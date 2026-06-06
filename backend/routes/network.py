"""
Network Routes Blueprint
Handles citation network analysis endpoints
"""

import os
import uuid
import logging


from flask import Blueprint, request, jsonify

from errors import APIError, ValidationError
from utils import sanitize_filename
from config import config

logger = logging.getLogger(__name__)

network_bp = Blueprint('network', __name__, url_prefix='/api/network')

def allowed_file(filename: str) -> bool:
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in config.ALLOWED_EXTENSIONS

def validate_file_upload(file) -> tuple:
    if not file:
        return False, "No file provided"

    if file.filename == '':
        return False, "No file selected"

    if not allowed_file(file.filename):
        return False, f"File type not allowed. Allowed types: {', '.join(config.ALLOWED_EXTENSIONS)}"

    return True, None

@network_bp.route('', methods=['POST'])
def analyze_network():
    if 'file' not in request.files:
        raise ValidationError('No file part in request')

    file = request.files['file']
    analysis_type = request.form.get('type', 'forward')

    is_valid, error_msg = validate_file_upload(file)
    if not is_valid:
        raise ValidationError(error_msg)

    valid_types = ['forward', 'backward', 'cross']
    if analysis_type not in valid_types:
        raise ValidationError(
            f'Invalid analysis type. Must be one of: {valid_types}',
            payload={'valid_types': valid_types}
        )

    if file:
        safe_name = sanitize_filename(file.filename)
        if not safe_name:
            raise ValidationError('Invalid filename')

        unique_filename = f"{uuid.uuid4().hex}_{safe_name}"
        filepath = os.path.join(config.UPLOAD_FOLDER, unique_filename)

        os.makedirs(config.UPLOAD_FOLDER, exist_ok=True)
        file.save(filepath)

        try:
            from modules.Forward_Reference import build_forward_network
            from modules.Backward_Reference import build_reference_network
            from modules.Cross_Reference import build_cross_reference_network

            G = None
            suggestions = []
            all_papers = []

            def log_progress(msg):
                logger.info(f"[PROGRESS]: {msg}")

            if analysis_type == "forward":
                G, suggestions, all_papers = build_forward_network(filepath, progress_callback=log_progress)
            elif analysis_type == "backward":
                G, suggestions, all_papers = build_reference_network(filepath, progress_callback=log_progress)
            elif analysis_type == "cross":
                G = build_cross_reference_network(filepath, progress_callback=log_progress)
                all_papers = []
                for n in G.nodes():
                    data = G.nodes[n]
                    all_papers.append({
                        'Number': len(all_papers) + 1,
                        'DOI': n,
                        'Title': data.get('title', 'No Title'),
                        'Publication Year': data.get('year', 0),
                        'Corresponding Author': data.get('author', 'Unknown'),
                        'Global Citation Count': data.get('citations', 0),
                        'Local Citation Count': G.in_degree(n)
                    })
                suggestions = []

            if G is None or G.number_of_nodes() < 2:
                raise APIError('Could not build network from the provided file', 400)

            nodes = []
            for node_id in G.nodes():
                node_data = dict(G.nodes[node_id])
                node_data['id'] = node_id
                if 'is_main' in node_data:
                    node_data['is_main'] = "True" if node_data['is_main'] else "False"
                nodes.append(node_data)

            edges = []
            for source, target in G.edges():
                edges.append({
                    'source': source,
                    'target': target
                })

            graph_json = {
                'nodes': nodes,
                'edges': edges
            }

            return jsonify({
                'success': True,
                'elements': graph_json,
                'suggestions': suggestions,
                'all_papers': all_papers,
                'stats': {
                    'nodes': G.number_of_nodes(),
                    'edges': G.number_of_edges()
                }
            })

        except APIError:
            raise
        except Exception as e:
            logger.error(f"Network analysis error: {e}", exc_info=True)
            raise APIError('An internal error occurred during analysis', 500)

        finally:
            if os.path.exists(filepath):
                os.remove(filepath)

    raise ValidationError('No file provided')
