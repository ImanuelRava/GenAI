"""
Visualization Routes Blueprint
Handles GNN and PCA visualization endpoints
"""

import logging
from typing import Dict, Any

from flask import Blueprint, request, jsonify

from errors import ValidationError, APIError
from modules.gnn_viz import (
    generate_sample_graph,
    simulate_message_passing,
    get_molecule_data,
    get_gnn_embedding_demo
)
from modules.pca_viz import (
    generate_2d_data,
    generate_scree_data,
    get_chemistry_pca_data
)

logger = logging.getLogger(__name__)

viz_bp = Blueprint('viz', __name__, url_prefix='/api')

@viz_bp.route('/gnn/graph')
def api_gnn_graph():
    try:
        num_nodes = request.args.get('nodes', 6, type=int)
        num_nodes = max(2, min(num_nodes, 50))

        data = generate_sample_graph(num_nodes)
        return jsonify({
            'success': True,
            'data': data
        })
    except Exception as e:
        logger.error(f"GNN graph error: {e}")
        raise APIError(f"Error generating graph: {str(e)}", 500)

@viz_bp.route('/gnn/message-passing', methods=['POST'])
def api_gnn_message_passing():
    try:
        data = request.get_json()
        nodes = data.get('nodes', [])
        edges = data.get('edges', [])
        current_step = data.get('currentStep', 0)

        result = simulate_message_passing(nodes, edges, current_step)
        return jsonify({
            'success': True,
            'data': result
        })
    except Exception as e:
        logger.error(f"Message passing error: {e}")
        raise APIError(f"Error simulating message passing: {str(e)}", 500)

@viz_bp.route('/gnn/molecule/<molecule_type>')
def api_gnn_molecule(molecule_type: str):
    try:
        data = get_molecule_data(molecule_type)
        return jsonify({
            'success': True,
            'data': data
        })
    except Exception as e:
        logger.error(f"Molecule data error: {e}")
        raise APIError(f"Error getting molecule data: {str(e)}", 500)

@viz_bp.route('/gnn/embeddings')
def api_gnn_embeddings():
    try:
        data = get_gnn_embedding_demo()
        return jsonify({
            'success': True,
            'data': data
        })
    except Exception as e:
        logger.error(f"GNN embeddings error: {e}")
        raise APIError(f"Error getting embeddings: {str(e)}", 500)

@viz_bp.route('/pca/data/<data_type>')
def api_pca_data(data_type: str):
    try:
        n_samples = request.args.get('samples', 60, type=int)
        n_samples = max(10, min(n_samples, 500))

        valid_data_types = ['clustered', 'linear', 'circular', 'structured']
        if data_type not in valid_data_types:
            data_type = 'structured'

        data = generate_2d_data(data_type, n_samples)
        return jsonify({
            'success': True,
            'data': data
        })
    except Exception as e:
        logger.error(f"PCA data error: {e}")
        raise APIError(f"Error generating PCA data: {str(e)}", 500)

@viz_bp.route('/pca/scree')
def api_pca_scree():
    try:
        num_features = request.args.get('features', 10, type=int)
        num_features = max(2, min(num_features, 100))

        data_type = request.args.get('type', 'structured')
        valid_data_types = ['structured', 'random', 'linear']
        if data_type not in valid_data_types:
            data_type = 'structured'

        data = generate_scree_data(num_features, data_type)
        return jsonify({
            'success': True,
            'data': data
        })
    except Exception as e:
        logger.error(f"Scree data error: {e}")
        raise APIError(f"Error generating scree data: {str(e)}", 500)

@viz_bp.route('/pca/chemistry/<dataset>')
def api_pca_chemistry(dataset: str):
    try:
        data = get_chemistry_pca_data(dataset)
        return jsonify({
            'success': True,
            'data': data
        })
    except Exception as e:
        logger.error(f"Chemistry PCA error: {e}")
        raise APIError(f"Error getting chemistry PCA data: {str(e)}", 500)
