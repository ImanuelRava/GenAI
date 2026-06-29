"""
Visualization Routes Blueprint
Handles GNN and PCA visualization endpoints.

Optional dependencies: numpy. If missing, endpoints return 503.
"""

import logging
from typing import Dict, Any

from flask import Blueprint, request, jsonify

from core.errors import ValidationError, APIError

logger = logging.getLogger(__name__)

viz_bp = Blueprint('viz', __name__, url_prefix='/api')

# ---------------------------------------------------------------------------
# Lazy-load optional heavy modules (numpy, sklearn, etc.)
# ---------------------------------------------------------------------------

_VIZ_AVAILABLE = False
_mod_gnn = None
_mod_pca = None

try:
    import importlib
    _mod_gnn = importlib.import_module('modules.gnn_viz')
    _mod_pca = importlib.import_module('modules.pca_viz')
    _VIZ_AVAILABLE = True
except ImportError as exc:
    logger.warning(
        "Visualization modules unavailable (%s). "
        "Install numpy to enable GNN/PCA endpoints.",
        exc,
    )


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _viz_unavailable():
    return jsonify({
        'success': False,
        'error': 'Visualization modules not available. Install numpy.',
        'available': False,
    }), 503


# ---------------------------------------------------------------------------
# GNN endpoints
# ---------------------------------------------------------------------------

@viz_bp.route('/gnn/graph')
def api_gnn_graph():
    if not _VIZ_AVAILABLE:
        return _viz_unavailable()
    try:
        num_nodes = request.args.get('nodes', 6, type=int)
        num_nodes = max(2, min(num_nodes, 50))
        data = _mod_gnn.generate_sample_graph(num_nodes)
        return jsonify({'success': True, 'data': data})
    except (KeyboardInterrupt, SystemExit):
        raise
    except Exception as e:
        logger.error(f"GNN graph error: {e}")
        raise APIError(f"Error generating graph: {str(e)}", 500)


@viz_bp.route('/gnn/message-passing', methods=['POST'])
def api_gnn_message_passing():
    if not _VIZ_AVAILABLE:
        return _viz_unavailable()
    try:
        data = request.get_json()
        nodes = data.get('nodes', [])
        edges = data.get('edges', [])
        current_step = data.get('currentStep', 0)
        result = _mod_gnn.simulate_message_passing(nodes, edges, current_step)
        return jsonify({'success': True, 'data': result})
    except (KeyboardInterrupt, SystemExit):
        raise
    except Exception as e:
        logger.error(f"Message passing error: {e}")
        raise APIError(f"Error simulating message passing: {str(e)}", 500)


@viz_bp.route('/gnn/molecule/<molecule_type>')
def api_gnn_molecule(molecule_type: str):
    if not _VIZ_AVAILABLE:
        return _viz_unavailable()
    try:
        data = _mod_gnn.get_molecule_data(molecule_type)
        return jsonify({'success': True, 'data': data})
    except (KeyboardInterrupt, SystemExit):
        raise
    except Exception as e:
        logger.error(f"Molecule data error: {e}")
        raise APIError(f"Error getting molecule data: {str(e)}", 500)


@viz_bp.route('/gnn/embeddings')
def api_gnn_embeddings():
    if not _VIZ_AVAILABLE:
        return _viz_unavailable()
    try:
        data = _mod_gnn.get_gnn_embedding_demo()
        return jsonify({'success': True, 'data': data})
    except (KeyboardInterrupt, SystemExit):
        raise
    except Exception as e:
        logger.error(f"GNN embeddings error: {e}")
        raise APIError(f"Error getting embeddings: {str(e)}", 500)


# ---------------------------------------------------------------------------
# PCA endpoints
# ---------------------------------------------------------------------------

@viz_bp.route('/pca/data/<data_type>')
def api_pca_data(data_type: str):
    if not _VIZ_AVAILABLE:
        return _viz_unavailable()
    try:
        n_samples = request.args.get('samples', 60, type=int)
        n_samples = max(10, min(n_samples, 500))

        # Must match the data_types handled by modules.pca_viz.generate_2d_data.
        valid_data_types = ['clusters', 'clustered', 'linear', 'circular', 'structured']
        if data_type not in valid_data_types:
            data_type = 'structured'

        data = _mod_pca.generate_2d_data(data_type, n_samples)
        return jsonify({'success': True, 'data': data})
    except (KeyboardInterrupt, SystemExit):
        raise
    except Exception as e:
        logger.error(f"PCA data error: {e}")
        raise APIError(f"Error generating PCA data: {str(e)}", 500)


@viz_bp.route('/pca/scree')
def api_pca_scree():
    if not _VIZ_AVAILABLE:
        return _viz_unavailable()
    try:
        num_features = request.args.get('features', 10, type=int)
        num_features = max(2, min(num_features, 100))

        data_type = request.args.get('type', 'structured')
        # Must match the data_types handled by modules.pca_viz.generate_scree_data.
        valid_data_types = ['structured', 'moderate', 'linear', 'random']
        if data_type not in valid_data_types:
            data_type = 'structured'

        data = _mod_pca.generate_scree_data(num_features, data_type)
        return jsonify({'success': True, 'data': data})
    except (KeyboardInterrupt, SystemExit):
        raise
    except Exception as e:
        logger.error(f"Scree data error: {e}")
        raise APIError(f"Error generating scree data: {str(e)}", 500)


@viz_bp.route('/pca/chemistry/<dataset>')
def api_pca_chemistry(dataset: str):
    if not _VIZ_AVAILABLE:
        return _viz_unavailable()
    try:
        data = _mod_pca.get_chemistry_pca_data(dataset)
        return jsonify({'success': True, 'data': data})
    except (KeyboardInterrupt, SystemExit):
        raise
    except Exception as e:
        logger.error(f"Chemistry PCA error: {e}")
        raise APIError(f"Error getting chemistry PCA data: {str(e)}", 500)
