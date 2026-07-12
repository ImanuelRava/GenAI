"""
RAL (Redox-Active Ligands) Database API Routes

Provides REST endpoints for searching and browsing the RAL research
database (ligand electronic properties + reductive coupling literature).

Mirrors the pattern of ``routes/database.py`` but scoped to RAL data.
"""

import logging
from typing import Dict, Any

from flask import Blueprint, request, jsonify

from core.errors import APIError, NotFoundError, ValidationError

logger = logging.getLogger(__name__)

ral_database_bp = Blueprint(
    'ral_database', __name__, url_prefix='/api/ral-database'
)

try:
    from modules.ral_database import get_ral_database, RALDatabase
    from modules.ral_rag import get_ral_rag, RALRAG
    RAL_DATABASE_AVAILABLE = True
except ImportError:
    RAL_DATABASE_AVAILABLE = False
    logger.warning(
        "RAL database modules not available. "
        "RAL Database API endpoints disabled."
    )

try:
    from modules.ral_similarity import get_similarity_engine
    RAL_SIMILARITY_AVAILABLE = True
except ImportError:
    RAL_SIMILARITY_AVAILABLE = False


def _require_db():
    """Raise 503 if RAL database module is not available."""
    if not RAL_DATABASE_AVAILABLE:
        raise APIError("RAL database module not available", 503)


@ral_database_bp.route('/status')
def ral_database_status():
    """Get RAL database status and statistics."""
    _require_db()

    try:
        db = get_ral_database()
        if not db._loaded:
            loaded = db.load()
            if not loaded:
                return jsonify({
                    'success': False,
                    'error': 'Failed to load RAL database',
                    'available': False,
                }), 503

        stats = db.get_statistics()
        return jsonify({
            'success': True,
            'available': True,
            'loaded': db._loaded,
            'statistics': stats,
        })
    except (KeyboardInterrupt, SystemExit):
        raise
    except Exception as e:
        logger.error(f"RAL database status error: {e}")
        return jsonify({
            'success': False, 'error': str(e), 'available': False,
        }), 500


@ral_database_bp.route('/ligand-classes')
def ligand_classes():
    """Get all ligand classes with descriptor ranges and counts."""
    _require_db()

    try:
        db = get_ral_database()
        classes = db.get_ligand_classes()
        return jsonify({
            'success': True,
            'count': len(classes),
            'classes': classes,
        })
    except (KeyboardInterrupt, SystemExit):
        raise
    except Exception as e:
        logger.error(f"Ligand classes error: {e}")
        raise APIError(f"Error: {str(e)}", 500)


@ral_database_bp.route('/search/ligands')
def search_ligands():
    """Search ligands by name, keyword, or class.

    Query params:
        q: search query (required)
        class: filter by ligand class (optional)
        limit: max results, default 10, max 50
    """
    _require_db()

    query = request.args.get('q', '')
    if not query:
        raise ValidationError("Query parameter 'q' is required", field='q')

    limit = min(int(request.args.get('limit', 10)), 50)
    ligand_class = request.args.get('class')

    try:
        db = get_ral_database()
        results = db.search_ligands(query, limit=limit,
                                     ligand_class=ligand_class)
        return jsonify({
            'success': True,
            'query': query,
            'class_filter': ligand_class,
            'count': len(results),
            'results': results,
        })
    except (KeyboardInterrupt, SystemExit):
        raise
    except Exception as e:
        logger.error(f"Ligand search error: {e}")
        raise APIError(f"Search error: {str(e)}", 500)


@ral_database_bp.route('/ligands/<path:name>')
def get_ligand_by_name(name: str):
    """Get a specific ligand's full electronic properties by name."""
    _require_db()

    try:
        db = get_ral_database()
        ligand = db.get_ligand_by_name(name)

        if ligand:
            return jsonify({'success': True, 'ligand': ligand})
        else:
            raise NotFoundError(f"Ligand not found: {name}")
    except NotFoundError:
        raise
    except (KeyboardInterrupt, SystemExit):
        raise
    except Exception as e:
        logger.error(f"Get ligand error: {e}")
        raise APIError(f"Error: {str(e)}", 500)


@ral_database_bp.route('/ligands/by-class/<ligand_class>')
def get_ligands_by_class(ligand_class: str):
    """Get all ligands in a given class with full descriptors."""
    _require_db()

    try:
        db = get_ral_database()
        results = db.get_ligands_by_class(ligand_class)

        if not results:
            raise NotFoundError(
                f"No ligands found for class: {ligand_class}"
            )

        return jsonify({
            'success': True,
            'class': ligand_class,
            'count': len(results),
            'ligands': results,
        })
    except NotFoundError:
        raise
    except (KeyboardInterrupt, SystemExit):
        raise
    except Exception as e:
        logger.error(f"Get ligands by class error: {e}")
        raise APIError(f"Error: {str(e)}", 500)


@ral_database_bp.route('/compare-classes/<class_a>/<class_b>')
def compare_classes(class_a: str, class_b: str):
    _require_db()

    try:
        db = get_ral_database()
        comparison = db.compare_ligand_classes(class_a, class_b)

        if comparison is None:
            raise NotFoundError(
                f"One or both classes not found: {class_a}, {class_b}"
            )

        return jsonify({
            'success': True,
            'class_a': class_a,
            'class_b': class_b,
            'comparison': comparison,
        })
    except NotFoundError:
        raise
    except (KeyboardInterrupt, SystemExit):
        raise
    except Exception as e:
        logger.error(f"Compare classes error: {e}")
        raise APIError(f"Error: {str(e)}", 500)


@ral_database_bp.route('/search/reactions')
def search_reactions():
    """Search reaction-ligand literature entries by keyword.

    Query params:
        q: search query (required)
        class: filter by mapped ligand class (optional)
        limit: max results, default 10, max 50
    """
    _require_db()

    query = request.args.get('q', '')
    if not query:
        raise ValidationError("Query parameter 'q' is required", field='q')

    limit = min(int(request.args.get('limit', 10)), 50)
    ligand_class = request.args.get('class')

    try:
        db = get_ral_database()
        results = db.search_reactions(query, limit=limit,
                                       ligand_class=ligand_class)
        return jsonify({
            'success': True,
            'query': query,
            'class_filter': ligand_class,
            'count': len(results),
            'results': results,
        })
    except (KeyboardInterrupt, SystemExit):
        raise
    except Exception as e:
        logger.error(f"Reaction search error: {e}")
        raise APIError(f"Search error: {str(e)}", 500)


@ral_database_bp.route('/reactions/<path:doi>')
def get_reaction_by_doi(doi: str):
    """Get a specific reaction-ligand entry by DOI."""
    _require_db()

    try:
        db = get_ral_database()
        entry = db.get_reaction_by_doi(doi)

        if entry:
            return jsonify({'success': True, 'reaction': entry})
        else:
            raise NotFoundError(f"Reaction not found for DOI: {doi}")
    except NotFoundError:
        raise
    except (KeyboardInterrupt, SystemExit):
        raise
    except Exception as e:
        logger.error(f"Get reaction error: {e}")
        raise APIError(f"Error: {str(e)}", 500)


@ral_database_bp.route('/reactions/by-class/<ligand_class>')
def get_reactions_by_class(ligand_class: str):
    """Get all curated reactions whose optimum ligand maps to a given class."""
    _require_db()

    try:
        db = get_ral_database()
        results = db.get_reactions_by_class(ligand_class)

        return jsonify({
            'success': True,
            'class': ligand_class,
            'count': len(results),
            'reactions': results,
        })
    except (KeyboardInterrupt, SystemExit):
        raise
    except Exception as e:
        logger.error(f"Get reactions by class error: {e}")
        raise APIError(f"Error: {str(e)}", 500)


@ral_database_bp.route('/context')
def get_rag_context():
    """Get RAG context for a query (for debugging/testing).

    Query params:
        q: the user query (required)
    """
    _require_db()

    query = request.args.get('q', '')
    if not query:
        raise ValidationError("Query parameter 'q' is required", field='q')

    try:
        rag = get_ral_rag()
        context = rag.retrieve_context(query)
        scores = rag.analyze_query(query)

        return jsonify({
            'success': True,
            'query': query,
            'intent_scores': scores,
            'context': {
                'ligands': context.ligands,
                'reactions': context.reactions,
                'detected_class': context.detected_class,
                'has_class_info': context.ligand_class_info is not None,
                'formatted_context': context.formatted_context,
            },
        })
    except (KeyboardInterrupt, SystemExit):
        raise
    except Exception as e:
        logger.error(f"Get RAG context error: {e}")
        raise APIError(f"Error: {str(e)}", 500)


@ral_database_bp.route('/statistics')
def ral_statistics():
    """Get RAL database statistics."""
    _require_db()

    try:
        db = get_ral_database()
        stats = db.get_statistics()
        return jsonify({
            'success': True,
            'statistics': stats,
        })
    except (KeyboardInterrupt, SystemExit):
        raise
    except Exception as e:
        logger.error(f"RAL statistics error: {e}")
        raise APIError(f"Error: {str(e)}", 500)


# ---------------------------------------------------------------------------
# Similarity & Recommendation endpoints
# ---------------------------------------------------------------------------

@ral_database_bp.route('/similarity/<ligand_name>')
def find_similar_ligands(ligand_name: str):
    """Find ligands with similar electronic profiles.

    Uses UMAP (preferred) or PCA (fallback) for dimensionality
    reduction, then KNN in the embedding space.

    Query params:
        k: number of results (default 5, max 20)
        same_class: 'true' to only return same-class ligands
        different_class: 'true' to only return different-class ligands
    """
    if not RAL_SIMILARITY_AVAILABLE:
        raise APIError("Similarity engine not available (needs numpy/sklearn/umap)", 503)

    try:
        engine = get_similarity_engine()
        if not engine._fitted:
            return jsonify({
                'success': False,
                'error': 'Similarity engine not fitted',
            }), 503

        k = min(int(request.args.get('k', 5)), 20)
        same_class = request.args.get('same_class', '').lower() == 'true'
        different_class = request.args.get('different_class', '').lower() == 'true'

        results = engine.recommend_for_ligand(
            ligand_name, k=k,
            same_class_only=same_class,
            different_class_only=different_class,
        )

        return jsonify({
            'success': True,
            'query_ligand': ligand_name,
            'method': engine._method,
            'count': len(results),
            'recommendations': [
                {
                    'name': r.name,
                    'class': r.ligand_class,
                    'distance': round(r.distance, 4),
                    'cosine_similarity': round(r.cosine_similarity, 4),
                    'embedding_coords': [round(c, 4) for c in r.embedding_coords],
                    'features': r.features,
                }
                for r in results
            ],
        })
    except (KeyboardInterrupt, SystemExit):
        raise
    except Exception as e:
        logger.error(f"Similarity search error: {e}")
        raise APIError(f"Error: {str(e)}", 500)


@ral_database_bp.route('/embedding/data')
@ral_database_bp.route('/pca/data')  # backward-compatible
def embedding_visualization_data():
    """Get embedding projection of all ligands for scatter plot visualisation.

    Returns 2D/3D coordinates, class labels, cluster IDs, and method info.
    Uses UMAP coordinates when available, PCA as fallback.
    """
    if not RAL_SIMILARITY_AVAILABLE:
        raise APIError("Similarity engine not available", 503)

    try:
        engine = get_similarity_engine()
        if not engine._fitted:
            return jsonify({'success': False, 'error': 'Not fitted'}), 503

        data = engine.get_embedding_data()
        return jsonify({
            'success': True,
            'embedding': data,
            'pca': data,  # backward-compatible
        })
    except (KeyboardInterrupt, SystemExit):
        raise
    except Exception as e:
        logger.error(f"Embedding data error: {e}")
        raise APIError(f"Error: {str(e)}", 500)


@ral_database_bp.route('/pca/loadings')
def pca_loadings():
    """Get PCA loadings — correlation of original features with each PC.

    Useful for understanding which electronic descriptors drive the
    clustering (e.g. "PC1 is mainly HOMO-LUMO gap").
    """
    if not RAL_SIMILARITY_AVAILABLE:
        raise APIError("Similarity engine not available", 503)

    try:
        engine = get_similarity_engine()
        if not engine._fitted:
            return jsonify({'success': False, 'error': 'Not fitted'}), 503

        loadings = engine.get_pca_loadings()
        return jsonify({
            'success': True,
            'loadings': loadings,
        })
    except (KeyboardInterrupt, SystemExit):
        raise
    except Exception as e:
        logger.error(f"PCA loadings error: {e}")
        raise APIError(f"Error: {str(e)}", 500)


@ral_database_bp.route('/clusters')
def get_clusters():
    """Get all ligand clusters with member lists and centroid descriptors."""
    if not RAL_SIMILARITY_AVAILABLE:
        raise APIError("Similarity engine not available", 503)

    try:
        engine = get_similarity_engine()
        if not engine._fitted or engine._cluster_labels is None:
            return jsonify({
                'success': False,
                'error': 'Clustering not fitted',
            }), 503

        clusters = engine.get_all_clusters()
        return jsonify({
            'success': True,
            'n_clusters': len(clusters),
            'clusters': [
                {
                    'cluster_id': c.cluster_id,
                    'ligand_count': c.ligand_count,
                    'ligand_names': c.ligand_names,
                    'class_distribution': c.class_distribution,
                    'centroid_features': c.centroid_features,
                }
                for c in clusters
            ],
        })
    except (KeyboardInterrupt, SystemExit):
        raise
    except Exception as e:
        logger.error(f"Clusters error: {e}")
        raise APIError(f"Error: {str(e)}", 500)


@ral_database_bp.route('/clusters/<int:cluster_id>/members')
def get_cluster_members(cluster_id: int):
    """Get all ligands in a specific cluster."""
    if not RAL_SIMILARITY_AVAILABLE:
        raise APIError("Similarity engine not available", 503)

    try:
        engine = get_similarity_engine()
        if not engine._fitted or engine._cluster_labels is None:
            return jsonify({'success': False, 'error': 'Not fitted'}), 503

        members = engine.get_cluster_members(cluster_id)
        return jsonify({
            'success': True,
            'cluster_id': cluster_id,
            'method': engine._method,
            'count': len(members),
            'members': members,
        })
    except (KeyboardInterrupt, SystemExit):
        raise
    except Exception as e:
        logger.error(f"Cluster members error: {e}")
        raise APIError(f"Error: {str(e)}", 500)


@ral_database_bp.route('/similarity/method-info')
def similarity_method_info():
    """Get information about the similarity engine's method and parameters.

    Returns which dimensionality reduction method is active (UMAP or PCA),
    the number of components, UMAP hyperparameters (if applicable),
    and PCA variance explained.
    """
    if not RAL_SIMILARITY_AVAILABLE:
        raise APIError("Similarity engine not available", 503)

    try:
        engine = get_similarity_engine()
        info = engine.get_method_info()
        return jsonify({
            'success': True,
            'method_info': info,
        })
    except (KeyboardInterrupt, SystemExit):
        raise
    except Exception as e:
        logger.error(f"Method info error: {e}")
        raise APIError(f"Error: {str(e)}", 500)