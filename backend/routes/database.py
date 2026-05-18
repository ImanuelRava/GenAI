import logging
from typing import Dict, Any

from flask import Blueprint, request, jsonify

from errors import APIError, NotFoundError
from cache import cached

logger = logging.getLogger(__name__)

database_bp = Blueprint('database', __name__, url_prefix='/api/database')

try:
    from modules.nicobot_database import get_database, NiCOBotDatabase
    from modules.nicobot_rag import get_rag, NiCOBotRAG
    DATABASE_AVAILABLE = True
except ImportError:
    DATABASE_AVAILABLE = False
    logger.warning("Database modules not available. Database API endpoints disabled.")


@database_bp.route('/status')
def database_status():
    """Get database status and statistics."""
    if not DATABASE_AVAILABLE:
        return jsonify({
            'success': False,
            'error': 'Database module not available',
            'available': False
        }), 503

    try:
        db = get_database()
        if not db._loaded:
            loaded = db.load()
            if not loaded:
                return jsonify({
                    'success': False,
                    'error': 'Failed to load database',
                    'available': False
                }), 503

        stats = db.get_statistics()
        return jsonify({
            'success': True,
            'available': True,
            'loaded': db._loaded,
            'statistics': stats
        })
    except Exception as e:
        logger.error(f"Database status error: {e}")
        return jsonify({
            'success': False,
            'error': str(e),
            'available': False
        }), 500


@database_bp.route('/search/compounds')
def search_compounds():
    """Search for compounds by name or SMILES."""
    if not DATABASE_AVAILABLE:
        raise APIError("Database module not available", 503)

    query = request.args.get('q', '')
    limit = min(int(request.args.get('limit', 10)), 50)
    category = request.args.get('category')

    if not query:
        raise APIError("Query parameter 'q' is required", 400)

    try:
        db = get_database()
        results = db.search_compounds(query, limit=limit)

        if category:
            results = [r for r in results if r.get('category') == category]

        return jsonify({
            'success': True,
            'query': query,
            'count': len(results),
            'results': results
        })
    except Exception as e:
        logger.error(f"Compound search error: {e}")
        raise APIError(f"Search error: {str(e)}", 500)


@database_bp.route('/search/papers')
def search_papers():
    """Search for papers by title or keyword."""
    if not DATABASE_AVAILABLE:
        raise APIError("Database module not available", 503)

    query = request.args.get('q', '')
    limit = min(int(request.args.get('limit', 10)), 50)

    if not query:
        raise APIError("Query parameter 'q' is required", 400)

    try:
        db = get_database()
        results = db.search_papers(query, limit=limit)

        return jsonify({
            'success': True,
            'query': query,
            'count': len(results),
            'results': results
        })
    except Exception as e:
        logger.error(f"Paper search error: {e}")
        raise APIError(f"Search error: {str(e)}", 500)


@database_bp.route('/compounds/<path:smiles>')
def get_compound(smiles: str):
    """Get compound information by SMILES."""
    if not DATABASE_AVAILABLE:
        raise APIError("Database module not available", 503)

    try:
        db = get_database()
        compound = db.get_compound_by_smiles(smiles)

        if compound:
            return jsonify({
                'success': True,
                'compound': {
                    'name': compound.name,
                    'smiles': compound.smiles,
                    'category': compound.category,
                    'leaving_group': compound.leaving_group,
                    'type': compound.compound_type
                }
            })
        else:
            raise NotFoundError(f"Compound not found: {smiles}")
    except NotFoundError:
        raise
    except Exception as e:
        logger.error(f"Get compound error: {e}")
        raise APIError(f"Error: {str(e)}", 500)


@database_bp.route('/papers/<path:doi>')
def get_paper(doi: str):
    """Get paper information by DOI."""
    if not DATABASE_AVAILABLE:
        raise APIError("Database module not available", 503)

    try:
        db = get_database()
        paper = db.get_paper_by_doi(doi)

        if paper:
            return jsonify({
                'success': True,
                'paper': {
                    'doi': paper.doi,
                    'title': paper.title,
                    'published_date': paper.published_date,
                    'authors': paper.authors,
                    'reaction_type': paper.reaction_type,
                    'strength': paper.strength,
                    'reference_count': len(paper.references)
                }
            })
        else:
            raise NotFoundError(f"Paper not found: {doi}")
    except NotFoundError:
        raise
    except Exception as e:
        logger.error(f"Get paper error: {e}")
        raise APIError(f"Error: {str(e)}", 500)


@database_bp.route('/papers/<path:doi>/related')
def get_related_papers(doi: str):
    """Get papers related through citations."""
    if not DATABASE_AVAILABLE:
        raise APIError("Database module not available", 503)

    limit = min(int(request.args.get('limit', 5)), 20)

    try:
        db = get_database()
        related = db.get_related_papers(doi, limit=limit)

        return jsonify({
            'success': True,
            'doi': doi,
            'count': len(related),
            'related_papers': [
                {
                    'doi': p.doi,
                    'title': p.title,
                    'published_date': p.published_date
                }
                for p in related
            ]
        })
    except Exception as e:
        logger.error(f"Get related papers error: {e}")
        raise APIError(f"Error: {str(e)}", 500)


@database_bp.route('/leaving-groups')
def get_leaving_groups():
    """Get all available leaving groups."""
    if not DATABASE_AVAILABLE:
        raise APIError("Database module not available", 503)

    try:
        db = get_database()
        lgs = db.get_leaving_groups()

        return jsonify({
            'success': True,
            'leaving_groups': lgs
        })
    except Exception as e:
        logger.error(f"Get leaving groups error: {e}")
        raise APIError(f"Error: {str(e)}", 500)


@database_bp.route('/reactions')
def get_reactions():
    """Get all available reaction types."""
    if not DATABASE_AVAILABLE:
        raise APIError("Database module not available", 503)

    try:
        db = get_database()
        reactions = db.get_reaction_types()

        return jsonify({
            'success': True,
            'count': len(reactions),
            'reactions': reactions
        })
    except Exception as e:
        logger.error(f"Get reactions error: {e}")
        raise APIError(f"Error: {str(e)}", 500)


@database_bp.route('/context')
def get_context():
    """Get database context for a query (for debugging/testing RAG)."""
    if not DATABASE_AVAILABLE:
        raise APIError("Database module not available", 503)

    query = request.args.get('q', '')

    if not query:
        raise APIError("Query parameter 'q' is required", 400)

    try:
        rag = get_rag()
        context = rag.retrieve_context(query)

        return jsonify({
            'success': True,
            'query': query,
            'context': {
                'compounds': context.compounds,
                'papers': context.papers,
                'reactions': context.reactions,
                'has_general_info': bool(context.general_info),
                'formatted_context': context.formatted_context
            }
        })
    except Exception as e:
        logger.error(f"Get context error: {e}")
        raise APIError(f"Error: {str(e)}", 500)


@database_bp.route('/statistics')
def get_statistics():
    """Get database statistics."""
    if not DATABASE_AVAILABLE:
        raise APIError("Database module not available", 503)

    try:
        db = get_database()
        stats = db.get_statistics()

        return jsonify({
            'success': True,
            'statistics': stats
        })
    except Exception as e:
        logger.error(f"Get statistics error: {e}")
        raise APIError(f"Error: {str(e)}", 500)
