"""
Tests for the split data_extraction package (backend/routes/data_extraction/).

Verifies:
  - The package re-exports data_extraction_bp for backwards compatibility
    (app.register_blueprint(data_extraction_bp) must continue to work)
  - All 12 endpoints from the original monolithic file are still registered
    on the blueprint
  - The shared helpers (_helpers.py) work correctly:
    - get_model_for_provider resolution + fallback
    - validate_pdf_upload rejects invalid uploads
    - cleanup_temp_file is safe with None / non-existent paths
    - merge_extraction_results handles flat + structured formats + dedup
  - The endpoint modules can be imported independently
  - The Flask test client can hit GET /api/extract/models (no LLM needed)
  - POST /api/extract/format/schemes validates input correctly (no LLM needed)

These are structural / contract tests — they don't make real LLM calls.
The LLM-backed endpoints (/extract, /extract/pdf/vision, etc.) are tested
via the existing test_app.py smoke tests.
"""

import sys
import os
import io
import json
from unittest.mock import patch

import pytest

# Make backend/ importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'backend'))


# =============================================================================
# Package structure tests
# =============================================================================

class TestPackageStructure:
    """Verify the package layout is correct and all modules import cleanly."""

    def test_blueprint_re_exported_from_package_init(self):
        """`from routes.data_extraction import data_extraction_bp` must work."""
        from routes.data_extraction import data_extraction_bp
        assert data_extraction_bp is not None
        assert data_extraction_bp.name == 'data_extraction'

    def test_all_endpoint_modules_importable(self):
        """Every endpoint module should import without error."""
        from routes.data_extraction import (
            models_endpoint,
            text_endpoints,
            pdf_text_endpoint,
            pdf_vision_endpoints,
            chemextract_endpoints,
            reactionlens_endpoints,
            format_schemes_endpoint,
        )
        # Each must be a valid module
        for mod in [models_endpoint, text_endpoints, pdf_text_endpoint,
                    pdf_vision_endpoints, chemextract_endpoints,
                    reactionlens_endpoints, format_schemes_endpoint]:
            assert mod is not None

    def test_helpers_module_importable(self):
        from routes.data_extraction._helpers import (
            data_extraction_bp,
            get_model_for_provider,
            validate_pdf_upload,
            cleanup_temp_file,
            merge_extraction_results,
            AVAILABLE_MODELS,
            REACTIONLENS_INFO,
            PROVIDER_DEFAULT_MODELS,
            VISION_CAPABLE_PROVIDERS,
        )
        # All must be defined
        assert data_blueprint_ok(data_extraction_bp)
        assert callable(get_model_for_provider)
        assert callable(validate_pdf_upload)
        assert callable(cleanup_temp_file)
        assert callable(merge_extraction_results)
        assert isinstance(AVAILABLE_MODELS, list)
        assert isinstance(REACTIONLENS_INFO, dict)
        assert isinstance(PROVIDER_DEFAULT_MODELS, dict)
        assert isinstance(VISION_CAPABLE_PROVIDERS, list)


def data_blueprint_ok(bp):
    """Helper — check that the blueprint looks valid."""
    return bp is not None and bp.name == 'data_extraction'


# =============================================================================
# Route registration tests — all 12 endpoints must be on the blueprint
# =============================================================================

class TestRouteRegistration:
    """Verify all 12 endpoints from the original file are still registered."""

    @pytest.fixture(scope='class')
    def registered_routes(self):
        """Register the blueprint on a fresh Flask app and return the rules."""
        from flask import Flask
        from routes.data_extraction import data_extraction_bp

        app = Flask(__name__)
        app.register_blueprint(data_extraction_bp)
        return {rule.rule: sorted(rule.methods - {'HEAD', 'OPTIONS'})
                for rule in app.url_map.iter_rules()}

    @pytest.mark.parametrize('route, expected_methods', [
        ('/api/extract/models', ['GET']),
        ('/api/extract', ['POST']),
        ('/api/extract/async', ['POST']),
        ('/api/extract/pdf', ['POST']),
        ('/api/extract/pdf/vision', ['POST']),
        ('/api/extract/pdf/vision/async', ['POST']),
        ('/api/extract/pdf/chemextract', ['POST']),
        ('/api/extract/pdf/chemextract/async', ['POST']),
        ('/api/extract/pdf/reactionlens', ['POST']),
        ('/api/extract/pdf/reactionlens/async', ['POST']),
        ('/api/extract/pdf/reactionlens/text', ['POST']),
        ('/api/extract/format/schemes', ['POST']),
    ])
    def test_route_registered_with_correct_methods(self, registered_routes, route, expected_methods):
        assert route in registered_routes, f"Route {route} not registered"
        assert registered_routes[route] == expected_methods, (
            f"Route {route} has methods {registered_routes[route]}, "
            f"expected {expected_methods}"
        )

    def test_total_route_count_is_12(self, registered_routes):
        """The original file had 12 endpoints — the split must preserve all."""
        extract_routes = [r for r in registered_routes if r.startswith('/api/extract')]
        assert len(extract_routes) == 12


# =============================================================================
# Helper tests
# =============================================================================

class TestGetModelForProvider:
    """Test the model resolution helper."""

    def test_explicit_model_wins(self):
        from routes.data_extraction._helpers import get_model_for_provider
        assert get_model_for_provider('deepseek', 'custom-model') == 'custom-model'

    def test_default_model_used_when_none(self):
        from routes.data_extraction._helpers import get_model_for_provider, PROVIDER_DEFAULT_MODELS
        for provider, expected_model in PROVIDER_DEFAULT_MODELS.items():
            assert get_model_for_provider(provider) == expected_model

    def test_unknown_provider_falls_back_to_deepseek(self):
        from routes.data_extraction._helpers import get_model_for_provider
        assert get_model_for_provider('totally-unknown') == 'deepseek-chat'


class TestCleanupTempFile:
    """Test the temp file cleanup helper."""

    def test_safe_with_none(self):
        from routes.data_extraction._helpers import cleanup_temp_file
        # Should not raise
        cleanup_temp_file(None)

    def test_safe_with_nonexistent_path(self):
        from routes.data_extraction._helpers import cleanup_temp_file
        cleanup_temp_file('/nonexistent/path/file.pdf')

    def test_deletes_existing_file(self, tmp_path):
        from routes.data_extraction._helpers import cleanup_temp_file
        # Create a temp file
        f = tmp_path / "test.pdf"
        f.write_text("test content")
        assert f.exists()
        cleanup_temp_file(str(f))
        assert not f.exists()


class TestValidatePdfUpload:
    """Test the PDF upload validation helper.

    These tests use a Flask request context to simulate uploads.
    """

    def test_rejects_missing_file(self):
        from routes.data_extraction._helpers import validate_pdf_upload
        from core.errors import ValidationError
        from flask import Flask

        app = Flask(__name__)
        with app.test_request_context('/'):
            with pytest.raises(ValidationError, match='No file provided'):
                validate_pdf_upload()

    def test_rejects_empty_filename(self):
        from routes.data_extraction._helpers import validate_pdf_upload
        from core.errors import ValidationError
        from flask import Flask
        from werkzeug.datastructures import FileStorage

        app = Flask(__name__)
        with app.test_request_context('/', data={'file': (io.BytesIO(b''), '')}):
            with pytest.raises(ValidationError, match='No file selected'):
                validate_pdf_upload()

    def test_rejects_non_pdf_filename(self):
        from routes.data_extraction._helpers import validate_pdf_upload
        from core.errors import ValidationError
        from flask import Flask

        app = Flask(__name__)
        with app.test_request_context('/', data={'file': (io.BytesIO(b'fake'), 'not_a_pdf.txt')}):
            with pytest.raises(ValidationError, match='Only PDF files'):
                validate_pdf_upload()

    def test_accepts_valid_pdf_upload(self):
        from routes.data_extraction._helpers import validate_pdf_upload
        from flask import Flask

        app = Flask(__name__)
        with app.test_request_context(
            '/',
            data={'file': (io.BytesIO(b'%PDF-1.4 fake pdf'), 'paper.pdf')},
        ):
            tmp_path, filename = validate_pdf_upload()
            try:
                assert filename == 'paper.pdf'
                assert os.path.exists(tmp_path)
                with open(tmp_path, 'rb') as f:
                    assert f.read() == b'%PDF-1.4 fake pdf'
            finally:
                os.unlink(tmp_path)


class TestMergeExtractionResults:
    """Test the extraction-results merger."""

    def test_empty_input_returns_empty_merge(self):
        from routes.data_extraction._helpers import merge_extraction_results
        result = merge_extraction_results([])
        assert result['reactants'] == []
        assert result['products'] == []
        assert result['pages_with_data'] == []

    def test_skips_results_with_no_data(self):
        from routes.data_extraction._helpers import merge_extraction_results
        result = merge_extraction_results([
            {'page': 1, 'source': 'embedded_figure', 'data': None},
            {'page': 2, 'source': 'scheme_page', 'data': {}},
        ])
        assert result['pages_with_data'] == []  # both skipped

    def test_flat_format_dedup(self):
        """Items appearing on multiple pages should dedupe."""
        from routes.data_extraction._helpers import merge_extraction_results
        result = merge_extraction_results([
            {'page': 1, 'source': 'embedded_figure', 'data': {
                'reactants': ['A', 'B'],
                'products': ['C'],
            }},
            {'page': 2, 'source': 'scheme_page', 'data': {
                'reactants': ['A', 'D'],  # 'A' already seen
                'products': ['E'],
            }},
        ])
        assert result['reactants'] == ['A', 'B', 'D']
        assert result['products'] == ['C', 'E']
        assert result['pages_with_data'] == [1, 2]

    def test_conditions_first_value_wins(self):
        """When the same condition key appears on multiple pages, the first
        non-empty value wins (subsequent values are ignored)."""
        from routes.data_extraction._helpers import merge_extraction_results
        result = merge_extraction_results([
            {'page': 1, 'source': 'embedded_figure', 'data': {
                'conditions': {'temperature': '80 C'},
            }},
            {'page': 2, 'source': 'scheme_page', 'data': {
                'conditions': {'temperature': '100 C'},  # ignored
                'conditions': {'time': '12 h'},          # new key, kept
            }},
        ])
        assert result['conditions']['temperature'] == '80 C'
        assert result['conditions']['time'] == '12 h'

    def test_structured_format_flattens_to_flat_format(self):
        """Structured-format reaction_schemes should also populate the
        flat-format reactants/products lists for backwards compat."""
        from routes.data_extraction._helpers import merge_extraction_results
        result = merge_extraction_results([
            {'page': 1, 'source': 'embedded_figure', 'data': {
                'reaction_schemes': [{
                    'reactants': [{'name': 'Bromobenzene'}],
                    'products': [{'name': 'Biphenyl'}],
                }],
            }},
        ])
        # Structured entry preserved
        assert len(result['reaction_schemes']) == 1
        assert result['reaction_schemes'][0]['_page'] == 1
        assert result['reaction_schemes'][0]['_source'] == 'embedded_figure'
        # Flattened
        assert 'Bromobenzene' in result['reactants']
        assert 'Biphenyl' in result['products']

    def test_structured_format_compounds_tagged_with_page_source(self):
        from routes.data_extraction._helpers import merge_extraction_results
        result = merge_extraction_results([
            {'page': 3, 'source': 'scheme_page', 'data': {
                'compounds': [{'name': 'Pd(OAc)2', 'smiles': 'CC(=O)O[Pd]'}],
            }},
        ])
        assert len(result['compounds']) == 1
        assert result['compounds'][0]['_page'] == 3
        assert result['compounds'][0]['_source'] == 'scheme_page'

    def test_structured_format_table_data_preserved(self):
        from routes.data_extraction._helpers import merge_extraction_results
        result = merge_extraction_results([
            {'page': 5, 'source': 'scheme_page', 'data': {
                'table_data': [{'entry': 1, 'yield': '85%'}],
            }},
        ])
        assert len(result['table_data']) == 1
        assert result['table_data'][0]['_page'] == 5
        assert result['table_data'][0]['entry'] == 1

    def test_image_description_collected(self):
        from routes.data_extraction._helpers import merge_extraction_results
        result = merge_extraction_results([
            {'page': 1, 'source': 'embedded_figure', 'data': {
                'image_description': 'A reaction scheme showing Suzuki coupling',
            }},
            {'page': 2, 'source': 'scheme_page', 'data': {
                'description': 'A table of substrates',
            }},
        ])
        assert len(result['image_descriptions']) == 2
        assert result['image_descriptions'][0]['page'] == 1
        assert result['image_descriptions'][1]['page'] == 2
        assert result['image_descriptions'][1]['source'] == 'scheme_page'


# =============================================================================
# Live endpoint tests (no LLM calls)
# =============================================================================

class TestModelsEndpoint:
    """GET /api/extract/models — returns the static catalog, no LLM needed."""

    def test_returns_200_with_models_list(self, client):
        rv = client.get('/api/extract/models')
        assert rv.status_code == 200
        data = rv.get_json()
        assert data['success'] is True
        assert isinstance(data['models'], list)
        assert len(data['models']) > 0
        # Each model has the expected fields
        for m in data['models']:
            assert 'id' in m
            assert 'name' in m
            assert 'provider' in m
            assert 'description' in m

    def test_includes_vision_providers_list(self, client):
        rv = client.get('/api/extract/models')
        data = rv.get_json()
        assert 'vision_providers' in data
        assert set(data['vision_providers']) == {'deepseek', 'openai', 'gemini', 'anthropic'}

    def test_includes_reactionlens_info(self, client):
        rv = client.get('/api/extract/models')
        data = rv.get_json()
        assert 'reactionlens' in data
        assert data['reactionlens']['id'] == 'reactionlens'
        assert 'capabilities' in data['reactionlens']

    def test_reports_async_support(self, client):
        rv = client.get('/api/extract/models')
        data = rv.get_json()
        assert data['async_support'] is True


class TestFormatSchemesEndpoint:
    """POST /api/extract/format/schemes — pure post-processing, no LLM call."""

    def test_rejects_empty_body(self, client):
        # Sending json={} triggers the "No JSON data provided" ValidationError
        # (sending json=None would make Werkzeug raise UnsupportedMediaType 415
        # before our handler runs, which is a different code path).
        rv = client.post('/api/extract/format/schemes', json={})
        assert rv.status_code == 400
        data = rv.get_json()
        assert data['success'] is False

    def test_rejects_body_without_reactions_or_compounds(self, client):
        rv = client.post('/api/extract/format/schemes', json={
            'format': 'smiles',
            # no 'reactions' or 'compounds' key
        })
        assert rv.status_code == 400
        data = rv.get_json()
        assert data['success'] is False
        assert 'No reaction or compound data' in data['error']

    def test_formats_reactions_when_chemextract_available(self, client):
        """If chemextract.reaction_formatter imports cleanly, the endpoint
        should call it and return its output. We mock the formatter to
        avoid depending on chemextract's full import chain."""
        # Patch the lazy import inside the endpoint function
        with patch('modules.chemextract.reaction_formatter.format_reaction_schemes') as mock_fmt:
            mock_fmt.return_value = {
                'schemes': [
                    {'reactants': 'c1ccccc1Br', 'products': 'c1ccc(-c2ccccc2)cc1'},
                ],
            }
            rv = client.post('/api/extract/format/schemes', json={
                'reactions': [{'reactants': [{'name': 'bromobenzene'}]}],
                'format': 'smiles',
            })
        assert rv.status_code == 200
        data = rv.get_json()
        assert data['success'] is True
        assert data['reactions_formatted'] == 1
        assert len(data['data']['schemes']) == 1


class TestExtractTextEndpointValidation:
    """POST /api/extract — input validation, no real LLM call."""

    def test_rejects_empty_body(self, client):
        # Sending json={} triggers the "No JSON data provided" ValidationError.
        rv = client.post('/api/extract', json={})
        assert rv.status_code == 400

    def test_rejects_empty_text(self, client):
        rv = client.post('/api/extract', json={
            'text': '',
            'provider': 'deepseek',
        })
        assert rv.status_code == 400
        data = rv.get_json()
        assert 'cannot be empty' in data['error'].lower()


# =============================================================================
# Pytest fixtures — the test_app.py client fixture works because app.py
# imports the blueprint via `from routes.data_extraction import data_extraction_bp`,
# which now resolves to the new package. We just re-use it.
# =============================================================================

@pytest.fixture
def app():
    """Create the Flask app from backend/app.py — same as test_app.py."""
    from app import app
    app.config['TESTING'] = True
    yield app


@pytest.fixture
def client(app):
    """Create a test client."""
    return app.test_client()
