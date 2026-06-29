"""
Tests for the narrowed exception-handling cleanup.

Verifies the key behavioral changes from the broad `except Exception` cleanup:

  1. **LLM HTTP wrappers narrow to specific exceptions** — JSON parse errors
     and unexpected response shapes return None, but programming bugs
     (AttributeError, ImportError) propagate.

  2. **PDF cascading extractors narrow to I/O + parse errors** — OSError,
     ValueError, RuntimeError are caught (allowing fallback to the next
     extractor), but programming bugs propagate.

  3. **Top-level route handlers re-raise KeyboardInterrupt / SystemExit** —
     Ctrl+C during a request now actually interrupts the server instead of
     being swallowed and returning a 500 JSON error.

  4. **Cache I/O narrows to OSError + JSONDecodeError** — corrupt cache
     files return None (cache miss), but programming bugs propagate.

  5. **Retry wrappers re-raise KeyboardInterrupt / SystemExit** — so
     Ctrl+C during a retry wait immediately exits instead of waiting for
     the retry loop to finish.

  6. **Pseudo-SMILES cleanup catches only the specific exceptions** that
     PIL and the pseudo-SMILES detector can raise.

These tests lock in the cleanup so it doesn't regress.
"""

import sys
import os
import json
import asyncio
from unittest.mock import patch, MagicMock, AsyncMock

import pytest

# Make backend/ importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'backend'))

from llm.client import LLMClient, retry_with_backoff, retry_with_backoff_async


# =============================================================================
# LLM HTTP wrapper — narrowed exception handling
# =============================================================================

class TestLLMClientNarrowedExceptions:
    """Verify LLMClient.chat catches only the expected exceptions."""

    def test_chat_returns_none_on_json_decode_error(self):
        """ValueError (JSON decode error) should be caught → return None."""
        c = LLMClient(provider='deepseek', api_key='sk-test')
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.side_effect = ValueError("not valid JSON")
        with patch('llm.client.requests.post', return_value=mock_response):
            result = c.chat([{"role": "user", "content": "hi"}])
            assert result is None

    def test_chat_returns_none_on_unexpected_response_shape(self):
        """KeyError (missing 'choices' key in response) → return None."""
        c = LLMClient(provider='deepseek', api_key='sk-test')
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {}  # missing 'choices'
        with patch('llm.client.requests.post', return_value=mock_response):
            result = c.chat([{"role": "user", "content": "hi"}])
            assert result is None

    def test_chat_returns_none_on_type_error(self):
        """TypeError (None where dict expected) → return None."""
        c = LLMClient(provider='deepseek', api_key='sk-test')
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"choices": [None]}  # None, not dict
        with patch('llm.client.requests.post', return_value=mock_response):
            result = c.chat([{"role": "user", "content": "hi"}])
            assert result is None

    def test_chat_propagates_attribute_error(self):
        """AttributeError (programming bug) should propagate, NOT be caught."""
        c = LLMClient(provider='deepseek', api_key='sk-test')
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"choices": [{"message": "not_a_dict"}]}
        # _extract_text_from_response will try data["choices"][0]["message"]["content"]
        # but "message" is a string, so ["content"] raises TypeError.
        # That IS caught. We need a different bug to test propagation.
        # Let's mock requests.post to raise AttributeError directly.
        with patch('llm.client.requests.post', side_effect=AttributeError("bug")):
            # AttributeError is NOT in the caught list, so it propagates.
            with pytest.raises(AttributeError, match="bug"):
                c.chat([{"role": "user", "content": "hi"}])

    def test_chat_propagates_import_error(self):
        """ImportError (programming bug) should propagate."""
        c = LLMClient(provider='deepseek', api_key='sk-test')
        with patch('llm.client.requests.post', side_effect=ImportError("missing module")):
            with pytest.raises(ImportError, match="missing module"):
                c.chat([{"role": "user", "content": "hi"}])

    def test_vision_returns_none_on_json_decode_error(self):
        """Vision path should also catch JSON decode errors."""
        c = LLMClient(provider='deepseek', api_key='sk-test')
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.side_effect = ValueError("bad JSON")
        with patch('llm.client.requests.post', return_value=mock_response):
            result = c.vision([
                {"role": "user", "content": [
                    {"type": "text", "text": "x"},
                    {"type": "image_url", "image_url": {"url": "data:image/png;base64,abc"}},
                ]}
            ])
            assert result is None

    @pytest.mark.asyncio
    async def test_chat_async_returns_none_on_json_decode_error(self):
        """Async chat should catch JSON decode errors too."""
        c = LLMClient(provider='deepseek', api_key='sk-test')

        mock_response = AsyncMock()
        mock_response.status = 200
        mock_response.json = AsyncMock(side_effect=ValueError("bad JSON"))
        mock_response.text = AsyncMock(return_value='')

        mock_session = MagicMock()
        mock_session.post.return_value.__aenter__ = AsyncMock(return_value=mock_response)
        mock_session.post.return_value.__aexit__ = AsyncMock(return_value=None)
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=None)

        with patch('llm.client.aiohttp.ClientSession', return_value=mock_session):
            result = await c.chat_async([{"role": "user", "content": "hi"}])
            assert result is None


# =============================================================================
# Retry wrappers — KeyboardInterrupt / SystemExit re-raise
# =============================================================================

class TestRetryKeyboardInterrupt:
    """Verify retry_with_backoff re-raises KeyboardInterrupt immediately."""

    def test_sync_retry_raises_keyboard_interrupt_immediately(self):
        """KeyboardInterrupt should NOT be retried — it propagates at once."""
        call_count = 0
        def func():
            nonlocal call_count
            call_count += 1
            raise KeyboardInterrupt()
        with pytest.raises(KeyboardInterrupt):
            retry_with_backoff(func, max_retries=5, retry_delay=0.01)
        # Should have been called exactly once (no retries).
        assert call_count == 1

    def test_sync_retry_raises_system_exit_immediately(self):
        """SystemExit should NOT be retried — it propagates at once."""
        call_count = 0
        def func():
            nonlocal call_count
            call_count += 1
            raise SystemExit(1)
        with pytest.raises(SystemExit):
            retry_with_backoff(func, max_retries=5, retry_delay=0.01)
        assert call_count == 1

    @pytest.mark.asyncio
    async def test_async_retry_raises_keyboard_interrupt_immediately(self):
        call_count = 0
        async def func():
            nonlocal call_count
            call_count += 1
            raise KeyboardInterrupt()
        with pytest.raises(KeyboardInterrupt):
            await retry_with_backoff_async(func, max_retries=5, retry_delay=0.01)
        assert call_count == 1

    @pytest.mark.asyncio
    async def test_async_retry_raises_system_exit_immediately(self):
        call_count = 0
        async def func():
            nonlocal call_count
            call_count += 1
            raise SystemExit(0)
        with pytest.raises(SystemExit):
            await retry_with_backoff_async(func, max_retries=5, retry_delay=0.01)
        assert call_count == 1

    def test_sync_retry_still_retries_normal_exceptions(self):
        """Regular exceptions should still be retried (not affected by the fix)."""
        call_count = 0
        def func():
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise ValueError("transient")
            return "success"
        result = retry_with_backoff(func, max_retries=5, retry_delay=0.01)
        assert result == "success"
        assert call_count == 3


# =============================================================================
# Cache I/O — narrowed to OSError + JSONDecodeError
# =============================================================================

class TestCacheNarrowedExceptions:
    """Verify the chemextract cache catches only I/O + JSON errors."""

    def test_load_cache_returns_none_on_corrupt_json(self):
        """Corrupt JSON in cache file → return None (cache miss)."""
        from modules.chemextract.cache import _get_cached_result
        # Mock the cache file to be readable but contain corrupt JSON.
        mock_file = MagicMock()
        mock_file.__enter__ = MagicMock(return_value=mock_file)
        mock_file.__exit__ = MagicMock(return_value=None)
        with patch('modules.chemextract.cache._pdf_content_hash', return_value='fake_hash'), \
             patch('modules.chemextract.cache.os.path.exists', return_value=True), \
             patch('modules.chemextract.cache.os.path.getmtime', return_value=1e9), \
             patch('modules.chemextract.cache.json.load', side_effect=json.JSONDecodeError("err", "doc", 0)), \
             patch('builtins.open', return_value=mock_file):
            result = _get_cached_result('/fake.pdf', 'model', 'provider')
            assert result is None

    def test_load_cache_returns_none_on_os_error(self):
        """OSError reading cache file → return None (cache miss)."""
        from modules.chemextract.cache import _get_cached_result
        with patch('modules.chemextract.cache._pdf_content_hash', return_value='fake_hash'), \
             patch('modules.chemextract.cache.os.path.exists', return_value=True), \
             patch('modules.chemextract.cache.os.path.getmtime', return_value=1e9), \
             patch('builtins.open', side_effect=OSError("permission denied")):
            result = _get_cached_result('/fake.pdf', 'model', 'provider')
            assert result is None

    def test_save_cache_swallows_os_error(self):
        """OSError writing cache → log warning, don't crash."""
        from modules.chemextract.cache import _save_cached_result
        with patch('modules.chemextract.cache._pdf_content_hash', return_value='fake_hash'), \
             patch('builtins.open', side_effect=OSError("disk full")):
            # Should NOT raise.
            _save_cached_result({"data": 1}, '/fake.pdf', 'model', 'provider')

    def test_save_cache_swallows_type_error(self):
        """TypeError (non-serializable object) → log warning, don't crash."""
        from modules.chemextract.cache import _save_cached_result
        # Patch json.dump at the cache module level (where it was imported).
        with patch('modules.chemextract.cache._pdf_content_hash', return_value='fake_hash'), \
             patch('modules.chemextract.cache.json.dump', side_effect=TypeError("not serializable")):
            _save_cached_result({"data": 1}, '/fake.pdf', 'model', 'provider')


# =============================================================================
# PDF processor — cascading extractors narrow to I/O + parse errors
# =============================================================================

class TestPdfProcessorNarrowedExceptions:
    """Verify the PDF text extraction cascade catches only I/O + parse errors."""

    def test_extract_text_cascades_on_pypdf_os_error(self):
        """If pypdf raises OSError, the cascade falls through to the next extractor."""
        from modules.chemextract.pdf_processor import extract_text_from_pdf
        # Force pypdf to be available and raise OSError
        with patch('modules.chemextract.pdf_processor.HAS_PYPDF', True), \
             patch('modules.chemextract.pdf_processor.pypdf') as mock_pypdf:
            mock_pypdf.PdfReader.side_effect = OSError("corrupt PDF")
            # Also disable the other extractors so we get the final ImportError
            with patch('modules.chemextract.pdf_processor.HAS_PDFPLUMBER', False), \
                 patch('modules.chemextract.pdf_processor.HAS_PYMUPDF', False):
                with pytest.raises(ImportError, match="No PDF text extraction library"):
                    extract_text_from_pdf('/fake.pdf')

    def test_extract_text_propagates_attribute_error(self):
        """AttributeError (programming bug in pypdf mock) should propagate."""
        from modules.chemextract.pdf_processor import extract_text_from_pdf
        with patch('modules.chemextract.pdf_processor.HAS_PYPDF', True), \
             patch('modules.chemextract.pdf_processor.pypdf') as mock_pypdf:
            # Make pypdf.PdfReader raise AttributeError (not caught by the
            # narrowed except clause).
            mock_pypdf.PdfReader.side_effect = AttributeError("bug in code")
            with pytest.raises(AttributeError, match="bug in code"):
                extract_text_from_pdf('/fake.pdf')


# =============================================================================
# Route handlers — KeyboardInterrupt / SystemExit propagate
# =============================================================================

class TestRouteHandlerKeyboardInterrupt:
    """Verify top-level route handlers let KeyboardInterrupt / SystemExit through."""

    def test_database_status_propagates_keyboard_interrupt(self, client):
        """If get_database() raises KeyboardInterrupt, the route should let it
        propagate instead of swallowing it into a 500 JSON response."""
        with patch('routes.database.get_database', side_effect=KeyboardInterrupt()):
            with pytest.raises(KeyboardInterrupt):
                client.get('/api/database/status')

    def test_database_status_propagates_system_exit(self, client):
        with patch('routes.database.get_database', side_effect=SystemExit(0)):
            with pytest.raises(SystemExit):
                client.get('/api/database/status')

    def test_visualization_propagates_keyboard_interrupt(self, client):
        """Vision endpoints should also let KeyboardInterrupt through."""
        with patch('routes.visualization._mod_gnn.generate_sample_graph',
                   side_effect=KeyboardInterrupt()):
            with pytest.raises(KeyboardInterrupt):
                client.get('/api/gnn/graph')

    def test_extract_models_still_works_normally(self, client):
        """Sanity check: normal requests still get a normal response."""
        rv = client.get('/api/extract/models')
        assert rv.status_code == 200

    def test_extract_text_endpoint_propagates_keyboard_interrupt(self, client):
        """If the text extraction LLM call raises KeyboardInterrupt, the route
        should let it propagate instead of catching it as a 500 error."""
        # The /api/extract endpoint calls call_text_llm from chemextract.standalone.
        # We mock it to raise KeyboardInterrupt.
        with patch('modules.chemextract.standalone.call_text_llm',
                   side_effect=KeyboardInterrupt()):
            with pytest.raises(KeyboardInterrupt):
                client.post('/api/extract', json={
                    'text': 'some text',
                    'provider': 'deepseek',
                    'api_key': 'sk-test',
                })

    def test_nicobot_chat_propagates_keyboard_interrupt(self, client):
        """NiCOBot chat should let KeyboardInterrupt through."""
        with patch('chat.nicobot.get_llm_response', side_effect=KeyboardInterrupt()):
            with pytest.raises(KeyboardInterrupt):
                client.post('/api/nicobot/chat', json={
                    'message': 'hi',
                    'provider': 'deepseek',
                    'api_key': 'sk-test',
                })


# =============================================================================
# Pytest fixtures
# =============================================================================

@pytest.fixture
def app():
    from app import app
    app.config['TESTING'] = True
    yield app


@pytest.fixture
def client(app):
    return app.test_client()
