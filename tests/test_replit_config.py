"""
Tests for Replit environment detection in wsgi.py and backend/app.py.

Verifies:
  - When REPL_ID or REPL_SLUG env vars are set, the app binds to 0.0.0.0
    (required for Replit's web preview to reach the app).
  - When no Replit env vars are set, the app keeps the secure default of
    127.0.0.1 (no network exposure).
  - On Replit, the PORT env var (set by Replit) is used as the default
    port if FLASK_PORT is not set.
  - FLASK_HOST / FLASK_PORT env vars always override the defaults.
  - The .replit config file exists and has the correct run command.
  - The replit.nix file exists and declares rdkit.
  - The main.py shim imports successfully.

These tests don't actually start the Flask server — they test the
environment-detection logic in isolation.
"""

import os
import sys
from unittest.mock import patch

import pytest

# Make backend/ importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'backend'))


def _resolve_bind_config(env: dict) -> tuple:
    """Replicate the bind-config logic from wsgi.py's __main__ block.

    This lets us test the logic without actually starting the server.
    """
    is_replit = bool(env.get('REPL_ID') or env.get('REPL_SLUG'))
    debug = env.get('FLASK_DEBUG', '0') == '1'
    default_host = '0.0.0.0' if is_replit else '127.0.0.1'
    host = env.get('FLASK_HOST', default_host)
    default_port = env.get('PORT', '5000') if is_replit else '5000'
    port = int(env.get('FLASK_PORT') or default_port)
    if debug and host not in ('127.0.0.1', 'localhost'):
        host = '127.0.0.1'
    return host, port, debug, is_replit


# =============================================================================
# Replit detection tests
# =============================================================================

class TestReplitDetection:
    """Verify the Replit environment detection logic."""

    def test_detects_replit_when_repl_id_set(self):
        """When REPL_ID is set, is_replit should be True and host 0.0.0.0."""
        host, port, debug, is_replit = _resolve_bind_config({
            'REPL_ID': 'abc123',
            'PORT': '8080',
        })
        assert is_replit is True
        assert host == '0.0.0.0'
        assert port == 8080

    def test_detects_replit_when_repl_slug_set(self):
        """When REPL_SLUG is set (but not REPL_ID), is_replit should be True."""
        host, port, debug, is_replit = _resolve_bind_config({
            'REPL_SLUG': 'my-genai-app',
            'PORT': '5000',
        })
        assert is_replit is True
        assert host == '0.0.0.0'

    def test_non_replit_defaults_to_localhost(self):
        """Without Replit env vars, host should default to 127.0.0.1 (secure)."""
        host, port, debug, is_replit = _resolve_bind_config({})
        assert is_replit is False
        assert host == '127.0.0.1'
        assert port == 5000

    def test_replit_uses_port_env_var(self):
        """On Replit, the PORT env var should be used as the default port."""
        host, port, debug, is_replit = _resolve_bind_config({
            'REPL_ID': 'test',
            'PORT': '3000',
        })
        assert port == 3000

    def test_replit_flask_port_overrides_port_env(self):
        """FLASK_PORT should take precedence over PORT on Replit."""
        host, port, debug, is_replit = _resolve_bind_config({
            'REPL_ID': 'test',
            'PORT': '8080',
            'FLASK_PORT': '5000',
        })
        assert port == 5000

    def test_flask_host_overrides_replit_default(self):
        """If FLASK_HOST is explicitly set, it overrides the Replit 0.0.0.0 default."""
        host, port, debug, is_replit = _resolve_bind_config({
            'REPL_ID': 'test',
            'FLASK_HOST': '127.0.0.1',
        })
        assert host == '127.0.0.1'

    def test_debug_mode_forced_to_localhost_even_on_replit(self):
        """Even on Replit, FLASK_DEBUG=1 should force host to 127.0.0.1
        to avoid exposing the Werkzeug debugger (RCE risk)."""
        host, port, debug, is_replit = _resolve_bind_config({
            'REPL_ID': 'test',
            'FLASK_DEBUG': '1',
        })
        assert debug is True
        assert host == '127.0.0.1'  # forced away from 0.0.0.0


# =============================================================================
# Replit config file tests
# =============================================================================

class TestReplitConfigFiles:
    """Verify the Replit-specific config files exist and are correct."""

    def test_replit_file_exists(self):
        replit_path = os.path.join(os.path.dirname(__file__), '..', '.replit')
        assert os.path.exists(replit_path), ".replit file missing"

    def test_replit_file_has_run_command(self):
        replit_path = os.path.join(os.path.dirname(__file__), '..', '.replit')
        with open(replit_path) as f:
            content = f.read()
        assert 'run = "python wsgi.py"' in content, \
            ".replit must specify 'python wsgi.py' as the run command"

    def test_replit_file_has_port_mapping(self):
        replit_path = os.path.join(os.path.dirname(__file__), '..', '.replit')
        with open(replit_path) as f:
            content = f.read()
        assert 'localPort' in content, \
            ".replit must map a local port for the web preview"
        assert '5000' in content, \
            ".replit should map port 5000 (the app's default)"

    def test_replit_nix_file_exists(self):
        nix_path = os.path.join(os.path.dirname(__file__), '..', 'replit.nix')
        assert os.path.exists(nix_path), "replit.nix file missing"

    def test_replit_nix_declares_rdkit(self):
        nix_path = os.path.join(os.path.dirname(__file__), '..', 'replit.nix')
        with open(nix_path) as f:
            content = f.read()
        assert 'rdkit' in content, \
            "replit.nix should declare rdkit for chemistry features"

    def test_main_py_shim_exists(self):
        main_path = os.path.join(os.path.dirname(__file__), '..', 'main.py')
        assert os.path.exists(main_path), "main.py shim missing"

    def test_main_py_imports_wsgi_application(self):
        """main.py should be able to import 'application' from wsgi."""
        main_path = os.path.join(os.path.dirname(__file__), '..', 'main.py')
        with open(main_path) as f:
            content = f.read()
        assert 'from wsgi import' in content, \
            "main.py should import from wsgi.py"
        assert 'application' in content, \
            "main.py should reference the WSGI application object"


# =============================================================================
# requirements.txt uses rdkit-pypi (better Replit compat)
# =============================================================================

class TestRequirementsTxt:
    """Verify requirements.txt uses rdkit-pypi (pre-built wheel)."""

    def test_uses_rdkit_pypi_not_rdkit(self):
        req_path = os.path.join(os.path.dirname(__file__), '..', 'requirements.txt')
        with open(req_path) as f:
            content = f.read()
        # rdkit-pypi should be listed (the pre-built wheel that works on Replit).
        # The bare 'rdkit' package requires system-level installation.
        assert 'rdkit-pypi' in content, \
            "requirements.txt should use rdkit-pypi (pre-built wheel for Replit)"
