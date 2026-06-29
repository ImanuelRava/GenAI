"""
main.py — Replit entry point shim.

Replit's default Python template looks for ``main.py`` as the run target.
This shim simply imports and runs the real WSGI application from
``wsgi.py`` so that:

  1. Clicking the green "Run" button in Replit works (it runs main.py).
  2. The ``python main.py`` command (Replit's default run command) works.
  3. The real application logic stays in ``wsgi.py`` for production
     deployments (gunicorn, waitress, etc.).

If you've changed the run command in ``.replit`` to ``python wsgi.py``
directly, you don't need this file — but it's harmless to keep.
"""

# Import the WSGI application + the dev-server entry point from wsgi.py.
# wsgi.py handles .env loading, sys.path setup, and the __main__ block
# that actually starts the Flask dev server.
from wsgi import *  # noqa: F401, F403
from wsgi import application, _load_env  # noqa: F401

if __name__ == '__main__':
    # Delegate to wsgi.py's __main__ block by re-importing it as a module
    # and calling its main logic. The simplest way is to exec the module's
    # __main__ block — but since wsgi.py already has the if __name__ guard,
    # we just call application.run() with the same env-var-driven config.
    import os
    import logging

    logger = logging.getLogger('main')

    # Replit sets PORT and REPL_ID env vars. If we detect Replit, bind to
    # 0.0.0.0 so the web preview can reach the app.
    is_replit = bool(os.environ.get('REPL_ID') or os.environ.get('REPL_SLUG'))

    debug = os.environ.get('FLASK_DEBUG', '0') == '1'
    host = os.environ.get('FLASK_HOST', '0.0.0.0' if is_replit else '127.0.0.1')
    port = int(os.environ.get('FLASK_PORT') or os.environ.get('PORT') or '5000')

    if debug and host not in ('127.0.0.1', 'localhost'):
        logger.warning(
            "FLASK_DEBUG=1 with FLASK_HOST=%s — Werkzeug debugger would be "
            "exposed to the network (RCE risk). Forcing host=127.0.0.1.", host,
        )
        host = '127.0.0.1'

    logger.info("Starting development server on http://%s:%d (debug=%s, replit=%s)",
                host, port, debug, is_replit)
    application.run(debug=debug, host=host, port=port)
