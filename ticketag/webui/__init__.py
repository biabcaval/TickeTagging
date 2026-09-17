"""Web UI: a small FastAPI app for classifying a ticket via either backend.

Everything specific to serving a browser-based interface lives here. The
shared core (``models``, ``pipeline``) knows nothing about HTTP or JSON.

The FastAPI instance itself is deliberately not re-exported here: doing so
would bind the name ``app`` on this package, shadowing the ``app`` submodule
for any later ``import ticketag.webui.app``. Import it directly instead, e.g.
``from ticketag.webui.app import app`` or ``uvicorn ticketag.webui.app:app``.
"""

from .app import run

__all__ = ["run"]
