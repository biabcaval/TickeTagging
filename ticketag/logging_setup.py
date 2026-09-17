"""Process-wide logging setup used by every CLI and the web UI.

Keeps stderr as the human-readable stream so stdout (JSON/CSV results)
stays machine-readable.
"""

from __future__ import annotations

import logging
import sys


def configure_logging(verbose: bool = False) -> None:
    """Send progress and warnings to stderr so stdout stays machine readable."""
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
        stream=sys.stderr,
    )
