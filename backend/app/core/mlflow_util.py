"""Helpers for optional MLflow tracking (avoid long retries when server is down)."""

import logging
import socket
from urllib.parse import urlparse

logger = logging.getLogger(__name__)


def mlflow_server_reachable(tracking_uri: str, timeout_sec: float = 1.5) -> bool:
    """
    Return True if we can open a TCP connection to the tracking URI host:port.
    file:// and unrecognized schemes are treated as reachable (no probe).
    """
    raw = (tracking_uri or "").strip()
    if not raw or raw.startswith("file:"):
        return True
    parsed = urlparse(raw)
    if parsed.scheme not in ("http", "https"):
        return True
    host = parsed.hostname
    if not host:
        return False
    port = parsed.port
    if port is None:
        port = 443 if parsed.scheme == "https" else 80
    try:
        with socket.create_connection((host, port), timeout=timeout_sec):
            pass
        return True
    except OSError as e:
        logger.debug("MLflow tracking URI not reachable (%s:%s): %s", host, port, e)
        return False
