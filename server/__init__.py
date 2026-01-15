"""
Server module for web and WebSocket communication.
"""

from .web_server import WebServer
from .websocket_server import WebSocketServer

__all__ = [
    'WebServer',
    'WebSocketServer',
]
