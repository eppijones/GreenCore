"""
Simple HTTP server for serving static web files.

Serves the Three.js 3D putting simulator interface.
"""

import os
import threading
from http.server import HTTPServer, SimpleHTTPRequestHandler
from functools import partial


class CORSRequestHandler(SimpleHTTPRequestHandler):
    """HTTP request handler with CORS support and custom directory."""
    
    def __init__(self, *args, directory=None, **kwargs):
        self.custom_directory = directory
        super().__init__(*args, directory=directory, **kwargs)
    
    def end_headers(self):
        """Add CORS headers."""
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')
        self.send_header('Cache-Control', 'no-cache, no-store, must-revalidate')
        super().end_headers()
    
    def do_OPTIONS(self):
        """Handle CORS preflight."""
        self.send_response(200)
        self.end_headers()
    
    def log_message(self, format, *args):
        """Suppress default logging."""
        pass  # Silent by default
    
    def translate_path(self, path):
        """Translate URL path to filesystem path."""
        # Remove leading slash and any query parameters
        path = path.split('?')[0]
        path = path.split('#')[0]
        
        if path == '/' or path == '':
            path = '/index.html'
        
        # Use custom directory if set
        if self.custom_directory:
            return os.path.join(self.custom_directory, path.lstrip('/'))
        
        return super().translate_path(path)


class WebServer:
    """
    Simple HTTP server for serving static files.
    
    Runs in a separate thread to serve the web-based 3D simulator.
    """
    
    def __init__(
        self, 
        host: str = "localhost", 
        port: int = 8080,
        web_directory: str = "web"
    ):
        """
        Initialize web server.
        
        Args:
            host: Host address to bind to.
            port: Port number for HTTP connections.
            web_directory: Directory containing static files.
        """
        self.host = host
        self.port = port
        self.web_directory = os.path.abspath(web_directory)
        
        self._server: HTTPServer = None
        self._thread: threading.Thread = None
        self._running = False
    
    def start(self):
        """Start the web server in a background thread."""
        if self._running:
            print("[WebServer] Already running")
            return
        
        if not os.path.isdir(self.web_directory):
            print(f"[WebServer] Warning: Web directory not found: {self.web_directory}")
            os.makedirs(self.web_directory, exist_ok=True)
        
        # Create handler with custom directory
        handler = partial(CORSRequestHandler, directory=self.web_directory)
        
        try:
            self._server = HTTPServer((self.host, self.port), handler)
            self._running = True
            
            self._thread = threading.Thread(target=self._run_server, daemon=True)
            self._thread.start()
            
            print(f"[WebServer] Serving on http://{self.host}:{self.port}")
            print(f"[WebServer] Web directory: {self.web_directory}")
            
        except OSError as e:
            print(f"[WebServer] Failed to start: {e}")
            raise
    
    def stop(self):
        """Stop the web server."""
        self._running = False
        
        if self._server:
            self._server.shutdown()
            self._server = None
        
        if self._thread:
            self._thread.join(timeout=2.0)
            self._thread = None
        
        print("[WebServer] Stopped")
    
    def _run_server(self):
        """Run the server (in background thread)."""
        try:
            self._server.serve_forever()
        except Exception as e:
            if self._running:  # Only log if not intentionally stopped
                print(f"[WebServer] Error: {e}")
    
    @property
    def url(self) -> str:
        """Get server URL."""
        return f"http://{self.host}:{self.port}"
    
    @property
    def is_running(self) -> bool:
        """Check if server is running."""
        return self._running
