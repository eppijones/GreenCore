"""
WebSocket server for real-time shot event broadcasting.

Pushes shot events to connected browser clients for 3D visualization.
"""

import asyncio
import json
import threading
from typing import Set, Optional, Any
from queue import Queue, Empty

import websockets


class WebSocketServer:
    """
    WebSocket server for broadcasting shot events to browser clients.
    
    Runs in a separate thread with its own event loop.
    Thread-safe message queue for sending events from the main thread.
    """
    
    def __init__(self, host: str = "localhost", port: int = 8765):
        """
        Initialize WebSocket server.
        
        Args:
            host: Host address to bind to.
            port: Port number for WebSocket connections.
        """
        self.host = host
        self.port = port
        
        # Connected clients
        self._clients: Set = set()
        
        # Message queue for thread-safe communication
        self._message_queue: Queue = Queue()
        
        # Server state
        self._server = None
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._thread: Optional[threading.Thread] = None
        self._running = False
        
        # Stats
        self._messages_sent = 0
    
    def start(self):
        """Start the WebSocket server in a background thread."""
        if self._running:
            print("[WebSocket] Already running")
            return
        
        self._running = True
        self._thread = threading.Thread(target=self._run_server, daemon=True)
        self._thread.start()
        print(f"[WebSocket] Starting server on ws://{self.host}:{self.port}")
    
    def stop(self):
        """Stop the WebSocket server."""
        self._running = False
        
        if self._loop:
            # Schedule shutdown
            self._loop.call_soon_threadsafe(self._loop.stop)
        
        if self._thread:
            self._thread.join(timeout=2.0)
        
        print("[WebSocket] Stopped")
    
    def broadcast(self, message: dict):
        """
        Broadcast a message to all connected clients.
        
        Thread-safe - can be called from any thread.
        
        Args:
            message: Dictionary to send as JSON.
        """
        self._message_queue.put(message)
    
    def _run_server(self):
        """Run the WebSocket server (in background thread)."""
        # Create new event loop for this thread
        self._loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self._loop)
        
        try:
            self._loop.run_until_complete(self._server_main())
        except Exception as e:
            print(f"[WebSocket] Server error: {e}")
        finally:
            self._loop.close()
    
    async def _server_main(self):
        """Main async server loop."""
        # Start WebSocket server
        async with websockets.serve(
            self._handle_client,
            self.host,
            self.port,
            ping_interval=20,
            ping_timeout=20,
        ) as server:
            self._server = server
            print(f"[WebSocket] Server listening on ws://{self.host}:{self.port}")
            
            # Process message queue
            while self._running:
                await self._process_queue()
                await asyncio.sleep(0.01)  # Small delay to prevent busy loop
    
    async def _handle_client(self, websocket):
        """
        Handle a client connection.
        
        Args:
            websocket: Client WebSocket connection.
        """
        # Register client
        self._clients.add(websocket)
        client_id = id(websocket)
        print(f"[WebSocket] Client connected: {client_id} (total: {len(self._clients)})")
        
        # Send welcome message
        try:
            await websocket.send(json.dumps({
                "type": "connected",
                "message": "Connected to Putting Launch Monitor"
            }))
        except Exception:
            pass
        
        try:
            # Keep connection alive and handle incoming messages
            async for message in websocket:
                # Handle any client messages (for future use)
                try:
                    data = json.loads(message)
                    await self._handle_client_message(websocket, data)
                except json.JSONDecodeError:
                    pass
        except websockets.exceptions.ConnectionClosed:
            pass
        finally:
            # Unregister client
            self._clients.discard(websocket)
            print(f"[WebSocket] Client disconnected: {client_id} (total: {len(self._clients)})")
    
    async def _handle_client_message(
        self, 
        websocket, 
        data: dict
    ):
        """Handle incoming client message."""
        msg_type = data.get("type", "")
        
        if msg_type == "ping":
            await websocket.send(json.dumps({"type": "pong"}))
        elif msg_type == "reset":
            # Client requested simulation reset
            print("[WebSocket] Client requested reset")
    
    async def _process_queue(self):
        """Process pending messages from the queue."""
        try:
            while True:
                message = self._message_queue.get_nowait()
                await self._broadcast_to_all(message)
        except Empty:
            pass
    
    async def _broadcast_to_all(self, message: dict):
        """Broadcast message to all connected clients."""
        if not self._clients:
            return
        
        json_message = json.dumps(message)
        
        # Send to all clients, remove any that fail
        disconnected = set()
        
        for client in self._clients:
            try:
                await client.send(json_message)
                self._messages_sent += 1
            except websockets.exceptions.ConnectionClosed:
                disconnected.add(client)
            except Exception as e:
                print(f"[WebSocket] Send error: {e}")
                disconnected.add(client)
        
        # Remove disconnected clients
        self._clients -= disconnected
    
    @property
    def client_count(self) -> int:
        """Get number of connected clients."""
        return len(self._clients)
    
    @property
    def is_running(self) -> bool:
        """Check if server is running."""
        return self._running
    
    @property
    def messages_sent(self) -> int:
        """Get total messages sent."""
        return self._messages_sent
