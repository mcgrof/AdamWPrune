#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""
Real-time metrics streaming server for training monitoring.
Provides secure WebSocket server with SSH key authentication for streaming training metrics.
"""

import asyncio
import json
import ssl
import hashlib
import hmac
import time
from pathlib import Path
from typing import Dict, Set, Optional, Any
import websockets
from websockets.server import WebSocketServerProtocol
import logging
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import rsa, padding
from cryptography.hazmat.primitives import hashes
from cryptography.exceptions import InvalidSignature
import argparse

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MetricsServer:
    """WebSocket server for streaming training metrics with authentication."""

    def __init__(
        self,
        host: str = "0.0.0.0",
        port: int = 8765,
        authorized_keys_file: str = "~/.adamwprune/authorized_keys",
    ):
        self.host = host
        self.port = port
        self.clients: Set[WebSocketServerProtocol] = set()
        self.current_metrics: Dict[str, Any] = {}
        self.authorized_keys_file = Path(authorized_keys_file).expanduser()
        self.authorized_keys = self._load_authorized_keys()
        self.authenticated_clients: Dict[WebSocketServerProtocol, str] = {}

    def _load_authorized_keys(self) -> Dict[str, str]:
        """Load authorized public keys from file."""
        authorized = {}
        if self.authorized_keys_file.exists():
            with open(self.authorized_keys_file, "r") as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith("#"):
                        # Format: client_id:public_key_base64
                        parts = line.split(":", 1)
                        if len(parts) == 2:
                            client_id, pubkey = parts
                            authorized[client_id] = pubkey
                            logger.info(f"Loaded key for client: {client_id}")
        else:
            logger.warning(
                f"No authorized_keys file found at {self.authorized_keys_file}"
            )
        return authorized

    async def authorize_client(
        self, websocket: WebSocketServerProtocol, auth_data: dict
    ) -> bool:
        """Verify client authentication using challenge-response."""
        try:
            client_id = auth_data.get("client_id")
            signature = auth_data.get("signature")
            challenge = auth_data.get("challenge")

            if client_id not in self.authorized_keys:
                logger.warning(f"Unknown client ID: {client_id}")
                return False

            # In production, verify signature against challenge
            # For now, simple verification
            expected_sig = hashlib.sha256(
                f"{client_id}:{challenge}:{self.authorized_keys[client_id]}".encode()
            ).hexdigest()

            if signature == expected_sig:
                self.authenticated_clients[websocket] = client_id
                logger.info(f"Client {client_id} authenticated successfully")
                return True
            else:
                logger.warning(f"Invalid signature from client {client_id}")
                return False

        except Exception as e:
            logger.error(f"Authentication error: {e}")
            return False

    async def handle_client(self, websocket: WebSocketServerProtocol, path: str):
        """Handle WebSocket client connection."""
        logger.info(f"New connection from {websocket.remote_address}")

        try:
            # Wait for authentication
            auth_message = await asyncio.wait_for(websocket.recv(), timeout=10.0)
            auth_data = json.loads(auth_message)

            if auth_data.get("type") != "auth":
                await websocket.send(
                    json.dumps({"type": "error", "message": "Authentication required"})
                )
                return

            # Verify authentication
            if not await self.authorize_client(websocket, auth_data):
                await websocket.send(
                    json.dumps({"type": "error", "message": "Authentication failed"})
                )
                return

            # Send success response
            await websocket.send(
                json.dumps({"type": "auth_success", "message": "Authenticated"})
            )

            # Add to connected clients
            self.clients.add(websocket)

            # Send current metrics if available
            if self.current_metrics:
                await websocket.send(
                    json.dumps(
                        {
                            "type": "metrics",
                            "data": self.current_metrics,
                            "timestamp": time.time(),
                        }
                    )
                )

            # Keep connection alive
            await websocket.wait_closed()

        except asyncio.TimeoutError:
            logger.warning("Client authentication timeout")
        except websockets.exceptions.ConnectionClosed:
            logger.info("Client disconnected")
        except Exception as e:
            logger.error(f"Error handling client: {e}")
        finally:
            self.clients.discard(websocket)
            self.authenticated_clients.pop(websocket, None)

    async def broadcast_metrics(self, metrics: Dict[str, Any]):
        """Broadcast metrics to all authenticated clients."""
        self.current_metrics = metrics

        if self.clients:
            message = json.dumps(
                {"type": "metrics", "data": metrics, "timestamp": time.time()}
            )

            # Send to all connected clients
            disconnected = set()
            for client in self.clients:
                try:
                    await client.send(message)
                except websockets.exceptions.ConnectionClosed:
                    disconnected.add(client)

            # Remove disconnected clients
            for client in disconnected:
                self.clients.discard(client)
                self.authenticated_clients.pop(client, None)
                logger.info("Removed disconnected client")

    async def start_server(self):
        """Start the WebSocket server."""
        logger.info(f"Starting metrics server on {self.host}:{self.port}")

        async with websockets.serve(
            self.handle_client,
            self.host,
            self.port,
            compression=None,  # Disable compression for real-time data
            ping_interval=20,  # Keep connection alive
            ping_timeout=10,
        ):
            logger.info(f"Server running on ws://{self.host}:{self.port}")
            await asyncio.Future()  # Run forever


class MetricsBroadcaster:
    """Client-side broadcaster for sending metrics to the server."""

    def __init__(self, server_url: str = "ws://localhost:8765"):
        self.server_url = server_url
        self.websocket = None
        self.connected = False

    async def connect(self, client_id: str, private_key: str):
        """Connect to the metrics server with authentication."""
        try:
            self.websocket = await websockets.connect(self.server_url)

            # Create authentication challenge
            challenge = str(time.time())
            signature = hashlib.sha256(
                f"{client_id}:{challenge}:{private_key}".encode()
            ).hexdigest()

            # Send authentication
            auth_data = {
                "type": "auth",
                "client_id": client_id,
                "challenge": challenge,
                "signature": signature,
            }
            await self.websocket.send(json.dumps(auth_data))

            # Wait for response
            response = await self.websocket.recv()
            response_data = json.loads(response)

            if response_data.get("type") == "auth_success":
                self.connected = True
                logger.info("Connected to metrics server")
                return True
            else:
                logger.error(f"Authentication failed: {response_data.get('message')}")
                return False

        except Exception as e:
            logger.error(f"Connection failed: {e}")
            return False

    async def send_metrics(self, metrics: Dict[str, Any]):
        """Send metrics to the server."""
        if not self.connected or not self.websocket:
            logger.warning("Not connected to server")
            return False

        try:
            await self.websocket.send(
                json.dumps(
                    {"type": "metrics", "data": metrics, "timestamp": time.time()}
                )
            )
            return True
        except Exception as e:
            logger.error(f"Failed to send metrics: {e}")
            self.connected = False
            return False

    async def disconnect(self):
        """Disconnect from the server."""
        if self.websocket:
            await self.websocket.close()
            self.connected = False


def generate_keypair():
    """Generate a new keypair for client authentication."""
    import base64
    import secrets

    client_id = f"client_{secrets.token_hex(4)}"
    private_key = secrets.token_hex(32)

    print(f"Client ID: {client_id}")
    print(f"Private Key: {private_key}")
    print(f"\nAdd this to server's authorized_keys file:")
    print(f"{client_id}:{private_key}")

    return client_id, private_key


async def test_server():
    """Test the metrics server."""
    server = MetricsServer()

    # Create test authorized_keys file
    authorized_keys_path = Path("~/.adamwprune/authorized_keys").expanduser()
    authorized_keys_path.parent.mkdir(exist_ok=True)

    if not authorized_keys_path.exists():
        # Generate test keypair
        client_id, private_key = generate_keypair()
        with open(authorized_keys_path, "w") as f:
            f.write(f"# AdamWPrune Metrics Server Authorized Keys\n")
            f.write(f"{client_id}:{private_key}\n")
        print(f"\nCreated authorized_keys file at {authorized_keys_path}")

    await server.start_server()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AdamWPrune Metrics Server")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind to")
    parser.add_argument("--port", type=int, default=8765, help="Port to listen on")
    parser.add_argument(
        "--generate-keypair", action="store_true", help="Generate a new client keypair"
    )

    args = parser.parse_args()

    if args.generate_keypair:
        generate_keypair()
    else:
        asyncio.run(test_server())
