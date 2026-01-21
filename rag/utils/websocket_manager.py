import asyncio
import uuid
from fastapi import WebSocket
from starlette.websockets import WebSocketState

class ConnectionManager:
    def __init__(self):
        self.active_connections = {}
        self.lock = asyncio.Lock()
    
    async def connect(self, websocket: WebSocket, connection_id: str):
        async with self.lock:
            await websocket.accept()
            self.active_connections[connection_id] = websocket
    
    async def disconnect(self, connection_id: str):
        async with self.lock:
            if connection_id in self.active_connections:
                del self.active_connections[connection_id]
    
    async def get_connection(self, connection_id: str):
        async with self.lock:
            return self.active_connections.get(connection_id)

# 创建全局连接管理器实例
manager = ConnectionManager()