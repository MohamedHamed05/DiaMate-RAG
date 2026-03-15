import json
import logging
import redis

logger = logging.getLogger('uvicorn.error')

KEY_PREFIX = "chat:"


class RedisStore:
    def __init__(self, host: str, port: int, db: int = 0, ttl: int = 86400):
        self.ttl = ttl
        try:
            self.client = redis.Redis(host=host, port=port, db=db, decode_responses=True)
            self.client.ping()
            self.connected = True
        except redis.ConnectionError:
            logger.warning("Redis is not available. Chat memory will be disabled.")
            self.client = None
            self.connected = False

    def _key(self, session_id: str) -> str:
        return f"{KEY_PREFIX}{session_id}"

    def get_history(self, session_id: str) -> list[dict]:
        if not self.connected:
            return []
        try:
            items = self.client.lrange(self._key(session_id), 0, -1)
            return [json.loads(item) for item in items]
        except Exception as e:
            logger.warning(f"Failed to get chat history: {e}")
            return []

    def get_recent_history(self, session_id: str, last_n: int) -> list[dict]:
        if not self.connected:
            return []
        try:
            items = self.client.lrange(self._key(session_id), -last_n, -1)
            return [json.loads(item) for item in items]
        except Exception as e:
            logger.warning(f"Failed to get recent chat history: {e}")
            return []

    def add_message(self, session_id: str, role: str, content: str):
        if not self.connected:
            return
        try:
            key = self._key(session_id)
            self.client.rpush(key, json.dumps({"role": role, "content": content}))
            self.client.expire(key, self.ttl)
        except Exception as e:
            logger.warning(f"Failed to save chat message: {e}")

    def clear_history(self, session_id: str) -> bool:
        if not self.connected:
            return False
        try:
            self.client.delete(self._key(session_id))
            return True
        except Exception as e:
            logger.warning(f"Failed to clear chat history: {e}")
            return False

    def get_sessions(self) -> list[str]:
        if not self.connected:
            return []
        try:
            sessions = []
            cursor = 0
            while True:
                cursor, keys = self.client.scan(cursor, match=f"{KEY_PREFIX}*", count=100)
                for key in keys:
                    sessions.append(key[len(KEY_PREFIX):])
                if cursor == 0:
                    break
            return sessions
        except Exception as e:
            logger.warning(f"Failed to list sessions: {e}")
            return []
