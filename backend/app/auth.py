import base64
import hashlib
from datetime import UTC, datetime, timedelta

import bcrypt
import jwt


class AuthError(Exception):
    pass


def _prehash(password: str) -> bytes:
    # bcrypt only reads the first 72 bytes; pre-hashing removes that limit safely.
    return base64.b64encode(hashlib.sha256(password.encode()).digest())


def hash_password(password: str) -> str:
    return bcrypt.hashpw(_prehash(password), bcrypt.gensalt()).decode()


def verify_password(password: str, password_hash: str) -> bool:
    try:
        return bcrypt.checkpw(_prehash(password), password_hash.encode())
    except ValueError:
        return False


def create_token(user_id: int, secret: str, minutes: int = 720, algorithm: str = "HS256") -> str:
    now = datetime.now(UTC)
    payload = {"sub": str(user_id), "iat": now, "exp": now + timedelta(minutes=minutes)}
    return jwt.encode(payload, secret, algorithm=algorithm)


def decode_token(token: str, secret: str, algorithm: str = "HS256") -> int:
    try:
        payload = jwt.decode(
            token, secret, algorithms=[algorithm], options={"require": ["exp", "sub"]}
        )
        return int(payload["sub"])
    except (jwt.PyJWTError, ValueError, KeyError) as exc:
        raise AuthError("invalid or expired token") from exc
