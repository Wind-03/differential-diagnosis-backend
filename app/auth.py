"""JWT authentication and authorization helpers."""
from __future__ import annotations

import uuid
from datetime import datetime, timedelta, timezone

import jwt
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from pwdlib import PasswordHash
from sqlalchemy.orm import Session

from app.config import settings
from app.db import get_db
from app.models import User

password_hash = PasswordHash.recommended()
bearer = HTTPBearer(auto_error=False)


def hash_password(password: str) -> str:
    return password_hash.hash(password)


def verify_password(password: str, hashed: str) -> bool:
    try:
        return password_hash.verify(password, hashed)
    except Exception:
        return False


def create_access_token(user: User) -> str:
    now = datetime.now(timezone.utc)
    payload = {
        "sub": user.id,
        "type": "access",
        "role": user.role,
        "verification_status": user.verification_status,
        "iat": now,
        "exp": now + timedelta(minutes=settings.access_token_minutes),
    }
    return jwt.encode(payload, settings.jwt_secret, algorithm=settings.jwt_algorithm)


def get_current_user(
    credentials: HTTPAuthorizationCredentials | None = Depends(bearer),
    db: Session = Depends(get_db),
) -> User:
    if credentials is None or credentials.scheme.lower() != "bearer":
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Authentication required")
    try:
        payload = jwt.decode(
            credentials.credentials,
            settings.jwt_secret,
            algorithms=[settings.jwt_algorithm],
            options={"require": ["sub", "type", "iat", "exp"]},
        )
        if payload.get("type") != "access":
            raise ValueError("invalid token type")
        user_id = payload["sub"]
    except (jwt.PyJWTError, ValueError, KeyError, TypeError) as exc:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid or expired token") from exc

    user = db.get(User, user_id)
    if user is None:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="User no longer exists")
    return user


def require_practitioner(user: User = Depends(get_current_user)) -> User:
    if user.role != "practitioner":
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Practitioner access required")
    return user


def create_user(db: Session, email: str, password: str, name :str) -> User:
    user = User(
        id=str(uuid.uuid4()),
        email=email.strip().lower(),
        name=name.strip().lower(),
        password_hash=hash_password(password),
        role="practitioner",
        verification_status="self_attested",
    )
    db.add(user)
    db.commit()
    db.refresh(user)
    return user
