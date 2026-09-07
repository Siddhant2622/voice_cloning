"""
api/auth.py
===========
Optional Bearer token API key authentication.
"""
from __future__ import annotations

from fastapi import Depends, HTTPException, Security, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from api.config import get_settings

_bearer = HTTPBearer(auto_error=False)


def require_api_key(
    credentials: HTTPAuthorizationCredentials | None = Security(_bearer),
) -> None:
    """
    FastAPI dependency. Raises 401 if API key authentication is enabled
    and the provided Bearer token is not in the valid key list.
    """
    settings = get_settings()
    if not settings.api_key_required:
        return  # Auth disabled — allow everything

    if credentials is None or credentials.scheme.lower() != "bearer":
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing or invalid Authorization header. Use: Authorization: Bearer <api_key>",
            headers={"WWW-Authenticate": "Bearer"},
        )

    if credentials.credentials not in settings.valid_api_keys:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Invalid API key.",
        )
