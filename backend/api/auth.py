from __future__ import annotations

import os
from dataclasses import dataclass

import jwt
from fastapi import Depends, HTTPException
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from jwt import PyJWKClient

from .dependencies import get_jwks_client

SUPABASE_ISSUER = f"{os.environ['SUPABASE_URL']}/auth/v1"

# auto_error=False so a missing header raises our own 401 below (matching the
# invalid-token case) instead of HTTPBearer's default 403. Registering this as
# the dependency's security scheme is also what makes /docs show an "Authorize"
# button instead of a plain, easy-to-miss header input field.
bearer_scheme = HTTPBearer(auto_error=False)


@dataclass(frozen=True)
class AuthUser:
    id: str
    email: str | None


def get_current_user(
    credentials: HTTPAuthorizationCredentials | None = Depends(bearer_scheme),
    jwks_client: PyJWKClient = Depends(get_jwks_client),
) -> AuthUser:
    if credentials is None:
        raise HTTPException(401, "Missing or malformed Authorization header")
    token = credentials.credentials

    try:
        # signing_key.algorithm_name comes from the JWKS entry itself, not the
        # token's own header — trusting the token's declared alg would let a
        # forged token pick whichever algorithm is easiest to forge.
        signing_key = jwks_client.get_signing_key_from_jwt(token)
        payload = jwt.decode(
            token,
            signing_key.key,
            algorithms=[signing_key.algorithm_name],
            audience="authenticated",
            issuer=SUPABASE_ISSUER,
        )
    except jwt.PyJWTError:
        raise HTTPException(401, "Invalid or expired token")

    return AuthUser(id=payload["sub"], email=payload.get("email"))
