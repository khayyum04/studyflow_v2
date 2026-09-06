from __future__ import annotations

import os
from dataclasses import dataclass

import jwt
from fastapi import Depends, Header, HTTPException
from jwt import PyJWKClient

from .dependencies import get_jwks_client

SUPABASE_ISSUER = f"{os.environ['SUPABASE_URL']}/auth/v1"


@dataclass(frozen=True)
class AuthUser:
    id: str
    email: str | None


def get_current_user(
    authorization: str | None = Header(None),
    jwks_client: PyJWKClient = Depends(get_jwks_client),
) -> AuthUser:
    if authorization is None or not authorization.startswith("Bearer "):
        raise HTTPException(401, "Missing or malformed Authorization header")
    token = authorization.removeprefix("Bearer ")

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
