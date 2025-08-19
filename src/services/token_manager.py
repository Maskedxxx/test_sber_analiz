import time
import uuid
from dataclasses import dataclass
from typing import Optional

import requests

from utils.config import config


@dataclass
class _Token:
    access_token: str
    expires_at: float  # epoch seconds


class GigaChatTokenManager:
    def __init__(self, *, timeout: int = 10):
        self._timeout = timeout
        self._token: Optional[_Token] = None

    def _fetch(self) -> _Token:
        headers = {
            "Content-Type": "application/x-www-form-urlencoded",
            "Accept": "application/json",
            "RqUID": str(uuid.uuid4()),
            "Authorization": f"Basic {config.gigachat_auth_key}",
        }
        data = {"scope": config.gigachat_scope}
        resp = requests.post(
            config.gigachat_auth_url,
            headers=headers,
            data=data,
            timeout=self._timeout,
            verify=config.gigachat_verify_ssl,
        )
        resp.raise_for_status()
        payload = resp.json()
        access_token = payload["access_token"]
        # expires_at может присутствовать; иначе берём 30 мин. (минус малая дельта в get())
        expires_at = float(payload.get("expires_at", time.time() + 1799))
        return _Token(access_token=access_token, expires_at=expires_at)

    def get(self) -> str:
        # Обновляем заранее за 60 секунд до истечения
        if self._token and (self._token.expires_at - 60) > time.time():
            return self._token.access_token
        self._token = self._fetch()
        return self._token.access_token

