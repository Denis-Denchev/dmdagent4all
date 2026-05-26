from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol


@dataclass(frozen=True)
class SecretRef:
    connector: str
    name: str


class SecretStore(Protocol):
    def get(self, ref: SecretRef) -> bytes:
        """Return secret bytes without exposing them to model context."""

    def set(self, ref: SecretRef, value: bytes) -> None:
        """Store secret bytes in an OS-backed or encrypted secret store."""

    def delete(self, ref: SecretRef) -> None:
        """Delete secret bytes."""
