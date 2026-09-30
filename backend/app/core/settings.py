"""Backend settings (environment-driven)."""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from src.utils.config import PROJECT_ROOT, load_config, resolve_path


@dataclass
class Settings:
    workspace: str | None = field(default_factory=lambda: os.environ.get("ENTIQ_WORKSPACE"))
    database_url: str = ""
    max_upload_mb: int = int(os.environ.get("ENTIQ_MAX_UPLOAD_MB", "300"))
    cors_origins: list[str] = field(default_factory=lambda: os.environ.get(
        "ENTIQ_CORS", "http://localhost:8000,http://127.0.0.1:8000,http://localhost:5173").split(","))
    frontend_dir: Path = PROJECT_ROOT / "frontend"

    def __post_init__(self) -> None:
        base = Path(self.workspace) if self.workspace else PROJECT_ROOT / "data"
        base.mkdir(parents=True, exist_ok=True)
        self.database_url = os.environ.get("DATABASE_URL", f"sqlite:///{(base / 'entiq.db').as_posix()}")
        self.uploads_dir = base / "uploads"
        self.uploads_dir.mkdir(parents=True, exist_ok=True)

    @property
    def cfg(self) -> dict:
        return load_config()

    def path(self, key: str) -> Path:
        return resolve_path(self.cfg, key)

    @property
    def demo_mode(self) -> bool:
        return bool(self.workspace) and "demo" in Path(self.workspace).name.lower()


settings = Settings()
