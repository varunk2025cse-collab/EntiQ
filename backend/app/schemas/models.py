"""Request / response schemas."""
from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

ID_PATTERN = r"^[A-Za-z0-9][A-Za-z0-9_\-.]{0,63}$"


class RunRequest(BaseModel):
    model_config = ConfigDict(protected_namespaces=())
    dataset_id: str = Field(pattern=ID_PATTERN)
    split: str = Field(default="test", pattern=r"^(train|test)$")
    model_version: str | None = Field(default=None, pattern=ID_PATTERN)


class TrainRequest(BaseModel):
    model: str | None = Field(default=None, pattern=r"^[a-z0-9_]{1,40}$")
    promote: bool = False


class PromoteRequest(BaseModel):
    version: str = Field(pattern=ID_PATTERN)
