"""Which models an issue-worm run would actually load.

Answers a different question from "what is pulled?". A config naming a 27B
through ``CODER_TARGETS`` looks harmless in a survey of pulled models and
is a hang waiting to happen when the job starts.

The precedence rules mirror issue-worm's own, or this would bless a model
the run would not load:

* ``CODER_TARGETS`` beats ``CODER_OLLAMA_MODEL`` -- that entry overwrites
  the coder's ``OLLAMA_ENDPOINT``/``OLLAMA_MODEL`` on every dispatch.
* A role whose source is not ``local`` is skipped: a cloud role cannot
  overcommit local VRAM.
* A real environment variable beats the file, matching
  ``load_dotenv(override=False)``.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

ROLES = ("coder", "analyser", "triage")
LOCAL_SOURCE = "local"


@dataclass(frozen=True)
class RoleModel:
    role: str
    model: str | None
    source: str
    note: str = ""

    @property
    def checkable(self) -> bool:
        return self.model is not None


def read_env_file(path: str | os.PathLike[str]) -> dict[str, str]:
    """``KEY=VALUE`` pairs from an env file. Missing file raises OSError."""
    values: dict[str, str] = {}
    for raw in Path(path).read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        key, sep, value = line.partition("=")
        if not sep:
            continue
        values[key.strip()] = value.strip().strip('"').strip("'")
    return values


def resolve(name: str, file_values: dict[str, str]) -> str | None:
    """A real environment variable, else the file, else None."""
    real = os.environ.get(name)
    if real and real.strip():
        return real.strip()
    value = file_values.get(name)
    if value and value.strip():
        return value.strip()
    return None


def coder_target_models(raw: str | None) -> list[str]:
    """Distinct model names from ``name:host:port:model`` entries.

    Split at most three times: an Ollama model name carries a ``:tag`` of
    its own, so everything after the port is the model.
    """
    models: list[str] = []
    for entry in (raw or "").split(","):
        parts = entry.strip().split(":", 3)
        if len(parts) == 4:
            model = parts[3].strip()
            if model and model not in models:
                models.append(model)
    return models


def role_models(file_values: dict[str, str]) -> list[RoleModel]:
    """One entry per role, in pipeline order."""
    resolved: list[RoleModel] = []
    for role in ROLES:
        prefix = role.upper()
        source = (resolve(f"{prefix}_MODEL_SOURCE", file_values) or LOCAL_SOURCE).lower()
        if source != LOCAL_SOURCE:
            resolved.append(
                RoleModel(role, None, source, "not local Ollama; cannot overcommit local VRAM")
            )
            continue

        models: list[str] = []
        if role == "coder":
            models = coder_target_models(resolve("CODER_TARGETS", file_values))
        if not models:
            single = resolve(f"{prefix}_OLLAMA_MODEL", file_values)
            if single:
                models = [single]
        if not models:
            resolved.append(
                RoleModel(role, None, source, "names no model; the runtime default applies and cannot be sized")
            )
            continue
        for model in models:
            resolved.append(RoleModel(role, model, source))
    return resolved


def models_to_check(file_values: dict[str, str]) -> list[str]:
    """Distinct local models a run would load, preserving role order."""
    seen: list[str] = []
    for entry in role_models(file_values):
        if entry.model and entry.model not in seen:
            seen.append(entry.model)
    return seen
