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
import re
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


_EXPANSION = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)(?::-([^}]*))?\}")


def _expand(value: str, so_far: dict[str, str]) -> str:
    """Substitute ``${VAR}`` and ``${VAR:-default}``.

    python-dotenv expands inside single quotes too, unlike a POSIX shell,
    so this is applied to every value regardless of quoting. An unset name
    with no default becomes empty, which is also what python-dotenv does --
    leaving the literal ``${VAR}`` would let a placeholder be sized as if
    it were a model name.
    """

    def replace(match: re.Match[str]) -> str:
        name, default = match.group(1), match.group(2)
        if name in so_far and so_far[name]:
            return so_far[name]
        from_env = os.environ.get(name)
        if from_env:
            return from_env
        return default or ""

    return _EXPANSION.sub(replace, value)


def _parse_value(raw: str, so_far: dict[str, str]) -> str:
    """One ``KEY=`` right-hand side, the way python-dotenv reads it.

    Quoted values keep their interior padding and lose anything after the
    closing quote; unquoted values lose a ``#`` comment only when it is
    preceded by whitespace, so ``has#hash`` survives intact.
    """
    text = raw.lstrip()
    if text[:1] in ('"', "'"):
        quote = text[0]
        end = text.find(quote, 1)
        if end != -1:
            return _expand(text[1:end], so_far)
        # Unterminated quote: fall through and treat it as unquoted rather
        # than silently returning the rest of the line as a quoted value.
        text = text[1:]
    comment = re.search(r"(?:^|\s)#", text)
    if comment:
        text = text[: comment.start()]
    return _expand(text.strip(), so_far)


def read_env_file(path: str | os.PathLike[str]) -> dict[str, str]:
    """``KEY=VALUE`` pairs from an env file. Missing file raises OSError.

    Parsing follows python-dotenv rather than being merely approximate:
    this module exists to predict what ``load_dotenv`` will hand the run,
    and a parser that disagreed about a value would, in the words of the
    module docstring, bless a model the run would not load.

    Not modelled: backslash escapes inside double quotes. A model name or
    endpoint has no use for them, and guessing at half of an escape
    grammar would be worse than not claiming it.
    """
    values: dict[str, str] = {}
    for raw in Path(path).read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export ") :].lstrip()
        key, sep, value = line.partition("=")
        if not sep:
            continue
        values[key.strip()] = _parse_value(value, values)
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
