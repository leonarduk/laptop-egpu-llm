"""A small Ollama HTTP client, stdlib only.

No `requests`: this has to run on a fresh Ubuntu box and on Windows with
nothing installed but Python, and everything needed is one JSON POST.
"""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from dataclasses import dataclass

DEFAULT_ENDPOINT = "http://localhost:11434"


class OllamaUnavailable(RuntimeError):
    """The server could not be reached, or refused the request.

    Like GpuUnavailable, this means the question could not be answered --
    which is not the same as the answer being "no", and callers map it to
    a different exit code.
    """


@dataclass(frozen=True)
class ModelInfo:
    name: str
    size_bytes: int

    @property
    def size_gib(self) -> float:
        return self.size_bytes / 1024**3


@dataclass(frozen=True)
class LoadedModel:
    """A model currently resident, and how much of it reached the GPU.

    ``size_vram_bytes`` below ``size_bytes`` means layers are on CPU. That
    is the number that explains a disappointing token rate, so it is
    carried everywhere speed is reported.
    """

    name: str
    size_bytes: int
    size_vram_bytes: int
    expires_at: str = ""

    @property
    def offload_fraction(self) -> float:
        if self.size_bytes <= 0:
            return 0.0
        return self.size_vram_bytes / self.size_bytes


class OllamaClient:
    def __init__(self, endpoint: str = DEFAULT_ENDPOINT, timeout: float = 30.0):
        self.endpoint = endpoint.rstrip("/")
        self.timeout = timeout

    def _request(self, path: str, payload: dict | None = None, timeout: float | None = None) -> dict:
        url = f"{self.endpoint}{path}"
        data = None if payload is None else json.dumps(payload).encode("utf-8")
        request = urllib.request.Request(
            url, data=data, headers={"Content-Type": "application/json"}
        )
        try:
            with urllib.request.urlopen(request, timeout=timeout or self.timeout) as response:
                body = response.read().decode("utf-8")
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", "replace").strip()
            raise OllamaUnavailable(f"{url} returned {exc.code}: {detail}") from exc
        except (urllib.error.URLError, OSError) as exc:
            raise OllamaUnavailable(f"{url} unreachable: {exc}") from exc

        try:
            parsed = json.loads(body)
        except json.JSONDecodeError as exc:
            raise OllamaUnavailable(f"{url} returned unparsable JSON: {exc}") from exc
        if not isinstance(parsed, dict):
            raise OllamaUnavailable(f"{url} returned {type(parsed).__name__}, not an object")
        return parsed

    def list_models(self) -> list[ModelInfo]:
        """Every pulled model, largest first."""
        payload = self._request("/api/tags", timeout=10)
        models = [
            ModelInfo(name=m.get("name", ""), size_bytes=int(m.get("size", 0)))
            for m in payload.get("models", [])
            if m.get("name")
        ]
        return sorted(models, key=lambda m: m.size_bytes, reverse=True)

    def loaded_models(self) -> list[LoadedModel]:
        payload = self._request("/api/ps", timeout=10)
        return [
            LoadedModel(
                name=m.get("name", ""),
                size_bytes=int(m.get("size", 0)),
                size_vram_bytes=int(m.get("size_vram", 0)),
                expires_at=str(m.get("expires_at", "")),
            )
            for m in payload.get("models", [])
            if m.get("name")
        ]

    def capabilities(self, model: str) -> list[str]:
        """What the model can do, e.g. ``["completion", "tools"]`` or
        ``["embedding"]``.

        Asked rather than inferred from the name: an embedding model has no
        ``generate`` endpoint at all, and guessing from "embed" appearing
        in a name would be wrong in both directions.
        """
        payload = self._request("/api/show", {"model": model}, timeout=20)
        caps = payload.get("capabilities")
        return [str(c) for c in caps] if isinstance(caps, list) else []

    def generate(self, model: str, prompt: str, num_predict: int, timeout: float = 600) -> dict:
        return self._request(
            "/api/generate",
            {
                "model": model,
                "prompt": prompt,
                "stream": False,
                "options": {"num_predict": num_predict},
            },
            timeout=timeout,
        )

    def embed(self, model: str, text: str, timeout: float = 600) -> dict:
        return self._request(
            "/api/embed", {"model": model, "input": text}, timeout=timeout
        )

    def unload(self, model: str, timeout: float = 60) -> None:
        """Evict a model from VRAM. The server stays up and nothing is
        deleted -- ``keep_alive: 0`` with an empty prompt is an eviction,
        not a very short request."""
        self._request(
            "/api/generate",
            {"model": model, "prompt": "", "keep_alive": 0},
            timeout=timeout,
        )
