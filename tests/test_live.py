from ollama_tools import live
from ollama_tools.client import LoadedModel, ModelInfo, OllamaUnavailable
from ollama_tools.gpu import GIB, Gpu

MIB = 1024 * 1024


def test_reclaim_splits_by_each_cards_usage():
    a = Gpu(0, "a", 8 * GIB, 2 * GIB)  # 6 GiB used
    b = Gpu(1, "b", 16 * GIB, 4 * GIB)  # 12 GiB used
    ra, rb = live.reclaim_resident([a, b], 9 * GIB)
    assert ra.free_bytes == 2 * GIB + 3 * GIB
    assert rb.free_bytes == 4 * GIB + 6 * GIB


def test_reclaim_never_exceeds_used():
    a = Gpu(0, "a", 8 * GIB, 6 * GIB)  # 2 GiB used
    (ra,) = live.reclaim_resident([a], 50 * GIB)
    assert ra.free_bytes == 8 * GIB


def test_reclaim_nothing_held_is_identity():
    a = Gpu(0, "a", 8 * GIB, 6 * GIB)
    assert live.reclaim_resident([a], 0) == [a]


def test_pick_tier_skips_unpulled_then_falls_back():
    tiers = ((10, "big"), (5, "mid"))
    assert live.pick_tier(tiers, "tiny", 20, frozenset({"mid:latest"})) == "mid"
    assert live.pick_tier(tiers, "tiny", 20, frozenset()) == "tiny"
    assert live.pick_tier(tiers, "tiny", 20, None) == "big"


class _Client:
    def __init__(self, endpoint):
        self.endpoint = endpoint

    def loaded_models(self):
        return [LoadedModel("m:latest", 10, 7), LoadedModel("n:latest", 5, 3)]

    def list_models(self):
        return [ModelInfo("qwen3.8-100k", 1), ModelInfo("qwen2.5-coder:7b", 1)]


class _DownClient:
    def __init__(self, endpoint):
        pass

    def loaded_models(self):
        raise OllamaUnavailable("down")

    list_models = loaded_models


def test_live_queries(monkeypatch):
    monkeypatch.setattr(live, "OllamaClient", _Client)
    assert live.resident_vram_bytes() == 10
    assert live.installed_models() == frozenset({"qwen3.8-100k:latest", "qwen2.5-coder:7b"})


def test_live_queries_degrade_when_ollama_is_down(monkeypatch):
    monkeypatch.setattr(live, "OllamaClient", _DownClient)
    assert live.resident_vram_bytes() == 0
    assert live.installed_models() is None


def test_endpoint_env(monkeypatch):
    monkeypatch.setenv("OLLAMA_ENDPOINT", "http://box:1")
    assert live._endpoint(None) == "http://box:1"
    monkeypatch.delenv("OLLAMA_ENDPOINT")
    assert live._endpoint(None) == "http://localhost:11434"
