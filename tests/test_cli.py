"""CLI behaviour, driven through the real argument parser.

The exit codes are the documented contract, so they are what these assert
on -- 0 fine, 1 does not fit, 2 could not be established.
"""

import pytest

from ollama_tools import cli
from ollama_tools.client import LoadedModel, ModelInfo, OllamaUnavailable
from ollama_tools.gpu import GIB, Gpu

MIB = 1024 * 1024
ONE_CARD = [Gpu(0, "RTX 5070 Laptop", 8151 * MIB, 7700 * MIB)]

SEVEN_B = int(4.36 * GIB)
TWENTY_SEVEN_B = int(11.29 * GIB)


class StubClient:
    def __init__(self, sizes=None, loaded=None, ps_error=None, capabilities=None, generate=None):
        self._sizes = sizes or {}
        self._loaded = loaded or []
        self._ps_error = ps_error
        self._capabilities = capabilities or ["completion"]
        self._generate = generate or {
            "eval_count": 100,
            "eval_duration": 2_000_000_000,
            "prompt_eval_count": 10,
            "prompt_eval_duration": 100_000_000,
            "total_duration": 2_200_000_000,
        }
        self.ps_calls = 0

    def list_models(self):
        return [ModelInfo(n, s) for n, s in sorted(self._sizes.items(), key=lambda kv: -kv[1])]

    def loaded_models(self):
        self.ps_calls += 1
        if self._ps_error:
            raise self._ps_error
        return self._loaded

    def capabilities(self, model):
        return self._capabilities

    def generate(self, model, prompt, num_predict, timeout=600):
        return self._generate

    def embed(self, model, text, timeout=600):
        return {"embeddings": [[0.0] * 768]}

    def unload(self, model, timeout=60):
        pass


@pytest.fixture
def wire(monkeypatch):
    """Point the CLI at a stub client and a fixed GPU inventory."""

    def _wire(client, gpus=None):
        monkeypatch.setattr(cli, "query_gpus", lambda: list(gpus or ONE_CARD))
        monkeypatch.setattr(cli, "OllamaClient", lambda *a, **k: client)
        return client

    return _wire


def run(argv):
    return cli.main(argv)


def test_model_that_fits_exits_zero(wire):
    wire(StubClient({"qwen2.5-coder:7b": SEVEN_B}))
    assert run(["fit", "qwen2.5-coder:7b"]) == cli.OK


def test_model_that_does_not_fit_exits_one(wire):
    wire(StubClient({"qwen3.8-64k:latest": TWENTY_SEVEN_B}))
    assert run(["fit", "qwen3.8-64k:latest"]) == cli.DOES_NOT_FIT


def test_survey_reports_oversized_models_without_failing(wire):
    """A survey asking 'what can I run?' finding big models is the expected
    answer, not an error."""
    wire(StubClient({"big": TWENTY_SEVEN_B, "small": SEVEN_B}))
    assert run(["fit"]) == cli.OK


def test_all_unpulled_models_are_reported_together(wire, capsys):
    """Issue #10: the first miss used to abort, hiding every other model."""
    wire(StubClient({"pulled": SEVEN_B}))
    code = run(["fit", "missing-a", "pulled", "missing-b"])
    assert code == cli.CANNOT_ANSWER
    out = capsys.readouterr()
    combined = out.out + out.err
    assert "missing-a" in combined and "missing-b" in combined
    # ...and the model that *is* pulled still got a verdict.
    assert "fits" in out.out


def test_a_model_that_does_not_fit_outranks_one_that_is_unpulled(wire):
    """Both refuse, but a definite hazard is the more actionable report."""
    wire(StubClient({"toobig": TWENTY_SEVEN_B}))
    assert run(["fit", "toobig", "missing"]) == cli.DOES_NOT_FIT


def test_gpu_unavailable_is_cannot_answer_not_does_not_fit(wire, monkeypatch):
    wire(StubClient({"m": SEVEN_B}))
    monkeypatch.setattr(
        cli, "query_gpus", lambda: (_ for _ in ()).throw(cli.GpuUnavailable("no nvidia-smi"))
    )
    assert run(["fit", "m"]) == cli.CANNOT_ANSWER


def test_server_down_is_cannot_answer(wire):
    class Dead(StubClient):
        def list_models(self):
            raise OllamaUnavailable("connection refused")

    wire(Dead())
    assert run(["fit", "m"]) == cli.CANNOT_ANSWER


def test_bench_keeps_its_results_when_the_post_run_read_fails(wire, capsys):
    """Issue #14: the benchmark ran and printed timings. A server that dies
    before the offload re-read costs us that figure, not the run -- the
    same reasoning `stop` already applied to its own re-read."""
    client = wire(
        StubClient({"m": SEVEN_B}, ps_error=OllamaUnavailable("server went away"))
    )
    code = run(["bench", "m"])
    out = capsys.readouterr().out
    assert code == cli.OK
    assert "50.0 tok/s" in out
    assert "the timings above stand" in out
    assert client.ps_calls  # the re-read really was attempted


def test_bench_still_fails_when_the_benchmark_itself_fails(wire):
    """The contract the fix must not weaken."""

    class Failing(StubClient):
        def generate(self, model, prompt, num_predict, timeout=600):
            raise OllamaUnavailable("died mid-run")

    wire(Failing({"m": SEVEN_B}))
    assert run(["bench", "m"]) == cli.CANNOT_ANSWER


def test_bench_reports_offload_when_the_read_succeeds(wire, capsys):
    wire(StubClient({"m": SEVEN_B}, loaded=[LoadedModel("m", 1000, 800)]))
    assert run(["bench", "m"]) == cli.OK
    assert "80%" in capsys.readouterr().out


def test_bench_of_an_embedding_model_uses_the_embed_endpoint(wire, capsys):
    """An embedding model has no generate endpoint at all; the stub's
    generate() would return completion timings if it were wrongly used."""
    wire(StubClient({"nomic-embed-text": 280 * MIB}, capabilities=["embedding"]))
    assert run(["bench", "nomic-embed-text"]) == cli.OK
    out = capsys.readouterr().out
    assert "768 dims" in out
    assert "tok/s" not in out


def test_stop_without_a_target_is_a_usage_error(wire):
    wire(StubClient(loaded=[LoadedModel("m", 1000, 1000)]))
    assert run(["stop"]) == cli.CANNOT_ANSWER


def test_stop_all_unloads_everything_resident(wire, capsys):
    wire(StubClient(loaded=[LoadedModel("m", 1000, 1000)]))
    assert run(["stop", "--all"]) == cli.OK
    assert "Unloading m" in capsys.readouterr().out


def test_ps_with_nothing_resident(wire, capsys):
    wire(StubClient())
    assert run(["ps"]) == cli.OK
    assert "Nothing resident" in capsys.readouterr().out
