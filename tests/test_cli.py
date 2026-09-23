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
    def __init__(
        self,
        sizes=None,
        loaded=None,
        ps_error=None,
        capabilities=None,
        generate=None,
        ps_error_after=None,
        show=None,
    ):
        self._sizes = sizes or {}
        self._loaded = loaded or []
        self._ps_error = ps_error
        # `stop` reads resident state twice - once to decide what to unload,
        # once to report what is left. This fails only the later call, which
        # is the case where the unload itself succeeded.
        self._ps_error_after = ps_error_after
        self._capabilities = capabilities or ["completion"]
        self._generate = generate or {
            "eval_count": 100,
            "eval_duration": 2_000_000_000,
            "prompt_eval_count": 10,
            "prompt_eval_duration": 100_000_000,
            "total_duration": 2_200_000_000,
        }
        self.ps_calls = 0
        self._show = show or {}
        self.show_calls = []

    def list_models(self):
        return [ModelInfo(n, s) for n, s in sorted(self._sizes.items(), key=lambda kv: -kv[1])]

    def loaded_models(self):
        self.ps_calls += 1
        if self._ps_error:
            raise self._ps_error
        if self._ps_error_after and self.ps_calls > self._ps_error_after:
            raise OllamaUnavailable("server went away mid-unload")
        return self._loaded

    def capabilities(self, model):
        return self._capabilities

    def show(self, model):
        self.show_calls.append(model)
        payload = self._show.get(model)
        if isinstance(payload, Exception):
            raise payload
        return payload or {}

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
    # Issue #16: the `if live:` guard is the behaviour the fix introduced,
    # and the warning alone does not pin it. Move that print outside the
    # guard and this would report "GPU offload: 0%" -- a figure invented
    # from a read that never returned -- while still passing everything
    # above.
    # "GPU offload:" with the colon, because the warning above says
    # "...re-read GPU offload afterwards" and would match a looser needle.
    assert "GPU offload:" not in out
    # Issue #18: the cause reaches the user. "could not re-read" reads the
    # same whether the server died, refused the connection or returned a
    # 500, and those want different responses.
    assert "server went away" in out


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


def test_stop_reports_why_the_final_read_failed(wire, capsys):
    """Issue #18, and the `stop` half of #14's reasoning: the unload
    succeeded, so this still exits 0 - but it says what went wrong rather
    than claiming 0.00 GB is held, which it has no basis for."""
    wire(StubClient(loaded=[LoadedModel("m", 1000, 1000)], ps_error_after=1))
    assert run(["stop", "--all"]) == cli.OK
    out = capsys.readouterr().out
    assert "could not re-read resident state" in out
    assert "server went away mid-unload" in out
    # The false claim #17 would have introduced by collapsing "read failed"
    # into an empty list.
    assert "still held by 0 resident" not in out


def test_ps_with_nothing_resident(wire, capsys):
    wire(StubClient())
    assert run(["ps"]) == cli.OK
    assert "Nothing resident" in capsys.readouterr().out


def test_coder_model_picks_7b_for_single_8gb_card(wire, capsys):
    wire(StubClient(), gpus=ONE_CARD)
    assert run(["coder-model"]) == cli.OK
    assert "coder model: qwen2.5-coder:7b" in capsys.readouterr().out


def test_coder_model_picks_32b_for_both_egpus(wire, capsys):
    egpu = Gpu(1, "RTX 5060 Ti", 16311 * MIB, 16000 * MIB)
    twin = Gpu(2, "RTX 5060 Ti (twin)", 16311 * MIB, 16000 * MIB)
    wire(StubClient(), gpus=[egpu, twin])
    assert run(["coder-model"]) == cli.OK
    assert "coder model: qwen2.5-coder:32b" in capsys.readouterr().out


def test_coder_model_cli_agrees_with_the_library_function(wire, capsys):
    """The CLI's `_describe_gpus`-derived budget and the library's
    `budget_bytes` must not diverge -- pin them to the same output."""
    from ollama_tools.coder_model import get_coder_model
    from ollama_tools.gpu import CONSERVATIVE

    gpus = [Gpu(0, "RTX 5070 Laptop", 8151 * MIB, 7700 * MIB), Gpu(1, "RTX 5060 Ti", 16311 * MIB, 16000 * MIB)]
    wire(StubClient(), gpus=gpus)
    assert run(["coder-model"]) == cli.OK
    out = capsys.readouterr().out
    assert f"coder model: {get_coder_model(CONSERVATIVE, gpus)}" in out


def test_coder_model_accepts_proportional_strategy(wire, capsys):
    gpus = [Gpu(0, "RTX 5070 Laptop", 8151 * MIB, 7700 * MIB), Gpu(1, "RTX 5060 Ti", 16311 * MIB, 16000 * MIB)]
    wire(StubClient(), gpus=gpus)
    assert run(["coder-model", "--strategy", "proportional"]) == cli.OK
    # 7700 + 16000 MiB free, proportional sum clears the 18 GiB top tier.
    assert "coder model: qwen2.5-coder:32b" in capsys.readouterr().out


def test_coder_model_falls_back_when_gpu_unavailable(monkeypatch, capsys):
    def raise_unavailable():
        raise cli.GpuUnavailable("nvidia-smi not found")

    monkeypatch.setattr(cli, "query_gpus", raise_unavailable)
    assert run(["coder-model"]) == cli.OK
    assert "qwen2.5-coder:0.5b" in capsys.readouterr().out


def test_general_model_picks_9b_for_single_8gb_card(wire, capsys):
    wire(StubClient(), gpus=ONE_CARD)
    assert run(["general-model"]) == cli.OK
    assert "general model: qwen3.5:9b" in capsys.readouterr().out


def test_general_model_picks_qwen3_216k_for_both_egpus(wire, capsys):
    egpu = Gpu(1, "RTX 5060 Ti", 16311 * MIB, 16000 * MIB)
    twin = Gpu(2, "RTX 5060 Ti (twin)", 16311 * MIB, 16000 * MIB)
    wire(StubClient(), gpus=[egpu, twin])
    assert run(["general-model"]) == cli.OK
    assert "general model: qwen3.8-216k" in capsys.readouterr().out


def test_general_model_falls_back_when_gpu_unavailable(monkeypatch, capsys):
    def raise_unavailable():
        raise cli.GpuUnavailable("nvidia-smi not found")

    monkeypatch.setattr(cli, "query_gpus", raise_unavailable)
    assert run(["general-model"]) == cli.OK
    assert "gemma3:4b" in capsys.readouterr().out


def test_general_model_cli_agrees_with_the_library_function(wire, capsys):
    """The CLI's `_describe_gpus`-derived budget and the library's
    `budget_bytes` must not diverge -- pin them to the same output."""
    from ollama_tools.general_model import get_general_model
    from ollama_tools.gpu import CONSERVATIVE

    gpus = [Gpu(0, "RTX 5070 Laptop", 8151 * MIB, 7700 * MIB), Gpu(1, "RTX 5060 Ti", 16311 * MIB, 16000 * MIB)]
    wire(StubClient(), gpus=gpus)
    assert run(["general-model"]) == cli.OK
    out = capsys.readouterr().out
    assert f"general model: {get_general_model(CONSERVATIVE, gpus)}" in out


def test_general_model_accepts_proportional_strategy(wire, capsys):
    gpus = [Gpu(0, "RTX 5070 Laptop", 8151 * MIB, 7700 * MIB), Gpu(1, "RTX 5060 Ti", 16311 * MIB, 16000 * MIB)]
    wire(StubClient(), gpus=gpus)
    assert run(["general-model", "--strategy", "proportional"]) == cli.OK
    # 7700 + 16000 MiB free, proportional sum clears the 18 GiB top tier.
    assert "general model: qwen3.8-216k" in capsys.readouterr().out


def test_endpoint_is_accepted_after_the_subcommand(wire):
    """`ollama-tools stop --all --endpoint X` is what people type; argparse
    rejects a parent-only flag there. Both orders must work."""
    wire(StubClient())
    assert run(["--endpoint", "http://host:1", "ps"]) == cli.OK
    assert run(["ps", "--endpoint", "http://host:1"]) == cli.OK


def test_subcommand_endpoint_does_not_clobber_the_parent_value():
    """The SUPPRESS default matters: without it the subparser's own default
    would overwrite an endpoint given before the subcommand."""
    args = cli.build_parser().parse_args(["--endpoint", "http://given:1", "ps"])
    assert args.endpoint == "http://given:1"


def test_endpoint_defaults_when_given_nowhere():
    assert cli.build_parser().parse_args(["ps"]).endpoint == "http://localhost:11434"


def test_bench_says_so_when_the_model_is_no_longer_resident(wire, capsys):
    """Issue #21: read succeeded, model gone. Distinct from a failed read,
    which prints the other warning -- both used to be an omitted offload
    line and nothing else."""
    wire(StubClient({"m": SEVEN_B}, loaded=[]))
    assert run(["bench", "m"]) == cli.OK
    out = capsys.readouterr().out
    assert "no longer resident" in out
    assert "GPU offload:" not in out
    # Not the read-failure warning: the read worked.
    assert "Could not re-read" not in out


def test_bench_read_failure_and_absent_model_say_different_things(wire, capsys):
    """The pair that motivated #21 -- the two paths must not be mistakable
    for each other."""
    wire(StubClient({"m": SEVEN_B}, ps_error=OllamaUnavailable("boom")))
    run(["bench", "m"])
    failed = capsys.readouterr().out

    wire(StubClient({"m": SEVEN_B}, loaded=[]))
    run(["bench", "m"])
    absent = capsys.readouterr().out

    assert "Could not re-read" in failed and "no longer resident" not in failed
    assert "no longer resident" in absent and "Could not re-read" not in absent


# --parallel: the KV cache is computed per slot and multiplied.

FOURTEEN_B = int(8.37 * GIB)
BOTH_CARDS = [
    Gpu(0, "RTX 5070 Laptop", 8151 * MIB, 7700 * MIB),
    Gpu(1, "RTX 5060 Ti", 16311 * MIB, 16000 * MIB),
]
QWEN25_14B_SHOW = {
    "model_info": {
        "general.architecture": "qwen2",
        "qwen2.block_count": 48,
        "qwen2.attention.head_count": 40,
        "qwen2.attention.head_count_kv": 8,
        "qwen2.embedding_length": 5120,
        "qwen2.context_length": 32768,
    }
}
HYBRID_SHOW = {
    "model_info": {
        "general.architecture": "qwen35",
        "qwen35.block_count": 64,
        "qwen35.full_attention_interval": 4,
    }
}


def fourteen_b(**extra):
    return StubClient({"qwen2.5-coder:14b": FOURTEEN_B}, show={"qwen2.5-coder:14b": QWEN25_14B_SHOW}, **extra)


def test_without_parallel_the_kv_cache_is_not_consulted(wire):
    """The default check is unchanged, and must not need /api/show."""
    client = wire(fourteen_b(), gpus=BOTH_CARDS)
    assert run(["fit", "qwen2.5-coder:14b", "--strategy", "proportional"]) == cli.OK
    assert client.show_calls == []


def test_parallel_slots_that_fit(wire, capsys):
    """10.04 GB of weights+headroom plus 7 x 1.69 GB q4_0 slots = 21.9 GB."""
    wire(fourteen_b(), gpus=BOTH_CARDS)
    argv = ["fit", "qwen2.5-coder:14b", "--strategy", "proportional", "--parallel", "7", "--kv-cache-type", "q4_0"]
    assert run(argv) == cli.OK
    out = capsys.readouterr().out
    assert "7 x 1.69 GB KV" in out
    assert "OLLAMA_NUM_PARALLEL" in out


def test_too_many_parallel_slots_do_not_fit(wire):
    wire(fourteen_b(), gpus=BOTH_CARDS)
    argv = ["fit", "qwen2.5-coder:14b", "--strategy", "proportional", "--parallel", "9", "--kv-cache-type", "q4_0"]
    assert run(argv) == cli.DOES_NOT_FIT


def test_kv_cache_type_defaults_to_the_largest(wire):
    """f16 is the conservative default: 4 x 6 GB slots cannot fit."""
    wire(fourteen_b(), gpus=BOTH_CARDS)
    assert run(["fit", "qwen2.5-coder:14b", "--strategy", "proportional", "--parallel", "4"]) == cli.DOES_NOT_FIT


def test_num_ctx_alone_means_one_slot(wire, capsys):
    client = wire(fourteen_b(), gpus=BOTH_CARDS)
    assert run(["fit", "qwen2.5-coder:14b", "--num-ctx", "8192"]) == cli.OK
    assert client.show_calls == ["qwen2.5-coder:14b"]
    assert "1 x 1.50 GB KV (8,192 ctx per slot)" in capsys.readouterr().out


def test_hybrid_architecture_cannot_be_answered(wire, capsys):
    wire(StubClient({"qwen3.8-216k:latest": TWENTY_SEVEN_B}, show={"qwen3.8-216k:latest": HYBRID_SHOW}), gpus=BOTH_CARDS)
    assert run(["fit", "qwen3.8-216k:latest", "--parallel", "2"]) == cli.CANNOT_ANSWER
    assert "measure-context-ceiling" in capsys.readouterr().err


def test_a_model_that_does_not_fit_outranks_one_that_cannot_be_sized(wire):
    wire(
        StubClient(
            {"qwen2.5-coder:14b": FOURTEEN_B, "hybrid": TWENTY_SEVEN_B},
            show={"qwen2.5-coder:14b": QWEN25_14B_SHOW, "hybrid": HYBRID_SHOW},
        ),
        gpus=BOTH_CARDS,
    )
    assert run(["fit", "qwen2.5-coder:14b", "hybrid", "--parallel", "4"]) == cli.DOES_NOT_FIT


def test_show_failing_is_cannot_answer(wire):
    wire(StubClient({"m": SEVEN_B}, show={"m": OllamaUnavailable("boom")}))
    assert run(["fit", "m", "--parallel", "2"]) == cli.CANNOT_ANSWER


def test_start_refuses_when_the_slots_do_not_fit(wire):
    wire(fourteen_b(), gpus=BOTH_CARDS)
    assert run(["start", "qwen2.5-coder:14b", "--parallel", "4"]) == cli.DOES_NOT_FIT


@pytest.mark.parametrize("command", ["start", "bench"])
def test_num_ctx_is_fit_only(command):
    """start and bench load at the manifest's num_ctx; an override there
    would pass a check the real load then fails."""
    with pytest.raises(SystemExit):
        cli.build_parser().parse_args([command, "m", "--num-ctx", "8192"])


@pytest.mark.parametrize("value", ["0", "-1", "two"])
def test_parallel_must_be_a_positive_integer(value):
    with pytest.raises(SystemExit):
        cli.build_parser().parse_args(["fit", "m", "--parallel", value])
