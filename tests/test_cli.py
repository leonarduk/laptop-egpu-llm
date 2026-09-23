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

# The real pair, as nvidia-smi reports it with nothing loaded.
LAPTOP_8GB = Gpu(0, "RTX 5070 Laptop", 8151 * MIB, 7700 * MIB)
EGPU_16GB = Gpu(1, "RTX 5060 Ti", 16311 * MIB, 15600 * MIB)

OLLAMA_ENV = (
    "OLLAMA_KV_CACHE_TYPE",
    "OLLAMA_FLASH_ATTENTION",
    "OLLAMA_NUM_PARALLEL",
    "OLLAMA_CONTEXT_LENGTH",
)


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
        # /api/show payloads by model name. Absent means an empty payload,
        # which carries no architecture: the fit check falls back to file
        # size plus headroom, as it did before the KV estimate existed.
        self._show = show or {}
        self.show_calls = []
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
        payload = self._show.get(model, {})
        if isinstance(payload, Exception):
            raise payload
        return payload

    def generate(self, model, prompt, num_predict, timeout=600):
        return self._generate

    def embed(self, model, text, timeout=600):
        return {"embeddings": [[0.0] * 768]}

    def unload(self, model, timeout=60):
        pass


@pytest.fixture
def wire(monkeypatch):
    """Point the CLI at a stub client and a fixed GPU inventory."""

    # The KV estimate reads these from the environment; the machine the
    # tests run on (this one sets q4_0) must not change the arithmetic.
    for var in OLLAMA_ENV:
        monkeypatch.delenv(var, raising=False)

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


def test_fit_shows_the_kv_cache_as_its_own_component(wire, capsys, monkeypatch, qwen35_show):
    """qwen3.8-100k on the real pair: 16.2 GiB of weights + 20% + the
    measured 1759.5 MiB cache. The cache is what the old file-size check
    could not see."""
    monkeypatch.setenv("OLLAMA_KV_CACHE_TYPE", "q4_0")
    monkeypatch.setenv("OLLAMA_FLASH_ATTENTION", "1")
    weights = int(16.2 * GIB)
    wire(
        StubClient({"qwen3.8-100k:latest": weights}, show={"qwen3.8-100k:latest": qwen35_show}),
        gpus=[LAPTOP_8GB, EGPU_16GB],
    )
    assert run(["fit", "qwen3.8-100k", "--strategy", "proportional"]) == cli.OK
    out = capsys.readouterr().out
    assert "OLLAMA_KV_CACHE_TYPE=q4_0" in out
    assert "assumes the Ollama server was started with the same settings" in out
    assert "16.20 GiB + 20% + KV  1.72 GiB =  21.16 GiB  fits" in out
    assert "KV: q4_0, num_ctx 100000 (Modelfile), 16 of 64 layers, 1 slot(s)" in out


def test_the_kv_cache_can_be_what_tips_a_model_over(wire, monkeypatch, qwen35_show):
    """Same model, same budget, f16 cache: 1759.5 MiB becomes ~6.1 GiB,
    and the file-size-only check would have said it fits (10 GiB + 20%
    against the ~15.0 GiB conservative budget)."""
    weights = int(10.0 * GIB)
    client = StubClient({"qwen3.8-100k:latest": weights}, show={"qwen3.8-100k:latest": qwen35_show})
    wire(client, gpus=[LAPTOP_8GB, EGPU_16GB])
    monkeypatch.setenv("OLLAMA_KV_CACHE_TYPE", "q4_0")
    assert run(["fit", "qwen3.8-100k"]) == cli.OK
    monkeypatch.setenv("OLLAMA_KV_CACHE_TYPE", "f16")
    assert run(["fit", "qwen3.8-100k"]) == cli.DOES_NOT_FIT


def test_fit_falls_back_to_file_size_and_says_so_when_show_fails(wire, capsys):
    wire(StubClient({"m:latest": SEVEN_B}, show={"m:latest": OllamaUnavailable("500 boom")}))
    assert run(["fit", "m"]) == cli.OK
    out = capsys.readouterr().out
    assert "KV not estimated (/api/show failed: 500 boom); file size + 20% only" in out
    assert "KV     ?" in out


def test_fit_falls_back_when_show_lacks_the_architecture(wire, capsys):
    wire(StubClient({"m:latest": SEVEN_B}))
    assert run(["fit", "m"]) == cli.OK
    assert "KV not estimated (/api/show returned no model_info)" in capsys.readouterr().out


def test_a_tagless_name_finds_the_latest_tag(wire, capsys):
    """/api/tags only lists `name:latest`; `fit qwen3.8-216k` used to
    report a pulled model as not pulled."""
    client = wire(StubClient({"qwen3.8-216k:latest": SEVEN_B}))
    assert run(["fit", "qwen3.8-216k"]) == cli.OK
    out = capsys.readouterr().out
    assert "not pulled" not in out
    assert client.show_calls == ["qwen3.8-216k:latest"]


def test_a_tagless_name_matches_a_resident_model_for_stop(wire, capsys):
    wire(StubClient(loaded=[LoadedModel("qwen3.8-216k:latest", 1000, 1000)]))
    assert run(["stop", "qwen3.8-216k"]) == cli.OK
    assert "Unloading qwen3.8-216k:latest" in capsys.readouterr().out


def test_bench_finds_offload_for_a_tagless_name(wire, capsys):
    wire(StubClient({"m:latest": SEVEN_B}, loaded=[LoadedModel("m:latest", 1000, 800)]))
    assert run(["bench", "m"]) == cli.OK
    assert "GPU offload: 80%" in capsys.readouterr().out


@pytest.mark.parametrize("value", ["-10", "ten", "1.5"])
def test_headroom_must_be_a_non_negative_whole_number(wire, capsys, value):
    """Rejected by argparse as a usage error (exit 2), not a traceback from
    required_bytes exiting 1 -- which reads as "does not fit"."""
    wire(StubClient({"m:latest": SEVEN_B}))
    with pytest.raises(SystemExit) as caught:
        run(["fit", "m", "--headroom", value])
    assert caught.value.code == cli.CANNOT_ANSWER
    assert "--headroom" in capsys.readouterr().err


def test_zero_headroom_is_allowed(wire):
    wire(StubClient({"m:latest": SEVEN_B}))
    assert run(["fit", "m", "--headroom", "0"]) == cli.OK


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


def picked(capsys):
    """(stdout, stderr) of a picker run. stdout is the contract: the bare
    model name and a newline, nothing else, so a script can capture it."""
    captured = capsys.readouterr()
    return captured.out, captured.err


def test_coder_model_picks_7b_for_single_8gb_card(wire, capsys):
    wire(StubClient(), gpus=ONE_CARD)
    assert run(["coder-model"]) == cli.OK
    out, err = picked(capsys)
    assert out == "qwen2.5-coder:7b\n"
    # The diagnostics are still there, just not on stdout.
    assert "GPUs present" in err and "coder model: qwen2.5-coder:7b" in err


def test_coder_model_picks_216k_for_two_16gb_cards(wire, capsys):
    egpu = Gpu(1, "RTX 5060 Ti", 16311 * MIB, 16000 * MIB)
    twin = Gpu(2, "RTX 5060 Ti (twin)", 16311 * MIB, 16000 * MIB)
    wire(StubClient(), gpus=[egpu, twin])
    assert run(["coder-model"]) == cli.OK
    assert picked(capsys)[0] == "qwen3.8-216k\n"


def test_pickers_on_the_real_8gb_plus_16gb_pair(wire, capsys):
    """The machine this was built for. Conservative caps the asymmetric
    pair at 2 x 7700 MiB (~15.0 GiB), which reaches the 100k build but not
    the 216k one; proportional sums to ~22.8 GiB and reaches both. The
    default stays conservative -- the safe side of an unknown split."""
    wire(StubClient(), gpus=[LAPTOP_8GB, EGPU_16GB])
    for command in ("coder-model", "general-model"):
        assert run([command]) == cli.OK
        assert picked(capsys)[0] == "qwen3.8-100k\n"
        assert run([command, "--strategy", "proportional"]) == cli.OK
        assert picked(capsys)[0] == "qwen3.8-216k\n"


def test_coder_model_cli_agrees_with_the_library_function(wire, capsys):
    """The CLI's `_describe_gpus`-derived budget and the library's
    `budget_bytes` must not diverge -- pin them to the same output."""
    from ollama_tools.coder_model import get_coder_model
    from ollama_tools.gpu import CONSERVATIVE

    gpus = [LAPTOP_8GB, EGPU_16GB]
    wire(StubClient(), gpus=gpus)
    assert run(["coder-model"]) == cli.OK
    assert picked(capsys)[0] == f"{get_coder_model(CONSERVATIVE, gpus)}\n"


def test_coder_model_falls_back_when_gpu_unavailable(monkeypatch, capsys):
    def raise_unavailable():
        raise cli.GpuUnavailable("nvidia-smi not found")

    monkeypatch.setattr(cli, "query_gpus", raise_unavailable)
    assert run(["coder-model"]) == cli.OK
    out, err = picked(capsys)
    assert out == "qwen2.5-coder:0.5b\n"
    assert "nvidia-smi not found" in err


def test_general_model_picks_9b_for_single_8gb_card(wire, capsys):
    wire(StubClient(), gpus=ONE_CARD)
    assert run(["general-model"]) == cli.OK
    out, err = picked(capsys)
    assert out == "qwen3.5:9b\n"
    assert "general model: qwen3.5:9b" in err


def test_general_model_picks_qwen3_216k_for_two_16gb_cards(wire, capsys):
    egpu = Gpu(1, "RTX 5060 Ti", 16311 * MIB, 16000 * MIB)
    twin = Gpu(2, "RTX 5060 Ti (twin)", 16311 * MIB, 16000 * MIB)
    wire(StubClient(), gpus=[egpu, twin])
    assert run(["general-model"]) == cli.OK
    assert picked(capsys)[0] == "qwen3.8-216k\n"


def test_general_model_falls_back_when_gpu_unavailable(monkeypatch, capsys):
    def raise_unavailable():
        raise cli.GpuUnavailable("nvidia-smi not found")

    monkeypatch.setattr(cli, "query_gpus", raise_unavailable)
    assert run(["general-model"]) == cli.OK
    assert picked(capsys)[0] == "gemma3:4b\n"


def test_general_model_cli_agrees_with_the_library_function(wire, capsys):
    """The CLI's `_describe_gpus`-derived budget and the library's
    `budget_bytes` must not diverge -- pin them to the same output."""
    from ollama_tools.general_model import get_general_model
    from ollama_tools.gpu import CONSERVATIVE

    gpus = [LAPTOP_8GB, EGPU_16GB]
    wire(StubClient(), gpus=gpus)
    assert run(["general-model"]) == cli.OK
    assert picked(capsys)[0] == f"{get_general_model(CONSERVATIVE, gpus)}\n"


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
