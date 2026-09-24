"""Command line for the Ollama/VRAM tooling.

Exit codes are the contract, so this can gate a script:

    0 - fine
    1 - does not fit; nothing was loaded
    2 - the question could not be answered (no nvidia-smi, no GPU, server
        unreachable, model not pulled); also nothing was loaded

There is deliberately no --force. If you believe a model fits because the
runtime places layers proportionally, ``--strategy proportional`` says so
in terms the check can act on. "Load it anyway" is not a claim, it is the
hang this whole package exists to prevent.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys

from . import bench as bench_mod
from .client import OllamaClient, OllamaUnavailable, normalise_model_name
from .coder_model import CODER_FALLBACK, coder_model_for_budget
from .envfile import models_to_check, read_env_file, role_models
from .fit import (
    DEFAULT_HEADROOM_PERCENT,
    KV_CACHE_TYPES,
    KvUnknown,
    estimate_kv_cache,
    judge,
    kv_cache_type,
    num_parallel,
)
from .general_model import GENERAL_FALLBACK, general_model_for_budget
from .gpu import CONSERVATIVE, GIB, GpuUnavailable, STRATEGIES, budget_bytes, query_gpus

OK, DOES_NOT_FIT, CANNOT_ANSWER = 0, 1, 2


def _gib(value: float) -> str:
    return f"{value / GIB:.2f} GiB"


def _non_negative_int(text: str) -> int:
    """argparse type for --headroom. Rejected here, as a usage error (exit
    2), rather than reaching required_bytes' ValueError and surfacing as a
    traceback with exit 1 -- which is the "does not fit" code."""
    try:
        value = int(text)
    except ValueError:
        raise argparse.ArgumentTypeError(f"{text!r} is not a whole number") from None
    if value < 0:
        raise argparse.ArgumentTypeError(f"{value} is negative; headroom must be 0 or more")
    return value


def _positive_int(text: str) -> int:
    """argparse type for --parallel and --num-ctx: a usage error (exit 2)
    for 0, negatives and non-numbers, not a silent fallback to 1."""
    try:
        value = int(text)
    except ValueError:
        raise argparse.ArgumentTypeError(f"{text!r} is not a whole number") from None
    if value < 1:
        raise argparse.ArgumentTypeError(f"{value} must be at least 1")
    return value


def _kv_env(args) -> tuple[dict[str, str], str]:
    """The environment the KV estimate reads, with any flags laid over this
    shell's, and a note saying where the settings came from.

    The flags exist because this shell's environment is only a stand-in for
    the server's (docs/ollama-multi-gpu.md section 3): passing what the
    server was actually started with beats hoping the two match."""
    env = dict(os.environ)
    overridden = []
    if getattr(args, "parallel", None) is not None:
        env["OLLAMA_NUM_PARALLEL"] = str(args.parallel)
        overridden.append("--parallel")
    if getattr(args, "kv_cache_type", None) is not None:
        env["OLLAMA_KV_CACHE_TYPE"] = args.kv_cache_type
        # Naming a quantised type is a statement that the server uses it,
        # which it can only do with flash attention on.
        env["OLLAMA_FLASH_ATTENTION"] = "1"
        overridden.append("--kv-cache-type")
    if overridden:
        source = f"{' and '.join(overridden)} over this shell's environment"
    else:
        source = "read from this shell -- assumes the Ollama server was started with the same settings"
    return env, source


def _refuse(message: str, code: int) -> int:
    print(f"\nREFUSED: {message}", file=sys.stderr)
    return code


def _describe_gpus(gpus, strategy: str, out=None) -> int:
    """Print the inventory and budget to ``out`` (stdout by default; the
    pickers send it to stderr so their stdout is just the model name)."""
    out = out or sys.stdout

    def say(text: str = "") -> None:
        print(text, file=out)

    say("GPUs present")
    for gpu in gpus:
        say(f"  [{gpu.index}] {gpu.name} - {gpu.free_gib:.2f} GiB free of {gpu.total_gib:.2f} GiB")

    total = sum(g.free_bytes for g in gpus)
    even = min(g.free_bytes for g in gpus) * len(gpus)
    budget = budget_bytes(gpus, strategy)

    say(f"\n  free, summed             : {_gib(total)}")
    if len(gpus) > 1:
        say(f"  free, even-split ceiling : {_gib(even)}  ({len(gpus)} x smallest card)")
        if even < total:
            say(
                "\n  These cards are asymmetric. An even split wastes the larger one -\n"
                "  see docs/lmstudio-multi-gpu.md for moving off 'Split evenly'."
            )
    say(f"  budget ({strategy}): {_gib(budget)}")

    if len(gpus) == 1:
        say(
            "\n  Only one GPU is present. If the eGPU should be attached:\n"
            "    - Hot-plugging will not fix it. Cold boot with the enclosure attached.\n"
            "    - On Windows 11 that means Restart, not Shut down: with Fast Startup,\n"
            "      'Shut down' is a hybrid hibernate and does not force a full POST.\n"
            "    - If attached but missing, run diagnostics/Test-DriverConflict.ps1;\n"
            "      two NVIDIA driver versions leave one card on Code 31."
        )
    return budget


def _sizes(client: OllamaClient) -> dict[str, int]:
    return {normalise_model_name(m.name): m.size_bytes for m in client.list_models()}


def _resolve_requested(args, client: OllamaClient, sizes: dict[str, int]):
    """(models, explicitly_requested). An explicit list or an env file are
    both statements of intent; a bare survey is a question."""
    if getattr(args, "env_file", None):
        try:
            values = read_env_file(args.env_file)
        except OSError as exc:
            raise OllamaUnavailable(f"env file could not be read: {exc}") from exc
        print(f"Roles configured by {args.env_file}")
        for entry in role_models(values):
            if entry.model:
                print(f"  {entry.role:<9} -> {entry.model}")
            else:
                print(f"  {entry.role:<9} -> {entry.source} ({entry.note})")
        print()
        return [normalise_model_name(m) for m in models_to_check(values)], True
    if getattr(args, "model", None):
        return [normalise_model_name(m) for m in args.model], True
    return sorted(sizes, key=lambda n: sizes[n], reverse=True), False


def _check(args) -> int:
    try:
        gpus = query_gpus()
    except GpuUnavailable as exc:
        return _refuse(f"{exc}. Not loading anything onto an unknown GPU setup.", CANNOT_ANSWER)

    budget = _describe_gpus(gpus, args.strategy)
    client = OllamaClient(args.endpoint)
    try:
        sizes = _sizes(client)
        models, requested = _resolve_requested(args, client, sizes)
    except OllamaUnavailable as exc:
        return _refuse(f"{exc}", CANNOT_ANSWER)

    if not models:
        print("Nothing to check.")
        return OK

    env, env_source = _kv_env(args)
    num_ctx = getattr(args, "num_ctx", None)
    print(
        f"\nModels  (KV cache at OLLAMA_KV_CACHE_TYPE={kv_cache_type(env)} x "
        f"OLLAMA_NUM_PARALLEL={num_parallel(env)},\n"
        f"         {env_source})"
    )
    verdicts = []
    # Collected rather than returned on: a multi-role config with one model
    # missing should say so once, with every other role's verdict alongside,
    # instead of making you rerun to discover them one at a time.
    #
    # Only reachable for an explicit list or --env-file. A survey builds its
    # list from `sizes` itself, so every name is pulled by construction.
    unpulled: list[str] = []
    for name in models:
        if name not in sizes:
            unpulled.append(name)
            print(f"  {name:<40} not pulled - size unknown")
            continue
        kv, why_not = _kv_estimate(client, name, env, num_ctx)
        verdict = judge(name, sizes[name], budget, args.headroom, kv.bytes if kv else 0)
        verdicts.append(verdict)
        state = "fits" if verdict.fits else f"TOO BIG by {verdict.short_gib:.2f} GiB"
        kv_part = f"KV {verdict.kv_gib:5.2f} GiB" if kv else "KV     ?    "
        print(
            f"  {name:<40} {verdict.size_gib:6.2f} GiB + {args.headroom}% + {kv_part} "
            f"= {verdict.needed_gib:6.2f} GiB  {state}"
        )
        if kv:
            print(
                f"  {'':<40} KV: {kv.cache_type}, num_ctx {kv.num_ctx} ({kv.ctx_source}), "
                f"{kv.kv_layers} of {kv.block_count} layers, {kv.parallel} slot(s)"
            )
        else:
            print(
                f"  {'':<40} KV not estimated ({why_not}); "
                f"file size + {args.headroom}% only"
            )

    too_big = [v for v in verdicts if not v.fits]
    # "Will not fit" outranks "could not be sized" when both are true: both
    # refuse and nothing loads either way, but a definite hazard is the more
    # actionable thing to put in front of someone.
    if requested and too_big:
        names = ", ".join(v.model for v in too_big)
        return _refuse(
            f"{names} will not fit in {_gib(budget)}. "
            "Do not load it - this is the configuration that hangs the machine.",
            DOES_NOT_FIT,
        )
    if unpulled:
        listed = "".join(f"\n  ollama pull {name}" for name in unpulled)
        return _refuse(
            f"{len(unpulled)} model(s) are not pulled, so their size is unknown:{listed}",
            CANNOT_ANSWER,
        )
    if too_big:
        print(f"\n{len(too_big)} of {len(verdicts)} pulled models do not fit right now.")
    print("\nOK - nothing requested exceeds the VRAM present.")
    return OK


def _kv_estimate(client: OllamaClient, name: str, env, num_ctx: int | None = None):
    """(estimate, None) or (None, why). Never raises: a model whose
    architecture cannot be read is still judged, on file size plus headroom
    as before, and the output says which of the two it got."""
    try:
        return estimate_kv_cache(client.show(name), env, num_ctx), None
    except OllamaUnavailable as exc:
        return None, f"/api/show failed: {exc}"
    except KvUnknown as exc:
        return None, str(exc)


def _start(args) -> int:
    code = _check(args)
    if code != OK:
        print(f"\nNOT LOADING {args.model[0]} - the fit check returned {code}.", file=sys.stderr)
        return code

    model = args.model[0]
    client = OllamaClient(args.endpoint)
    try:
        capabilities = client.capabilities(model)
    except OllamaUnavailable as exc:
        return _refuse(str(exc), CANNOT_ANSWER)

    # An embedding model has no interactive session to open; warming it with
    # one embed call is the equivalent of loading it.
    if bench_mod.is_embedding_model(capabilities):
        print(f"\n{model} is an embedding model - warming it with one /api/embed call.")
        try:
            run = bench_mod.run_embedding(client, model, bench_mod.DEFAULT_EMBED_TEXT)
        except OllamaUnavailable as exc:
            return _refuse(str(exc), CANNOT_ANSWER)
        print(f"Loaded. {run.dimensions} dimensions, {run.total_seconds:.2f}s.")
        return OK

    print(f"\nFit check passed; starting {model}")
    argv = ["ollama", "run", model]
    if args.prompt:
        argv.append(args.prompt)
    try:
        return subprocess.run(argv).returncode
    except OSError as exc:
        return _refuse(f"could not run ollama: {exc}", CANNOT_ANSWER)


def _ps(args) -> int:
    client = OllamaClient(args.endpoint)
    try:
        loaded = client.loaded_models()
    except OllamaUnavailable as exc:
        return _refuse(str(exc), CANNOT_ANSWER)

    if not loaded:
        print("Nothing resident - all VRAM is free for the next load.")
        return OK

    print(f"{'model':<40}{'total GiB':>10}{'VRAM GiB':>10}{'offload':>9}")
    for model in loaded:
        print(
            f"{model.name:<40}{model.size_bytes / GIB:10.2f}"
            f"{model.size_vram_bytes / GIB:10.2f}{model.offload_fraction * 100:8.0f}%"
        )
    held = sum(m.size_vram_bytes for m in loaded)
    print(f"\n{_gib(held)} held by {len(loaded)} resident model(s).")
    if any(m.offload_fraction < 0.99 for m in loaded):
        print(
            "\nAt least one model is partly on CPU. It will run, slowly - that is\n"
            "what caps the token rate. Free VRAM or drop to a smaller model."
        )
    return OK


def _stop(args) -> int:
    client = OllamaClient(args.endpoint)
    try:
        loaded = {normalise_model_name(m.name) for m in client.loaded_models()}
    except OllamaUnavailable as exc:
        return _refuse(str(exc), CANNOT_ANSWER)

    if not loaded:
        print("Nothing resident - all VRAM is already free.")
        return OK

    targets = sorted(loaded) if args.all else [normalise_model_name(m) for m in args.model or []]
    for name in targets:
        if name not in loaded:
            print(f"{name} is not resident; nothing to unload.")
            continue
        print(f"Unloading {name} ...")
        try:
            client.unload(name)
        except OllamaUnavailable as exc:
            return _refuse(f"failed to unload {name}: {exc}", CANNOT_ANSWER)

    try:
        still = client.loaded_models()
        held = sum(m.size_vram_bytes for m in still)
        print(f"\n{_gib(held)} still held by {len(still)} resident model(s).")
    except OllamaUnavailable as exc:
        # The unload succeeded; failing to re-read state is not a failure.
        # The cause is carried through because "could not re-read" reads the
        # same whether the server died, refused the connection or returned a
        # 500, and those want different responses from whoever sees it.
        print(f"\nUnloaded (could not re-read resident state: {exc}).")
    return OK


def _pick(args, kind: str, fallback: str, for_budget) -> int:
    """Shared body of the pickers. stdout carries the model name and
    nothing else, so ``MODEL=$(ollama-tools coder-model)`` works; the GPU
    inventory and any fallback reason go to stderr."""
    try:
        gpus = query_gpus()
    except GpuUnavailable as exc:
        print(f"{exc}. Falling back to the smallest {kind} model.", file=sys.stderr)
        print(fallback)
        return OK

    budget = _describe_gpus(gpus, args.strategy, out=sys.stderr)
    model = for_budget(budget)
    print(f"\n{kind} model: {model}", file=sys.stderr)
    print(model)
    return OK


def _coder_model(args) -> int:
    return _pick(args, "coder", CODER_FALLBACK, coder_model_for_budget)


def _general_model(args) -> int:
    return _pick(args, "general", GENERAL_FALLBACK, general_model_for_budget)


def _bench(args) -> int:
    code = _check(args)
    if code != OK:
        print(f"\nNOT BENCHMARKING {args.model[0]} - the fit check returned {code}.", file=sys.stderr)
        return code

    model = args.model[0]
    client = OllamaClient(args.endpoint)
    try:
        capabilities = client.capabilities(model)
        embedding = bench_mod.is_embedding_model(capabilities)
        for attempt in range(1, args.repeat + 1):
            print(f"\nRun {attempt} of {args.repeat}")
            if embedding:
                run = bench_mod.run_embedding(client, model, bench_mod.DEFAULT_EMBED_TEXT)
                print(f"  embedding: {run.dimensions} dims in {run.total_seconds:.3f}s")
            else:
                run = bench_mod.run_generation(client, model, args.prompt or bench_mod.DEFAULT_PROMPT, args.tokens)
                print(
                    f"  generation {run.generation_tokens_per_second:.1f} tok/s, "
                    f"prompt {run.prompt_tokens_per_second:.1f} tok/s, "
                    f"{run.total_seconds:.1f}s total"
                )
    except OllamaUnavailable as exc:
        return _refuse(str(exc), CANNOT_ANSWER)

    # Deliberately outside the block above, for the reason _stop gives: the
    # benchmark has already run and printed its timings, so a server that
    # goes away before this follow-up read has cost us the offload figure,
    # not the results. Returning CANNOT_ANSWER here would report a run that
    # succeeded as a run that failed.
    # None means the read failed; a list means it succeeded, and an empty one
    # means the model genuinely is not resident. Collapsing those two into []
    # made the missing offload line mean two different things, with the
    # difference visible only as the presence of a warning further up.
    try:
        wanted = normalise_model_name(model)
        live: list | None = [
            m for m in client.loaded_models() if normalise_model_name(m.name) == wanted
        ]
    except OllamaUnavailable as exc:
        live = None
        print(
            "\nCould not re-read GPU offload afterwards; the timings above "
            f"stand. ({exc})"
        )

    if live:
        model_state = live[0]
        pct = model_state.offload_fraction * 100
        print(
            f"\nGPU offload: {pct:.0f}% "
            f"({model_state.size_vram_bytes / GIB:.2f} GiB of {model_state.size_bytes / GIB:.2f} GiB in VRAM)"
        )
        if pct < 99:
            print("Part of this model is on CPU, which is what caps the rate above.")
    elif live is not None:
        # Read fine, model gone. Unusual right after a benchmark, but it
        # happens with OLLAMA_KEEP_ALIVE=0 or when a concurrent load evicts
        # it -- and saying so beats omitting the line and leaving the reader
        # to wonder which of the two happened.
        print(
            f"\n{model} is no longer resident, so GPU offload could not be "
            "read. A keep-alive of 0, or another model loaded since, will do "
            "this; the timings above stand."
        )
    if not embedding:
        print(
            "\nGeneration is memory-bandwidth-bound and prompt eval is compute-bound,\n"
            "so compare like with like when putting these in the README."
        )
    return OK


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="ollama-tools",
        description=(
            "Check a model fits in attached VRAM before anything loads it. "
            "Windows and Linux; requires an NVIDIA GPU with nvidia-smi on "
            "PATH -- other GPUs exit 2 rather than guess at a VRAM figure."
        ),
    )
    parser.add_argument("--endpoint", default="http://localhost:11434")
    subparsers = parser.add_subparsers(dest="command", required=True)

    def add_common(sub):
        # Also on every subcommand, because `ollama-tools stop --all
        # --endpoint X` is what people type and argparse would otherwise
        # reject it as an unrecognised argument. SUPPRESS so an absent flag
        # here leaves the top-level value alone instead of a subparser
        # default silently clobbering it.
        sub.add_argument("--endpoint", default=argparse.SUPPRESS)

    def add_fit_options(sub, with_model="optional"):
        if with_model == "optional":
            sub.add_argument("model", nargs="*", help="models to check; default is everything pulled")
            sub.add_argument("--env-file", help="check what an issue-worm run would load instead")
        else:
            sub.add_argument("model", nargs=1)
        sub.add_argument("--strategy", choices=STRATEGIES, default=CONSERVATIVE)
        sub.add_argument(
            "--headroom",
            type=_non_negative_int,
            default=DEFAULT_HEADROOM_PERCENT,
            help=(
                "percent added to the weights for what is not computed (compute "
                f"graph, CUDA context); default {DEFAULT_HEADROOM_PERCENT}. The KV "
                "cache is estimated separately from /api/show, using this shell's "
                "OLLAMA_KV_CACHE_TYPE, OLLAMA_NUM_PARALLEL and OLLAMA_CONTEXT_LENGTH "
                "-- so it assumes the server runs with the same settings, unless "
                "--parallel / --kv-cache-type say otherwise"
            ),
        )
        sub.add_argument(
            "--parallel",
            type=_positive_int,
            help=(
                "parallel slots to budget a full KV cache for, instead of this "
                "shell's OLLAMA_NUM_PARALLEL; pass what the server runs with"
            ),
        )
        sub.add_argument(
            "--kv-cache-type",
            choices=sorted(KV_CACHE_TYPES),
            help=(
                "the KV cache type the server actually uses, instead of this shell's "
                "OLLAMA_KV_CACHE_TYPE. Ollama only uses a quantised cache with flash "
                "attention on, so pass f16 if the server has it off"
            ),
        )
        if with_model == "optional":
            # fit only: start and bench load at the Modelfile's num_ctx, so an
            # override there would pass a check the real load then fails.
            sub.add_argument(
                "--num-ctx",
                type=_positive_int,
                help=(
                    "context per slot instead of the Modelfile's num_ctx, e.g. to "
                    "size a tag before rebuilding it (capped at the trained length)"
                ),
            )

    check = subparsers.add_parser("fit", help="does it fit? (checks only, loads nothing)")
    add_fit_options(check)
    add_common(check)
    check.set_defaults(func=_check)

    start = subparsers.add_parser("start", help="load a model, only if it fits")
    add_fit_options(start, with_model="one")
    start.add_argument("--prompt", help="run one prompt instead of an interactive session")
    add_common(start)
    start.set_defaults(func=_start)

    ps = subparsers.add_parser("ps", help="what is resident, and how much reached the GPU")
    add_common(ps)
    ps.set_defaults(func=_ps)

    coder_model = subparsers.add_parser(
        "coder-model",
        help=(
            "pick the coder model that fits the VRAM attached right now; prints "
            "only the name on stdout, the GPU inventory on stderr (always exits 0)"
        ),
    )
    coder_model.add_argument("--strategy", choices=STRATEGIES, default=CONSERVATIVE)
    add_common(coder_model)
    coder_model.set_defaults(func=_coder_model)

    general_model = subparsers.add_parser(
        "general-model",
        help=(
            "pick the general-purpose model that fits the VRAM attached right now; "
            "prints only the name on stdout, the GPU inventory on stderr (always exits 0)"
        ),
    )
    general_model.add_argument("--strategy", choices=STRATEGIES, default=CONSERVATIVE)
    add_common(general_model)
    general_model.set_defaults(func=_general_model)

    stop = subparsers.add_parser("stop", help="unload to reclaim VRAM; the server stays up")
    stop.add_argument("model", nargs="*")
    stop.add_argument("--all", action="store_true")
    add_common(stop)
    stop.set_defaults(func=_stop)

    benchmark = subparsers.add_parser("bench", help="tok/s and GPU offload")
    add_fit_options(benchmark, with_model="one")
    benchmark.add_argument("--prompt")
    benchmark.add_argument("--tokens", type=int, default=200)
    benchmark.add_argument("--repeat", type=int, default=1)
    add_common(benchmark)
    benchmark.set_defaults(func=_bench)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "stop" and not args.all and not args.model:
        print("stop needs a model name or --all", file=sys.stderr)
        return CANNOT_ANSWER
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
