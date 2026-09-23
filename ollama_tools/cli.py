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
import subprocess
import sys

from . import bench as bench_mod
from .client import OllamaClient, OllamaUnavailable
from .coder_model import CODER_FALLBACK, coder_model_for_budget
from .envfile import models_to_check, read_env_file, role_models
from .fit import DEFAULT_HEADROOM_PERCENT, judge
from .kvcache import DEFAULT_KV_CACHE_TYPE, KV_CACHE_TYPES, KvCacheUnknown, kv_shape
from .general_model import GENERAL_FALLBACK, general_model_for_budget
from .gpu import CONSERVATIVE, GIB, GpuUnavailable, STRATEGIES, budget_bytes, query_gpus

OK, DOES_NOT_FIT, CANNOT_ANSWER = 0, 1, 2


def _gib(value: float) -> str:
    return f"{value / GIB:.2f} GB"


def _refuse(message: str, code: int) -> int:
    print(f"\nREFUSED: {message}", file=sys.stderr)
    return code


def _describe_gpus(gpus, strategy: str) -> int:
    print("GPUs present")
    for gpu in gpus:
        print(f"  [{gpu.index}] {gpu.name} - {gpu.free_gib:.2f} GB free of {gpu.total_gib:.2f} GB")

    total = sum(g.free_bytes for g in gpus)
    even = min(g.free_bytes for g in gpus) * len(gpus)
    budget = budget_bytes(gpus, strategy)

    print(f"\n  free, summed             : {_gib(total)}")
    if len(gpus) > 1:
        print(f"  free, even-split ceiling : {_gib(even)}  ({len(gpus)} x smallest card)")
        if even < total:
            print(
                "\n  These cards are asymmetric. An even split wastes the larger one -\n"
                "  see docs/lmstudio-multi-gpu.md for moving off 'Split evenly'."
            )
    print(f"  budget ({strategy}): {_gib(budget)}")

    if len(gpus) == 1:
        print(
            "\n  Only one GPU is present. If the eGPU should be attached:\n"
            "    - Hot-plugging will not fix it. Cold boot with the enclosure attached.\n"
            "    - On Windows 11 that means Restart, not Shut down: with Fast Startup,\n"
            "      'Shut down' is a hybrid hibernate and does not force a full POST.\n"
            "    - If attached but missing, run diagnostics/Test-DriverConflict.ps1;\n"
            "      two NVIDIA driver versions leave one card on Code 31."
        )
    return budget


def _sizes(client: OllamaClient) -> dict[str, int]:
    return {m.name: m.size_bytes for m in client.list_models()}


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
        return models_to_check(values), True
    if getattr(args, "model", None):
        return list(args.model), True
    return sorted(sizes, key=lambda n: sizes[n], reverse=True), False


def _kv_request(args):
    """(parallel, kv_cache_type, num_ctx), or parallel None for the
    headroom-only check. Naming a KV type or context without a slot count
    is still a question about the KV cache, so it means one slot."""
    num_ctx = getattr(args, "num_ctx", None)
    kv_type = args.kv_cache_type
    parallel = args.parallel
    if parallel is None and (num_ctx or kv_type):
        parallel = 1
    return parallel, kv_type or DEFAULT_KV_CACHE_TYPE, num_ctx


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

    parallel, kv_type, num_ctx = _kv_request(args)
    if parallel is not None:
        print(f"\nBudgeting {parallel} parallel slot(s), KV cache {kv_type}")

    print("\nModels")
    verdicts = []
    # Collected rather than returned on: a multi-role config with one model
    # missing should say so once, with every other role's verdict alongside,
    # instead of making you rerun to discover them one at a time.
    #
    # Only reachable for an explicit list or --env-file. A survey builds its
    # list from `sizes` itself, so every name is pulled by construction.
    unpulled: list[str] = []
    # Same collect-don't-return reasoning, for models whose KV cache cannot
    # be computed (hybrid architectures, /api/show failing): (name, why).
    unsizable: list[tuple[str, str]] = []
    for name in models:
        if name not in sizes:
            unpulled.append(name)
            print(f"  {name:<40} not pulled - size unknown")
            continue
        kv_line = ""
        kv_slot = 0
        if parallel is not None:
            try:
                shape = kv_shape(client.show(name), num_ctx)
            except OllamaUnavailable as exc:
                unsizable.append((name, f"/api/show failed: {exc}"))
                print(f"  {name:<40} KV cache could not be read - see below")
                continue
            except KvCacheUnknown as exc:
                unsizable.append((name, str(exc)))
                print(f"  {name:<40} KV cache cannot be computed - see below")
                continue
            kv_slot = shape.slot_bytes(kv_type)
            kv_line = (
                f"\n  {'':<40} + {parallel} x {_gib(kv_slot)} KV "
                f"({shape.context:,} ctx per slot)"
            )
        verdict = judge(name, sizes[name], budget, args.headroom, parallel, kv_slot)
        verdicts.append(verdict)
        state = "fits" if verdict.fits else f"TOO BIG by {verdict.short_gib:.2f} GB"
        print(
            f"  {name:<40} {verdict.size_gib:6.2f} GB + {args.headroom}%"
            f"{kv_line} = {verdict.needed_gib:6.2f} GB  {state}"
        )

    if parallel is not None:
        # The tool cannot see the server's environment (docs/ollama-multi-gpu.md
        # section 3), so it can only answer the question it was asked.
        print(
            f"\nThis budgets {parallel} slot(s) with a {kv_type} KV cache. Ollama allocates\n"
            "whatever the *server* was started with - OLLAMA_NUM_PARALLEL, and\n"
            "OLLAMA_KV_CACHE_TYPE (only with OLLAMA_FLASH_ATTENTION=1). If those\n"
            "differ, this answered a different question from the one the load will ask."
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
    if unsizable:
        listed = "".join(f"\n  {name}: {why}" for name, why in unsizable)
        return _refuse(
            f"{len(unsizable)} model(s) have a KV cache that could not be sized:{listed}",
            CANNOT_ANSWER,
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

    print(f"{'model':<40}{'total':>10}{'in VRAM':>10}{'offload':>9}")
    for model in loaded:
        print(
            f"{model.name:<40}{model.size_bytes / GIB:9.2f}G"
            f"{model.size_vram_bytes / GIB:9.2f}G{model.offload_fraction * 100:8.0f}%"
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
        loaded = {m.name for m in client.loaded_models()}
    except OllamaUnavailable as exc:
        return _refuse(str(exc), CANNOT_ANSWER)

    if not loaded:
        print("Nothing resident - all VRAM is already free.")
        return OK

    targets = sorted(loaded) if args.all else list(args.model or [])
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


def _coder_model(args) -> int:
    try:
        gpus = query_gpus()
    except GpuUnavailable as exc:
        print(f"{exc}. Falling back to the smallest coder model.")
        print(CODER_FALLBACK)
        return OK

    budget = _describe_gpus(gpus, args.strategy)
    model = coder_model_for_budget(budget)
    print(f"\ncoder model: {model}")
    return OK


def _general_model(args) -> int:
    try:
        gpus = query_gpus()
    except GpuUnavailable as exc:
        print(f"{exc}. Falling back to the smallest general model.")
        print(GENERAL_FALLBACK)
        return OK

    budget = _describe_gpus(gpus, args.strategy)
    model = general_model_for_budget(budget)
    print(f"\ngeneral model: {model}")
    return OK


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
        live: list | None = [m for m in client.loaded_models() if m.name == model]
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
            f"({model_state.size_vram_bytes / GIB:.2f} GB of {model_state.size_bytes / GIB:.2f} GB in VRAM)"
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


def _positive_int(text: str) -> int:
    value = int(text)
    if value < 1:
        raise argparse.ArgumentTypeError("must be at least 1")
    return value


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
        sub.add_argument("--headroom", type=int, default=DEFAULT_HEADROOM_PERCENT)
        sub.add_argument(
            "--parallel",
            type=_positive_int,
            help=(
                "budget this many parallel slots (OLLAMA_NUM_PARALLEL), each with its own "
                "full KV cache computed from the model's shape; default is the "
                "headroom-only check"
            ),
        )
        sub.add_argument(
            "--kv-cache-type",
            choices=sorted(KV_CACHE_TYPES),
            help=(
                f"the server's OLLAMA_KV_CACHE_TYPE (default {DEFAULT_KV_CACHE_TYPE}, the "
                "largest; this tool cannot read the server's environment)"
            ),
        )
        if with_model == "optional":
            # fit only: start and bench load at the manifest's num_ctx, so an
            # override there would pass a check the real load then fails.
            sub.add_argument(
                "--num-ctx",
                type=_positive_int,
                help="context per slot instead of the manifest's num_ctx, e.g. before rebuilding a tag",
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
        help="pick the coder model that fits the VRAM attached right now (always exits 0)",
    )
    coder_model.add_argument("--strategy", choices=STRATEGIES, default=CONSERVATIVE)
    add_common(coder_model)
    coder_model.set_defaults(func=_coder_model)

    general_model = subparsers.add_parser(
        "general-model",
        help="pick the general-purpose model that fits the VRAM attached right now (always exits 0)",
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
