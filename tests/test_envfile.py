import pytest

from ollama_tools.envfile import (
    coder_target_models,
    models_to_check,
    read_env_file,
    resolve,
    role_models,
)


def write(tmp_path, text):
    path = tmp_path / ".env"
    path.write_text(text, encoding="utf-8")
    return path


def test_reads_pairs_ignoring_comments_and_blanks(tmp_path):
    path = write(tmp_path, "# a comment\n\nCODER_MODEL_SOURCE=local\n  TRIAGE_OLLAMA_MODEL = gemma3:4b \n")
    values = read_env_file(path)
    assert values["CODER_MODEL_SOURCE"] == "local"
    assert values["TRIAGE_OLLAMA_MODEL"] == "gemma3:4b"


def test_strips_surrounding_quotes(tmp_path):
    values = read_env_file(write(tmp_path, 'CODER_OLLAMA_MODEL="qwen2.5-coder:7b"\n'))
    assert values["CODER_OLLAMA_MODEL"] == "qwen2.5-coder:7b"


def test_value_containing_equals_survives(tmp_path):
    values = read_env_file(write(tmp_path, "REMOTE_LLM_ENDPOINT=https://x/y?a=b\n"))
    assert values["REMOTE_LLM_ENDPOINT"] == "https://x/y?a=b"


def test_missing_file_raises(tmp_path):
    with pytest.raises(OSError):
        read_env_file(tmp_path / "nope.env")


def test_real_environment_beats_the_file(monkeypatch):
    """Mirrors load_dotenv(override=False). Without this the checker could
    bless a model the run would not load."""
    monkeypatch.setenv("CODER_MODEL_SOURCE", "cloud")
    assert resolve("CODER_MODEL_SOURCE", {"CODER_MODEL_SOURCE": "local"}) == "cloud"


def test_blank_real_variable_falls_through_to_the_file(monkeypatch):
    monkeypatch.setenv("CODER_MODEL_SOURCE", "   ")
    assert resolve("CODER_MODEL_SOURCE", {"CODER_MODEL_SOURCE": "local"}) == "local"


def test_coder_target_keeps_a_models_own_colon_tag():
    assert coder_target_models("desk:192.168.1.20:11434:qwen2.5-coder:7b") == [
        "qwen2.5-coder:7b"
    ]


def test_coder_target_lists_each_distinct_model_once():
    assert coder_target_models("a:h:1:m1,b:h:2:m2,c:h:3:m1") == ["m1", "m2"]
    assert coder_target_models("") == []
    assert coder_target_models(None) == []


def test_coder_targets_wins_over_the_role_model(monkeypatch):
    """That entry overwrites the coder's OLLAMA_MODEL on every dispatch, so
    it is what would actually load -- and is how a 27B sneaks in behind a
    config that looks like it names a 7B."""
    for name in ("CODER_MODEL_SOURCE", "ANALYSER_MODEL_SOURCE", "TRIAGE_MODEL_SOURCE"):
        monkeypatch.delenv(name, raising=False)
    values = {
        "CODER_MODEL_SOURCE": "local",
        "CODER_TARGETS": "local:localhost:11434:qwen3.8-64k:latest",
        "CODER_OLLAMA_MODEL": "qwen2.5-coder:7b",
    }
    coder = [r for r in role_models(values) if r.role == "coder"]
    assert [r.model for r in coder] == ["qwen3.8-64k:latest"]


def test_non_local_role_is_skipped(monkeypatch):
    """A cloud role cannot overcommit local VRAM."""
    monkeypatch.delenv("TRIAGE_MODEL_SOURCE", raising=False)
    values = {"TRIAGE_MODEL_SOURCE": "cloud", "TRIAGE_OLLAMA_MODEL": "qwen2.5-coder:14b"}
    triage = [r for r in role_models(values) if r.role == "triage"][0]
    assert triage.model is None
    assert triage.source == "cloud"


def test_role_defaults_to_local_when_unset(monkeypatch):
    for name in ("CODER_MODEL_SOURCE", "ANALYSER_MODEL_SOURCE", "TRIAGE_MODEL_SOURCE"):
        monkeypatch.delenv(name, raising=False)
    coder = [r for r in role_models({"CODER_OLLAMA_MODEL": "gemma3:4b"}) if r.role == "coder"][0]
    assert coder.source == "local"
    assert coder.model == "gemma3:4b"


def test_local_role_with_no_model_is_reported_not_silently_dropped(monkeypatch):
    for name in ("CODER_MODEL_SOURCE", "ANALYSER_MODEL_SOURCE", "TRIAGE_MODEL_SOURCE"):
        monkeypatch.delenv(name, raising=False)
    coder = [r for r in role_models({}) if r.role == "coder"][0]
    assert coder.model is None
    assert "cannot be sized" in coder.note


def test_models_to_check_dedupes_across_roles(monkeypatch):
    for name in ("CODER_MODEL_SOURCE", "ANALYSER_MODEL_SOURCE", "TRIAGE_MODEL_SOURCE"):
        monkeypatch.delenv(name, raising=False)
    values = {
        "CODER_OLLAMA_MODEL": "qwen2.5-coder:7b",
        "ANALYSER_OLLAMA_MODEL": "qwen2.5-coder:7b",
        "TRIAGE_OLLAMA_MODEL": "gemma3:4b",
    }
    assert models_to_check(values) == ["qwen2.5-coder:7b", "gemma3:4b"]


# --- parity with python-dotenv (issue #11) ---------------------------------
# The runtime loads this file with python-dotenv, so anywhere the two
# disagree about a value this module is predicting the wrong thing. Only a
# differential test can hold that line; a hand-written expectation just
# encodes whatever this parser happens to do today.

PARITY_CASES = [
    "PLAIN=value",
    'DQUOTED="value"',
    "SQUOTED='value'",
    'DQUOTED_COMMENT="a" # trailing',
    "SQUOTED_COMMENT='a' # trailing",
    "UNQUOTED_COMMENT=plain # trailing",
    "HASH_NO_SPACE=has#hash",
    'PADDED="  keeps padding  "',
    "TRAILING_WS=value    ",
    "APOSTROPHE=it's",
    'NESTED_QUOTES=\'say "hi"\'',
    "EMPTY=",
    "EXPORTED=exported",
    "TARGET=local:localhost:11434:qwen2.5-coder:7b",
    "EXPAND=${PARITY_SET}",
    "EXPAND_SQUOTED='${PARITY_SET}'",
    "EXPAND_MISSING=${PARITY_UNSET}",
    "EXPAND_DEFAULT=${PARITY_UNSET:-fallback}",
    "EXPAND_INLINE=pre-${PARITY_SET}-post",
]


def test_parsing_matches_python_dotenv(tmp_path, monkeypatch):
    dotenv = pytest.importorskip(
        "dotenv", reason="python-dotenv is a test-only dependency (pip install -e .[dev])"
    )
    monkeypatch.setenv("PARITY_SET", "from-env")
    monkeypatch.delenv("PARITY_UNSET", raising=False)

    path = tmp_path / ".env"
    path.write_text("\n".join(PARITY_CASES) + "\n", encoding="utf-8")

    assert read_env_file(path) == dict(dotenv.dotenv_values(path))


def test_inline_comment_is_not_part_of_the_value(tmp_path):
    """The bug issue #11 was actually about: the value kept ' # comment'."""
    values = read_env_file(write(tmp_path, 'CODER_OLLAMA_MODEL="qwen2.5-coder:7b" # the safe one\n'))
    assert values["CODER_OLLAMA_MODEL"] == "qwen2.5-coder:7b"


def test_expansion_is_applied_so_a_placeholder_is_never_sized(tmp_path, monkeypatch):
    """Issue #11 missed this one, and .env already uses ${...}. An
    unexpanded '${HOST}' would be carried into a model name."""
    monkeypatch.setenv("PARITY_MODEL", "qwen2.5-coder:7b")
    values = read_env_file(write(tmp_path, "CODER_OLLAMA_MODEL=${PARITY_MODEL}\n"))
    assert values["CODER_OLLAMA_MODEL"] == "qwen2.5-coder:7b"


def test_nested_quotes_were_never_broken(tmp_path):
    """Issue #11's headline example claimed this produced 'say "hi'. It did
    not, before or after the fix - kept as a regression guard."""
    assert read_env_file(write(tmp_path, "A='say \"hi\"'\n"))["A"] == 'say "hi"'
