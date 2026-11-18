"""Integration tests for the Click CLI in src/cli.py."""

import pytest
from click.testing import CliRunner

from src.cli import cli


@pytest.fixture
def runner():
    """Provide a shared Click CliRunner instance for all CLI tests."""
    return CliRunner()


def test_cli_clean_remove_missing(runner):
    """Test clean remove-missing command."""
    result = runner.invoke(cli, ["clean", "remove-missing", "a,,None,b,nan"])
    assert result.exit_code == 0
    assert result.output.strip() == "['a', 'b']"


def test_cli_clean_remove_missing_empty(runner):
    """Test clean remove-missing with an empty input string."""
    result = runner.invoke(cli, ["clean", "remove-missing", ""])
    assert result.exit_code == 0
    assert result.output.strip() == "[]"


def test_cli_clean_fill_missing_default(runner):
    """Test clean fill-missing command with default fill value."""
    result = runner.invoke(cli, ["clean", "fill-missing", "a,,None,b"])
    assert result.exit_code == 0
    assert result.output.strip() == "['a', '0', '0', 'b']"


def test_cli_clean_fill_missing_custom(runner):
    """Test clean fill-missing command with a custom --fill-value."""
    result = runner.invoke(
        cli, ["clean", "fill-missing", "a,,nan,b", "--fill-value", "NA"]
    )
    assert result.exit_code == 0
    assert result.output.strip() == "['a', 'NA', 'NA', 'b']"


def test_cli_numeric_normalize_default(runner):
    """Test numeric normalize command with default [0, 1] range."""
    result = runner.invoke(cli, ["numeric", "normalize", "1,2,3"])
    assert result.exit_code == 0
    assert result.output.strip() == "[0.0, 0.5, 1.0]"


def test_cli_numeric_normalize_custom(runner):
    """Test numeric normalize command with custom min/max options."""
    result = runner.invoke(
        cli, ["numeric", "normalize", "1,2,3", "--min-val", "0", "--max-val", "10"]
    )
    assert result.exit_code == 0
    assert result.output.strip() == "[0.0, 5.0, 10.0]"


def test_cli_numeric_standardize(runner):
    """Test numeric standardize command."""
    result = runner.invoke(cli, ["numeric", "standardize", "2,4,4,4,5,5,7,9"])
    assert result.exit_code == 0
    assert result.output.strip() == "[-1.5, -0.5, -0.5, -0.5, 0.0, 0.0, 1.0, 2.0]"


def test_cli_numeric_clip(runner):
    """Test numeric clip command with custom bounds."""
    result = runner.invoke(
        cli, ["numeric", "clip", "-1,0.5,2", "--min-clip", "0", "--max-clip", "1"]
    )
    assert result.exit_code == 0
    assert result.output.strip() == "[0.0, 0.5, 1.0]"


def test_cli_numeric_to_integers(runner):
    """Test numeric to-integers command."""
    result = runner.invoke(cli, ["numeric", "to-integers", "1,abc,2,3.5,4"])
    assert result.exit_code == 0
    assert result.output.strip() == "[1, 2, 4]"


def test_cli_numeric_log(runner):
    """Test numeric log command filters non-positive numbers."""
    result = runner.invoke(cli, ["numeric", "log", "-2,0,1"])
    assert result.exit_code == 0
    assert result.output.strip() == "[0.0]"


def test_cli_numeric_empty_input(runner):
    """Test numeric command with empty input string."""
    result = runner.invoke(cli, ["numeric", "standardize", ""])
    assert result.exit_code == 0
    assert result.output.strip() == "[]"


def test_cli_text_tokenize(runner):
    """Test text tokenize command."""
    result = runner.invoke(cli, ["text", "tokenize", "Hello, World! 123"])
    assert result.exit_code == 0
    assert result.output.strip() == "['hello', 'world', '123']"


def test_cli_text_remove_punctuation(runner):
    """Test text remove-punctuation command."""
    result = runner.invoke(cli, ["text", "remove-punctuation", "Hello, World! 123"])
    assert result.exit_code == 0
    assert result.output.strip() == "Hello World 123"


def test_cli_text_remove_stopwords(runner):
    """Test text remove-stopwords command."""
    result = runner.invoke(
        cli, ["text", "remove-stopwords", "This is a test", "--stop-words", "is,a"]
    )
    assert result.exit_code == 0
    assert result.output.strip() == "this test"


def test_cli_struct_shuffle(runner):
    """Test struct shuffle command with deterministic seed."""
    first = runner.invoke(cli, ["struct", "shuffle", "a,b,c,d", "--seed", "42"])
    second = runner.invoke(cli, ["struct", "shuffle", "a,b,c,d", "--seed", "42"])
    assert first.exit_code == 0
    assert second.exit_code == 0
    assert first.output == second.output


def test_cli_struct_flatten(runner):
    """Test struct flatten command."""
    result = runner.invoke(cli, ["struct", "flatten", "1,2;3,4;5"])
    assert result.exit_code == 0
    assert result.output.strip() == "['1', '2', '3', '4', '5']"


def test_cli_struct_unique(runner):
    """Test struct unique command."""
    result = runner.invoke(cli, ["struct", "unique", "a,b,a,c,b"])
    assert result.exit_code == 0
    assert result.output.strip() == "['a', 'b', 'c']"
