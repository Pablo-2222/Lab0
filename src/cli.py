"""Command Line Interface for data preprocessing functions."""

import click

from src.preprocessing import (
    clip_values,
    convert_to_integers,
    fill_missing_values,
    flatten_list,
    logarithmic_scale,
    normalize_min_max,
    remove_duplicated_values,
    remove_missing_values,
    remove_stop_words,
    select_alphanumerical_and_spaces,
    shuffle_list,
    standardize_z_score,
    tokenize_text,
)


def _parse_string_list(raw_values):
    """Parse a comma-separated string into a list, converting None/nan tokens."""
    if not raw_values:
        return []
    parsed = []
    for item in raw_values.split(","):
        cleaned = item.strip()
        if cleaned == "None":
            parsed.append(None)
        elif cleaned.lower() == "nan":
            parsed.append(float("nan"))
        else:
            parsed.append(cleaned)
    return parsed


def _parse_float_list(raw_values):
    """Parse a comma-separated string into a list of floats."""
    if not raw_values.strip():
        return []
    return [float(item.strip()) for item in raw_values.split(",") if item.strip()]


@click.group()
def cli():
    """CLI tool for data preprocessing operations."""


@cli.group()
def clean():
    """Functions related to data cleaning."""


@cli.group()
def numeric():
    """Functions related to dealing with numerical attributes."""


@cli.group()
def text():
    """Functions to deal with textual information."""


@cli.group()
def struct():
    """Functions related to the structure of data."""


@clean.command("remove-missing")
@click.argument("values")
def remove_missing(values):
    """Remove missing values (None, empty strings, nan) from a list.

    Example: python -m src.cli clean remove-missing "a,,None,b,nan"
    """
    result = remove_missing_values(_parse_string_list(values))
    click.echo(result)


@clean.command("fill-missing")
@click.argument("values")
@click.option(
    "--fill-value",
    type=str,
    default="0",
    show_default=True,
    help="Value used to replace missing values in the input list.",
)
def fill_missing(values, fill_value):
    """Replace missing values in a list with a given fill value.

    Example: python -m src.cli clean fill-missing "a,,None,b" --fill-value 0
    """
    result = fill_missing_values(_parse_string_list(values), fill_value=fill_value)
    click.echo(result)


@numeric.command("normalize", context_settings={"ignore_unknown_options": True})
@click.argument("values")
@click.option(
    "--min-val",
    type=float,
    default=0.0,
    show_default=True,
    help="Minimum value of the normalized range.",
)
@click.option(
    "--max-val",
    type=float,
    default=1.0,
    show_default=True,
    help="Maximum value of the normalized range.",
)
def normalize(values, min_val, max_val):
    """Normalize numerical values using the min-max method.

    Example: python -m src.cli numeric normalize "1,2,3,4" --min-val 0 --max-val 1
    """
    result = normalize_min_max(
        _parse_float_list(values), new_min=min_val, new_max=max_val
    )
    click.echo(result)


@numeric.command("standardize", context_settings={"ignore_unknown_options": True})
@click.argument("values")
def standardize(values):
    """Standardize numerical values using the z-score method.

    Example: python -m src.cli numeric standardize "1,2,3,4,5"
    """
    result = standardize_z_score(_parse_float_list(values))
    click.echo(result)


@numeric.command("clip", context_settings={"ignore_unknown_options": True})
@click.argument("values")
@click.option(
    "--min-clip",
    type=float,
    default=0.0,
    show_default=True,
    help="Minimum value to clip.",
)
@click.option(
    "--max-clip",
    type=float,
    default=1.0,
    show_default=True,
    help="Maximum value to clip.",
)
def clip(values, min_clip, max_clip):
    """Clip numerical values to a specified [min, max] range.

    Example: python -m src.cli numeric clip "-1,0.5,2" --min-clip 0 --max-clip 1
    """
    result = clip_values(
        _parse_float_list(values), min_clip=min_clip, max_clip=max_clip
    )
    click.echo(result)


@numeric.command("to-integers", context_settings={"ignore_unknown_options": True})
@click.argument("values")
def to_integers(values):
    """Convert numeric strings to integers, excluding non-numerical values.

    Example: python -m src.cli numeric to-integers "1,abc,2,3.5,4"
    """
    items = [item.strip() for item in values.split(",") if item.strip()]
    result = convert_to_integers(items)
    click.echo(result)


@numeric.command("log", context_settings={"ignore_unknown_options": True})
@click.argument("values")
def log_scale(values):
    """Transform positive numerical values to logarithmic scale.

    Example: python -m src.cli numeric log "-2,0,1,2.718,10"
    """
    result = logarithmic_scale(_parse_float_list(values))
    click.echo(result)


@text.command("tokenize")
@click.argument("input_text")
def tokenize(input_text):
    """Tokenize text into lower-cased alphanumeric words.

    Example: python -m src.cli text tokenize "Hello, World! 123"
    """
    result = tokenize_text(input_text)
    click.echo(result)


@text.command("remove-punctuation")
@click.argument("input_text")
def remove_punctuation(input_text):
    """Keep only alphanumeric characters and spaces from text.

    Example: python -m src.cli text remove-punctuation "Hello, World! 123"
    """
    result = select_alphanumerical_and_spaces(input_text)
    click.echo(result)


@text.command("remove-stopwords")
@click.argument("input_text")
@click.option(
    "--stop-words",
    type=str,
    default="",
    show_default=True,
    help="Comma-separated list of stop-words to remove.",
)
def remove_stopwords(input_text, stop_words):
    """Remove specified stop-words from lower-cased text.

    Example: python -m src.cli text remove-stopwords "This is a test" --stop-words "is,a"
    """
    stop_list = [w.strip().lower() for w in stop_words.split(",") if w.strip()]
    result = remove_stop_words(input_text, stop_list)
    click.echo(result)


@struct.command("shuffle")
@click.argument("values")
@click.option(
    "--seed",
    type=int,
    default=None,
    show_default=True,
    help="Integer seed to ensure reproducibility.",
)
def shuffle(values, seed):
    """Randomly shuffle a list of values.

    Example: python -m src.cli struct shuffle "a,b,c,d" --seed 42
    """
    items = [item.strip() for item in values.split(",") if item.strip()]
    result = shuffle_list(items, seed=seed)
    click.echo(result)


@struct.command("flatten")
@click.argument("lists_str")
def flatten(lists_str):
    """Flatten a list of lists (sublists separated by ';', items by ',').

    Example: python -m src.cli struct flatten "1,2;3,4;5"
    """
    nested = [
        [item.strip() for item in sublist.split(",") if item.strip()]
        for sublist in lists_str.split(";")
        if sublist.strip()
    ]
    result = flatten_list(nested)
    click.echo(result)


@struct.command("unique")
@click.argument("values")
def unique(values):
    """Remove duplicated values from a list while preserving order.

    Example: python -m src.cli struct unique "a,b,a,c,b"
    """
    items = [item.strip() for item in values.split(",") if item.strip()]
    result = remove_duplicated_values(items)
    click.echo(result)


if __name__ == "__main__":
    cli()
