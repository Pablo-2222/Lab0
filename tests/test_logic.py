"""Unit tests for the preprocessing functions in src/preprocessing.py."""

import math
import pytest

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


@pytest.fixture
def sample_numbers():
    """Provide a reusable list of numbers for numerical tests."""
    return [1, 2, 3, 4, 5]


@pytest.fixture
def sample_with_missing():
    """Provide a reusable list containing valid and missing values."""
    return ["a", None, "", float("nan"), "b"]


def test_remove_missing_values(sample_with_missing):
    """Verify None, empty strings, and NaN are removed."""
    assert remove_missing_values(sample_with_missing) == ["a", "b"]


def test_fill_missing_values_default(sample_with_missing):
    """Verify missing values are replaced with 0 by default."""
    assert fill_missing_values(sample_with_missing) == ["a", 0, 0, 0, "b"]


def test_fill_missing_values_custom(sample_with_missing):
    """Verify missing values are replaced with a custom value."""
    assert fill_missing_values(sample_with_missing, fill_value="NA") == [
        "a",
        "NA",
        "NA",
        "NA",
        "b",
    ]


@pytest.mark.parametrize(
    "values, expected",
    [
        ([1, 2, 2, 3, 1], [1, 2, 3]),
        (["a", "b", "a", "c"], ["a", "b", "c"]),
        ([], []),
    ],
)
def test_remove_duplicated_values(values, expected):
    """Verify duplicates are removed while preserving order."""
    assert remove_duplicated_values(values) == expected


def test_normalize_min_max_default(sample_numbers):
    """Verify min-max normalization to default [0.0, 1.0] range."""
    assert normalize_min_max(sample_numbers) == pytest.approx([0.0, 0.25, 0.5, 0.75, 1.0])


def test_normalize_min_max_custom_range(sample_numbers):
    """Verify min-max normalization to a custom [new_min, new_max] range."""
    assert normalize_min_max(sample_numbers, new_min=-1.0, new_max=1.0) == pytest.approx(
        [-1.0, -0.5, 0.0, 0.5, 1.0]
    )


def test_normalize_min_max_edge_cases():
    """Verify empty and constant lists are handled safely."""
    assert normalize_min_max([]) == []
    assert normalize_min_max([7, 7, 7], new_min=2.0, new_max=5.0) == [2.0, 2.0, 2.0]


def test_standardize_z_score(sample_numbers):
    """Verify z-score standardization produces zero mean and unit std."""
    result = standardize_z_score(sample_numbers)
    expected_std = math.sqrt(2.0)
    expected = [(x - 3.0) / expected_std for x in sample_numbers]
    assert result == pytest.approx(expected)


def test_standardize_z_score_edge_cases():
    """Verify empty and zero-variance lists return expected outputs."""
    assert standardize_z_score([]) == []
    assert standardize_z_score([4, 4, 4]) == [0.0, 0.0, 0.0]


@pytest.mark.parametrize(
    "values, min_clip, max_clip, expected",
    [
        ([-5, 0, 5, 10, 15], 0, 10, [0, 0, 5, 10, 10]),
        ([1.5, 2.5, 3.5], 2.0, 3.0, [2.0, 2.5, 3.0]),
    ],
)
def test_clip_values(values, min_clip, max_clip, expected):
    """Verify numerical clipping with required min and max bounds."""
    assert clip_values(values, min_clip, max_clip) == expected


def test_convert_to_integers():
    """Verify valid integer strings are converted and non-numeric ones ignored."""
    assert convert_to_integers(["1", "abc", "-3", "4.5", "10"]) == [1, -3, 10]


def test_logarithmic_scale():
    """Verify natural log is computed only for strictly positive numbers."""
    result = logarithmic_scale([-2, 0, 1, math.e, 10])
    assert result == pytest.approx([0.0, 1.0, math.log(10)])


@pytest.mark.parametrize(
    "text, expected",
    [
        ("Hello, World! 123", ["hello", "world", "123"]),
        ("MLOps: Lab_0 & CI!", ["mlops", "lab", "0", "ci"]),
        ("", []),
    ],
)
def test_tokenize_text(text, expected):
    """Verify text tokenization extracts lower-cased alphanumeric words."""
    assert tokenize_text(text) == expected


def test_select_alphanumerical_and_spaces():
    """Verify punctuation is removed while keeping alphanumeric chars and spaces."""
    assert select_alphanumerical_and_spaces("Hello, World! 2026.") == "Hello World 2026"


@pytest.mark.parametrize(
    "text, stop_words, expected",
    [
        ("This is a simple test", ["is", "a"], "this simple test"),
        ("The cat and the dog", ["the", "and"], "cat dog"),
    ],
)
def test_remove_stop_words(text, stop_words, expected):
    """Verify stop-words are removed from lower-cased text."""
    assert remove_stop_words(text, stop_words) == expected


def test_flatten_list():
    """Verify a list of lists is flattened into a single list."""
    assert flatten_list([[1, 2], [3], [], [4, 5]]) == [1, 2, 3, 4, 5]


def test_shuffle_list_reproducible(sample_numbers):
    """Verify shuffling with the same seed is reproducible and preserves elements."""
    first = shuffle_list(sample_numbers, seed=42)
    second = shuffle_list(sample_numbers, seed=42)
    assert first == second
    assert sorted(first) == sorted(sample_numbers)


def test_shuffle_list_no_seed(sample_numbers):
    """Verify shuffling without a seed preserves all original elements."""
    result = shuffle_list(sample_numbers)
    assert sorted(result) == sorted(sample_numbers)
