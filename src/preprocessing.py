import math
import random
import re


def remove_missing_values(values):
    """
    Removal of missing values
    Input: List of values, including missing values like None, '' and nan
    Output: list of values removing the missing ones
    """
    return [
        x
        for x in values
        if x is not None and x != "" and not (isinstance(x, float) and math.isnan(x))
    ]


def fill_missing_values(values, fill_value=0):
    """
    Filling missing values
    Inputs:
    List of values, including missing values
    The value to fill the missing values (default to 0)
    Output: list of values where the missing values are replaced with the given filling value
    """
    return [
        (
            fill_value
            if x is None or x == "" or (isinstance(x, float) and math.isnan(x))
            else x
        )
        for x in values
    ]


def remove_duplicated_values(values):
    """
    Removal of duplicated values
    Input: List of values
    Output: List of unique values
    """
    return list(dict.fromkeys(values))


def normalize_min_max(values, new_min=0.0, new_max=1.0):
    """
    Normalization of numerical values using the min-max method
    Inputs:
    List of numerical values
    New minimum (default to 0.0)
    New maximum (default to 1.0)
    Output: list of normalized values
    """
    if not values:
        return []
    min_val = min(values)
    max_val = max(values)
    if min_val == max_val:
        return [new_min for _ in values]
    return [
        new_min + (x - min_val) * (new_max - new_min) / (max_val - min_val)
        for x in values
    ]


def standardize_z_score(values):
    """
    Standardization of numerical values using the z-score method
    Input: List of numerical values
    Output: List of standardized values
    """
    if not values:
        return []
    mean = sum(values) / len(values)
    variance = sum((x - mean) ** 2 for x in values) / len(values)
    if variance == 0:
        return [0.0 for _ in values]
    std_dev = math.sqrt(variance)
    return [(x - mean) / std_dev for x in values]


def clip_values(values, min_clip, max_clip):
    """
    Clipping numerical values to a certain range
    Inputs:
    List of numerical values
    Minimum value to clip
    Maximum value to clip
    Output: List of clipped values
    """
    return [max(min_clip, min(x, max_clip)) for x in values]


def convert_to_integers(values):
    """
    Conversion of values to integers
    Input: List of strings (it can include both numerical and non-numerical values)
    Output: List of values converted to integers (non-numerical values are excluded from the output)
    """
    result = []
    for x in values:
        try:
            result.append(int(x))
        except ValueError:
            pass
    return result


def logarithmic_scale(values):
    """
    Logarithmic scale transformation
    Input: List of values
    Output: List of values converted to logarithmic scale (just the original positive numbers)
    """
    return [math.log(x) for x in values if x > 0]


def tokenize_text(text):
    """
    Tokenization of text into words, selecting only alphanumeric characters and lower-casing words
    Input: Text to be processed
    Output: Processed text
    """
    return re.findall(r"[a-z0-9]+", text.lower())


def select_alphanumerical_and_spaces(text):
    """
    Selection of alphanumerical characters and spaces
    Input: Text to be processed
    Output: Processed text
    """
    return "".join(c for c in text if c.isalnum() or c.isspace())


def remove_stop_words(text, stop_words):
    """
    Removal of stop-words from text (it should be lower-cased)
    Inputs:
    Text to be processed
    Stop words to be removed
    Output: Processed text
    """
    words = text.lower().split()
    filtered_words = [word for word in words if word not in stop_words]
    return " ".join(filtered_words)


def flatten_list(list_of_lists):
    """
    Flatten a list of lists
    Input: A list of lists
    Output: A flattened list
    """
    return [item for sublist in list_of_lists for item in sublist]


def shuffle_list(values, seed=None):
    """
    Random shuffling of a list of values
    Inputs:
    List of values
    Seed to ensure reproducibility
    Output: list of shuffled values
    """
    if seed is not None:
        random.seed(seed)
    shuffled = list(values)
    random.shuffle(shuffled)
    return shuffled
