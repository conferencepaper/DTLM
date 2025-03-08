import jellyfish
from collections import Counter
from fuzzywuzzy import fuzz

# Function to get the index range considering bounds
def get_index(i, diff, n_max):
    """
    Get index range with a given difference and maximum limit.

    Parameters:
    - i: Current index.
    - diff: Difference to apply to the index.
    - n_max: Maximum length (upper bound).

    Returns:
    A list of indices within the calculated range.
    """
    return [j for j in range(i, min(i + diff + 1, n_max))]

# Function to get the index of the maximum similarity score using Jaro-Winkler
def get_arg_max_jw(out, l_min_input):
    """
    Get the index of the element in l_min_input that has the maximum 
    Jaro-Winkler similarity score to the 'out' string.

    Parameters:
    - out: Target string.
    - l_min_input: List of strings to compare to the target.

    Returns:
    The index of the element in l_min_input with the highest similarity.
    """
    similarities = [jellyfish.jaro_winkler_similarity(out, i) for i in l_min_input]
    return similarities.index(max(similarities))

# Function to get the index of the maximum similarity score using FuzzyWuzzy
def get_arg_max_fuzz(out, l_min_input):
    """
    Get the index of the element in l_min_input that has the maximum 
    FuzzyWuzzy ratio to the 'out' string.

    Parameters:
    - out: Target string.
    - l_min_input: List of strings to compare to the target.

    Returns:
    The index of the element in l_min_input with the highest similarity.
    """
    ratios = [fuzz.ratio(out, i) for i in l_min_input]
    return ratios.index(max(ratios))

# Function to find the most frequent list (mode) from a list of lists
def most_frequent_list(all_l):
    """
    Find the most frequent list in a list of lists.

    Parameters:
    - all_l: List of lists.

    Returns:
    The most frequent list and its count.
    """
    my_list_tuples = [tuple(sublist) for sublist in all_l]  # Convert lists to tuples for hashing
    counter = Counter(my_list_tuples)
    most_common_element_tuple, count = counter.most_common(1)[0]  # Get most common element
    return list(most_common_element_tuple), count  # Convert tuple back to list

# Function to generate the transformed list using fuzzy matching
def get_list(l_input, l_out, similarity_func=get_arg_max_fuzz):
    """
    Generate a transformed list by matching elements in l_out to the most similar elements in l_input.

    Parameters:
    - l_input: Input list.
    - l_out: Output list to match against l_input.
    - similarity_func: Function to use for similarity calculation.

    Returns:
    A transformed list 'L' and indices where mismatches occurred.
    """
    n_max = len(l_input)
    diff = len(l_input) - len(l_out)
    L = []
    out_index = []
    in_idx = 0  # Pointer for l_input

    for out in range(len(l_out)):
        # Adjust index based on current position in l_input
        index = get_index(in_idx, diff, n_max)
        l_min_input = [l_input[j] for j in index]

        # Get the index of the best match in l_min_input
        index_local = similarity_func(l_out[out], l_min_input)
        index_max = index[index_local]

        if in_idx == index_max:
            # Direct match, append from l_out
            L.append(l_out[out])
            in_idx += 1  # Move to next element in l_input
        else:
            # Mismatch, append elements from l_input up to the match
            out_index.append(out)
            for j in range(in_idx, index_max):
                L.append(l_input[j])
            L.append(l_out[out])  # Append the matched element from l_out
            diff -= (index_max - in_idx)
            in_idx = index_max + 1  # Update in_idx to the position after index_max
    # Append any remaining elements from l_input
    while in_idx < n_max:
        L.append(l_input[in_idx])
        in_idx += 1

    return L, out_index


# Main function to process input and find the most common transformation
def missing_values(l_input, l_out, similarity_func=get_arg_max_jw):
    """
    Process the input and output lists, apply the transformation, and return the most frequent result.
    
    Parameters:
    - l_input: Input list.
    - l_out: Output list to transform.
    - similarity_func: Function to use for similarity calculation (default is Jaro-Winkler).

    Returns:
    The most frequent transformed list.
    """
    all_l = []
    all_out_index = []

    for j in range(len(l_out)):
        L, out_index = get_list(l_input[j:], l_out[j:], similarity_func)
        all_l.append(l_out[:j] + L)  # Store the combined result
        if out_index:
            all_out_index.append(out_index[0] + j)

    most_common_element, _ = most_frequent_list(all_l)
    return most_common_element