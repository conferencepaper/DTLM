import random

def check_pattern(lst):
    n = len(lst)
    # Loop through the list to check for consecutive patterns
    for i in range(n):
        for length in range(1, (n - i) // 2 + 1):  # Length of the potential repeated sequence
            # Check if the subsequence of this length is repeated consecutively
            if lst[i:i+length] == lst[i+length:i+2*length]:
                return True  # A pattern is found
    return False  # No consecutive patterns found
def shuffle_without_pattern(lst):
    original_indices = list(range(len(lst)))  # Create a list of indices corresponding to the original list
    while True:
        combined = list(zip(lst, original_indices))  # Combine the list elements with their indices
        random.shuffle(combined)  # Shuffle the combined list
        lst_shuffled, indices_shuffled = zip(*combined)  # Unzip to get shuffled list and corresponding indices
        
        if not check_pattern(lst_shuffled):  # Check if there are no patterns in the shuffled list
            break  # If no pattern, we are done
    return list(lst_shuffled), list(indices_shuffled)
def restore_original_list(shuffled_lst, shuffled_indices):
    # Initialize a list of the same length as the original list
    original_lst = [None] * len(shuffled_lst)
    # Place the shuffled elements back into their original positions
    for i, index in enumerate(shuffled_indices):
        original_lst[index] = shuffled_lst[i]
    return original_lst