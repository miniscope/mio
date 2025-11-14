"""
Helper functions for processing metadata in recordings.
"""

from typing import List


def make_combined_list(list_of_lists: List[List]) -> List:
    """
    Combine multiple lists into a single list with unique elements, preserving order.

    Parameters:
        list_of_lists: List[List]
            A list of lists to combine.

    Returns:
        List: A combined list with unique elements.
    """
    combined_list = []
    for lst in list_of_lists:
        for item in lst:
            if item not in combined_list:
                combined_list.append(item)
    return combined_list
