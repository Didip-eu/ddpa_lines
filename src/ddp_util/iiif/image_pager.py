# TODO(anguelos): move to a general didip flask module
def create_pagers(result_length, skip, item_count):
    """Creates REST API pagers.

    Args:
        result_length (int): total number of results
        skip (int): _description_
        item_count (int): _description_

    Returns:
        tuple[tuple[int, int], ...]: A tuple of tuples of the form (skip, item_count) for first, previous, current, following, and last pagers.
    """
    last_item = result_length - 1
    first = 0, max(min(item_count, result_length),1)
    prev = max(skip-item_count,0), max(min(item_count , last_item),1)
    current = min(max(skip,0),last_item), max(min(item_count , result_length-skip),1)
    following = min(skip+item_count,last_item),max(min(item_count, last_item-(skip+item_count)),1)
    last = max(last_item-item_count,0), max(min(item_count, result_length),1)
    return first, prev, current, following, last
