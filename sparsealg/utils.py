import numpy as np
import sys

def slice_to_indice(s:slice, dim:int)->list[int]:
    """Convert slice object to list

    Args:
        s (slice): Slice object
        dim (int): Length of vector or matrix

    Returns:
        list[int]: list of index
    """
    idx_step = 1 if s.step is None else s.step
    if s.start is None:
        idx_start = 0
    elif s.start < 0:
        idx_start = dim + s.start
    else:
        idx_start = s.start
    
    if s.stop is None:
        idx_stop = dim
    elif s.stop < 0:
        idx_stop = dim + s.stop
    else:
        idx_stop = s.stop

    return [id for id in range(idx_start, idx_stop, idx_step)]