import numpy as np
from sparsealg.svector import svector, scvector
from sparsealg import config
import sys

def to_dense(s_vec)->np.ndarray:
    """Convert sparse vector to dense vector.

    Args:
        s_vec (core.svector_meta): Sparse vector.
    Returns:
        np.ndarray: Dense vector.
    """
    d_vec = np.zeros(s_vec.dim, dtype = s_vec.dtype)
    for index, value in zip(s_vec.indice, s_vec.values):
        d_vec[index] = value
        
    return d_vec

def to_sparse(d_vec:np.ndarray):
    """Convert dense vector to sparse vector.

    Args:
        d_vec (np.ndarray): Dense vector.
    Returns:
        core.svector_meta: Sparse vector.
    """
    assert len(d_vec.shape) == 1

    indice = np.array([], dtype = np.int64)
    values = np.array([], dtype = d_vec.dtype)
    for idx, v in enumerate(d_vec):
        if v == 0.: continue #ゼロ要素は無視
        indice = np.append(indice, idx)
        values = np.append(values, v)

    if d_vec.dtype in config.allow_dtype_real:
        return svector(dim = len(d_vec), indice = indice, values = values, dtype = d_vec.dtype)
    elif d_vec.dtype in config.allow_dtype_complex:
        return scvector(dim = len(d_vec), indice = indice, values = values, dtype = d_vec.dtype)
    else:
        raise NotImplementedError


def dot(svec1:svector, svec2:svector):
    """Inner product for real sparse vectors.

    Args:
        svec1 (core.svector.svector): Sparse vector.
        svec2 (core.svector.svector): Sparse vector.
    Returns:
        (float): Inner product.
    """
    output = 0.
    for i, v in zip(svec1.indice, svec1.values):
        output += v*svec2[i]
    return output


def cdot(scvec1:scvector, scvec2:scvector):
    """Inner product for complex sparse vectors.

    Args:
        svec1 (core.svector.scvector): Sparse vector.
        svec2 (core.svector.scvector): Sparse vector.
    Returns:
        (complex): Inner product.
    """
    output = 0.
    for i, v in zip(scvec1.indice, scvec1.values):
        output += v*np.conjugate(scvec2[i])
    return output

def power(svec, p):
    """Return sparse vector powered by p

    Args:
        svec (sparsealg.core.svector_meta): Sparse vector.
        p (scalar): Scalar value. 
    Returns:
        sparsealg.core.svector_meta: Sparse vector.
    """
    return type(svec)(dim = svec.dim, indice = svec.indice, values = np.power(svec.values, p), dtype = svec.dtype)


def sum(svec):
    """Return sum

    Args:
        svec (sparsealg.core.svector_meta): Sparse vector
    Returns:
        Scalar: Sum of elements in sparse vector.
    """
    return svec.dtype(np.sum(svec.values))

def abs(svec) -> svector:
    """Return abs

    Args:
        svec (sparsealg.core.svector_meta): Sparse vector
    Returns:
        sparsealg.core.svector_meta: Abs of svec.
    """
    return svector(svec.dim, svec.indice, np.abs(svec.values))

def norm(svec, ord : int = 2):
    """Return norm

    Args:
        svec (sparsealg.core.svector_meta): Sparse vector
        ord (int): Order of norm.
    Returns:
        Scalar: Norm of svec.
    """
    return float(np.linalg.norm(svec.values, ord))