import numpy as np
import sys
from sparsealg import utils

class svector_meta:
    """Abstract class for sparse vector

    Args:
        allow_dtype (tuple[class]): Usable dtype in this class.
        dim (int): Dimension of vector.
        indice (np.ndarray): Indice of non zero elements.
        values (np.ndarray): Values of non zero elements. indice[i]'s value equals to values[i].
    Attributes:
        allow_dtype (tuple[class]): Usable dtype in this class.
        dim (int): Dimension of vector.
        indice (np.ndarray): Indice of non zero elements.
        values (np.ndarray): Values of non zero elements. indice[i]'s value equals to values[i].
    Note:
        * Indice[i]'s value equals to values[i].
    """
    allow_dtype = None
    def __init__(self, dim:int, indice:np.ndarray = None, values:np.ndarray = None, dtype = None):
        self.dim = dim
        assert dtype in self.allow_dtype, f"{dtype}"
        self.dtype = dtype
        self.indice = np.array([], dtype = int) if indice is None else np.copy(indice)
        self.values = np.array([], dtype = self.dtype) if values is None else np.copy(values)
    
    def __len__(self)->int:
        return self.dim
    
    @property
    def num_nonzero(self)->int:
        return len(self.indice)
    
    def copy(self):
        return type(self)(self.dim, self.indice, self.values, self.dtype)
    
    def where(self, i:int)->int:
        """Find the index number "j" where indice[j] == i

        Args:
            i (int): Index number.
        Returns:
            int: Index number "j".
        """
        assert i < self.dim #インデックス番号iが次元数以下か判定
        index = np.where(self.indice == i)[0]

        if len(index) > 1: raise Exception("Found multi defined elements")
        return None if len(index) == 0 else index[0]
    
    def __setitem__(self, indice, value):
        if isinstance(indice, (list, tuple)):
            indice = indice
        elif isinstance(indice, slice):
            indice = utils.slice_to_indice(indice, self.dim)
        else:
            indice = [indice]

        value = self.dtype(value)
        for index in indice:
            idx = self.where(index)
            if value == self.dtype(0.):
                if idx is not None:
                    self.indice = np.delete(self.indice, index)
                    self.values = np.delete(self.values, index)
            else:
                if idx is None:
                    self.indice = np.append(self.indice, index)
                    self.values = np.append(self.values, value)
                else:
                    self.values[idx] = value
    
    def __getitem__(self, i):
        if isinstance(i, (list, tuple)):
            indice = []
            for ii in i:
                if ii < 0:
                    indice.append(self.dim + ii)
                else:
                    indice.append(ii)
            dim_new = len(indice)
            indice_new = np.array([], dtype = int)
            values_new = np.array([], dtype = self.dtype)

            for idx in range(len(indice)):
                id = self.where(indice[idx])
                if id is not None:
                    indice_new = np.append(indice_new, idx)
                    values_new = np.append(values_new, self.values[id])
            return type(self)(dim = dim_new, indice = indice_new, values = values_new, dtype = self.dtype)
        
        elif isinstance(i, slice):
            indice = utils.slice_to_indice(i, self.dim)

            dim_new = len(indice)
            indice_new = np.array([], dtype = int)
            values_new = np.array([], dtype = self.dtype)
            for c, index in enumerate(indice):
                idx = self.where(index)
                if idx is not None:
                    indice_new = np.append(indice_new, c)
                    values_new = np.append(values_new, self.values[idx])

            return type(self)(dim = dim_new, indice = indice_new, values = values_new, dtype = self.dtype)

        else:
            index = self.where(i)
            return 0. if index is None else self.values[index]
    
    def __neg__(self):
        return type(self)(dim = self.dim, indice = self.indice, values = -self.values, dtype = self.dtype)
    
    def __add__(self, another):
        output = self.copy()

        if type(another) != type(self):
            i = np.arange(self.dim, dtype = int)
            v = self.dtype(another)*np.ones(self.dim, dtype = self.dtype)
            another = type(self)(dim = self.dim, indice = i, values = v, dtype = self.dtype)

        for i, v in zip(another.indice, another.values):
            output[i] += v
        
        return output
    
    def __radd__(self, another):
        return self.__add__(another)

    def __sub__(self, another):
        return self.__add__(-another)
    
    def __rsub__(self, another):
        return (-self).__add__(another)
    
    def __mul__(self, another):
        if type(another) == type(self):
            output = self.copy()
            for i, index in enumerate(output.indice):
                index_another = another.where(index)

                if index_another is None:
                    output[index] = 0.
                else:
                    output.values[i] *= another.values[index_another]
            return output
        else:
            if another == 0.:
                return type(self)(self.dim)
            else:
                output = self.copy()
                output.values *= self.dtype(another)
                return output
    
    def __rmul__(self, another):
        return self.__mul__(another)
    
    def __matmul__(self, another):
        raise NotImplementedError