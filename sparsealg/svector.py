import numpy as np
import sys
from sparsealg.core import svector_meta
from sparsealg import config

class svector(svector_meta):
    """Real sparse vector

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
    allow_dtype = config.allow_dtype_real

    def __matmul__(self, another) -> float:
        output = 0.
        for i, v in zip(self.indice, self.values):
            output += v*another[i]
        return output


class scvector(svector_meta):
    """Complex sparse vector

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
    allow_dtype = config.allow_dtype_complex
    @property
    def conjugate(self):
        return type(self)(self.dim, indice = self.indice, values = np.conjugate(self.values), dtype = self.dtype)
    
    @property
    def real(self):
        values_real_wzero = self.values.real
        values_real = np.array([], dtype = values_real_wzero.dtype)
        indice_real = np.array([], dtype = int)

        for value_real, index in zip(values_real_wzero, self.indice):
            if not np.isclose(value_real, 0.):
                indice_real = np.append(indice_real, index)
                values_real = np.append(values_real, value_real)
        
        return svector(dim = self.dim, indice = indice_real, values = values_real, dtype = values_real_wzero.dtype)
    
    @property
    def imag(self):
        values_imag_wzero = self.values.imag
        values_imag = np.array([], dtype = values_imag_wzero.dtype)
        indice_imag = np.array([], dtype = int)

        for value_imag, index in zip(values_imag_wzero, self.indice):
            if not np.isclose(value_imag, 0.):
                indice_imag = np.append(indice_imag, index)
                values_imag = np.append(values_imag, value_imag)
        
        return svector(dim = self.dim, indice = indice_imag, values = values_imag, dtype = values_imag_wzero.dtype)
    
    def __matmul__(self, another):
        output = 0.
        another_conjugate = another.conjugate
        for i, v in zip(self.indice, self.values):
            output += v*another_conjugate[i]
        return output