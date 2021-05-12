"""
@author: Inki
@contact: inki.yinji@qq.com
@version: Created in 2020 0703; last modified in 2020 1231.
@note: Distance or similarity function for single-instance learning (SIL),
and all vector data's type must be numpy.array.
"""

import os
import numpy as np
np.set_printoptions(precision=6)

__all__ = ['dis_euclidean',
           'kernel_gaussian',
           'kernel_rbf']


def dis_euclidean(para_arr1, para_arr2):
    """The eucildean distance, i.e.m $||para_arr1 - para_arr2||^2$
    @param:
        para_arr1:
            The given array, e.g., np.array([1, 2])
        para_arr2:
            The given array like para_arr1.
    @return
        A scalar.
    """
    return np.sqrt(np.sum((para_arr1 - para_arr2)**2))


def kernel_gaussian(para_arr1, para_arr2, para_gamma=1):
    """
    The details please refer the kernel_rbf.
    """
    return np.exp(-para_gamma * dis_euclidean(para_arr1, para_arr2)**2)
    

def kernel_rbf(para_arr1, para_arr2, para_gamma=1):
    r"""
    The Gaussian RBF kernel for SIL, i.e., $exp (\gama ||para_arr1 - para_arr2||^2)$.
    @param: 
    ------------
        para_arr1:
            The given array, e.g., np.array([1, 2])
        para_arr2:
            The given array like para_arr1.
        para_gama:
            The gama for RBF kernel.
    ------------
    @return:
    ------------
        A scalar.
    ------------
    """
    
    return np.exp(-para_gamma * dis_euclidean(para_arr1, para_arr2))
