"""
@author: Inki
@contact: inki.yinji@qq.com
@version: Created in 2020 0922; last modified in 2020 1020
@note: Distance or similarity function between instance and bag,
and all vector data's type must be numpy.array.
"""

import numpy as np
from I2I import kernel_rbf, dis_euclidean


def max_similarity(para_bag, para_ins, para_ins_dis='rbf', para_gamma=1):
    """
    Compute the similarity between bag and discriminative instance.
    :param
        para_bag:
            The given bag, and its have not instance label.
        para_ins:
            The given discriminative instance.
        para_ins_dis:
            The type of distance / similarity function for two instances.
                1. 'euclidean': the euclidean distance.
                2. 'rbf': the RBF kernel.
            And its default setting is 'rbf'.
        para_gamma:
            The gamma for RBF function.
    :return
        The maximum rbf value.
    """
    ret_dis = -np.inf

    if para_ins_dis == 'rbf':
        for ins in para_bag:
            ret_dis = max(ret_dis, kernel_rbf(ins, para_ins, para_gamma))
    elif para_ins_dis == 'euclidean':
        ret_dis = np.inf
        for ins in para_bag:
            ret_dis = min(ret_dis, dis_euclidean(ins, para_ins))
    elif para_ins_dis == 'rbf2':
        for ins in para_bag:
            ret_dis = max(ret_dis, np.exp(-para_gamma * (dis_euclidean(ins, para_ins)**2)))

    if ret_dis == -1:
        print("Fetal error: the similarity between bag and instance is -1.")

    return ret_dis


def ave_similarity(para_bag, para_ins, para_ins_dis='rbf', para_gamma=1):
    """
    Compute the average similarity.
    More detail please refer max_similarity.
    """
    ret_dis = 0

    if para_ins_dis == 'rbf':
        for ins in para_bag:
            ret_dis += kernel_rbf(ins, para_ins, para_gamma)
    elif para_ins_dis == 'euclidean':
        for ins in para_bag:
            ret_dis += dis_euclidean(ins, para_ins)
    elif para_ins_dis == 'rbf2':
        for ins in para_bag:
            ret_dis += np.exp(-para_gamma * (dis_euclidean(ins, para_ins)**2))

    if ret_dis == -1:
        print("Fetal error: the similarity between bag and instance is -1.")

    return ret_dis / len(para_bag)
