"""
@author: Inki
@contact: inki.yinji@gmail.com
@version: Created in 2020 1120, last modified in 2020 1120.
@note: The refer link: https://blog.csdn.net/weixin_44575152/article/details/109595872
"""
import os
import numpy as np
import pandas as pd
from Prototype import MIL
from I2B import max_similarity
from FunctionTool import print_progress_bar, load_file, get_k_cross_validation_index


class MILES(MIL):
    """
    The class of MILFM.
    @param:
        loops:
            The number of repetitions for kMeans.
        k:
            The number of clustering centers for kMeans.
        gamma:
            The gamma for i2i distance, include rbf and rbf2 (Gaussian distance).
    """

    def __init__(self, path, loops=5, k=10, k_c=50, gamma=1, bag_space=None):
        """
        The constructor.
        """
        super(MILES, self).__init__(path, bag_space=bag_space)
        self.loops = loops
        self.k = k
        self.k_c = k_c
        self.gamma = gamma
        self.mapping_path = ''
        self.full_mapping = []
        self.tr_idx = []
        self.te_idx = []
        self.__initialize_milfm()
        self.__full_mapping()

    def __initialize_milfm(self):
        """
        The initialize of MilFm.
        """
        self.mapping_path = 'D:/Data/TempData/Mapping/MilDm/' + self.data_name + '_max_rbf_' + str(self.gamma) + '.csv'

    def __full_mapping(self):
        """
        Mapping bags by using all instances.
        @Note:
            The size of data set instance space will greatly affect the running time.
        """

        self.full_mapping = np.zeros((self.num_bag, self.num_ins))
        if not os.path.exists(self.mapping_path) or os.path.getsize(self.mapping_path) == 0:
            print("Full mapping starting...")
            open(self.mapping_path, 'a').close()

            for i in range(self.num_bag):
                print_progress_bar(i, self.num_bag)
                # print("%d-th bag mapping..." % i)
                for j in range(self.num_ins):
                    self.full_mapping[i, j] = max_similarity(self.bag_space[i][0][:, :self.num_att],
                                                             self.ins_space[j], 'rbf', self.gamma)
            pd.DataFrame.to_csv(pd.DataFrame(self.full_mapping), self.mapping_path,
                                index=False, header=False, float_format='%.6f')
            print("Full mapping end...")
        else:
            temp_data = load_file(self.mapping_path)
            for i in range(self.num_bag):
                self.full_mapping[i] = [float(value) for value in temp_data[i].strip().split(',')]

    def get_mapping(self):
        """
        Get mapping.
        """

        self.tr_idx, self.te_idx = get_k_cross_validation_index(self.num_bag)
        for loop_k in range(self.k):

            # Step 1. Get iip.
            temp_tr_idx = self.tr_idx[loop_k]
            temp_iip = []
            for tr_idx in temp_tr_idx:
                for ins_i in range(self.bag_size[tr_idx]):
                    temp_iip.append(self.ins_idx[tr_idx] + ins_i)

            # Step 2. Mapping.
            temp_mapping = self.full_mapping[:, temp_iip]
            yield temp_mapping[temp_tr_idx], self.bag_lab[temp_tr_idx],\
                  temp_mapping[self.te_idx[loop_k]], self.bag_lab[self.te_idx[loop_k]], None
