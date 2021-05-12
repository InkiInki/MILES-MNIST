"""
@author: Inki
@contact: inki.yinji@qq.com
@version: Created in 2020 0907 1601, last modified in 2021 0415.
"""

import numpy as np
import warnings
import time
from MILES import MILES
from ClassifyTool import Classify
from MnistLoadTool import MnistLoader
warnings.filterwarnings('ignore')


def test_10cv():
    """
    """
    po_label = 9
    file_name = "mnist" + str(po_label) + ".none"  # "D:/Data/OneDrive/文档/Code/MIL1/Data/Text/talk_religion_misc.mat"
    mnist_path = "D:/Data/OneDrive/文档/Code/MIL1/Data"

    """======================================================="""
    loops = 5
    # tr_f1_k, tr_acc_k, tr_roc_k = np.zeros(loops), np.zeros(loops), np.zeros(loops)
    # tr_f1_s, tr_acc_s, tr_roc_s = np.zeros(loops), np.zeros(loops), np.zeros(loops)
    # tr_f1_j, tr_acc_j, tr_roc_j = np.zeros(loops), np.zeros(loops), np.zeros(loops)
    te_f1_k, te_acc_k, te_roc_k = np.zeros(loops), np.zeros(loops), np.zeros(loops)
    te_f1_s, te_acc_s, te_roc_s = np.zeros(loops), np.zeros(loops), np.zeros(loops)
    te_f1_j, te_acc_j, te_roc_j = np.zeros(loops), np.zeros(loops), np.zeros(loops)
    print("=================================================")
    print("MILDM with %s" % file_name.split("/")[-1].split(".")[0])
    bag_space = MnistLoader(seed=1, po_label=po_label, mnist_path=mnist_path).bag_space
    mil = MILES(file_name, bag_space=bag_space)
    for i in range(loops):
        classifier = Classify(["knn", "svm", "j48"], ["f1_score", "acc", "roc"])
        data_iter = mil.get_mapping()
        te_per = classifier.test(data_iter)
        # tr_f1_k[i], tr_acc_k[i], tr_roc_k[i] = tr_per["knn"][0], tr_per["knn"][1], tr_per["knn"][2]
        # tr_f1_s[i], tr_acc_s[i], tr_roc_s[i] = tr_per["svm"][0], tr_per["svm"][1], tr_per["svm"][2]
        # tr_f1_j[i], tr_acc_j[i], tr_roc_j[i] = tr_per["j48"][0], tr_per["j48"][1], tr_per["j48"][2]
        te_f1_k[i], te_acc_k[i], te_roc_k[i] = te_per["knn"][0], te_per["knn"][1], te_per["knn"][2]
        te_f1_s[i], te_acc_s[i], te_roc_s[i] = te_per["svm"][0], te_per["svm"][1], te_per["svm"][2]
        te_f1_j[i], te_acc_j[i], te_roc_j[i] = te_per["j48"][0], te_per["j48"][1], te_per["j48"][2]
        print("%.4lf, %.4lf, %.4lf; %.4lf, %.4lf, %.4lf; %.4lf, %.4lf, %.4lf; \n"
              % (te_f1_k[i], te_acc_k[i], te_roc_k[i],
                 te_f1_s[i], te_acc_s[i], te_roc_s[i],
                 te_f1_j[i], te_acc_j[i], te_roc_j[i]
                 ), end=" ")

    # print("%.4lf %.4lf %.4lf %.4lf %.4lf %.4lf "
    #       "%.4lf %.4lf %.4lf %.4lf %.4lf %.4lf "
    #       "%.4lf %.4lf %.4lf %.4lf %.4lf %.4lf " % (np.sum(tr_f1_k) / loops, np.std(tr_f1_k),
    #                                                 np.sum(tr_acc_k) / loops, np.std(tr_acc_k),
    #                                                 np.sum(tr_roc_k) / loops, np.std(tr_roc_k),
    #                                                 np.sum(tr_f1_s) / loops, np.std(tr_f1_s),
    #                                                 np.sum(tr_acc_s) / loops, np.std(tr_acc_s),
    #                                                 np.sum(tr_roc_s) / loops, np.std(tr_roc_s),
    #                                                 np.sum(tr_f1_j) / loops, np.std(tr_f1_j),
    #                                                 np.sum(tr_acc_j) / loops, np.std(tr_acc_j),
    #                                                 np.sum(tr_roc_j) / loops, np.std(tr_roc_j)), end="")
    print("knn-f1 std   knn-acc std   knn-roc std   svm-f1 std    svm-acc std   svm-roc std   "
          "j48-f1 std    j48-acc std   j48-roc std")
    print("%.4lf %.4lf %.4lf %.4lf %.4lf %.4lf "
          "%.4lf %.4lf %.4lf %.4lf %.4lf %.4lf "
          "%.4lf %.4lf %.4lf %.4lf %.4lf %.4lf " % (np.sum(te_f1_k) / loops, np.std(te_f1_k),
                                                    np.sum(te_acc_k) / loops, np.std(te_acc_k),
                                                    np.sum(te_roc_k) / loops, np.std(te_roc_k),
                                                    np.sum(te_f1_s) / loops, np.std(te_f1_s),
                                                    np.sum(te_acc_s) / loops, np.std(te_acc_s),
                                                    np.sum(te_roc_s) / loops, np.std(te_roc_s),
                                                    np.sum(te_f1_j) / loops, np.std(te_f1_j),
                                                    np.sum(te_acc_j) / loops, np.std(te_acc_j),
                                                    np.sum(te_roc_j) / loops, np.std(te_roc_j)), end="")


if __name__ == '__main__':
    s_t = time.time()
    test_10cv()
    print("%.4f" % (time.time() - s_t))
