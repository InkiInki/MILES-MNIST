"""
Author: Inki
Email: inki.yinji@qq.com
Create: 2021 0303
Last modify: 2021 0412
"""

import warnings
warnings.filterwarnings("ignore")


class Classify:
    """
    The class of classify.
    :param
        classifier_type:        The type of classifier.
            The default setting is knn.
    """

    def __init__(self, classifier_type=None, performance_type=None):

        self.__classifier_type = classifier_type
        self.__performance_type = performance_type
        self.tr_true_label_arr = {}
        self.tr_predict_arr = {}
        self.te_true_label_arr = {}
        self.te_predict_arr = {}
        self.tr_per = {}
        self.te_per = {}
        self.__init_classify()

    def __init_classify(self):
        """
        The initialize of Classify.
        """

        self.__classifier = []
        self.__performance_er = []
        if self.__classifier_type is None:
            self.__classifier_type = ["knn"]
        for classifier_type in self.__classifier_type:
            if classifier_type == "knn":
                from sklearn.neighbors import KNeighborsClassifier
                self.__classifier.append(KNeighborsClassifier(n_neighbors=3))
            elif classifier_type == "svm":
                from sklearn.svm import SVC
                self.__classifier.append(SVC(max_iter=10000))
            elif classifier_type == "j48":
                from sklearn.tree import DecisionTreeClassifier
                self.__classifier.append(DecisionTreeClassifier())
            self.tr_predict_arr[classifier_type], self.tr_true_label_arr[classifier_type] = [], []
            self.tr_per[classifier_type] = []
            self.te_predict_arr[classifier_type], self.te_true_label_arr[classifier_type] = [], []
            self.te_per[classifier_type] = []

        if self.__performance_type is None:
            self.__performance_type = ["f1_score"]
        for performance_type in self.__performance_type:
            if performance_type == "f1_score":
                from sklearn.metrics import f1_score
                self.__performance_er.append(f1_score)
            elif performance_type == "acc":
                from sklearn.metrics import accuracy_score
                self.__performance_er.append(accuracy_score)
            elif performance_type == "roc":
                from sklearn.metrics import roc_auc_score
                self.__performance_er.append(roc_auc_score)

    def test(self, data_iter, is_pre_tr=False):
        """
        :param
            data_iter:          Must is a iterator, and including
                                training data, training label, test data, test_label
        """
        for tr_data, tr_label, te_data, te_label, weight in data_iter:
            for classifier, classifier_type in zip(self.__classifier, self.__classifier_type):
                if weight is None:
                    model = classifier.fit(tr_data, tr_label)
                else:
                    model = classifier.fit(tr_data, tr_label, sample_weight=weight)

                if is_pre_tr:
                    predict = model.predict(tr_data)
                    self.tr_predict_arr[classifier_type].extend(predict)
                    self.tr_true_label_arr[classifier_type].extend(tr_label)

                predict = model.predict(te_data)
                self.te_predict_arr[classifier_type].extend(predict)
                self.te_true_label_arr[classifier_type].extend(te_label)

        for classifier_type in self.__classifier_type:
            for per_er in self.__performance_er:
                try:
                    self.tr_per[classifier_type].append(per_er(
                        self.tr_predict_arr[classifier_type],
                        self.tr_true_label_arr[classifier_type]
                    ))
                    self.te_per[classifier_type].append(per_er(
                        self.te_predict_arr[classifier_type],
                        self.te_true_label_arr[classifier_type]
                    ))
                except ValueError:
                    self.tr_per[classifier_type].append(0)
                    self.te_per[classifier_type].append(0)

        if is_pre_tr:
            return self.tr_per, self.te_per
        return self.te_per
