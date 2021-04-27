"""
@author: Inki
@contact: inki.yinji@gamil.com
@version: Created in 2020 1123, last modified in 2020 1123.
@note: You can refer this blog of https://blog.csdn.net/weixin_44575152/article/details/106600849
"""

import numpy as np
from Prototype import MIL
from sklearn.cluster import MiniBatchKMeans
from I2I import dis_euclidean


class miVLAD(MIL):
    """
    The algorithm of milVLAD.
    @param:
        k:
            The times of k-th cross validation.
        k_m:
            The clustering center numbers fro kMeans.
    @attribute:
        centers:
            The clustering centers.
    """

    def __init__(self, path, has_ins_label=True, k=10, k_m=1):
        super(miVLAD, self).__init__(path, has_ins_label)
        self.k = k
        self.k_m = k_m
        self.tr_idx = []
        self.te_idx = []

    def __bag_mapping(self, idx, centers, labels):
        """
        Mapping each given bag by using centers.
        @param:
            idx:
                The index of bag to be mapped.
            centers:
                The clustering centers of kMeans clustering algorithm.
            labels:
                The label for bag's instances which indicate the center to which the instance belongs.
        """
        ret_vec = np.zeros((self.k_m, self.dimensions))
        idx_ins = 0
        for ins in self.bags[idx, 0][:, :self.dimensions]:
            ret_vec[labels[idx_ins]] += ins - centers[labels[idx_ins]]
            idx_ins += 1
        ret_vec = np.resize(ret_vec, self.k_m * self.dimensions)
        ret_vec = np.sign(ret_vec) * np.sqrt(np.abs(ret_vec))
        return ret_vec / dis_euclidean(ret_vec, np.zeros_like(ret_vec))

    def get_mapping(self):
        """
        Mapping bags to vectors.
        """
        self.tr_idx, self.te_idx = self.get_index()
        for loop_k in range(self.k):
            # Step 1. Clustering to k_m blocks.
            temp_tr_ins = []
            temp_tr_ins_idx = [0]
            temp_tr_idx = self.tr_idx[loop_k]
            for tr_idx in temp_tr_idx:
                for ins in self.bags[tr_idx, 0][:, :self.dimensions]:
                    temp_tr_ins.append(ins)
                temp_tr_ins_idx.append(self.bags_size[tr_idx] + temp_tr_ins_idx[-1])
            temp_tr_ins = np.array(temp_tr_ins)
            temp_kmeans = MiniBatchKMeans(self.k_m)
            temp_kmeans.fit(temp_tr_ins)
            temp_centers = temp_kmeans.cluster_centers_
            temp_labels = temp_kmeans.labels_

            # Step 2. Mapping.
            temp_num_tr = len(temp_tr_idx)
            ret_tr_vec = np.zeros((temp_num_tr, self.k_m * self.dimensions))
            for idx_bag in range(temp_num_tr):
                ret_tr_vec[idx_bag] = self.__bag_mapping(temp_tr_idx[idx_bag], temp_centers,
                                                         temp_labels[temp_tr_ins_idx[idx_bag]:
                                                                     temp_tr_ins_idx[idx_bag + 1]])
            temp_te_idx = self.te_idx[loop_k]
            temp_num_te = len(temp_te_idx)
            ret_te_vec = np.zeros((temp_num_te, self.k_m * self.dimensions))
            for idx_bag in range(temp_num_te):
                temp_labels = []
                for ins in self.bags[temp_te_idx[idx_bag], 0][:, :self.dimensions]:
                    temp_dis = []
                    for center in temp_centers:
                        temp_dis.append(dis_euclidean(ins, center))
                    temp_sorted_dis_idx = np.argsort(temp_dis)
                    temp_labels.append(temp_sorted_dis_idx[0])
                ret_te_vec[idx_bag] = self.__bag_mapping(temp_te_idx[idx_bag], temp_centers, temp_labels)
            yield ret_tr_vec, self.bags_label[temp_tr_idx], ret_te_vec, self.bags_label[temp_te_idx], None
