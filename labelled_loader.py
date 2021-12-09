import numpy as np
import scipy.sparse as sp
import dgl
import tensorflow as tf
from pathlib import Path


class GraphConstruct:

    def __init__(self, triplets, num_nodes):
        self.data = triplets
        self.num_nodes = num_nodes
        return

    @staticmethod
    def normalize(adj):
        degree = np.array(adj.sum(1)) ** (-0.5)
        degree[np.isinf(degree)] = 0.0
        degree = sp.diags(degree.flatten())
        return degree.dot(adj.dot(degree))

    @staticmethod
    def row_normalize(mx):
        """ Row-normalize sparse matrix
            https://github.com/tkipf/pygcn/blob/master/pygcn/utils.py"""
        rowsum = np.array(mx.sum(1))
        r_inv = np.power(rowsum, -1).flatten()
        r_inv[np.isinf(r_inv)] = 0.
        r_mat_inv = sp.diags(r_inv)
        mx = r_mat_inv.dot(mx)
        return mx

    def get_adj(self, data=None):
        if data is None:
            data = self.data
        data_ = np.ones(len(data))
        row = data[:, 0]
        column = data[:, 2]
        adj = sp.coo_matrix(arg1=(data_, (row, column)),
                            shape=(self.num_nodes, self.num_nodes),
                            dtype=float)
        # adj = adj + adj.T - sp.diags([adj.diagonal()], [0])
        # adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
        # adj = self.row_normalize(adj)
        return adj

    def build_graph(self):
        graph = dgl.from_scipy(self.get_adj())
        graph.add_edges(self.data[:, 2], self.data[:, 0])
        # reverse_data = np.concatenate([np.expand_dims(self.data[:, 2], axis=1),
        #                                np.expand_dims(self.data[:, 1], axis=1),
        #                                np.expand_dims(self.data[:, 0], axis=1)],
        #                               axis=1)
        # double_data = np.concatenate([self.data, reverse_data], axis=0)
        # ind = np.lexsort([self.data[:, 2], self.data[:, 0]])
        # sorted_edge = self.data[ind][:, 1]
        sorted_edge = np.concatenate([self.data[:, 1],
                                      self.data[:, 1] + np.max(self.data[:, 1])])
        graph.edata['edge_type'] = tf.cast(tf.constant(sorted_edge), tf.int32)
        return graph

    def build_heterograph(self):
        data_dict = {
            ('node', e, 'node'):
                (tf.convert_to_tensor(self.data[self.data[:, 1] == e][:, 0]),
                 tf.convert_to_tensor(self.data[self.data[:, 1] == e][:, 2]))
            for e in range(dl.num_rels)}
        return dgl.heterograph(data_dict)

    def graph_split(self, train_trip, valid_trip, test_trip):
        def split(g, src, dst):
            bi_src = np.concatenate([src, dst])
            bi_dst = np.concatenate([dst, src])
            return dgl.edge_subgraph(g, edges=g.edge_ids(bi_src, bi_dst))

        graph = self.build_graph()
        graph = graph.to('/cpu:0')
        train_graph = split(graph, train_trip[:, 0], train_trip[:, 2])
        val_graph = split(graph, valid_trip[:, 0], valid_trip[:, 2])
        test_graph = split(graph, test_trip[:, 0], test_trip[:, 2])
        return train_graph.to('/cpu:0'), val_graph.to('/cpu:0'), test_graph.to('/cpu:0')


class DataLoader(GraphConstruct):

    def __init__(self,
                 path_raw_ent,
                 path_raw_rel,
                 path_raw_tr,
                 path_raw_val,
                 path_raw_te):
        # read raw data from txt
        self.n_list = self._read_raw_txt(path_raw_ent)
        self.e_list = self._read_raw_txt(path_raw_rel)
        train_list = self._read_raw_txt(path_raw_tr)
        val_list = self._read_raw_txt(path_raw_val)
        test_list = self._read_raw_txt(path_raw_te)
        self.data_name = Path(path_raw_tr).parent.name

        # map nodes and edges to build graphs
        self.train = self._dataset_map(train_list)
        self.valid = self._dataset_map(val_list)
        self.test = self._dataset_map(test_list)
        self.data = np.concatenate([self.train, self.valid, self.test], axis=0)
        self.num_nodes = len(self.n_list)
        self.num_rels = len(self.e_list)
        super(DataLoader, self).__init__(triplets=self.data, num_nodes=self.num_nodes)

        # get one-hot features
        self.node_feat, self.edge_feat = self._feature_map()

        # get labels
        label_map = {i: self.edge_feat[i] for i in range(len(self.edge_feat))}
        self.y_train = list(map(label_map.get, self.train[:, 1]))
        self.y_val = list(map(label_map.get, self.valid[:, 1]))
        self.y_test = list(map(label_map.get, self.test[:, 1]))

        return

    @staticmethod
    def _read_raw_txt(raw_path):
        return np.loadtxt(raw_path, delimiter='\t', dtype=str)

    @staticmethod
    def _n_or_e_map(n_or_e_list, map_list):
        map_list = dict(map_list)
        one_hot_map = dict(zip(map_list.values(), map_list.keys()))
        return list(map(one_hot_map.get, n_or_e_list))

    def _dataset_map(self, dataset):
        dataset[:, 0] = self._n_or_e_map(dataset[:, 0], self.n_list)
        dataset[:, 1] = self._n_or_e_map(dataset[:, 1], self.e_list)
        dataset[:, 2] = self._n_or_e_map(dataset[:, 2], self.n_list)
        return dataset.astype(int)

    def _feature_map(self):
        return np.eye(len(self.n_list)), np.eye(len(self.e_list))

    def __call__(self, *args, **kwargs):
        return self.graph_split(train_trip=self.train,
                                valid_trip=self.valid,
                                test_trip=self.test)


if __name__ == '__main__':
    dl = DataLoader(path_raw_ent='../Data/Labeled/FB15k-237/entities.dict',
                    path_raw_rel='../Data/Labeled/FB15k-237/relations.dict',
                    path_raw_tr='../Data/Labeled/FB15k-237/train.txt',
                    path_raw_val='../Data/Labeled/FB15k-237/valid.txt',
                    path_raw_te='../Data/Labeled/FB15k-237/test.txt')

    dl2 = DataLoader(path_raw_ent='../Data/Labeled/wn18/entities.dict',
                     path_raw_rel='../Data/Labeled/wn18/relations.dict',
                     path_raw_tr='../Data/Labeled/wn18/train.txt',
                     path_raw_val='../Data/Labeled/wn18/valid.txt',
                     path_raw_te='../Data/Labeled/wn18/test.txt')

    # g_ = dl.build_graph()
    # gg_ = dgl.data.FB15k237Dataset()[0]
    tr_g, val_g, te_g = dl()
