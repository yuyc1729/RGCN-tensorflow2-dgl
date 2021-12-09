import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import eigsh
import dgl
import tensorflow as tf
from sklearn.cluster import KMeans
from labelled_loader import GraphConstruct
import matplotlib.pyplot as plt
from pathlib import Path


class DataLoader2(GraphConstruct):

    def __init__(self, path_data):

        # read raw data from txt
        self.data_name = Path(path_data).stem
        self.data = self._read_raw_txt(path_data)
        if self.data.shape[1] > 2:
            self.data = self.data[:, 0:2]
        self.n_list = np.unique(self.data)
        node_list = np.arange(0, len(self.n_list))
        node_map = {self.n_list[i]: node_list[i] for i in range(len(self.n_list))}
        self.data[:, 0] = np.array(list(map(node_map.get, self.data[:, 0])))
        self.data[:, 1] = np.array(list(map(node_map.get, self.data[:, 1])))
        self.n_list = node_list

        g = dgl.graph((self.data[:, 0], self.data[:, 1]))
        self.num_nodes = len(self.n_list)
        self.data = np.insert(self.data, 1, 0, axis=1)

        # get train, valid and test data
        train_size = int(len(self.data) * 0.8)
        val_size = int(len(self.data) * 0.1)
        test_size = len(self.data) - train_size - val_size
        with tf.device('/cpu:0'):
            self.test, test_eids = self._sample_graph(g, sample_edge_size=test_size)
            g.remove_edges(test_eids)
            self.valid, valid_eids = self._sample_graph(g, sample_edge_size=val_size)
            g.remove_edges(valid_eids)
        self.train = np.concatenate([tf.expand_dims(g.edges()[0], axis=1).numpy(),
                                     tf.expand_dims(g.edges()[1], axis=1).numpy()], axis=1)

        # insert edges to form triplets
        self.train = np.insert(self.train, 1, 0, axis=1)
        self.valid = np.insert(self.valid, 1, 0, axis=1)
        self.test = np.insert(self.test, 1, 0, axis=1)
        self.train_nodes = np.unique(np.concatenate([self.train[:, 0], self.train[:, 2]]))
        self.val_nodes = np.unique(np.concatenate([self.valid[:, 0], self.valid[:, 2]]))
        self.test_nodes = np.unique(np.concatenate([self.test[:, 0], self.test[:, 2]]))

        # set self._node_feat as None if applying spectral clustering to classify the
        # nodes before training
        self._node_feat = np.eye(self.num_nodes)
        # self._node_feat = None
        if self._node_feat is None:
            # do the spectral clustering if not having done it yet
            self.result_dir = Path('./sc_results')
            self.sc_pic = self.result_dir / f'{self.data_name}_sc.png'
            self.feat_labels = self.result_dir / f'{self.data_name}_sc.npy'
            if not self.result_dir.exists():
                Path.mkdir(self.result_dir)
            if not self.sc_pic.exists():
                self.sc = self.spectral_clustering()
        self.num_rels = 1

        super(DataLoader2, self).__init__(triplets=self.data, num_nodes=self.num_nodes)

        return

    def _sample_graph(self, g, sample_edge_size):
        sampled_triplet = 0
        sampled_eids = 0
        k = 10
        while True:
            sampled_nodes = np.random.choice(np.arange(self.num_nodes), k)
            test_graph = dgl.sampling.sample_neighbors(g, nodes=sampled_nodes, fanout=2)
            if test_graph.num_edges() < sample_edge_size:
                k += 1
                continue
            else:
                sampled_triplet = np.concatenate(
                    [tf.expand_dims(test_graph.edges()[0], axis=1).numpy(),
                     tf.expand_dims(test_graph.edges()[1], axis=1).numpy()],
                    axis=1)
                sampled_eids = test_graph.edata[dgl.EID]
                break
        return sampled_triplet, sampled_eids

    @staticmethod
    def _read_raw_txt(raw_path):
        return np.loadtxt(raw_path, dtype=float).astype(int)

    def spectral_clustering(self, plot=True):
        adj_mat = self.get_adj(data=self.train)
        degrees = np.sum(adj_mat.toarray(), axis=1)
        deg_mat = sp.diags([degrees], [0])
        laplace = deg_mat - adj_mat

        # normalize
        degrees_sqrt = degrees ** (-0.5)
        degrees_sqrt[np.isinf(degrees_sqrt)] = 0.0
        degrees_sqrt = sp.diags(degrees_sqrt.flatten())
        laplace = degrees_sqrt.dot(laplace.dot(degrees_sqrt))

        # compute eigenvectors
        evals_large, evecs_large = eigsh(laplace, 3, which='LM')

        # k-means clustering
        sp_dict = {'class_rate': [], 'labels': [], 'inertia': []}
        for class_rate in [0.05, 0.1, 0.3, 0.5, 0.7, 0.9]:
            num_classes = int(evecs_large.shape[0] * class_rate)
            sp_kmeans = KMeans(n_clusters=num_classes)
            sp_kmeans.fit(evecs_large)
            sp_dict['class_rate'].append(num_classes)
            sp_dict['labels'].append(sp_kmeans.labels_)
            sp_dict['inertia'].append(sp_kmeans.inertia_)
        np.save(str(self.feat_labels), sp_dict['labels'][3])

        # plot to see the descending trend of inertia as the number of clusters increasing
        if plot:
            if not self.result_dir.exists():
                Path.mkdir(self.result_dir)
            fig, ax = plt.subplots()
            plt.xlabel(xlabel='Number of Clusters')
            plt.ylabel(ylabel='Sum of Squared Distances')
            plt.plot(sp_dict['class_rate'], sp_dict['inertia'])
            plt.show()
            fig.savefig(self.sc_pic)
        return sp_dict

    @property
    def node_feat(self):
        return self._node_feat

    @node_feat.getter
    def node_feat(self):
        if self._node_feat is None:
            if not self.feat_labels.exists():
                raise Exception('We should get the label of each node using '
                                'spectral clustering before getting the node features.')
            clustering_labels = np.load(str(self.feat_labels)).astype(int)
            self._node_feat = node_classification(clustering_labels, dataset=self)
        return self._node_feat

    def __call__(self, *args, **kwargs):
        return self.graph_split(train_trip=self.train,
                                valid_trip=self.valid,
                                test_trip=self.test)


def node_classification(clustering_labels, dataset):
    outside_nodes = np.setdiff1d(np.union1d(dataset.val_nodes, dataset.test_nodes),
                                 dataset.train_nodes)
    num_clustering_feat = np.max(clustering_labels) + 1
    num_feat = num_clustering_feat + len(outside_nodes)
    num_nodes = dataset.num_nodes

    data = np.ones(shape=(num_nodes,))
    row = np.arange(0, num_nodes)
    one_hot_node_feat = sp.coo_matrix(arg1=(data, (row, clustering_labels)),
                                      shape=(num_nodes, num_feat),
                                      dtype=float)
    one_hot_node_feat = tf.convert_to_tensor(one_hot_node_feat.toarray())
    if len(outside_nodes) != 0:
        outside_feat = np.eye(len(outside_nodes))
        outside_feat = np.concatenate([np.zeros(shape=(len(outside_nodes),
                                                       num_clustering_feat)),
                                       outside_feat], axis=1)
        one_hot_node_feat = tf.tensor_scatter_nd_update(
            tensor=one_hot_node_feat,
            indices=tf.expand_dims(outside_nodes, axis=1),
            updates=tf.convert_to_tensor(outside_feat)
        )
    return one_hot_node_feat


if __name__ == '__main__':
    dl = DataLoader2(path_data='../Data/Unlabeled/USAir.txt')
    num_rels = dl.num_rels
    prime_node_feat = dl.node_feat
    tr_g, val_g, te_g = dl()
    tr_node_feat = tf.gather(prime_node_feat, indices=tr_g.ndata[dgl.NID])
    val_node_feat = tf.gather(prime_node_feat, indices=val_g.ndata[dgl.NID])
    te_node_feat = tf.gather(prime_node_feat, indices=te_g.ndata[dgl.NID])
