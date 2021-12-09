# import os
#
# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import tensorflow as tf
from labelled_loader import DataLoader
from unlabelled_loader import DataLoader2, node_classification
from rgcn_model import LinkPrediction
from mpnn_model import LinkPredictionMPNN
import time
import numpy as np
import dgl
from sklearn.cluster import KMeans
from sklearn.cluster import SpectralClustering
import matplotlib.pyplot as plt
import pickle

"""
https://docs.dgl.ai/en/0.6.x/guide/training-link.html
"""

model_type = 'mpnn'


def sample_graph(g, sample_edge_size):
    sampled_eids = 0
    k = 100
    while True:
        sampled_nodes = np.random.choice(np.arange(g.num_nodes()), k)
        test_graph = dgl.sampling.sample_neighbors(g, nodes=sampled_nodes, fanout=2)
        if test_graph.num_edges() < sample_edge_size:
            k += 100
            continue
        else:
            sampled_eids = test_graph.edata[dgl.EID]
            test_graph = dgl.edge_subgraph(g, edges=sampled_eids)
            break
    try:
        test_graph = test_graph.to('/gpu:0')
    except dgl._ffi.base.DGLError:
        pass
    return test_graph, sampled_eids, test_graph.ndata[dgl.NID]


def compute_loss(pos_score, neg_score):
    # Margin loss
    n_edges = pos_score.shape[0]
    loss = 1 - tf.expand_dims(pos_score, axis=1) + tf.reshape(neg_score, (n_edges, -1))
    loss = tf.where(loss < 0.0, 0.0, loss)
    return tf.reduce_mean(loss)


def k_means(node_feat, plot, data_name):
    # k-means clustering
    sp_dict = {'class_rate': [], 'labels': [], 'inertia': []}
    for class_rate in [0.05, 0.1, 0.3, 0.5, 0.7, 0.9]:
        num_classes = int(node_feat.shape[0] * class_rate)
        sp_kmeans = KMeans(n_clusters=num_classes)
        sp_kmeans.fit(node_feat)
        sp_dict['class_rate'].append(num_classes)
        sp_dict['labels'].append(sp_kmeans.labels_)
        sp_dict['inertia'].append(sp_kmeans.inertia_)

    # plot to see the descending trend of inertia as the number of clusters increasing
    if plot:
        fig, ax = plt.subplots()
        plt.xlabel(xlabel='Number of Clusters')
        plt.ylabel(ylabel='Sum of Squared Distances')
        plt.plot(sp_dict['class_rate'], sp_dict['inertia'])
        plt.show()
        fig.savefig(f'{data_name}_km.png')

    with open(data_name + '.pkl', 'wb') as f:
        pickle.dump(sp_dict, f, pickle.HIGHEST_PROTOCOL)

    return sp_dict


def spectral_clustering(node_feat, data_name):
    # k-means clustering
    sp_dict = {'class_rate': [], 'labels': []}
    for class_rate in [0.5]:
        num_classes = int(node_feat.shape[0] * class_rate)
        sp_sc = SpectralClustering(n_clusters=num_classes,
                                   n_components=50)
        sp_sc.fit(node_feat)
        sp_dict['class_rate'].append(num_classes)
        sp_dict['labels'].append(sp_sc.labels_)
    with open(data_name + 'sc.pkl', 'wb') as f:
        pickle.dump(sp_dict, f, pickle.HIGHEST_PROTOCOL)
    return sp_dict


def main(dl, node_feat=None, update=False):
    num_rels = dl.num_rels
    tr_g, val_g, te_g = dl()
    prime_node_feat = dl.node_feat
    if node_feat is not None:
        prime_node_feat = node_feat
    prime_tr_node_feat = tf.gather(prime_node_feat, indices=tr_g.ndata[dgl.NID])
    prime_val_node_feat = tf.gather(prime_node_feat, indices=val_g.ndata[dgl.NID])
    prime_te_node_feat = tf.gather(prime_node_feat, indices=te_g.ndata[dgl.NID])

    # train
    print('Initialize Models')
    model_name = f'{model_type}-{dl.data_name}-2'
    h_dim = 200
    node_embed = tf.zeros(shape=(dl.num_nodes, h_dim))
    if model_type == 'rgcn':
        model = LinkPrediction(h_dim=h_dim, num_conv=2,
                               num_rels=2 * num_rels,
                               num_bases=2, activation='relu',
                               reg_coe=0.01, negative_rate=3)
    elif model_type == 'mpnn':
        model = LinkPredictionMPNN(h_dim=h_dim, num_conv=3,
                                   activation='relu',
                                   reg_coe=0.01, negative_rate=3)
    else:
        raise Exception('We only support RGCN and MPNN.')

    print('Start to Train')
    # preparation
    epochs = 6000
    lr = 0.001
    optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
    graph_batch_size = 1000
    val_freq = 50
    eval_batch_size = 100
    train_time = []
    loss_tracker_tr = []
    loss_tracker_val = []
    loss_tracker_te = []
    best_loss = 999.0
    patience = 30
    p = 0

    try:
        tr_g_sam = tr_g.to('/gpu:0')
    except dgl._ffi.base.DGLError:
        tr_g_sam = tr_g
    tr_node_feat = prime_tr_node_feat
    for epoch in range(epochs):
        # if not update:
        #     tr_g_sam, tr_g_e_ids, tr_g_n_ids = sample_graph(g=tr_g, sample_edge_size=graph_batch_size)
        #     tr_node_feat = tf.gather(prime_tr_node_feat, indices=tr_g_n_ids)
        t0 = time.process_time()
        with tf.GradientTape() as tape:
            pos_score, neg_score = model((tr_g_sam, tr_node_feat), training=True)
            loss = compute_loss(pos_score=pos_score, neg_score=neg_score)
        grads = tape.gradient(loss, model.trainable_weights)
        optimizer.apply_gradients(zip(grads, model.trainable_weights))
        loss_tracker_tr.append(tf.reduce_mean(loss).numpy())
        t1 = time.process_time()
        train_time.append(t1 - t0)

        print("Training: Epoch {:04d} | Loss {:.4f} ({:.4f}) | Time {:.4f}s ({:.4f})".
              format(epoch + 1, loss_tracker_tr[-1], np.mean(loss_tracker_tr),
                     train_time[-1], np.mean(train_time)))
        with open(file=f'{model_name}.txt', mode='a+') as log:
            log.writelines("Training: Epoch {:04d} | Loss {:.4f} ({:.4f}) | Time {:.4f}s ({:.4f})\n".
                           format(epoch + 1, loss_tracker_tr[-1], np.mean(loss_tracker_tr),
                                  train_time[-1], np.mean(train_time)))

        # validation
        if epoch % val_freq == 0:
            val_times = int(val_g.num_edges() / eval_batch_size)
            val_times = val_times if val_times > 30 else 30
            try:
                val_g_sam = val_g.to('/gpu:0')
            except dgl._ffi.base.DGLError:
                val_g_sam = val_g
            val_node_feat = prime_val_node_feat
            for val_epoch in range(val_times):
                # if not update:
                #     val_g_sam, val_e_ids, val_n_ids = sample_graph(val_g, sample_edge_size=eval_batch_size)
                #     val_node_feat = tf.gather(prime_val_node_feat, indices=val_n_ids)
                pos_score_val, neg_score_val = model((val_g_sam, val_node_feat), training=False)
                val_metric = compute_loss(pos_score=pos_score_val, neg_score=neg_score_val)
                loss_tracker_val.append(tf.reduce_mean(val_metric).numpy())

                print("Validating: Val_epoch {:04d} | Loss {:.4f} ({:.4f})".
                      format(val_epoch + 1, loss_tracker_val[-1], np.mean(loss_tracker_val)))
                with open(file=f'{model_name}.txt', mode='a+') as log:
                    log.writelines("Validating: Val_epoch {:04d} | Loss {:.4f} ({:.4f})\n".
                                   format(val_epoch + 1, loss_tracker_val[-1], np.mean(loss_tracker_val)))

            # early stop
            if best_loss > np.mean(loss_tracker_val):
                best_loss = np.mean(loss_tracker_val)
                p = 0
                model.save_weights(f'{model_name}/{model_name}.ckpt')
                if update:
                    n_ids = tr_g_sam.ndata[dgl.NID]
                    n_val_ids = val_g_sam.ndata[dgl.NID]
                    node_embed = tf.tensor_scatter_nd_update(
                        tensor=node_embed,
                        indices=tf.expand_dims(n_ids, axis=1),
                        updates=model.mpnn((tr_g_sam, tr_node_feat), training=False)
                    )
                    node_embed = tf.tensor_scatter_nd_update(
                        tensor=node_embed,
                        indices=tf.expand_dims(n_val_ids, axis=1),
                        updates=model.mpnn((val_g_sam, val_node_feat), training=False)
                    )
            else:
                p += 1
            if p >= patience:
                break

            loss_tracker_tr = []
            loss_tracker_val = []

    print("training done")
    print("\nstart testing:")
    test_times = int(te_g.num_edges() / eval_batch_size)
    test_times = test_times if test_times > 30 else 30
    try:
        te_g_sam = te_g.to('/gpu:0')
    except dgl._ffi.base.DGLError:
        te_g_sam = te_g
    te_node_feat = prime_te_node_feat
    for test_epoch in range(test_times):
        # if not update:
        #     te_g_sam, te_e_ids, te_n_ids = sample_graph(te_g, sample_edge_size=eval_batch_size)
        #     te_node_feat = tf.gather(prime_te_node_feat, indices=te_n_ids)
        model.load_weights(f'{model_name}/{model_name}.ckpt')
        pos_score_te, neg_score_te = model((te_g_sam, te_node_feat), training=False)
        te_metric = compute_loss(pos_score=pos_score_te, neg_score=neg_score_te)
        loss_tracker_te.append(tf.reduce_mean(te_metric).numpy())

        print("Testing: Test_epoch {:04d} | Loss {:.4f} ({:.4f})".
              format(test_epoch + 1, loss_tracker_te[-1], np.mean(loss_tracker_te)))
        with open(file=f'{model_name}.txt', mode='a+') as log:
            log.writelines("Testing: Test_epoch {:04d} | Loss {:.4f} ({:.4f})\n".
                           format(test_epoch + 1, loss_tracker_te[-1], np.mean(loss_tracker_te)))

        if update:
            n_te_ids = te_g_sam.ndata[dgl.NID]
            node_embed = tf.tensor_scatter_nd_update(
                tensor=node_embed,
                indices=tf.expand_dims(n_te_ids, axis=1),
                updates=model.mpnn((te_g_sam, te_node_feat), training=False)
            )

    return node_embed, model_name


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    # load data
    print('Load Data')
    # dl = DataLoader(path_raw_ent='../Data/Labeled/FB15k-237/entities.dict',
    #                 path_raw_rel='../Data/Labeled/FB15k-237/relations.dict',
    #                 path_raw_tr='../Data/Labeled/FB15k-237/train.txt',
    #                 path_raw_val='../Data/Labeled/FB15k-237/valid.txt',
    #                 path_raw_te='../Data/Labeled/FB15k-237/test.txt')
    dl = DataLoader2(path_data='../Data/Unlabeled/NS.txt')
    tr_, val_, te_ = dl()
    node_e, model_name = main(dl=dl, update=True)
    # # k-means clustering
    clustering_labels = k_means(node_feat=node_e, plot=True, data_name=model_name)
    clustering_labels = clustering_labels['labels'][3]
    # spectral clustering
    # clustering_labels = spectral_clustering(node_feat=node_e, data_name=model_name)
    # clustering_labels = clustering_labels['labels'][0]
    node_f = node_classification(clustering_labels=clustering_labels, dataset=dl)
    main(dl=dl, node_feat=node_f, update=False)
