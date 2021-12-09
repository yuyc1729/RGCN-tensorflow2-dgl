import dgl
from dgl.nn.tensorflow.conv import RelGraphConv
import tensorflow as tf
from tensorflow.keras import layers


def compute_norm(g):
    """ Edge norm is the number of predecessors of each edge type. """
    e_ids = tf.linspace(0, g.num_edges() - 1, g.num_edges())
    e_ids = tf.cast(e_ids, tf.int64)
    g.ndata['norm'] = tf.zeros(g.num_nodes())
    for e in tf.unique(g.edata['edge_type'])[0]:
        # get e_ids for the edge type e
        e_type = g.find_edges(e_ids[g.edata['edge_type'] == e])
        # the number of edges incident to each node for type e
        dst_node, dst_node_pos = tf.unique(e_type[1])
        num_e_t_node = tf.math.bincount(dst_node_pos)
        # update the node norm for edge type e
        node_norm = 1.0 / tf.cast(num_e_t_node, tf.float32)
        g.ndata['norm'] = tf.tensor_scatter_nd_update(
            tensor=g.ndata['norm'],
            indices=tf.expand_dims(dst_node, axis=1),
            updates=node_norm)
    g.apply_edges(lambda edges: {'norm': tf.expand_dims(edges.dst['norm'], axis=1)})
    g.ndata.pop('norm')
    return g.edata.pop('norm')


def construct_negative_graph(graph, node_embed, k, etype):
    edge_mask = graph.edata['edge_type'] == etype
    src, dst = graph.edges()[0][edge_mask], graph.edges()[1][edge_mask]
    selected_nodes = tf.unique(tf.concat([src, dst], axis=0))[0]
    selected_nodes = tf.sort(selected_nodes)
    neg_src = tf.repeat(src, k)
    neg_dst = tf.random.uniform(shape=(len(src) * k, ), minval=0, maxval=graph.num_nodes(), dtype=tf.int64)
    g = dgl.graph((neg_src, neg_dst))
    g.ndata['h'] = node_embed[:g.num_nodes(), ]
    # node_feat_neg = tf.zeros(shape=(g.num_nodes(), node_embed.shape[1]))
    # node_feat_neg = tf.tensor_scatter_nd_update(
    #     tensor=node_feat_neg,
    #     indices=tf.expand_dims(selected_nodes, axis=1),
    #     updates=tf.gather(node_embed, indices=selected_nodes)
    # )
    return g, graph.edge_ids(src, dst)


class RGCN(layers.Layer):
    def __init__(self, in_dim, out_dim, num_rel, num_bases,
                 kernel_init=tf.keras.initializers.glorot_uniform(),
                 bias_init=tf.keras.initializers.zeros(),
                 use_bias=False, low_mem=True):
        super(RGCN, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.num_rel = num_rel
        self.num_bases = num_bases
        self.use_bias = use_bias
        self.low_mem = low_mem

        # parameter matrices
        self.weight = tf.Variable(
            initial_value=kernel_init(shape=(num_bases, in_dim, out_dim)),
            trainable=True)
        if num_bases < num_rel:
            self.w_coe = tf.Variable(
                initial_value=kernel_init(shape=(num_rel, num_bases)),
                trainable=True)
        if use_bias:
            self.bias = tf.Variable(
                initial_value=bias_init(shape=(out_dim,)),
                trainable=True)
        self.self_loop = tf.Variable(
            initial_value=kernel_init(shape=(in_dim, out_dim)),
            trainable=True)
        return

    def rgcn_msg_func(self, edges):
        """
        based on https://docs.dgl.ai/en/0.7.x/_modules/dgl/nn/tensorflow/conv/relgraphconv.html#RelGraphConv
        """
        if self.num_bases < self.num_rel:
            weight = tf.einsum('ab, bcd -> acd', self.w_coe, self.weight)
        else:
            weight = self.weight  # weight.shape == (num_rel, in_dim, out_dim)

        if self.low_mem:
            msg_collection = tf.zeros(shape=(edges.src['h'].shape[0], self.out_dim))
            edge_types = tf.unique(edges.data['edge_type'])[0]
            for edge_type in edge_types:
                node_e_t_weight = weight[edge_type]
                node_e_t_mask = edges.data['edge_type'] == edge_type
                node_feat_e_t = tf.boolean_mask(edges.src['h'], node_e_t_mask)
                msg = tf.matmul(node_feat_e_t, node_e_t_weight)
                indices = tf.where(node_e_t_mask)
                msg_collection = tf.tensor_scatter_nd_update(
                    tensor=msg_collection,
                    indices=indices,
                    updates=msg)
        else:
            # gathering the weight matrices based on the order of the edge type of the graph
            node_e_t_weight = tf.gather(weight, edges.data['edge_type'])
            # node_e_t_weight.shape == (num_edges, in_dim, out_dim)
            # edges.src['h'].shape == (num_edges, in_dim)
            msg_collection = tf.einsum('ab, abc -> ac', edges.src['h'], node_e_t_weight)
        msg_collection = msg_collection * tf.cast(edges.data['norm'], tf.float32)
        return {'msg': msg_collection}

    def apply_node(self, nodes):
        h = nodes.data['msg']
        if self.use_bias:
            h = h + self.bias
        h = h + tf.matmul(nodes.data['h'], self.self_loop)
        return {'h': h}

    def call(self, inputs, *args, **kwargs):
        g, node_feat = inputs
        g.ndata['h'] = node_feat
        g.update_all(message_func=self.rgcn_msg_func,
                        reduce_func=dgl.function.sum('msg', 'msg'),
                        apply_node_func=self.apply_node)
        return g.ndata.pop('h')


class RGCNModel(tf.keras.Model):
    def __init__(self, h_dim, num_conv, num_rel, num_bases, activation):
        super(RGCNModel, self).__init__()
        self.num_conv = num_conv

        # the initial node features are embedded with one-hot encoding,
        # therefore the dim is num_nodes
        self.emb_layer = layers.Dense(units=h_dim)
        self.conv_layer = [RGCN(in_dim=h_dim,
                                out_dim=h_dim,
                                num_rel=num_rel,
                                num_bases=num_bases)
                           for _ in range(self.num_conv)]
        self.bn_layer = [layers.BatchNormalization()
                         for _ in range(self.num_conv)]
        self.act_layer = layers.Activation(activation=activation)
        self.drop_layer = [layers.Dropout(rate=0.3)
                           for _ in range(self.num_conv)]
        return

    def call(self, inputs, training=None, mask=None):
        g, node_embed = inputs
        node_embed = self.emb_layer(node_embed)
        g.edata['norm'] = compute_norm(g)
        for i in range(self.num_conv):
            node_embed = self.conv_layer[i]((g, node_embed))
            node_embed = self.bn_layer[i](node_embed, training=training)
            node_embed = self.act_layer(node_embed)
            node_embed = self.drop_layer[i](node_embed, training=training)
        return node_embed


class LinkPrediction(tf.keras.Model):
    def __init__(self,
                 h_dim,
                 num_conv,
                 num_rels,
                 num_bases,
                 activation,
                 reg_coe,
                 negative_rate):
        super(LinkPrediction, self).__init__()
        self.rgcn = RGCNModel(h_dim, num_conv,
                              num_rels, num_bases, activation)
        """ In DistMult, each edge type has a corresponding diagonal matrix for
            computing the score. """
        self.w_relation = tf.Variable(
            initial_value=tf.keras.initializers.glorot_uniform()(shape=(num_rels, h_dim)),
            trainable=True)
        self.num_rels = num_rels
        self.negative_rate = negative_rate
        self.reg_coe = reg_coe
        return

    def call(self, inputs, training=None, mask=None):
        graph, node_embed = inputs
        node_embed = self.rgcn((graph, node_embed), training=training)
        graph.ndata['h'] = node_embed
        neg_scores = tf.zeros(shape=(int(graph.num_edges() * self.negative_rate),))
        pos_scores = tf.zeros(shape=(graph.num_edges(), ))

        for edge_type in range(self.num_rels):
            w_relation = self.w_relation[edge_type]
            w_relation = tf.expand_dims(w_relation, axis=0)
            neg_graph, pos_e_ids = construct_negative_graph(graph=graph,
                                                            node_embed=node_embed,
                                                            k=self.negative_rate,
                                                            etype=edge_type)

            # update the positive graph
            e_t_mask = graph.edata['edge_type'] == edge_type
            e_t_ids = graph.edge_ids(u=graph.edges()[0][e_t_mask],
                                     v=graph.edges()[1][e_t_mask])
            graph.apply_edges(func=dgl.function.u_mul_v('h', 'h', 'hh'), edges=e_t_ids)
            pos_scores = tf.tensor_scatter_nd_update(
                tensor=pos_scores,
                indices=tf.expand_dims(e_t_ids, axis=1),
                updates=tf.reduce_sum(
                    tf.gather(graph.edata['hh'], indices=e_t_ids) *
                    tf.tile(w_relation, [len(e_t_ids), 1]), axis=1)
            )

            # update negative graphs
            neg_graph.apply_edges(func=dgl.function.u_mul_v('h', 'h', 'hh'))
            neg_graph.edata['score'] = tf.reduce_sum(
                neg_graph.edata['hh'] *
                tf.tile(w_relation, [neg_graph.num_edges(), 1]), axis=1)

            # update negative edges scores in corresponding positions
            pos_e_ids_neg = pos_e_ids * self.negative_rate
            neg_e_ids = tf.repeat(pos_e_ids_neg, self.negative_rate)
            for i in range(self.negative_rate - 1):
                indices = tf.linspace(i + 1,
                                      i + 1 + self.negative_rate * (len(pos_e_ids_neg) - 1),
                                      len(pos_e_ids_neg))
                indices = tf.cast(indices, tf.int32)
                updates = pos_e_ids_neg + i + 1
                neg_e_ids = tf.tensor_scatter_nd_update(tensor=neg_e_ids,
                                                        indices=tf.expand_dims(indices, axis=1),
                                                        updates=updates)
            neg_scores = tf.tensor_scatter_nd_update(tensor=neg_scores,
                                                     indices=tf.expand_dims(neg_e_ids, axis=1),
                                                     updates=neg_graph.edata['score'])
        return pos_scores, neg_scores
