import dgl
import tensorflow as tf
from tensorflow.keras import layers


def construct_negative_graph(graph, node_embed, k):
    src, dst = graph.edges()[0], graph.edges()[1]
    neg_src = tf.repeat(src, k)
    neg_dst = tf.random.uniform(shape=(len(src) * k,), minval=0, maxval=graph.num_nodes(), dtype=tf.int64)
    g = dgl.graph((neg_src, neg_dst))
    g.ndata['h'] = node_embed[:g.num_nodes(), ]
    # node_feat_neg = tf.zeros(shape=(g.num_nodes(), node_embed.shape[1]))
    # node_feat_neg = tf.tensor_scatter_nd_update(
    #     tensor=node_feat_neg,
    #     indices=tf.expand_dims(selected_nodes, axis=1),
    #     updates=tf.gather(node_embed, indices=selected_nodes)
    # )
    return g, graph.edge_ids(src, dst)


class MPNN(layers.Layer):
    def __init__(self, h_dim, use_bias=True,
                 kernel_init=tf.keras.initializers.glorot_uniform(),
                 bias_init=tf.keras.initializers.zeros()):
        super(MPNN, self).__init__()
        self.use_bias = use_bias
        self.w = tf.Variable(initial_value=kernel_init(shape=(2 * h_dim, h_dim)),
                             trainable=True)
        if use_bias:
            self.b = tf.Variable(initial_value=bias_init(shape=(h_dim,)),
                                 trainable=True)
        return

    @staticmethod
    def msg_func(edges):
        return {'msg': tf.concat([edges.src['h'], edges.dst['h']], axis=-1)}

    @staticmethod
    def reduce_func(nodes):
        return {'msg': tf.reduce_sum(nodes.mailbox['msg'], axis=1)}

    @staticmethod
    def apply_nodes(nodes):
        return {'h': tf.nn.sigmoid(nodes.data['msg']) + tf.nn.softplus(nodes.data['h'])}

    def call(self, inputs, *args, **kwargs):
        graph, node_feat = inputs
        graph.ndata['h'] = node_feat
        graph.apply_edges(func=self.msg_func)
        edge_feat = tf.matmul(graph.edata['msg'], self.w)
        if self.use_bias:
            edge_feat += self.b
        graph.edata['msg'] = edge_feat
        graph.update_all(message_func=dgl.function.copy_edge('msg', 'msg'),
                         reduce_func=self.reduce_func,
                         apply_node_func=self.apply_nodes)
        return graph.ndata.pop('h')


class MPNNModel(tf.keras.Model):
    def __init__(self, h_dim, num_conv, use_bias=True, activation='relu'):
        super(MPNNModel, self).__init__()
        self.num_conv = num_conv
        self.emb_layer = layers.Dense(h_dim)
        self.conv_layer = [MPNN(h_dim=h_dim, use_bias=use_bias)
                           for _ in range(num_conv)]
        self.bn_layer = [layers.BatchNormalization()
                         for _ in range(self.num_conv)]
        self.act_layer = layers.Activation(activation=activation)
        self.drop_layer = [layers.Dropout(rate=0.3)
                           for _ in range(self.num_conv)]

    def call(self, inputs, training=None, mask=None):
        g, node_embed = inputs
        node_embed = self.emb_layer(node_embed)
        for i in range(self.num_conv):
            node_embed = self.conv_layer[i]((g, node_embed))
            node_embed = self.bn_layer[i](node_embed, training=training)
            node_embed = self.act_layer(node_embed)
            node_embed = self.drop_layer[i](node_embed, training=training)
        return node_embed


class LinkPredictionMPNN(tf.keras.Model):
    def __init__(self,
                 h_dim,
                 num_conv,
                 activation,
                 reg_coe,
                 negative_rate):
        super(LinkPredictionMPNN, self).__init__()
        self.mpnn = MPNNModel(h_dim, num_conv, activation)
        self.h_dim = h_dim
        self.w_relation = tf.Variable(
            initial_value=tf.keras.initializers.glorot_uniform()(shape=(1, h_dim)),
            trainable=True)
        self.negative_rate = negative_rate
        self.reg_coe = reg_coe
        return

    def call(self, inputs, training=None, mask=None):
        graph, node_embed = inputs
        node_embed = self.mpnn((graph, node_embed), training=training)
        graph.ndata['h'] = node_embed
        neg_scores = tf.zeros(shape=(int(graph.num_edges() * self.negative_rate),))
        pos_scores = tf.zeros(shape=(graph.num_edges(),))
        neg_graph, pos_e_ids = construct_negative_graph(graph=graph,
                                                        node_embed=node_embed,
                                                        k=self.negative_rate)

        # update the positive graph
        graph.apply_edges(func=dgl.function.u_mul_v('h', 'h', 'hh'))
        pos_scores = tf.reduce_sum(
            graph.edata['hh'] *
            tf.tile(self.w_relation, [graph.num_edges(), 1]), axis=1)

        # update negative graphs
        neg_graph.apply_edges(func=dgl.function.u_mul_v('h', 'h', 'hh'))
        neg_graph.edata['score'] = tf.reduce_sum(
            neg_graph.edata['hh'] *
            tf.tile(self.w_relation, [neg_graph.num_edges(), 1]), axis=1)

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
