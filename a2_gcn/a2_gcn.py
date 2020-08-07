'''
Created on Oct 10, 2018
Tensorflow Implementation of An Attribute-aware Attentive GCN (a2-gcn) model in:
Liu Fan et al. A2-GCN: An Attribute-aware Attentive GCNModel for Recommendation. In TKDE.

@author: Liu Fan (liufancs@gmail.com)
'''
import tensorflow as tf
from utility.helper import *
from utility.batch_test import *

class a2_gcn(object):
    def __init__(self, data_config, pretrain_data):
        # argument settings
        self.adj_type = args.adj_type
        self.pretrain_data = pretrain_data
        self.n_users = data_config['n_users']
        self.n_items = data_config['n_items']
        self.n_tags = data_config['n_tags']
        self.n_relations = args.n_relations
        self.train_items = data_config['train_items']
        self.weight_size = eval(args.layer_size)
        self.n_layers = len(self.weight_size)
        self.n_fold = 100
        self.norm_adj = data_config['norm_adj']
        self.n_nonzero_elems = self.norm_adj.count_nonzero()
        self.lr = args.lr
        self.emb_dim = args.embed_size
        self.batch_size = data_config['batch_size']
        self.regs = eval(args.regs)
        self.decay = self.regs[0]
        self.verbose = args.verbose

        self.all_u_list = data_config['all_u_list']
        self.all_t_list = data_config['all_t_list']
        self.all_v_list = data_config['all_v_list']
        self.all_r_list = data_config['all_r_list']
        self.all_tag_list = data_config['all_tag_list']
        self.mask_tag = data_config['mask_tag']
        self.mask_tags = data_config['mask_tags']
        self.A_in = data_config['norm_adj']
        self.A_values = tf.placeholder(tf.float32, shape=[len(self.all_v_list)])
        '''
        *********************************************************
        Create Placeholder for Input Data & Dropout.
        '''
        # placeholder definition
        self.users = tf.placeholder(tf.int32, shape=(None,))
        self.pos_items = tf.placeholder(tf.int32, shape=(None,))
        self.neg_items = tf.placeholder(tf.int32, shape=(None,))

        self.h0 = tf.placeholder(tf.int32, shape=[None], name='h0')
        self.r0 = tf.placeholder(tf.int32, shape=[None], name='r0')
        self.pos_t0 = tf.placeholder(tf.int32, shape=[None], name='pos_t0')
        self.pos_tags_u = tf.placeholder(tf.int32, shape=[None, None], name='pos_tags_u')
        self.mask_pos_tags_u = tf.placeholder(tf.int32, shape=[None, None], name='mask_pos_tags_u')
        self.mask_pos_tags_us = tf.placeholder(tf.float32, shape=[None, None], name='mask_pos_tags_us')

        self.node_dropout_flag = args.node_dropout_flag
        self.node_dropout = tf.placeholder(tf.float32, shape=None)
        self.mess_dropout = tf.placeholder(tf.float32, shape=None)

        """
        *********************************************************
        Create Model Parameters (i.e., Initialize Weights).
        """
        # initialization of model parameters
        self.weights = self._init_weights()
        self._build_model_phase_II()
        self.ua_embeddings, self.ia_embeddings, self.ta_embeddings = self._create_embed()

        """
        *********************************************************
        Establish the final representations for user-item pairs in batch.
        """
        self.u_g_embeddings = tf.nn.embedding_lookup(self.ua_embeddings, self.users)
        self.pos_i_g_embeddings = tf.nn.embedding_lookup(self.ia_embeddings, self.pos_items)
        self.neg_i_g_embeddings = tf.nn.embedding_lookup(self.ia_embeddings, self.neg_items)

        """
        *********************************************************
        Inference for the testing phase.
        """
        self.batch_ratings = tf.matmul(self.u_g_embeddings, self.pos_i_g_embeddings, transpose_a=False, transpose_b=True)

        """
        *********************************************************
        Generate Predictions & Optimize via BPR loss.
        """
        self.mf_loss, self.emb_loss, self.reg_loss = self.create_bpr_loss(self.u_g_embeddings,
                                                                          self.pos_i_g_embeddings,
                                                                          self.neg_i_g_embeddings)
        self.loss = self.mf_loss + self.emb_loss + self.reg_loss

        self.opt = tf.train.AdamOptimizer(learning_rate=self.lr).minimize(self.loss)

    def _init_weights(self):
        all_weights = dict()
        initializer = tf.contrib.layers.xavier_initializer()

        if self.pretrain_data is None:
            all_weights['user_embedding'] = tf.Variable(initializer([self.n_users, self.emb_dim]),
                                                        name='user_embedding')
            all_weights['item_embedding'] = tf.Variable(initializer([self.n_items, self.emb_dim]),
                                                        name='item_embedding')
            print('using xavier initialization')
        else:
            all_weights['user_embedding'] = tf.Variable(initial_value=self.pretrain_data['user_embeded'],
                                                        trainable=True,
                                                        name='user_embedding', dtype=tf.float32)
            all_weights['item_embedding'] = tf.Variable(initial_value=self.pretrain_data['item_embeded'],
                                                        trainable=True,
                                                        name='item_embedding', dtype=tf.float32)
            print('using pretrained initialization')

        all_weights['tag_embedding'] = tf.Variable(initializer([self.n_tags, self.emb_dim]), name='tag_embedding')
        all_weights['mask_tag'] = tf.concat([tf.zeros([1, self.emb_dim]), tf.ones([1, self.emb_dim])], axis=0)
        print('using xavier initialization')
        all_weights['trans_w'] = tf.Variable(initializer([self.n_relations, self.emb_dim, self.emb_dim]),
                                             name='trans_w')

        self.weight_size_list = [self.emb_dim] + self.weight_size

        for k in range(self.n_layers):
            all_weights['W_gc_%d' % k] = tf.Variable(
                initializer([self.weight_size_list[k], self.weight_size_list[k + 1]]), name='W_gc_%d' % k)
            all_weights['b_gc_%d' % k] = tf.Variable(
                initializer([1, self.weight_size_list[k + 1]]), name='b_gc_%d' % k)

            all_weights['W_bi_%d' % k] = tf.Variable(
                initializer([self.weight_size_list[k], self.weight_size_list[k + 1]]), name='W_bi_%d' % k)
            all_weights['b_bi_%d' % k] = tf.Variable(
                initializer([1, self.weight_size_list[k + 1]]), name='b_bi_%d' % k)

        return all_weights

    def _split_A_hat(self, X):
        A_fold_hat = []
        fold_len = (self.n_users + self.n_items + self.n_tags) // self.n_fold
        for i_fold in range(self.n_fold):
            start = i_fold * fold_len
            if i_fold == self.n_fold -1:
                end = self.n_users + self.n_items + self.n_tags
            else:
                end = (i_fold + 1) * fold_len
            A_fold_hat.append(self._convert_sp_mat_to_sp_tensor(X[start:end]))
        return A_fold_hat

    def _split_A_hat_node_dropout(self, X):
        A_fold_hat = []
        fold_len = (self.n_users + self.n_items + self.n_tags) // self.n_fold
        for i_fold in range(self.n_fold):
            start = i_fold * fold_len
            if i_fold == self.n_fold -1:
                end = self.n_users + self.n_items + self.n_tags
            else:
                end = (i_fold + 1) * fold_len
            # A_fold_hat.append(self._convert_sp_mat_to_sp_tensor(X[start:end]))
            temp = self._convert_sp_mat_to_sp_tensor(X[start:end])
            n_nonzero_temp = X[start:end].count_nonzero()
            A_fold_hat.append(self._dropout_sparse(temp, 1 - self.node_dropout, n_nonzero_temp))
        return A_fold_hat

    def _create_embed(self):
        # Generate a set of adjacency sub-matrix.
        A = self.A_in
        if self.node_dropout_flag:
            # node dropout.
            A_fold_hat = self._split_A_hat_node_dropout(A)
        else:
            A_fold_hat = self._split_A_hat(A)

        ego_embeddings = tf.concat([self.weights['user_embedding'], self.weights['item_embedding'], self.weights['tag_embedding']], axis=0)

        all_embeddings = [ego_embeddings]
        for k in range(0, self.n_layers):
            temp_embed = []
            for f in range(self.n_fold):
                temp_embed.append(tf.sparse_tensor_dense_matmul(A_fold_hat[f], ego_embeddings))

            # sum messages of neighbors.
            side_embeddings = tf.concat(temp_embed, 0)
            # transformed sum messages of neighbors.
            sum_embeddings = tf.nn.leaky_relu(
                tf.matmul(side_embeddings, self.weights['W_gc_%d' % k]) + self.weights['b_gc_%d' % k])

            # bi messages of neighbors.
            bi_embeddings = tf.multiply(ego_embeddings, side_embeddings)
            # transformed bi messages of neighbors.
            bi_embeddings = tf.nn.leaky_relu(
                tf.matmul(bi_embeddings, self.weights['W_bi_%d' % k]) + self.weights['b_bi_%d' % k])

            # non-linear activation.
            ego_embeddings = sum_embeddings + bi_embeddings

            # message dropout.
            ego_embeddings = tf.nn.dropout(ego_embeddings, 1 - self.mess_dropout)

            # normalize the distribution of embeddings.
            norm_embeddings = tf.nn.l2_normalize(ego_embeddings, axis=1)

            all_embeddings += [norm_embeddings]

        all_embeddings = tf.concat(all_embeddings, 1)
        u_g_embeddings, i_g_embeddings, t_g_embeddings = tf.split(all_embeddings, [self.n_users, self.n_items, self.n_tags], 0)
        return u_g_embeddings, i_g_embeddings, t_g_embeddings

    def create_bpr_loss(self, users, pos_items, neg_items):
        pos_scores = tf.reduce_sum(tf.multiply(users, pos_items), axis=1)
        neg_scores = tf.reduce_sum(tf.multiply(users, neg_items), axis=1)
        regularizer = tf.nn.l2_loss(users) + tf.nn.l2_loss(pos_items) + tf.nn.l2_loss(neg_items)
        regularizer = regularizer/self.batch_size

        mf_loss = tf.reduce_sum(tf.nn.softplus(-(pos_scores - neg_scores)))
        emb_loss = self.decay * regularizer
        reg_loss = tf.constant(0.0, tf.float32, [1])
        return mf_loss, emb_loss, reg_loss

    def _convert_sp_mat_to_sp_tensor(self, X):
        coo = X.tocoo().astype(np.float32)
        indices = np.mat([coo.row, coo.col]).transpose()
        return tf.SparseTensor(indices, coo.data, coo.shape)

    def _dropout_sparse(self, X, keep_prob, n_nonzero_elems):
        """
        Dropout for sparse tensors.
        """
        noise_shape = [n_nonzero_elems]
        random_tensor = keep_prob
        random_tensor += tf.random_uniform(noise_shape)
        dropout_mask = tf.cast(tf.floor(random_tensor), dtype=tf.bool)
        pre_out = tf.sparse_retain(X, dropout_mask)

        return pre_out * tf.div(1., keep_prob)

    def _build_model_phase_II(self):
        self.A_score = self._generate_score(self.h0, self.pos_t0, self.r0, self.pos_tags_u,self.mask_pos_tags_u,self.mask_pos_tags_us)
        self.A_out = self._create_attentive_A_out()

    def _create_attentive_A_out(self):
        indices = np.mat([self.all_u_list, self.all_t_list]).transpose()
        A = tf.sparse_softmax(tf.SparseTensor(indices, self.A_values, self.A_in.shape))
        return A

    def _generate_score(self, u, t, r, pos_tags_u, mask_pos_tags_u, mask_pos_tags_us):
        embeddings = tf.concat([self.weights['user_embedding'], self.weights['item_embedding'], self.weights['tag_embedding']], axis=0)
        pos_t = tf.nn.embedding_lookup(embeddings, pos_tags_u)
        embeddings = tf.expand_dims(embeddings, 1)
        mask_pos_tags_u = tf.nn.embedding_lookup(self.weights['mask_tag'], mask_pos_tags_u)
        p_t = tf.divide(tf.reduce_sum(tf.multiply(pos_t, mask_pos_tags_u), axis=1), mask_pos_tags_us)
        u_e = tf.nn.embedding_lookup(embeddings, u)
        t_e = tf.nn.embedding_lookup(embeddings, t)
        t_e_1 = tf.matmul(p_t, self.weights['W_gc_0']) + self.weights['b_gc_0']
        t_e = tf.nn.leaky_relu(t_e + tf.expand_dims(t_e_1, axis=1))

        # transform weight
        trans_w = tf.nn.embedding_lookup(self.weights['trans_w'], r)
        u_e = tf.reshape(u_e, [-1, self.emb_dim])
        t_e = tf.reshape(tf.matmul(t_e, trans_w), [-1, self.emb_dim])
        score = tf.reduce_sum(tf.multiply(u_e, t_e), 1)
        return score

    def update_attentive_A(self, sess):
        fold_len = len(self.all_u_list) // self.n_fold
        kg_score = []

        for i_fold in range(self.n_fold):
            start = i_fold * fold_len
            if i_fold == self.n_fold - 1:
                end = len(self.all_u_list)
            else:
                end = (i_fold + 1) * fold_len

            feed_dict = {
                self.h0: self.all_u_list[start:end],
                self.r0: self.all_r_list[start:end],
                self.pos_t0: self.all_t_list[start:end],
                self.pos_tags_u: self.all_tag_list[start:end],
                self.mask_pos_tags_u: self.mask_tag[start:end],
                self.mask_pos_tags_us: self.mask_tags[start:end]
            }
            A_score = sess.run(self.A_score, feed_dict=feed_dict)
            kg_score += list(A_score)
        kg_score = np.array(kg_score)
        new_A = sess.run(self.A_out, feed_dict={self.A_values: kg_score})
        new_A_values = new_A.values
        new_A_indices = new_A.indices
        rows = new_A_indices[:, 0]
        cols = new_A_indices[:, 1]
        self.A_in = sp.coo_matrix((new_A_values, (rows, cols)), shape=(self.n_users + self.n_items + self.n_tags, self.n_users + self.n_items + self.n_tags))

def load_pretrained_data():
    pretrain_path = args.dataset+'.npz'
    try:
        pretrain_data = np.load(pretrain_path)
        print('load the pretrained embeddings.')
    except Exception:
        pretrain_data = None
    return pretrain_data

if __name__ == '__main__':

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)
    config = dict()
    config['n_users'] = data_generator.n_users
    config['n_items'] = data_generator.n_items
    config['n_tags'] = data_generator.n_tags
    config['train_items'] = data_generator.train_items
    config['batch_size'] = data_generator.batch_size

    """
    *********************************************************
    Generate the Laplacian matrix, where each entry defines the decay factor (e.g., p_ui) between two connected nodes.
    """
    plain_adj, norm_adj, mean_adj = data_generator.get_adj_mat()
    all_u_list, all_t_list, all_r_list, all_v_list ,all_tag_list, mask_tag, mask_tags = data_generator._get_all_data(norm_adj)
    config['all_u_list'] = all_u_list
    config['all_r_list'] = all_r_list
    config['all_t_list'] = all_t_list
    config['all_v_list'] = all_v_list
    config['all_tag_list'] = all_tag_list
    config['mask_tag'] = mask_tag
    config['mask_tags'] = mask_tags

    if args.adj_type == 'plain':
        config['norm_adj'] = plain_adj
        print('use the plain adjacency matrix')

    elif args.adj_type == 'norm':
        config['norm_adj'] = norm_adj
        print('use the normalized adjacency matrix')

    else:
        config['norm_adj'] = mean_adj + sp.eye(mean_adj.shape[0])
        print('use the mean adjacency matrix')

    if args.pretrain == -1:
        pretrain_data = load_pretrained_data()
    else:
        pretrain_data = None

    t0 = time()
    model = a2_gcn(data_config=config, pretrain_data=pretrain_data)

    """
    *********************************************************
    Save the model parameters.
    """
    saver = tf.train.Saver()

    if args.save_flag == 1:
        weights_save_path = '%sweights/%s/%s/r%s' % (args.weights_path, args.dataset,
                                                            str(args.lr), '-'.join([str(r) for r in eval(args.regs)]))
        ensureDir(weights_save_path)
        save_saver = tf.train.Saver(max_to_keep=1)

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)

    """
    *********************************************************
    Reload the pretrained model parameters.
    """
    sess.run(tf.global_variables_initializer())
    cur_best_pre_0 = 0.
    """
    Train.
    """
    loss_loger, pre_loger, rec_loger, ndcg_loger, hit_loger = [], [], [], [], []
    stopping_step = 0
    should_stop = False

    for epoch in range(args.epoch):
        t1 = time()
        loss, mf_loss, emb_loss, reg_loss = 0., 0., 0., 0.
        n_batch = data_generator.n_train // args.batch_size + 1

        for idx in range(n_batch):
            users, pos_items, neg_items = data_generator.sample()
            _, batch_loss, batch_mf_loss, batch_emb_loss, batch_reg_loss = sess.run([model.opt, model.loss, model.mf_loss, model.emb_loss, model.reg_loss],
                               feed_dict={model.users: users, model.pos_items: pos_items,
                                          model.node_dropout: args.node_dropout,
                                          model.mess_dropout: args.mess_dropout,
                                          model.neg_items: neg_items
                                          })
            loss += batch_loss
            mf_loss += batch_mf_loss
            emb_loss += batch_emb_loss
            reg_loss += batch_reg_loss
        model.update_attentive_A(sess)

        # print the test evaluation metrics each 10 epochs; pos:neg = 1:10.
        if (epoch + 1) % 10 != 0:
            if args.verbose > 0 and epoch % args.verbose == 0:
                perf_str = 'Epoch %d [%.1fs]: train==[%.5f=%.5f + %.5f]' % (
                    epoch, time() - t1, loss, mf_loss, reg_loss)
                print(perf_str)
            continue

        t2 = time()
        users_to_test = list(data_generator.test_set.keys())
        ret = test(sess, model, users_to_test, drop_flag=True)
        t3 = time()

        loss_loger.append(loss)
        rec_loger.append(ret['recall'])
        pre_loger.append(ret['precision'])
        ndcg_loger.append(ret['ndcg'])
        hit_loger.append(ret['hit_ratio'])

        if args.verbose > 0:
            perf_str = 'Epoch %d [%.1fs + %.1fs]: train==[%.5f=%.5f + %.5f + %.5f], recall=[%.5f, %.5f], ' \
                       'precision=[%.5f, %.5f], hit=[%.5f, %.5f], ndcg=[%.5f, %.5f]' % \
                       (epoch, t2 - t1, t3 - t2, loss, mf_loss, emb_loss, reg_loss, ret['recall'][0], ret['recall'][1],
                        ret['precision'][0], ret['precision'][1], ret['hit_ratio'][0], ret['hit_ratio'][1],
                        ret['ndcg'][0], ret['ndcg'][1])
            print(perf_str)

        cur_best_pre_0, stopping_step, should_stop = early_stopping(ret['ndcg'][0], cur_best_pre_0,
                                                                    stopping_step, expected_order='acc', flag_step=10)

        # *********************************************************
        # early stopping when cur_best_pre_0 is decreasing for ten successive steps.
        if should_stop == True:
            break

        # *********************************************************
        # save the user & item embeddings for pretraining.
        if ret['ndcg'][0] == cur_best_pre_0 and args.save_flag == 1:
            save_saver.save(sess, weights_save_path + '/weights', global_step=epoch)
            np.savez(args.dataset + '.agcn', user_embeded=sess.run(model.weights)['user_embedding'],
                     item_embeded=sess.run(model.weights)['item_embedding'])
            print('save the weights in path: ', weights_save_path)

    recs = np.array(rec_loger)
    pres = np.array(pre_loger)
    ndcgs = np.array(ndcg_loger)
    hit = np.array(hit_loger)

    best_rec_0 = max(ndcgs[:, 0])
    idx = list(ndcgs[:, 0]).index(best_rec_0)

    final_perf = "Best Iter=[%d]@[%.1f]\trecall=[%s], precision=[%s], hit=[%s], ndcg=[%s]" % \
                 (idx, time() - t0, '\t'.join(['%.5f' % r for r in recs[idx]]),
                  '\t'.join(['%.5f' % r for r in pres[idx]]),
                  '\t'.join(['%.5f' % r for r in hit[idx]]),
                  '\t'.join(['%.5f' % r for r in ndcgs[idx]]))
    print(final_perf)

    save_path = '%soutput/%s.result' % (args.proj_path, args.dataset)
    ensureDir(save_path)
    f = open(save_path, 'a')

    f.write(
        'embed_size=%d, lr=%.4f, node_dropout=%s, mess_dropout=%s, regs=%s, adj_type=%s\n\t%s\n'
        % (args.embed_size, args.lr, args.node_dropout, args.mess_dropout, args.regs,
           args.adj_type, final_perf))
    f.close()
