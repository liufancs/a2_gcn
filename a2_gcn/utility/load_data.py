'''
Created on Oct 10, 2018
Tensorflow Implementation of Neural Graph Collaborative Filtering (NGCF) model in:
Wang Xiang et al. Neural Graph Collaborative Filtering. In SIGIR 2019.

@author: Xiang Wang (xiangwang@u.nus.edu)
'''
import numpy as np
import random as rd
import scipy.sparse as sp
from time import time
import collections
from utility.parser import parse_args
args = parse_args()

class Data(object):
    def __init__(self, path, batch_size):
        self.path = path
        train_file = path + '/train.txt'
        test_file = path + '/test.txt'
        tag_file = path + '/tag.txt'

        #get number of users and items
        self.n_users, self.n_items, self.n_tags = 0, 0, 0
        self.n_train, self.n_test, self.n_t= 0, 0, 0
        self.neg_pools = {}

        self.exist_users = []
        self.exist_items = []
        #read train_file
        #n_users: number of users, n_items:number of items
        #n_train: iteractions of all train nodes
        with open(train_file) as f:
            for l in f.readlines():
                if len(l) > 0:
                    l = l.strip('\n').split(' ')
                    items = [int(i) for i in l[1:]]
                    uid = int(l[0])
                    self.exist_users.append(uid)
                    self.n_items = max(self.n_items, max(items))
                    self.n_users = max(self.n_users, uid)
                    self.n_train += len(items)

        #n_test: interactions of all test nodes
        with open(test_file) as f:
            for l in f.readlines():
                if len(l) > 0:
                    l = l.strip('\n')
                    try:
                        items = [int(i) for i in l.split(' ')[1:]]
                    except Exception:
                        continue
                    self.n_items = max(self.n_items, max(items))
                    self.n_test += len(items)
        self.n_items += 1
        self.n_users += 1
        #n_tags: number of tags
        with open(tag_file) as f:
            for l in f.readlines():
                if len(l) > 0:
                    l = l.strip('\n')
                    self.n_t+=1
                try:
                    tag = int(l.split(' ')[1])
                    item = int(l.split(' ')[0])
                    if item not in self.exist_items:
                        self.exist_items.append(item)
                except Exception:
                    continue
                self.n_tags = max(self.n_tags, tag)
        self.n_tags += 1

        #self.print_statistics()

        self.R = sp.dok_matrix((self.n_users, self.n_items), dtype=np.float32)
        self.R_tag = sp.dok_matrix((self.n_items, self.n_tags), dtype=np.float32)

        self.train_items, self.test_set ,self.item_tags = {}, {}, {}  # train and test data for items(per user)
        self.batch_size = batch_size


        with open(train_file) as f_train:
            with open(test_file) as f_test:
                for l in f_train.readlines():
                    if len(l) == 0: break
                    l = l.strip('\n')
                    items = [int(i) for i in l.split(' ')]
                    uid, train_items = items[0], items[1:]
                    for i in train_items:
                        self.R[uid, i] = 1.

                    self.train_items[uid] = train_items

                for l in f_test.readlines():
                    if len(l) == 0: break
                    l = l.strip('\n')
                    try:
                        items = [int(i) for i in l.split(' ')]
                    except Exception:
                        continue

                    uid, test_items = items[0], items[1:]
                    self.test_set[uid] = test_items

        with open(tag_file) as f_tag:
            for l in f_tag.readlines():
                if len(l) == 0: break
                l = l.strip('\n')
                try:
                    i, t = int(l.split(' ')[0]), int(l.split(' ')[1])
                    self.R_tag[i, t] = 1.
                    if i not in self.item_tags.keys():
                        self.item_tags[i] = []
                    self.item_tags[i].append(t)
                except Exception:
                    continue
        self.n_tag_max = 0
        for i in self.item_tags.keys():
            if self.n_tag_max < len(self.item_tags[i]):
                self.n_tag_max = len(self.item_tags[i])

    def get_adj_mat(self):
        try:
            t1 = time()
            adj_mat = sp.load_npz(self.path + '/s_adj_mat.npz')
            norm_adj_mat = sp.load_npz(self.path + '/s_norm_adj_mat.npz')
            mean_adj_mat = sp.load_npz(self.path + '/s_mean_adj_mat.npz')
            print('already load adj matrix', adj_mat.shape, time() - t1)

        except Exception:
            adj_mat, norm_adj_mat, mean_adj_mat = self.create_adj_mat()
            sp.save_npz(self.path + '/s_adj_mat.npz', adj_mat)
            sp.save_npz(self.path + '/s_norm_adj_mat.npz', norm_adj_mat)
            sp.save_npz(self.path + '/s_mean_adj_mat.npz', mean_adj_mat)
        return adj_mat, norm_adj_mat, mean_adj_mat

    # create three adjacency matrix
    def create_adj_mat(self):
        t1 = time()
        adj_mat = sp.dok_matrix((self.n_users + self.n_items + self.n_tags, self.n_users + self.n_items + self.n_tags), dtype=np.float32)
        adj_mat = adj_mat.tolil()
        R = self.R.tolil()
        R_tag = self.R_tag.tolil()

        adj_mat[:self.n_users, self.n_users:self.n_users + self.n_items] = R
        adj_mat[self.n_users:self.n_users + self.n_items, :self.n_users] = R.T

        adj_mat[self.n_users:self.n_users + self.n_items, self.n_users + self.n_items:] = R_tag
        adj_mat[self.n_users + self.n_items:, self.n_users:self.n_users + self.n_items] = R_tag.T

        adj_mat = adj_mat.todok()
        print('already create adjacency matrix', adj_mat.shape, time() - t1)

        t2 = time()

        def normalized_adj_single(adj):
            rowsum = np.array(adj.sum(1))

            d_inv = np.power(rowsum, -1).flatten()
            d_inv[np.isinf(d_inv)] = 0.
            d_mat_inv = sp.diags(d_inv)

            norm_adj = d_mat_inv.dot(adj)
            # norm_adj = adj.dot(d_mat_inv)
            print('generate single-normalized adjacency matrix.')
            return norm_adj.tocoo()

        norm_adj_mat = normalized_adj_single(adj_mat + sp.eye(adj_mat.shape[0]))
        mean_adj_mat = normalized_adj_single(adj_mat)

        print('already normalize adjacency matrix', time() - t2)
        return adj_mat.tocsr(), norm_adj_mat.tocsr(), mean_adj_mat.tocsr()

    # Random 100 items per user as negative pools
    def negative_pool(self):
        t1 = time()
        for u in self.train_items.keys():
            neg_items = list(set(range(self.n_items)) - set(self.train_items[u]))
            pools = [rd.choice(neg_items) for _ in range(100)]
            self.neg_pools[u] = pools
        print('refresh negative pools', time() - t1)

    # sample training data
    def sample(self):
        if self.batch_size <= self.n_users:
            users = rd.sample(self.exist_users, self.batch_size)
        else:
            users = [rd.choice(self.exist_users) for _ in range(self.batch_size)]

        # sample num positive items for user u
        def sample_pos_items_for_u(u, num):
            pos_items = self.train_items[u]
            n_pos_items = len(pos_items)
            pos_batch = []
            while True:
                if len(pos_batch) == num: break
                pos_id = np.random.randint(low=0, high=n_pos_items, size=1)[0]
                pos_i_id = pos_items[pos_id]

                if pos_i_id not in pos_batch:
                    pos_batch.append(pos_i_id)
            return pos_batch

        def sample_neg_items_for_u(u, num):
            neg_items = []
            while True:
                if len(neg_items) == num: break
                neg_id = np.random.randint(low=0, high=self.n_items,size=1)[0]
                if neg_id not in self.train_items[u] and neg_id not in neg_items:
                    neg_items.append(neg_id)
            return neg_items


        def sample_neg_items_for_u_from_pools(u, num):
            neg_items = list(set(self.neg_pools[u]) - set(self.train_items[u]))
            return rd.sample(neg_items, num)

        pos_items, neg_items = [], []
        for u in users:
            pos_items += sample_pos_items_for_u(u, 1)
            neg_items += sample_neg_items_for_u(u, 1)
        return users, pos_items, neg_items

    def get_num_users_items(self):
        return self.n_users, self.n_items

    def _get_all_data(self, norm_adj):
        def _reorder_list(org_list, order):
            new_list = np.array(org_list)
            new_list = new_list[order]
            return new_list

        lap = norm_adj.tocoo()
        all_h_list = list(lap.row)
        all_t_list = list(lap.col)
        all_v_list = list(lap.data)
        all_r_list = [0] * len(all_h_list)
        all_tag_list = [[]] * len(all_h_list)
        mask_tag = [[]] * len(all_h_list)
        mask_tags = [[]] * len(all_h_list)

        org_h_dict = dict()

        for idx, h in enumerate(all_h_list):
            if h not in org_h_dict.keys():
                org_h_dict[h] = [[], [], []]

            org_h_dict[h][0].append(all_t_list[idx])
            org_h_dict[h][1].append(all_r_list[idx])
            org_h_dict[h][2].append(all_v_list[idx])

        sorted_h_dict = dict()
        for h in org_h_dict.keys():
            org_t_list, org_r_list, org_v_list = org_h_dict[h]
            sort_t_list = np.array(org_t_list)
            sort_order = np.argsort(sort_t_list)

            sort_t_list = _reorder_list(org_t_list, sort_order)
            sort_r_list = _reorder_list(org_r_list, sort_order)
            sort_v_list = _reorder_list(org_v_list, sort_order)

            sorted_h_dict[h] = [sort_t_list, sort_r_list, sort_v_list]
        print('\tsort meta-data done.')

        od = collections.OrderedDict(sorted(sorted_h_dict.items()))
        new_h_list, new_t_list, new_r_list, new_v_list = [], [], [], []

        for h, vals in od.items():
            new_h_list += [h] * len(vals[0])
            new_t_list += list(vals[0])
            new_r_list += list(vals[1])
            new_v_list += list(vals[2])

        all_h_list, all_t_list, all_r_list, all_v_list = new_h_list, new_t_list, new_r_list, new_v_list


        for i in range(len(all_h_list)):
            if all_h_list[i] == all_t_list[i]:
                all_r_list[i] = args.n_relations
                all_tag_list[i] = [self.n_users + self.n_items + self.n_tags - 1] * (self.n_tag_max)
                mask_tag[i] = [0] * self.n_tag_max
                mask_tags[i] = [1] * 64
            else:
                if all_h_list[i] < self.n_users:
                    all_r_list[i] = 0
                    item = all_t_list[i] - self.n_users
                    try:
                        all_tag_list[i] = (np.asarray(self.item_tags[item]) + self.n_users + self.n_items).tolist() + [self.n_users + self.n_items+self.n_tags - 1] * (self.n_tag_max - len(self.item_tags[item]))
                        mask_tag[i]=[1]*len(self.item_tags[item]) + [0] * (self.n_tag_max - len(self.item_tags[item]))
                        mask_tags[i]=[len(self.item_tags[item])]*64
                    except:
                        all_tag_list[i]=[self.n_users + self.n_items + self.n_tags - 1] * (self.n_tag_max)
                        mask_tag[i] = [0] * self.n_tag_max
                        mask_tags[i] = [1] * 64
                if all_h_list[i] < self.n_users + self.n_items and all_h_list[i] >= self.n_users and all_t_list[i] < self.n_users+self.n_items:
                    all_r_list[i] = 1
                    all_tag_list[i] = [self.n_users + self.n_items + self.n_tags - 1] * (self.n_tag_max)
                    mask_tag[i] = [0] * self.n_tag_max
                    mask_tags[i] = [1] * 64
                if all_h_list[i] < self.n_users + self.n_items and all_h_list[i] >= self.n_users and all_t_list[i] >= self.n_users + self.n_items:
                    all_r_list[i] = 2
                    all_tag_list[i] = [self.n_users + self.n_items + self.n_tags - 1] * (self.n_tag_max)
                    mask_tag[i] = [0] * self.n_tag_max
                    mask_tags[i] = [1] * 64
                if all_h_list[i] >= self.n_users + self.n_items:
                    all_r_list[i] = 3
                    all_tag_list[i] = [self.n_users + self.n_items + self.n_tags - 1] * (self.n_tag_max)
                    mask_tag[i] = [0] * self.n_tag_max
                    mask_tags[i] = [1] * 64
        return all_h_list, all_t_list, all_r_list, all_v_list, all_tag_list, mask_tag, mask_tags

    def print_statistics(self):
        print('n_users=%d, n_items=%d, n_tags=%d' % (self.n_users, self.n_items, self.n_tags))
        print('n_interactions=%d' % (self.n_train + self.n_test))
        print('n_train=%d, n_test=%d, sparsity=%.5f' % (self.n_train, self.n_test, (self.n_train + self.n_test)/(self.n_users * self.n_items)))
