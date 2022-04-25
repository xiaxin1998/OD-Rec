import sys
import copy
import random
import numpy as np
from collections import defaultdict
from operator import itemgetter


def count_hot(sessions, target):
    cnt_hot = {}
    for sess in sessions:
        for i in sess:
            if i not in cnt_hot:
                cnt_hot[i] = 1
            else:
                cnt_hot[i] += 1
    cnt_hot = sorted(cnt_hot.items(), key=lambda kv: kv[1], reverse=True)
    # hot_item = list(cnt_hot.keys())
    length = len(cnt_hot)
    hot = [cnt_hot[i][0] for i in range(int(length*0.2))]
    most_pop = [cnt_hot[i][0] for i in range(int(length*0.05))]
    return cnt_hot, hot, most_pop


class Data():
    def __init__(self, data, train_data, shuffle=False, n_node=None, train=None):
        self.raw = np.asarray(data[0])
        self.n_node = n_node
        self.targets = np.asarray(data[1])
        self.length = len(self.raw)
        self.shuffle = shuffle
        if train == 1:
            self.item_dict, self.hot_item, self.most_pop = count_hot(data[0], data[1])
        else:
            self.item_dict, self.hot_item, self.most_pop = train_data.item_dict, train_data.hot_item, train_data.most_pop

    def generate_batch(self, batch_size):
        if self.shuffle:
            shuffled_arg = np.arange(self.length)
            np.random.shuffle(shuffled_arg)
            self.raw = self.raw[shuffled_arg]
            self.targets = self.targets[shuffled_arg]
        n_batch = int(self.length / batch_size)
        if self.length % batch_size != 0:
            n_batch += 1
        slices = np.split(np.arange(n_batch * batch_size), n_batch)
        slices[-1] = np.arange(self.length - batch_size, self.length)
        return slices

    def get_slice(self, index):
        items, num_node = [], []
        hot_mask, cold_mask, hot_sess_len, hot_sess_items,  hot_tar, cold_only_index,\
        hot_only_index, cold_sess_items, cold_sess_len, hot_cold_tar = [], [], [], [], [], [], [], [], [], []
        inp = self.raw[index]
        for session in inp:
            num_node.append(len(np.nonzero(session)[0]))
        max_n_node = np.max(num_node)
        # print(max_n_node)
        session_len = []
        mask = []
        tar = self.targets[index]
        hot = set(self.hot_item)

        for t, session in enumerate(inp):
            nonzero_elems = np.nonzero(session)[0]
            length = len(nonzero_elems)
            sess_items = session + (max_n_node - length) * [0]
            session_len.append([length])
            items.append(sess_items)
            mask.append([1] * length + (max_n_node - length) * [0])
            if length > 1:
                if len(set(session).intersection(hot)) > 0 and len(set(session).intersection(hot)) < len(set(session)):
                    hot_item, cold_item = [], []
                    # hot_item_batch, cold_item_batch = [], []
                    for cnt, item in enumerate(session):
                        if item in hot:
                            hot_item.append(item)
                        else:
                            cold_item.append(item)
                    hot_sess_items.append(hot_item + (max_n_node - len(hot_item)) * [0])
                    cold_sess_items.append(cold_item + (max_n_node - len(cold_item)) * [0])
                    hot_sess_len.append([len(hot_item)])
                    cold_sess_len.append([len(cold_item)])
                    hot_cold_tar.append(tar[t])
                    hot_mask.append([1] * len(hot_item) + (max_n_node - len(hot_item)) * [0])
                    cold_mask.append([1] * len(cold_item) + (max_n_node - len(cold_item)) * [0])
                elif len(set(session).intersection(hot)) == 0:
                    cold_only_index.append(t)
                else:
                    hot_only_index.append(t)
        return items, tar, session_len, mask, hot_sess_items, cold_sess_items, hot_sess_len, cold_sess_len, \
               hot_cold_tar, hot_mask, cold_mask, hot_only_index, cold_only_index

