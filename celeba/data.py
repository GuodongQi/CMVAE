import os

import csv
import numpy as np


class Data:
    def __init__(self, data_dir):
        self.K_mtr = 8192  # total num meta_train classes
        self.K_mva = 985  # total num meta_val classes
        self.K_mte = 1000  # total num meta_test classes

        x_mtr = np.load(os.path.join(data_dir, 'train_features.npy'))
        np.random.shuffle(x_mtr)

        len_threshold = 20

        x_mva = np.load(os.path.join(data_dir, 'val_features.npy'))
        x_mva_ = []
        csvdata_val = self.loadCSV(os.path.join(data_dir, 'val.csv'))  # csv path
        self.data_val = []
        length = 0
        for i, (k, v) in enumerate(csvdata_val.items()):
            self.data_val.append(v)  # [[img1, img2, ...], [img111, ...]]
            if len(v) > len_threshold:
                x_mva_.append(x_mva[length: length + len(v)])
            length += len(v)

        x_mte = np.load(os.path.join(data_dir, 'test_features.npy'))
        x_mte_ = []
        csvdata_test = self.loadCSV(os.path.join(data_dir, 'test.csv'))  # csv path
        self.data_test = []
        length = 0
        for i, (k, v) in enumerate(csvdata_test.items()):
            self.data_test.append(v)  # [[img1, img2, ...], [img111, ...]]
            if len(v) > len_threshold:
                x_mte_.append(x_mte[length: length + len(v)])
            length += len(v)

        self.x_mtr = x_mtr
        self.x_mva = x_mva_
        self.x_mte = x_mte_

        self.generate_test_episode(5, 5, 15)

    def loadCSV(self, csvf):
        dictLabels = {}
        with open(csvf) as csvfile:
            csvreader = csv.reader(csvfile, delimiter=',')
            next(csvreader, None)  # skip (filename, label)
            for i, row in enumerate(csvreader):
                filename = row[0]
                label = row[1]
                # append filename to current label
                if label in dictLabels.keys():
                    dictLabels[label].append(filename)
                else:
                    dictLabels[label] = [filename]
        return dictLabels

    def generate_test_episode(self, way, shot, query, n_episodes=1, test=True):
        generate_label = lambda way, n_samp: np.repeat(np.eye(way), n_samp, axis=0)
        n_way, n_shot, n_query = way, shot, query
        (K, x) = self.K_mte if test else self.K_mva, self.x_mte if test else self.x_mva

        xtr, ytr, xte, yte = [], [], [], []
        for t in range(n_episodes):
            # sample WAY classes
            classes = np.random.choice(range(len(x)), size=n_way, replace=False)

            xtr_t = []
            xte_t = []
            for k in list(classes):
                # sample SHOT and QUERY instances

                idx = np.random.choice(range(len(x[k])), size=n_shot + n_query, replace=False)

                x_k = x[k][idx]
                xtr_t.append(x_k[:n_shot])
                xte_t.append(x_k[n_shot:])

            xtr.append(np.concatenate(xtr_t, 0))
            xte.append(np.concatenate(xte_t, 0))
            ytr.append(generate_label(n_way, n_shot))
            yte.append(generate_label(n_way, n_query))

        xtr, ytr = np.stack(xtr, 0), np.stack(ytr, 0)
        ytr = np.argmax(ytr, -1)
        xte, yte = np.stack(xte, 0), np.stack(yte, 0)
        yte = np.argmax(yte, -1)
        return [xtr, ytr, xte, yte]
