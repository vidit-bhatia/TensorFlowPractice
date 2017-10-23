import cPickle
import numpy as np
import os


class CifarLoader(object):
    def __init__(self, source_files):
        self._source = source_files
        self._i = 0
        self.images = None
        self.labels = None
        self.DATA_PATH = "data/"

    def one_hot(self, vec, vals=10):
        n = len(vec)
        out = np.zeros((n, vals))
        out[range(n), vec] = 1
        return out


    def load(self):
        data = [self.unpickle(f) for f in self._source]
        images = np.vstack([d["data"] for d in data])
        n = len(images)
        self.images = images.reshape(n, 3, 32, 32).transpose(0, 2, 3, 1)\
                          .astype(float) / 255
        self.labels = self.one_hot(np.hstack([d["labels"] for d in data]), 10)
        return self

    def next_batch(self, batch_size):
        x, y = (self.images[self._i:self._i+batch_size],
               self.labels[self._i:self._i+batch_size])
        self._i = (self._i + batch_size) % len(self.images)
        return x, y

    def unpickle(self, file):
        with open(os.path.join(self.DATA_PATH, file), 'rb') as fo:
            dict = cPickle.load(fo)
        return dict

