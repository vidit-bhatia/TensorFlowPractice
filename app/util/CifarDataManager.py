from CifarLoader import *

class CifarDataManager(object):
    def __init__(self):
        self.train = CifarLoader(["data_batch_{0}".format(i)
                                  for i in range(1, 6)]).load()
        self.test = CifarLoader(["test_batch"]).load()