from util.CifarDataManager import *
import matplotlib.pyplot as plt
import numpy as np
import random

def display_cifar(images, size):
    n = range(len(images))
    plt.figure()
    plt.gca().set_axis_off()
    im = np.vstack([np.hstack([images[random.choice(n)] for i in range(size)])
                    for i in range(size)])
    plt.imshow(im)
    plt.show()


d = CifarDataManager()
print "Number of train images: {0}".format(len(d.train.images))
print "Number of train labels: {0}".format(len(d.train.labels))
print "Number of test images: {0}".format(len(d.test.images))
print "Number of test images: {0}".format(len(d.test.labels))
images = d.train.images
display_cifar(images, 10)