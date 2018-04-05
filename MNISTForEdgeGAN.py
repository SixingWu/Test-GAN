from EdgeGAN import EdgeGAN
from tensorflow.examples.tutorials.mnist import input_data
from config import Config
import time
import numpy as np
from numpy import random
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
def plot(samples):
    fig = plt.figure(figsize=(4, 4))
    gs = gridspec.GridSpec(4, 4)
    gs.update(wspace=0.05, hspace=0.05)

    for i, sample in enumerate(samples):  # [i,samples[i]] imax=16
        ax = plt.subplot(gs[i])
        plt.axis('off')
        ax.set_xticklabels([])
        ax.set_aspect('equal')
        plt.imshow(sample.reshape(28, 28), cmap='Greys_r')

    return fig

config = Config()
config.batch_size=16
gan = EdgeGAN(config)
gan.build_graph()
gan.init_session()

current_time = time.time()
for i in range(0,50000):
    X, Y = mnist.train.next_batch(config.batch_size)
    Y_mask = random.random_integers(0, 1, np.shape(Y)) + 0.0
    Y_mask -= Y
    Y_mask = np.maximum(0,Y_mask)
    YS = abs(Y_mask)
    YS = YS / np.sum(YS)
    res = gan.train_step(X_data=X, Y_data=Y)
    print(res)