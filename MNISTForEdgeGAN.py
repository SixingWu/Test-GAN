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
    res = gan.train_step(X_data=X, Y_data=Y)
    if i % 100 == 0:
        print(res)
    if i % 1000 == 0:
        X, Y = mnist.train.next_batch(5)
        probs = gan.infer_step(X_data=X, Multiple_Y_data=Y)
        print("Testing:#########")
        print(Y)
        print(probs)
        print("Testing:#########")
