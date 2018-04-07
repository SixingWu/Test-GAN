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

def do_infer(config,X_data):
    num_class = config.num_class
    MX = []
    MY = []
    for x in X_data:
        for i in range(num_class):
            MX.append(x)
            y = np.zeros([num_class])
            y[i] = 1.0
            MY.append(y)

    probs = gan.infer_step(MX,MY)
    probs = np.reshape(probs,[-1, num_class])
    lables = np.argmax(probs, axis=-1)
    return probs,lables


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
        X, Y = mnist.train.next_batch(config.batch_size)
        probs = do_infer(config,X)
        print("Testing:#########")
        print(np.argmax(Y, axis=-1))
        print(probs[1])

        truth = np.argmax(Y, axis=-1)
        probs[1]
        print(sum([a==b for (a,b) in zip(truth,probs[1])]))
        print("Testing:#########")



