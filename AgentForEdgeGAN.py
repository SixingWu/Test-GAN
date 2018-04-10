from EdgeGAN import EdgeGAN
from tensorflow.examples.tutorials.mnist import input_data
from config import Config
from Data_Util import DataUtil
import time
import numpy as np




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
    return probs, lables


#data = input_data.read_data_sets('MNIST_data', one_hot=True).train

mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
debug = False
path = '/Users/mebiuw/Downloads/small_set.txt'
path = '/ldev/wsx/tmp/netemb/github/dataset/generated_data/eco_blogCatalog3.txt.labeled.reindex'
config = Config()
if not debug:
    data = DataUtil(path)
    config.x_dim = data.num_vertex
    config.input_dim = data.num_vertex
    config.num_class = data.num_class

    config.batch_size=16
gan = EdgeGAN(config)
gan.build_graph()
gan.init_session()

current_time = time.time()

for i in range(0,50000):
    if debug:
        X, Y =mnist.train.next_batch(config.batch_size)
    else:
        X, Y = data.next_batch(config.batch_size)
    res = gan.train_step(X_data=X, Y_data=Y)
    if i % 100 == 0:
        print(res)
    if i % 1000 == 0:
        if debug:
            X, Y = mnist.test.next_batch(config.batch_size)
        else:
            X, Y = data.next_batch(config.batch_size,'test')
        probs,answers = do_infer(config,X)
        print("Testing:#########")
        base_scores = 0
        truth = []
        for y_line in Y:
            res = set()
            for index, y in enumerate(y_line):
                if y > 0.9999:
                    res.add(index)
                    base_scores += 1
            truth.append(res)
        base_scores /= (config.num_class * len(Y))


        acc = 0
        for index, answer in enumerate(answers):
            if answer in truth[index]:
                acc += 1
        print(np.argmax(Y, axis=-1))
        print(answers)
        print("%f\t%f" % (base_scores, acc/len(Y)))
        print("Testing:#########")



