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



mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
debug = False
path = '/ldev/wsx/tmp/netemb/github/dataset/generated_data/eco_blogCatalog3.txt.labeled.reindex'
config = Config()
if not debug:
    print('normal mode')
    data = DataUtil(path)
    config.x_dim = data.num_vertex
    config.input_dim = data.num_vertex
    config.num_class = data.num_class
    config.batch_size = 16
    data.generate_negative_set()
else:
    path = r'C:\Users\v-sixwu\Downloads\all.txt'
    data = DataUtil(path)
    config.x_dim = data.num_vertex
    config.input_dim = data.num_vertex
    config.num_class = data.num_class
    config.batch_size = 16
    data.generate_negative_set(1000)
gan = EdgeGAN(config)
gan.build_graph()
gan.init_session()


current_time = time.time()

for i in range(0,50000):
    if debug:
        X, Y, h, t, ih, it = data.next_batch(config.batch_size, 'train')
    else:
        X, Y,h,t,ih,it = data.next_batch(config.batch_size)
    res = gan.train_step(X,Y,h,t,ih,it)
    if i % 100 == 0:
        print(res)
    if i % 1000 == 0:
        gan.save_to_checkpoint()
        if i % 10000 == 0:
            negative_tuple_nums = config.batch_size * 10000
            data.generate_negative_set(negative_tuple_nums)
        if debug:
            X, Y, h, t, ih, it = data.next_batch(config.batch_size, 'test')
        else:
            X, Y,h,t,ih,it = data.next_batch(config.batch_size,'test')
        probs,answers = do_infer(config,X)
        print("Testing:#########")
        base_scores = 0
        truth = []
        for y_line in Y:
            res = set()
            for index, y in enumerate(y_line):
                if y > 0:
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



