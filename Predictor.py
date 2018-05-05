from EdgeGAN import EdgeGAN
from tensorflow.examples.tutorials.mnist import input_data
from config import Config
from Data_Util import DataUtil
import time
import numpy as np
import sys




def main(arg):
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

    debug = False
    path = arg[0] # '/ldev/wsx/tmp/netemb/github/dataset/generated_data/eco_blogCatalog3.txt.labeled.reindex'
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
    config.batch_size = 256
    config.checkpoint_path += arg[1]+'/'
    gan = EdgeGAN(config)
    gan.build_graph()
    gan.init_session()


    current_time = time.time()

    with open(arg[2],'w+') as fout:
        print("Predicting start" )
        i = -1
        while True:
            i += 1
            try:
                X = data.next_infer_batch(config.batch_size)
                probs, answers = do_infer(config, X)
                for answer in answers:
                    fout.write('%s\n' % answer)
            except EOFError as e:
                print(e)
                print("Predicting %d is finished" )
                break;


if __name__ == "__main__":
    main(sys.argv[1:])
