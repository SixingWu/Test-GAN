from Multiclassifier_exp import MultiClassificationGAN
from config import Config
from Data_Util import DataUtil

data = DataUtil()
config = Config()
config.x_dim = data.num_vertex
config.input_dim = data.num_vertex
config.num_class = data.num_class
config.checkpoint_path = 'netemb/'
gan = MultiClassificationGAN(config)
gan.init_session()

for i in range(0,50000):
    X, Y = data.next_batch(config.batch_size)
    res = gan.train_step( X_data=X, Y_data=Y)
    if i % 500 == 0:
        if i >0:
            gan.save_to_checkpoint()
        samples, labels = gan.figure_step(Y)
        print(labels)
        print(gan.test_step(X_data=X,Y_data=Y))
        print(res)