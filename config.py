import tensorflow as tf
class Config:
    # checkpoint_path
    batch_size = 32
    checkpoint_path = 'model/'
    steps_per_epoch = 10000
    epochs = 10
    infor_step = 100
    middle_size = 8
    # input
    dtype=tf.float32
    num_class = 10
    input_dim = 28 * 28
    operation_dim = 256
    batch_size= 24
    embed=False
    x_dim=input_dim

    # generator
    z_dim = 64

