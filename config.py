import tensorflow as tf
class Config:
    # training process
    batch_size = 32
    show_res_per_steps = 100
    internal_test_per_steps = 10 * show_res_per_steps
    checkpoint_per_steps = 10 * show_res_per_steps
    negative_sampling_per_step = 10 * show_res_per_steps
    negative_sampling_ratio = 1
    epochs = 10
    # checkpoint_path

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

