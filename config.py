class Config:
    # checkpoint_path
    checkpoint_path = 'model/'
    steps_per_epoch = 10000
    epochs = 10
    infor_step = 100

    # input
    num_class = 10
    input_dim = 28 * 28
    batch_size=16
    embed=False
    x_dim=input_dim

    # generator
    z_dim = 64

