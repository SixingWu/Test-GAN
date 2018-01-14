import numpy as np
def log(log, mode='debug'):
    print(log)

def one_hot(array, nb_classes):
    targets = np.array(array).reshape(-1)
    one_hot_targets = np.eye(nb_classes)[targets]
    return one_hot_targets