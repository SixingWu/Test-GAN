from Multiclassifier import MultiClassificationGAN
from tensorflow.examples.tutorials.mnist import input_data
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


gan = MultiClassificationGAN()
gan.init_session()

for i in range(0,50000):
    X, Y = mnist.train.next_batch(gan.batch_size)
    res = gan.train_step( X_data=X, Y_data=Y, YS_data=Y)
    if i % 20000 == 0:
        samples, labels = gan.figure_step(Y)
        print(labels)
        try:
            fig = plot(samples)
            # plt.savefig('out/{}.png'.format(str(i).zfill(3)), bbox_inches='tight')
            i += 1
            plt.show()
            plt.close(fig)
        except Exception as e:
            print(e)
        finally:
            try:
                plt.close(fig)
            except:
                pass

        print(res)