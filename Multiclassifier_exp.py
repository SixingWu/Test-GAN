import tensorflow as tf
import numpy as np
from Util import log
from Util import one_hot

class MultiClassificationGAN:

    def _sample_Z(self, m):
        '''Uniform prior for G(Z)'''
        return np.random.uniform(-1., 1., size=[m, self.z_dim])

    def _weight_var(self, shape, name):
        return tf.get_variable(name=name, shape=shape, initializer=tf.contrib.layers.xavier_initializer())

    def _bias_var(self, shape, name):
        return tf.get_variable(name=name, shape=shape, initializer=tf.constant_initializer(0))

    def _create_distriminator(self, x, y, input_dim, num_class):
        """

        :param x: batch_size * dimension
        :param y: batch_size * num_class
        :return:
        """
        # Transform Y
        D_W_all = self._weight_var([num_class, input_dim * 128], 'Discriminator_all')
        D_b_all = self._bias_var([num_class, 128], 'Discriminator_b1')

        # batch * input_dim * 128
        D_W1 = tf.reshape(tf.matmul(y,D_W_all), [-1, input_dim, 128])
        D_b1 = tf.reshape(tf.matmul(y,D_b_all), [-1, 128])


        # Generate Parameter

        D_W2 = self._weight_var([128, 1], 'Discriminator_W2')
        D_b2 = self._bias_var([1], 'Discriminator_b2')
        var_list = [D_W_all,D_b_all,D_W2,D_b2]
        x = tf.reshape(x, [-1, 1, input_dim])
        D_h1 = tf.nn.relu(tf.reshape(tf.matmul(x, D_W1), [-1, 128]) + D_b1)
        D_logit = tf.matmul(D_h1, D_W2) + D_b2
        D_prob = tf.nn.sigmoid(D_logit)

        return D_prob, D_logit, var_list

    def _create_generator(self, z, y, output_dim, num_class, z_dim):

        # z2 = W * z1 + b
        G_W2 = self._weight_var([128, output_dim], 'G_W2')
        G_b2 = self._bias_var([output_dim], 'G_B2')


        # Transform Y
        G_W_all = self._weight_var([num_class, z_dim * 128], 'Generator_all')
        G_b_all = self._bias_var([num_class, 128], 'Generator_b1')

        var_list = [G_W_all, G_b_all, G_W2, G_b2]

        # batch * z_dim * 128
        G_W1 = tf.reshape(tf.matmul(y, G_W_all), [-1, z_dim, 128])
        G_b1 = tf.reshape(tf.matmul(y, G_b_all), [-1, 128])
        z = tf.reshape(z, [-1, 1, z_dim])
        G_h1 = tf.nn.relu(tf.reshape(tf.matmul(z, G_W1), [-1, 128]) + G_b1)
        G_log_prob = tf.matmul(G_h1, G_W2) + G_b2
        G_prob = tf.nn.sigmoid(G_log_prob)

        return G_prob, var_list

    def _create_classifer(self, x, y, input_dim, num_class):

        C_W1 = self._weight_var([input_dim, 128], 'C_W1')
        C_b1 = self._bias_var([128], 'C_b1')

        C_W2 = self._weight_var([128, num_class], 'C_W2')
        C_b2 = self._bias_var([num_class], 'C_b2')

        var_list = [C_W1, C_W2, C_b1, C_b2]


        C_h1 = tf.nn.relu(tf.matmul(x, C_W1) + C_b1)
        C_logits = tf.matmul(C_h1, C_W2) + C_b2
        C_prob = tf.nn.softmax(C_logits)
        cross_entropy = tf.reduce_mean(-tf.reduce_sum(y * tf.log(C_prob), reduction_indices=[1]))
        C_label = tf.arg_max(C_prob, dimension=-1)
        C_onehot = tf.one_hot(C_label, depth=num_class, dtype=tf.float32)
        return C_onehot,cross_entropy, var_list

    def _build_graph(self, input_dim, num_class,z_dim):

        self.global_step = tf.Variable(initial_value=0, dtype=tf.int64, trainable=False)

        self.X_input = tf.placeholder(tf.float32, shape=[None, self.config.x_dim], name='X')
        self.Z = tf.placeholder(tf.float32, shape=[None, z_dim], name='Z')
        self.Y = tf.placeholder(tf.float32, shape=[None, num_class], name='Y_Ground_truth')

        if self.config.embed:
            log('embedding mode %d*%d' % (self.config.x_num, input_dim))
            self.embeddings = tf.get_variable(shape=[self.config.x_num, input_dim], name='embedding')
            ids = tf.cast(tf.reshape(self.X_input,[-1]), tf.int64)
            self.X = tf.nn.embedding_lookup(self.embeddings, ids)
        else:
            log('standard mode')
            self.X = self.X_input

        with tf.variable_scope('multi_class_gan', reuse=tf.AUTO_REUSE):
            # classier, input x,ys
            predicted_Y,self.C_loss, classifer_vars = self._create_classifer(self.X, self.Y, input_dim, num_class)
            generated_fakes, generator_vars = self._create_generator(self.Z, self.Y, input_dim, num_class, z_dim)
            self.G_sample = generated_fakes
            D_classifer, D_logit_classifer, discriminator_vars = self._create_distriminator(self.X, predicted_Y, input_dim, num_class)
            D_real, D_logit_real, _ = self._create_distriminator(self.X, self.Y, input_dim, num_class)
            D_fake, D_logit_fake,_  = self._create_distriminator(generated_fakes, self.Y, input_dim, num_class)

        # Inference Function:
        self.infer_discriminator, _, _ = self._create_distriminator(self.X , self.Y,input_dim, num_class)
        # Loss
        self.D_loss = - tf.reduce_mean(tf.log(D_real) + tf.log(1. - D_fake) + tf.log(D_classifer))
        # 对于判别网络, 希望D_fake尽可能大，这样可以迷惑生成网络，
        self.G_loss = -tf.reduce_mean(tf.log(D_fake))
        # Classifier
        self.C_loss2 = - tf.reduce_mean(tf.log(D_classifer))

        def optimize_with_clip(loss, var_list, global_step=None):
            optimizer = tf.train.AdamOptimizer(0.0005)
            grads = optimizer.compute_gradients(loss=loss, var_list=var_list)
            for i, (g, v) in enumerate(grads):
                if g is not None:
                    grads[i] = (tf.clip_by_norm(g, 5), v)  # clip gradients
            train_op = optimizer.apply_gradients(grads, global_step=global_step)
            return train_op

        # TODO 参数问题，学习那些参数？
        self.D_optimizer = optimize_with_clip(self.D_loss, var_list=discriminator_vars, global_step=self.global_step)
        self.C_optimizer = optimize_with_clip(self.C_loss, var_list=classifer_vars)
        self.G_optimizer = optimize_with_clip(self.G_loss, var_list=generator_vars)
        self.C2_optimizer = optimize_with_clip(self.C_loss2, var_list=discriminator_vars)
        # self.D_optimizer = tf.train.AdamOptimizer(0.0005).minimize(self.D_loss, var_list=discriminator_vars)
        # self.C_optimizer = tf.train.AdamOptimizer(0.0005).minimize(self.C_loss, var_list=classifer_vars)
        # self.G_optimizer = tf.train.AdamOptimizer(0.0005).minimize(self.G_loss, var_list=generator_vars)

        log('Graph has been built')


    def figure_step(self,y):
        y_index = tf.argmax(y, dimension=-1)
        samples, label = self.sess.run([self.G_sample, y_index], feed_dict={
            self.Z: self._sample_Z(16), self.Y: y})
        return samples, label


    def inference_step(self, X_data):
        """
        利用GAN去计算每个分类的类别，X——data会自动的拓展到合适的num数目
        :param X_data:
        :return:
        """
        num_class = self.config.num_class

        Y_data = [i for i in range(num_class) for x in X_data]
        X_data = [x for i in range(num_class) for x in X_data]
        Y_data = one_hot(Y_data, num_class)

        probs = self.sess.run([self.infer_discriminator], feed_dict={
            self.X_input: X_data,self.Y: Y_data})

        # batch_size * num_class
        probs = np.reshape(probs, [-1, num_class])
        print(probs)
        predict_label = np.argmax(probs, axis=-1)
        return predict_label

    def test_step(self, X_data, Y_data):
        Y_hat = self.inference_step(X_data)

        pt = 0
        Y_data = np.argmax(Y_data, -1)
        print(Y_data)
        print(Y_hat)
        for y,yh in zip(Y_data,Y_hat):
            if y == yh:
                pt += 1
        return pt / len(Y_data)

    def train_step(self, X_data, Y_data):
        # Discriminator
        batch_size = self.batch_size
        _, D_loss_curr = self.sess.run([self.D_optimizer, self.D_loss], feed_dict={
            self.X_input: X_data, self.Z: self._sample_Z(batch_size), self.Y: Y_data})

        # Generator & Classifier
        _, G_loss_curr = self.sess.run([self.G_optimizer, self.G_loss], feed_dict={
            self.Z: self._sample_Z(batch_size), self.Y: Y_data})
        _, C_loss_curr,_,_= self.sess.run([self.C_optimizer, self.C_loss, self.C2_optimizer, self.C_loss2], feed_dict={
            self.X_input: X_data, self.Z: self._sample_Z(batch_size),  self.Y: Y_data})

        step = self.sess.run(self.global_step)
        return step, D_loss_curr, G_loss_curr, C_loss_curr

    def init_session(self, mode='Train'):
        log('initializing the model...')
        log('train_mode: %s' % mode)
        self.saver = tf.train.Saver()
        self.sess = tf.Session()

        # check from checkpoint
        ckpt_path = self.config.checkpoint_path
        log('check the checkpoint_path : %s' % ckpt_path)
        ckpt = tf.train.get_checkpoint_state(ckpt_path)
        if ckpt and ckpt.model_checkpoint_path:
            log('restoring from %s' % ckpt.model_checkpoint_path)
            self.saver.restore(self.sess, ckpt.model_checkpoint_path)
        elif mode != 'Train':
            raise FileNotFoundError('Inference mode asks the checkpoint !')
        else:
            log('does not find the checkpoint, use fresh parameters')
            self.sess.run(tf.global_variables_initializer())


    def save_to_checkpoint(self, path=None):
        if path is None:
            path = self.config.checkpoint_path
        self.saver.save(self.sess, path + 'model.ckpt', global_step=self.global_step)
        log('checkpoint has been saved to :'+path + 'model.ckpt')


    def __init__(self, config):
        self.config = config
        self.z_dim = self.config.z_dim
        self.batch_size = self.config.batch_size
        self._build_graph(self.config.input_dim, self.config.num_class, self.z_dim)








