import tensorflow as tf
import numpy as np
from config import Config



class EdgeGAN:
    def __init__(self, config):
        self.config = config
    def _weight_var(self, shape, name):
        return tf.get_variable(name=name, shape=shape, initializer=tf.contrib.layers.xavier_initializer())

    def _bias_var(self, shape, name):
        return tf.get_variable(name=name, shape=shape, initializer=tf.constant_initializer(0))

    def create_discriminator_or_learner(self, name, x, y):
        """
        给定X 判别是否是来自某个集合
        :return:
        """
        config = self.config
        with tf.variable_scope(name, reuse=tf.AUTO_REUSE ):
            y_prob = tf.nn.softmax(y)
            xy_concatenation = tf.concat([x,y_prob], axis=-1)
            W = self._weight_var([config.x_dim+config.num_class, 1],'W')
            b = self._bias_var([1], 'b')
            probs = tf.nn.sigmoid(tf.matmul(xy_concatenation, W) + b)
            probs = tf.reshape(probs, [-1])
        return probs

    def create_generator(self, name, y, z):
        """
        根据y生成X
        :param name:
        :param y:
        :return:
        """
        config = self.config
        with tf.variable_scope(name):
            y_prob = tf.nn.softmax(y)
            yz_concatenation = tf.concat([z, y_prob], axis=-1)
            W = self._weight_var([config.num_class+config.z_dim, config.x_dim], 'W')
            b = self._bias_var([config.x_dim], 'b')
            GX = tf.nn.tanh(tf.matmul(yz_concatenation,W) + b)
        return GX

    def create_classifer(self,name,x,y):
        """
        分类器
        :param name:
        :param x:
        :param y:
        :return:
        """
        config = self.config
        with tf.variable_scope(name):
            y_prob = tf.nn.softmax(y)
            W = self._weight_var([config.x_dim, config.num_class], 'W')
            b = self._bias_var([config.num_class], 'b')
            logits = tf.matmul(x,W) + b
            probs = tf.nn.softmax(logits)
        return probs

    def _sample_Z(self, m):
        '''Uniform prior for G(Z)'''
        return np.random.uniform(-1., 1., size=[m, self.config.z_dim])


    def build_graph(self):
        config = self.config
        """
        Placeholders
        """
        self.global_step = tf.Variable(initial_value=0, dtype=tf.int64, trainable=False)
        self.X = tf.placeholder(config.dtype, shape=[None, config.x_dim], name='X')
        self.Y = tf.placeholder(config.dtype, shape=[None, config.num_class], name='Y')
        self.Z = tf.placeholder(config.dtype, shape=[None, config.z_dim], name='Y')

        """
        for Discrminator
        """
        discrminator_objective_term = - tf.log(self.create_discriminator_or_learner("Discriminator", self.X, self.Y))
        classifier_Y = self.create_classifer("Classifier",self.X,self.Y)
        learner_objective_term = - tf.log(self.create_discriminator_or_learner("Learner", self.X, classifier_Y))
        Generated_X = self.create_generator("Generator",self.Y,self.Z)
        generator_objective_term = - tf.log(

            self.create_discriminator_or_learner("Discriminator",Generated_X, self.Y)
            #+ self.create_discriminator_or_learner("Learner",Generated_X, self.Y)
        )
        - tf.log(

            #self.create_discriminator_or_learner("Discriminator", Generated_X, self.Y)
            self.create_discriminator_or_learner("Learner", Generated_X, self.Y)
        )

        prob_Y = tf.nn.softmax(self.Y)
        KL_term = tf.reduce_sum(tf.multiply(prob_Y, tf.log(tf.div(prob_Y,classifier_Y))))
        # 分开优化
        loss = generator_objective_term + discrminator_objective_term + learner_objective_term + KL_term #+generator_objective_term
        self.loss = tf.reduce_mean(loss)
        self.train_op = tf.train.AdamOptimizer(0.0005).minimize(self.loss)

    def init_session(self):
        config = tf.ConfigProto(allow_soft_placement=True,
                                log_device_placement=False)
        config.gpu_options.allow_growth = True
        self.sess = tf.Session(config=config)
        self.sess.run(tf.global_variables_initializer())

    """
    Training
    """
    def train_step(self,X_data, Y_data):
        # Discriminator
        batch_size = self.config.batch_size
        _, loss = self.sess.run([self.train_op, self.loss], feed_dict={
            self.X: X_data, self.Z: self._sample_Z(batch_size), self.Y: Y_data})

        step = self.sess.run(self.global_step)
        return step, loss


