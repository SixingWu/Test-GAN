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

            trainable_parameters = [W,b]

        return probs, trainable_parameters

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
            GX = tf.nn.sigmoid(tf.matmul(yz_concatenation,W) + b)
            trainable_parameters = [W, b]

        return GX, trainable_parameters

    def create_classifer(self,name,x):
        """
        分类器
        :param name:
        :param x:
        :param y:
        :return:
        """
        config = self.config
        with tf.variable_scope(name):
            W = self._weight_var([config.x_dim, config.num_class], 'W')
            b = self._bias_var([config.num_class], 'b')
            logits = tf.matmul(x,W) + b
            probs = tf.nn.softmax(logits, dim=-1)
            trainable_parameters = [W, b]
        return logits, probs, trainable_parameters

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

        # Maximize
        d_probs, d_paras = self.create_discriminator_or_learner("Discriminator", self.X, self.Y)
        discrminator_objective_term = - tf.log(d_probs)

        classifier_logits, classifier_Y, c_paras = self.create_classifer("Classifier",self.X)

        l_probs, l_paras = self.create_discriminator_or_learner("Learner", self.X, classifier_Y)
        learner_objective_term = - tf.log(l_probs)

        Generated_X, g_paras = self.create_generator("Generator",self.Y,self.Z)

        gd_probs, _ = self.create_discriminator_or_learner("Discriminator", Generated_X, self.Y)
        gl_probs, _ = self.create_discriminator_or_learner("Learner", Generated_X, self.Y)

        generator_objective_term = - tf.log(1.0 - gd_probs)- tf.log( 1.0 - gl_probs)
        prob_Y = tf.maximum( 1e-10, tf.nn.softmax(self.Y))
        # KL_term = tf.reduce_sum(tf.multiply(prob_Y, tf.log(tf.div(prob_Y, classifier_Y))))
        MSE_term = tf.reduce_mean(tf.square(prob_Y - classifier_Y))

        # TODO 检查这里
        CEE = tf.nn.softmax_cross_entropy_with_logits(labels=self.Y, logits=classifier_logits)
        ECEE = tf.reduce_mean(-tf.reduce_sum(prob_Y * tf.log(classifier_Y), reduction_indices=[1]))
        # 定义损失和训练函数

        self.discriminator_loss = tf.reduce_mean(discrminator_objective_term)
        self.train_discriminator_op = self.optimize_with_clip(self.discriminator_loss, var_list=d_paras)
        self.learner_loss = tf.reduce_mean(learner_objective_term)
        self.train_learner_op = self.optimize_with_clip(self.learner_loss, var_list=l_paras)
        self.generator_loss = tf.reduce_mean(generator_objective_term)
        # TODO 检查是否应该同时训练generator和discriminator等的参数
        self.train_generator_op = self.optimize_with_clip(self.generator_loss, var_list=g_paras+d_paras+l_paras)
        self.classifier_loss = tf.reduce_mean(CEE)
        self.train_classifier_op = self.optimize_with_clip(self.classifier_loss, var_list=c_paras, global_step=self.global_step)


        """
        For Inference
        """
        self.classifier_res = classifier_Y

    def optimize_with_clip(self, loss, var_list, global_step=None):
        optimizer = tf.train.AdamOptimizer(0.001)
        grads = optimizer.compute_gradients(loss=loss, var_list=var_list)
        for i, (g, v) in enumerate(grads):
            if g is not None:
                grads[i] = (tf.clip_by_norm(g, 1), v)  # clip gradients
        train_op = optimizer.apply_gradients(grads, global_step=global_step)
        return train_op

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
        Z = self._sample_Z(batch_size)
        _, discriminator_loss = self.sess.run([self.train_discriminator_op, self.discriminator_loss], feed_dict={
            self.X: X_data, self.Z: Z, self.Y: Y_data})
        _, learner_loss = self.sess.run([self.train_learner_op, self.learner_loss], feed_dict={
            self.X: X_data, self.Z: Z, self.Y: Y_data})
        _, generator_loss = self.sess.run([self.train_generator_op, self.generator_loss], feed_dict={
            self.X: X_data, self.Z: Z, self.Y: Y_data})
        _, classifier_loss = self.sess.run([self.train_classifier_op, self.classifier_loss], feed_dict={
            self.X: X_data, self.Z: Z, self.Y: Y_data})
        step = self.sess.run(self.global_step)

        loss = [discriminator_loss,learner_loss,generator_loss,classifier_loss]
        #loss = classifier_loss
        return step, loss

    """
    Infer or Test
    """

    def infer_step(self, X_data, Multiple_Y_data):
        # Discriminator
        batch_size = self.config.batch_size
        Z = self._sample_Z(batch_size)
        probs = self.sess.run([self.classifier_res], feed_dict={
            self.X: X_data, self.Y: Multiple_Y_data})
        return probs


