import tensorflow as tf
import math

class ArcMarginProduct(tf.keras.layers.Layer):
    '''
    Implements large margin arc distance with sub centers

    Reference:
        https://arxiv.org/pdf/1801.07698.pdf
        https://github.com/lyakaap/Landmark2019-1st-and-3rd-Place-Solution/
    '''
    def __init__(self, n_classes, s=30, m=0.50, easy_margin=False,
                 ls_eps=0.0, **kwargs):

        super(ArcMarginProduct, self).__init__(**kwargs)

        self.n_classes = n_classes
        self.s = s
        self.m = m
        self.ls_eps = ls_eps
        self.easy_margin = easy_margin
        self.cos_m = tf.math.cos(m)
        self.sin_m = tf.math.sin(m)
        self.th = tf.math.cos(math.pi - m)
        self.mm = tf.math.sin(math.pi - m) * m

    def get_config(self):

        config = super().get_config().copy()
        config.update({
            'n_classes': self.n_classes,
            's': self.s,
            'm': self.m,
            'ls_eps': self.ls_eps,
            'easy_margin': self.easy_margin,
        })
        return config

    def build(self, input_shape):
        super(ArcMarginProduct, self).build(input_shape.as_list()[1])
        self.W = self.add_weight(
            name='W',
            shape=(input_shape.as_list()[1],self.n_classes),
            initializer='glorot_uniform',
            dtype='float32',
            trainable=True,
            regularizer=None
            )

    def call(self, inputs):
        X, y = inputs #
        
        y = tf.cast(y, dtype=tf.int32)
        
        angle = tf.matmul(
              tf.math.l2_normalize(X),
              tf.math.l2_normalize(self.W)
            )

        angle = tf.math.acos(tf.transpose(angle, perm=[1,0]))

        # https://hav4ik.github.io/articles/deep-metric-learning-survey#fig-net-gem-af
        one_hot = tf.cast(
            tf.one_hot(y, depth=self.n_classes),
            dtype=angle.dtype
        )

        margin_angle = angle  + one_hot * self.m


        cosine = tf.math.cos(margin_angle)

        output = cosine
        output *= self.s
        return output