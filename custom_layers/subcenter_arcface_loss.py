import tensorflow as tf
import math

class SubcenterArcMarginProduct(tf.keras.layers.Layer):
    '''
    Implements large margin arc distance with sub centers

    Reference:
        https://arxiv.org/pdf/1801.07698.pdf
        https://github.com/lyakaap/Landmark2019-1st-and-3rd-Place-Solution/
        https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123560715.pdf
            blob/master/src/modeling/metric_learning.py
    '''
    def __init__(self, n_classes, s=30, m=0.50,k=3, easy_margin=False,
                 ls_eps=0.0, **kwargs):

        super(SubcenterArcMarginProduct, self).__init__(**kwargs)

        self.n_classes = n_classes
        self.s = s
        self.m = m
        self.k=k
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
            'k':self.k,
            'ls_eps': self.ls_eps,
            'easy_margin': self.easy_margin,
        })
        return config

# TODO: make sure there is NO bias term when building this weight
    def build(self, input_shape):
        super(SubcenterArcMarginProduct, self).build(input_shape[0])
        self.W = self.add_weight(
            name='W',
            shape=(self.n_classes,int(input_shape[0][-1]),self.k), #  # (n_class, img_embed, self.k)  --> I think... int(input_shape[0][-1]) same as emb dim (b/c of previous dense)
            initializer='glorot_uniform',
            dtype='float32',
            trainable=True,
            regularizer=None
            )

    # trying to implement from page 6 of https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123560715.pdf
    """Fig. 2. Training the deep face recognition model by minimizing the proposed sub-center ArcFace loss. The main contribution in this paper is highlighted by the blue dashed
        box. Based on a l2 normalisation step on both embedding feature xi ∈ R 512×1 and all sub-centers W ∈ R N×K×512, we get the subclass-wise similarity score S ∈ R
        N×K by a matrix multiplication WT xi. After a max pooling step, we can easily get the class-wise similarity score S 0 ∈ R N×1. The following steps are same as ArcFace"""
    
    #@tf.function(jit_compile=True)
    def call(self, inputs):
        X, y = inputs #
 
        y = tf.cast(y, dtype=tf.int32)
        

        # expecting  (? x 256) x (256 x num_classes x 3 sub centers) = (? x num_classes x 3 subcenters)
        # not the case with batches... seems like it needs (? x 256) X (numclasses x 256 x subcenters)... WHY?!?!?! 

        class_by_centers = tf.matmul(   # (? x 256) X (numclasses x 256 x subcenters) -> (numclasses x ? x subcenters)
              tf.math.l2_normalize(X),
              tf.math.l2_normalize(self.W)
            )

        max_centers = tf.math.reduce_max(class_by_centers, axis=[2]) # (numclasses x ? x subcenters) -> (numclasses x ?)
        angle = tf.math.acos(tf.transpose(max_centers, perm=[1,0]))            # (numclasses x ?) -> (? x numclasses)


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