import abc
from datetime import datetime
from collections import Counter

import tensorflow as tf
from keras.callbacks import Callback

class EmbeddingCallback(Callback):
    def __init__(self, x_val, y_val, save_dir:str, embedding_dim:int=64, distance_type:str='euclidean', write_to_csv:bool=True):
        super().__init__()

        self.x_val = x_val
        self.y_val = y_val
        self.save_dir = save_dir
        self.distance_type = distance_type
        self.embedding_dim = embedding_dim
        self.write_to_csv = write_to_csv
    
    def on_test_end(self, logs=None):
        return self.run()

    # take model up to pt just before training head                                        done
    # apply pooling steps to get embs of size self.emb_dim                                 done
    # calculate distances between all pts      
    # calculate scores based on those dists                                                
    def run(self):
        embeddings = self._get_embeddings()
        

    def _get_embeddings(self):
        return self.model.predict(self.x_val)

    def _calc_distance(self, emb1, emb2):
        distance = False

        if self.distance_type.lower().strip() == 'euclidean':
            distance = tf.math.sqrt(tf.math.pow(emb1,2)+tf.math.pow(emb2,2))

        if self.distance_type.lower().strip() == 'cosine':
            norm_emb1 = tf.nn.l2_normalize(emb1,0)        
            norm_emb2 = tf.nn.l2_normalize(emb2,0)
            distance = tf.reduce_sum(tf.multiply(norm_emb1,norm_emb2))

        return distance

   