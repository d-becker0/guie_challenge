from os import mkdir
from os.path import isdir
from datetime import datetime
from collections import Counter

from pandas import DataFrame

from keras.callbacks import Callback

from utils import evaluation_metrics


class TestEmbeddingCallback(Callback):
    def __init__(self, x_test, y_test, save_dir:str, embedding_dim:int=64,num_classes:int=100, distance_type:str='euclidean', save=True):
        super().__init__()
        self.x_test = x_test
        self.y_test = y_test
        self.save_dir = save_dir
        self.num_classes = num_classes
        self.distance_type = distance_type
        self.embedding_dim = embedding_dim
        self.save = save
        self.epoch = 1

        if not isdir(self.save_dir):
            mkdir(self.save_dir)
 
    def on_test_end(self, logs=None):  # seems like logs don't print... idk how to find what it contains
        return self.run()

    # take model up to pt just before training head                                        done  (in model definition)
    # apply pooling steps to get embs of size self.emb_dim                                 done  (in model definition)
    # calculate distances between all pts                                                  
    # calculate scores based on those dists                                                
    def run(self):
        class_preds, embeddings = self._get_embeddings()
        embedding_data, tree = evaluation_metrics.test_embeddings(self.y_test,class_preds,embeddings,self.num_classes)
        tree.build(20)
        emb_df = self._build_df(embedding_data, tree)
        score = self._score(emb_df)
        print("Competition score was", score)
        if self.save:
            emb_df.to_csv(self.save_dir+'/'+str(self.epoch),index=False)
        self.epoch+=1


    # self.model assigned by tf Callback
    # model takes x and y sets (y is not used in predict, but is still required)
    # returns predicted classes and an embedding for each input image
    def _get_embeddings(self):
        pred_class, embeddings = self.model.predict((self.x_test, self.y_test))
        return pred_class,embeddings

    def _build_df(self, embedding_data, tree):
        emb_df = DataFrame(embedding_data)
        emb_df['nearest_neighbors'] = emb_df['annoy_idx'].apply(lambda row: evaluation_metrics.n_neighbors(row,tree,neighbor_count=5))
        emb_df['neighbor_classes'] = emb_df.apply(lambda row: evaluation_metrics.neighbor_classes(row,emb_df,true_classes=True), axis=1)
        emb_df['neighbor_pred_classes'] = emb_df.apply(lambda row: evaluation_metrics.neighbor_classes(row,emb_df,true_classes=False), axis=1)
        emb_df['matching_neighbors'] = emb_df.apply(lambda row: evaluation_metrics.matching_neighbors(row,true_classes=True), axis=1)
        emb_df['matching_neighbor_preds'] = emb_df.apply(lambda row: evaluation_metrics.matching_neighbors(row,true_classes=False), axis=1)

        return emb_df

    def _score(self, emb_df):
        return evaluation_metrics.competition_score(emb_df,5)