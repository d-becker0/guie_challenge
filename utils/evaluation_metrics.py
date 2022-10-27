from annoy import AnnoyIndex
from numpy import sqrt, dot, argmax

def test_embeddings(true_class,pred_class,pred_emb,num_classes):   
    embedding_data = []
    embedding_dim = len(pred_emb[0])
    tree = AnnoyIndex(embedding_dim, 'euclidean')
    
    # class with max prob is predicted
    pred_class = argmax(pred_class, axis=1)
    print(pred_class)

    for i,pred_tensor, y_array, embedding in zip(range(len(true_class)),pred_class,true_class,pred_emb):
        pred = int(pred_tensor)
        y = int(y_array)

        tree.add_item(i, embedding)
        assert y < num_classes and pred < num_classes
        embedding_data.append({'annoy_idx':i,'true_class':y,'pred_class':pred,'embedding':embedding})

    return embedding_data, tree

def dist_to_origin(embedding):
    return sqrt(dot(embedding,embedding))

def n_neighbors(annoy_idx, tree, neighbor_count=5):
    # the closest embedding is always the same exact embedding, must exclude
    neighbor_count+=1
    return tree.get_nns_by_item(annoy_idx, neighbor_count)[1:]

def neighbor_classes(row, emb_df, true_classes = True):
    if true_classes:
        get_column = 'true_class'
    else:
        get_column = 'pred_class'
    return emb_df.loc[emb_df['annoy_idx'].isin(row.nearest_neighbors),get_column].to_list()
    
def matching_neighbors(row, true_classes = True):
    if true_classes:
        neighbor_class_col = 'neighbor_classes'
        get_column = 'true_class'
    else:
        neighbor_class_col = 'neighbor_pred_classes'
        get_column = 'pred_class'
    return len([True for neighbor_class in row[neighbor_class_col] if row[get_column] == neighbor_class])

def competition_score(emb_df, neighbor_count):
    # Average of (correct neighbors / looked at neighbors)
    # ex: 1/N * (  sum(  1/neighbor_count * per_item_match_count  )  ) = 1/N * ( 1/neighbor_count * total_match_count)
    print(emb_df['matching_neighbors'])
    total_matches = emb_df['matching_neighbors'].sum()
    print('Total matches:',total_matches)
    score = total_matches / (neighbor_count * len(emb_df))
    return score


