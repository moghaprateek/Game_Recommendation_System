### Supported Function To Develop Recommendation Engine ###

import pandas as pd
import numpy as np
from scipy import sparse
from lightfm import LightFM

from sklearn.metrics.pairwise import cosine_similarity


def build_matrix(data_frame,u_column, i_column, r_column, norm= False, threshold = None):
    matrix_inter = data_frame.groupby([u_column, i_column])[r_column] \
            .sum().unstack().reset_index(). \
            fillna(0).set_index(u_column)
    if norm:
        matrix_inter = matrix_inter.applymap(lambda x: 1 if x > threshold else 0)
    return matrix_inter

def build_dic_user(matrix_inter):
    u_id = list(matrix_inter.index)
    u_dic = {}
    counter = 0 
    for i in u_id:
        u_dic[i] = counter
        counter += 1
    return u_dic

def build_dic_item(data_frame,column_id,column_n):
    item_dict ={}
    for i in range(data_frame.shape[0]):
        item_dict[(data_frame.loc[i,column_id])] = data_frame.loc[i,column_n]
    return item_dict

def apply_model(matrix_inter, component_val=30, loss='warp', epoch=30, n_jobs = 4):
    x = sparse.csr_matrix(matrix_inter.values)
    model = LightFM(no_components= component_val, loss=loss)
    model.fit(x,epochs=epoch,num_threads = n_jobs)
    return model

def find_recom(rec_mode, matrix_inter, player_id, dic_player, 
                               dic_game,threshold = 0,num_items = 10, show_known = True, show_recs = True):
    n_users, n_items = matrix_inter.shape
    x_player = dic_player[player_id]
    rating = pd.Series(rec_mode.predict(x_player,np.arange(n_items)))
    rating.index = matrix_inter.columns
    ratings = list(pd.Series(rating.sort_values(ascending=False).index))
    game_val = list(pd.Series(matrix_inter.loc[player_id,:] \
                                 [matrix_inter.loc[player_id,:] > threshold].index).sort_values(ascending=False))
    ratings = [x for x in ratings if x not in game_val]
    rate_list = ratings[0:num_items]
    game_val = list(pd.Series(game_val).apply(lambda x: dic_game[x]))
    ratings = list(pd.Series(rate_list).apply(lambda x: dic_game[x]))
    
    if show_known == True:
        print("Games Likes:")
        counter = 1
        for i in game_val:
            print(str(counter) + '- ' + i)
            counter+=1
            
    if show_recs == True:
        print("\n Game Recommended Items:")
        counter = 1
        for i in ratings:
            print(str(counter) + '- ' + i)
            counter+=1
    return ratings

def build_embedding_mat(rec_mode,matrix_inter):
    df_item_norm_sparse = sparse.csr_matrix(rec_mode.item_embeddings)
    similarities = cosine_similarity(df_item_norm_sparse)
    item_emdedding_matrix = pd.DataFrame(similarities)
    item_emdedding_matrix.columns = matrix_inter.columns
    item_emdedding_matrix.index = matrix_inter.columns
    
    return item_emdedding_matrix

def return_recomm(embedd_mat, game_id,
                             dic_game, n_items = 10, show = True):
    recommended_items = list(pd.Series(embedd_mat.loc[game_id,:]. \
                                  sort_values(ascending = False).head(n_items+1). \
                                  index[1:n_items+1]))
    if show == True:
        print("Item of interest: {0}".format(dic_game[game_id]))
        print("Similar items:")
        counter = 1
        for i in recommended_items:
            print(str(counter) + '- ' +  dic_game[i])
            counter+=1
    return recommended_items