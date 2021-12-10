import numpy as np, pandas as pd
import sys 
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.base import BaseEstimator, TransformerMixin
from scipy.sparse import lil_matrix
import algorithm.model_config as cfg


class UserItemIdMapper(BaseEstimator, TransformerMixin):    
    ''' Generates sequential user and item ids for internal use.'''
    def __init__(self, user_id_col, item_id_col, user_id_int_col, item_id_int_col): 
        super().__init__()
        self.user_id_col = user_id_col
        self.user_id_int_col = user_id_int_col
        self.item_id_col = item_id_col
        self.item_id_int_col = item_id_int_col
        self.new_to_orig_user_map = None
        self.new_to_orig_item_map = None

    
    def fit(self, data): 

        self.user_ids = data[[self.user_id_col]].drop_duplicates()

        # self.user_ids = self.user_ids.sample(n=50, replace=False, random_state=42)        
        
        self.user_ids[self.user_id_int_col] = self.user_ids[self.user_id_col].factorize()[0]
        self.users_orig_to_new = dict( zip(self.user_ids[self.user_id_col], 
            self.user_ids[self.user_id_int_col]) )   

        self.item_ids = data[[self.item_id_col]].drop_duplicates()        
        
        self.item_ids[self.item_id_int_col] = self.item_ids[self.item_id_col].factorize()[0]

        self.items_orig_to_new = dict( zip(self.item_ids[self.item_id_col], 
            self.item_ids[self.item_id_int_col]) )
        
        self.users_new_to_orig = { v:k for k,v in self.users_orig_to_new.items()}
        self.items_new_to_orig = { v:k for k,v in self.items_orig_to_new.items()}      

        return self


    def transform(self, df): 
        idx1 = df[self.user_id_col].isin(self.users_orig_to_new.keys())
        idx2 = df[self.item_id_col].isin(self.items_orig_to_new.keys())
        df = df.loc[idx1 & idx2].copy()        

        df[self.user_id_int_col] = df[self.user_id_col].map(self.users_orig_to_new)
        df[self.item_id_int_col] = df[self.item_id_col].map(self.items_orig_to_new)  

        return df


    def inverse_transform(self, df): 
        df.sort_values(by=[self.user_id_int_col, self.item_id_int_col], inplace=True)
        df[self.user_id_col] = df[self.user_id_int_col].map(self.users_new_to_orig)
        df[self.item_id_col] = df[self.item_id_int_col].map(self.items_new_to_orig)
        return df
    


class RatingsScaler(BaseEstimator, TransformerMixin):  
    ''' Scale ratings '''
    def __init__(self, ratings_col, ratings_int_col, scaler_type='minmax'): 
        super().__init__()
        self.ratings_col = ratings_col
        self.ratings_int_col = ratings_int_col

        if scaler_type == 'minmax':
            self.scaler = MinMaxScaler()
        elif scaler_type == 'standard':
            self.scaler = StandardScaler()
        else:
            raise Exception(f"Undefined scaler type {scaler_type}")

    
    def fit(self, data): 
        self.scaler.fit(data[[self.ratings_col]])
        return self
        

    def transform(self, data):        
        if not self.ratings_col in data.columns: return data                        
        
        data[self.ratings_int_col] = self.scaler.transform(data[[self.ratings_col]])            
        return data


    def inverse_transform(self, data): 
        rescaled_data = data
        rescaled_data[cfg.PRED_RATING_COL] = self.scaler.inverse_transform(data[[cfg.PRED_RATING_INT_COL]])
        return rescaled_data



class SparseMatrixCreator(BaseEstimator, TransformerMixin):  
    ''' create sparse NxM matrix of users and ratings '''
    def __init__(self, user_id_int_col, item_id_int_col, ratings_int_col):
        super().__init__()
        self.user_id_int_col = user_id_int_col
        self.item_id_int_col = item_id_int_col
        self.ratings_int_col = ratings_int_col
        self.N = None; self.M = None
        self.nonzero_const = 1e-9

    
    def fit(self, df): 
        self.N = df[self.user_id_int_col].max() + 1 # number of users
        self.M = df[self.item_id_int_col].max() + 1 # number of items

        self.R = lil_matrix((self.N, self.M))
        self.R[ df[self.user_id_int_col] , df[self.item_id_int_col] ] = df[self.ratings_int_col] + self.nonzero_const
        
        self.mask = lil_matrix((self.N, self.M))
        self.mask[ df[self.user_id_int_col] , df[self.item_id_int_col] ] = 1
        return self
        

    def transform(self, df):
        
        given_N = df[self.user_id_int_col].max() + 1 # number of users
        if given_N > self.N: 
            raise Exception(f"Index of user {given_N} cannot be greater than fitted bound {self.N}")
        
        given_M = df[self.item_id_int_col].max() + 1 # number of items
        if given_M > self.M: 
            raise Exception(f"Index of item {given_M} cannot be greater than fitted bound {self.M}")  

        Y_R = lil_matrix((self.N, self.M))
        Y_M = lil_matrix((self.N, self.M))

        if  self.ratings_int_col in df: 
            Y_R[ df[self.user_id_int_col] , df[self.item_id_int_col] ] = df[self.ratings_int_col] + self.nonzero_const
        else: 
            Y_R[ df[self.user_id_int_col] , df[self.item_id_int_col] ] = self.nonzero_const
        
        Y_M[ df[self.user_id_int_col] , df[self.item_id_int_col] ] = 1
        
        user_ids_int = df[self.user_id_int_col].drop_duplicates()    

        X_R = self.R[user_ids_int, :]
        X_M = self.mask[user_ids_int, :]

        Y_R = Y_R[user_ids_int, :]
        Y_M = Y_M[user_ids_int, :]   

        return (X_R, X_M, Y_R, Y_M, user_ids_int)


    def inverse_transform(self, data, mask, user_ids_int): 
        nonzero_idxs = mask.nonzero() 
        df = pd.DataFrame()
        df['sparse_row_id'] = nonzero_idxs[0]
        df[self.item_id_int_col] = nonzero_idxs[1]
        df[cfg.PRED_RATING_INT_COL] = data[nonzero_idxs[0], nonzero_idxs[1]]

        sparse_row_users_int_ids_map = { k:v for k,v in zip(np.arange(len(user_ids_int)), user_ids_int )}

        df.insert(0, self.user_id_int_col, df['sparse_row_id'].map(sparse_row_users_int_ids_map))
        del df['sparse_row_id']

        return df

    


