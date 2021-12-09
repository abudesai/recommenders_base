import numpy as np, pandas as pd
import sys 
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.base import BaseEstimator, TransformerMixin
from scipy.sparse import lil_matrix, csr_matrix, save_npz, load_npz
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
        # self.user_ids = self.user_ids.sample(n=20000, replace=False, random_state=42)        
        
        self.user_ids[self.user_id_int_col] = self.user_ids[self.user_id_col].factorize()[0]
        self.users_orig_to_new = dict( zip(self.user_ids[self.user_id_col], 
            self.user_ids[self.user_id_int_col]) )            

        idx = data[self.user_id_col].isin(self.users_orig_to_new.keys())
        filtered_data = data.loc[idx]

        self.item_ids = data[[self.item_id_col]].drop_duplicates()        
        
        self.item_ids[self.item_id_int_col] = self.item_ids[self.item_id_col].factorize()[0]

        self.items_orig_to_new = dict( zip(self.item_ids[self.item_id_col], 
            self.item_ids[self.item_id_int_col]) )

        
        self.users_new_to_orig = { v:k for k,v in self.users_orig_to_new.items()}
        self.items_new_to_orig = { v:k for k,v in self.items_orig_to_new.items()}      

        print("user orig to new for orig=0:, ", self.users_orig_to_new[0])
        print("user new to orig for new=26348:, ", self.users_new_to_orig[26348])

        
        print("item orig to new for orig=1:, ", self.items_orig_to_new[1])
        print("item new to orig for new=9:, ", self.items_new_to_orig[9])
        return self


    def transform(self, df): 
        idx1 = df[self.user_id_col].isin(self.users_orig_to_new.keys())
        idx2 = df[self.item_id_col].isin(self.items_orig_to_new.keys())
        df = df.loc[idx1 & idx2].copy()        

        df[self.user_id_int_col] = df[self.user_id_col].map(self.users_orig_to_new)
        df[self.item_id_int_col] = df[self.item_id_col].map(self.items_orig_to_new)

        if df.shape[0]<400000:
            print("="*80)
            df.sort_values(by=[self.user_id_col,self.item_id_col], inplace=True)
            print("mapping", df.shape)
            print(df.head())
        return df
        

    def inverse_transform(self, df): 
        df[self.user_id_col] = df[self.user_id_int_col].map(self.users_new_to_orig)
        df[self.item_id_col] = df[self.item_id_int_col].map(self.items_new_to_orig)
        df.sort_values(by=[self.user_id_col,self.item_id_col], inplace=True)
        print(df.head())
        sys.exit()

        # del df[self.user_id_int_col]
        # del df[self.item_id_int_col]
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
        data[self.ratings_int_col] = self.scaler.transform(data[[self.ratings_col]])
        
        if data.shape[0]<400000:            
            print("="*80)
            print("scaling", data.shape, data[self.ratings_int_col].sum())
            print("row sum", data['user_id_int'].sum())
            print("col sum", data['item_id_int'].sum())
            print(data.head())
            print("="*80)
            
        return data


    def inverse_transform(self, data): 
        rescaled_data = data
        rescaled_data[cfg.PRED_RATING_COL] = self.scaler.inverse_transform(data[[cfg.PRED_RATING_INT_COL]])
        # del rescaled_data[cfg.PRED_RATING_INT_COL]
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
        # print("fitted R", self.R.shape) 
        return self
        

    def transform(self, df):
        
        given_N = df[self.user_id_int_col].max() + 1 # number of users
        if given_N > self.N: 
            raise Exception(f"Index of user {given_N} cannot be greater than fitted bound {self.N}")
        
        given_M = df[self.item_id_int_col].max() + 1 # number of items
        if given_M > self.M: 
            raise Exception(f"Index of item {given_M} cannot be greater than fitted bound {self.M}")   

        Y_R = lil_matrix((self.N, self.M))
        Y_R[ df[self.user_id_int_col] , df[self.item_id_int_col] ] = df[self.ratings_int_col] + self.nonzero_const
        
        Y_M = lil_matrix((self.N, self.M))
        Y_M[ df[self.user_id_int_col] , df[self.item_id_int_col] ] = 1
        
        given_users = df[self.user_id_int_col].drop_duplicates()        
        X_R = self.R[given_users, :]
        X_M = self.mask[given_users, :]
        Y_R = Y_R[given_users, :]
        Y_M = Y_M[given_users, :]        
        # print(X_R.shape, X_M.shape, Y_R.shape, Y_M.shape)
        # print(X_R.mean(), X_M.mean(), Y_R.mean(), Y_M.mean())

        if df.shape[0]<400000:  
            print("="*80)
            print("sparse", Y_R.shape, Y_R.sum())
            nz = Y_R.nonzero()
            print('row sum', len(nz[0]), nz[0][:5])
            print('col sum', len(nz[1]), nz[1][:5])
            print("="*80)

        return (X_R, X_M, Y_R, Y_M)


    def inverse_transform(self, data, mask): 
        nonzero_idxs = mask.nonzero() 
        df = pd.DataFrame()
        df[self.user_id_int_col] = nonzero_idxs[0]
        df[self.item_id_int_col] = nonzero_idxs[1]
        df[cfg.PRED_RATING_INT_COL] = data[nonzero_idxs[0], nonzero_idxs[1]]
        return df

    


