from sklearn.pipeline import Pipeline
import pandas as pd
import algorithm.model_config as cfg
import algorithm.preprocessors as pp 

def get_preprocess_pipeline(): 
    pipeline = Pipeline(
        [
            # generate sequential ids for users and items
            (
                cfg.USER_ITEM_MAPPER,
                (
                    pp.UserItemIdMapper(
                        user_id_col = cfg.USER_ID_COL, 
                        item_id_col= cfg.ITEM_ID_COL, 
                        user_id_int_col= cfg.USER_ID_INT_COL, 
                        item_id_int_col = cfg.ITEM_ID_INT_COL, 
                    )
                )
            ), 
            # min max scale ratings
            (
                cfg.RATINGS_SCALER,
                (
                    pp.RatingsScaler(
                        ratings_col = cfg.RATING_COL, 
                        ratings_int_col = cfg.RATING_INT_COL, 
                        scaler_type='minmax',   # minmax, standard
                    )
                )
            )
        ]
    )

    return pipeline




if __name__ == '__main__':

    data_path = './../../data/jester/processed/train/ratings_train.csv'
    train_data = pd.read_csv(data_path)
    print("orig data --------")
    print(train_data.shape)
    print(train_data.head())

    preprocess_pipe = get_preprocess_pipeline()

    df2 = preprocess_pipe.fit_transform(train_data)
    print("transformed data --------")
    print(df2.head())
    print(df2.shape)
