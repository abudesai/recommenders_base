
import numpy as np, pandas as pd
import os
from sklearn.utils import shuffle
import joblib

import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Embedding, Dot, Add, Flatten, \
    Concatenate, Dense, Activation
from tensorflow.keras.regularizers import l2
from tensorflow.keras.optimizers import SGD, Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, Callback



import algorithm.model_config as cfg


COST_THRESHOLD = float('inf')



class InfCostStopCallback(Callback):
    def on_epoch_end(self, epoch, logs={}):
        loss_val = logs.get('loss')
        if(loss_val == COST_THRESHOLD or tf.math.is_nan(loss_val)):
            print("\nCost is inf, so stopping training!!")
            self.model.stop_training = True


class MatrixFactorizerRes():

    def __init__(self, N, M, K, hidden_dim, l2_reg=0., lr = 0.1, momentum = 0.9, **kwargs  ):
        super(MatrixFactorizerRes, self).__init__(**kwargs)
        self.N = N
        self.M = M
        self.K = K
        self.hidden_dim = hidden_dim
        self.l2_reg = l2_reg
        self.lr = lr
        self.momentum = momentum

        self.model = self.build_model()
        self.model.compile(
            loss='mse',
            # optimizer='adam',
            # optimizer=Adam(lr=0.01),
            optimizer=SGD(learning_rate=self.lr, momentum=self.momentum),
            metrics=['mse'],
        )

    def build_model(self): 
        N, M, K = self.N, self.M, self.K
        
        # keras model
        u = Input(shape=(1,))
        m = Input(shape=(1,))
        u_embedding = Embedding(N, K)(u) # (N, 1, K)
        m_embedding = Embedding(M, K)(m) # (N, 1, K)


        ##### main branch
        u_bias = Embedding(N, 1)(u) # (N, 1, 1)
        m_bias = Embedding(M, 1)(m) # (N, 1, 1)
        x = Dot(axes=2)([u_embedding, m_embedding]) # (N, 1, 1)
        x = Add()([x, u_bias, m_bias])
        x = Flatten()(x) # (N, 1)


        ##### side branch
        u_embedding = Flatten()(u_embedding) # (N, K)
        m_embedding = Flatten()(m_embedding) # (N, K)
        y = Concatenate()([u_embedding, m_embedding]) # (N, 2K)
        y = Dense(self.hidden_dim)(y)
        y = Activation('elu')(y)
        # y = Dropout(0.5)(y)
        y = Dense(1)(y)


        ##### merge
        x = Add()([x, y])

        model = Model(inputs=[u, m], outputs=x)
        return model


    def fit(self, X, y, validation_split=None, batch_size=256, epochs=100, verbose=0): 
                
        early_stop_loss = 'val_loss' if validation_split is not None else 'loss'
        early_stop_callback = EarlyStopping(monitor=early_stop_loss, min_delta = 1e-4, patience=3) 
        infcost_stop_callback = InfCostStopCallback()

        history = self.model.fit(
                x = [ X.iloc[:, 0], X.iloc[:, 1] ],
                y = y, 
                validation_split = validation_split,
                batch_size = batch_size,
                epochs=epochs,
                verbose=verbose,
                shuffle=True,
                callbacks=[early_stop_callback, infcost_stop_callback]
            )
        return history


    def predict(self, X): 
        preds = self.model.predict([ X.iloc[:, 0], X.iloc[:, 1] ])
        return preds 

    def summary(self):
        self.model.summary()

    def save(self, model_path): 
        model_params = {
            "N": self.N,
            "M": self.M,
            "K": self.K,
            "hidden_dim": self.hidden_dim,
            "l2_reg": self.l2_reg,
            "lr": self.lr,
            "momentum": self.momentum,
        }
        joblib.dump(model_params, os.path.join(model_path, cfg.MODEL_PARAMS_FNAME))

        self.model.save_weights(os.path.join(model_path, cfg.MODEL_WTS_FNAME))


    @staticmethod
    def load(model_path): 
        model_params = joblib.load(os.path.join(model_path, cfg.MODEL_PARAMS_FNAME))
        mf = MatrixFactorizerRes(**model_params)
        mf.model.load_weights(os.path.join(model_path, cfg.MODEL_WTS_FNAME))
        return mf


if __name__ == '__main__': 

    data_type = 'train'
    train_data = pd.read_csv(f'./../../')
    print(train_data.shape)

    dataset = 'jester'
    data_type = 'test'
    test_data = pd.read_csv(f'./../../data/{dataset}/processed/{data_type}/ratings_{data_type}.csv')
    print(test_data.shape)

    N = M = 100

    user_ids = np.arange(N)
    item_ids = np.arange(M)

    mf = MatrixFactorizerRes(N, M, K=10)
    preds = mf.predict()