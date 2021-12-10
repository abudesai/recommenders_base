
import numpy as np, pandas as pd
import os, sys
from sklearn.utils import shuffle
import joblib

import tensorflow as tf
from keras import backend as K
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dropout, Embedding, Dot, Add, Flatten, \
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


class AutoRec():

    def __init__(self, M, drop_out=0.7, l2_reg=0., lr = 0.1, momentum = 0.9, **kwargs  ):
        self.M = M
        self.drop_out = drop_out 
        self.l2_reg = l2_reg
        self.lr = lr
        self.momentum = momentum
        self.mu = None

        self.model = self.build_model()

        def custom_loss(y_true, y_pred):
            mask = K.cast(K.not_equal(y_true, 0), dtype='float32')
            diff = y_pred - y_true
            sqdiff = diff * diff * mask
            sse = K.sum(K.sum(sqdiff))
            n = K.sum(K.sum(mask))
            return sse / n
            

        self.model.compile(
            loss=custom_loss,
            # optimizer=Adam(lr=0.01),
            optimizer=SGD(learning_rate=self.lr, momentum=self.momentum),
            metrics=[custom_loss],
        )


    def build_model(self): 
        # build the model - just a 1 hidden layer autoencoder
        i = Input(shape=(self.M,))
        x = Dropout(self.drop_out)(i)
        num_hidden = min(self.M // 10, 50)
        x = Dense(num_hidden, activation='tanh', kernel_regularizer=l2(self.l2_reg))(x)
        x = Dense(self.M, kernel_regularizer=l2(self.l2_reg))(x)
        model = Model(i, x)
        return model 


    def fit(self, train_data_tup, valid_data_tup, 
            batch_size=256, epochs=100, verbose=0): 
               
        def generator(X_R, X_M, Y_R, Y_M):
            while True:
                if X_R.shape[0] % batch_size == 0:
                    num_batches = X_R.shape[0] // batch_size
                else:
                    num_batches = X_R.shape[0] // batch_size + 1

                for i in range(num_batches ):
                    upper = min((i+1)*batch_size, X_M.shape[0])
                    x_r = X_R[i*batch_size:upper].toarray()
                    x_m = X_M[i*batch_size:upper].toarray()

                    y_r = Y_R[i*batch_size:upper].toarray()
                    y_m = Y_M[i*batch_size:upper].toarray()

                    x_r = x_r - self.mu * x_m
                    y_r = y_r - self.mu * y_m

                    yield x_r, y_r  # returns X and Y

        train_X_R, train_X_M, train_Y_R, train_Y_M = train_data_tup
        if valid_data_tup is not None: 
            valid_X_R, valid_X_M, valid_Y_R, valid_Y_M = valid_data_tup
            validation_data = generator(valid_X_R, valid_X_M, valid_Y_R, valid_Y_M)
        else:
            validation_data = None

        self.mu = train_X_R.sum() / train_X_M.sum()
        # print("mu: ", self.mu)

        early_stop_loss = 'val_loss' if validation_data != None else 'loss'
        early_stop_callback = EarlyStopping(monitor=early_stop_loss, min_delta = 1e-4, patience=5) 
        infcost_stop_callback = InfCostStopCallback()


        history = self.model.fit(
                x=generator(train_X_R, train_X_M, train_Y_R, train_Y_M),
                validation_data=validation_data,
                batch_size = batch_size,
                epochs=epochs,
                steps_per_epoch=train_X_R.shape[0] // batch_size + 1,
                validation_steps=valid_X_R.shape[0] // batch_size + 1,
                verbose=verbose,
                shuffle=True,
                callbacks=[early_stop_callback, infcost_stop_callback]
            )
        return history


    def predict(self, data, data_mask):
        R = data - self.mu * data_mask
        preds = self.model.predict(R, batch_size=1024) + self.mu
        return preds 

    def summary(self):
        self.model.summary()

    def save(self, model_path): 
        model_params = {
            "M": self.M,
            "drop_out": self.drop_out,
            "l2_reg": self.l2_reg,
            "lr": self.lr,
            "momentum": self.momentum,
            "mu": self.mu,
        }
        joblib.dump(model_params, os.path.join(model_path, cfg.MODEL_PARAMS_FNAME))

        self.model.save_weights(os.path.join(model_path, cfg.MODEL_WTS_FNAME))


    @staticmethod
    def load(model_path): 
        model_params = joblib.load(os.path.join(model_path, cfg.MODEL_PARAMS_FNAME))
        mf = AutoRec(**model_params)
        mf.mu = model_params['mu']
        mf.model.load_weights(os.path.join(model_path, cfg.MODEL_WTS_FNAME)).expect_partial()
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

    mf = AutoRec(N, M, K=10)
    preds = mf.predict()