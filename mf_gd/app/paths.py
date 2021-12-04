import os

volume_path = './ml_vol/'
train_data_path = os.path.join(volume_path, 'data', 'train', 'ratings_train.csv')
test_data_path = os.path.join(volume_path, 'data', 'test', 'ratings_test.csv')
logs_path = os.path.join(volume_path, 'logs')
model_path = os.path.join(volume_path, 'model')
output_path = os.path.join(volume_path, 'output')

