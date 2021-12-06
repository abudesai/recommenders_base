
MODEL_NAME = 'matrix_factorization'


USER_ID_COL = 'user_id'
ITEM_ID_COL = 'item_id'
RATING_COL = 'rating'


PRED_RATING_INT_COL = 'pred_int_rating'
PRED_RATING_COL = 'pred_rating'


USER_ID_INT_COL = 'user_id_int'
ITEM_ID_INT_COL = 'item_id_int'
RATING_INT_COL = 'rating_int'


VALIDATION_SPLIT = 0.1

# Pipeline steps
USER_ITEM_MAPPER = 'user_item_mapper'
RATINGS_SCALER = 'ratings_scaler'
SPARSE_MATRIX_CREATOR = 'sparse_matrix_creator'

# Model file names
PREPROCESSOR_FNAME = 'preprocess_pipe.save'
MODEL_WTS_FNAME = 'model_weights'
MODEL_PARAMS_FNAME = 'model_params'


# Scoring file name
SCORING_FNAME = 'scores.csv'

# Loss Metric
loss_metric = 'mse'

#------------------------------------------------
# Output Files
PREDICTIONS_FNAME = 'predictions.csv'
HPO_RESULTS_FNAME = 'hpo_results.csv'

