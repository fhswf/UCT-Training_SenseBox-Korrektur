import tensorflow as tf

def compute_MSE(y_true, y_pred):
    return tf.reduce_mean(tf.math.squared_difference(y_true, y_pred))

def compute_RMSE(y_true, y_pred):
    return tf.sqrt(compute_MSE(y_true, y_pred))

def compute_MaxError(y_true, y_pred):
    return tf.reduce_max(tf.abs(y_true - y_pred))

def compute_ErrorProportionOverThreshold(y_true, y_pred, threshold):
    error = tf.abs(y_true - y_pred)
    over_threshold = tf.math.greater(error, threshold)
    proportion = tf.reduce_sum(tf.cast(over_threshold, tf.int32)) / tf.size(y_true)
    return proportion*100

def compute_metrics(y_true, y_pred):
    results = {
        'MSE': compute_MSE(y_true, y_pred).numpy(),
        'RMSE': compute_RMSE(y_true, y_pred).numpy(),
        'MaxError': compute_MaxError(y_true, y_pred).numpy(),
        'PoT@2': compute_ErrorProportionOverThreshold(y_true, y_pred, threshold=2.0).numpy()
    }
    return results 