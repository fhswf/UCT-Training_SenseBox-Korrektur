import numpy as np
import pandas as pd
import tensorflow as tf


def rescale_continuous_features(scaler, x, continuous_features, feature_metadata=None):
    if isinstance(x, tf.Tensor):
        x_scaled = x.numpy().copy()
        x_scaled[:, continuous_features] = scaler.transform(x.numpy()[:, continuous_features])
        x_scaled = tf.convert_to_tensor(x_scaled, dtype=x.dtype)
    elif isinstance(x, pd.DataFrame):
        x_scaled = pd.DataFrame(scaler.transform(x[continuous_features].values), columns=continuous_features)

        # add back features that were not rescaled
        for f in feature_metadata:
            if f['type'] == 'categorical':
                x_scaled[f['name']] = x[f['name']].values

        if 'messzeit_utc' in x.columns:
            x_scaled['messzeit_utc'] = x['messzeit_utc'].values

        # same order of columns
        x_scaled = x_scaled[x.columns]
    elif isinstance(x, np.ndarray):
        x_scaled = x.copy()
        x_scaled[:, continuous_features] = scaler.transform(x[:, continuous_features])
    else:
        raise ValueError(f"Unsupported data type for rescaling: {type(x)}")

    assert x.shape == x_scaled.shape and type(x) == type(x_scaled), "Mismatch after rescaling continuous features."
    return x_scaled