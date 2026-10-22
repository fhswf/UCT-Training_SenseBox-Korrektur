import tensorflow as tf

from data.load_data import load_datasets

from sklearn.preprocessing import StandardScaler

from model.build_model import build_model
from model.train import SLTrainer

from utils.misc import init_session
from utils.tf_utils import make_tf_datasets, init_learning_rate


def sl_setup(config):
    """
    Setup for Supervised Learning (SL) training.
    Initializes the training environment for a supervised learning model based on the provided configuration.
    """

    print("Initializing Supervised Learning Setup...")
    config["training"]["session_name"] = init_session(prefix="SL_", directories=[config["training"]["save_model_dir"], config["validation"]["save_results_dir"]])
       
    # Datasets
    batch_size_l = config["training"]["batch_size_labeled"]
    ds_labeled, ds_unlabeled, post_train_eval_data, feature_metadata = load_datasets(config)
    ds_unlabeled = None

    # drop messzeit_utc
    for df in [ds_labeled['X_train'], ds_labeled['X_val'], ds_labeled['X_test']]:
        if 'messzeit_utc' in df.columns:
            df.drop(columns=['messzeit_utc'], inplace=True)

    print(f"Labeled Train DS shapes: {ds_labeled['X_train'].shape}, {ds_labeled['y_train'].shape}")
    print(f"Labeled Val DS shapes: {ds_labeled['X_val'].shape}, {ds_labeled['y_val'].shape}")
    print(f"Test DS shape: {ds_labeled['X_test'].shape}, {ds_labeled['y_test'].shape}")

    continuous_features = [f['name'] for f in feature_metadata if f['type'] == 'continuous']
    continuous_indices = [idx for idx, f in enumerate(feature_metadata) if f['type'] == 'continuous']

    tfds_train_labeled, _, tfds_val_labeled = make_tf_datasets(ds_labeled, ds_unlabeled, batch_size_l=batch_size_l)

    # Scaler
    scaler = eval(config["preprocessing"]["split_and_scaling"]["scaler"])()  
    scaler.fit(ds_labeled['X_train'][continuous_features].values)

    # Learning rate and optimizer
    learning_rate = init_learning_rate(config, steps_per_epoch=len(tfds_train_labeled))

    if hasattr(tf.keras.optimizers, "AdamW"):
        optimizer = tf.keras.optimizers.AdamW(learning_rate=learning_rate)
    else:
        optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

    # Feature Extractor and Regressor
    model = build_model(input_dim=tfds_train_labeled.element_spec[0].shape[1], hidden_units=config["model"]["hidden_units"])    

    trainer = SLTrainer(
        train_ds_l=tfds_train_labeled,
        test_ds=tfds_val_labeled,
        scaler=scaler,
        model=model,
        optimizer=optimizer,
        loss_l_fn=tf.keras.losses.MeanSquaredError(),
        train_cfg=config["training"],
        val_cfg=config["validation"]
    )
    trainer.continuous_features_mask = tf.convert_to_tensor([i in continuous_indices for i in range(tfds_train_labeled.element_spec[0].shape[-1])], dtype=tf.bool)
    return trainer, post_train_eval_data, ds_labeled, ds_unlabeled, continuous_features, feature_metadata