import tensorflow as tf


def make_tf_datasets(ds_labeled, ds_unlabeled, batch_size_l, batch_size_u=None, oversample_labeled_data=False):
    """
    Creates TensorFlow datasets for labeled and unlabeled data, with optional oversampling of labeled data.
    """

    assert len(ds_labeled['X_train']) == len(ds_labeled['y_train']), "Mismatch in labeled train data." 
    assert len(ds_labeled['X_val']) == len(ds_labeled['y_val']), "Mismatch in labeled validation data." 
    if ds_unlabeled is not None:
        assert len(ds_unlabeled['X_train']) == len(ds_unlabeled['pseudo_label']), "Mismatch in unlabeled data."

    # train dataset
    if oversample_labeled_data:
        train_labeled = (
            tf.data.Dataset.from_tensor_slices((ds_labeled['X_train'], ds_labeled['y_train']))
            .shuffle(buffer_size=len(ds_labeled['X_train']), reshuffle_each_iteration=True)
            .repeat() # oversample labeled data
            .batch(batch_size_l, drop_remainder=True)
            .prefetch(tf.data.AUTOTUNE)
        )
    else:
        train_labeled = (
            tf.data.Dataset.from_tensor_slices((ds_labeled['X_train'], ds_labeled['y_train']))
            .shuffle(buffer_size=len(ds_labeled['X_train']), reshuffle_each_iteration=True)
            .batch(batch_size_l, drop_remainder=True)
            .prefetch(tf.data.AUTOTUNE)
        )

    # unlabeled dataset
    if ds_unlabeled is None or batch_size_u is None:
        train_unlabeled = None
    else:
        train_unlabeled = (
            tf.data.Dataset.from_tensor_slices((ds_unlabeled['X_train'], ds_unlabeled['pseudo_label']))
            .shuffle(buffer_size=len(ds_unlabeled), reshuffle_each_iteration=True)
            .batch(batch_size_u, drop_remainder=True)
            .repeat()
            .prefetch(tf.data.AUTOTUNE)
        )

    # validation dataset
    val_labeled = (
        tf.data.Dataset.from_tensor_slices((ds_labeled['X_val'], ds_labeled['y_val']))
        .batch(batch_size_l, drop_remainder=False)
        .prefetch(tf.data.AUTOTUNE)
    )

    return train_labeled, train_unlabeled, val_labeled

def init_learning_rate(config, steps_per_epoch):
    """
    Initializes the learning rate schedule.
    """

    learning_rate = config["training"]["learning_rate"]
    lr_schedule_cfg = config["training"].get("lr_schedule", {"type": "none"})
    total_epochs = config["training"]["epochs"]
    warmup_epochs = lr_schedule_cfg.get("warmup_epochs", 5)
    debug = config["setup"].get("debug", False)
    if lr_schedule_cfg["type"] == "cosine_decay" and not debug and (total_epochs > warmup_epochs):
        lr_min = lr_schedule_cfg.get("lr_min", 1e-5)
        lr_max = learning_rate
        total_steps_warmup = steps_per_epoch * warmup_epochs
        total_steps_decay = steps_per_epoch * (total_epochs - warmup_epochs)

        learning_rate = tf.keras.optimizers.schedules.CosineDecay(
            initial_learning_rate=lr_min,
            decay_steps=total_steps_decay,
            alpha=lr_min,
            name='CosineDecay',
            warmup_target=lr_max,
            warmup_steps=total_steps_warmup
        )
        print(f"Using Cosine Decay LR Schedule: lr_min={lr_min}, lr_max={lr_max}, warmup_epochs={warmup_epochs}, total_epochs={total_epochs}")
    return learning_rate