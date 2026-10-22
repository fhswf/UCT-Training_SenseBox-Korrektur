import argparse
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' # Suppress TensorFlow logging

import pandas as pd
import tensorflow as tf
import yaml

from dotenv import load_dotenv
from evaluation.metrics import compute_metrics
from preprocessing.rescaling import rescale_continuous_features
from utils.misc import combine_evaluation_data, save_summary, save_scaler
from utils.train_setup import sl_setup
from utils.visualization import plot_learning_curves, plot_evaluation_overview

load_dotenv()


# Args
parser = argparse.ArgumentParser(description="Training script")
parser.add_argument("--config", type=str, default="config/config.yaml", help="Path to the config file")
args = parser.parse_args()


if __name__ == "__main__":
    with open(args.config, "r") as file:
        print(f"Loading config file: {args.config}")
        config = yaml.safe_load(file)
        for section, params in config.items():
            print(f"{section}:")
            for key, value in params.items():
                print(f"    {key}: {value}")

    # Setup
    trainer, post_train_eval_data, ds_labeled, ds_unlabeled, continuous_features, feature_metadata = sl_setup(config)

    # Train the model
    model = trainer.train(epochs=config["training"]["epochs"])

    save_scaler(trainer.scaler, config["training"]["save_model_dir"], config["training"]["session_name"])

    # Evaluate on test set
    print("Evaluating on test set ...")
    y_pred = model.predict(
        rescale_continuous_features(trainer.scaler, ds_labeled['X_test'], continuous_features, feature_metadata),
        batch_size=config["training"]["batch_size_labeled"]
    )
    val_results = compute_metrics(tf.convert_to_tensor(ds_labeled['y_test'], dtype=tf.float32), 
                                tf.squeeze(tf.convert_to_tensor(y_pred, dtype=tf.float32)))
    pre_train_results = compute_metrics(tf.convert_to_tensor(ds_labeled['y_test'], dtype=tf.float32),
                                        tf.squeeze(tf.convert_to_tensor(post_train_eval_data['temperature_value'], dtype=tf.float32)))

    # Results
    print(f"\nResults for Session {config['training']['session_name']}:\nPre-Training: {pre_train_results}\nPost-Training: {val_results}")
    print(f"Labeled Train DS shapes: {ds_labeled['X_train'].shape}, {ds_labeled['y_train'].shape} -- Batch Size: {config['training']['batch_size_labeled']}")
    print(f"Labeled Val DS shapes: {ds_labeled['X_val'].shape}, {ds_labeled['y_val'].shape} -- Batch Size: {config['training']['batch_size_labeled']}")
    if ds_unlabeled is not None:
        print(f"Unlabeled Train DS shape: {ds_unlabeled['X_train'].shape}, {ds_unlabeled['pseudo_label'].shape} -- Batch Size: {config['training']['batch_size_unlabeled']}")
    print(f"Test DS shape: {ds_labeled['X_test'].shape}, {ds_labeled['y_test'].shape} -- Batch Size: {config['training']['batch_size_labeled']}")
    print(f"Model Size: {model.count_params()} params")

    df_eval = combine_evaluation_data(ds_labeled, post_train_eval_data, y_pred)
    df_eval.to_csv(f"{config['validation']['save_results_dir']}/evaluation_details__{config['training']['session_name']}.csv", index=False, sep=';', decimal=',')
    plot_evaluation_overview(df_eval[["measurement_sb", "target", "predictions"]], pre_train_results, val_results, config["training"]["session_name"], config["validation"]["save_results_dir"], binwidth=0.1, include_pdf=False)
    plot_learning_curves(config["training"]["session_name"], config["validation"]["save_results_dir"], eval_metric=config["validation"]["metric_name"])

    # Summary
    config["results"] = {
        "MSE": float(f"{val_results['MSE']:.2f}"),
        "RMSE": float(f"{val_results['RMSE']:.2f}"),
        "MaxError": float(f"{val_results['MaxError']:.2f}"),
        "PoT@2": float(f"{val_results['PoT@2']:.2f}"),
    }
    save_summary(config, config["validation"]["save_results_dir"], session_name=config["training"]["session_name"])