import tensorflow as tf

from preprocessing.rescaling import rescale_continuous_features
from tqdm import tqdm
from utils.logger import CSVLogger


class SLTrainer():
    """
    Trainer class for supervised learning models. Handles training, validation, logging, checkpointing, and early stopping.
    """

    def __init__(self, train_ds_l, test_ds, scaler, model, optimizer, loss_l_fn, train_cfg, val_cfg):
        self.train_cfg = train_cfg
        self.val_cfg = val_cfg

        self.scaler = scaler
        self.model = model
        self.optimizer = optimizer

        self.logger = CSVLogger(
            filename=f"{val_cfg['save_results_dir']}/results__{train_cfg['session_name']}.csv", 
            fieldnames=["epoch", "lr", "loss", "loss_l", "loss_u", f"{val_cfg['metric_name']}", "val_loss", f"val_{val_cfg['metric_name']}"]
        )
        
        # checkpoint 
        self.ckpt_path = f"{train_cfg['save_model_dir']}/model__{train_cfg['session_name']}.weights.h5"
        self.best_val_loss = float("inf")

        # early stopping
        self.early_stopping = train_cfg["early_stopping"]
        self.wait_epochs_early_stopping = None

        # datasets
        self.train_data_l = train_ds_l
        self.test_data = test_ds
        
        # loss function and evaluation metric
        self.loss_l_fn = loss_l_fn
        self.eval_metric = eval(val_cfg["metric_obj"])()

        # use compatible reset function
        if hasattr(self.eval_metric, "reset_states"):
            self.eval_metric.reset_fn = self.eval_metric.reset_states
        else:
            self.eval_metric.reset_fn = self.eval_metric.reset_state

    def train(self, epochs=1):
        for epoch in range(epochs):
            train_stats = self.train_one_epoch(epoch)
            val_stats = self.validate(epoch)

            # stats
            current_lr = float(self.optimizer.learning_rate)
            epoch_stats = {"epoch": epoch, "lr": current_lr, **{k:v.numpy() for k,v in train_stats.items()}, **{k:v.numpy() for k,v in val_stats.items()}}
            stats_str = " | ".join([f"{k}: {v:.5f}" for k, v in epoch_stats.items() if k != "epoch"])
            print(f"Epoch {epoch} | {stats_str}")
            self.logger.log_epoch(epoch_stats)

            # checkpoint and early stopping
            if val_stats["val_loss"] < self.best_val_loss - self.early_stopping["min_delta"]:
                self.wait_epochs_early_stopping = 0
                self.best_val_loss = val_stats["val_loss"]
                self.model.save_weights(self.ckpt_path)
            else:
                self.wait_epochs_early_stopping += 1

            if self.early_stopping["enable"] and (self.wait_epochs_early_stopping >= self.early_stopping["patience"]):
                print(f"Early stopping triggered after {self.wait_epochs_early_stopping} epochs without improvement. Best val_loss: {self.best_val_loss:.4f}")
                break

        # return best model
        self.model.load_weights(self.ckpt_path)
        return self.model

    def train_one_epoch(self, epoch):
        losses_l, metrics = 0.0, 0.0
        total_steps = len(self.train_data_l)
        for step, (x_l, y_l) in enumerate(tqdm(self.train_data_l, total=total_steps, desc=f"Epoch {epoch}")):
            with tf.GradientTape() as tape:
                # Rescaling
                x_l = rescale_continuous_features(self.scaler, x_l, self.continuous_features_mask)

                # Predictions
                _, y_pred = self.model(x_l, training=True)
                y_pred = tf.squeeze(y_pred)

                # Compute loss
                loss_l = self.loss_l_fn(y_l, y_pred)
                
            # Backprop
            grads = tape.gradient(loss_l, self.model.trainable_variables)
            self.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))

            # Metrics
            self.eval_metric.update_state(y_l, y_pred)
            metric = self.eval_metric.result()
            self.eval_metric.reset_fn()

            losses_l += loss_l
            metrics += metric

        stats_epoch = {
            "loss_l": losses_l/(total_steps),
            self.val_cfg['metric_name']: metrics/(total_steps)
        } 
        return stats_epoch 

    def validate(self, epoch):
        losses, metrics = 0.0, 0.0

        for step, (x_test, y_test) in enumerate(self.test_data):
            x_test = rescale_continuous_features(self.scaler, x_test, self.continuous_features_mask)

            y_pred = self.model(x_test, training=False)
            y_pred = tf.squeeze(y_pred)

            loss = self.loss_l_fn(y_test, y_pred)
            self.eval_metric.update_state(y_test, y_pred)
            metric = self.eval_metric.result()
            self.eval_metric.reset_fn()

            losses += loss
            metrics += metric

        stats_epoch = {
            "val_loss": losses/(step+1), 
            f"val_{self.val_cfg['metric_name']}": metrics/(step+1)
        } 
        return stats_epoch