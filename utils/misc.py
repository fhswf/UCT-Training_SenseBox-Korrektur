import datetime
import joblib
import os
import shutil
import uuid
import yaml
import psutil


def save_scaler(scaler, directory, session_name):
    scaler_save_path = f'{directory}/scaler__{session_name}.pkl'
    joblib.dump(scaler, scaler_save_path)
    print(f"Scaler saved to {scaler_save_path}")

def save_model(model, directory, session_name):
    model_save_path = directory + f"/weights__{session_name}.keras"
    model.save(model_save_path)
    print(f"Model saved to {model_save_path}")

def save_summary(summary, directory, session_name):
    summary_save_path = f"{directory}/summary__{session_name}.yaml"
    with open(summary_save_path, 'w') as f:
        yaml.dump(summary, f)
    print(f"Summary saved to {summary_save_path}")

def load_scaler(directory, session_name):
    scaler = joblib.load(f"{directory}/scaler__{session_name}.pkl") 
    return scaler

def load_model(model, directory, session_name):
    model.load_weights(f"{directory}/model__{session_name}.h5")
    return model

def init_session(prefix="", directories=[]):
    session_name = '{date:%Y-%m-%d}__'.format(date=datetime.datetime.now()) + prefix + uuid.uuid4().hex
    print(f"Session name: {session_name}")

    # clear working dirs
    to_clear = []
    for dir in to_clear:
        if os.path.exists(dir):
            shutil.rmtree(dir)

    # create directories
    for dir in directories:
        if not os.path.exists(dir):
            os.makedirs(dir)

    return session_name

def log_profiling_session_start(session_name, directory="output/profiling"):
    if not os.path.exists(directory):
        os.makedirs(directory)
    log_path = os.path.join(directory, f"profiling__{session_name}.txt")
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(f"---\nSession start: {datetime.datetime.now().isoformat()} | {session_name}\n---\n")

def log_datapreparation_profile(df, step_name, session_name, directory="output/profiling"):
    if not os.path.exists(directory):
        os.makedirs(directory)
    log_path = os.path.join(directory, f"profiling__{session_name}.txt")
    process = psutil.Process(os.getpid())
    ram_bytes = process.memory_info().rss           # Resident Set Size (RAM in Bytes)
    ram_gb = ram_bytes / 1e9
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(f"{datetime.datetime.now().isoformat()} | {step_name}\n")
        f.write(f"Shape: {df.shape}\n")
        f.write(f"RAM usage (process): {ram_gb:.2f} GB\n")
        f.write("-" * 40 + "\n")

def log_training_data_profile(obj, step_name, session_name, directory="output/profiling"):
    if not os.path.exists(directory):
        os.makedirs(directory)
    log_path = os.path.join(directory, f"profiling__{session_name}.txt")
    process = psutil.Process(os.getpid())
    ram_bytes = process.memory_info().rss
    ram_gb = ram_bytes / 1e9
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(f"{datetime.datetime.now().isoformat()} | {step_name}\n")
        if isinstance(obj, list):
            f.write(f"List of {len(obj)} DataFrames:\n")
            for i, d in enumerate(obj):
                if hasattr(d, "shape"):
                    f.write(f"  DataFrame {i}: shape={d.shape}\n")
        elif hasattr(obj, "shape"):
            f.write(f"Shape: {obj.shape}\n")
        else:
            f.write(f"Type: {type(obj)}\n")
        f.write(f"RAM usage (process): {ram_gb:.2f} GB\n")
        f.write("-" * 40 + "\n")

def combine_evaluation_data(ds_labeled, post_train_eval_data, y_pred):
    """
    Utility function to combine evaluation data for analysis.
    """

    df_eval = ds_labeled['X_test'].copy()

    assert len(df_eval) == len(post_train_eval_data), "Length of dataframes does not match."
    assert df_eval.index.equals(post_train_eval_data.index), "Indices of dataframes do not match."

    df_eval["messzeit_utc"] = post_train_eval_data["messzeit_utc"]
    df_eval["quell_id"] = post_train_eval_data["quell_id"]

    if df_eval["temperature_value"].equals(post_train_eval_data["temperature_value"]):
        df_eval.rename(columns={"temperature_value": "measurement_sb"}, inplace=True)
    else:
        print("Warning: 'temperature_value' columns do not match. Using post_train_eval_data's 'temperature_value' as 'measurement_sb'.")
        df_eval["measurement_sb"] = post_train_eval_data["temperature_value"]
    
    if "is_dwd_filled" in post_train_eval_data.columns:
        df_eval["is_dwd_filled"] = post_train_eval_data["is_dwd_filled"]
    df_eval['target'] = ds_labeled['y_test']
    df_eval['predictions'] = y_pred
    df_eval['abs_error_model'] = abs(df_eval['target'] - df_eval['predictions'])
    df_eval['abs_error_sb'] = abs(df_eval['target'] - df_eval['measurement_sb'])
    
    fixed_columns = ["messzeit_utc", "quell_id", "target", "predictions", "measurement_sb", "temperature_value_DWD001", "is_dwd_filled", 
                     "temperature_value_neighbor_1", "temperature_value_neighbor_2",
                     "abs_error_model", "abs_error_sb", 
                     "lux_value", "lux_value_neighbor_1", "lux_value_neighbor_2",
                     "uv_value", "uv_value_neighbor_1", "uv_value_neighbor_2",
                     "humidity_value", "humidity_value_neighbor_1", "humidity_value_neighbor_2",
                     "pressure_value", "pressure_value_neighbor_1", "pressure_value_neighbor_2",
    ]    
    fixed_columns = [col for col in fixed_columns if col in df_eval.columns]
    df_eval = df_eval[fixed_columns + [col for col in df_eval.columns if col not in fixed_columns]]
    df_eval = df_eval.sort_values(by="abs_error_model", ascending=False)
    return df_eval