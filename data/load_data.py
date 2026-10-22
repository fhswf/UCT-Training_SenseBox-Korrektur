import datetime
import numpy as np
import os
import pandas as pd

from preprocessing.extract_JSON import extract_measurement_data_from_json, extract_additonal_parameters_from_json, extract_quell_properties, extract_raster_properties
from preprocessing.neighboring_senseBoxes import determine_neighboring_senseBoxes
from preprocessing.target_station import determine_target_stations
from preprocessing.DWD_10min import attach_dwd_data_10min
from preprocessing.NaN_incorrect_values import nan_handling
from preprocessing.feature_engineering import calculate_ewma, calculate_rolling_mean, messzeit_utc_handling, transform_categorical_features
from preprocessing.plausibility_checks import plausibility_check, filter_inconsistent_neighbors_DWD

from sklearn.model_selection import train_test_split
from sqlalchemy import create_engine

def load_datasets(config):
    """
    Load datasets based on the provided configuration.
    Retrieves data from the database, applies preprocessing steps, and prepares the datasets for training and evaluation.
    """

    df = load_data(config["data"]["db"], config["data"]["tables"])
    df = df.sort_values(by=["messzeit_utc"])

    # Neighboring senseBoxes
    if config["preprocessing"].get("neighboring_senseBoxes", False):
        df = determine_neighboring_senseBoxes(df, 
                                              config["preprocessing"].get("number_of_neighbors", 0), 
                                              config["preprocessing"].get("measured_values_neighbors"), 
                                              config["preprocessing"].get("plausibility_checks"),
                                              config["preprocessing"].get("neighbors_reduction_method"),
                                              config["preprocessing"].get("consistency_threshold_neighbors", None),
                                              config["preprocessing"].get("use_climatological_neighbors", False))

    # Targets (DecentLab data)
    if config["preprocessing"].get("number_of_DL_stations", 0) > 0:
        df = determine_target_stations(df,
                                        config["preprocessing"].get("use_climatological_targets", False),
                                        config["preprocessing"].get("number_of_DL_stations", 0),
                                        config["preprocessing"]["measured_values_DL_stations"],
                                        distance_threshold_meters=config["preprocessing"].get("distance_threshold_DL", None))

    # DWD data    
    if config["preprocessing"].get("DWD_10min", False):
        df =  attach_dwd_data_10min(df, config["preprocessing"]["DWD_stations_10min"], config["preprocessing"]["DWD_measured_values_10min"])

    df = df[df["quell_typ"]== "senseBox"]

    # JSON data
    print(f"[{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Extracting JSON data...")
    df = extract_measurement_data_from_json(df, config["preprocessing"]["SB_measured_values"])
    df = extract_additonal_parameters_from_json(df, config["preprocessing"]["additional_parameters"])
    df = extract_quell_properties(df, config["preprocessing"]["quell_properties"])
    df = extract_raster_properties(df, config["preprocessing"]["raster_properties"])
    print(f"   Shape: {df.shape}")

    # Plausibility Check: Filter invalid measurements
    if config["preprocessing"]["plausibility_checks"].get("enable", False):
        df = plausibility_check(df, config["preprocessing"]["plausibility_checks"])

    drop_cols = config["preprocessing"].get("drop_columns", [])
    drop_cols = drop_cols + [col for col in df.columns if col.endswith("__OLD")] # drop old neighbor columns after plausibility check
    drop_cols = [col for col in drop_cols if col in df.columns]
    df = df.drop(columns=drop_cols)

    # Drop inconsistent neighbors compared to DWD
    if config["preprocessing"]["consistency_threshold_neighbors"] and 'temperature_value_DWD001' in df.columns:
        print(f"   Filtering inconsistent neighboring senseBoxes based on DWD measurements ...")
        df = filter_inconsistent_neighbors_DWD(df, config["preprocessing"]["consistency_threshold_neighbors"])

    # NaN-handling
    df = nan_handling(df, config["preprocessing"]["NaN_handling"], session_name=config["training"]["session_name"])

    # Feature Engineering
    print(f"[{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Applying feature engineering...")
    df = calculate_rolling_mean(df, config["preprocessing"]["feature_engineering"]["ema_15"].get("columns", []))
    df = calculate_ewma(df, config["preprocessing"]["feature_engineering"]["ewma_15"].get("columns", []))
    df = transform_categorical_features(df)
    df = messzeit_utc_handling(df, config["preprocessing"]["feature_engineering"]["measurement_time_handling"])
    print(f"   Shape: {df.shape}")

    if "niota_standort" in df.columns:
        df = df.drop(columns=["niota_standort"])

    # List of numerical features
    measured_values = []
    for measured_values_cfg in [
        config["preprocessing"].get("measured_values_neighbors", False),
        config['preprocessing']['DWD_measured_values_10min'],
        config['preprocessing']['SB_measured_values'],
        config['preprocessing']['raster_properties'],
    ]:
        if measured_values_cfg:
            if isinstance(measured_values_cfg, list):
                for v in measured_values_cfg:
                    measured_values.append(v)
            else:
                for k, v in measured_values_cfg.items():
                    if isinstance(v, list):
                        for sub_v in v:
                            measured_values.append(f"{k}_{sub_v}")
                    elif isinstance(v, dict):
                        for sub_k, sub_v in v.items():
                            for sub_sub_v in sub_v:
                                measured_values.append(f"{k}_{sub_k}_{sub_sub_v}")
                    else:
                        measured_values.append(f"{k}_{v}")

    # rename temperature_main_value (smoothed DWD data) for compliance with other temperature features
    if 'temperature_main_value' in measured_values:
        measured_values.remove('temperature_main_value')
        measured_values.append('temperature_value')
     
    # remove duplicate entries
    measured_values = list(set(measured_values))

    # specific features
    measured_values.extend(["Gelaendehoehe", "Montagehoehe", "Gelaendeneigung"])
    if "GEL_FORM" in measured_values:
        measured_values.remove("GEL_FORM")

    if config["data"].get("prepare_dataset_splits", True):
        return prepare_for_splits(df,
                            target_name=config["preprocessing"]["split_and_scaling"]["target_name"],
                            test_size=config["preprocessing"]["split_and_scaling"]["test_size_random_split"],
                            random_state=config["preprocessing"]["split_and_scaling"]["random_state"],
                            measured_values=measured_values,
                            test_split=config["preprocessing"]["split_and_scaling"].get("test_split", False),
                            unlabeled_ids_or_size=config["preprocessing"]["split_and_scaling"].get("unlabeled_ids_or_size", False),
                            allow_drop_quell_id=config["preprocessing"].get("allow_drop_quell_id", True),
                            train_val_split=config["preprocessing"]["split_and_scaling"].get("train_val_split", True),
                            test_split_columns_to_keep=config["validation"].get("test_split_columns_to_keep", [])
                            )
    else:
        return df

def prepare_for_splits(df, target_name, test_size, random_state, measured_values, test_split=False, unlabeled_ids_or_size=False, allow_drop_quell_id=True, train_val_split=True, test_split_columns_to_keep=[]):
    """
    Prepare the dataset for training and evaluation.
    Handles the splitting of the dataset into training, validation, and test sets based on the provided parameters.
    """

    df.rename(columns={"temperature_value_dl_1": target_name}, inplace=True)
    if "quell_id" in df.columns and not test_split:
        df = df.drop(columns=["quell_id"])

    # convert to float32
    cols_to_exclude = ['quell_id', 'messzeit_utc', 'is_dwd_filled']
    cols_to_convert = [col for col in df.columns if col not in cols_to_exclude]
    df[cols_to_convert] = df[cols_to_convert].astype(np.float32)

    # extract labeled/unlabeled datasets
    mask_labeled = df[target_name].notna()
    df_labeled = df[mask_labeled].copy().reset_index(drop=True)
    ds_unlabeled = df[~mask_labeled].drop(columns=[target_name]).reset_index(drop=True)

    # filter unlabeled data by specific ids/size
    if unlabeled_ids_or_size:
        print(f"Filtering unlabeled data. Using ids {unlabeled_ids_or_size}")
        ds_unlabeled = ds_unlabeled[ds_unlabeled['quell_id'].isin(unlabeled_ids_or_size)].reset_index(drop=True)
            
    # determine what type of test split should be used 
    if test_split:
        if isinstance(test_split, list):
            # check first entry to determine if it's a time-based or ID-based split
            if test_split[0].startswith("SB"):
                # ID-based split
                test_ids = test_split
            else:
                # time-based split
                test_ids = None
                test_start, test_end = test_split
        elif isinstance(test_split, int):
            # split by number of ids
            unique_ids = df_labeled['quell_id'].unique()
            rng = np.random.RandomState()
            rng.shuffle(unique_ids)
            test_ids = unique_ids[:test_split]
        elif isinstance(test_split, float):
            # split by fraction 
            unique_ids = df_labeled['quell_id'].unique()
            rng = np.random.RandomState()
            rng.shuffle(unique_ids)
            n_test = int(len(unique_ids) * test_split)
            test_ids = unique_ids[:n_test]
        else:
            raise ValueError("test_split must be list, int or float.")

        # perform train/val/test split
        if test_ids is not None:
            # keep out test_ids for test set
            print(f"Using ID-based test split with test IDs: {test_ids}")
            X_train = df_labeled[~df_labeled['quell_id'].isin(test_ids)]
            y_train = df_labeled[~df_labeled['quell_id'].isin(test_ids)][target_name]        
            X_test = df_labeled[df_labeled['quell_id'].isin(test_ids)]
            y_test = df_labeled[df_labeled['quell_id'].isin(test_ids)][target_name]
            print(f"Train IDs: {X_train['quell_id'].unique()}")
            print(f"Test IDs: {X_test['quell_id'].unique()}")
        else:
            # use time window for test set
            print(f"Using time-based test split: {test_start} to {test_end}")
            X_train = df_labeled[~((df_labeled['messzeit_utc'] >= test_start) & (df_labeled['messzeit_utc'] <= test_end))]
            y_train = df_labeled[~((df_labeled['messzeit_utc'] >= test_start) & (df_labeled['messzeit_utc'] <= test_end))][target_name]
            X_test = df_labeled[(df_labeled['messzeit_utc'] >= test_start) & (df_labeled['messzeit_utc'] <= test_end)]
            y_test = df_labeled[(df_labeled['messzeit_utc'] >= test_start) & (df_labeled['messzeit_utc'] <= test_end)][target_name]
            print(f"Train time range: {X_train['messzeit_utc'].min()} to {X_train['messzeit_utc'].max()}")
            print(f"Test time range: {X_test['messzeit_utc'].min()} to {X_test['messzeit_utc'].max()}")

        if X_train.empty:
            print("WARNING: Train set is empty after train/test splitting. This should only happen during testing. Copying test set into train for compliance.")
            X_train = X_test.copy()
            y_train = y_test.copy()

        # split train into train/val
        if train_val_split:
            X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=test_size, random_state=random_state)
        else:
            # empty validation set
            X_val = pd.DataFrame(columns=X_train.columns)
            y_val = pd.Series(dtype=y_train.dtype)

    else:
        # random train/test split (no validation set)
        X = df_labeled.drop(columns=[target_name])
        y = df_labeled[target_name]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    
    for data in [X_train, X_val, X_test, y_train, y_val, y_test]:
        data.reset_index(drop=True, inplace=True)

    # Keep specific test split data for post-training evaluation 
    test_split_columns_to_keep = [col for col in test_split_columns_to_keep if col in X_test.columns]
    post_train_eval_data = X_test[test_split_columns_to_keep].copy()

    # clean up 
    # drop target and optionally quell_id
    X_train = X_train.drop(columns=[target_name])
    X_test = X_test.drop(columns=[target_name])
    if train_val_split:
        X_val = X_val.drop(columns=[target_name])
    
    if allow_drop_quell_id:
        ds_unlabeled = ds_unlabeled.drop(columns=["quell_id"])
        X_train = X_train.drop(columns=["quell_id"])
        X_test = X_test.drop(columns=["quell_id"])
        if train_val_split:
            X_val = X_val.drop(columns=["quell_id"])

    for data in [X_train, X_val, X_test, ds_unlabeled]:
        if "is_dwd_filled" in data.columns:
            data.drop(columns=["is_dwd_filled"], inplace=True)

    # determine feature metadata 
    training_features = pd.concat([ds_unlabeled, X_train])
    for col in ["quell_id", "messzeit_utc"]:
        if col in training_features.columns:
            training_features = training_features.drop(columns=[col])

    # std without outliers
    sigma = (training_features.quantile(0.75) - training_features.quantile(0.25)) / 1.349

    feature_metadata = []
    for col in training_features.columns:
        if col in cols_to_exclude:
            continue

        if col.split("_neighbor")[0] in measured_values or col.split("_DWD")[0] in measured_values:
            ftype = "continuous"
        else:
            ftype = "categorical"

        std = sigma[col] if ftype == "continuous" else None

        feature_metadata.append({
            "name": col,
            "type": ftype,
            "std": std
        })
    
    ds_labeled = {
        "X_train": X_train,
        "y_train": y_train,
        "X_val": X_val,
        "y_val": y_val,
        "X_test": X_test,
        "y_test": y_test,
    }

    return ds_labeled, ds_unlabeled, post_train_eval_data, feature_metadata

def load_data(db_config, table_list):
    """
    Load data from the database based on the provided configuration.
    """

    print(f"[{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Loading data from {table_list[0]['start_date']} to {table_list[0]['end_date']} ...")
    # DB credentials
    user, password, host, port = [os.getenv(var) for var in ["DB_USER", "DB_PASSWORD", "DB_HOST", "DB_PORT"]]
    dbname = db_config["dbname"]
    schema = db_config["schema"]

    # DB connection
    try:
        engine = create_engine(f"postgresql+psycopg2://{user}:{password}@{host}:{port}/{dbname}")
        print("   Verbindung zur Datenbank erfolgreich hergestellt!")
    except Exception as e:
        print(f"   Fehler beim Verbinden mit der Datenbank: {e}")
        return None

    dfs = {}
    for table in table_list:
        name = table["name"]
        columns = table.get("columns", [])
        column_str = ", ".join(columns) if columns else "*"

        start = table.get("start_date")
        end = table.get("end_date")
        quell_typ = table.get("quell_typ")
        typ = table.get("typ")

        # WHERE-conditions
        conditions = []
        # time range condition for 'messzeit_utc' in table 'messungen'
        if name == "messungen" and (not columns or "messzeit_utc" in columns):
            if start and end:
                conditions.append(f"messzeit_utc BETWEEN '{start}' AND '{end}'")
            elif start:
                conditions.append(f"messzeit_utc >= '{start}'")
            elif end:
                conditions.append(f"messzeit_utc <= '{end}'")

        # Bedingung für 'quell_typ' in Tabelle 'messungen'
        if name == "messungen" and quell_typ:
            if isinstance(quell_typ, list) and len(quell_typ) > 0:
                joined = ", ".join([f"'{qt}'" for qt in quell_typ])
                conditions.append(f"quell_typ IN ({joined})")

        # exclude fport 10 in 'messungen' table
        if name == "messungen":
            conditions.append("(additional_parameters->'lora'->>'fport' IS NULL OR (additional_parameters->'lora'->>'fport')::integer <> 10)")

        # condition for 'typ' in table 'quelle'
        if name == "quelle" and typ is not None:
            if isinstance(typ, list) and len(typ) > 0:
                joined = ", ".join([f"'{t}'" for t in typ])
                conditions.append(f"typ IN ({joined})")

        # condition for 'id' in table 'location'
        if name == "location":
            conditions.append("id >= 100000 AND id < 200000")

        where_clause = f"WHERE {' AND '.join(conditions)}" if conditions else ""
        full_table_name = f"{schema}.{name}"

        query = f"SELECT {column_str} FROM {full_table_name} {where_clause};"

        try:
            dfs[name] = pd.read_sql_query(query, con=engine)
            print(f"   Tabelle '{name}' geladen: {len(dfs[name])} Zeilen")
        except Exception as e:
            print(f"   Fehler beim Laden von {name}: {e}")

    # merge
    if "quelle" in dfs and "location" in dfs:
        try:
            dfs["quelle"] = pd.merge(dfs["quelle"], dfs["location"], left_on="raster_id", right_on="id", how="left")
            dfs["quelle"] = dfs["quelle"].drop(columns=["id_y"])
            dfs["quelle"] = dfs["quelle"].rename(columns={"id_x": "id", "properties": "raster_properties"})
            print(f"   DataFrames 'quelle' und 'location' erfolgreich zusammengeführt.")
        except Exception as e:
            print(f"   Fehler beim Zusammenführen der DataFrames 'quelle' und 'location': {e}")

    # merge messungen and quelle
    if "messungen" in dfs and "quelle" in dfs:
        try:
            if dbname == "UCT":
                dfs["quelle"] = dfs["quelle"][["id", "from_date", "until_date", "location_features", "quell_properties"]]
            elif dbname == "UrbanClimateTwin":
                dfs["quelle"] = dfs["quelle"][["id", "from_date", "until_date", "quell_properties", "location_id", "raster_properties"]]
            else:
                raise ValueError(f"Unknown dbname: {dbname}")
            
            # merge based on 'quell_id' and 'id'
            merged_df = pd.merge(dfs["messungen"], dfs["quelle"], left_on="quell_id", right_on="id", how="left")
            if "from_date" in merged_df.columns and "until_date" in merged_df.columns:
                
                # Keep only rows that meet the conditions
                merged_df = merged_df[
                    (merged_df["messzeit_utc"] >= merged_df["from_date"]) &
                    (merged_df["messzeit_utc"] <= merged_df["until_date"])
                ]

            if "location_id_x" in merged_df.columns and "location_id_y" in merged_df.columns:
                assert all(merged_df["location_id_x"] == merged_df["location_id_y"]), "Mismatch between location_id_x and location_id_y after merge"
                merged_df = merged_df.drop(columns=["location_id_y"])
                merged_df = merged_df.rename(columns={"location_id_x": "location_id"})

            if dbname == "UrbanClimateTwin" and "location_center" in merged_df.columns and "location" not in merged_df.columns:
                merged_df = merged_df.rename(columns={"location_center": "location"})
            
            print(f"   DataFrames 'messungen' und 'quelle' erfolgreich zusammengeführt.")
            merged_df = merged_df.drop(columns=["id_y"])
            merged_df = merged_df.rename(columns={"id_x": "id"})
        except Exception as e:
            print(f"   Fehler beim Zusammenführen der DataFrames: {e}")

    # apply sb_filter for senseBox data
    sb_filter = table_list[0].get("sb_filter", [])
    if len(sb_filter) > 0:
        qids = merged_df['quell_id'].unique()
        keep = [qid for qid in qids if not qid.startswith("SB") or qid in sb_filter]
        merged_df = merged_df[merged_df['quell_id'].isin(keep)]
    
    print(f"   Shape: {merged_df.shape}")
    return merged_df if "messungen" in dfs and "quelle" in dfs else dfs.get("messungen", None)	