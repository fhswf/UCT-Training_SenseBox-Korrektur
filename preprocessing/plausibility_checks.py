import numpy as np
import datetime


def plausibility_check(df, config):
    print(f"[{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Performing plausibility checks on the data ...")

    # Stage 1: Physical boundaries and faulty patterns - Flags 1/0/NaN: implausible/plausible/No check performed
    df = stage_1_check(df, config)

    # Stage 2: Temporal plausibility checks (e.g. unrealistic changes over time) - Flags 10/0/NaN: implausible/plausible/No check performed
    df = stage_2_check(df, config)

    # Stage 3: Spatial plausibility checks (e.g. neighboring senseBoxes should have similar temperature values) - Flags 100/0/NaN: implausible/plausible/No check performed
    df = stage_3_check(df, config)
    
    # Aggregate flags into a single plausiblity flag
    flag_cols = ["implausibility_flag_stage_1", "implausibility_flag_stage_2", "implausibility_flag_stage_3"]
    flag_cols = [col for col in flag_cols if col in df.columns]
    df["implausibility_flag"] = df[flag_cols].sum(axis=1)
    print(f"   Implausibility flag distribution:\n{df['implausibility_flag'].value_counts(dropna=False)}")
    
    # Apply filtering based on given flags
    if config.get("filter_implausible_samples", False):
        df = df[df["implausibility_flag"] == 0]

    # drop flag columns
    to_drop = ["median_neighbors" , "mad"] + [col for col in df.columns if col.startswith("implausibility_flag")]
    to_drop = [col for col in to_drop if col in df.columns]
    df = df.drop(columns=to_drop)

    print(f"   Shape: {df.shape}")
    return df

def stage_1_check(df, config):
    boundaries = config.get("boundaries", {})
    specific_values = config.get("specific_values", {})

    if boundaries or specific_values:
        df['implausibility_flag_stage_1'] = np.nan

    # Check for values outside of physical boundaries and set flags
    for feature, limits in boundaries.items():
        if feature in df.columns:
            min_val, max_val = limits
            df['implausibility_flag_stage_1'] = np.where((df[feature] <= min_val) | (df[feature] >= max_val), 1, 0)

    # Check for specific values and set flags
    for feature, values in specific_values.items():
        if feature in df.columns:
            df['implausibility_flag_stage_1'] = np.where(df[feature].isin(values), 1, 0)
    return df

def stage_2_check(df, config):
    df["implausibility_flag_stage_2"] = np.nan
    # TODO: Implement grouping by ID and time windows (split by gaps), compute gradients + gradient threshold 
    return df

def stage_3_check(df, config):
    neighbors = [f"temperature_value_neighbor_{idx}__OLD" for idx in range(1, 6)]
    neighbors = [col for col in neighbors if col in df.columns]
    if len(neighbors) < 3:
        print("   Not enough neighboring features for spatial plausibility check (need at least 3). Skipping stage 3.")
    else: 
        df["implausibility_flag_stage_3"] = np.nan

        # compute median of neighbors and MAD of sensor value compared to neighbors
        df['median_neighbors'] = df[neighbors].median(axis=1)
        df['mad'] = abs(df['temperature_value'] - df['median_neighbors'])

        # compute MAD threshold for stage-1-adjusted data
        mad_threshold = df[df['implausibility_flag_stage_1'] == 0]['mad'].quantile(config.get("mad_threshold_quantile", 0.995))
        print(f"   MAD threshold for spatial plausibility check (Stage 3): {mad_threshold:.2f}°C")

        # set flag based on MAD threshold
        df['implausibility_flag_stage_3'] = np.where(df['mad'] > mad_threshold, 100, 0)
    return df

def filter_inconsistent_neighbors_DWD(df, threshold):
    neighbor_cols = [col for col in df.columns if col.startswith("temperature_value_neighbor_")]
    if threshold is not None and "temperature_value_DWD001" in df.columns:
        for col in neighbor_cols:
            mask_consistency_dwd = abs(df["temperature_value_DWD001"] - df[col]) < threshold
            print(f"   Removing rows where temperature difference between DWD and {col} exceeds consistency threshold of {threshold}°C: {(~mask_consistency_dwd).sum()} rows removed")
            df = df[mask_consistency_dwd] 
    
    print(f"   Shape: {df.shape}")
    return df