import datetime
import geopandas as gpd
import pandas as pd
import numpy as np

from shapely import wkb

from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import MinMaxScaler


def determine_target_stations(df, use_climatological_targets, number, measured_values_DL, distance_threshold_meters=None):
    """
    Determines valid target stations for each measurement, either based on geographic proximity or climatological similarity.
    """

    print(f"[{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Merging DL stations ...")

    if use_climatological_targets:
        feature_weights = {'NDVI': 2.0}
        climatological_targets = determine_climatological_targets(df, feature_weights=feature_weights)

    df["geometry"] = df["location"].apply(lambda x: wkb.loads(x, hex=True) if isinstance(x, str) else x)
    gdf = gpd.GeoDataFrame(df, geometry="geometry", crs="25832")

    # filter DL stations
    dl_stations = gdf[gdf["quell_typ"].str.startswith("DL")].copy()
    dl_stations = dl_stations[["quell_id", "geometry", "from_date", "until_date"]].drop_duplicates()

    def find_geographic_targets(row): 
        valid_dls = dl_stations[
            (dl_stations["from_date"] <= row["messzeit_utc"]) & 
            (dl_stations["until_date"] >= row["messzeit_utc"])
        ]

        neighbors = (
            valid_dls.distance(row["geometry"])
            .sort_values()
            .head(number)
        )

        if distance_threshold_meters is not None:
            idx = neighbors.index[0]
            if neighbors[idx] >= distance_threshold_meters:
                neighbors = []

        neighbors = neighbors.index.tolist() if len(neighbors) > 0 else []
        return [dl_stations.loc[idx, "quell_id"] for idx in neighbors if idx in dl_stations.index]

    def find_climatological_targets(row):
        quell_id = row["quell_id"]
        candidates = climatological_targets[climatological_targets["quell_id"] == quell_id]

        if row['quell_typ'] != 'senseBox' or candidates.empty:
            return []

        valid_dls = dl_stations[
            (dl_stations["from_date"] <= row["messzeit_utc"]) & 
            (dl_stations["until_date"] >= row["messzeit_utc"])
        ]

        # get potential targets
        targets = []
        for cand_row in candidates.iterrows():
            if cand_row[1]["from_date"] == row["from_date"] and cand_row[1]["until_date"] == row["until_date"]:
                targets = cand_row[1]["target_ids"]

        # Filter targets to only include those that are in valid_dls
        valid_targets = valid_dls["quell_id"].tolist()
        final_targets = [t for t in targets if t in valid_targets]
        return final_targets

    if use_climatological_targets:
        gdf["dl_neighbors"] = gdf.apply(find_climatological_targets, axis=1)
    else:
        gdf["dl_neighbors"] = gdf.apply(find_geographic_targets, axis=1)

    for i in range(number):
        gdf[f"neighbor_DL_id_{i+1}"] = gdf["dl_neighbors"].apply(lambda lst: lst[i] if i < len(lst) else None)

    gdf = gdf.drop(columns=["dl_neighbors"])

    copy4merge = gdf[["quell_id", "messzeit_utc", "measurement_data"]].copy()

    # Drop DL duplicates (constant measurement values for longer time periods)
    # for decentlab quell_ids, find rows where quell_id, tag_monat_jahr and measurement_data have duplicates
    copy4merge["tag_monat_jahr"] = copy4merge["messzeit_utc"].dt.to_period("D")
    dl_rows = copy4merge[gdf["quell_typ"].str.startswith("DL")].copy()
    #print(dl_rows["tag_monat_jahr"].value_counts())
    dl_rows["measurement_data"] = dl_rows["measurement_data"].apply(lambda x: str(x))
    dl_dupes = dl_rows[dl_rows.duplicated(subset=["quell_id", "tag_monat_jahr", "measurement_data"], keep=False)]
    to_drop = dl_dupes.index
    
    # determine last occurence of each duplicate
    dl_dupes = dl_dupes.sort_values("messzeit_utc")
    last_occurences = dl_dupes[~dl_dupes.duplicated(subset=["quell_id", "tag_monat_jahr", "measurement_data"], keep="last")]

    # Also drop subsequent 3 measurements (within 30-minute time frame) after each time period with duplicates per DL-ID, as they are likely to be faulty
    subsequent_measurements = []
    for idx, row in last_occurences.iterrows():
        quell_id = row["quell_id"]
        tag_monat_jahr = row["tag_monat_jahr"]
        messzeit_utc = row["messzeit_utc"]

        next_rows = copy4merge[
            (copy4merge["quell_id"] == quell_id) &
            (copy4merge["tag_monat_jahr"] == tag_monat_jahr) &
            (copy4merge["messzeit_utc"] > messzeit_utc) &
            (copy4merge["messzeit_utc"] <= messzeit_utc + pd.Timedelta("35m"))
        ].sort_values("messzeit_utc").head(3)

        subsequent_measurements.extend(next_rows.index.tolist())

    print(f"   Dropping {len(to_drop)} duplicate DL rows and {len(subsequent_measurements)} subsequent rows within 30 minutes after duplicates.")
    #print(copy4merge.loc[to_drop, ['quell_id', 'tag_monat_jahr']].groupby(['quell_id', 'tag_monat_jahr']).nunique().index)

    # Combine duplicate and subsequent measurements to drop
    to_drop = to_drop.union(pd.Index(subsequent_measurements))
    copy4merge = copy4merge.drop(index=to_drop).reset_index(drop=True)
    copy4merge.drop(columns=["tag_monat_jahr"], inplace=True)

    for i in range(1, number + 1):
        gdf = pd.merge_asof(
            gdf.sort_values("messzeit_utc"),
            copy4merge.sort_values("messzeit_utc"),
            on="messzeit_utc",
            direction="nearest",
            tolerance=pd.Timedelta("20m"),
            left_by=f"neighbor_DL_id_{i}",
            right_by="quell_id",
            suffixes=("", f"_dl_{i}")
        )

    for i in range(1, number + 1):
        for key, subkeys in measured_values_DL.items():
            for subkey in subkeys:
                column_name = f"{key}_{subkey}_dl_{i}"
                gdf[column_name] = gdf.apply(
                    lambda row: row[f"measurement_data_dl_{i}"].get(key, {}).get(subkey)
                    if isinstance(row[f"measurement_data_dl_{i}"], dict) else None,
                    axis=1
                )
    gdf = gdf.drop(columns=[f"measurement_data_dl_{i}" for i in range(1, number + 1)])
    gdf = gdf.drop(columns=[f"neighbor_DL_id_{i}" for i in range(1, number + 1)])
    gdf = gdf.drop(columns=[f"quell_id_dl_{i}" for i in range(1, number + 1)])
    gdf = gdf.drop(columns=["geometry"])
    print(f"   Shape: {gdf.shape}")
    return gdf

def determine_climatological_targets(df, feature_weights, metric="euclidean"):
    """
    Determines climatological target stations for each senseBox measurement based on climatological similarity.
    Returns a DataFrame with the senseBox IDs and their corresponding target station IDs.
    """

    # extract unique senseBox sources from measurements df
    sensors = df[df["quell_typ"].isin(["senseBox", "DL Windsensor", "DL Wetterstation"])][["quell_id", "from_date", "until_date", "raster_properties"]].drop_duplicates(subset=["quell_id", "from_date", "until_date"])

    # drop rows with missing raster_properties
    sensors = sensors.dropna(subset=["raster_properties"]) 

    # unpack raster_properties column
    properties_df = sensors['raster_properties'].apply(pd.Series)
    sensors = pd.concat([sensors.drop(columns=['raster_properties']), properties_df], axis=1)

    # Geländeform als binäres Merkmal: {1, 2} -> {0, 1}
    sensors['GEL_FORM'] = sensors['GEL_FORM'].apply(lambda x: 1 if x == 2 else 0)

    # drop specific properties
    sensors = sensors.drop(columns=['LT_4', 'LT_14', 'PET_4', 'PET_14', 'LSTDS_TM', 'LSTNS_TM', 'KLIMATOP', 'FRISCHL', 'LAGE', 'DISTANZ', 'VEGET_D', 'LSTDS_MA', 'LSTDS_MI', 'LSTNS_MA', 'LSTNS_MI', 'LSTW1_TM', 'LSTW2_TM', 'P_D_WG_F', 'GRU_VOL_S', 'WI_G_8100', 'COOL_O', 'FT_71_00', 'ST_71_00'])

    # assign feature weights
    for col in sensors.columns:
        if col not in feature_weights and col not in ['quell_id', 'from_date', 'until_date']:
            feature_weights[col] = 1.0  # default weight if not specified

    # rescale features
    scaler = MinMaxScaler()
    scale_cols = [col for col in sensors.columns if col not in ['quell_id', 'raster_id', 'center', 'from_date', 'until_date', 'location_id']]   
    sensors_scaled = sensors.copy()
    sensors_scaled[scale_cols] = scaler.fit_transform(sensors_scaled[scale_cols])

    # split sensors into sensebox and DL 
    senseboxes_scaled = sensors_scaled[sensors_scaled['quell_id'].str.startswith('SB')].copy().reset_index(drop=True)
    decentlab_scaled = sensors_scaled[sensors_scaled['quell_id'].str.startswith('DL')].copy().reset_index(drop=True)

    # apply feature weights (sqrt for euclidean distance)
    weights = np.array([feature_weights[col] for col in scale_cols])
    decentlab_features = decentlab_scaled[scale_cols].values * np.sqrt(weights)
    sensebox_features = senseboxes_scaled[scale_cols].values * np.sqrt(weights)

    # Search DecentLab feature vectors using SenseBox feature vectors as queries
    k = len(decentlab_scaled)
    nn = NearestNeighbors(n_neighbors=k, metric=metric)
    nn.fit(decentlab_features)
    distances, indices = nn.kneighbors(sensebox_features)

    decentlab_ids = decentlab_scaled["quell_id"].values

    # Store the sorted DecentLab candidates for each SenseBox
    decentlab_neighbors = []
    for row_idx, (dist_row, idx_row) in enumerate(zip(distances, indices)):
        decentlab_neighbors.append(
            {
                "id": senseboxes_scaled.iloc[row_idx]["quell_id"],
                "neighbors": decentlab_ids[idx_row].tolist(),
                "distances": dist_row.tolist(),
            }
        )

    decentlab_neighbors_df = pd.DataFrame(decentlab_neighbors)

    # Keep only targets with overlapping dates
    final_targets = senseboxes_scaled[["quell_id", "from_date", "until_date"]].copy()
    final_targets["target_ids"] = None

    for idx, sensebox in final_targets.iterrows():
        candidates = decentlab_neighbors_df.loc[
            decentlab_neighbors_df["id"] == sensebox["quell_id"]
        ].iloc[0]

        valid_targets = []
        for target_id, distance in zip(candidates["neighbors"], candidates["distances"]):
            target = decentlab_scaled.loc[decentlab_scaled["quell_id"] == target_id].iloc[0]

            if (
                sensebox["from_date"] <= target["until_date"]
                and target["from_date"] <= sensebox["until_date"]
            ):
                valid_targets.append((target_id, distance))

        final_targets.at[idx, "target_ids"] = [target_id for target_id, _ in valid_targets[:k]]

    return final_targets