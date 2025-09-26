import sys
import pandas as pd
import geopandas as gpd
from shapely import wkt
from shapely.geometry import Point
from plotnine import *

sys.path.append("/usr/local/repositories/features_query/code/ncsstech_pysda")

# import pysda
import sdapoly  # , sdaprop, sdainterp

if False:
    # Champaign
    lon = -88.262429
    lat = 40.064603

    shp_path = "/usr/local/data_science_box/tmp/field_boundaries/champaign.shp"


def load_ssurgo_features():
    ssurgo_features_df = pd.read_parquet(
        "/usr/local/tmp/ssurgo_features (2).parquet"  # for mac
    )

    return ssurgo_features_df


def get_soil_features_query(geometry, ssurgo_features_df):
    # loc_gdf = create_buffer(lat, lon, buffer_m=buffer_m)
    loc_gdf = gpd.GeoDataFrame({"geometry": [geometry]}, crs="EPSG:4326")

    soils_gdf = sdapoly.gdf(loc_gdf)

    soils_gdf["mukey"] = soils_gdf["mukey"].astype(int)
    soils_gdf["geometry"] = soils_gdf["geometry"].buffer(0)
    soils_gdf["area_m2"] = soils_gdf["geometry"].to_crs(epsg=5070).area
    soils_gdf = soils_gdf.sort_values("area_m2", ascending=False).reset_index(drop=True)
    soils_gdf.drop(["geom", "geometry"], axis=1)  # Check it out
    # myaoi.plot()
    # ggplot(soils_gdf) + geom_map(aes(fill="factor(mukey)"))
    # soils_gdf[["mukey", "area_m2"]]

    # Get the largest soil, but if it is missing, it returns a following one
    for i in range(len(soils_gdf)):
        soil_features_df = ssurgo_features_df.loc[
            ssurgo_features_df["mukey"] == soils_gdf["mukey"][i]
        ].copy()
        if len(soil_features_df) > 0:
            break
    soil_features_df.reset_index(drop=True, inplace=True)
    return soil_features_df


# def get_soil_features_cached(spatial_gdf, ssurgo_features_df, src_dir=None):
#     """
#     Function that filters the soil features for a given mukey
#     """
#     src_dir = "/usr/local/tmp/gSSURGO/processed"
#     src_dir = f"{src_dir}/gSSURGO/"

#     state_abbv = spatial_gdf["state_abbv"].str.lower().values[0]
#     file_name = f"{src_dir}/gssurgo_map_{state_abbv}.gpkg"
#     soils_gdf = gpd.read_file(file_name, bbox=spatial_gdf)
#     soils_gdf.columns = soils_gdf.columns.str.lower()
#     soils_gdf = soils_gdf.explode(index_parts=True).reset_index(drop=True)
#     soils_gdf = soils_gdf.clip(spatial_gdf)

#     soils_gdf["mukey"] = soils_gdf["mukey"].astype(int)
#     soils_gdf["geometry"] = soils_gdf["geometry"].buffer(0)
#     soils_gdf["area_m2"] = soils_gdf["geometry"].to_crs(epsg=5070).area
#     soils_gdf = soils_gdf.sort_values("area_m2", ascending=False).reset_index(drop=True)

#     soil_features_df = ssurgo_features_df.loc[
#         ssurgo_features_df["mukey"] == soils_gdf["mukey"][0]
#     ].copy()
#     soil_features_df = soil_features_df.reset_index(drop=True).drop("muname", axis=1)
#     return soil_features_df


from shapely.geometry import MultiPolygon

def get_soil_features_query(geometry, ssurgo_features_df):
    if geometry.geom_type == "Polygon":
        geometry = MultiPolygon([geometry])
    geometry = geometry.buffer(0)

    loc_gdf = gpd.GeoDataFrame({"geometry": [geometry]}, crs="EPSG:4326")

    soils_gdf = sdapoly.gdf(loc_gdf)
    if soils_gdf is None or len(soils_gdf) == 0:
        return pd.DataFrame()

    soils_gdf = soils_gdf.copy()
    soils_gdf["mukey"] = soils_gdf["mukey"].astype(int)

    if "geometry" in soils_gdf.columns:
        soils_gdf["geometry"] = soils_gdf["geometry"].buffer(0)
        soils_gdf["area_m2"] = soils_gdf["geometry"].to_crs(epsg=5070).area
    else:
        soils_gdf["area_m2"] = 0.0

    agg = (
        soils_gdf.groupby("mukey", as_index=False)
        .agg(area_m2=("area_m2", "sum"))
        .sort_values("area_m2", ascending=False)
        .reset_index(drop=True)
    )
    total = float(agg["area_m2"].sum()) if len(agg) else 0.0
    agg["pct_area"] = agg["area_m2"] / total if total > 0 else 0.0

    ssurgo_features_df = ssurgo_features_df.copy()
    ssurgo_features_df["mukey"] = ssurgo_features_df["mukey"].astype(int)
    out = agg.merge(ssurgo_features_df, on="mukey", how="left")

    out = out.reset_index(drop=True)
    return out

from shapely.geometry import MultiPolygon
import pandas as pd
import geopandas as gpd
import sys

if __name__ == "__main__":
    import pandas as pd
    from shapely import wkt
    from time import perf_counter

    try:
        from tqdm.auto import tqdm
        USE_TQDM = True
    except Exception:
        USE_TQDM = False

    INPUT_CSV  = "/usr/local/tmp/counties_unique_with_state.csv"
    GEOM_COL   = "geometry_wkt"
    OUTPUT_CSV = "/usr/local/tmp/soil_features_out.csv"

    df = pd.read_csv(INPUT_CSV)
    if GEOM_COL not in df.columns:
        raise ValueError(f"Η column '{GEOM_COL}' not in CSV.")

    ssurgo_df = load_ssurgo_features()

    results = []
    n = len(df)
    start = perf_counter()

    base_iter = df.iterrows()
    if USE_TQDM:
        base_iter = tqdm(base_iter, total=n, desc="Processing polygons", unit="poly")

    for j, (idx, row) in enumerate(base_iter, 1):
        wkt_str = row[GEOM_COL]
        if pd.isna(wkt_str) or not isinstance(wkt_str, str) or not wkt_str.strip():
            print(f"[SKIP] Row {idx}: empty/no valid WKT", flush=True)
            continue
        try:
            geom = wkt.loads(wkt_str)
            if not geom.is_valid:
                geom = geom.buffer(0)
        except Exception as e:
            print(f"[SKIP] Row {idx}: error in WKT ({e})", flush=True)
            continue

        out = get_soil_features_query(geom, ssurgo_df)
        if out is None or len(out) == 0:
            if not USE_TQDM:
                print(f"[INFO] Row {idx}: δεν βρέθηκαν mukey", flush=True)
            continue

        meta = {col: row[col] for col in df.columns if col != GEOM_COL}
        out.insert(0, "src_row", idx)
        for k, v in meta.items():
            out[k] = v
        results.append(out)

        # Fallback progress when there is no tqdm
        if not USE_TQDM and (j % 10 == 0 or j == n):
            elapsed = perf_counter() - start
            rate = j / elapsed if elapsed > 0 else 0.0
            eta_s = (n - j) / rate if rate > 0 else 0.0
            print(f"[{j}/{n}] {rate:.2f} poly/s | ETA {eta_s/60:.1f} min", flush=True)

    if results:
        final = pd.concat(results, ignore_index=True)
        final.to_csv(OUTPUT_CSV, index=False)
        print(f"OK: writen {len(final)} lines in {OUTPUT_CSV}", flush=True)
    else:
        print("No results.", flush=True)
