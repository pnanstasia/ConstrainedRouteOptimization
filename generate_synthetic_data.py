import random
from pathlib import Path
import pandas as pd
import numpy as np
import math
import osmnx as ox
import networkx as nx

SEED = 20260321
rng = random.Random(SEED)
np.random.seed(SEED)

ox.settings.use_cache = True
ox.settings.log_console = False

cities = {
    "kyiv": {
        "city_name": "Kyiv",
        "city_name_uk": "Київ",
        "depot_name": "Nova Poshta depot",
        "depot_address": "Velyka Okruzhna Road, 98A, Kyiv",
        "depot_lat": 50.4012,
        "depot_lon": 30.3578,
        "center_lat": 50.4547,
        "center_lon": 30.5238,
        "days": 25,
        "points_range": [
            400,
            600
        ],
        "fixed_crew": 20,
        "supports_night": True,
        "type_probs": {
            "ATM": 0.25,
            "TT": 0.44,
            "TOBO": 0.11,
            "TC": 0.2
        },
        "bbox": {
            "lat_min": 50.12,
            "lat_max": 50.72,
            "lon_min": 30.05,
            "lon_max": 30.95
        },
        "clusters": [
            [50.4505, 30.5235, 1.10],
            [50.4870, 30.3900, 0.85],
            [50.5160, 30.4980, 0.80],
            [50.4115, 30.5200, 0.80],
            [50.4500, 30.6000, 0.95],
            [50.3960, 30.6100, 0.90],
            [50.5160, 30.6150, 0.75],

            [50.5210, 30.7900, 0.55],
            [50.5550, 30.2100, 0.50],
            [50.3620, 30.2160, 0.40],
            [50.5930, 30.4950, 0.35],
            [50.3460, 30.8940, 0.35]
        ],
        "cluster_sigma_lat": 0.025,
        "cluster_sigma_lon": 0.034,
        "service_radius_km": 35,
        "uniform_share": 0.12
    },
    "varash": {
        "city_name": "Varash",
        "city_name_uk": "Вараш",
        "depot_name": "Nova Poshta depot",
        "depot_address": "Enerhetykiv St, 5, Varash, Rivne Oblast",
        "depot_lat": 51.3489,
        "depot_lon": 25.8472,
        "center_lat": 51.3482,
        "center_lon": 25.8501,
        "days": 25,
        "points_range": [
            40,
            70
        ],
        "fixed_crew": 2,
        "supports_night": False,
        "type_probs": {
            "ATM": 0.3,
            "TT": 0.21,
            "TOBO": 0.13,
            "TC": 0.36
        },
        "bbox": {
            "lat_min": 51.071,
            "lat_max": 51.611,
            "lon_min": 25.419,
            "lon_max": 26.281
        },
        "clusters": [
            [51.3410, 25.8500, 1.00], 
            [51.3460, 25.8420, 0.80],
            [51.3340, 25.8550, 0.70],
            [51.4320, 26.1210, 0.55], 
            [51.2720, 25.9260, 0.45],
            [51.3000, 25.8450, 0.35],
            [51.4780, 25.7450, 0.30],
            [51.1560, 25.8480, 0.40],
            [51.2950, 25.5940, 0.35]
        ],
        "cluster_sigma_lat": 0.015, 
        "cluster_sigma_lon": 0.022,
        "service_radius_km": 20,
        "uniform_share": 0.12
    }
}

night_prob = {'ATM': 0.26, 'TT': 0.08, 'TOBO': 0.34, 'TC': 0.32}
tw_prob = {'TT': 0.3}

def weighted_choice(prob_map, r):
    '''
    find the type of the point
    '''
    keys = list(prob_map.keys())
    probs = np.array(list(prob_map.values()), dtype=float)
    idx = np.searchsorted(np.cumsum(probs), r.random())
    return keys[min(idx, len(keys)-1)]


def prepare_road_nodes(city_cfg):
    '''
    search of the nodes for the city based on the graph and taking into account boundaries and service zone
    '''
    center = (city_cfg["center_lat"], city_cfg["center_lon"])
    dist_m = int((city_cfg["service_radius_km"] + 5) * 1000)

    G = ox.graph_from_point(
        center,
        dist=dist_m,
        network_type="drive",
        simplify=True
    )

    largest_cc = max(nx.weakly_connected_components(G), key=len)
    G = G.subgraph(largest_cc).copy()

    nodes = ox.graph_to_gdfs(G, edges=False)
    nodes = nodes.rename(columns={"y": "lan", "x": "lot"}).reset_index(drop=True)

    bbox = city_cfg["bbox"]
    mask_bbox = (
        (nodes["lan"] >= bbox["lat_min"]) &
        (nodes["lan"] <= bbox["lat_max"]) &
        (nodes["lot"] >= bbox["lon_min"]) &
        (nodes["lot"] <= bbox["lon_max"])
    )

    lat_km = (nodes["lan"] - city_cfg["center_lat"]) * 111.0
    lon_km = (
        (nodes["lot"] - city_cfg["center_lon"]) *
        111.0 *
        math.cos(math.radians(city_cfg["center_lat"]))
    )
    mask_radius = (lat_km**2 + lon_km**2) <= city_cfg["service_radius_km"]**2

    nodes = nodes[mask_bbox & mask_radius].copy()

    if nodes.empty:
        raise ValueError(f"No drivable road nodes found for {city_cfg['city_name']}.")

    return nodes[["lan", "lot"]].reset_index(drop=True)


def sample_point(city_cfg, road_nodes):
    '''
    choose the point based on the cluster
    '''
    if rng.random() < city_cfg["uniform_share"]:
        idx = rng.randrange(len(road_nodes))
        row = road_nodes.iloc[idx]
        return float(row["lan"]), float(row["lot"])

    clusters = city_cfg["clusters"]
    weights = np.array([w for _, _, w in clusters], dtype=float)
    weights /= weights.sum()

    k = int(np.searchsorted(np.cumsum(weights), rng.random()))
    clat, clon, _ = clusters[min(k, len(clusters) - 1)]

    sigma_lat_km = city_cfg["cluster_sigma_lat"] * 111.0
    sigma_lon_km = (
        city_cfg["cluster_sigma_lon"] *
        111.0 *
        math.cos(math.radians(clat))
    )

    lat_km = (road_nodes["lan"].to_numpy() - clat) * 111.0
    lon_km = (
        (road_nodes["lot"].to_numpy() - clon) *
        111.0 *
        math.cos(math.radians(clat))
    )
    d2 = (lat_km / sigma_lat_km) ** 2 + (lon_km / sigma_lon_km) ** 2
    probs = np.exp(-0.5 * d2)

    if probs.sum() == 0:
        idx = rng.randrange(len(road_nodes))
    else:
        probs = probs / probs.sum()
        idx = np.random.choice(len(road_nodes), p=probs)

    row = road_nodes.iloc[int(idx)]
    return float(row["lan"]), float(row["lot"])

def sample_time_window(ptype, is_night):
    '''
    setting time boundaries
    '''
    if is_night:
        return ("18:00:00", "23:59:59")

    if ptype != "TT":
        return "", ""

    if rng.random() >= tw_prob["TT"]:
        return "", ""

    options = [
        ("08:00:00", "18:00:00"),("08:30:00", "14:30:00"), ("08:30:00", "16:30:00"),
        ("09:00:00", "11:00:00"),("09:00:00", "13:00:00"),("09:00:00", "15:00:00"),
        ("09:00:00", "16:00:00"),("09:00:00", "17:00:00"),("09:00:00", "18:00:00"),
        ("09:30:00", "11:30:00"),("09:30:00", "15:30:00"),("09:30:00", "16:00:00"),
        ("09:30:00", "17:00:00"),("10:00:00", "12:00:00"),("10:00:00", "13:00:00"),
        ("10:00:00", "14:00:00"),("10:00:00", "15:00:00"),("10:00:00", "16:00:00"),
        ("10:00:00", "17:00:00"),("10:00:00", "18:00:00"),("10:00:00", "19:00:00"),
        ("10:00:00", "20:00:00"),("10:30:00", "16:00:00"),("10:30:00", "16:30:00"),
        ("11:00:00", "14:00:00"),("11:00:00", "15:00:00"),("11:00:00", "16:00:00"),
        ("11:00:00", "17:00:00"),("11:00:00", "18:00:00"),("11:00:00", "19:00:00"),
        ("11:30:00", "14:30:00"),("11:30:00", "17:30:00"),("12:00:00", "16:00:00"),
        ("12:00:00", "17:00:00"),("12:00:00", "18:00:00"),("12:00:00", "19:00:00"),
        ("13:00:00", "16:00:00"),("13:00:00", "16:30:00"),("13:00:00", "17:00:00"),
        ("13:00:00", "18:00:00"),("13:00:00", "19:00:00"),("13:00:00", "20:00:00"),
        ("13:30:00", "16:00:00"),("13:30:00", "16:30:00"),("13:30:00", "17:00:00"),
        ("14:00:00", "16:00:00"),("14:00:00", "16:30:00"),("14:00:00", "17:00:00"),
        ("14:00:00", "18:00:00"),("14:00:00", "19:00:00"),("14:00:00", "20:00:00"),
        ("14:30:00", "17:30:00"),("15:00:00", "17:00:00"),("15:00:00", "17:30:00"),
        ("15:00:00", "18:00:00"),("15:00:00", "19:00:00"),("15:30:00", "18:30:00"),
        ("16:00:00", "19:00:00"),("17:00:00", "20:00:00"),("17:00:00", "20:50:00"),
        ("19:00:00", "21:00:00")
    ]
    return options[rng.randrange(len(options))]

def generate(output_dir="synthetic_data_kyiv_varash"):
    '''
    min function to generate all points for cities
    '''
    base = Path(output_dir)
    base.mkdir(parents=True, exist_ok=True)
    for key in cities:
        (base / key).mkdir(exist_ok=True)

    road_nodes_by_city = {
        city_key: prepare_road_nodes(cfg)
        for city_key, cfg in cities.items()
    }

    summary_rows = []

    for city_key, cfg in cities.items():
        city_name = city_key.upper()
        road_nodes = road_nodes_by_city[city_key]

        for day in range(1, cfg["days"] + 1):
            n = rng.randint(*cfg["points_range"])
            rows = []
            used_coords = set()

            for i in range(1, n + 1):
                ptype = weighted_choice(cfg["type_probs"], rng)

                is_night = (
                    cfg["supports_night"]
                    and (rng.random() < night_prob.get(ptype))
                )
                tw_start, tw_end = sample_time_window(ptype, is_night)

                for _ in range(100):
                    lat, lon = sample_point(cfg, road_nodes)
                    coord_key = (round(lat, 6), round(lon, 6))
                    if coord_key not in used_coords:
                        used_coords.add(coord_key)
                        break

                rows.append({
                    "point_id": f"{city_name}_D{day:02d}_{i:04d}",
                    "point_type": ptype,
                    "lot": round(lon, 6),
                    "lan": round(lat, 6),
                    "night": "так" if is_night else "ні",
                    "tw_start": tw_start,
                    "tw_end": tw_end,
                })

            df = pd.DataFrame(rows)
            out_path = base / city_key / f"{city_key}_day_{day:02d}.csv"
            df.to_csv(out_path, index=False, encoding="utf-8-sig")

        summary_rows.append({
            "city_code": city_key,
            "city_name": cfg["city_name"],
            "fixed_crew": cfg["fixed_crew"],
            "depot_name": cfg["depot_name"],
            "depot_address": cfg["depot_address"],
            "depot_lot": cfg["depot_lon"],
            "depot_lan": cfg["depot_lat"],
        })

    pd.DataFrame(summary_rows).to_csv(
        base / "general.csv",
        index=False,
        encoding="utf-8-sig"
    )

if __name__ == "__main__":
    generate()
