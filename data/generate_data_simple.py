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
    "dubno": {
        "city_name": "Dubno",
        "city_name_uk": "Дубно",
        "depot_name": "Nova Poshta depot #1",
        "depot_address": "Hrushevskoho St, 1, Dubno, Rivne Oblast",
        "depot_lat": 50.4042,
        "depot_lon": 25.7350,
        "center_lat": 50.4115,
        "center_lon": 25.7380,
        "days": 25,
        "points_range": [30, 60],
        "fixed_crew": 2,
        "supports_night": False,
        "type_probs": {
            "ATM": 0.35,
            "TT": 0.30,
            "TOBO": 0.10,
            "TC": 0.25
        },
        "bbox": {
            "lat_min": 50.350,
            "lat_max": 50.460,
            "lon_min": 25.650,
            "lon_max": 25.820
        },
        "service_radius_km": 15
    },
    "ternopil": {
        "city_name": "Ternopil",
        "city_name_uk": "Тернопіль",
        "depot_name": "Nova Poshta depot #1",
        "depot_address": "Zakhidna St, 2, Ternopil",
        "depot_lat": 49.5410,
        "depot_lon": 25.5650,
        "center_lat": 49.5535,
        "center_lon": 25.5948,
        "days": 25,
        "points_range": [
            150,
            300
        ],
        "fixed_crew": 10,
        "supports_night": True,
        "type_probs": {
            "ATM": 0.30,
            "TT": 0.35,
            "TOBO": 0.15,
            "TC": 0.20
        },
        "bbox": {
            "lat_min": 49.480,
            "lat_max": 49.620,
            "lon_min": 25.500,
            "lon_max": 25.720
        },
        "service_radius_km": 25
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


def sample_point(road_nodes):
    '''
    choose the point based on the cluster
    '''
    idx = rng.randrange(len(road_nodes))
    row = road_nodes.iloc[idx]
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

def generate(output_dir="synthetic_data_ternopil_dubno"):
    '''
    main function to generate all points for cities
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
                    lat, lon = sample_point(road_nodes)
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
