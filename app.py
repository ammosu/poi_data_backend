from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from sklearn.neighbors import KDTree
import pandas as pd
import numpy as np
import joblib
import os
from geopy.distance import geodesic

app = FastAPI()

# 允許所有來源的跨域請求（可以根據需要進行限制）
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 可以根據需要限制來源
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 讀入POI數據
poi_data = pd.read_csv('poi_all.csv')
poi_data['coordinates'] = list(zip(poi_data.lat, poi_data.lng))

# 構建並保存KD-tree
def build_and_save_kdtrees(poi_data):
    trees = {}
    for model_type, group in poi_data.groupby('model_type'):
        coords = np.array(list(group['coordinates']))
        tree = KDTree(coords, leaf_size=2)
        trees[model_type] = tree
        joblib.dump(tree, f'kdtree_{model_type}.pkl')
    return trees

# 加載KD-tree
def load_kdtrees(poi_data):
    trees = {}
    for model_type in poi_data['model_type'].unique():
        if os.path.exists(f'kdtree_{model_type}.pkl'):
            trees[model_type] = joblib.load(f'kdtree_{model_type}.pkl')
        else:
            raise FileNotFoundError(f'KD-tree for model_type {model_type} not found')
    return trees

# 如果KD-tree文件存在則加載，否則構建並保存
if all(os.path.exists(f'kdtree_{model_type}.pkl') for model_type in poi_data['model_type'].unique()):
    trees = load_kdtrees(poi_data)
else:
    trees = build_and_save_kdtrees(poi_data)

@app.get("/nearest-poi")
def get_nearest_poi(lat: float, lng: float, model_type: str = Query(...)):
    coords = np.array([[lat, lng]])

    if model_type == "all":
        all_pois = []

        # 遍歷所有KD-tree並找出最近的POI
        for tree_model_type, tree in trees.items():
            dist, inds = tree.query(coords, k=10)
            for i, distance in enumerate(dist[0]):
                poi_candidate = poi_data[poi_data['model_type'] == tree_model_type].iloc[inds[0][i]]
                candidate_distance = geodesic((lat, lng), (poi_candidate["lat"], poi_candidate["lng"])).meters
                all_pois.append({
                    "name": poi_candidate["name"],
                    "model_type": tree_model_type,
                    "distance": round(candidate_distance, 2),
                    "latitude": poi_candidate["lat"],
                    "longitude": poi_candidate["lng"]
                })

        # 排序所有POI並取前10個
        all_pois = sorted(all_pois, key=lambda x: x["distance"])[:10]

        if not all_pois:
            raise HTTPException(status_code=404, detail="No POI found")

        return all_pois
    else:
        if model_type not in trees:
            raise HTTPException(status_code=404, detail="Model type not found")

        tree = trees[model_type]
        dist, inds = tree.query(coords, k=10)
        nearest_pois = []

        for i, distance in enumerate(dist[0]):
            nearest_poi = poi_data[poi_data['model_type'] == model_type].iloc[inds[0][i]]
            distance_m = geodesic((lat, lng), (nearest_poi["lat"], nearest_poi["lng"])).meters
            nearest_pois.append({
                "name": nearest_poi["name"],
                "model_type": model_type,
                "distance": round(distance_m, 2),
                "latitude": nearest_poi["lat"],
                "longitude": nearest_poi["lng"]
            })

        return nearest_pois

# 運行應用
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=3000)
