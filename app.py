from fastapi import FastAPI, HTTPException, Query, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from sklearn.neighbors import KDTree
import pandas as pd
import numpy as np
import joblib
import os
from geopy.distance import geodesic
import logging

app = FastAPI()

# 允許所有來源的跨域請求（可以根據需要進行限制）
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 可以根據需要限制來源
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 設置日誌
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

poi_data = None
trees = {}

# 構建KD-tree（不保存到磁碟）
def build_kdtrees(poi_data):
    trees = {}
    for poi_type, group in poi_data.groupby('poi_type'):
        coords = np.array(list(group['coordinates']))
        tree = KDTree(coords, leaf_size=2)
        trees[poi_type] = tree
    return trees

@app.post("/upload-poi")
async def upload_poi(file: UploadFile = File(...)):
    global poi_data, trees
    try:
        poi_data = pd.read_csv(file.file)
        poi_data['coordinates'] = list(zip(poi_data.lat, poi_data.lng))
        trees = build_kdtrees(poi_data)
        return {"message": "POI data uploaded and KD-trees built successfully"}
    except Exception as e:
        logger.error(f"An error occurred while processing the uploaded POI data: {e}")
        raise HTTPException(status_code=500, detail="An error occurred while processing the uploaded POI data")

@app.get("/poi/nearest")
def get_nearest_poi(lat: float, lng: float, poi_type: str = Query(...)):
    global poi_data, trees
    if poi_data is None or trees == {}:
        raise HTTPException(status_code=400, detail="POI data not uploaded")

    coords = np.array([[lat, lng]])

    if poi_type == "all":
        all_pois = []

        # 遍歷所有KD-tree並找出最近的POI
        for tree_poi_type, tree in trees.items():
            dist, inds = tree.query(coords, k=10)
            for i, distance in enumerate(dist[0]):
                poi_candidate = poi_data[poi_data['poi_type'] == tree_poi_type].iloc[inds[0][i]]
                candidate_distance = geodesic((lat, lng), (poi_candidate["lat"], poi_candidate["lng"])).meters
                all_pois.append({
                    "name": poi_candidate["name"],
                    "poi_type": tree_poi_type,
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
        if poi_type not in trees:
            raise HTTPException(status_code=404, detail="Model type not found")

        tree = trees[poi_type]
        dist, inds = tree.query(coords, k=10)
        nearest_pois = []

        for i, distance in enumerate(dist[0]):
            nearest_poi = poi_data[poi_data['poi_type'] == poi_type].iloc[inds[0][i]]
            distance_m = geodesic((lat, lng), (nearest_poi["lat"], nearest_poi["lng"])).meters
            nearest_pois.append({
                "name": nearest_poi["name"],
                "poi_type": poi_type,
                "distance": round(distance_m, 2),
                "latitude": nearest_poi["lat"],
                "longitude": nearest_poi["lng"]
            })

        return nearest_pois

@app.post("/clear-kdtrees")
def clear_kdtrees():
    global poi_data, trees
    poi_data = None
    trees = {}
    return {"message": "KD-trees and POI data cleared successfully"}

# 運行應用
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
