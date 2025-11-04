from fastapi import FastAPI, Query
from typing import List, Dict

app = FastAPI()

@app.get("/")
def read_root():
    return {"message": "SmartHandle AI server is running 🚴‍♀️"}

# ✅ /recommend 엔드포인트
@app.get("/recommend")
def recommend_route(
    lat: float = Query(..., description="출발지 위도"),
    lng: float = Query(..., description="출발지 경도"),
    distance: float = Query(..., description="목표 거리 (km 단위)")
):
    """
    운동 목표 거리 기반 임시 추천 경로 반환 (AI 로직은 이후 추가 예정)
    """
    # 임시 추천 경로 (나중에 실제 AI 알고리즘으로 교체 예정)
    dummy_routes = [
        {"id": 1, "name": "추천코스 A", "distance": round(distance * 0.95, 2)},
        {"id": 2, "name": "추천코스 B", "distance": round(distance * 1.03, 2)},
        {"id": 3, "name": "추천코스 C", "distance": round(distance * 1.08, 2)}
    ]

    return {
        "start": {"lat": lat, "lng": lng},
        "target_distance": distance,
        "routes": dummy_routes
    }