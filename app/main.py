from fastapi import FastAPI, Query
from typing import List, Dict

app = FastAPI(
    title="SmartHandle API",
    description="SmartHandle 네비+진동용 백엔드",
    version="0.1.0"
)

# 👉 서버가 만든 경로를 잠깐 저장해둘 곳 (임시 메모리)
generated_routes: Dict[int, Dict] = {}
route_counter = 1  # 경로 id 증가용

def make_dummy_path(lat: float, lng: float, distance_km: float):
    """
    진짜 AI 경로가 생기기 전까지 쓰는 더미 경로.
    시작점에서 살짝씩 방향을 바꾼 4개의 점을 만든다고 생각하면 돼.
    """
    step = distance_km * 0.001  # 거리 비례로 위도/경도 차이 조금 주기
    return [
        {"lat": lat, "lng": lng},
        {"lat": lat + step, "lng": lng},
        {"lat": lat + step, "lng": lng + step},
        {"lat": lat, "lng": lng + step},
        {"lat": lat, "lng": lng},  # 다시 출발점으로 돌아오는 느낌
    ]


@app.get("/recommend")
def recommend_route(
    lat: float = Query(..., description="출발지 위도"),
    lng: float = Query(..., description="출발지 경도"),
    distance: float = Query(..., description="목표 거리 (km 단위)")
):
    """
    목표거리 기반으로 3개 정도 코스 후보를 만들어서 내려줌.
    나중에 진짜 AI로직으로 교체 가능.
    """
    global route_counter, generated_routes

    # 후보 3개 만들기 (거리만 살짝 다르게)
    candidates = []
    variants = [0.95, 1.0, 1.08]
    names = ["추천코스 A", "추천코스 B", "추천코스 C"]

    for ratio, name in zip(variants, names):
        this_id = route_counter
        route_counter += 1

        this_distance = round(distance * ratio, 2)
        this_path = make_dummy_path(lat, lng, this_distance)

        route_data = {
            "id": this_id,
            "name": name,
            "distance": this_distance,
            "start": {"lat": lat, "lng": lng},
            "path": this_path,           # 지도에 찍을 점들
            "turns": [                   # 나중에 ESP32 진동에 쓸 자리
                {"seq": 1, "type": "left", "at": this_path[1]},
                {"seq": 2, "type": "right", "at": this_path[3]},
            ]
        }

        # 👉 서버 메모리에 저장해두기 (사용자 선택 시 꺼내줄 거)
        generated_routes[this_id] = route_data
        candidates.append({
            "id": this_id,
            "name": name,
            "distance": this_distance
        })

    return {
        "start": {"lat": lat, "lng": lng},
        "target_distance": distance,
        "routes": candidates
    }