from fastapi import FastAPI, HTTPException
import math
import random
import requests

app = FastAPI()

# ======================================================
# Kakao Directions API
# ======================================================
KAKAO_REST_API_KEY = "여기에_네_카카오_REST_API_KEY"
KAKAO_DIRECTIONS_URL = "https://apis-navi.kakaomobility.com/v1/directions"

# ======================================================
# 서버 캐시 (카드 ↔ 주행 공용)
# ======================================================
ROUTE_CACHE: dict[int, dict] = {}   # route_id -> {polyline, distance_km, duration_min}


# ======================================================
# 자전거 시간 계산 (자동차 도로 + 자전거 속도)
# ======================================================
def bike_duration_min(distance_km: float) -> int:
    BIKE_SPEED_KMH = 14.0
    return int((distance_km / BIKE_SPEED_KMH) * 60)


# ======================================================
# 각도 기반 목적지 생성
# ======================================================
def destination_by_angle(lat, lng, distance_km, angle_deg):
    half = distance_km / 2.0
    delta = half / 111.0
    rad = math.radians(angle_deg)

    dlat = delta * math.cos(rad)
    dlng = delta * math.sin(rad)

    return lat + dlat, lng + dlng


# ======================================================
# Kakao Directions 호출
# ======================================================
def kakao_directions(origin_lat, origin_lng, dest_lat, dest_lng):
    headers = {
        "Authorization": f"KakaoAK {KAKAO_REST_API_KEY}"
    }
    params = {
        "origin": f"{origin_lng},{origin_lat}",
        "destination": f"{dest_lng},{dest_lat}",
        "priority": "RECOMMEND",
        "alternatives": False
    }

    res = requests.get(
        KAKAO_DIRECTIONS_URL,
        headers=headers,
        params=params,
        timeout=5
    )
    res.raise_for_status()
    return res.json()


# ======================================================
# 후보 경로 생성
# ======================================================
def generate_candidate(lat, lng, target_km, angle):
    try:
        dest_lat, dest_lng = destination_by_angle(lat, lng, target_km, angle)
        result = kakao_directions(lat, lng, dest_lat, dest_lng)

        route = result["routes"][0]
        summary = route["summary"]

        go_distance_km = summary["distance"] / 1000.0

        # polyline 추출
        polyline = []
        for section in route["sections"]:
            for road in section["roads"]:
                polyline.extend(road["vertexes"])

        if len(polyline) < 10:
            return None

        # 왕복 / 단방향
        if target_km >= 6:
            polyline = polyline + list(reversed(polyline))
            total_distance_km = go_distance_km * 2
        else:
            total_distance_km = go_distance_km

        return {
            "polyline": polyline,
            "distance_km": round(total_distance_km, 2),
            "duration_min": bike_duration_min(total_distance_km)
        }

    except Exception as e:
        print("[CANDIDATE ERROR]", e)
        return None


# ======================================================
# 추천 API (카드 + 미리보기)
# ======================================================
@app.get("/recommend")
def recommend(lat: float, lng: float, distance: float):
    angle_pool = [0, 45, 90, 135, 180, 225, 270, 315]
    random.shuffle(angle_pool)

    candidates = []

    for angle in angle_pool:
        cand = generate_candidate(lat, lng, distance, angle)
        if cand:
            score = abs(cand["distance_km"] - distance)
            cand["score"] = score
            candidates.append(cand)

        if len(candidates) >= 5:
            break

    if not candidates:
        return {"routes": []}

    candidates.sort(key=lambda x: x["score"])
    ROUTE_CACHE.clear()

    results = []
    for idx, c in enumerate(candidates[:3]):
        route_id = idx + 1

        ROUTE_CACHE[route_id] = c

        results.append({
            "id": route_id,
            "distance_km": c["distance_km"],
            "duration_min": c["duration_min"],
            "polyline": c["polyline"]   # ✅ 카드 미리보기용
        })

    return {"routes": results}


# ======================================================
# 주행 화면용 API
# ======================================================
@app.get("/route/{route_id}")
def get_route(route_id: int):
    if route_id not in ROUTE_CACHE:
        raise HTTPException(status_code=404, detail="Route not found")

    return ROUTE_CACHE[route_id]
