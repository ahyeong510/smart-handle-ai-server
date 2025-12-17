from fastapi import FastAPI
import requests
import math
import os
import random
from dotenv import load_dotenv

load_dotenv()

app = FastAPI()

# =========================
# 🔐 API KEYS (여기 중요!!)
# =========================
KAKAO_REST_API_KEY = os.getenv("KAKAO_REST_KEY")   # ✅ .env 이름과 일치
GOOGLE_ELEVATION_API_KEY = os.getenv("GOOGLE_ELEVATION_API_KEY")

KAKAO_DIRECTIONS_URL = "https://apis-navi.kakaomobility.com/v1/directions"
GOOGLE_ELEVATION_URL = "https://maps.googleapis.com/maps/api/elevation/json"

# =========================
# 고정 파라미터
# =========================
TOLERANCE = 0.30
RANDOM_SAMPLES = 80
MAX_CANDIDATES = 12
ELEV_SAMPLE_POINTS = 40

# =========================
# 거리 계산
# =========================
def haversine(p1, p2):
    R = 6371000
    lat1, lon1 = map(math.radians, p1)
    lat2, lon2 = map(math.radians, p2)
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = math.sin(dlat/2)**2 + math.cos(lat1)*math.cos(lat2)*math.sin(dlon/2)**2
    return 2 * R * math.asin(math.sqrt(a))

# =========================
# 목적지 계산
# =========================
def destination_point(lat, lon, bearing, distance_km):
    R = 6371
    b = math.radians(bearing)
    lat1 = math.radians(lat)
    lon1 = math.radians(lon)
    d = distance_km / R

    lat2 = math.asin(
        math.sin(lat1)*math.cos(d) +
        math.cos(lat1)*math.sin(d)*math.cos(b)
    )
    lon2 = lon1 + math.atan2(
        math.sin(b)*math.sin(d)*math.cos(lat1),
        math.cos(d)-math.sin(lat1)*math.sin(lat2)
    )

    return math.degrees(lat2), math.degrees(lon2)

# =========================
# 랜덤 목적지 생성
# =========================
def generate_random_destinations(lat, lon, target_km):
    r_min = target_km * 0.7
    r_max = target_km * 1.3
    dests = []

    for _ in range(RANDOM_SAMPLES):
        bearing = random.uniform(0, 360)
        r = random.uniform(r_min, r_max)
        dests.append(destination_point(lat, lon, bearing, r))

    return dests

# =========================
# Kakao Directions
# =========================
def get_route(start, dest):
    headers = {
        "Authorization": f"KakaoAK {KAKAO_REST_API_KEY}"
    }
    params = {
        "origin": f"{start[1]},{start[0]}",
        "destination": f"{dest[1]},{dest[0]}",
        "priority": "RECOMMEND"
    }
    return requests.get(KAKAO_DIRECTIONS_URL, headers=headers, params=params, timeout=10).json()

# =========================
# polyline
# =========================
def extract_polyline(route):
    points = []
    try:
        roads = route["routes"][0]["sections"][0]["roads"]
        for r in roads:
            v = r["vertexes"]
            for i in range(0, len(v), 2):
                points.append((v[i+1], v[i]))
    except:
        return []
    return points

# =========================
# elevation
# =========================
def get_elevations(points):
    locs = "|".join([f"{lat},{lon}" for lat, lon in points])
    params = {"locations": locs, "key": GOOGLE_ELEVATION_API_KEY}
    res = requests.get(GOOGLE_ELEVATION_URL, params=params).json()
    if res.get("status") != "OK":
        return []
    return [r["elevation"] for r in res["results"]]

# =========================
# 난이도
# =========================
def analyze(points, elev):
    ascent = 0
    max_grade = 0

    for i in range(len(points)-1):
        d = haversine(points[i], points[i+1])
        if d <= 0:
            continue
        dh = elev[i+1] - elev[i]
        grade = abs(dh/d)*100
        if dh > 0:
            ascent += dh
        max_grade = max(max_grade, grade)

    score = ascent*0.5 + max_grade*2
    return {
        "total_ascent_m": round(ascent, 1),
        "max_grade_percent": round(max_grade, 1),
        "difficulty_score": round(score, 1)
    }

# =========================
# API
# =========================
@app.post("/ai/recommend")
def recommend(lat: float, lon: float, target_km: float):

    if not KAKAO_REST_API_KEY or not GOOGLE_ELEVATION_API_KEY:
        return {
            "message": "API 키 설정 필요",
            "kakao_loaded": bool(KAKAO_REST_API_KEY),
            "google_loaded": bool(GOOGLE_ELEVATION_API_KEY)
        }

    target_m = target_km * 1000
    min_d = target_m * (1 - TOLERANCE)
    max_d = target_m * (1 + TOLERANCE)

    results = []
    for dlat, dlon in generate_random_destinations(lat, lon, target_km):
        route = get_route((lat, lon), (dlat, dlon))
        if "routes" not in route:
            continue

        summary = route["routes"][0]["summary"]
        dist = summary.get("distance", 0)
        if not (min_d <= dist <= max_d):
            continue

        poly = extract_polyline(route)
        if len(poly) < 5:
            continue

        elev = get_elevations(poly[:40])
        if len(elev) != len(poly[:40]):
            continue

        diff = analyze(poly[:40], elev)
        results.append({
            "distance_m": dist,
            **diff,
            "polyline": poly
        })

    if len(results) < 3:
        return {"message": "후보 경로 부족", "count": len(results)}

    results.sort(key=lambda x: x["difficulty_score"])

    return {
        "EASY": results[0],
        "NORMAL": results[len(results)//2],
        "HARD": results[-1]
    }