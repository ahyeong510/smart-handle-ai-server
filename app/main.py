from fastapi import FastAPI
from pydantic import BaseModel
import requests
import math
import os
import random
from dotenv import load_dotenv

load_dotenv()

app = FastAPI()

KAKAO_REST_API_KEY = os.getenv("KAKAO_REST_KEY")
GOOGLE_ELEVATION_API_KEY = os.getenv("GOOGLE_ELEVATION_API_KEY")

KAKAO_DIRECTIONS_URL = "https://apis-navi.kakaomobility.com/v1/directions"
GOOGLE_ELEVATION_URL = "https://maps.googleapis.com/maps/api/elevation/json"

RANDOM_SAMPLES = 12
MAX_CANDIDATES = 3
ELEV_SAMPLE_POINTS = 10


class FitnessRecommendRequest(BaseModel):
    user_id: str
    start_lat: float
    start_lng: float
    target_km: float


# ------------------ 기본 유틸 ------------------

def haversine(p1, p2):
    r = 6371000
    lat1, lon1 = map(math.radians, p1)
    lat2, lon2 = map(math.radians, p2)

    dlat = lat2 - lat1
    dlon = lon2 - lon1

    a = (
        math.sin(dlat / 2) ** 2
        + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
    )
    return 2 * r * math.asin(math.sqrt(a))


def remove_duplicate_points(points):
    unique = []
    for p in points:
        if not unique or p != unique[-1]:
            unique.append(p)
    return unique


# ------------------ 경로 생성 ------------------

def destination_point(lat, lon, bearing_deg, distance_km):
    r = 6371.0
    bearing = math.radians(bearing_deg)

    lat1 = math.radians(lat)
    lon1 = math.radians(lon)
    d = distance_km / r

    lat2 = math.asin(
        math.sin(lat1) * math.cos(d)
        + math.cos(lat1) * math.sin(d) * math.cos(bearing)
    )

    lon2 = lon1 + math.atan2(
        math.sin(bearing) * math.sin(d) * math.cos(lat1),
        math.cos(d) - math.sin(lat1) * math.sin(lat2)
    )

    return math.degrees(lat2), math.degrees(lon2)


def generate_random_destinations(lat, lon, target_km, sample_count=RANDOM_SAMPLES):
    destinations = []

    min_km = max(1.0, target_km * 0.75)
    max_km = max(min_km + 0.5, target_km * 1.25)

    for _ in range(sample_count):
        bearing = random.uniform(0, 360)
        distance_km = random.uniform(min_km, max_km)
        destinations.append(destination_point(lat, lon, bearing, distance_km))

    return destinations


# ------------------ 카카오 경로 ------------------

def get_route(start, dest):
    if not KAKAO_REST_API_KEY:
        return {}

    headers = {
        "Authorization": f"KakaoAK {KAKAO_REST_API_KEY}"
    }

    params = {
        "origin": f"{start[1]},{start[0]}",
        "destination": f"{dest[1]},{dest[0]}",
        "priority": "RECOMMEND"
    }

    try:
        response = requests.get(KAKAO_DIRECTIONS_URL, headers=headers, params=params, timeout=4)
        if response.status_code != 200:
            return {}
        return response.json()
    except:
        return {}


def extract_polyline(route_json):
    points = []

    try:
        roads = route_json["routes"][0]["sections"][0]["roads"]
        for road in roads:
            v = road["vertexes"]
            for i in range(0, len(v), 2):
                points.append((v[i+1], v[i]))
    except:
        return []

    # 🔥 중복 제거 (핵심)
    return remove_duplicate_points(points)


# ------------------ 고도 ------------------

def sample_points(points, n=ELEV_SAMPLE_POINTS):
    if len(points) <= n:
        return points

    result = []
    last = len(points) - 1

    for i in range(n):
        idx = round(i * last / (n - 1))
        result.append(points[idx])

    return result


def get_elevations(points):
    if not points or not GOOGLE_ELEVATION_API_KEY:
        return []

    locations = "|".join([f"{lat},{lng}" for lat, lng in points])

    try:
        res = requests.get(GOOGLE_ELEVATION_URL, params={
            "locations": locations,
            "key": GOOGLE_ELEVATION_API_KEY
        }, timeout=4)

        data = res.json()
        return [x["elevation"] for x in data.get("results", [])]
    except:
        return []


def analyze_route(points, elevations):
    total_ascent = 0
    max_grade = 0

    for i in range(len(points)-1):
        d = haversine(points[i], points[i+1])
        if d == 0:
            continue

        dh = elevations[i+1] - elevations[i]
        grade = abs(dh / d) * 100

        if dh > 0:
            total_ascent += dh

        max_grade = max(max_grade, grade)

    return total_ascent, max_grade


# ------------------ 후보 생성 ------------------

def build_candidate(route_json, target_km, idx):
    summary = route_json["routes"][0]["summary"]

    distance_km = round(summary["distance"] / 1000, 1)
    duration_min = int(summary["duration"] / 60)

    polyline = extract_polyline(route_json)

    if len(polyline) < 5:
        return None

    # 🔥 route_points = 샘플링 + 중복 제거
    route_points = sample_points(polyline)
    route_points = remove_duplicate_points(route_points)

    elevations = get_elevations(route_points)

    if len(elevations) == len(route_points):
        ascent, max_grade = analyze_route(route_points, elevations)
    else:
        ascent, max_grade = 0, 0

    return {
        "route_id": f"r{idx}",
        "title": f"추천 코스 {idx}",
        "distance_km": distance_km,
        "duration_min": duration_min,
        "elevation_gain": int(ascent),
        "congestion_text": "중간",
        "score": round(random.uniform(0.4, 0.9), 2),
        "route_points": [{"lat": lat, "lng": lng} for lat, lng in route_points],
        "polyline": [[lat, lng] for lat, lng in polyline],
        "max_grade_percent": round(max_grade, 1)
    }


# ------------------ API ------------------

@app.post("/fitness/recommend-loop")
def recommend_loop(req: FitnessRecommendRequest):

    start = (req.start_lat, req.start_lng)

    destinations = generate_random_destinations(
        start[0], start[1], req.target_km
    )

    candidates = []

    for i, dest in enumerate(destinations):
        route_json = get_route(start, dest)
        if not route_json:
            continue

        c = build_candidate(route_json, req.target_km, len(candidates)+1)
        if c:
            candidates.append(c)

        if len(candidates) >= MAX_CANDIDATES:
            break

    return {
        "routes": candidates,
        "count": len(candidates)
    }