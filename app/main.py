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
ELEV_SAMPLE_POINTS = 5


class FitnessRecommendRequest(BaseModel):
    user_id: str
    start_lat: float
    start_lng: float
    target_km: float


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
        response = requests.get(
            KAKAO_DIRECTIONS_URL,
            headers=headers,
            params=params,
            timeout=4
        )
        if response.status_code != 200:
            return {}
        return response.json()
    except Exception:
        return {}


def extract_polyline(route_json):
    points = []

    try:
        routes = route_json.get("routes", [])
        if not routes:
            return []

        sections = routes[0].get("sections", [])
        if not sections:
            return []

        roads = sections[0].get("roads", [])
        for road in roads:
            vertexes = road.get("vertexes", [])
            for i in range(0, len(vertexes), 2):
                lng = vertexes[i]
                lat = vertexes[i + 1]
                points.append((lat, lng))
    except Exception:
        return []

    return points


def sample_points(points, n=ELEV_SAMPLE_POINTS):
    if not points:
        return []

    if len(points) <= n:
        return points

    if n <= 1:
        return [points[0]]

    sampled = []
    last_index = len(points) - 1

    for i in range(n):
        idx = round(i * last_index / (n - 1))
        sampled.append(points[idx])

    return sampled


def get_elevations(points):
    if not points or not GOOGLE_ELEVATION_API_KEY:
        return []

    locations = "|".join([f"{lat},{lng}" for lat, lng in points])
    params = {
        "locations": locations,
        "key": GOOGLE_ELEVATION_API_KEY
    }

    try:
        response = requests.get(GOOGLE_ELEVATION_URL, params=params, timeout=4)
        data = response.json()

        if data.get("status") != "OK":
            return []

        return [item["elevation"] for item in data.get("results", [])]
    except Exception:
        return []


def analyze_route(points, elevations):
    if len(points) < 2 or len(elevations) < 2 or len(points) != len(elevations):
        return {
            "total_ascent_m": 0.0,
            "max_grade_percent": 0.0,
            "difficulty_score": 0.0
        }

    total_ascent = 0.0
    max_grade = 0.0

    for i in range(len(points) - 1):
        distance_m = haversine(points[i], points[i + 1])
        if distance_m <= 0:
            continue

        delta_h = elevations[i + 1] - elevations[i]
        grade = abs(delta_h / distance_m) * 100

        if delta_h > 0:
            total_ascent += delta_h

        max_grade = max(max_grade, grade)

    difficulty_score = total_ascent * 0.5 + max_grade * 2

    return {
        "total_ascent_m": round(total_ascent, 1),
        "max_grade_percent": round(max_grade, 1),
        "difficulty_score": round(difficulty_score, 1)
    }


def convert_score(distance_km, target_km, difficulty_score):
    distance_gap = abs(distance_km - target_km)
    distance_penalty = min(25.0, distance_gap * 12.0)

    difficulty_penalty = min(60.0, difficulty_score * 0.6)

    raw = 100.0 - distance_penalty - difficulty_penalty
    raw = max(0.0, min(100.0, raw))

    return round(raw / 100.0, 2)


def classify_congestion(distance_km, ascent_m):
    score = distance_km * 0.7 + (ascent_m / 100.0) * 0.3

    if score < 3.0:
        return "낮음"
    elif score < 5.5:
        return "중간"
    else:
        return "높음"


def route_points_objects(polyline):
    return [{"lat": lat, "lng": lng} for lat, lng in polyline]


def build_signature(polyline, distance_km):
    if not polyline:
        return None

    end_lat, end_lng = polyline[-1]
    return (
        round(end_lat, 3),
        round(end_lng, 3),
        round(distance_km, 1)
    )


def build_candidate(route_json, target_km, seen_signatures, index):
    routes = route_json.get("routes", [])
    if not routes:
        return None

    summary = routes[0].get("summary", {})
    distance_m = summary.get("distance", 0)
    duration_sec = summary.get("duration", 0)

    distance_km = round(distance_m / 1000.0, 1)
    polyline = extract_polyline(route_json)

    if len(polyline) < 5:
        return None

    signature = build_signature(polyline, distance_km)
    if signature in seen_signatures:
        return None

    seen_signatures.add(signature)

    analysis = {
        "total_ascent_m": 0.0,
        "max_grade_percent": 0.0,
        "difficulty_score": 0.0
    }

    try:
        elev_points = sample_points(polyline, ELEV_SAMPLE_POINTS)
        elevations = get_elevations(elev_points)

        if len(elev_points) >= 2 and len(elevations) == len(elev_points):
            analysis = analyze_route(elev_points, elevations)
    except Exception:
        pass

    elevation_gain = int(round(analysis["total_ascent_m"]))
    congestion_text = classify_congestion(distance_km, elevation_gain)
    score = convert_score(
        distance_km=distance_km,
        target_km=target_km,
        difficulty_score=analysis["difficulty_score"]
    )

    return {
        "route_id": f"r{index}",
        "title": f"추천 코스 {index}",
        "distance_km": distance_km,
        "duration_min": max(1, int(round(duration_sec / 60))),
        "elevation_gain": elevation_gain,
        "congestion_text": congestion_text,
        "score": score,
        "route_points": route_points_objects(polyline),
        "polyline": [[lat, lng] for lat, lng in polyline],
        "max_grade_percent": analysis["max_grade_percent"],
        "difficulty_score": analysis["difficulty_score"]
    }


def collect_candidates(start, target_km, min_dist_km, max_dist_km, candidates, seen_signatures):
    destinations = generate_random_destinations(start[0], start[1], target_km)

    for dest in destinations:
        route_json = get_route(start, dest)
        routes = route_json.get("routes", [])
        if not routes:
            continue

        summary = routes[0].get("summary", {})
        distance_m = summary.get("distance", 0)
        distance_km = round(distance_m / 1000.0, 1)

        if not (min_dist_km <= distance_km <= max_dist_km):
            continue

        candidate = build_candidate(
            route_json=route_json,
            target_km=target_km,
            seen_signatures=seen_signatures,
            index=len(candidates) + 1
        )

        if candidate is None:
            continue

        candidates.append(candidate)

        if len(candidates) >= MAX_CANDIDATES:
            break


@app.get("/")
def root():
    return {"message": "server is running"}


@app.post("/fitness/recommend-loop")
def recommend_loop(request: FitnessRecommendRequest):
    start = (request.start_lat, request.start_lng)
    target_km = request.target_km

    candidates = []
    seen_signatures = set()

    # 1차: 기본 조건
    collect_candidates(
        start=start,
        target_km=target_km,
        min_dist_km=max(1.0, target_km * 0.7),
        max_dist_km=target_km * 1.3,
        candidates=candidates,
        seen_signatures=seen_signatures
    )

    # 2차: 후보 부족하면 조건 완화
    if len(candidates) < MAX_CANDIDATES:
        collect_candidates(
            start=start,
            target_km=target_km,
            min_dist_km=max(1.0, target_km * 0.55),
            max_dist_km=target_km * 1.45,
            candidates=candidates,
            seen_signatures=seen_signatures
        )

    # 점수 높은 순, 목표 거리와 가까운 순, 고도 낮은 순
    candidates.sort(
        key=lambda x: (
            -x["score"],
            abs(x["distance_km"] - target_km),
            x["elevation_gain"]
        )
    )

    routes_result = candidates[:MAX_CANDIDATES]

    return {
        "routes": routes_result,
        "count": len(routes_result)
    }