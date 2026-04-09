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

RANDOM_SAMPLES = 30
MAX_CANDIDATES = 3
ELEV_SAMPLE_POINTS = 10


class FitnessRecommendRequest(BaseModel):
    user_id: str
    start_lat: float
    start_lng: float
    target_km: float


def haversine(p1, p2):
    R = 6371000
    lat1, lon1 = map(math.radians, p1)
    lat2, lon2 = map(math.radians, p2)
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = math.sin(dlat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
    return 2 * R * math.asin(math.sqrt(a))


def destination_point(lat, lon, bearing, distance_km):
    R = 6371
    b = math.radians(bearing)
    lat1 = math.radians(lat)
    lon1 = math.radians(lon)
    d = distance_km / R

    lat2 = math.asin(
        math.sin(lat1) * math.cos(d) +
        math.cos(lat1) * math.sin(d) * math.cos(b)
    )
    lon2 = lon1 + math.atan2(
        math.sin(b) * math.sin(d) * math.cos(lat1),
        math.cos(d) - math.sin(lat1) * math.sin(lat2)
    )

    return math.degrees(lat2), math.degrees(lon2)


def generate_random_destinations(lat, lon, target_km):
    dests = []
    for _ in range(RANDOM_SAMPLES):
        bearing = random.uniform(0, 360)
        r = random.uniform(target_km * 0.7, target_km * 1.3)
        dests.append(destination_point(lat, lon, bearing, r))
    return dests


def get_route(start, dest):
    if not KAKAO_REST_API_KEY:
        return {}

    headers = {"Authorization": f"KakaoAK {KAKAO_REST_API_KEY}"}
    params = {
        "origin": f"{start[1]},{start[0]}",
        "destination": f"{dest[1]},{dest[0]}",
        "priority": "RECOMMEND"
    }

    try:
        res = requests.get(
            KAKAO_DIRECTIONS_URL,
            headers=headers,
            params=params,
            timeout=10
        )
        if res.status_code != 200:
            return {}
        return res.json()
    except Exception:
        return {}


def extract_polyline(route):
    points = []
    try:
        routes = route.get("routes", [])
        if not routes:
            return []

        sections = routes[0].get("sections", [])
        if not sections:
            return []

        roads = sections[0].get("roads", [])
        for r in roads:
            v = r.get("vertexes", [])
            for i in range(0, len(v), 2):
                points.append((v[i + 1], v[i]))  # (lat, lng)
    except Exception:
        return []

    return points


def sample_points(points, n=ELEV_SAMPLE_POINTS):
    if not points:
        return []

    if len(points) <= n:
        return points

    sampled = []
    last_index = len(points) - 1
    for i in range(n):
        idx = round(i * last_index / (n - 1))
        sampled.append(points[idx])

    return sampled


def get_elevations(points):
    if not points or not GOOGLE_ELEVATION_API_KEY:
        return []

    locs = "|".join([f"{lat},{lon}" for lat, lon in points])
    params = {"locations": locs, "key": GOOGLE_ELEVATION_API_KEY}

    try:
        res = requests.get(GOOGLE_ELEVATION_URL, params=params, timeout=10).json()
        if res.get("status") != "OK":
            return []
        return [r["elevation"] for r in res.get("results", [])]
    except Exception:
        return []


def analyze(points, elev):
    if len(points) < 2 or len(elev) < 2 or len(points) != len(elev):
        return {
            "total_ascent_m": 0.0,
            "max_grade_percent": 0.0,
            "difficulty_score": 0.0
        }

    ascent = 0.0
    max_grade = 0.0

    for i in range(len(points) - 1):
        d = haversine(points[i], points[i + 1])
        if d <= 0:
            continue

        dh = elev[i + 1] - elev[i]
        grade = abs(dh / d) * 100

        if dh > 0:
            ascent += dh
        max_grade = max(max_grade, grade)

    difficulty_score = ascent * 0.5 + max_grade * 2

    return {
        "total_ascent_m": round(ascent, 1),
        "max_grade_percent": round(max_grade, 1),
        "difficulty_score": round(difficulty_score, 1)
    }


def convert_score(difficulty_score):
    score = max(0.0, 100.0 - difficulty_score)
    return round(score / 100.0, 2)


def classify_congestion(distance_km):
    # 임시 규칙 기반
    if distance_km < 2.5:
        return "낮음"
    if distance_km < 4.5:
        return "중간"
    return "높음"


def route_points_objects(polyline):
    return [{"lat": lat, "lng": lng} for lat, lng in polyline]


@app.get("/")
def root():
    return {"message": "server is running"}


@app.post("/fitness/recommend-loop")
def recommend(request: FitnessRecommendRequest):
    lat = request.start_lat
    lon = request.start_lng
    target_km = request.target_km

    target_m = target_km * 1000
    min_d = target_m * 0.7
    max_d = target_m * 1.3

    routes_result = []
    seen_signatures = set()

    for dlat, dlon in generate_random_destinations(lat, lon, target_km):
        route = get_route((lat, lon), (dlat, dlon))

        routes = route.get("routes", [])
        if not routes:
            continue

        summary = routes[0].get("summary")
        if not summary:
            continue

        dist = summary.get("distance", 0)
        duration_sec = summary.get("duration", 0)

        if not (min_d <= dist <= max_d):
            continue

        poly = extract_polyline(route)
        if len(poly) < 5:
            continue

        # 중복 비슷한 경로 제거용 간단 서명
        signature = (
            round(poly[-1][0], 3),
            round(poly[-1][1], 3),
            round(dist / 100)
        )
        if signature in seen_signatures:
            continue
        seen_signatures.add(signature)

        elev_points = sample_points(poly, ELEV_SAMPLE_POINTS)
        elev_values = get_elevations(elev_points)
        analysis = analyze(elev_points, elev_values)

        elevation_gain = int(round(analysis["total_ascent_m"]))
        score = convert_score(analysis["difficulty_score"])
        congestion_text = classify_congestion(dist / 1000)

        routes_result.append({
            "route_id": f"r{len(routes_result) + 1}",
            "title": f"추천 코스 {len(routes_result) + 1}",
            "distance_km": round(dist / 1000, 1),
            "duration_min": int(duration_sec / 60),
            "elevation_gain": elevation_gain,
            "congestion_text": congestion_text,
            "score": score,

            # 앱 매핑 편하게 둘 다 제공
            "route_points": route_points_objects(poly),
            "polyline": [[p_lat, p_lng] for p_lat, p_lng in poly],

            # 디버깅/확인용
            "max_grade_percent": analysis["max_grade_percent"],
            "difficulty_score": analysis["difficulty_score"]
        })

        if len(routes_result) >= MAX_CANDIDATES:
            break

    return {"routes": routes_result}