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
KAKAO_LOCAL_CATEGORY_URL = "https://dapi.kakao.com/v2/local/search/category.json"

RANDOM_SAMPLES = 12
MAX_CANDIDATES = 3
ELEV_SAMPLE_POINTS = 10


class FitnessRecommendRequest(BaseModel):
    user_id: str
    start_lat: float
    start_lng: float
    target_km: float

class TourRecommendRequest(BaseModel):
    start_lat: float
    start_lng: float
    radius_m: int = 5000


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


def normalize_low_better(value, min_value, max_value):
    if max_value == min_value:
        return 1.0
    return 1.0 - ((value - min_value) / (max_value - min_value))


def get_default_user_weights():
    return {
        "elevation": 0.4,
        "turn": 0.3,
        "duration": 0.3
    }


# ------------------ 목적지 생성 ------------------

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
def search_tour_places(lat, lng, radius_m):
    print("관광지 검색 시작:", lat, lng, radius_m)

    if not KAKAO_REST_API_KEY:
        print("KAKAO_REST_API_KEY 없음")
        return []

    headers = {
        "Authorization": f"KakaoAK {KAKAO_REST_API_KEY}"
    }

    params = {
        "category_group_code": "AT4",
        "x": lng,
        "y": lat,
        "radius": radius_m,
        "sort": "distance",
        "size": 15
    }

    try:
        response = requests.get(
            KAKAO_LOCAL_CATEGORY_URL,
            headers=headers,
            params=params,
            timeout=4
        )

        print("Kakao Local 상태코드:", response.status_code)
        print("Kakao Local 응답:", response.text[:500])

        if response.status_code != 200:
            return []

        data = response.json()
        places = data.get("documents", [])

        print("검색된 관광지 개수:", len(places))

        for p in places[:5]:
            print("관광지:", p.get("place_name"), p.get("x"), p.get("y"))

        return places

    except Exception as e:
        print("관광지 검색 오류:", e)
        return []


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
        roads = route_json["routes"][0]["sections"][0]["roads"]
        for road in roads:
            v = road["vertexes"]
            for i in range(0, len(v), 2):
                points.append((v[i + 1], v[i]))
    except Exception:
        return []

    return remove_duplicate_points(points)


# ------------------ 고도 분석 ------------------

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
        res = requests.get(
            GOOGLE_ELEVATION_URL,
            params={
                "locations": locations,
                "key": GOOGLE_ELEVATION_API_KEY
            },
            timeout=4
        )

        data = res.json()
        return [x["elevation"] for x in data.get("results", [])]
    except Exception:
        return []


def analyze_route(points, elevations):
    total_ascent = 0
    max_grade = 0

    for i in range(len(points) - 1):
        d = haversine(points[i], points[i + 1])
        if d == 0:
            continue

        dh = elevations[i + 1] - elevations[i]
        grade = abs(dh / d) * 100

        if dh > 0:
            total_ascent += dh

        max_grade = max(max_grade, grade)

    return total_ascent, max_grade


# ------------------ 회전 수 계산 ------------------

def calculate_bearing(p1, p2):
    lat1, lon1 = map(math.radians, p1)
    lat2, lon2 = map(math.radians, p2)

    dlon = lon2 - lon1

    y = math.sin(dlon) * math.cos(lat2)
    x = (
        math.cos(lat1) * math.sin(lat2)
        - math.sin(lat1) * math.cos(lat2) * math.cos(dlon)
    )

    bearing = math.degrees(math.atan2(y, x))
    return (bearing + 360) % 360


def calculate_turn_count(points, threshold_deg=35):
    if len(points) < 3:
        return 0

    count = 0

    for i in range(1, len(points) - 1):
        b1 = calculate_bearing(points[i - 1], points[i])
        b2 = calculate_bearing(points[i], points[i + 1])

        diff = abs(b2 - b1)
        if diff > 180:
            diff = 360 - diff

        if diff >= threshold_deg:
            count += 1

    return count


# ------------------ 후보 생성 ------------------

def build_candidate(route_json, target_km, idx):
    try:
        summary = route_json["routes"][0]["summary"]
    except Exception:
        return None

    distance_km = round(summary["distance"] / 1000, 1)
    duration_min = int(summary["duration"] / 60)

    polyline = extract_polyline(route_json)

    if len(polyline) < 2:
        print("polyline 부족:", len(polyline))
        return None

    turn_count = calculate_turn_count(polyline)

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
        "turn_count": turn_count,
        "score": 0.0,
        "route_points": [{"lat": lat, "lng": lng} for lat, lng in route_points],
        "polyline": [[lat, lng] for lat, lng in polyline],
        "max_grade_percent": round(max_grade, 1)
    }

def build_tour_candidate(route_json, place, idx):
    try:
        summary = route_json["routes"][0]["summary"]
    except Exception:
        return None

    distance_km = round(summary["distance"] / 1000, 1)
    duration_min = int(summary["duration"] / 60)

    polyline = extract_polyline(route_json)

    print("polyline 개수:", len(polyline))

    if len(polyline) < 5:
        return None

    turn_count = calculate_turn_count(polyline)

    route_points = sample_points(polyline)
    route_points = remove_duplicate_points(route_points)

    elevations = get_elevations(route_points)

    if len(elevations) == len(route_points):
        ascent, max_grade = analyze_route(route_points, elevations)
    else:
        ascent, max_grade = 0, 0

    place_name = place.get("place_name", "관광지")
    address = place.get("road_address_name") or place.get("address_name", "")

    return {
        "route_id": f"tour_{idx}",
        "title": f"{place_name} 관광 코스",
        "place_name": place_name,
        "address": address,
        "distance_km": distance_km,
        "duration_min": duration_min,
        "elevation_gain": int(ascent),
        "turn_count": turn_count,
        "score": 0.0,
        "route_points": [{"lat": lat, "lng": lng} for lat, lng in route_points],
        "polyline": [[lat, lng] for lat, lng in polyline],
        "max_grade_percent": round(max_grade, 1)
    }

# ------------------ 점수 계산 ------------------

def apply_scores(candidates, user_weights):
    if not candidates:
        return candidates

    min_elev = min(c["elevation_gain"] for c in candidates)
    max_elev = max(c["elevation_gain"] for c in candidates)

    min_turn = min(c["turn_count"] for c in candidates)
    max_turn = max(c["turn_count"] for c in candidates)

    min_dur = min(c["duration_min"] for c in candidates)
    max_dur = max(c["duration_min"] for c in candidates)

    for c in candidates:
        elev_score = normalize_low_better(c["elevation_gain"], min_elev, max_elev)
        turn_score = normalize_low_better(c["turn_count"], min_turn, max_turn)
        dur_score = normalize_low_better(c["duration_min"], min_dur, max_dur)

        final_score = (
            user_weights["elevation"] * elev_score
            + user_weights["turn"] * turn_score
            + user_weights["duration"] * dur_score
        )

        c["score"] = round(final_score, 2)

    candidates.sort(key=lambda x: x["score"], reverse=True)
    return candidates

def apply_tour_scores(candidates):
    if not candidates:
        return candidates

    min_turn = min(c["turn_count"] for c in candidates)
    max_turn = max(c["turn_count"] for c in candidates)

    min_dur = min(c["duration_min"] for c in candidates)
    max_dur = max(c["duration_min"] for c in candidates)

    min_dist = min(c["distance_km"] for c in candidates)
    max_dist = max(c["distance_km"] for c in candidates)

    for c in candidates:
        turn_score = normalize_low_better(c["turn_count"], min_turn, max_turn)
        dur_score = normalize_low_better(c["duration_min"], min_dur, max_dur)
        dist_score = normalize_low_better(c["distance_km"], min_dist, max_dist)

        final_score = (
            0.3 * turn_score
            + 0.4 * dur_score
            + 0.3 * dist_score
        )

        c["score"] = round(final_score, 2)

    candidates.sort(key=lambda x: x["score"], reverse=True)
    return candidates

# ------------------ API ------------------

@app.get("/")
def root():
    return {"message": "SMART HANDLE AI SERVER RUNNING"}


@app.post("/fitness/recommend-loop")
def recommend_loop(req: FitnessRecommendRequest):
    start = (req.start_lat, req.start_lng)

    destinations = generate_random_destinations(
        start[0], start[1], req.target_km
    )

    candidates = []

    for dest in destinations:
        route_json = get_route(start, dest)
        if not route_json:
            continue

        c = build_candidate(route_json, req.target_km, len(candidates) + 1)
        if c:
            candidates.append(c)

        if len(candidates) >= MAX_CANDIDATES:
            break

    user_weights = get_default_user_weights()
    candidates = apply_scores(candidates, user_weights)

    return {
        "routes": candidates,
        "count": len(candidates)
    }

@app.post("/tour/recommend")
def recommend_tour_routes(req: TourRecommendRequest):
    start = (req.start_lat, req.start_lng)

    radius_m = max(3000, min(req.radius_m, 10000))

    print("관광지 추천 요청:", req.start_lat, req.start_lng, radius_m)

    places = search_tour_places(
        lat=req.start_lat,
        lng=req.start_lng,
        radius_m=radius_m
    )

    print("places 개수:", len(places))

    candidates = []

    for place in places:
        print("후보 관광지:", place.get("place_name"))

        try:
            dest_lat = float(place["y"])
            dest_lng = float(place["x"])
        except Exception as e:
            print("좌표 변환 실패:", e)
            continue

        dest = (dest_lat, dest_lng)

        route_json = get_route(start, dest)

        if not route_json:
            print("경로 생성 실패:", place.get("place_name"))
            continue

        candidate = build_tour_candidate(
            route_json=route_json,
            place=place,
            idx=len(candidates) + 1
        )

        if candidate:
            print("후보 추가 성공:", candidate["title"])
            candidates.append(candidate)
        else:
            print("후보 생성 실패:", place.get("place_name"))

        if len(candidates) >= MAX_CANDIDATES:
            break

    candidates = apply_tour_scores(candidates)

    print("최종 추천 개수:", len(candidates))

    return {
        "routes": candidates,
        "count": len(candidates)
    }