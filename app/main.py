from fastapi import FastAPI
from pydantic import BaseModel
import requests
import math
import os
import random
import xml.etree.ElementTree as ET
from dotenv import load_dotenv

load_dotenv()

app = FastAPI()

KAKAO_REST_API_KEY = os.getenv("KAKAO_REST_KEY")
GOOGLE_ELEVATION_API_KEY = os.getenv("GOOGLE_ELEVATION_API_KEY")
CONGESTION_API_KEY = os.getenv("CONGESTION_API_KEY")

KAKAO_DIRECTIONS_URL = "https://apis-navi.kakaomobility.com/v1/directions"
GOOGLE_ELEVATION_URL = "https://maps.googleapis.com/maps/api/elevation/json"
KAKAO_COORD2ADDRESS_URL = "https://dapi.kakao.com/v2/local/geo/coord2address.json"
GYEONGGI_CONGESTION_URL = "http://openapigw.gyeonggi.go.kr/api/rest/getRoadLinkTrafficInfo"

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


def dedupe_strings_keep_order(values):
    seen = set()
    result = []
    for v in values:
        if v and v not in seen:
            seen.add(v)
            result.append(v)
    return result


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
            vertexes = road["vertexes"]
            for i in range(0, len(vertexes), 2):
                points.append((vertexes[i + 1], vertexes[i]))
    except Exception:
        return []

    return remove_duplicate_points(points)


# ------------------ 고도 ------------------

def sample_points(points, n=ELEV_SAMPLE_POINTS):
    if not points:
        return []

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
    total_ascent = 0.0
    max_grade = 0.0

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


# ------------------ 혼잡도 ------------------

def is_in_gyeonggi(lat, lng):
    """
    좌표가 경기도인지 확인
    """
    if not KAKAO_REST_API_KEY:
        return False

    headers = {
        "Authorization": f"KakaoAK {KAKAO_REST_API_KEY}"
    }
    params = {
        "x": lng,
        "y": lat
    }

    try:
        res = requests.get(
            KAKAO_COORD2ADDRESS_URL,
            headers=headers,
            params=params,
            timeout=4
        )

        if res.status_code != 200:
            return False

        data = res.json()
        docs = data.get("documents", [])
        if not docs:
            return False

        addr = docs[0].get("address")
        if addr and "경기도" in addr.get("region_1depth_name", ""):
            return True

        return False
    except Exception:
        return False


def get_road_name(lat, lng):
    """
    좌표 -> 도로명
    """
    if not KAKAO_REST_API_KEY:
        return None

    headers = {
        "Authorization": f"KakaoAK {KAKAO_REST_API_KEY}"
    }
    params = {
        "x": lng,
        "y": lat
    }

    try:
        res = requests.get(
            KAKAO_COORD2ADDRESS_URL,
            headers=headers,
            params=params,
            timeout=4
        )
        if res.status_code != 200:
            return None

        data = res.json()
        docs = data.get("documents", [])
        if not docs:
            return None

        road_addr = docs[0].get("road_address")
        if road_addr and road_addr.get("road_name"):
            return road_addr["road_name"]

        addr = docs[0].get("address")
        if addr and addr.get("address_name"):
            return addr["address_name"]

        return None
    except Exception:
        return None


def fetch_gyeonggi_congestion_rows():
    """
    경기도 구간 소통 정보 XML -> [{'routenm': ..., 'roadrank': ...}]
    """
    if not CONGESTION_API_KEY:
        return []

    try:
        res = requests.get(
            GYEONGGI_CONGESTION_URL,
            params={"serviceKey": CONGESTION_API_KEY},
            timeout=8
        )
        if res.status_code != 200:
            return []

        root = ET.fromstring(res.text)
        rows = []

        for item in root.findall(".//itemlist"):
            routenm = item.findtext("routenm")
            roadrank = item.findtext("roadrank")

            if not routenm or not roadrank:
                continue

            try:
                rank_value = float(roadrank)
            except ValueError:
                continue

            rows.append({
                "routenm": routenm.strip(),
                "roadrank": rank_value
            })

        return rows
    except Exception:
        return []


def match_roadrank_for_route_names(route_names, congestion_rows):
    matched_ranks = []

    for route_name in route_names:
        route_name = route_name.strip()

        for row in congestion_rows:
            api_name = row["routenm"]

            if route_name == api_name or route_name in api_name or api_name in route_name:
                matched_ranks.append(row["roadrank"])

    return matched_ranks


def classify_congestion_from_rank(avg_rank):
    if avg_rank >= 100:
        return 0.8, "높음"
    elif avg_rank >= 60:
        return 0.5, "중간"
    else:
        return 0.2, "낮음"


def get_congestion_info(route_points):
    """
    1) 경기도인지 먼저 확인
    2) 경기도면 도로명 추출
    3) 경기도 구간 소통 정보와 매칭
    """
    if not route_points:
        return {
            "congestion_score": -1.0,
            "congestion_text": "정보없음",
            "matched_roads": []
        }

    # 경기도 여부 먼저 확인
    check_point = route_points[len(route_points) // 2]

    if not is_in_gyeonggi(check_point[0], check_point[1]):
        return {
            "congestion_score": -1.0,
            "congestion_text": "경기외지역",
            "matched_roads": []
        }

    if not CONGESTION_API_KEY:
        return {
            "congestion_score": -1.0,
            "congestion_text": "대기중",
            "matched_roads": []
        }

    sampled_for_congestion = sample_points(route_points, n=min(5, len(route_points)))

    route_names = []
    for lat, lng in sampled_for_congestion:
        road_name = get_road_name(lat, lng)
        if road_name:
            route_names.append(road_name)

    route_names = dedupe_strings_keep_order(route_names)

    if not route_names:
        return {
            "congestion_score": -1.0,
            "congestion_text": "도로없음",
            "matched_roads": []
        }

    congestion_rows = fetch_gyeonggi_congestion_rows()
    if not congestion_rows:
        return {
            "congestion_score": -1.0,
            "congestion_text": "API오류",
            "matched_roads": route_names
        }

    matched_ranks = match_roadrank_for_route_names(route_names, congestion_rows)

    if not matched_ranks:
        return {
            "congestion_score": -1.0,
            "congestion_text": "매칭실패",
            "matched_roads": route_names
        }

    avg_rank = sum(matched_ranks) / len(matched_ranks)
    congestion_score, congestion_text = classify_congestion_from_rank(avg_rank)

    return {
        "congestion_score": round(congestion_score, 2),
        "congestion_text": congestion_text,
        "matched_roads": route_names
    }


# ------------------ 점수 ------------------

def calc_route_score(distance_km, target_km, elevation_gain, congestion_score):
    score = 100.0

    score -= abs(distance_km - target_km) * 12.0
    score -= elevation_gain * 0.25

    if congestion_score >= 0:
        score -= congestion_score * 20.0

    score = max(0.0, min(100.0, score))
    return round(score / 100.0, 2)


# ------------------ 후보 생성 ------------------

def build_candidate(route_json, target_km, idx):
    summary = route_json["routes"][0]["summary"]

    distance_km = round(summary["distance"] / 1000, 1)
    duration_min = int(summary["duration"] / 60)

    polyline = extract_polyline(route_json)

    if len(polyline) < 5:
        return None

    route_points = sample_points(polyline)
    route_points = remove_duplicate_points(route_points)

    elevations = get_elevations(route_points)

    if len(elevations) == len(route_points):
        ascent, max_grade = analyze_route(route_points, elevations)
    else:
        ascent, max_grade = 0.0, 0.0

    elevation_gain = int(round(ascent))

    congestion = get_congestion_info(route_points)

    score = calc_route_score(
        distance_km=distance_km,
        target_km=target_km,
        elevation_gain=elevation_gain,
        congestion_score=congestion["congestion_score"]
    )

    return {
        "route_id": f"r{idx}",
        "title": f"추천 코스 {idx}",
        "distance_km": distance_km,
        "duration_min": duration_min,
        "elevation_gain": elevation_gain,
        "congestion_score": congestion["congestion_score"],
        "congestion_text": congestion["congestion_text"],
        "score": score,
        "route_points": [{"lat": lat, "lng": lng} for lat, lng in route_points],
        "polyline": [[lat, lng] for lat, lng in polyline],
        "max_grade_percent": round(max_grade, 1),
        "matched_roads": congestion["matched_roads"]
    }


# ------------------ API ------------------

@app.get("/")
def root():
    return {"message": "server is running"}


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

    return {
        "routes": candidates,
        "count": len(candidates)
    }