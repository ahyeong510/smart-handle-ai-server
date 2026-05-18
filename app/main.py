from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
import requests
import math
import os
import random
import re
from dotenv import load_dotenv

load_dotenv()

app = FastAPI()

KAKAO_REST_API_KEY = os.getenv("KAKAO_REST_KEY")
GOOGLE_ELEVATION_API_KEY = os.getenv("GOOGLE_ELEVATION_API_KEY")
TOUR_API_KEY = os.getenv("TOUR_API_KEY")

KAKAO_DIRECTIONS_URL = "https://apis-navi.kakaomobility.com/v1/directions"
KAKAO_LOCAL_CATEGORY_URL = "https://dapi.kakao.com/v2/local/search/category.json"
GOOGLE_ELEVATION_URL = "https://maps.googleapis.com/maps/api/elevation/json"

TOUR_API_SEARCH_KEYWORD_URL = "https://apis.data.go.kr/B551011/KorService2/searchKeyword2"
TOUR_API_DETAIL_COMMON_URL = "https://apis.data.go.kr/B551011/KorService2/detailCommon2"

RANDOM_SAMPLES = 12
MAX_CANDIDATES = 3
ELEV_SAMPLE_POINTS = 10

tour_description_cache = {}


class RideHistoryItem(BaseModel):
    elevationGain: int
    turnCount: int
    durationMin: int
    completionPercent: int
    satisfaction: str


class FitnessRecommendRequest(BaseModel):
    user_id: str
    start_lat: float
    start_lng: float
    target_km: float
    ride_history: List[RideHistoryItem] = []


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


def clamp(value, min_value=0.0, max_value=1.0):
    return max(min_value, min(max_value, value))


def get_default_user_weights():
    return {
        "elevation": 0.4,
        "turn": 0.3,
        "duration": 0.3
    }


# ------------------ 사용자 맞춤 추천 ------------------

def satisfaction_to_score(satisfaction: str) -> float:
    if satisfaction == "만족":
        return 1.0
    elif satisfaction == "보통":
        return 0.6
    elif satisfaction == "불만족":
        return 0.2
    return 0.5


def compute_feedback_score(completion_percent: int, satisfaction: str) -> float:
    completion_score = clamp(completion_percent / 100.0)
    satisfaction_score = satisfaction_to_score(satisfaction)
    return 0.6 * completion_score + 0.4 * satisfaction_score


def get_user_preference_from_history(rides: List[dict]):
    if not rides:
        return get_default_user_weights()

    scored_rides = []
    for ride in rides:
        feedback = compute_feedback_score(
            ride.get("completionPercent", 0),
            ride.get("satisfaction", "")
        )
        scored_rides.append({
            "feedback": feedback,
            "elevation": ride.get("elevationGain", 0),
            "turn": ride.get("turnCount", 0),
            "duration": ride.get("durationMin", 0)
        })

    good_rides = [r for r in scored_rides if r["feedback"] >= 0.7]

    if not good_rides:
        good_rides = scored_rides

    avg_elev = sum(r["elevation"] for r in good_rides) / len(good_rides)
    avg_turn = sum(r["turn"] for r in good_rides) / len(good_rides)
    avg_dur = sum(r["duration"] for r in good_rides) / len(good_rides)

    elev_pref = 1.0 / (avg_elev + 1)
    turn_pref = 1.0 / (avg_turn + 1)
    dur_pref = 1.0 / (avg_dur + 1)

    total = elev_pref + turn_pref + dur_pref
    if total == 0:
        return get_default_user_weights()

    return {
        "elevation": elev_pref / total,
        "turn": turn_pref / total,
        "duration": dur_pref / total
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


# ------------------ 관광지 필터링 ------------------

def filter_tour_places(places):
    exclude_keywords = [
        "카페",
        "외식",
        "상점가",
        "골목형상점가",
        "맛집",
        "식당",
        "음식",
        "주차장",
        "화장실",
        "매점",
        "휴게소",
        "센터",
    ]

    include_keywords = [
        "공원",
        "호수",
        "저수지",
        "산",
        "성",
        "궁",
        "사찰",
        "절",
        "박물관",
        "미술관",
        "기념관",
        "전망대",
        "유적",
        "문화재",
        "생태",
        "숲",
        "둘레길",
        "산책로",
        "수목원",
        "성곽길",
        "팔색길",
    ]

    result = []

    for place in places:
        name = place.get("place_name", "")
        category = place.get("category_name", "")
        text = name + " " + category

        if any(word in text for word in exclude_keywords):
            continue

        if any(word in text for word in include_keywords):
            result.append(place)

    if len(result) >= 3:
        return result

    return places


# ------------------ 카카오 Local 관광지 검색 ------------------

def search_tour_places(lat, lng, radius_m):
    print("관광지 검색 시작:", lat, lng, radius_m)

    if not KAKAO_REST_API_KEY:
        print("KAKAO_REST_KEY 없음")
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

        filtered_places = filter_tour_places(places)

        print("필터링 후 관광지 개수:", len(filtered_places))

        for p in filtered_places:
            print("필터링 통과 관광지:", p.get("place_name"))

        return filtered_places

    except Exception as e:
        print("관광지 검색 오류:", e)
        return []


# ------------------ 한국관광공사 TourAPI 설명 조회 ------------------

def clean_html(text):
    if not text:
        return ""

    text = re.sub(r"<[^>]*>", " ", str(text))
    text = (
        text.replace("&nbsp;", " ")
        .replace("&lt;", "<")
        .replace("&gt;", ">")
        .replace("&amp;", "&")
        .replace("&quot;", '"')
    )
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def normalize_tour_api_items(items, place_name, label):
    if not items:
        print(f"TourAPI {label} 결과 없음:", place_name)
        return []

    if isinstance(items, dict):
        return [items]

    if isinstance(items, list):
        return [item for item in items if isinstance(item, dict)]

    if isinstance(items, str):
        print(f"TourAPI {label} 결과 문자열:", place_name, items[:100])
        return []

    print(f"TourAPI {label} 결과 형식 이상:", place_name, type(items))
    return []


def extract_tour_api_items(data, place_name, label):
    response = data.get("response", {})
    body = response.get("body", {})
    items = body.get("items", {})

    if not items:
        print(f"TourAPI {label} items 없음:", place_name)
        return []

    if isinstance(items, str):
        print(f"TourAPI {label} items 문자열:", place_name, items[:100])
        return []

    if not isinstance(items, dict):
        print(f"TourAPI {label} items 형식 이상:", place_name, type(items))
        return []

    raw_items = items.get("item", [])
    return normalize_tour_api_items(raw_items, place_name, label)


def make_tour_search_keywords(place_name):
    keywords = [place_name]

    if "수원팔색길" in place_name:
        keywords.append("수원 팔색길")
        keywords.append("팔색길")
        keywords.append("수원화성")
        keywords.append("수원 화성")
        keywords.append("화성행궁")

    if "화성성곽길" in place_name:
        keywords.append("수원화성")
        keywords.append("수원 화성")
        keywords.append("화성행궁")

    if "광교저수지" in place_name or "청송못" in place_name:
        keywords.append("광교호수공원")
        keywords.append("광교 호수공원")

    result = []
    for keyword in keywords:
        if keyword and keyword not in result:
            result.append(keyword)

    return result


def search_tour_content(place_name):
    keywords = make_tour_search_keywords(place_name)

    for keyword in keywords:
        try:
            search_params = {
                "serviceKey": TOUR_API_KEY,
                "MobileOS": "ETC",
                "MobileApp": "SmartHandle",
                "_type": "json",
                "numOfRows": 3,
                "pageNo": 1,
                "keyword": keyword
            }

            search_res = requests.get(
                TOUR_API_SEARCH_KEYWORD_URL,
                params=search_params,
                timeout=5
            )

            print("TourAPI 키워드 검색:", place_name, "=>", keyword, search_res.status_code)

            if search_res.status_code != 200:
                print("TourAPI 키워드 검색 실패:", search_res.text[:300])
                continue

            try:
                search_data = search_res.json()
            except Exception as e:
                print("TourAPI 키워드 검색 JSON 파싱 실패:", keyword, e)
                print("응답 일부:", search_res.text[:300])
                continue

            search_items = extract_tour_api_items(
                search_data,
                keyword,
                "검색"
            )

            if not search_items:
                continue

            first_item = search_items[0]
            content_id = first_item.get("contentid")
            content_type_id = first_item.get("contenttypeid")
            title = first_item.get("title", "")

            if not content_id:
                print("TourAPI contentId 없음:", keyword)
                continue

            print("TourAPI 검색 매칭 성공:", place_name, "=>", keyword, "/", title)

            return content_id, content_type_id, title

        except Exception as e:
            print("TourAPI 키워드 검색 오류:", place_name, keyword, e)
            continue

    return None, None, ""


def get_tour_description(place_name):
    if not place_name:
        return ""

    if place_name in tour_description_cache:
        return tour_description_cache[place_name]

    if not TOUR_API_KEY:
        print("TOUR_API_KEY 없음")
        tour_description_cache[place_name] = ""
        return ""

    try:
        content_id, content_type_id, matched_title = search_tour_content(place_name)

        if not content_id:
            print("TourAPI 최종 검색 결과 없음:", place_name)
            tour_description_cache[place_name] = ""
            return ""

        detail_params = {
            "serviceKey": TOUR_API_KEY,
            "MobileOS": "ETC",
            "MobileApp": "SmartHandle",
            "_type": "json",
            "contentId": content_id,
            "contentTypeId": content_type_id,
            "defaultYN": "Y",
            "overviewYN": "Y"
        }

        detail_res = requests.get(
            TOUR_API_DETAIL_COMMON_URL,
            params=detail_params,
            timeout=5
        )

        print("TourAPI 상세 조회:", place_name, "=>", matched_title, detail_res.status_code)

        if detail_res.status_code != 200:
            print("TourAPI 상세 조회 실패:", detail_res.text[:300])
            tour_description_cache[place_name] = ""
            return ""

        try:
            detail_data = detail_res.json()
        except Exception as e:
            print("TourAPI 상세 조회 JSON 파싱 실패:", place_name, e)
            print("응답 일부:", detail_res.text[:300])
            tour_description_cache[place_name] = ""
            return ""

        detail_items = extract_tour_api_items(
            detail_data,
            place_name,
            "상세"
        )

        if not detail_items:
            tour_description_cache[place_name] = ""
            return ""

        overview = detail_items[0].get("overview", "")
        description = clean_html(overview)

        if description:
            print("TourAPI 설명 조회 성공:", place_name, "=>", matched_title)
        else:
            print("TourAPI overview 없음:", place_name, "=>", matched_title)

        tour_description_cache[place_name] = description
        return description

    except Exception as e:
        print("TourAPI 설명 조회 오류:", place_name, e)
        tour_description_cache[place_name] = ""
        return ""


# ------------------ 카카오 경로 ------------------

def get_route(start, dest):
    if not KAKAO_REST_API_KEY:
        print("KAKAO_REST_KEY 없음")
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

        print("Kakao Directions 상태코드:", response.status_code)
        print("Kakao Directions 응답:", response.text[:300])

        if response.status_code != 200:
            return {}

        return response.json()

    except Exception as e:
        print("Kakao Directions 오류:", e)
        return {}


def get_route_with_waypoints(start, tour_places):
    if not KAKAO_REST_API_KEY:
        print("KAKAO_REST_KEY 없음")
        return {}

    if len(tour_places) < 2:
        return {}

    headers = {
        "Authorization": f"KakaoAK {KAKAO_REST_API_KEY}"
    }

    destination_place = tour_places[-1]
    destination_lat = float(destination_place["y"])
    destination_lng = float(destination_place["x"])

    waypoint_places = tour_places[:-1]

    waypoints = []
    for place in waypoint_places:
        lat = float(place["y"])
        lng = float(place["x"])
        name = place.get("place_name", "관광지")
        waypoints.append(f"{lng},{lat},name={name}")

    params = {
        "origin": f"{start[1]},{start[0]}",
        "destination": f"{destination_lng},{destination_lat},name={destination_place.get('place_name', '관광지')}",
        "waypoints": "|".join(waypoints),
        "priority": "RECOMMEND"
    }

    try:
        response = requests.get(
            KAKAO_DIRECTIONS_URL,
            headers=headers,
            params=params,
            timeout=4
        )

        print("관광 코스 Directions 상태코드:", response.status_code)
        print("관광 코스 Directions 응답:", response.text[:300])

        if response.status_code != 200:
            return {}

        return response.json()

    except Exception as e:
        print("관광 코스 Directions 오류:", e)
        return {}


def make_tour_place_groups(places, group_size=3, group_count=6):
    if len(places) < group_size:
        return []

    groups = []
    used_keys = set()

    for _ in range(group_count * 3):
        group = random.sample(places, group_size)

        key = tuple(sorted([p.get("id", p.get("place_name", "")) for p in group]))
        if key in used_keys:
            continue

        used_keys.add(key)

        group.sort(key=lambda p: int(p.get("distance", 999999)))
        groups.append(group)

        if len(groups) >= group_count:
            break

    return groups


def extract_polyline(route_json):
    points = []

    try:
        sections = route_json["routes"][0]["sections"]

        for section in sections:
            roads = section.get("roads", [])

            for road in roads:
                vertexes = road.get("vertexes", [])

                for i in range(0, len(vertexes), 2):
                    lng = vertexes[i]
                    lat = vertexes[i + 1]
                    points.append((lat, lng))

    except Exception as e:
        print("polyline 추출 오류:", e)
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

def build_candidate(route_json, idx):
    try:
        summary = route_json["routes"][0]["summary"]
    except Exception:
        return None

    distance_km = round(summary["distance"] / 1000, 1)
    duration_min = int(summary["duration"] / 60)

    polyline = extract_polyline(route_json)

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

    print("관광지 polyline 개수:", len(polyline))

    if len(polyline) < 2:
        print("polyline 부족")
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


def build_tour_course_candidate(route_json, tour_places, idx):
    candidate = build_tour_candidate(
        route_json=route_json,
        place=tour_places[-1],
        idx=idx
    )

    if not candidate:
        return None

    place_names = [p.get("place_name", "관광지") for p in tour_places]

    candidate["route_id"] = f"tour_course_{idx}"
    candidate["title"] = " → ".join(place_names)
    candidate["place_name"] = place_names[-1]
    candidate["address"] = tour_places[-1].get("road_address_name") or tour_places[-1].get("address_name", "")

    candidate["tour_places"] = []

    for p in tour_places:
        place_name = p.get("place_name", "관광지")
        address = p.get("road_address_name") or p.get("address_name", "")
        description = get_tour_description(place_name) or ""

        candidate["tour_places"].append({
            "name": place_name,
            "lat": float(p["y"]),
            "lng": float(p["x"]),
            "address": address,
            "description": description
        })

    return candidate


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
        req.start_lat,
        req.start_lng,
        req.target_km
    )

    candidates = []
    for dest in destinations:
        route_json = get_route(start, dest)
        if not route_json:
            continue

        c = build_candidate(route_json, len(candidates) + 1)
        if c:
            candidates.append(c)

        if len(candidates) >= MAX_CANDIDATES:
            break

    rides = [item.dict() for item in req.ride_history]
    user_weights = get_user_preference_from_history(rides)

    candidates = apply_scores(candidates, user_weights)

    return {
        "routes": candidates,
        "count": len(candidates),
        "user_weights": user_weights
    }


@app.post("/tour/recommend")
def recommend_tour_routes(req: TourRecommendRequest):
    start = (req.start_lat, req.start_lng)

    radius_m = max(3000, min(req.radius_m, 10000))

    print("관광지 코스 추천 요청:", req.start_lat, req.start_lng, radius_m)

    places = search_tour_places(
        lat=req.start_lat,
        lng=req.start_lng,
        radius_m=radius_m
    )

    print("검색된 관광지 개수:", len(places))

    if len(places) < 3:
        print("관광지 3개 미만이라 코스 생성 불가")
        return {
            "routes": [],
            "count": 0
        }

    place_groups = make_tour_place_groups(
        places=places,
        group_size=3,
        group_count=8
    )

    print("생성된 관광 코스 후보 그룹 수:", len(place_groups))

    candidates = []

    for group in place_groups:
        print("관광 코스 후보:", [p.get("place_name") for p in group])

        route_json = get_route_with_waypoints(
            start=start,
            tour_places=group
        )

        if not route_json:
            print("관광 코스 경로 생성 실패")
            continue

        candidate = build_tour_course_candidate(
            route_json=route_json,
            tour_places=group,
            idx=len(candidates) + 1
        )

        if candidate:
            print("관광 코스 후보 추가 성공:", candidate["title"])
            candidates.append(candidate)
        else:
            print("관광 코스 후보 생성 실패")

        if len(candidates) >= MAX_CANDIDATES:
            break

    candidates = apply_tour_scores(candidates)

    print("최종 관광 코스 추천 개수:", len(candidates))

    return {
        "routes": candidates,
        "count": len(candidates)
    }