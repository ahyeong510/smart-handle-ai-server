from fastapi import FastAPI, Query
from pydantic import BaseModel
from typing import List, Optional
import requests
import math
import os
import random
import re
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
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
tour_detail_cache = {}


class RideHistoryItem(BaseModel):
    distanceKm: float = 0.0
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
    def make_label(completion_percent: int, satisfaction: str) -> int:
        """
        만족 여부를 0/1로 변환
        1 = 사용자가 좋아한 주행
        0 = 사용자가 별로라고 판단한 주행
        """
        feedback = compute_feedback_score(completion_percent, satisfaction)

        if feedback >= 0.7:
            return 1
        return 0


    def build_ml_dataset(rides: List[dict]):
        X = []
        y = []

        for ride in rides:
            distance = float(ride.get("distanceKm", 0.0))
            elevation = float(ride.get("elevationGain", 0))
            turn = float(ride.get("turnCount", 0))
            duration = float(ride.get("durationMin", 0))

            completion = int(ride.get("completionPercent", 0))
            satisfaction = ride.get("satisfaction", "보통")

            X.append([
                distance,
                elevation,
                turn,
                duration
            ])

            y.append(make_label(completion, satisfaction))

        return X, y


    def predict_ml_scores(candidates, rides: List[dict]):
        """
        후보 경로마다 사용자가 만족할 확률을 계산함.
        데이터가 부족하면 None 반환.
        """
        if len(rides) < 5:
            print("ML 미사용: 기록 5개 미만")
            return None

        X, y = build_ml_dataset(rides)

        if len(set(y)) < 2:
            print("ML 미사용: 만족/불만족 데이터가 한쪽만 있음")
            return None

        try:
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(np.array(X))

            model = LogisticRegression()
            model.fit(X_scaled, y)

            candidate_X = []

            for c in candidates:
                candidate_X.append([
                    float(c.get("distance_km", 0.0)),
                    float(c.get("elevation_gain", 0)),
                    float(c.get("turn_count", 0)),
                    float(c.get("duration_min", 0))
                ])

            candidate_X_scaled = scaler.transform(np.array(candidate_X))

            probabilities = model.predict_proba(candidate_X_scaled)[:, 1]

            return probabilities.tolist()

        except Exception as e:
            print("Logistic Regression 오류:", e)
            return None
    completion_score = clamp(completion_percent / 100.0)
    satisfaction_score = satisfaction_to_score(satisfaction)
    return 0.6 * completion_score + 0.4 * satisfaction_score

def make_label(completion_percent: int, satisfaction: str) -> int:
    feedback = compute_feedback_score(completion_percent, satisfaction)

    if feedback >= 0.7:
        return 1
    return 0


def build_ml_dataset(rides):
    X = []
    y = []

    for ride in rides:
        distance = float(ride.get("distanceKm", 0.0))
        elevation = float(ride.get("elevationGain", 0))
        turn = float(ride.get("turnCount", 0))
        duration = float(ride.get("durationMin", 0))

        completion = int(ride.get("completionPercent", 0))
        satisfaction = ride.get("satisfaction", "보통")

        X.append([
            distance,
            elevation,
            turn,
            duration
        ])

        y.append(make_label(completion, satisfaction))

    return X, y


def predict_ml_scores(candidates, rides):
    if len(rides) < 5:
        print("ML 미사용: 기록 5개 미만")
        return None

    X, y = build_ml_dataset(rides)

    if len(set(y)) < 2:
        print("ML 미사용: 만족/불만족 데이터 부족")
        return None

    try:
        scaler = StandardScaler()

        X_scaled = scaler.fit_transform(np.array(X))

        model = LogisticRegression()

        model.fit(X_scaled, y)

        candidate_X = []

        for c in candidates:
            candidate_X.append([
                float(c.get("distance_km", 0.0)),
                float(c.get("elevation_gain", 0)),
                float(c.get("turn_count", 0)),
                float(c.get("duration_min", 0))
            ])

        candidate_X_scaled = scaler.transform(
            np.array(candidate_X)
        )

        probabilities = model.predict_proba(
            candidate_X_scaled
        )[:, 1]

        return probabilities.tolist()

    except Exception as e:
        print("Logistic Regression 오류:", e)
        return None

def get_user_preference_from_history(rides: List[dict]):
    """
    사용자가 만족하고 완주율이 높았던 기록을 기반으로
    평균 운동 패턴을 계산한다.
    """
    if not rides:
        return None

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
        return None

    total_feedback = sum(r["feedback"] for r in good_rides)

    return {
        "target_elevation": sum(r["elevation"] * r["feedback"] for r in good_rides) / total_feedback,
        "target_turn": sum(r["turn"] * r["feedback"] for r in good_rides) / total_feedback,
        "target_duration": sum(r["duration"] * r["feedback"] for r in good_rides) / total_feedback
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
    try:
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

        item = items.get("item")

        if not item:
            print(f"TourAPI {label} item 없음:", place_name)
            return []

        if isinstance(item, dict):
            return [item]

        if isinstance(item, list):
            return [x for x in item if isinstance(x, dict)]

        print(f"TourAPI {label} item 형식 이상:", place_name, type(item))
        return []

    except Exception as e:
        print(f"TourAPI {label} 파싱 오류:", place_name, e)
        return []

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

    allowed_content_types = {"12", "14", "15", "25", "28"}

    for keyword in keywords:
        try:
            search_params = {
                "serviceKey": TOUR_API_KEY,
                "MobileOS": "ETC",
                "MobileApp": "SmartHandle",
                "_type": "json",
                "numOfRows": 10,
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

            selected_item = None

            for item in search_items:
                content_type_id = str(item.get("contenttypeid", ""))
                title = item.get("title", "")

                if content_type_id in allowed_content_types:
                    selected_item = item
                    break

            if not selected_item:
                print("TourAPI 관광지 타입 결과 없음:", keyword)
                continue

            content_id = selected_item.get("contentid")
            content_type_id = selected_item.get("contenttypeid")
            title = selected_item.get("title", "")

            if not content_id:
                print("TourAPI contentId 없음:", keyword)
                continue

            print(
                "TourAPI 검색 매칭 성공:",
                place_name,
                "=>",
                keyword,
                "/",
                title,
                "/",
                content_type_id
            )

            return content_id, content_type_id, title

        except Exception as e:
            print("TourAPI 키워드 검색 오류:", place_name, keyword, e)
            continue

    return None, None, ""

def get_fallback_tour_description(place_name):
    if (
        "광교저수지" in place_name
        or "청송못" in place_name
        or "광교호수공원" in place_name
    ):
        return (
            "광교호수공원은 수원시와 용인시에 걸쳐 있는 대형 호수공원입니다. "
            "넓은 산책로와 자전거길, 수변 경관이 조성되어 있어 시민들이 휴식과 운동을 즐기기 좋은 장소입니다."
        )

    if (
        "수원팔색길" in place_name
        or "화성성곽길" in place_name
        or "여우길" in place_name
        or "모수길" in place_name
        or "효행길" in place_name
    ):
        return (
            "수원팔색길은 수원의 자연, 역사, 문화 공간을 연결한 도보 여행길입니다. "
            "코스마다 하천, 마을길, 공원, 수원화성 주변을 지나며 수원의 다양한 풍경을 느낄 수 있습니다."
        )

    if "수원화성" in place_name or "화성행궁" in place_name:
        return (
            "수원화성은 조선 정조 때 축조된 성곽으로, 군사적 기능과 도시 계획이 결합된 대표적인 문화유산입니다. "
            "화성행궁과 성곽길 주변은 수원의 역사와 문화를 체험할 수 있는 관광지입니다."
        )

    return ""

def get_empty_tour_detail(place_name=""):
    return {
        "content_id": "",
        "content_type_id": "",
        "tour_title": place_name or "",
        "addr1": "",
        "overview": ""
    }


def get_tour_detail(place_name):
    """
    Kakao Local에서 받은 관광지명으로 TourAPI searchKeyword2를 호출해 contentId를 찾고,
    detailCommon2로 title / addr1 / overview를 가져온다.
    """
    if not place_name:
        return get_empty_tour_detail()

    if place_name in tour_detail_cache:
        return tour_detail_cache[place_name]

    result = get_empty_tour_detail(place_name)

    if not TOUR_API_KEY:
        print("TOUR_API_KEY 없음")
        result["overview"] = get_fallback_tour_description(place_name)
        tour_detail_cache[place_name] = result
        return result

    try:
        content_id, content_type_id, matched_title = search_tour_content(place_name)

        if not content_id:
            print("TourAPI 최종 검색 결과 없음:", place_name)
            result["overview"] = get_fallback_tour_description(place_name)
            tour_detail_cache[place_name] = result
            return result

        # KorService2 detailCommon2는 예전 detailCommon1 옵션
        # (defaultYN, addrinfoYN, overviewYN, contentTypeId 등)을 보내면
        # INVALID_REQUEST_PARAMETER_ERROR가 날 수 있어서 contentId 중심으로만 요청한다.
        detail_params = {
            "serviceKey": TOUR_API_KEY,
            "MobileOS": "ETC",
            "MobileApp": "SmartHandle",
            "_type": "json",
            "numOfRows": 10,
            "pageNo": 1,
            "contentId": content_id
        }

        detail_res = requests.get(
            TOUR_API_DETAIL_COMMON_URL,
            params=detail_params,
            timeout=5
        )

        print("TourAPI 상세 조회:", place_name, "=>", matched_title, detail_res.status_code)

        if detail_res.status_code != 200:
            print("TourAPI 상세 조회 실패:", detail_res.text[:300])
            result["content_id"] = str(content_id or "")
            result["content_type_id"] = str(content_type_id or "")
            result["tour_title"] = matched_title or place_name
            result["overview"] = get_fallback_tour_description(place_name)
            tour_detail_cache[place_name] = result
            return result

        try:
            detail_data = detail_res.json()
        except Exception as e:
            print("TourAPI 상세 조회 JSON 파싱 실패:", place_name, e)
            print("응답 일부:", detail_res.text[:300])
            result["content_id"] = str(content_id or "")
            result["content_type_id"] = str(content_type_id or "")
            result["tour_title"] = matched_title or place_name
            result["overview"] = get_fallback_tour_description(place_name)
            tour_detail_cache[place_name] = result
            return result

        detail_items = extract_tour_api_items(
            detail_data,
            place_name,
            "상세"
        )

        if not detail_items:
            result["content_id"] = str(content_id or "")
            result["content_type_id"] = str(content_type_id or "")
            result["tour_title"] = matched_title or place_name
            result["overview"] = get_fallback_tour_description(place_name)
            tour_detail_cache[place_name] = result
            return result

        item = detail_items[0]
        overview = clean_html(item.get("overview", ""))

        if not overview:
            overview = get_fallback_tour_description(place_name)

        result = {
            "content_id": str(item.get("contentid") or content_id or ""),
            "content_type_id": str(item.get("contenttypeid") or content_type_id or ""),
            "tour_title": clean_html(item.get("title") or matched_title or place_name),
            "addr1": clean_html(item.get("addr1") or ""),
            "overview": overview
        }

        print("TourAPI 상세 정보 성공:", place_name, "=>", result["tour_title"])
        tour_detail_cache[place_name] = result
        return result

    except Exception as e:
        print("TourAPI 상세 정보 조회 오류:", place_name, e)
        result["overview"] = get_fallback_tour_description(place_name)
        tour_detail_cache[place_name] = result
        return result


def get_tour_description(place_name):
    detail = get_tour_detail(place_name)
    return detail.get("overview", "")

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
        tour_detail = get_tour_detail(place_name)
        description = tour_detail.get("overview", "") or ""

        candidate["tour_places"].append({
            "name": place_name,
            "lat": float(p["y"]),
            "lng": float(p["x"]),
            "address": address,

            # TourAPI 정보
            "content_id": tour_detail.get("content_id", ""),
            "content_type_id": tour_detail.get("content_type_id", ""),
            "tour_title": tour_detail.get("tour_title", ""),
            "addr1": tour_detail.get("addr1", ""),

            # Android TTS에서 읽을 설명
            "description": description
        })

    return candidate


# ------------------ 점수 계산 ------------------

def similarity_score(value, target):
    """
    후보 경로 값이 사용자가 만족했던 평균 패턴과 가까울수록 높은 점수.
    target과 완전히 같으면 1.0, 많이 다르면 0에 가까워진다.
    """
    if target <= 0:
        return 1.0 if value <= 0 else 0.5

    diff_ratio = abs(value - target) / target
    return clamp(1.0 - diff_ratio)


def apply_scores(candidates, user_pattern, rides=None):
    if not candidates:
        return candidates

    if rides is None:
        rides = []

    min_elev = min(c["elevation_gain"] for c in candidates)
    max_elev = max(c["elevation_gain"] for c in candidates)

    min_turn = min(c["turn_count"] for c in candidates)
    max_turn = max(c["turn_count"] for c in candidates)

    min_dur = min(c["duration_min"] for c in candidates)
    max_dur = max(c["duration_min"] for c in candidates)

    rule_scores = []

    for c in candidates:
        elev_score = normalize_low_better(c["elevation_gain"], min_elev, max_elev)
        turn_score = normalize_low_better(c["turn_count"], min_turn, max_turn)
        dur_score = normalize_low_better(c["duration_min"], min_dur, max_dur)

        # 기록이 없으면 기본 쉬운 코스 기준
        if not user_pattern:
            rule_score = (
                0.4 * elev_score
                + 0.3 * turn_score
                + 0.3 * dur_score
            )
        else:
            # user_pattern이 있어도 기본 규칙 점수 사용
            rule_score = (
                0.4 * elev_score
                + 0.3 * turn_score
                + 0.3 * dur_score
            )

        rule_scores.append(rule_score)

    ml_scores = predict_ml_scores(candidates, rides)

    for i, c in enumerate(candidates):
        rule_score = rule_scores[i]

        if ml_scores is not None:
            ml_score = ml_scores[i]

            final_score = (
                0.5 * rule_score
                + 0.5 * ml_score
            )

            c["ml_score"] = round(ml_score, 2)
            c["rule_score"] = round(rule_score, 2)
            c["score_type"] = "logistic_regression"

        else:
            final_score = rule_score

            c["ml_score"] = None
            c["rule_score"] = round(rule_score, 2)
            c["score_type"] = "rule_based"

        c["score"] = round(final_score, 2)

    candidates.sort(key=lambda x: x["score"], reverse=True)
    return candidates

    # 기록이 있으면 사용자가 만족했던 운동 패턴과의 유사도로 추천
    for c in candidates:
        elev_score = similarity_score(
            c["elevation_gain"],
            user_pattern["target_elevation"]
        )
        turn_score = similarity_score(
            c["turn_count"],
            user_pattern["target_turn"]
        )
        dur_score = similarity_score(
            c["duration_min"],
            user_pattern["target_duration"]
        )

        final_score = (
            0.35 * elev_score
            + 0.30 * turn_score
            + 0.35 * dur_score
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


@app.get("/tour/info")
def get_tour_info(
    content_id: str = Query(..., description="TourAPI contentId"),
    content_type_id: Optional[str] = Query(None, description="TourAPI contentTypeId")
):
    """
    테스트용 API.
    contentId를 직접 넣어서 detailCommon2 연결이 되는지 확인한다.
    최종 앱에서는 /tour/recommend 응답에 설명까지 포함해서 내려준다.
    """
    if not TOUR_API_KEY:
        return {
            "success": False,
            "message": "TOUR_API_KEY 없음",
            "contentId": content_id,
            "contentTypeId": content_type_id or "",
            "title": "",
            "addr1": "",
            "overview": ""
        }

    try:
        # KorService2 detailCommon2는 예전 detailCommon1 옵션
        # (defaultYN, addrinfoYN, overviewYN, contentTypeId 등)을 보내면
        # INVALID_REQUEST_PARAMETER_ERROR가 날 수 있어서 contentId 중심으로만 요청한다.
        params = {
            "serviceKey": TOUR_API_KEY,
            "MobileOS": "ETC",
            "MobileApp": "SmartHandle",
            "_type": "json",
            "numOfRows": 10,
            "pageNo": 1,
            "contentId": content_id
        }

        res = requests.get(
            TOUR_API_DETAIL_COMMON_URL,
            params=params,
            timeout=5
        )

        print("/tour/info 상태코드:", res.status_code)
        print("/tour/info 응답 전체:", res.text[:2000])

        if res.status_code != 200:
            return {
                "success": False,
                "message": "TourAPI 호출 실패",
                "status_code": res.status_code,
                "raw": res.text[:2000],
                "contentId": content_id,
                "contentTypeId": content_type_id or "",
                "title": "",
                "addr1": "",
                "overview": ""
            }

        data = res.json()

        items = extract_tour_api_items(
            data,
            content_id,
            "tour/info"
        )

        if not items:
            return {
                "success": False,
                "message": "TourAPI 결과 없음",
                "contentId": content_id,
                "contentTypeId": content_type_id or "",
                "title": "",
                "addr1": "",
                "overview": ""
            }

        item = items[0]

        return {
            "success": True,
            "contentId": str(item.get("contentid") or content_id),
            "contentTypeId": str(item.get("contenttypeid") or content_type_id or ""),
            "title": clean_html(item.get("title", "")),
            "addr1": clean_html(item.get("addr1", "")),
            "overview": clean_html(item.get("overview", ""))
        }

    except Exception as e:
        print("/tour/info 오류:", e)
        return {
            "success": False,
            "message": str(e),
            "contentId": content_id,
            "contentTypeId": content_type_id or "",
            "title": "",
            "addr1": "",
            "overview": ""
        }

@app.get("/tour/search-test")
def tour_search_test(keyword: str = Query(..., description="검색할 관광지명")):
    """
    TourAPI searchKeyword2 테스트용.
    관광지명으로 contentId가 잡히는지 확인한다.
    """
    if not TOUR_API_KEY:
        return {
            "success": False,
            "message": "TOUR_API_KEY 없음",
            "items": []
        }

    try:
        params = {
            "serviceKey": TOUR_API_KEY,
            "MobileOS": "ETC",
            "MobileApp": "SmartHandle",
            "_type": "json",
            "numOfRows": 10,
            "pageNo": 1,
            "keyword": keyword
        }

        res = requests.get(
            TOUR_API_SEARCH_KEYWORD_URL,
            params=params,
            timeout=5
        )

        print("/tour/search-test 상태코드:", res.status_code)
        print("/tour/search-test 응답:", res.text[:500])

        if res.status_code != 200:
            return {
                "success": False,
                "message": "TourAPI 검색 호출 실패",
                "status_code": res.status_code,
                "raw": res.text[:500],
                "items": []
            }

        data = res.json()

        items = extract_tour_api_items(
            data,
            keyword,
            "tour/search-test"
        )

        result_items = []

        for item in items:
            result_items.append({
                "contentId": str(item.get("contentid", "")),
                "contentTypeId": str(item.get("contenttypeid", "")),
                "title": clean_html(item.get("title", "")),
                "addr1": clean_html(item.get("addr1", "")),
                "mapx": item.get("mapx", ""),
                "mapy": item.get("mapy", "")
            })

        return {
            "success": True,
            "keyword": keyword,
            "count": len(result_items),
            "items": result_items
        }

    except Exception as e:
        print("/tour/search-test 오류:", e)
        return {
            "success": False,
            "message": str(e),
            "items": []
        }


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
    user_pattern = get_user_preference_from_history(rides)

    candidates = apply_scores(candidates, user_pattern, rides)

    return {
        "routes": candidates,
        "count": len(candidates),
        "user_pattern": user_pattern
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