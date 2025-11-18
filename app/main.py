import requests
from dotenv import load_dotenv
import os
from fastapi import FastAPI, Query, Path, HTTPException
from typing import List, Dict
from pathlib import Path as FilePath
from fastapi.responses import JSONResponse

app = FastAPI(
    title="SmartHandle API",
    description="SmartHandle 네비+진동용 백엔드",
    version="0.1.0"
)

# 👉 서버가 만든 경로를 잠깐 저장해둘 곳 (임시 메모리)
generated_routes: Dict[int, Dict] = {}
route_counter = 1  # 경로 id 증가용
# --- add: load .env and read key ---
ENV_PATH = FilePath(__file__).resolve().parent.parent / ".env"
load_dotenv(dotenv_path=ENV_PATH, override=True)

KAKAO_REST_KEY = os.getenv("KAKAO_REST_KEY")

# 확인용 출력
print("[ENV] .env exists?", ENV_PATH.exists())
print("[ENV] KEY loaded?", "YES" if KAKAO_REST_KEY else "NO")


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

# --- add: Kakao Directions call helper ---
def get_kakao_route(start_lat, start_lng, end_lat, end_lng):
    """카카오 다이렉션 REST API 호출"""
    url = "https://apis-navi.kakaomobility.com/v1/directions"
    headers = {"Authorization": f"KakaoAK {KAKAO_REST_KEY}"}
    params = {
        "origin": f"{start_lng},{start_lat}",       # 카카오는 lng,lat 순서
        "destination": f"{end_lng},{end_lat}",
        "priority": "RECOMMEND",
    }
    res = requests.get(url, headers=headers, params=params)
    if res.status_code == 200:
        return res.json()
    print("🚨 Kakao API 실패:", res.status_code, res.text)
    return None

def parse_kakao_route(route_json):
    """
    Kakao Directions 응답(JSON)에서
    - path: 좌표 리스트
    - turns: 회전 이벤트 리스트
    를 추출해서 돌려주는 함수
    """

    if not route_json or "routes" not in route_json:
        return [], []

    try:
        route = route_json["routes"][0]
        section = route["sections"][0]

        # --- path 생성 ---
        path = []
        for road in section["roads"]:
            v = road["vertexes"]
            for i in range(0, len(v), 2):
                lng = v[i]
                lat = v[i + 1]
                path.append({"lat": lat, "lng": lng})

        # 중복 제거
        cleaned_path = []
        seen = set()
        for p in path:
            key = (p["lat"], p["lng"])
            if key not in seen:
                cleaned_path.append(p)
                seen.add(key)

        # --- turns 생성 ---
        guides = section.get("guides", [])
        turns = []

        turn_map = {
            0: "straight",
            1: "right",
            2: "left",
            3: "u_turn",
            4: "roundabout",
            5: "depart",
            6: "arrive"
        }

        for idx, g in enumerate(guides, start=1):
            t = g.get("type")
            if t in turn_map:
                turns.append({
                    "seq": idx,
                    "type": turn_map[t],
                    "at": {"lat": g.get("y"), "lng": g.get("x")}
                })

        return cleaned_path, turns

    except Exception as e:
        print("💥 parse_kakao_route 오류:", repr(e))
        return [], []


@app.get("/recommend")
def recommend_route(
    lat: float = Query(..., description="출발지 위도"),
    lng: float = Query(..., description="출발지 경도"),
    distance: float = Query(..., description="목표 거리 (km 단위)")
):
    """
    목표거리 기반으로 3개 정도 코스 후보를 만들어서 내려줌.
    - Kakao Directions를 실제로 호출해서 path/turns를 만들고
    - 실패하면 더미 경로로 폴백
    - AI 추천(목표 거리와 가장 가까운 코스) 포함
    """
    try:
        print(f"[DEBUG] /recommend lat={lat}, lng={lng}, distance={distance}")
        print("[DEBUG] KEY loaded?", "YES" if KAKAO_REST_KEY else "NO")

        candidates = []
        variants = [0.95, 1.0, 1.08]
        names = ["추천코스 A", "추천코스 B", "추천코스 C"]

        global route_counter, generated_routes

        for ratio, name in zip(variants, names):
            this_id = route_counter
            route_counter += 1

            this_distance = round(distance * ratio, 2)

            # 👉 임시 목적지: 출발지에서 북동쪽으로 살짝 이동 (나중에 로직 바꿔도 됨)
            end_lat = lat + 0.01 * ratio
            end_lng = lng + 0.01 * ratio

            path = []
            turns = []
            source = "dummy"  # 기본값은 더미

            # ✅ 1) Kakao Directions 실제 호출 + 파싱
            try:
                kakao_json = get_kakao_route(lat, lng, end_lat, end_lng)
                if kakao_json:
                    path, turns = parse_kakao_route(kakao_json)
                    if path:
                        source = "kakao"
                        print(f"✅ Kakao route OK for {name}: points={len(path)}, turns={len(turns)}")
                    else:
                        print(f"❌ Kakao route parse empty for {name}, fallback to dummy")
                else:
                    print(f"❌ Kakao returned None for {name}, fallback to dummy")
            except Exception as e:
                print(f"💥 Kakao route error for {name}:", repr(e))

            # ❗ Kakao 실패 or path 비었으면 더미 경로 사용
            if not path:
                path = make_dummy_path(lat, lng, this_distance)
                turns = [
                    {"seq": 1, "type": "left",  "at": path[1]},
                    {"seq": 2, "type": "right", "at": path[3]},
                ]

            one_route = {
                "id": this_id,
                "name": name,
                "distance": this_distance,
                "start": {"lat": lat, "lng": lng},
                "path": path,
                "turns": turns,
                "source": source,
            }

            # 상세 조회용 저장 ( /route/{id} )
            generated_routes[this_id] = one_route

            # 리스트에는 요약 + 카카오/더미 정보만
            candidates.append({
                "id": this_id,
                "name": name,
                "distance": this_distance,
                "source": source,
                "path_point_count": len(path),
                "turn_count": len(turns),
            })

        # ⭐ AI 추천: 목표 거리와 가장 가까운 코스를 하나 선택
        best_route = min(candidates, key=lambda c: abs(c["distance"] - distance))

        return {
            "start": {"lat": lat, "lng": lng},
            "target_distance": distance,
            "routes": candidates,
            "recommended_route_id": best_route["id"],
            "recommended_distance": best_route["distance"],
        }

    except Exception as e:
        print("💥 /recommend 처리 중 예외:", repr(e))
        return JSONResponse(
            status_code=500,
            content={
                "error": "internal_error",
                "detail": str(e),
            },
        )



@app.get("/route/{route_id}")
def get_route_detail(
    route_id: int = Path(..., description="추천 코스 id")
):
    """
    /recommend에서 받은 id 값을 이용해서
    해당 경로의 전체 정보(path, turns 등)를 돌려주는 API.
    앱에서 유저가 코스를 선택한 뒤 이걸 호출하면 됨.
    """
    route = generated_routes.get(route_id)
    if not route:
        # 없는 id이면 404 에러
        raise HTTPException(
            status_code=404,
            detail={"error": "route_not_found", "route_id": route_id}
        )
    return route
