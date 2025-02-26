import time
import pandas as pd
import json
import uuid
import openai
import requests
import re
import os
import numpy as np
import os
from dotenv import load_dotenv
load_dotenv(r"C:\Users\senbo\Desktop\taba_project\.env")  # .env 파일 로드
api_key = os.getenv("OPENAI_API_KEY")
client_id=os.getenv("client_id")
client_secret=os.getenv("client_secret")
MODEL = "gpt-4o"
client = openai.OpenAI(api_key=api_key)
secret_key=os.getenv("secret_key")
api_url=os.getenv("api_url")
def remove_bounding_boxes(data):
    """Bounding Box 값을 제거하고 저장하는 함수"""
    bounding_boxes = {}
    
    def traverse(node, path=""):
        if isinstance(node, dict):
            if "bounding_box" in node:
                bounding_boxes[path] = node["bounding_box"]
                del node["bounding_box"]
            for key, value in node.items():
                traverse(value, f"{path}.{key}" if path else key)
        elif isinstance(node, list):
            for idx, item in enumerate(node):
                traverse(item, f"{path}[{idx}]")

    traverse(data)
    return bounding_boxes

def restore_bounding_boxes(data, bounding_boxes):
    """저장된 Bounding Box 값을 복원하는 함수"""
    def traverse(node, path=""):
        if isinstance(node, dict):
            for key, value in node.items():
                traverse(value, f"{path}.{key}" if path else key)
            if path in bounding_boxes:
                node["bounding_box"] = bounding_boxes[path]
        elif isinstance(node, list):
            for idx, item in enumerate(node):
                traverse(item, f"{path}[{idx}]")
    traverse(data)

def process_all_json(input_dir):
    try:
        # 파일 경로 설정
        files = {
            "coai": os.path.join(input_dir, "coai_result_a.json"),
            "ledger": os.path.join(input_dir, "ledger_result.json"),
            "reg": os.path.join(input_dir, "reg_result.json")
        }

        with open(files["coai"], 'r', encoding='utf-8') as f:
            coai_data = json.load(f)
        with open(files["ledger"], 'r', encoding='utf-8') as f:
            ledger_data = {"page1": json.load(f)}
        with open(files["reg"], 'r', encoding='utf-8') as f:
            reg_data = json.load(f)

        # 데이터 통합
        merged_data = {
            "contract": coai_data,
            "building_registry": ledger_data,
            "registry_document": reg_data
        }

        # 1단계: 소유자 수 조정
        name_count = sum(1 for key in ledger_data["page1"].keys() if key.startswith("성명"))
        owners = []
        for page_key, page_content in reg_data.items():
            if not isinstance(page_content, dict):
                continue
            
            for key, value in page_content.items():
                if key.startswith("소유자"):
                    owner_info = {
                        "page": page_key,
                        "key": key,
                        "y1": value["bounding_box"]["y1"],
                        "text": value.get("text", "")
                    }
                    owners.append(owner_info)

        # 소유자 수 조정
        owners.sort(key=lambda x: x["y1"])
        owners_to_remove = len(owners) - name_count

        if owners_to_remove > 0:
            for i in range(owners_to_remove):
                owner = owners[i]
                del merged_data["registry_document"][owner["page"]][owner["key"]]
        
        return merged_data
    
    except Exception as e:
        print(f"\n오류 발생: {str(e)}")
        raise

# 주소 앞에 [집합건물] 헤드 지우기
def remove_brackets(address):
    # 정규표현식을 사용하여 [...]로 둘러싸인 부분을 찾아 제거
    cleaned_address = re.sub(r'\[.*?\]', '', address)
    # 추가 공백 정리 (여러 공백을 하나로 줄이기)
    cleaned_address = re.sub(r'\s+', ' ', cleaned_address).strip()
    return cleaned_address
# 네이버 Geocoding API 호출 함수 정의
def geocode_address(address):
    url = f"https://naveropenapi.apigw.ntruss.com/map-geocode/v2/geocode?query={address}"
    headers = {
        'X-NCP-APIGW-API-KEY-ID': client_id,
        'X-NCP-APIGW-API-KEY': client_secret
    }
    response = requests.get(url, headers=headers)
    if response.status_code == 200:
        data = response.json()
        if data['addresses']:
            location = data['addresses'][0]
            return location['y'], location['x']
        else:
            return None, None
    else:
        print(f"Error {response.status_code}: {response.text}")
        return None, None
#gpt 기동
def analyze_with_gpt(analysis_data):
    message_content = f"다음 데이터를 분석하고 JSON 형식으로 응답해주세요. {analysis_data}"
    
    response = client.chat.completions.create(
        model=MODEL,
        messages=[{
            "role": "user",
            "content": message_content
        }],
        response_format={"type": "json_object"},
        max_tokens=3000
    )
    return json.loads(response.choices[0].message.content.strip())
#주소 확인
def parse_address(address):
    parsed_result = {}

    match = re.search(r"^(서울특별시|부산광역시|경기도|대구광역시|인천광역시|광주광역시|대전광역시|울산광역시|세종특별자치시|제주특별자치도|충청북도|충청남도|전라북도|전라남도|경상북도|경상남도)\s+(\S+구|\S+시|\S+군)", address)
    if match:
        parsed_result["시도"] = match.group(1)
        parsed_result["시군구"] = match.group(2)
        address = address.replace(match.group(0), "").strip() 

    match = re.search(r"(\S+동\d*|\S+읍|\S+면)(?:가)?", address)
    if match:
        full_dong = match.group(0)
        parsed_result["동리"] = full_dong
        address = address.replace(full_dong, "").strip()

    match = re.search(r"(?:제)?(\d+)동", address)
    if match:
        parsed_result["동명"] = match.group(1)
        address = re.sub(r"제?\d+동", "", address).strip()

    address = re.sub(r"제?\d+층", "", address).strip()

    match = re.search(r"(?:제)?(\d+)호", address)
    if match:
        parsed_result["호명"] = match.group(1)
        address = re.sub(r"제?\d+호", "", address).strip()

    building_match = re.search(r"([가-힣A-Za-z0-9]+(?:[가-힣A-Za-z0-9\s]+)?(?:아파트|빌라|오피스텔|타워|팰리스|파크|하이츠|프라자|빌딩|스카이|센터|시티|맨션|코아|플라자|타운|힐스))", address)
    if building_match:
        parsed_result["건물명"] = building_match.group(1)
        address = address.replace(building_match.group(1), "").strip()

    for key in ["시도", "시군구", "동리", "동명", "호명"]:
        if key not in parsed_result:
            parsed_result[key] = "nan"

    return parsed_result
#공시가 구하기
def price(address):
    result = parse_address(address)
    print(result)
    # 모든 시도에 대한 GCS 파일 경로 매핑
    gcs_urls = {
        "서울특별시": "https://storage.googleapis.com/jipsin/storage/seoul.csv",
        "부산광역시": "https://storage.googleapis.com/jipsin/storage/busan.csv",
        "대구광역시": "https://storage.googleapis.com/jipsin/storage/daegu.csv",
        "인천광역시": "https://storage.googleapis.com/jipsin/storage/incheon.csv",
        "광주광역시": "https://storage.googleapis.com/jipsin/storage/gwangju.csv",
        "대전광역시": "https://storage.googleapis.com/jipsin/storage/daejeon.csv",
        "울산광역시": "https://storage.googleapis.com/jipsin/storage/ulsan.csv",
        "세종특별자치시": "https://storage.googleapis.com/jipsin/storage/sejong.csv",
        "경기도": "https://storage.googleapis.com/jipsin/storage/gyeonggi.csv",
        "강원특별자치도": "https://storage.googleapis.com/jipsin/storage/gangwon.csv",
        "충청북도": "https://storage.googleapis.com/jipsin/storage/chungbuk.csv",
        "충청남도": "https://storage.googleapis.com/jipsin/storage/chungnam.csv",
        "전라북도": "https://storage.googleapis.com/jipsin/storage/jeunbuk.csv",
        "전라남도": "https://storage.googleapis.com/jipsin/storage/jeunnam.csv",
        "경상북도": "https://storage.googleapis.com/jipsin/storage/gyeongbuk.csv",
        "경상남도": "https://storage.googleapis.com/jipsin/storage/gyeongnam.csv",
        "제주특별자치도": "https://storage.googleapis.com/jipsin/storage/jeju.csv",
    }
    gcs_url = gcs_urls.get(result["시도"], None)

    if gcs_url:
        df = pd.read_csv(gcs_url)
    else:
        print("해당 시도에 대한 GCS 데이터 없음")
    cost = df[
        (df['시도']==result["시도"]) &
        (df['시군구']==result["시군구"]) &
        (df['동리']==result["동리"]) &
        (df["동명"]==result["동명"]) &
        (df["호명"]==result["호명"])
    ]

    if cost.empty:
        cost = df[
            (df['시도']==result["시도"]) &
            (df['시군구']==result["시군구"]) &
            (df['동리']==result["동리"])
        ]
        

    cost_records = cost.to_dict(orient='records')

    # DataFame에서 직접 공시가격 확인 (GPT 호출 없이)
    if not cost.empty:
        # 결과가 1개만 있으면 바로 반환
        if len(cost) == 1:
            direct_price = cost.iloc[0]['공시가격']
            return {"공시가격": direct_price, "method": "direct_match"}
    
    # GPT 분석 사용
    if len(cost_records) == 0:
        print("검색 결과가 없습니다. 데이터베이스에 해당 주소와 유사한 항목이 없습니다.")
        return {"error": "해당 주소를 찾을 수 없습니다.", "공시가격": "NA"}
    else:
        parsed_info = {
            "원본주소": address,
            "파싱결과": result,
            "건물명_추출": result.get("건물명", "알 수 없음"),
            "검색결과수": len(cost_records)
        }

        prompt = {
            "task": "주소 유사도 분석 및 공시가격 추출",
            "parsed_info": parsed_info,
            "candidate_data": cost_records,
            "instruction": "위 원본 주소와 가장 유사한 후보 데이터를 찾아 해당 행의 '공시가격' 값을 JSON 형식으로 반환해주세요. 단지명과 동호수가 가장 중요한 매칭 기준입니다. 반드시 '공시가격' 키에 공시가격 값을 포함해야 합니다."
        }
        
        prompt_json = json.dumps(prompt, ensure_ascii=False, indent=2)
        try:
            gpt_result = analyze_with_gpt(prompt_json)
            
            if 'public_price' in gpt_result:
                return {"공시가격": gpt_result['public_price'], "method": "gpt_analysis"}
            elif '공시가격' in gpt_result:
                return {"공시가격": gpt_result['공시가격'], "method": "gpt_analysis"}
            else:
                return {"공시가격": cost.iloc[0]['공시가격'], "method": "fallback_first_result"}
                
        except Exception as e:
            if not cost.empty:
                return {"공시가격": cost.iloc[0]['공시가격'], "method": "fallback_after_error"}
            return {"error": f"GPT API 오류: {str(e)}", "공시가격": "NA"}
#좌표로 면적 찾기
def building(data):
    result_dict = {}
    counter = 1
    # 1. coai_result_a에서 소재지+임차할부분 합쳐서 지오코딩
    for key, sub_data in data["contract"].items():
        if isinstance(sub_data, dict):
            if "소재지" in sub_data and "임차할부분" in sub_data:
                # 소재지와 임차할부분 합치기
                combined_address = sub_data["소재지"]["text"] + " " + sub_data["임차할부분"]["text"]
                
                lat, lng = geocode_address(combined_address)
                
                if lat and lng:
                    address_key = f"location_{counter}"
                    result_dict[address_key] = {
                        "address": combined_address,
                        "latitude": lat,
                        "longitude": lng,
                        "source": "coai_combined"
                    }
                    # counter 증가 추가
                    counter += 1

    # 2. ledger_result에서 주소 정보 추출하여 지오코딩
    for key, sub_data in data["building_registry"].items():
        if isinstance(sub_data, dict):
            # 도로명주소 지오코딩
            if "도로명주소" in sub_data:
                address = sub_data["도로명주소"]["text"]
                lat, lng = geocode_address(address)
                
                if lat and lng:
                    address_key = f"location_{counter}"
                    result_dict[address_key] = {
                        "address": address,
                        "latitude": lat,
                        "longitude": lng,
                        "source": "ledger_도로명주소"
                    }
                    # counter 증가 추가
                    counter += 1

    # 3. reg_result에서 건물주소 지오코딩
    for key, sub_data in data.get("registry_document", {}).items():
        if isinstance(sub_data, dict) and "건물주소" in sub_data:
            address = sub_data["건물주소"]["text"]
            address=remove_brackets(address)

            lat, lng = geocode_address(address)
            
            if lat and lng:
                address_key = f"location_{counter}"
                result_dict[address_key] = {
                    "address": address,
                    "latitude": lat,
                    "longitude": lng,
                    "source": "reg_건물주소"
                }
                # counter 증가 추가
                counter += 1
    json.dumps(result_dict, ensure_ascii=False, indent=2)
    prompt = {
        "task": "주소 유사도 분석 및 도로명 주소 추출",
        "location": result_dict,
        "instruction": "위 값의 정보를 이용해서 같은 장소인지 확인하고, 같은 장소라면 reg_건물주소를 출력해줘. 아니면 'nan'을 출력해줘. 다른 말은 들어가면 안돼"
    }
    prompt_json = json.dumps(prompt, ensure_ascii=False, indent=2)
    result = analyze_with_gpt(prompt_json)
    return result['result']
#실행(수정사항 포함)

def solution(data):

    promt = (f"""
{data}에서 'contract'는 계약서, 'building_registry'는 건축물 대장, 'registry_document'는 등기부등본이다.

다음 항목들을 분석하여 문제가 있으면 각 항목별로 notice와 solution을 추가해주세요:

1. 등기부등본에 '신탁', '압류', '가처분', '가압류', '가등기'가 있는지 확인
2. 건축물대장에 '위반건축물'이 있는지 확인
3. 등기부등본과 계약서상의 주소 및 면적이 일치하는지 확인
4. 계약기간과 임대차 기간이 일치하는지 확인
5. 특약사항과 특약에 해당하는 내용 요약

원본 데이터 구조를 유지하면서, 분석한 항목에 'notice'와 'solution' 필드를 추가해주세요.
예를 들어, 등기부등본에 '가압류'가 있다면:
"""
"""
```
"가압류": {
  "text": "...",
  "bounding_box": {...},
  "notice": "가압류가 설정되어 있어 권리 침해 우려가 있습니다",
  "solution": "가압류 해제 후 계약 진행 권장"
}
```

위반건축물이 있다면:
```
"위반건축물": {
  "text": "...",
  "bounding_box": {...},
  "notice": "위반건축물로 등록되어 있어 법적 문제가 있습니다",
  "solution": "위반 내용 확인 및 시정 후 계약 진행 권장"
}
```

주소/면적/계약기간 불일치는 해당 필드에 notice와 solution을 추가해주세요.
특약사항은 해당 필드에 notice로 요약 내용을 추가해주세요.

문제가 없는 항목은 다음과 같이 추가해주세요:
```
"notice": "문제 없음",
"solution": "계약 진행 가능"
```

원본 데이터의 모든 구조를 유지하고, 필요한 필드에만 notice와 solution을 추가하는 방식으로 결과를 JSON 형태로 반환해주세요.
""")
    result = analyze_with_gpt(promt)

    return result

def find_keys_in_json(data):
    """
    JSON 데이터에서 특정 키들을 찾아 결과를 반환하는 함수
    
    Args:
        data (dict): 검색할 JSON 데이터
        
    Returns:
        dict: 찾은 키와 해당 값
    """
    # 찾을 키 목록
    target_keys = [
        "임대인", "성명1", "성명2", "소유자_3", "소유자_4",  # 임대인/소유자 관련
        "위반건축물",  # 건축물 위반사항
        "신탁", "가압류", "가처분",  # 권리 제한 관련
        "보증금_1", "보증금_2", "차임_1", "차임_2",  # 금액 관련
        "채권최고액",  # 채권 관련
        "관리비_정액", "관리비_비정액",  # 관리비 관련
        "임대차기간", "계약기간",  # 기간 관련
        "특약", "특약사항",  # 특약 관련
        "집합건물", "면적"  # 건물 유형 및 면적 관련
    ]
    
    # 결과 저장할 딕셔너리
    result = {
        "contract": {},
        "building_registry": {},
        "registry_document": {}
    }
    
    # 계약서(contract) 검색
    if "contract" in data:
        for page_key, page_data in data["contract"].items():
            for key, value in page_data.items():
                if key in target_keys:
                    result["contract"][key] = value
    
    # 건축물대장(building_registry) 검색
    if "building_registry" in data:
        for page_key, page_data in data["building_registry"].items():
            for key, value in page_data.items():
                if key in target_keys:
                    result["building_registry"][key] = value
    
    # 등기부등본(registry_document) 검색
    if "registry_document" in data:
        for page_key, page_data in data["registry_document"].items():
            for key, value in page_data.items():
                if key in target_keys:
                    result["registry_document"][key] = value
    
    # 최상위 레벨에 있을 수 있는 면적 정보 확인
    if isinstance(data, dict):
        address_info = {}
        
        # 최상위 레벨에서 시도, 시군구, 동리 등의 주소 정보와 면적 정보 수집
        for key in ["시도", "시군구", "동리", "동명", "호명", "건물명"]:
            if key in data:
                address_info[key] = data[key]
        
        # 면적 관련 정보가 있으면 별도 섹션에 추가
        if address_info:
            result["address_info"] = address_info
        
        # 공시가격 정보가 있으면 추가
        if "공시가격" in data:
            result["property_value"] = {"공시가격": data["공시가격"]}
    
    return result


def request():
    output_path = r"C:\Users\senbo\Desktop\taba_project\ai_series\result\sol_1.json"
    data=process_all_json(r"C:\Users\senbo\Desktop\taba_project\ai_series\result")
    with open(r"C:\Users\senbo\Desktop\taba_project\ai_series\result\sol.json", "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=4)
    bounding_boxes = remove_bounding_boxes(data)
    res_1 = building(data)
    if res_1 != "nan" and res_1 != "NA":
        try:
            res = price(res_1)
            # res가 문제 없는지 확인
            if res and isinstance(res, dict) and '공시가격' in res:
                cost = int(res['공시가격'])
            else:
                print("공시가격 정보를 찾을 수 없습니다.")
                cost = 'na'
        except Exception as e:
            print(f"가격 처리 중 오류 발생: {e}")
            cost = 'nan'
    else:
        cost = 'nan'
    # JSON 데이터 분석 및 처리
    print(cost)
    # Bounding Box 복원
    restore_bounding_boxes(data, bounding_boxes)
    found_keys = find_keys_in_json(data)

    print(json.dumps(found_keys, indent=2, ensure_ascii=False))
    result = solution(found_keys)
    # 결과 저장
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=4)





request()