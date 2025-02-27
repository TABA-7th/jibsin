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

def process_all_json(input_dir):
    try:
        # 파일 경로 설정
        files = {
            "coai": os.path.join(input_dir, "coai_result_a.json"),
            "ledger": os.path.join(input_dir, "ledger_result.json"),
            "reg": os.path.join(input_dir, "test_bui_1.json")
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


#------------------------[수정사항]--------------------------
def restore_bounding_boxes(data, bounding_boxes):
    """저장된 Bounding Box 값을 복원하는 함수"""
    def traverse(node, path=""):
        if isinstance(node, dict):
            if path in bounding_boxes:
                node["bounding_box"] = bounding_boxes[path]
            for key, value in node.items():
                new_path = f"{path}.{key}" if path else key
                traverse(value, new_path)
        elif isinstance(node, list):
            for idx, item in enumerate(node):
                new_path = f"{path}[{idx}]"
                traverse(item, new_path)
    
    # 깊은 복사로 입력 데이터 보존
    import copy
    result = copy.deepcopy(data)
    
    # 복원 실행
    traverse(result)
    
    # 결과 반환 (이 부분이 누락되어 있었음)
    return result
def building(data):
    result_dict = {}
    counter = 1
    address_list = []
    used_keys = []

    for key, sub_data in data["contract"].items():
        if isinstance(sub_data, dict):
            if "소재지" in sub_data and "임차할부분" in sub_data:
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
                    used_keys.append(address_key)
                    address_list.append(combined_address)
                    counter += 1

    for key, sub_data in data["building_registry"].items():
        if isinstance(sub_data, dict) and "도로명주소" in sub_data:
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
                used_keys.append(address_key)
                address_list.append(address)
                counter += 1

    for key, sub_data in data.get("registry_document", {}).items():
        if isinstance(sub_data, dict) and "건물주소" in sub_data:
            address = remove_brackets(sub_data["건물주소"]["text"])
            lat, lng = geocode_address(address)
            if lat and lng:
                address_key = f"location_{counter}"
                result_dict[address_key] = {
                    "address": address,
                    "latitude": lat,
                    "longitude": lng,
                    "source": "reg_건물주소"
                }
                used_keys.append(address_key)
                address_list.append(address)
                counter += 1

    json.dumps(result_dict, ensure_ascii=False, indent=2)
    prompt = {
        "task": "주소 유사도 분석 및 도로명 주소 추출",
        "location": result_dict,
        "addresses": address_list,
        "instruction": "각 주소별 유사도를 분석하고 같은 장소인지 확인하여 모두 같은 장소라면 reg_건물주소를 result 값으로 출력해줘. 아니면 'nan'을 result 값으로 출력해줘. 다른 말은 들어가면 안돼"
    }

    prompt_json = json.dumps(prompt, ensure_ascii=False, indent=2)
    result = analyze_with_gpt(prompt_json)
    print(result)
    return result['result']

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
        "(채권최고액)",  # 채권 관련
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

    
    return result

def solution_1(data): #등본, 건축물 대장 상 위험 매물, 면적, 계약기간, 임대차 기간, 특약 요약, 주소

    promt = (f"""
{data}에서 'contract'는 계약서, 'building_registry'는 건축물 대장, 'registry_document'는 등기부등본이다.

다음 항목들을 분석하여 문제가 있으면 각 항목별로 notice와 solution을 추가해주세요:

1. 등기부등본에 '신탁', '압류', '가처분', '가압류', '가등기'가 있는지 확인
2. 건축물대장에 '위반건축물'이 있는지 확인
3. 등기부등본과 계약서상의 면적이 일치하는지 확인
4. 계약기간과 임대차 기간이 일치하는지 확인
5. 특약사항과 특약에 임차인에게 불리한 조항 확인
6. 관리비_비정액에 값이 있고 관리비_정액에 값이 없으면 경고
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

면적/계약기간 불일치는 해당 필드에 notice와 solution을 추가해주세요.
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
def solution_2(data): #사용자 이름
    promt = (f"""
{data}에서 'contract'는 계약서, 'building_registry'는 건축물 대장, 'registry_document'는 등기부등본이다.
계약서에서 '임대인', 건축물대장에서 '성명', 등기부등본에서 '소유자'이 일치하는지 확인 할 것.
성명, 소유자가 1명이 아닌 경우 공동명의로 판단한다.
성명끼리는 같은 notice와 solution을 출력한다.
소유자끼리는 같은 notice와 solution을 출력한다.
"""
"""
소유자가 한 명이 아니라면 '임대인'의 notice에 공지한다.
```
"임대인": {
  "text": "...",
  "bounding_box": {...},
  "notice": "소유자가 공동명의로 확인됩니다",
  "solution": "다른 소유주의 확인 필요"
}
```
건축물대장 '성명'과 등기부등본의 '소유자', 계약서의 '임대인' 중 일치하지 않는 것이 있다면
```
"소유자": {
  "text": "...",
  "bounding_box": {...},
  "notice": "건축물 대장 혹은 계약서의 임대인과 일치하지 않습니다",
  "solution": "임대인을 확실하게 확인하여 주십시오."
}
```
```
"성명": {
  "text": "...",
  "bounding_box": {...},
  "notice": "건축물 대장 혹은 계약서의 임대인과 일치하지 않습니다",
  "solution": "임대인을 확실하게 확인하여 주십시오."
}
```

임대인/성명/소유자 불일치는 해당 필드에 notice와 solution을 추가해주세요.
문제가 없는 항목은 다음과 같이 추가해주세요:
```
"notice": "문제 없음",
"solution": "계약 진행 가능"
```

원본 데이터의 모든 구조를 유지하고, 필요한 필드에만 notice와 solution을 추가하는 방식으로 결과를 JSON 형태로 반환해주세요.
""")
    result = analyze_with_gpt(promt)

    return  result
def solution_3(data,cost): #보증금, 근저당권, 공시가
    promt = (f"""
{data}에서 'contract'는 계약서, 'building_registry'는 건축물 대장, 'registry_document'는 등기부등본이다. {cost}는 공시가격이다.
"""
f"""
다음 항목들을 분석하여 문제가 있으면 각 항목별로 notice와 solution을 추가해주세요:
'보증금', '채권최고액' 외에는 notice, solution을 추가하지 않는다.
1. 보증금 일관성 확인:
   - 보증금_1과 보증금_2의 금액이 다른 경우 오류 메시지를 출력
   - 금액 차이가 있는 경우 두 보증금 필드 모두에 오류 표시
원본 데이터 구조를 유지하면서, 분석한 항목에 'notice'와 'solution' 필드를 추가해주세요.
이외의 다른 정보는 무시한다.
예를 들어, 보증금_1과 보증금_2의 금액이 다른 경우:

```json
"보증금_1": {{
  "text": "...",
  "bounding_box": {{...}},
  "notice": "보증금_2와 금액이 다릅니다",
  "solution": "계약서 내용 확인 후 보증금 금액을 일치시켜야 합니다."
}}
채권최고액에 대한 분석 결과는 다음과 같이 추가해주세요:
"채권최고액": {{
  "text": "...",
  "bounding_box": {{...}},
  "notice": "채권최고액이 보증금과 공시가격({cost})를 초과하는지 확인하세요",
  "solution": "채권최고액은 보증금과 공시가격의 차이 이내로 설정하는 것이 안전합니다."
}}
공시가격이 :
"보증금_1": {{
  "text": "...",
  "bounding_box": {{...}},
  "notice": "공시가격 정보가 없어 적정 보증금 여부를 판단할 수 없습니다.",
  "solution": "국토교통부 부동산 공시가격 알리미 등을 통해 공시가격을 확인하세요."
}}
문제가 없는 항목은 다음과 같이 추가해주세요:
"notice": "문제 없음",
"solution": "계약 진행 가능"
""")
    result = analyze_with_gpt(promt)

    return  result

def merge_analysis(sol_json, analysis_jsons):
    """
    구조가 동일한 여러 JSON에서 notice와 solution을 병합
    모든 notice와 solution을 가져옴 (기본 메시지 포함)
    
    Args:
        sol_json (dict): 원본 JSON
        analysis_jsons (list): 분석 결과 JSON 리스트
    
    Returns:
        dict: 병합된 JSON
    """
    # 각 섹션과 필드 순회
    for section_key, section in sol_json.items():
        for subsection_key, subsection in section.items():
            for field_key, field_value in list(subsection.items()):  # list()로 감싸서 반복 중 수정 가능하게 함
                notices = []
                solutions = []
                
                # 각 분석 JSON에서 값 확인
                for analysis in analysis_jsons:
                    # 동일한 경로에 필드가 있는지 확인
                    if (section_key in analysis and 
                        subsection_key in analysis[section_key] and 
                        field_key in analysis[section_key][subsection_key]):
                        
                        analysis_field = analysis[section_key][subsection_key][field_key]
                        
                        # notice와 solution이 있는지 확인
                        if isinstance(analysis_field, dict):
                            if "notice" in analysis_field:
                                # 모든 notice 포함 (문제 없음도 포함)
                                if analysis_field["notice"] not in notices:
                                    notices.append(analysis_field["notice"])
                            
                            if "solution" in analysis_field:
                                # 모든 solution 포함 (계약 진행 가능도 포함)
                                if analysis_field["solution"] not in solutions:
                                    solutions.append(analysis_field["solution"])
                
                # 결과 추가
                if isinstance(field_value, dict):
                    # 이미 딕셔너리인 경우
                    if notices:
                        sol_json[section_key][subsection_key][field_key]["notice"] = "; ".join(notices)
                    
                    if solutions:
                        sol_json[section_key][subsection_key][field_key]["solution"] = "; ".join(solutions)
                else:
                    # 딕셔너리가 아닌 경우 변환
                    if notices or solutions:
                        new_field = {"text": field_value}
                        
                        if notices:
                            new_field["notice"] = "; ".join(notices)
                        
                        if solutions:
                            new_field["solution"] = "; ".join(solutions)
                        
                        sol_json[section_key][subsection_key][field_key] = new_field
    
    return sol_json
    

def request():
    output_path = r"C:\Users\senbo\Desktop\taba_project\ai_series\result\sol_1.json"
    data=process_all_json(r"C:\Users\senbo\Desktop\taba_project\ai_series\result")
    bounding_boxes = remove_bounding_boxes(data)
    res_1 = building(data)
    used_keys = [
    "소재지",
    "임차할부분",
    "도로명주소",
    "건물주소"
]
    # 디버깅: 타입과 정확한 값 확인
    print(f"res_1의 타입: {type(res_1)}, 값: {repr(res_1)}")

    # 보다 안전한 조건식
    if res_1 and res_1 not in ["nan", "NA", "NaN", "NAN", float('nan'), None]:
        try:
            res = price(res_1)
            if res and isinstance(res, dict) and '공시가격' in res:
                cost = int(res['공시가격'])
                for key in used_keys:
                    if key in data:
                        data[key]["notice"] = "문제 없음"
                        data[key]["solution"] = "계약 진행 가능"
            else:
                print("공시가격 정보를 찾을 수 없습니다.")
                cost = 'na'
        except Exception as e:
            print(f"가격 처리 중 오류 발생: {e}")
            cost = 'nan'
    else:
        cost = 'nan'
        print(f"주소 불일치 감지: res_1 = {res_1}")
        
        # used_keys가 None이거나 비어있는지 확인
        if used_keys is None:
            print("used_keys가 None입니다. 기본 키를 사용합니다.")
            used_keys = ["주소", "소재지", "건물주소"]  # 기본 키 설정
        
        # used_keys가 비어있는지 확인
        if not used_keys:
            print("used_keys가 비어있습니다. 기본 키를 사용합니다.")
            used_keys = ["주소", "소재지", "건물주소"]  # 기본 키 설정
        
        print(f"사용할 키: {used_keys}")
        
        # data 내에서 주소 관련 키를 찾아 notice 추가
        for section in ["contract", "building_registry", "registry_document"]:
            if section in data:
                for subsection_key, subsection in data[section].items():
                    for key in used_keys:
                        if key in subsection:
                            subsection[key]["notice"] = "주소가 일치하지 않습니다"
                            subsection[key]["solution"] = "주소 확인이 필요합니다."
                            print(f"{section}.{subsection_key}.{key}에 notice 추가 완료")

    # JSON 데이터 분석 및 처리
    print(cost)
    # Bounding Box 복원


    result_1 = solution_1(data)
    result_2 = solution_2(data)
    result_3 = solution_3(data,cost)
    merged_json = merge_analysis(data, [result_1, result_2, result_3])
    result_json= restore_bounding_boxes(merged_json, bounding_boxes)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(result_json, f, ensure_ascii=False, indent=4)
    



start=time.time()
request()
end=time.time()
print(end-start)