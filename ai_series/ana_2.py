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
            ledger_data = {"1": json.load(f)}
        with open(files["reg"], 'r', encoding='utf-8') as f:
            reg_data = json.load(f)

        # 데이터 통합
        merged_data = {
            "coai_result_a": coai_data,
            "ledger_result": ledger_data,
            "reg_result": reg_data
        }

        # 1단계: 소유자 수 조정
        name_count = sum(1 for key in ledger_data["1"].keys() if key.startswith("성명"))
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
                del merged_data["reg_result"][owner["page"]][owner["key"]]
        
        return merged_data
    
    except Exception as e:
        print(f"\n오류 발생: {str(e)}")
        raise


def clean_json(data,res_1,cost):
    def analyze_with_gpt(analysis_data):
        response = client.chat.completions.create(
            model=MODEL,
            messages=[{
                "role": "user",
                "content": analysis_data
            }],
            response_format={"type": "json_object"},
            max_tokens=3000
        )
        return json.loads(response.choices[0].message.content.strip())
    def ana_1(data):
        """임대인 정보 분석"""
        analysis_data = {
            "contract": [],
            "ledger": [],
            "register": []
        }

        if "coai_result_a" in data:
            for key, sub_data in data["coai_result_a"].items():
                if isinstance(sub_data, dict) and "임대인" in sub_data:
                    analysis_data["contract"].append(sub_data["임대인"]["text"])

        if "ledger_result" in data:
            for key, sub_data in data["ledger_result"].items():
                if isinstance(sub_data, dict):
                    for sub_key, value in sub_data.items():
                        if sub_key.startswith("성명"):
                            analysis_data["ledger"].append(value["text"])

        if "reg_result" in data:
            for key, sub_data in data["reg_result"].items():
                if isinstance(sub_data, dict):
                    for sub_key, value in sub_data.items():
                        if sub_key.startswith("소유자"):
                            analysis_data["register"].append(value["text"])

        prompt = (
        f"다음은 부동산 계약 관련 문서에서 추출한 임대인 정보입니다:"
        f"계약서 상 임대인: {', '.join(analysis_data['contract'])}"
        f"건축물대장 소유자: {', '.join(analysis_data['ledger'])}"
        f"등기부등본 소유자: {', '.join(analysis_data['register'])}"
                    
                    """부동산 계약서에서 임대 목적물의 상태와 권리 관계를 점검하고, 문제가 없는지 확인해줘.  
다음 항목을 기준으로 분석하고, 수정 또는 보완이 필요한 부분을 구체적으로 설명해줘.  

**임대 목적물의 기본 정보 확인**  
- 계약서에 기재된 **주소(동·호수 포함), 면적이 정확한가?**  
- 등기부등본과 계약서상의 주소 및 면적이 일치하는가?  

**건물의 권리관계 확인**  
- 등기부등본을 확인하여 **근저당, 가압류, 압류, 가등기, 경매개시결정을 확인했는가?**  
- 건축물대장을 확인하여 **위반건축물을 확인했는가?**

⚠ **위 항목에서 문제가 발견될 경우, 해결 방법과 법적 보호 조치를 상세히 설명해줘.**  


                    내부적으로 모든 분석을 수행한 후, 최종적으로 아래 **JSON 형식으로만** 응답해.
                    ```json
                    {
                      "notice": "발견된 문제 요약",
                      "solution": "해결 방법 요약"
                    }
                    ```
                    **출력 규칙:**
                    - 문제가 있으면 `notice`에 **주요 문제 요약**을 입력하고, `solution`에 **해결 방법**을 제공해.
                    - 문제가 없으면 다음과 같이 응답해:
                      ```json
                      {
                        "notice": "문제 없음",
                        "solution": "계약 진행 가능"
                      }
                      ```
                    - JSON 형식 외의 설명을 포함하지 마.
                    """)
        result = analyze_with_gpt(prompt)

        # 분석 결과 추가
        if "coai_result_a" in data:
            for key, sub_data in data["coai_result_a"].items():
                if isinstance(sub_data, dict) and "임대인" in sub_data:
                    data["coai_result_a"][key]["임대인"]["notice"] = result.get("notice", "")
                    data["coai_result_a"][key]["임대인"]["solution"] = result.get("solution", "")

        if "ledger_result" in data:
            for key, sub_data in data["ledger_result"].items():
                if isinstance(sub_data, dict):
                    for sub_key, value in sub_data.items():
                        if sub_key.startswith("성명"):
                            data["ledger_result"][key][sub_key]["notice"] = result.get("notice", "")
                            data["ledger_result"][key][sub_key]["solution"] = result.get("solution", "")

        if "reg_result" in data:
            for key, sub_data in data["reg_result"].items():
                if isinstance(sub_data, dict):
                    for sub_key, value in sub_data.items():
                        if sub_key.startswith("소유자"):
                            data["reg_result"][key][sub_key]["notice"] = result.get("notice", "")
                            data["reg_result"][key][sub_key]["solution"] = result.get("solution", "")

        return data
    
    def ana_2(data,res_1):
        """위치 및 면적 분석"""
        analysis_data = {
            "contract": {},
            "ledger": {},
            "register": {}
        }
        
        if "coai_result_a" in data:
            for key, sub_data in data["coai_result_a"].items():
                if isinstance(sub_data, dict):
                    for target_key in ["소재지", "임차할부분", "면적"]:
                        if target_key in sub_data:
                            analysis_data["contract"][target_key] = sub_data[target_key]["text"]

        if "ledger_result" in data:
            for key, sub_data in data["ledger_result"].items():
                if isinstance(sub_data, dict):
                    for target_key in ["대지위치", "도로명주소", "면적"]:
                        if target_key in sub_data:
                            analysis_data["ledger"][target_key] = sub_data[target_key]["text"]

        if "reg_result" in data:
            for key, sub_data in data["reg_result"].items():
                if isinstance(sub_data, dict):
                    for target_key in ["건물주소"]:
                        if target_key in sub_data:
                            analysis_data["register"][target_key] = sub_data[target_key]["text"]

        prompt = (f"""
        다음은 부동산 계약 관련 문서의 위치 및 면적 정보입니다:

        계약서 정보:
        - 면적: {analysis_data['contract'].get('면적', 'NA')}

        건축물대장 정보:
        - 면적: {analysis_data['ledger'].get('면적', 'NA')}

        주소 일치 여부:
        - {res_1}
"""
"""
                    부동산 계약서에서 임대 목적물의 상태와 권리 관계를 점검하고, 문제가 없는지 확인해줘.  
다음 항목을 기준으로 분석하고, 수정 또는 보완이 필요한 부분을 구체적으로 설명해줘.  

**임대 목적물의 기본 정보 확인**  
- 계약서에 기재된 **면적이 정확한가?**
- **주소 일치 여부**가 nan이 아니면 주소가 일치하는 것으로 간주한다.  
- 등기부등본과 계약서상의 주소 및 면적이 일치하는가?  

⚠ **위 항목에서 문제가 발견될 경우, 해결 방법과 법적 보호 조치를 상세히 설명해줘.**  


                    내부적으로 모든 분석을 수행한 후, 최종적으로 아래 **JSON 형식으로만** 응답해.
                    ```json
                    {
                      "notice": "발견된 문제 요약",
                      "solution": "해결 방법 요약"
                    }
                    ```
                    **출력 규칙:**
                    - 문제가 있으면 `notice`에 **주요 문제 요약**을 입력하고, `solution`에 **해결 방법**을 제공해.
                    - 문제가 없으면 다음과 같이 응답해:
                      ```json
                      {
                        "notice": "문제 없음",
                        "solution": "계약 진행 가능"
                      }
                      ```
                    - JSON 형식 외의 설명을 포함하지 마.
        """)

        result = analyze_with_gpt(prompt)

        # 분석 결과 추가
        if "coai_result_a" in data:
            for key, sub_data in data["coai_result_a"].items():
                if isinstance(sub_data, dict):
                    for target_key in ["소재지", "임차할부분", "면적"]:
                        if target_key in sub_data:
                            data["coai_result_a"][key][target_key]["notice"] = result.get("notice", "")
                            data["coai_result_a"][key][target_key]["solution"] = result.get("solution", "")

        if "ledger_result" in data:
            for key, sub_data in data["ledger_result"].items():
                if isinstance(sub_data, dict):
                    for target_key in ["대지위치", "도로명주소", "면적"]:
                        if target_key in sub_data:
                            data["ledger_result"][key][target_key]["notice"] = result.get("notice", "")
                            data["ledger_result"][key][target_key]["solution"] = result.get("solution", "")

        if "reg_result" in data:
            for key, sub_data in data["reg_result"].items():
                if isinstance(sub_data, dict):
                    if "건물주소" in sub_data:
                        data["reg_result"][key]["건물주소"]["notice"] = result.get("notice", "")
                        data["reg_result"][key]["건물주소"]["solution"] = result.get("solution", "")

        return data
    #수정
    def ana_2_2(data):

        analysis_data = {
            "ledger": {},
            "register": {}
        }
        if "ledger_result" in data:
            for key, sub_data in data["ledger_result"].items():
                if isinstance(sub_data, dict):
                    for target_key in ["위반건축물"]:
                        if target_key in sub_data:
                            analysis_data["ledger"][target_key] = sub_data[target_key]["text"]

        if "reg_result" in data:
            for key, sub_data in data["reg_result"].items():
                if isinstance(sub_data, dict):
                    for target_key in ["신탁", "가압류", "가처분"]:
                        if target_key in sub_data:
                            analysis_data["register"][target_key] = sub_data[target_key]["text"]

        prompt = (f"""
        다음은 부동산 계약 관련 문서의 위험매물 정보입니다.
"""
"""
                    부동산 계약서에서 임대 목적물의 상태와 권리 관계를 점검하고, 문제가 없는지 확인해줘.  
다음 항목을 기준으로 분석하고, 수정 또는 보완이 필요한 부분을 구체적으로 설명해줘.  

**건물의 권리관계 확인**  
- 등기부등본을 확인하여 **근저당, 가압류, 압류, 가등기, 경매개시결정을 확인했는가?**  
- 건축물대장을 확인하여 **위반건축물을 확인했는가?**

⚠ **위 항목에서 문제가 발견될 경우, 해결 방법과 법적 보호 조치를 상세히 설명해줘.**  


                    내부적으로 모든 분석을 수행한 후, 최종적으로 아래 **JSON 형식으로만** 응답해.
                    ```json
                    {
                      "notice": "발견된 문제 요약",
                      "solution": "해결 방법 요약"
                    }
                    ```
                    **출력 규칙:**
                    - 문제가 있으면 `notice`에 **주요 문제 요약**을 입력하고, `solution`에 **해결 방법**을 제공해.
                    - 문제가 없으면 다음과 같이 응답해:
                      ```json
                      {
                        "notice": "문제 없음",
                        "solution": "계약 진행 가능"
                      }
                      ```
                    - JSON 형식 외의 설명을 포함하지 마.
        """)

        result = analyze_with_gpt(prompt)

        if "ledger_result" in data:
            for key, sub_data in data["ledger_result"].items():
                if isinstance(sub_data, dict):
                    for target_key in ["위반건축물"]:
                        if target_key in sub_data:
                            data["ledger_result"][key][target_key]["notice"] = result.get("notice", "")
                            data["ledger_result"][key][target_key]["solution"] = result.get("solution", "")

        if "reg_result" in data:
            for key, sub_data in data["reg_result"].items():
                if isinstance(sub_data, dict):
                    for target_key in ["신탁", "가압류", "가처분"]:
                        data["reg_result"][key][target_key]["notice"] = result.get("notice", "")
                        data["reg_result"][key][target_key]["solution"] = result.get("solution", "")
        return data
    
    def ana_3(data,cost):
        if isinstance(cost, (int, float)):  # 숫자인 경우만 실행
            cost=cost*70/69
        else:
            cost="nan"
        if "coai_result_a" in data:
            for key, sub_data in data["coai_result_a"].items():
                if isinstance(sub_data, dict):
                    payment_info = {}
                    # 기본 정보 수집
                    for sub_key in sub_data:
                        if any(sub_key.startswith(prefix) for prefix in ["보증금", "차임", "관리비"]):
                            payment_info[sub_key] = sub_data[sub_key]["text"]
                    
                    # 기본값으로 채권최고액 설정
                    payment_info["채권최고액"] = "0원"
                    
                    # reg_result에서 채권최고액 추가
                    if "reg_result" in data:
                        for page_data in data["reg_result"].values():
                            if "(채권최고액)" in page_data:
                                payment_info["채권최고액"] = page_data["(채권최고액)"]["text"]
                    
                    # 모든 정보가 수집된 후 프롬프트 생성
                    if payment_info:
                        prompt = (f"""
                        다음은 부동산 계약의 보증금, 임대료, 관리비, 채권최고액 정보입니다:
                        {', '.join(f'{k}: {v}' for k, v in payment_info.items())}
                        건물의 가치: {cost}

"""
#-------------------------------------------------------------------------------------------------                        
"""
                    부동산 계약서에서 임대 목적물의 상태와 권리 관계를 점검하고, 문제가 없는지 확인해줘.  
다음 항목을 기준으로 분석하고, 수정 또는 보완이 필요한 부분을 구체적으로 설명해줘.  

계약서에서 보증금 및 월세 조건이 명확하게 작성되었는지 분석해줘.
(보증금+채권최고액)과 건물의 가치를 비교해줘. 건물의 가치가 높으면 안전하고, 건물의 가치가 낮으면 위험하다고 경고해줘.
보증금+채권최고액이 높으면 위험하다는 경고가 필요해.
보증금과 월세 금액이 명확하게 기재되었는지 확인해줘. 
관리비가 별도로 청구되는지, 포함된 항목(전기, 수도, 가스 등)이 적절히 기재되었는지 점검해줘.
분석 후 누락된 정보나 모호한 부분을 지적해줘."
⚠ **위 항목에서 문제가 발견될 경우, 해결 방법과 법적 보호 조치를 상세히 설명해줘.**  


                    내부적으로 모든 분석을 수행한 후, 최종적으로 아래 **JSON 형식으로만** 응답해.
                    ```json
                    {
                      "notice": "발견된 문제 요약",
                      "solution": "해결 방법 요약"
                    }
                    ```
                    **출력 규칙:**
                    - 문제가 있으면 `notice`에 **주요 문제 요약**을 입력하고, `solution`에 **해결 방법**을 제공해.
                    - 문제가 없으면 다음과 같이 응답해:
                      ```json
                      {
                        "notice": "문제 없음",
                        "solution": "계약 진행 가능"
                      }
                      ```
                    - JSON 형식 외의 설명을 포함하지 마.
                    """)
                        result = analyze_with_gpt(prompt)
                        
                        # 결과를 각 필드에 추가
                        for sub_key in [k for k in payment_info.keys() if k != "채권최고액"]:
                            if sub_key in data["coai_result_a"][key]:  # 원본 데이터에 있는 키만 처리
                                data["coai_result_a"][key][sub_key]["notice"] = result.get("notice", "")
                                data["coai_result_a"][key][sub_key]["solution"] = result.get("solution", "")
        return data
    def ana_4(data):
            """계약기간 분석"""
            if "coai_result_a" in data:
                for key, sub_data in data["coai_result_a"].items():
                    if isinstance(sub_data, dict):
                        period_info = {}
                        for period_key in ["계약기간", "임대차기간"]:
                            if period_key in sub_data:
                                period_info[period_key] = sub_data[period_key]["text"]
                        
                        if period_info:
                            prompt = (f"""
                            다음은 부동산 계약의 기간 정보입니다:
                            
                            {', '.join(f'{k}: {v}' for k, v in period_info.items())}"""
                            
"""
다음 항목을 기준으로 분석하고, 수정 또는 보완이 필요한 부분을 구체적으로 설명해줘.  

계약서의 계약 기간 및 갱신 조건을 검토해줘.

계약 기간이 정확하게 (예: 2025년 2월 21일 ~ 2027년 2월 20일) 기재되어 있는지 확인해줘.
계약 갱신청구권(최소 2년 거주 보장)에 대한 내용이 포함되어 있는지 점검해줘.
중도 해지 시 위약금이나 해지 절차가 명확히 정의되어 있는지 분석해줘.
위 조건을 기준으로 검토하고, 누락된 사항이 있으면 지적해줘."

⚠ **위 항목에서 문제가 발견될 경우, 법적 보호를 받을 수 있는 방법을 설명해줘.**  


                    내부적으로 모든 분석을 수행한 후, 최종적으로 아래 **JSON 형식으로만** 응답해.
                    ```json
                    {
                      "notice": "발견된 문제 요약",
                      "solution": "해결 방법 요약"
                    }
                    ```
                    **출력 규칙:**
                    - 문제가 있으면 `notice`에 **주요 문제 요약**을 입력하고, `solution`에 **해결 방법**을 제공해.
                    - 문제가 없으면 다음과 같이 응답해:
                      ```json
                      {
                        "notice": "문제 없음",
                        "solution": "계약 진행 가능"
                      }
                      ```
                    - JSON 형식 외의 설명을 포함하지 마.
                    """
                            )
                            result = analyze_with_gpt(prompt)
                            
                            # 결과를 각 필드에 추가
                            for period_key in period_info.keys():
                                data["coai_result_a"][key][period_key]["notice"] = result.get("notice", "")
                                data["coai_result_a"][key][period_key]["solution"] = result.get("solution", "")
            return data
    def ana_5(data):
        if "coai_result_a" in data:
            for key, sub_data in data["coai_result_a"].items():
                if isinstance(sub_data, dict):
                    special_terms = {}
                    for sub_key in sub_data:
                        if sub_key.startswith("특약"):
                            special_terms[sub_key] = sub_data[sub_key]["text"]
                    
                    if special_terms:
                        prompt = (f"""
                        다음은 부동산 계약의 특약사항입니다:
                        
                        {', '.join(f'{k}: {v}' for k, v in special_terms.items())}"""
                        """
다음 항목을 기준으로 분석하고, 수정 또는 보완이 필요한 부분을 구체적으로 설명해줘.  

계약서의 계약 기간 및 갱신 조건을 검토해줘.

계약 기간이 정확하게 (예: 2025년 2월 21일 ~ 2027년 2월 20일) 기재되어 있는지 확인해줘.
계약 갱신청구권(최소 2년 거주 보장)에 대한 내용이 포함되어 있는지 점검해줘.
중도 해지 시 위약금이나 해지 절차가 명확히 정의되어 있는지 분석해줘.
위 조건을 기준으로 검토하고, 누락된 사항이 있으면 지적해줘."

⚠ **위 항목에서 문제가 발견될 경우, 법적 보호를 받을 수 있는 방법을 설명해줘.**  


                    내부적으로 모든 분석을 수행한 후, 최종적으로 아래 **JSON 형식으로만** 응답해.
                    ```json
                    {
                      "notice": "발견된 문제 요약",
                      "solution": "해결 방법 요약"
                    }
                    ```
                    **출력 규칙:**
                    - 문제가 있으면 `notice`에 **주요 문제 요약**을 입력하고, `solution`에 **해결 방법**을 제공해.
                    - 문제가 없으면 다음과 같이 응답해:
                      ```json
                      {
                        "notice": "문제 없음",
                        "solution": "계약 진행 가능"
                      }
                      ```
                    - JSON 형식 외의 설명을 포함하지 마.
                    """
                        )
                        result = analyze_with_gpt(prompt)
                        
                        # 결과를 각 필드에 추가
                        for sub_key in special_terms.keys():
                            data["coai_result_a"][key][sub_key]["notice"] = result.get("notice", "")
                            data["coai_result_a"][key][sub_key]["solution"] = result.get("solution", "")
        return data

    data = ana_1(data)
    data = ana_2(data, res_1)
    data = ana_2_2(data)
    data = ana_3(data, cost)
    data = ana_4(data)
    data = ana_5(data)
    return data
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
    for key, sub_data in data["coai_result_a"].items():
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
    for key, sub_data in data["ledger_result"].items():
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
    for key, sub_data in data.get("reg_result", {}).items():
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
def request():
    output_path = r"C:\Users\senbo\Desktop\taba_project\ai_series\result\sol_1.json"
    data=process_all_json(r"C:\Users\senbo\Desktop\taba_project\ai_series\result")
    bounding_boxes = remove_bounding_boxes(data)
    res_1 = building(data)
    if res_1 != "nan":
        res = price(res_1)
        cost=int(res['공시가격'])
    else:
        cost = 'nan'
    # JSON 데이터 분석 및 처리
    data = clean_json(data,res_1,cost)
    # Bounding Box 복원
    restore_bounding_boxes(data, bounding_boxes)
    
    # 결과 저장
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=4)


request()