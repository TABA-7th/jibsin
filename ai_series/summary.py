import openai
import os
import base64
import pandas as pd
import requests
import json
import uuid
import time
import re
import openai
from dotenv import load_dotenv

MODEL = "gpt-4o"
load_dotenv(r"C:\Users\senbo\Desktop\taba_project\.env")  # .env 파일 로드
api_key = os.getenv("OPENAI_API_KEY")
client = openai.OpenAI(api_key=api_key)


def clean_json(input):
    with open(input, 'r', encoding='utf-8') as f:
        input_json = json.load(f)

    result = {}
    
    # 각 최상위 키에 대해 처리
    for top_key, top_value in input_json.items():
        result[top_key] = {}
        
        # 각 섹션(페이지) 처리
        for section_key, section_value in top_value.items():
            result[top_key][section_key] = {}
            
            # 각 항목 처리
            for item_key, item_value in section_value.items():
                # 딕셔너리가 아닌 경우 건너뛰기
                if not isinstance(item_value, dict):
                    continue
                
                # "notice" 키가 있는 항목만 유지
                if "notice" in item_value:
                    # 새 항목 생성 (bounding_box 제외)
                    new_item = {}
                    for field_key, field_value in item_value.items():
                        if field_key != "bounding_box":
                            new_item[field_key] = field_value
                    
                    # 결과에 추가
                    result[top_key][section_key][item_key] = new_item
            
            # 빈 섹션이면 삭제
            if not result[top_key][section_key]:
                del result[top_key][section_key]
        
        # 빈 최상위 키면 삭제
        if not result[top_key]:
            del result[top_key]
        with open(r'C:\Users\senbo\Desktop\taba_project\ai_series\result\solution_1.json', 'w', encoding='utf-8') as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
    return result


def analyze_with_gpt(analysis_data):
    prompt="""
임대차 계약서를 분석하고 다음 JSON 형식으로 결과를 반환해 주세요.  
각 항목에는 "text" (내용)과 "check" (문제 여부, true/false)를 포함해야 합니다.  
또한, 계약의 전체 요약 정보를 제공하는 "summary" 키를 추가해야 합니다.  

{
  "summary": {
    "text": "[계약의 전체적인 요약 및 주요 문제점]",
    "check": [true/false]  // 전체 계약에 큰 문제가 있으면 true, 없으면 false
  },
  "contract_details": {
    "임대인": {
      "text": "[임대인 이름]",
      "check": [true/false]  // 임대인 정보에 문제가 있으면 true
    },
    "소재지": {
      "text": "[임대차 건물의 주소]",
      "check": [true/false]
    },
    "임차할부분": {
      "text": "[임차 대상 공간]",
      "check": [true/false]
    },
    "면적": {
      "text": "[전용 면적 m²]",
      "check": [true/false]
    },
    "계약기간": {
      "text": "[계약 시작일 ~ 종료일]",
      "check": [true/false]  // 갱신청구권 언급이 없으면 true
    },
    "보증금": {
      "text": "[보증금 금액]",
      "check": [true/false]  // 보증금 관련 정보가 불명확하면 true
    },
    "차임": {
      "text": "[월세 금액 및 지불 조건]",
      "check": [true/false]
    },
    "특약사항": {
      "text": "[특약 조항 요약]",
      "check": [true/false]  // 특약에서 보호 조항이 미흡하면 true
    },
    "등기부등본": {
      "text": "[건물 소유자 및 주요 정보]",
      "check": [true/false]
    }
  }
}

1. "check" 값이 **true**이면, 해당 항목에 문제가 있는 것이며, **false**이면 문제가 없는 것입니다.
2. "summary" 항목은 계약의 전체적인 요약과 주요 문제점을 포함해야 합니다.
3. 계약 기간, 보증금, 차임, 특약사항 등에서 법적 보호가 부족하면 "check"를 true로 설정해야 합니다.
4. 모든 값은 JSON 형식을 유지하며, 계약서 원문을 기반으로 분석해야 합니다.

반환 형식은 **위 JSON 구조를 유지하며**, 실제 계약서 정보를 바탕으로 정확하게 분석하여 채워 주세요.

"""
    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "user", "content": f"다음 JSON 데이터를 분석해 주세요:\n\n```json\n{analysis_data}\n```\n\n이 데이터에서 'notice'와 'solution' 정보를 기반으로 계약의 주요 문제점과 해결책을 요약해주세요."},
            {"role": "user", "content": f"출력 양식은 다음과 같습니다. {prompt}"}
        ],
        response_format={"type": "json_object"},
        max_tokens=3000
    )
    return json.loads(response.choices[0].message.content.strip())

def summary_result(data):
    result = analyze_with_gpt(data)
    return result


def request():
    data=clean_json(r'C:\Users\senbo\Desktop\taba_project\ai_series\result\solution.json')
    result = summary_result(data)
    with open(r'C:\Users\senbo\Desktop\taba_project\ai_series\result\solution_2.json', "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=4)



request()

