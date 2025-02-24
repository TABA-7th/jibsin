import openai
import os
import base64
import pandas as pd
import requests
import json
import uuid
import time
import re

# GPT 모델 설정
MODEL = "gpt-4o"

# 네이버 OCR API 키 및 URL 설정
client = openai.OpenAI()
# 네이버 OCR 호출 함수
def read_ocr(secret_key, api_url, image_file):
    request_json = {
        'images': [{'format': 'jpg', 'name': 'demo'}],
        'requestId': str(uuid.uuid4()),
        'version': 'V2',
        'timestamp': int(round(time.time() * 1000))
    }

    payload = {'message': json.dumps(request_json).encode('UTF-8')}
    files = [('file', open(image_file, 'rb'))]
    headers = {'X-OCR-SECRET': secret_key}

    response = requests.post(api_url, headers=headers, data=payload, files=files)
    if response.status_code == 200:
        ocr_results = response.json()
        all_data = []
        for image_result in ocr_results['images']:
            for field in image_result['fields']:
                text = field['inferText']
                bounding_box = field['boundingPoly']['vertices']
                x1, y1 = int(bounding_box[0]['x']), int(bounding_box[0]['y'])
                x2, y2 = int(bounding_box[2]['x']), int(bounding_box[2]['y'])
                all_data.append({
                    "Text": text,
                    "x1": x1, "y1": y1,
                    "x2": x2, "y2": y2
                })
        df = pd.DataFrame(all_data)
        return df
    else:
        raise ValueError(f"❌ OCR 요청 실패: {response.status_code} - {response.text}")

# GPT 호출 함수
def read_image(client, image_path, MODEL, df):
    # OCR 데이터 JSON 변환
    df_json = json.dumps(df.to_dict(orient="records"), ensure_ascii=False)

    with open(image_path, "rb") as image_file:
        base64_image = base64.b64encode(image_file.read()).decode("utf-8")

    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "user", "content": "출력은 요청 정보만 {'key': 'value'} 형태의 딕셔너리로 출력해줘"},
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": (
                            f"다음은 OCR 분석을 위한 데이터입니다.\n\n"
                            f"✅ **OCR 데이터 (df_json):**\n{json.dumps(df_json, ensure_ascii=False)}\n\n"
                            f"💡 **목표:**\n"
                            f"주어진 문서에서 다음 정보를 정확하게 추출하세요, 반드시 key값으로 추출해야 합니다.:\n"
                            f"1. **건축물대장**\n"
                            f"2. **대지위치**\n"
                            f"3. **도로명주소**(대지 위치 같은 y좌표를 갖는다,  [시/도] [시/군/구] [도로명] [건물번호]의 구조로 이루어진다.)\n"
                            f"4. **위반건축물** (건축물대장 옆에 있으며, OCR 데이터에서 없으면 'NA'로 처리하며, 좌표값은 {json.dumps({'x1': 0, 'y1': 0, 'x2': 0, 'y2': 0})} 으로 설정)\n"
                            f"5. **소유자의 성명과 주소** (각각 독립적인 키로 반환하되, **여러 개일 경우 성명1, 성명2 등으로 추가 key를 생성한다, 다른 key값은 하나만 존재합니다.**)\n"
                            f"6. **구조** (소유자의 **성명** 옆에 위치, 예: 철근콘크리트구조)\n"
                            f"7. **면적** (소유자의 **성명** 옆에 위치, 예: 88.8)\n\n"
                            f"8. **발급일자** (예: yyyy년mm월dd일)\n\n"

                            f"📌 **출력 규칙:**\n"
                            f"- 반드시 `{{'key': 'value'}}` 형태의 **JSON 형식**으로 출력하세요.\n"
                            f"- OCR 데이터에서 **각 정보(성명, 주소)의 바운딩 박스(`bounding_box`)를 각각 포함**해야 합니다.\n"
                            f"- 값이 존재하지 않는 경우 `'text': 'NA'`를 반환하세요.\n\n"

                            f"🔹 **출력 형식 예시:**\n"
                            f"```json\n"
                            f"{{\n"
                            f"  \"건축물대장\": {{\n"
                            f"    \"text\": \"집합건축물대장(전유부,갑)\",\n"
                            f"    \"bounding_box\": {{ \"x1\": 379, \"y1\": 62, \"x2\": 595, \"y2\": 86 }}\n"
                            f"  }},\n"
                            f"  \"대지위치\": {{\n"
                            f"    \"text\": \"서울특별시 서대문구 창천동\",\n"
                            f"    \"bounding_box\": {{ \"x1\": 273, \"y1\": 134, \"x2\": 394, \"y2\": 147 }}\n"
                            f"  }},\n"
                            f"  \"도로명주소\": {{\n"
                            f"    \"text\": \"경기도 하남시 미사강변한강로\",\n"
                            f"    \"bounding_box\": {{ \"x1\": 273, \"y1\": 134, \"x2\": 394, \"y2\": 147 }}\n"
                            f"  }},\n"
                            f"  \"위반건축물\": {{\n"
                            f"    \"text\": \"NA\",\n"
                            f"    \"bounding_box\": {json.dumps({'x1': 0, 'y1': 0, 'x2': 0, 'y2': 0})}\n"
                            f"  }},\n"
                            f"  \"성명\": {{\n"
                            f"    \"text\": \"김나연\",\n"
                            f"    \"bounding_box\": {{ \"x1\": 528, \"y1\": 252, \"x2\": 561, \"y2\": 267 }}\n"
                            f"  }},\n"
                            f"  \"주소\": {{\n"
                            f"    \"text\": \"서울특별시 강남구 테헤란로 123\",\n" 
                            f"    \"bounding_box\": {{ \"x1\": 500, \"y1\": 400, \"x2\": 750, \"y2\": 430 }}\n"
                            f"  }},\n"
                            f"  \"구조\": {{\n"
                            f"    \"text\": \"철근콘크리트구조3\",\n"
                            f"    \"bounding_box\": {{ \"x1\": 500, \"y1\": 500, \"x2\": 750, \"y2\": 530 }}\n"
                            f"  }},\n"
                            f"  \"면적\": {{\n"
                            f"    \"text\": \"88.8\",\n"
                            f"    \"bounding_box\": {{ \"x1\": 500, \"y1\": 600, \"x2\": 750, \"y2\": 630 }}\n"
                            f"  }}\n"
                            f"  \"발급일자\": {{\n"
                            f"    \"text\": \"2025년 2월 11일\",\n"
                            f"    \"bounding_box\": {{ \"x1\": 500, \"y1\": 600, \"x2\": 750, \"y2\": 630 }}\n"
                            f"  }}\n"
                            f"}}\n"
                            f"```\n\n"

                            f"⚠️ **주의사항:**\n"
                            f"- JSON 형식을 반드시 준수하세요.\n"
                            f"- 'bounding box'는 'text'에 해당하는 내용의 ocr 좌표를 모두 포함해야 합니다.\n"
                            f"- 양식은 모두 통일 되어야 합니다.\n"
                            f"- 추가적인 설명 없이 JSON 형태만 출력하세요."
                        )
                    },
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}
                    }
                ]
            }
        ],
        max_tokens=1000

    )


    return response.choices[0].message.content

# JSON 변환 및 정리
def fix_json_format(text: str) -> str:
    text = text.replace("```json", "").replace("```", "").strip()
    text = re.sub(r'(\d{1,3}),(\d{3})', r'\1\2', text)
    return text

# JSON 파일 저장
def save_json(text: str, output_file: str) -> str:
    try:
        text = fix_json_format(text)
        data = json.loads(text)

        os.makedirs(os.path.dirname(output_file), exist_ok=True)  # 폴더가 없으면 생성

        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=4)

        return output_file
    except json.JSONDecodeError as e:
        return f"❌ JSON 변환 실패: {e}"

# 실행 함수
def request(image_path,output_file_path):
    # image_path = os.path.abspath(r"C:\Users\senbo\Desktop\taba\python\sibal\t2.jpg")
    # output_file_path = os.path.abspath(r"C:\Users\senbo\Desktop\taba\python\eee\cleaned_ocr_result.json")

    if not os.path.exists(image_path):
        raise FileNotFoundError(f"❌ 이미지 파일을 찾을 수 없습니다: {image_path}")

    df = read_ocr(key, url, image_path)
    if df.empty:
        raise ValueError("❌ OCR 결과가 없습니다.")

    text = read_image(client, image_path, MODEL, df)
    result = save_json(text, output_file_path)

# request()
