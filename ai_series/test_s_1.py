import time
import pandas as pd
import cv2
import json
from PIL import Image
import requests
import uuid
import time
import openai
import re
import base64
import numpy as np
from io import BytesIO
from PIL import Image
from dotenv import load_dotenv
import os
load_dotenv(r"C:\Users\senbo\Desktop\taba_project\.env")  # .env 파일 로드
api_key = os.getenv("OPENAI_API_KEY")
client_id=os.getenv("client_id")
client_secret=os.getenv("client_secret")
client = openai.OpenAI(api_key=api_key)
secret_key=os.getenv("secret_key")
api_url=os.getenv("api_url")

MODEL = "gpt-4o"
#계약서원본양식
def base_xy():
    rows = [
        ['등기사항전부증명서',348,112,934,162],
        ['집합건물',520,166,766,216],
        ['[집합건물] 건물주소',26,298,908,346],
        ['[표제부](1동의 건물의 표시)',94,354,632,392],
        ['표시번호',34,406,130,444],
        ['접수',172,414,268,440],
        ['소재지번, 건물 명칭 및 번호', 318,410,590,440],
        ['([도로명주소])',312,456,580,642],
        ['건물내역',668,410,808,446],
        ['등기 원인 및 기타사항',904,404,1140,448],
        ['열람일시',22,1620,456,1656],
        ['(대지권이 목적인 토지의 표시)',408,2456,788,2496],
        ['[표제부] (전유부분의 건물의 표시)',80,2672,684,2720],
        ['표시번호',40,2740,130,2776],
        ['접수',166,2732,280,2776],
        ['건물번호',322,2732,480,2780],
        ['(건물번호)',316,2784,490,2842],
        ['건물내역',522,2742,694,2770],
        ['(건물내역)',506,2790,706,2850],
        ['등기원인 및 기타사항',806,2736,1064,2772],
        ['[갑 구] (소유권에 관한 사항)',86,3842,654,3898],
        ['순위번호',46,3908,134,3948], 
        ['등기목적',170,3910,314,3944],
        ['접수', 390,3904,490,3946],
        ['등기원인',524,3906,668,3952],
        ['관리자 및 기타사항', 824,3902,1030,3946],
        ['소유자', 824,3902,1030,4462],
        ['[을 구] (소유권 이외의 권리에 대한 사항)', 88,4562,796,4608],
        ['순위번호',46,4628,134,4658],
        ['등기목적',170,4628,314,4658],
        ['접수', 390,4628,490,4658],
        ['등기원인',524,4628,668,4658],
        ['관리자 및 기타사항',824,4628,1030,4658],
        ['(채권최고액)',718,4662,1156,4752],
        ['이하여백',410,4952,689,4990]
    ]
    xy = pd.DataFrame(columns=['Text', 'x1', 'y1', 'x2', 'y2'])
    xy = pd.concat([xy, pd.DataFrame(rows, columns=xy.columns)], ignore_index=True)
    return xy

def merge_images(image_paths, output_path_2):
    # target_size = (1240, 1755)  # 원하는 이미지 크기

    # # 이미지 불러와 크기 조정
    # images = []
    # for img_path in image_paths:
    #     image = cv2.imread(img_path)  # OpenCV로 이미지 읽기
    #     resized_image = cv2.resize(image, target_size, interpolation=cv2.INTER_AREA)  # 크기 조정
    #     images.append(Image.fromarray(cv2.cvtColor(resized_image, cv2.COLOR_BGR2RGB)))  # OpenCV → PIL 변환
    
    images = []
    for img_path in image_paths:
        image = cv2.imread(img_path)  # OpenCV로 이미지 읽기
        images.append(Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB)))  # OpenCV → PIL 변환

    total_height = sum(img.height for img in images)
    max_width = max(img.width for img in images)

    merged_image = Image.new("RGB", (max_width, total_height))

    # 이미지 붙이기
    y_offset = 0
    for img in images:
        merged_image.paste(img, (0, y_offset))
        y_offset += img.height
    
    # 병합된 이미지 저장
    merged_image.save(output_path_2)
    
    return merged_image

def cre_ocr(image_path):
    # target_size = (1240, 1753)
    image = cv2.imread(image_path)
    # image_1 = cv2.resize(image, target_size, interpolation=cv2.INTER_AREA)
    request_json = {
        'images': [{'format': 'jpg', 'name': 'demo'}],
        'requestId': str(uuid.uuid4()),
        'version': 'V2',
        'timestamp': int(round(time.time() * 1000))
    }
    payload = {'message': json.dumps(request_json).encode('UTF-8')}
    headers = {'X-OCR-SECRET': secret_key}

    # 전처리된 이미지를 메모리에서 바로 전송
    _, img_encoded = cv2.imencode('.jpg', image)
    files = [('file', ('image.jpg', img_encoded.tobytes(), 'image/jpeg'))]

    response = requests.request("POST", api_url, headers=headers, data=payload, files=files)
 
    #가장 기본적인 파일 url 불러와서 보내는 방식
    payload = {'message': json.dumps(request_json).encode('UTF-8')}
    files = [('file', open(image_path, 'rb'))]
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

#-----------[추가함수]-------------
def get_image_height(image_path):
    with Image.open(image_path) as img:
        return img.height  # 이미지 높이 반환
def organize_by_pages(data, image_paths):
    # 이미지 높이 리스트 생성
    page_heights = [get_image_height(img) for img in image_paths]
    
    # 페이지 경계 계산
    page_boundaries = []
    current_height = 0
    for height in page_heights:
        page_boundaries.append({
            'start': current_height,
            'end': current_height + height
        })
        current_height += height

    # 결과를 저장할 딕셔너리 초기화 (키를 "1페이지" 형식으로 생성)
    result = {f"{i+1}페이지": {} for i in range(len(page_heights))}
    
    # 각 항목을 해당하는 페이지에 할당
    for key, value in data.items():
        y1 = value['bounding_box']['y1']
        
        # y1 값이 어느 페이지 범위에 속하는지 확인
        for page_num, boundary in enumerate(page_boundaries):
            if boundary['start'] <= y1 < boundary['end']:
                # 해당 페이지에 항목 추가 (페이지 번호는 1부터 시작)
                page_key = f"{page_num+1}페이지"
                new_value = value.copy()
                # y 좌표 조정 (페이지 시작점만큼 빼기)
                new_value['bounding_box']['y1'] -= boundary['start']
                new_value['bounding_box']['y2'] -= boundary['start']
                result[page_key][key] = new_value
                break
    
    return result
#수정
def read_ocr(image_paths, out_path):
# 이미지 높이 리스트 생성 (리스트 컴프리헨션 활용)
    page_height = [get_image_height(img) for img in image_paths]
    all_dfs = []
    y = 0
    for j, image_path in enumerate(image_paths):
        df = cre_ocr(image_path)
        df[["y1", "y2"]] += y  # y 값 일괄 변경
        all_dfs.append(df)
        y += page_height[j]  # y 값 업데이트

    merged_df = pd.concat(all_dfs, ignore_index=True)
    # merged_df.to_csv(r'C:\Users\senbo\Desktop\taba\python\rrr\test_3.csv')
    # merged_df=pd.read_csv(r'C:\Users\senbo\Desktop\taba\python\rrr\test_3.csv')

    xy=base_xy()
    xy_json = xy.to_json(orient="records", force_ascii=False)
    df_json = merged_df.to_json(orient="records", force_ascii=False)

    target_texts = {
            "종류": "등본 종류 (집합건물, 건물, 토지 중 하나)",
            "(건물주소)": "[등본종류] 도로명 주소 (예: [집합건물] 정왕대로 53번길 29)",
            "열람일시": "yyyy년 mm월 dd일 hh시mm분ss초",
            "(갑구)":"텍스트",
            "(소유권에 관한 사항)": "(소유권에 관한 사항)",
            "소유자":"이름",
            "신탁":"신탁 (예: 신탁, 이외의 다른 단어가 있으면 안됨)",
            "압류":"압류 (예: 압류, 이외의 다른 단어가 있으면 안됨)",
            "가처분":"가처분 (예: 가처분, 이외의 다른 단어가 있으면 안됨)",
            "가압류":"가압류 (예: 가압류, 이외의 다른 단어가 있으면 안됨)",
            "(소유권 이외의 권리에 대한 사항)":"(소유권 이외의 권리에 대한 사항)",
            "(채권최고액)": "최고채권액 금 ###원(예: 채권최고액 금1,000,000,000원)",
            "이하여백": "이 하 여 백"
        }

    # OpenAI API 요청
    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": (
                            f"다음은 OCR 분석을 위한 데이터입니다.\n\n"
                            f"**위치 데이터 (xy):**\n{xy_json}\n\n"
                            f"**내용 데이터 (df):**\n{df_json}\n\n"
                            f"**작업 목표:**\n"
                            f"- 내용이 없으면 'NA'로 표시\n\n"
                            f"- `xy` 데이터의 위치 정보(좌표)를 활용하여 `df` 데이터와 매칭. {xy_json}의 위치는 참고만하고 항상 {df_json}을 따른다.\n"
                            f"- 'xy' 데이터의 바운딩 박스 크기는 'df'에 맞게 조정된다"
                            f" **각 항목의 출력 형식:**\n"
                            + "\n".join([f"- **{key}**: {value}" for key, value in target_texts.items()]) +
                            f"\n\n**결과 형식:**\n"
                            f"- JSON 형식으로 반환 (각 항목의 바운딩 박스 포함)\n"
                            f"- **출력 데이터가 지정된 형식과 다를 경우 자동으로 변환하여 반환**\n\n"
                            f"**반환 예시:**\n"
                            f"{{\n"
                            f"  \"종류\": {{\"text\": \"집합건물\", \"bounding_box\": {{\"x1\": 100, \"y1\": 200, \"x2\": 300, \"y2\": 250}}}},\n"
                            f"  \"건물주소\": {{\"text\": \"정왕대로 53번길 29\", \"bounding_box\": {{\"x1\": 120, \"y1\": 220, \"x2\": 320, \"y2\": 270}}}},\n"
                            f"  \"(소유권에 관한 사항)\": {{\"text\": \"( 소유권에 관한 사항 )\", \"bounding_box\": {{\"x1\": 120, \"y1\": 220, \"x2\": 320, \"y2\": 270}}}},\n"
                            f"  \"소유주\": {{\"text\": \"( 소유권에 관한 사항 )\", \"bounding_box\": {{\"x1\": 120, \"y1\": 220, \"x2\": 320, \"y2\": 270}}}},\n"
                            f"  \"(소유권 이외의 권리에 관한 사항)\": {{\"text\": \"(소유권 이외의 권리에 관한 사항)\", \"bounding_box\": {{\"x1\": 120, \"y1\": 220, \"x2\": 320, \"y2\": 270}}}},\n"                            
                            f"  \"열람일시\": {{\"text\": \"2025년 02월 15일 14시 48분\", \"bounding_box\": {{\"x1\": 150, \"y1\": 250, \"x2\": 350, \"y2\": 300}}}},\n"
                            f"  \"채권최고액\": {{\"text\": \"채권최고액 금1,000,000,000원\", \"bounding_box\": {{\"x1\": 170, \"y1\": 270, \"x2\": 370, \"y2\": 320}}}}\n"
                            f"}}\n\n"
                            f"**주의사항:**\n"
                            f"- 모든 좌표는 df를 기준으로 출력한다."
                            f"- df를 항상 우선시한다."
                            f"- 텍스트가 여러 바운딩 박스에 걸쳐 있는 경우, 중심점 기준으로 판단\n"
                            f"- 내용이 없을 경우 `NA`로 반환, text 내용이 없는 경우 좌표를 0, 0, 0, 0으로 해줘.\n"
                            f"- df 기준으로 없는 내용을 추가하지 말것"
                            f"- 소유주는 '(소유권 이외의 권리에 관한 사항)'와 '(소유권에 관한 사항) 사이에 해당하는 모든 이름이다'"
                            f"- 소유주가 여러명인 경우 소유주_1, 소유주_2 의 형식으로 출력된다"
                            f"- 채권최고액은 '(소유권에 관한 사항)' 과 '이하여백' 사이에 해당하는 모든 금액이다."
                            f"- 채권최고액은 여러개인 경우 채권최고액_1, 채권최고액_2의 형식으로 출력된다."
                            f"- 채권최고액은 채권최고액_i 중 가장 i가 큰 것만을 출력한다."
                            f"- JSON 형식이 정확하도록 반환할 것!\n"
                            f"- JSON 형식 이외의 어떤 알림, 내용은 첨가하지 말것!\n"
                            f"- 반환 내용 외의 경고, 알림은 반환하지 말것\n"
                            f" '아래는 제공된 `xy` 및 `df` 데이터를 사용하여 각 항목을 분석한 결과입니다'와 같은 알림은 절대 금지\n"
                            f" OpenAI 응답내용금지\n"

                        )
                    }
                ]
            }
        ],
        max_tokens=5000,
        temperature=0.2,
        top_p=1.0
    )

    # 응답 처리
    text = response.choices[0].message.content.strip()
    text = fix_json_format(text)
    data=json.loads(text)


    # "(소유권에 관한 사항)"과 "(소유권 이외의 권리에 관한 사항)"을 삭제
    data.pop("(소유권에 관한 사항)", None)
    data.pop("(소유권 이외의 권리에 관한 사항)", None)

    

    return ttj(json.dumps(data, ensure_ascii=False, indent=4), out_path)
#--------------------------------------------
def fix_json_format(text: str) -> str:
    """JSON 형식 오류를 자동으로 수정하는 함수."""
    text = text.strip()

    # JSON 코드 블록 제거
    text = text.replace("```json", "").replace("```", "").strip()

    # 불필요한 텍스트(설명 등) 자동 제거
    json_end_index = text.rfind("}")
    if json_end_index != -1:
        text = text[:json_end_index+1]  # JSON 부분만 남김

    # JSON에서 누락된 ',' 자동 추가
    text = re.sub(r'}\s*{', '}, {', text)

    # JSON 내 숫자 쉼표 오류 수정 (100000,000원 -> 100,000,000원)
    text = re.sub(r'(\d{1,3})(\d{3},\d{3})', r'\1,\2', text)

    return text

#ttj 함수에 저장 기능 제거
def ttj(text: str, output_file: str) -> str:
    """OCR 결과 JSON 데이터를 정리하고 저장하는 함수."""
    try:
        text = fix_json_format(text)  # JSON 형식 자동 수정
        data = json.loads(text)  # JSON 변환

        def fix_text(value):
            if value == "NA":
                return value
            value = re.sub(r'(\d+)\s+(\d+)', r'\1,\2', value)  # 숫자 사이 공백을 콤마로 변환
            return value.strip()

        # 모든 JSON 필드의 값 자동 수정
        for key, value in data.items():
            if isinstance(value, dict) and "text" in value:
                value["text"] = fix_text(value["text"])

        return data

    except json.JSONDecodeError as e:
        print(f"❌ JSON 변환 실패: {e}")
        print("📌 오류 발생 JSON 내용:\n", text)  # JSON 디버깅 출력
        return f"❌ JSON 변환 실패: {e}"

def request(img_list,output_path, output_path_2):
# def request():
    # img_list = [rf"C:\Users\senbo\Desktop\taba\python\reg\000{i}.jpg" for i in range(1, 4)]
    # output_path = rf"C:\Users\senbo\Desktop\taba\python\rrr\test_bui_1.json"
    # output_path_2 = r"C:\Users\senbo\Desktop\taba\python\rrr\test_3_1.jpg"
    
    merge_images(img_list,output_path_2)
    data = read_ocr(img_list,output_path)

    organized_data = organize_by_pages(data, img_list)

    # 결과 저장
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(organized_data, f, ensure_ascii=False, indent=2)

# request()
