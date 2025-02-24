import firebase_admin
from firebase_admin import credentials, storage
import os

# ✅ Firebase Admin SDK 초기화 (중복 방지)
if not firebase_admin._apps:
    cred = credentials.Certificate(r"C:\Users\senbo\Desktop\taba_project\ai_series\firebase_key.json")
    firebase_admin.initialize_app(cred, {"storageBucket": "jibsin.appspot.com"})

# ✅ 업로드할 파일 경로 설정
image_path = r"C:\Users\senbo\Desktop\taba_project\ai_series\result\page_1_with_boxes.jpg"  # 업로드할 이미지 파일 경로
if not os.path.exists(image_path):
    print(f"❌ 파일이 존재하지 않습니다: {image_path}")
    exit()

# ✅ Firebase Storage 업로드
try:
    bucket = storage.bucket()
    blob = bucket.blob(f"images/{os.path.basename(image_path)}")  # Firebase Storage 내부 경로
    blob.upload_from_filename(image_path)  # 파일 업로드

    # 퍼블릭 URL 생성 (Firebase 보안 규칙 필요)
    blob.make_public()
    print(f"✅ 업로드 성공: {blob.public_url}")

except Exception as e:
    print(f"⚠️ 업로드 실패: {e}")
