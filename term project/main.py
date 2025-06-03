import subprocess
import sys
import os

# ✅ 출력 인코딩 문제 해결 (한글 깨짐 방지)
sys.stdout.reconfigure(encoding='utf-8')

print("📦 Term Project End-to-End 파이프라인 실행 시작")

# 1. 전처리 파이프라인 실행
print("\n▶️ 실행 중: preprocessing_pipeline.py")
subprocess.run(["python", "preprocessing_pipeline.py"], encoding="utf-8")

# 2. 모델링 파이프라인 실행
print("\n▶️ 실행 중: modeling_pipeline.py")
subprocess.run(["python", "modeling_pipeline.py"], encoding="utf-8")




print("\n✅ 전체 End-to-End 파이프라인 완료")
