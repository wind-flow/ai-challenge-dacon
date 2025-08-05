#!/usr/bin/env python3
"""
단순화된 데이터 생성 테스트
토픽/개념 없이 순수 RAG 기반
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent / "src"))

from generate_data.main import generate_data

# 테스트 설정
config = {
    'model_name': 'Qwen/Qwen2.5-1.5B-Instruct',  # 빠른 테스트
    'use_rag': True,
    'use_quantization': False,
    'num_questions': 3,
    'min_quality': 50,  # 낮춰서 더 많이 통과하도록
    'temperature': 0.7,
    'prompt_template': 'prompts/simple_qa.txt'  # 단순한 프롬프트
}

print("🧪 단순화된 데이터 생성 테스트")
print("✅ 토픽/개념 추출 없음")
print("✅ 순수 RAG 컨텍스트 기반")
print("✅ 단순한 출력 형식")
print("-" * 60)

# 실행
output_file = generate_data(config)

print("\n✅ 완료!")