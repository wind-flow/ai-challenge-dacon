#!/usr/bin/env python3
"""
새로운 프롬프트로 빠른 테스트
"""

import sys
sys.path.append("src")

from generate_data.main import generate_data

print("="*60)
print("📝 새로운 프롬프트 테스트")
print("="*60)

# 간단한 프롬프트 테스트
config = {
    'model_name': 'gpt2',  # 가장 작은 모델
    'use_rag': False,  # RAG 비활성화
    'use_quantization': False,
    'num_questions': 1,
    'min_quality': 30,  # 품질 기준 낮춤
    'temperature': 0.9,
    'prompt_template': 'prompts/simple.txt'  # 간단한 프롬프트
}

print("\n설정:")
print(f"- 모델: {config['model_name']}")
print(f"- 프롬프트: {config['prompt_template']}")
print(f"- 문제 수: {config['num_questions']}")

print("\n생성 시작...")

try:
    result = generate_data(config)
    print(f"\n✅ 완료!")
    print(f"결과 파일: {result}")
    
    # 생성된 내용 확인
    import json
    with open(result, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f, 1):
            data = json.loads(line)
            print(f"\n--- 문제 {i} ---")
            print(f"질문: {data['question'][:100]}...")
            print(f"답변: {data['answer']}")
            
except Exception as e:
    print(f"❌ 오류: {e}")
    import traceback
    traceback.print_exc()