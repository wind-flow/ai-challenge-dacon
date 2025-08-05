#!/usr/bin/env python3
"""
깔끔한 데이터 생성 테스트
질문과 답만 저장, 메타데이터는 별도 파일
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent / "src"))

from generate_data.main import generate_data

# 빠른 테스트 설정
config = {
    'model_name': 'Qwen/Qwen2.5-1.5B-Instruct',  # 작은 모델로 빠른 테스트
    'use_rag': True,
    'use_quantization': False,
    'num_questions': 3,  # 3개만 테스트
    'min_quality': 60,
    'temperature': 0.7,
    'prompt_template': 'prompts/training_data.txt'
}

print("🧪 깔끔한 데이터 생성 테스트")
print("-" * 60)

# 데이터 생성 실행
output_file = generate_data(config)

print("\n✅ 테스트 완료!")
print(f"생성된 파일: {output_file}")

# 생성된 파일 확인
import json
from pathlib import Path

if Path(output_file).exists():
    print("\n📄 생성된 데이터 샘플:")
    print("-" * 60)
    
    with open(output_file, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f, 1):
            if i > 3:  # 처음 3개만
                break
            data = json.loads(line)
            print(f"\n[{i}]")
            print(f"ID: {data.get('id', 'N/A')}")
            print(f"질문: {data.get('question', 'N/A')[:100]}...")
            print(f"답변: {data.get('answer', 'N/A')}")
            
# 메타데이터 파일 확인
metadata_file = output_file.replace('train_data', 'metadata').replace('.jsonl', '.json')
if Path(metadata_file).exists():
    print("\n📊 메타데이터 파일도 생성됨:", metadata_file)
    with open(metadata_file, 'r', encoding='utf-8') as f:
        metadata = json.load(f)
        print(f"총 {len(metadata.get('metadata', []))}개 메타데이터 저장")