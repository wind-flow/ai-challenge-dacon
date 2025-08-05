#!/usr/bin/env python3
"""
기존 생성된 데이터에서 질문과 답만 추출하는 테스트
"""

import json
import re
from pathlib import Path

def extract_answer(generated_text: str) -> str:
    """생성된 텍스트에서 답변 추출"""
    # 다양한 패턴으로 답변 찾기
    patterns = [
        r'\[답\]([\s\S]*?)\[', 
        r'\[답변\]([\s\S]*?)\[',
        r'\[ANSWER\]([\s\S]*?)\[',
        r'정답[:：]\s*([^\n]+)',
        r'답[:：]\s*([^\n]+)',
        r'\[정답\]\s*([1-5])\)',
    ]
    
    for pattern in patterns:
        match = re.search(pattern, generated_text, re.MULTILINE)
        if match:
            return match.group(1).strip()
    
    # 객관식 패턴 (1번, 2번 등)
    if any(num in generated_text for num in ['1)', '2)', '3)', '4)', '5)']):
        answer_match = re.search(r'정답.*?([1-5])\s*\)', generated_text)
        if answer_match:
            return answer_match.group(1) + "번"
    
    return "답변 추출 실패"

def clean_question(generated_text: str) -> str:
    """생성된 텍스트에서 질문만 추출"""
    # 질문 패턴
    patterns = [
        r'\[문제\]([\s\S]*?)\[',
        r'\[QUESTION\]([\s\S]*?)\[',
        r'문제[:：]\s*([^\n]+[?])',
    ]
    
    for pattern in patterns:
        match = re.search(pattern, generated_text, re.MULTILINE)
        if match:
            return match.group(1).strip()
    
    # 처음부터 물음표까지 찾기
    question_end = generated_text.find('?')
    if question_end > 0:
        return generated_text[:question_end+1].strip()
    
    # 첫 줄만 반환
    lines = generated_text.split('\n')
    if lines:
        return lines[0].strip()
    
    return generated_text[:200].strip()

# 기존 파일 읽기
input_file = "data/augmented/train_data_20250805_004816.jsonl"
output_file = "data/augmented/train_data_clean.jsonl"
metadata_file = "data/augmented/metadata_clean.json"

print(f"📖 입력 파일: {input_file}")
print("-" * 60)

clean_data = []
metadata_list = []

with open(input_file, 'r', encoding='utf-8') as f:
    for i, line in enumerate(f, 1):
        data = json.loads(line)
        
        # 질문과 답 추출
        question_text = data.get('question', '')
        answer = extract_answer(question_text)
        clean_q = clean_question(question_text)
        
        # 깔끔한 데이터
        clean_entry = {
            'id': data.get('id'),
            'question': clean_q,
            'answer': answer
        }
        clean_data.append(clean_entry)
        
        # 메타데이터
        metadata = {
            'id': data.get('id'),
            'concept': data.get('concept'),
            'quality_score': data.get('quality_score'),
            'context_used': data.get('context_used'),
            'timestamp': data.get('timestamp'),
            'model': data.get('model'),
            'original_length': len(question_text)
        }
        metadata_list.append(metadata)
        
        # 샘플 출력
        if i <= 3:
            print(f"\n[{i}] 원본 길이: {len(question_text)}자")
            print(f"질문: {clean_q[:80]}...")
            print(f"답변: {answer}")

# 저장
with open(output_file, 'w', encoding='utf-8') as f:
    for entry in clean_data:
        f.write(json.dumps(entry, ensure_ascii=False) + '\n')

with open(metadata_file, 'w', encoding='utf-8') as f:
    json.dump({
        'source_file': input_file,
        'total_entries': len(clean_data),
        'metadata': metadata_list
    }, f, ensure_ascii=False, indent=2)

print(f"\n\n✅ 변환 완료!")
print(f"📄 깔끔한 데이터: {output_file}")
print(f"📊 메타데이터: {metadata_file}")
print(f"📈 총 {len(clean_data)}개 항목 처리")