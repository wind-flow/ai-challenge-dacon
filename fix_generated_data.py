#!/usr/bin/env python3
"""
생성된 데이터를 정리하여 적절한 형식으로 변환
"""

import json
import re
from pathlib import Path
import sys

sys.path.append("src")
from generate_data.parser import QuestionParser

def extract_clean_data(raw_data):
    """
    원시 데이터에서 깨끗한 문제/답 추출
    """
    parser = QuestionParser()
    
    # 현재 Ko-PlatYi가 생성하는 형식 분석
    text = raw_data.get('question', '')
    
    # 첫 번째 줄을 질문으로
    lines = text.strip().split('\n')
    question = ""
    answer = ""
    choices = []
    
    # 질문 찾기 - 첫 번째 의미있는 문장
    for line in lines:
        line = line.strip()
        if line and not line.startswith('[') and '정답' not in line:
            # 물음표가 있으면 거기까지
            if '?' in line:
                question = line[:line.index('?')+1]
            else:
                question = line
            break
    
    # 선택지 찾기
    choice_pattern = r'^([1-4])\)\s*(.+)'
    for i, line in enumerate(lines):
        match = re.match(choice_pattern, line.strip())
        if match:
            choices.append(line.strip())
    
    # 정답 찾기 - 더 넓은 패턴
    for line in lines:
        # 다양한 정답 패턴
        patterns = [
            r'정답[:\]]\s*([1-4])',  # 정답: 1 또는 [정답] 2
            r'^\s*([1-4])\s*\)\s*$',  # 단독 라인의 1)
            r'답[:\s]+([1-4])',       # 답: 1
        ]
        
        for pattern in patterns:
            answer_match = re.search(pattern, line)
            if answer_match:
                answer = answer_match.group(1)
                break
        
        if answer:
            break
    
    # 해설 찾기 (선택사항)
    explanation = ""
    explanation_start = False
    for line in lines:
        if '해설' in line or '설명' in line:
            explanation_start = True
            continue
        if explanation_start and line.strip():
            explanation = line.strip()
            break
    
    return {
        'id': raw_data.get('id', 'UNKNOWN'),
        'question': question,
        'choices': choices[:4] if choices else [],
        'answer': answer,
        'explanation': explanation,
        'concept': raw_data.get('concept', ''),
        'quality_score': raw_data.get('quality_score', 0)
    }

def process_file(input_file, output_file):
    """
    파일 처리
    """
    cleaned_data = []
    
    with open(input_file, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                data = json.loads(line)
                clean = extract_clean_data(data)
                
                # 최소 요구사항 확인
                if clean['question'] and clean['answer']:
                    cleaned_data.append(clean)
                    print(f"✅ {clean['id']}: 처리 완료")
                else:
                    print(f"❌ {data.get('id', 'UNKNOWN')}: 불완전한 데이터")
                    print(f"   질문: {clean['question'][:50] if clean['question'] else 'None'}...")
                    print(f"   답변: {clean['answer'] if clean['answer'] else 'None'}")
                    
            except Exception as e:
                print(f"❌ 오류: {e}")
    
    # 결과 저장
    with open(output_file, 'w', encoding='utf-8') as f:
        for item in cleaned_data:
            # JSONL 형식으로 저장
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
    
    print(f"\n📊 처리 결과:")
    print(f"- 입력: {input_file}")
    print(f"- 출력: {output_file}")
    print(f"- 성공: {len(cleaned_data)}개")
    
    # 샘플 출력
    if cleaned_data:
        print(f"\n📝 샘플:")
        sample = cleaned_data[0]
        print(f"질문: {sample['question']}")
        if sample['choices']:
            print("선택지:")
            for choice in sample['choices']:
                print(f"  {choice}")
        print(f"정답: {sample['answer']}")

if __name__ == "__main__":
    # 특정 파일 처리
    input_file = Path("data/augmented/train_data_20250805_004816.jsonl")
    output_file = Path("data/augmented/cleaned_train_data.jsonl")
    
    if input_file.exists():
        print(f"🔄 데이터 정리 시작")
        print(f"입력: {input_file}")
        
        process_file(input_file, output_file)
    else:
        print(f"❌ 파일을 찾을 수 없습니다: {input_file}")