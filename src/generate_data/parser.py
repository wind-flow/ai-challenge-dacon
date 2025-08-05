#!/usr/bin/env python3
"""
FSKU 문제 파싱 유틸리티

생성된 텍스트에서 구조화된 문제 추출
"""

import re
from typing import Dict, Optional, List
import logging

logger = logging.getLogger(__name__)


class QuestionParser:
    """문제 파싱 클래스"""
    
    def parse_structured_output(self, text: str) -> Dict[str, str]:
        """
        구조화된 출력 파싱
        
        Returns:
            {
                'question': 질문 텍스트,
                'choices': ['1) ...', '2) ...', '3) ...', '4) ...'],
                'answer': '2',
                'explanation': '설명...'
            }
        """
        result = {
            'question': '',
            'choices': [],
            'answer': '',
            'explanation': ''
        }
        
        # QUESTION 추출
        question_match = re.search(r'\[QUESTION\](.*?)\[/QUESTION\]', text, re.DOTALL)
        if question_match:
            result['question'] = question_match.group(1).strip()
        else:
            # 대체 패턴
            question_match = re.search(r'\[문제\](.*?)(?:\[|$)', text, re.DOTALL)
            if question_match:
                result['question'] = question_match.group(1).strip()
        
        # CHOICES 추출
        choices_match = re.search(r'\[CHOICES\](.*?)\[/CHOICES\]', text, re.DOTALL)
        if choices_match:
            choices_text = choices_match.group(1).strip()
            # 선택지 파싱
            choice_pattern = r'([1-4])\)\s*(.+?)(?=(?:[1-4]\)|$))'
            choices = re.findall(choice_pattern, choices_text, re.DOTALL)
            result['choices'] = [f"{num}) {text.strip()}" for num, text in choices]
        else:
            # 대체 패턴
            choices_match = re.search(r'\[선택지\](.*?)(?:\[|$)', text, re.DOTALL)
            if choices_match:
                choices_text = choices_match.group(1).strip()
                choice_pattern = r'([1-4])\)\s*(.+?)(?=(?:[1-4]\)|$))'
                choices = re.findall(choice_pattern, choices_text, re.DOTALL)
                result['choices'] = [f"{num}) {text.strip()}" for num, text in choices]
        
        # ANSWER 추출
        answer_match = re.search(r'\[ANSWER\](.*?)\[/ANSWER\]', text, re.DOTALL)
        if answer_match:
            answer_text = answer_match.group(1).strip()
            # 숫자만 추출
            answer_num = re.search(r'[1-4]', answer_text)
            if answer_num:
                result['answer'] = answer_num.group()
        else:
            # 대체 패턴
            answer_match = re.search(r'\[정답\]\s*([1-4])', text)
            if answer_match:
                result['answer'] = answer_match.group(1)
        
        # EXPLANATION 추출
        explanation_match = re.search(r'\[EXPLANATION\](.*?)\[/EXPLANATION\]', text, re.DOTALL)
        if explanation_match:
            result['explanation'] = explanation_match.group(1).strip()
        else:
            # 대체 패턴
            explanation_match = re.search(r'\[해설\](.*?)(?:\[|$)', text, re.DOTALL)
            if explanation_match:
                result['explanation'] = explanation_match.group(1).strip()
        
        return result
    
    def parse_simple_format(self, text: str) -> Dict[str, str]:
        """
        간단한 형식 파싱 (현재 Ko-PlatYi가 생성하는 형식)
        """
        result = {
            'question': '',
            'choices': [],
            'answer': '',
            'explanation': ''
        }
        
        # 전체 텍스트를 줄 단위로 분리
        lines = text.strip().split('\n')
        
        # 질문 찾기 (첫 번째 의미있는 줄)
        for line in lines:
            if line.strip() and not line.startswith('['):
                result['question'] = line.strip()
                break
        
        # 선택지 찾기
        choices = []
        for i, line in enumerate(lines):
            # 1), 2), 3), 4) 패턴 찾기
            choice_match = re.match(r'^([1-4])\)\s*(.+)', line.strip())
            if choice_match:
                choices.append(line.strip())
        
        if len(choices) >= 4:
            result['choices'] = choices[:4]
        
        # 정답 찾기
        for line in lines:
            if '정답' in line or 'ANSWER' in line.upper():
                # 숫자 추출
                answer_num = re.search(r'[1-4]', line)
                if answer_num:
                    result['answer'] = answer_num.group()
                    break
        
        # 해설 찾기
        explanation_start = False
        explanation_lines = []
        for line in lines:
            if '해설' in line or 'EXPLANATION' in line.upper():
                explanation_start = True
                continue
            if explanation_start:
                explanation_lines.append(line.strip())
        
        if explanation_lines:
            result['explanation'] = ' '.join(explanation_lines[:3])  # 처음 3줄만
        
        return result
    
    def extract_question_answer(self, text: str) -> Dict[str, str]:
        """
        현재 생성되는 형식에서 질문과 답 추출
        """
        # 먼저 구조화된 형식 시도
        result = self.parse_structured_output(text)
        
        # 실패하면 간단한 형식 시도
        if not result['question']:
            result = self.parse_simple_format(text)
        
        # 그래도 실패하면 기본 추출
        if not result['question']:
            # 질문: 첫 번째 문장 또는 물음표까지
            question_end = text.find('?')
            if question_end > 0:
                result['question'] = text[:question_end+1].strip()
            else:
                # 첫 줄을 질문으로
                first_line = text.strip().split('\n')[0]
                result['question'] = first_line
            
            # 답: 정답이라는 단어 다음의 숫자
            answer_match = re.search(r'정답[:\s]*([1-4])', text)
            if answer_match:
                result['answer'] = answer_match.group(1)
        
        return result
    
    def format_for_evaluation(self, parsed: Dict[str, str]) -> str:
        """
        평가용 형식으로 포맷팅
        """
        formatted = f"문제: {parsed['question']}\n\n"
        
        if parsed['choices']:
            formatted += "선택지:\n"
            for choice in parsed['choices']:
                formatted += f"{choice}\n"
            formatted += "\n"
        
        formatted += f"정답: {parsed['answer']}\n"
        
        if parsed['explanation']:
            formatted += f"\n해설: {parsed['explanation']}"
        
        return formatted


# 테스트
if __name__ == "__main__":
    parser = QuestionParser()
    
    # 테스트 텍스트
    test_text = """
    [QUESTION]
    금융위원회의 주요 역할은 무엇인가요?
    [/QUESTION]
    
    [CHOICES]
    1) 통화정책 수립
    2) 금융정책 수립 및 감독
    3) 세금 징수
    4) 무역 정책 결정
    [/CHOICES]
    
    [ANSWER]
    2
    [/ANSWER]
    
    [EXPLANATION]
    금융위원회는 금융정책을 수립하고 금융기관을 감독하는 역할을 합니다.
    [/EXPLANATION]
    """
    
    result = parser.parse_structured_output(test_text)
    print("파싱 결과:")
    print(result)
    print("\n포맷팅:")
    print(parser.format_for_evaluation(result))