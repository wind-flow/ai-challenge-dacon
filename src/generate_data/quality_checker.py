#!/usr/bin/env python3
"""
생성된 데이터 품질 평가 모듈

금융 문제의 품질을 다각도로 평가하고 개선
"""

import re
from typing import Dict, List, Tuple, Optional
import logging

logger = logging.getLogger(__name__)


class QualityChecker:
    """품질 평가 및 개선 클래스"""
    
    def __init__(self):
        # 평가 가중치
        self.weights = {
            'length': 0.2,
            'structure': 0.25,
            'keywords': 0.3,
            'clarity': 0.25
        }
        
        # 금융 키워드 (외부 데이터에서 로드하는 것이 이상적)
        self.finance_keywords = self._load_finance_keywords()
        
        # 모호한 표현
        self.ambiguous_terms = [
            '대략', '아마', '어느정도', '일부', '몇몇',
            '가능한', '일반적으로', '보통', '대체로'
        ]
        
        # 품질 통계
        self.stats = {
            'total_evaluated': 0,
            'passed': 0,
            'failed': 0,
            'avg_score': 0
        }
    
    def _load_finance_keywords(self) -> List[str]:
        """금융 키워드 로드 (외부 데이터에서)"""
        # 실제로는 외부 데이터에서 로드해야 함
        # 여기서는 최소한의 키워드만
        return [
            '금리', '이자', '환율', '투자', '리스크', '위험',
            '시장', '은행', '금융', '자산', '부채', '자본',
            '보험', '연금', '증권', '채권', '주식', '펀드',
            '규제', '감독', '정책', '보안', '개인정보'
        ]
    
    def evaluate(self, text: str, concept: str = "") -> float:
        """
        품질 평가
        
        Args:
            text: 평가할 텍스트
            concept: 관련 개념 (선택)
            
        Returns:
            품질 점수 (0-100)
        """
        scores = {}
        
        # 1. 길이 평가
        scores['length'] = self._evaluate_length(text)
        
        # 2. 구조 평가
        scores['structure'] = self._evaluate_structure(text)
        
        # 3. 키워드 평가
        scores['keywords'] = self._evaluate_keywords(text, concept)
        
        # 4. 명확성 평가
        scores['clarity'] = self._evaluate_clarity(text)
        
        # 가중 평균
        total_score = sum(
            scores[key] * self.weights[key]
            for key in scores
        )
        
        # 통계 업데이트
        self._update_stats(total_score)
        
        return round(total_score, 1)
    
    def _evaluate_length(self, text: str) -> float:
        """길이 평가"""
        length = len(text)
        
        if 100 <= length <= 300:
            return 100
        elif 50 <= length < 100:
            return 70
        elif 300 < length <= 500:
            return 80
        elif length < 50:
            return 30
        else:  # > 500
            return 60
    
    def _evaluate_structure(self, text: str) -> float:
        """구조 평가"""
        score = 50  # 기본 점수
        
        # 질문 형식
        if '?' in text or any(end in text for end in ['는가', '까', '인가']):
            score += 20
        
        # 선택지 구조
        if any(marker in text for marker in ['1)', '2)', '3)', 'A)', 'B)', 'ㄱ)', 'ㄴ)']):
            score += 20
        
        # 줄바꿈으로 구조화
        if '\n' in text and len(text.split('\n')) > 2:
            score += 10
        
        return min(100, score)
    
    def _evaluate_keywords(self, text: str, concept: str) -> float:
        """키워드 평가"""
        score = 30  # 기본 점수
        
        # 개념 포함
        if concept and concept in text:
            score += 20
        
        # 금융 키워드 카운트
        keyword_count = sum(1 for kw in self.finance_keywords if kw in text)
        score += min(50, keyword_count * 10)
        
        return min(100, score)
    
    def _evaluate_clarity(self, text: str) -> float:
        """명확성 평가"""
        score = 100  # 시작 점수
        
        # 모호한 표현 체크
        for term in self.ambiguous_terms:
            if term in text:
                score -= 10
        
        # 너무 많은 조건문
        if text.count('만약') + text.count('경우') > 3:
            score -= 15
        
        # 이중 부정
        if '않지 않' in text or '없지 않' in text:
            score -= 20
        
        return max(0, score)
    
    def _update_stats(self, score: float):
        """통계 업데이트"""
        self.stats['total_evaluated'] += 1
        
        if score >= 70:
            self.stats['passed'] += 1
        else:
            self.stats['failed'] += 1
        
        # 평균 계산
        n = self.stats['total_evaluated']
        prev_avg = self.stats['avg_score']
        self.stats['avg_score'] = (prev_avg * (n-1) + score) / n
    
    def improve_quality(self, text: str, score: float) -> str:
        """
        품질 개선 제안
        
        Args:
            text: 원본 텍스트
            score: 현재 점수
            
        Returns:
            개선된 텍스트 또는 개선 제안
        """
        suggestions = []
        
        if score < 70:
            # 길이 문제
            if len(text) < 50:
                suggestions.append("문제를 더 구체적으로 작성하세요.")
            elif len(text) > 500:
                suggestions.append("문제를 더 간결하게 작성하세요.")
            
            # 키워드 부족
            keyword_count = sum(1 for kw in self.finance_keywords if kw in text)
            if keyword_count < 2:
                suggestions.append("금융 관련 용어를 더 포함하세요.")
            
            # 구조 문제
            if '?' not in text:
                suggestions.append("명확한 질문 형식으로 작성하세요.")
        
        return "\n".join(suggestions) if suggestions else "품질 양호"
    
    def get_statistics(self) -> Dict:
        """평가 통계 반환"""
        return self.stats.copy()
    
    def reset_statistics(self):
        """통계 초기화"""
        self.stats = {
            'total_evaluated': 0,
            'passed': 0,
            'failed': 0,
            'avg_score': 0
        }