#!/usr/bin/env python3
"""
외부 데이터에서 금융 개념 자동 추출

대회 규칙 준수: 수기 작성 금지, 외부 데이터 기반 추출
"""

import json
import re
import random
from pathlib import Path
from typing import List, Set, Dict, Tuple
from collections import Counter
import logging

logger = logging.getLogger(__name__)


class ConceptExtractor:
    """외부 데이터에서 금융 개념 자동 추출"""
    
    def __init__(self, data_dir: str = "data"):
        self.data_dir = Path(data_dir)
        self.concepts = set()
        self.concept_contexts = {}
        self.concept_frequencies = Counter()
        
        # 금융 용어 패턴 (정규식)
        self.patterns = [
            r'[가-힣]+(?:금리|이자율)',
            r'[가-힣]+(?:리스크|위험)',
            r'[가-힣]+(?:시장|거래소)',
            r'[가-힣]+(?:은행|금융)',
            r'[가-힣]+(?:투자|자산)',
            r'[가-힣]+(?:보험|연금)',
            r'[가-힣]+(?:증권|채권|주식)',
            r'[가-힣]+(?:규제|법규|정책)',
            r'[가-힣]+(?:보안|보호)',
            r'(?:전자|디지털|온라인)[가-힣]+',
            r'[A-Z]{2,10}',  # 영문 약어 (ETF, KOSPI 등)
        ]
    
    def extract_concepts(self) -> Set[str]:
        """
        외부 데이터에서 개념 추출
        
        Returns:
            추출된 개념 집합
        """
        # 외부 데이터 디렉토리
        external_dir = self.data_dir / "external"
        
        if not external_dir.exists():
            logger.warning(f"외부 데이터 디렉토리 없음: {external_dir}")
            logger.warning("data/external/ 폴더에 금융 관련 문서를 추가하세요.")
            return set()
        
        # 모든 텍스트 파일 처리
        for file_path in external_dir.glob("*.txt"):
            self._process_text_file(file_path)
        
        # JSON 파일 처리
        for file_path in external_dir.glob("*.json"):
            self._process_json_file(file_path)
        
        # CSV 파일 처리 (train.csv만, test.csv 제외)
        raw_dir = self.data_dir / "raw"
        if raw_dir.exists():
            for file_path in raw_dir.glob("*.csv"):
                if "test" not in file_path.name.lower():
                    self._process_csv_file(file_path)
        
        logger.info(f"총 {len(self.concepts)}개 개념 추출")
        return self.concepts
    
    def _process_text_file(self, file_path: Path):
        """텍스트 파일 처리"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
                self._extract_from_text(content)
        except Exception as e:
            logger.error(f"파일 읽기 실패 {file_path}: {e}")
    
    def _process_json_file(self, file_path: Path):
        """JSON 파일 처리"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                self._extract_from_json(data)
        except Exception as e:
            logger.error(f"JSON 파일 읽기 실패 {file_path}: {e}")
    
    def _process_csv_file(self, file_path: Path):
        """CSV 파일 처리"""
        try:
            import pandas as pd
            df = pd.read_csv(file_path)
            
            # 텍스트 컬럼에서 추출
            text_columns = ['Question', 'question', 'text', 'content']
            for col in text_columns:
                if col in df.columns:
                    for text in df[col].dropna():
                        self._extract_from_text(str(text))
        except Exception as e:
            logger.error(f"CSV 파일 읽기 실패 {file_path}: {e}")
    
    def _extract_from_text(self, text: str):
        """텍스트에서 개념 추출"""
        if not text:
            return
        
        # 패턴 매칭
        for pattern in self.patterns:
            matches = re.findall(pattern, text)
            for match in matches:
                if 2 <= len(match) <= 20:
                    self.concepts.add(match)
                    self.concept_frequencies[match] += 1
                    
                    # 컨텍스트 저장 (처음 발견된 경우)
                    if match not in self.concept_contexts:
                        context = self._extract_context(text, match)
                        if context:
                            self.concept_contexts[match] = context
        
        # "~란", "~는" 패턴으로 정의되는 용어
        definitions = re.findall(r'([가-힣]{2,10})(?:이란|란|는)\s', text)
        for term in definitions:
            self.concepts.add(term)
            self.concept_frequencies[term] += 1
    
    def _extract_from_json(self, data: any):
        """JSON 데이터에서 재귀적으로 추출"""
        if isinstance(data, str):
            self._extract_from_text(data)
        elif isinstance(data, dict):
            for value in data.values():
                self._extract_from_json(value)
        elif isinstance(data, list):
            for item in data:
                self._extract_from_json(item)
    
    def _extract_context(self, text: str, concept: str) -> str:
        """개념 주변 컨텍스트 추출"""
        # 개념이 포함된 문장 찾기
        sentences = re.split(r'[.。!?]', text)
        for sentence in sentences:
            if concept in sentence:
                return sentence.strip()
        return ""
    
    def get_weighted_concept(self) -> Dict[str, any]:
        """
        빈도 기반 가중치로 개념 선택
        
        Returns:
            {'concept': 개념, 'frequency': 빈도, 'context': 컨텍스트}
        """
        if not self.concepts:
            self.extract_concepts()
        
        if not self.concepts:
            return {'concept': '금융', 'frequency': 0, 'context': ''}
        
        # 빈도 기반 가중치 선택
        if self.concept_frequencies:
            concepts = list(self.concept_frequencies.keys())
            weights = [freq ** 0.5 for freq in self.concept_frequencies.values()]
            
            chosen = random.choices(concepts, weights=weights)[0]
        else:
            chosen = random.choice(list(self.concepts))
        
        return {
            'concept': chosen,
            'frequency': self.concept_frequencies.get(chosen, 0),
            'context': self.concept_contexts.get(chosen, '')
        }
    
    def get_related_concepts(self, concept: str, n: int = 5) -> List[str]:
        """관련 개념 찾기"""
        related = []
        
        # 같은 카테고리 찾기
        for c in self.concepts:
            if c != concept:
                # 공통 단어가 있으면 관련
                if any(word in c for word in concept.split()):
                    related.append(c)
        
        return related[:n]