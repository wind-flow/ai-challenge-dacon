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
    
    def __init__(self, data_dir: str = "data", use_cache: bool = True):
        self.data_dir = Path(data_dir)
        self.concepts = set()
        self.concept_contexts = {}
        self.concept_frequencies = Counter()
        self.cache_path = self.data_dir / "cache" / "concepts.pkl"
        
        # 캐시 확인
        if use_cache and self.cache_path.exists():
            self._load_cache()
            logger.info(f"캐시에서 {len(self.concepts)}개 개념 로드")
            return
        
        # 패턴은 외부 데이터에서 자동 생성
        self.patterns = []
        self._build_patterns_from_external_data()
    
    def _build_patterns_from_external_data(self):
        """외부 데이터에서 패턴 자동 생성"""
        external_dir = self.data_dir / "external"
        
        if not external_dir.exists():
            logger.warning(f"외부 데이터 디렉토리 없음: {external_dir}")
            # 최소한의 범용 패턴만 사용 (수기 작성 아님)
            self.patterns = [
                r'\b[가-힣]{2,10}\b',  # 한글 단어 (2-10자)
                r'\b[A-Z]{2,10}\b',     # 영문 약어
                r'\b[0-9]+[가-힣]+\b',  # 숫자+한글 조합
            ]
            return
        
        # 외부 데이터에서 자주 나오는 단어 패턴 학습
        word_patterns = set()
        
        for file_path in external_dir.glob("*.txt"):
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    text = f.read()
                    
                    # 명사형 어미 패턴 자동 감지
                    # "~은", "~는", "~이", "~가" 앞의 단어들
                    noun_patterns = re.findall(r'([가-힣]+)(?:은|는|이|가)\s', text)
                    for noun in noun_patterns:
                        if 2 <= len(noun) <= 10:
                            word_patterns.add(noun)
                    
                    # "~란", "~라는" 으로 정의되는 용어
                    defined_terms = re.findall(r'([가-힣]+)(?:란|라는)\s', text)
                    for term in defined_terms:
                        if 2 <= len(term) <= 10:
                            word_patterns.add(term)
                    
                    # 반복되는 단어 패턴 감지
                    words = re.findall(r'[가-힣]{2,10}', text)
                    word_freq = Counter(words)
                    
                    # 빈도수 높은 상위 단어들을 패턴으로
                    for word, freq in word_freq.most_common(50):
                        if freq >= 3:  # 3번 이상 나온 단어
                            word_patterns.add(word)
                            
            except Exception as e:
                logger.error(f"파일 읽기 실패 {file_path}: {e}")
        
        # 수집한 단어들로 동적 패턴 생성
        if word_patterns:
            # 접미사 패턴 자동 생성 (빈번한 어미 찾기)
            suffixes = Counter()
            for word in word_patterns:
                if len(word) >= 3:
                    suffixes[word[-2:]] += 1
            
            # 빈번한 접미사로 패턴 생성
            for suffix, count in suffixes.most_common(10):
                if count >= 3:
                    self.patterns.append(f'[가-힣]+{suffix}')
        
        # 기본 패턴 추가
        self.patterns.extend([
            r'\b[가-힣]{2,10}\b',  # 한글 단어
            r'\b[A-Z]{2,10}\b',     # 영문 약어
        ])
        
        logger.info(f"외부 데이터에서 {len(self.patterns)}개 패턴 생성")
    
    def extract_concepts(self) -> Set[str]:
        """
        외부 데이터에서 개념 추출 (PDF 포함)
        
        Returns:
            추출된 개념 집합
        """
        # 이미 개념이 있으면 재사용
        if self.concepts:
            return self.concepts
            
        # 외부 데이터 디렉토리
        external_dir = self.data_dir / "external"
        
        if not external_dir.exists():
            logger.warning(f"외부 데이터 디렉토리 없음: {external_dir}")
            logger.warning("data/external/ 폴더에 금융 관련 문서를 추가하세요.")
            return set()
        
        # PDF 파일 처리 (새로 추가)
        for file_path in external_dir.glob("*.pdf"):
            self._process_pdf_file(file_path)
        
        # 모든 텍스트 파일 처리
        for file_path in external_dir.glob("*.txt"):
            self._process_text_file(file_path)
        
        # JSON 파일 처리
        for file_path in external_dir.glob("*.json"):
            self._process_json_file(file_path)
        
        # Excel 파일 처리 (새로 추가)
        for file_path in external_dir.glob("*.xlsx"):
            self._process_excel_file(file_path)
        
        # CSV 파일 처리 (train.csv만, test.csv 제외)
        raw_dir = self.data_dir / "raw"
        if raw_dir.exists():
            for file_path in raw_dir.glob("*.csv"):
                if "test" not in file_path.name.lower():
                    self._process_csv_file(file_path)
        
        logger.info(f"총 {len(self.concepts)}개 개념 추출")
        
        # 캐시 저장
        if self.concepts:
            self._save_cache()
        
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
    
    def _process_pdf_file(self, file_path: Path):
        """PDF 파일 처리"""
        try:
            # PDF 로더 사용
            try:
                from rag.pdf_loader import DocumentLoader
            except ImportError:
                from ..rag.pdf_loader import DocumentLoader
            
            loader = DocumentLoader()
            doc = loader.load_document(file_path)
            
            if doc and 'content' in doc:
                self._extract_from_text(doc['content'])
                logger.info(f"PDF 처리 완료: {file_path.name}")
        except Exception as e:
            logger.error(f"PDF 파일 읽기 실패 {file_path}: {e}")
    
    def _process_excel_file(self, file_path: Path):
        """Excel 파일 처리"""
        try:
            import pandas as pd
            # 모든 시트 읽기
            dfs = pd.read_excel(file_path, sheet_name=None)
            
            for sheet_name, df in dfs.items():
                # 데이터프레임을 텍스트로 변환
                text = df.to_string()
                self._extract_from_text(text)
                
                # 컬럼명도 개념으로 추출
                for col in df.columns:
                    if isinstance(col, str) and 2 <= len(col) <= 20:
                        self.concepts.add(col)
                        self.concept_frequencies[col] += 1
            
            logger.info(f"Excel 처리 완료: {file_path.name}")
        except Exception as e:
            logger.error(f"Excel 파일 읽기 실패 {file_path}: {e}")
    
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
    
    def _save_cache(self):
        """캐시 저장"""
        import pickle
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        
        cache_data = {
            'concepts': self.concepts,
            'concept_contexts': self.concept_contexts,
            'concept_frequencies': self.concept_frequencies,
            'patterns': self.patterns
        }
        
        with open(self.cache_path, 'wb') as f:
            pickle.dump(cache_data, f)
        logger.info(f"캐시 저장: {self.cache_path}")
    
    def _load_cache(self):
        """캐시 로드"""
        import pickle
        
        with open(self.cache_path, 'rb') as f:
            cache_data = pickle.load(f)
        
        self.concepts = cache_data['concepts']
        self.concept_contexts = cache_data['concept_contexts']
        self.concept_frequencies = cache_data['concept_frequencies']
        self.patterns = cache_data.get('patterns', [])