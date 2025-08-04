#!/usr/bin/env python3
"""
RAG (Retrieval Augmented Generation) 문서 검색 모듈

외부 문서에서 관련 정보를 검색하여 생성 품질 향상
"""

import json
import re
from pathlib import Path
from typing import List, Dict, Optional, Tuple
import logging
from collections import defaultdict

logger = logging.getLogger(__name__)


class DocumentRetriever:
    """문서 검색 및 관리 클래스"""
    
    def __init__(self, data_dir: str = "data"):
        self.data_dir = Path(data_dir)
        self.documents = []
        self.document_index = defaultdict(list)  # 키워드 -> 문서 인덱스
        
        # 문서 로드
        self._load_documents()
        self._build_index()
    
    def _load_documents(self):
        """외부 문서 로드"""
        external_dir = self.data_dir / "external"
        
        if not external_dir.exists():
            logger.warning(f"외부 데이터 디렉토리 없음: {external_dir}")
            return
        
        # 텍스트 파일
        for file_path in external_dir.glob("*.txt"):
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                    self.documents.append({
                        'source': file_path.name,
                        'content': content,
                        'type': 'text'
                    })
            except Exception as e:
                logger.error(f"파일 로드 실패 {file_path}: {e}")
        
        # JSON 파일
        for file_path in external_dir.glob("*.json"):
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    # JSON을 텍스트로 변환
                    content = json.dumps(data, ensure_ascii=False)
                    self.documents.append({
                        'source': file_path.name,
                        'content': content,
                        'type': 'json'
                    })
            except Exception as e:
                logger.error(f"JSON 로드 실패 {file_path}: {e}")
        
        logger.info(f"총 {len(self.documents)}개 문서 로드")
    
    def _build_index(self):
        """문서 인덱스 구축 (간단한 역인덱스)"""
        for idx, doc in enumerate(self.documents):
            content = doc['content'].lower()
            
            # 단어 추출 (간단한 토크나이징)
            words = re.findall(r'[가-힣]+|[a-zA-Z]+', content)
            
            # 각 단어에 대해 문서 인덱스 저장
            for word in set(words):
                if len(word) >= 2:  # 2글자 이상만
                    self.document_index[word].append(idx)
        
        logger.info(f"인덱스 구축 완료: {len(self.document_index)}개 키워드")
    
    def search(self, query: str, top_k: int = 3) -> str:
        """
        관련 문서 검색
        
        Args:
            query: 검색 쿼리
            top_k: 반환할 문서 수
            
        Returns:
            관련 컨텍스트 문자열
        """
        if not self.documents:
            return ""
        
        # 쿼리 토크나이징
        query_lower = query.lower()
        query_words = re.findall(r'[가-힣]+|[a-zA-Z]+', query_lower)
        
        # 문서 점수 계산
        doc_scores = defaultdict(float)
        
        for word in query_words:
            if word in self.document_index:
                # 해당 단어를 포함한 문서들
                for doc_idx in self.document_index[word]:
                    doc_scores[doc_idx] += 1
        
        # 전체 쿼리 매칭 (더 높은 점수)
        for idx, doc in enumerate(self.documents):
            if query_lower in doc['content'].lower():
                doc_scores[idx] += 5
        
        if not doc_scores:
            # 점수가 있는 문서가 없으면 랜덤 선택
            import random
            selected_docs = random.sample(
                range(len(self.documents)),
                min(top_k, len(self.documents))
            )
        else:
            # 점수 기준 정렬
            sorted_docs = sorted(
                doc_scores.items(),
                key=lambda x: x[1],
                reverse=True
            )
            selected_docs = [idx for idx, _ in sorted_docs[:top_k]]
        
        # 컨텍스트 추출
        contexts = []
        for doc_idx in selected_docs:
            doc = self.documents[doc_idx]
            context = self._extract_context(doc['content'], query)
            
            if context:
                contexts.append(f"[출처: {doc['source']}]\n{context}")
        
        return "\n\n---\n\n".join(contexts)
    
    def _extract_context(self, content: str, query: str, max_length: int = 500) -> str:
        """
        문서에서 쿼리 관련 부분 추출
        
        Args:
            content: 문서 내용
            query: 검색 쿼리
            max_length: 최대 컨텍스트 길이
            
        Returns:
            추출된 컨텍스트
        """
        # 쿼리 주변 텍스트 추출
        query_lower = query.lower()
        content_lower = content.lower()
        
        idx = content_lower.find(query_lower)
        
        if idx != -1:
            # 쿼리 주변 추출
            start = max(0, idx - 100)
            end = min(len(content), idx + max_length - 100)
            excerpt = content[start:end]
            
            # 문장 단위로 자르기
            sentences = re.split(r'[.。!?]', excerpt)
            if sentences:
                # 쿼리가 포함된 문장들만
                relevant_sentences = [
                    s.strip() for s in sentences
                    if query_lower in s.lower()
                ]
                if relevant_sentences:
                    return '. '.join(relevant_sentences[:3])
        
        # 쿼리를 찾지 못한 경우 처음 부분 반환
        return content[:max_length]
    
    def add_document(self, content: str, source: str = "manual"):
        """
        문서 추가
        
        Args:
            content: 문서 내용
            source: 문서 출처
        """
        self.documents.append({
            'source': source,
            'content': content,
            'type': 'text'
        })
        
        # 인덱스 재구축
        self._build_index()
    
    def get_statistics(self) -> Dict:
        """검색 통계 반환"""
        return {
            'total_documents': len(self.documents),
            'total_keywords': len(self.document_index),
            'sources': list(set(doc['source'] for doc in self.documents))
        }