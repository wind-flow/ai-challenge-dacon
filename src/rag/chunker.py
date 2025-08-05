#!/usr/bin/env python3
"""
문서 청킹 모듈

긴 문서를 의미 있는 단위로 분할
"""

import re
from typing import List, Dict, Optional
import logging

logger = logging.getLogger(__name__)

# 한국어 토크나이저 import
try:
    from .korean_utils import get_best_tokenizer, split_korean_sentences
except ImportError:
    from korean_utils import get_best_tokenizer, split_korean_sentences


class DocumentChunker:
    """문서 청킹 클래스"""
    
    def __init__(self, 
                 chunk_size: int = 300,  # Colab과 동일하게 축소
                 chunk_overlap: int = 50,  # 오버랩 감소
                 min_chunk_size: int = 50,  # 최소 크기도 감소
                 use_korean_tokenizer: bool = True):
        """
        초기화
        
        Args:
            chunk_size: 청크 크기 (토큰 기준)
            chunk_overlap: 오버랩 크기
            min_chunk_size: 최소 청크 크기
            use_korean_tokenizer: 한국어 토크나이저 사용 여부
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.min_chunk_size = min_chunk_size
        
        # 한국어 토크나이저 초기화
        if use_korean_tokenizer:
            try:
                self.tokenizer = get_best_tokenizer()
                logger.info(f"한국어 토크나이저 사용: {self.tokenizer.tokenizer_type}")
            except:
                self.tokenizer = None
                logger.info("기본 토크나이저 사용")
        else:
            self.tokenizer = None
        
    def chunk_text(self, text: str, metadata: Dict = None) -> List[Dict]:
        """
        텍스트를 청크로 분할
        
        Args:
            text: 원본 텍스트
            metadata: 문서 메타데이터
            
        Returns:
            청크 리스트
        """
        if not text or len(text.strip()) < self.min_chunk_size:
            return []
        
        # 문장 단위로 분할
        sentences = self._split_into_sentences(text)
        
        # 청크 생성
        chunks = []
        current_chunk = []
        current_length = 0
        
        for sentence in sentences:
            sentence_length = self._estimate_tokens(sentence)
            
            # 현재 청크가 크기를 초과하면 새 청크 시작
            if current_length + sentence_length > self.chunk_size and current_chunk:
                # 청크 저장
                chunk_text = ' '.join(current_chunk)
                chunks.append(self._create_chunk(chunk_text, len(chunks), metadata))
                
                # 오버랩 처리
                if self.chunk_overlap > 0:
                    # 오버랩 크기만큼 이전 문장 유지
                    overlap_sentences = []
                    overlap_length = 0
                    
                    for sent in reversed(current_chunk):
                        sent_len = self._estimate_tokens(sent)
                        if overlap_length + sent_len <= self.chunk_overlap:
                            overlap_sentences.insert(0, sent)
                            overlap_length += sent_len
                        else:
                            break
                    
                    current_chunk = overlap_sentences
                    current_length = overlap_length
                else:
                    current_chunk = []
                    current_length = 0
            
            current_chunk.append(sentence)
            current_length += sentence_length
        
        # 마지막 청크 처리
        if current_chunk:
            chunk_text = ' '.join(current_chunk)
            if len(chunk_text.strip()) >= self.min_chunk_size:
                chunks.append(self._create_chunk(chunk_text, len(chunks), metadata))
        
        return chunks
    
    def chunk_document(self, document: Dict) -> List[Dict]:
        """
        문서를 청크로 분할
        
        Args:
            document: 문서 딕셔너리 (content, metadata)
            
        Returns:
            청크 리스트
        """
        content = document.get('content', '')
        metadata = document.get('metadata', {})
        
        # PDF인 경우 페이지별 처리 고려
        if metadata.get('type') == 'pdf' and 'pages' in document:
            chunks = []
            for page_info in document['pages']:
                page_chunks = self.chunk_text(
                    page_info['content'],
                    {**metadata, 'page': page_info['page']}
                )
                chunks.extend(page_chunks)
            return chunks
        else:
            # 일반 텍스트 처리
            return self.chunk_text(content, metadata)
    
    def _split_into_sentences(self, text: str) -> List[str]:
        """
        텍스트를 문장으로 분할
        
        Args:
            text: 원본 텍스트
            
        Returns:
            문장 리스트
        """
        try:
            # 한국어 특화 문장 분리 사용
            sentences = split_korean_sentences(text)
        except:
            # 기본 문장 분할
            sentence_endings = r'[.!?。]\s+'
            
            # 특수 케이스 처리 (예: 조항 번호)
            text = re.sub(r'(\d+)\.\s*(\d+)', r'\1_\2', text)  # 1.2 -> 1_2
            text = re.sub(r'(\d+)\.\s*([가-힣])', r'\1_\2', text)  # 1.가 -> 1_가
            
            # 문장 분할
            sentences = re.split(sentence_endings, text)
            
            # 특수 케이스 복원
            sentences = [s.replace('_', '.') for s in sentences]
            
            # 빈 문장 제거
            sentences = [s.strip() for s in sentences if s.strip()]
        
        # 너무 짧은 문장은 다음 문장과 결합
        combined_sentences = []
        temp_sentence = ""
        
        for sentence in sentences:
            if len(sentence) < 20 and temp_sentence:
                temp_sentence += " " + sentence
            else:
                if temp_sentence:
                    combined_sentences.append(temp_sentence)
                temp_sentence = sentence
        
        if temp_sentence:
            combined_sentences.append(temp_sentence)
        
        return combined_sentences
    
    def _estimate_tokens(self, text: str) -> int:
        """
        텍스트의 토큰 수 추정
        
        Args:
            text: 텍스트
            
        Returns:
            추정 토큰 수
        """
        if self.tokenizer:
            # 한국어 토크나이저 사용
            try:
                return self.tokenizer.estimate_tokens(text)
            except:
                pass
        
        # 기본 추정: 한글 2-3자 = 1토큰, 영어 4자 = 1토큰
        korean_chars = len(re.findall(r'[가-힣]', text))
        english_chars = len(re.findall(r'[a-zA-Z]', text))
        numbers = len(re.findall(r'\d', text))
        
        estimated_tokens = (korean_chars / 2.5) + (english_chars / 4) + (numbers / 3)
        return int(estimated_tokens)
    
    def _create_chunk(self, text: str, index: int, metadata: Dict = None) -> Dict:
        """
        청크 딕셔너리 생성
        
        Args:
            text: 청크 텍스트
            index: 청크 인덱스
            metadata: 메타데이터
            
        Returns:
            청크 딕셔너리
        """
        chunk = {
            'content': text.strip(),
            'chunk_id': index,
            'tokens': self._estimate_tokens(text),
            'metadata': metadata or {}
        }
        
        # 청크에서 핵심 키워드 추출 (선택적)
        keywords = self._extract_keywords(text)
        if keywords:
            chunk['keywords'] = keywords
        
        return chunk
    
    def _extract_keywords(self, text: str) -> List[str]:
        """
        텍스트에서 핵심 키워드 추출
        
        Args:
            text: 텍스트
            
        Returns:
            키워드 리스트
        """
        keywords = []
        
        # 조항 번호 패턴
        article_patterns = re.findall(r'제\d+조', text)
        keywords.extend(article_patterns)
        
        # 특수 용어 패턴 (괄호 안의 정의)
        definitions = re.findall(r'[가-힣]+(?:\([가-힣]+\))', text)
        keywords.extend(definitions[:5])  # 상위 5개만
        
        return keywords


if __name__ == "__main__":
    # 테스트
    chunker = DocumentChunker(chunk_size=300, chunk_overlap=50)
    
    test_text = """
    제1조(목적) 이 규정은 전자금융거래의 안전성과 신뢰성을 확보하기 위하여 
    금융회사 및 전자금융업자가 준수하여야 할 기준을 정함을 목적으로 한다.
    
    제2조(정의) 이 규정에서 사용하는 용어의 정의는 다음과 같다.
    1. "전자금융거래"란 금융회사 또는 전자금융업자가 전자적 장치를 통하여 
    금융상품 및 서비스를 제공하고, 이용자가 금융회사 또는 전자금융업자의 
    종사자와 직접 대면하거나 의사소통을 하지 아니하고 자동화된 방식으로 
    이를 이용하는 거래를 말한다.
    """
    
    chunks = chunker.chunk_text(test_text, {'source': 'test.txt'})
    
    for i, chunk in enumerate(chunks):
        print(f"\n청크 {i+1}:")
        print(f"- 토큰 수: {chunk['tokens']}")
        print(f"- 키워드: {chunk.get('keywords', [])}")
        print(f"- 내용: {chunk['content'][:100]}...")