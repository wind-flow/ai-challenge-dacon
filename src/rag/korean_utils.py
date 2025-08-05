#!/usr/bin/env python3
"""
한국어 특화 유틸리티

토크나이저, 임베딩 모델 등 한국어 최적화 기능
"""

import re
from typing import List, Optional
import logging

logger = logging.getLogger(__name__)


class KoreanTokenizer:
    """한국어 특화 토크나이저"""
    
    def __init__(self, tokenizer_type: str = "mecab"):
        """
        초기화
        
        Args:
            tokenizer_type: 토크나이저 종류 (mecab, kkma, hannanum, okt)
        """
        self.tokenizer_type = tokenizer_type
        self.tokenizer = None
        
        # 토크나이저 초기화
        self._initialize_tokenizer()
    
    def _initialize_tokenizer(self):
        """토크나이저 초기화"""
        try:
            if self.tokenizer_type == "mecab":
                # python-mecab-ko 우선 시도 (가장 빠름)
                try:
                    import mecab
                    self.tokenizer = mecab.MeCab()
                    logger.info("python-mecab-ko 토크나이저 초기화")
                    self.use_python_mecab = True
                except:
                    # KoNLPy MeCab 시도
                    try:
                        from konlpy.tag import Mecab
                        self.tokenizer = Mecab()
                        logger.info("KoNLPy MeCab 토크나이저 초기화")
                        self.use_python_mecab = False
                    except:
                        logger.warning("MeCab 사용 불가, Okt로 대체")
                        self.tokenizer_type = "okt"
            
            if self.tokenizer_type == "okt" or self.tokenizer is None:
                # Okt (구 Twitter, 균형잡힌 성능)
                from konlpy.tag import Okt
                self.tokenizer = Okt()
                logger.info("Okt 토크나이저 초기화")
                self.use_python_mecab = False
                
        except ImportError:
            logger.warning("KoNLPy가 설치되지 않았습니다.")
            logger.warning("pip install konlpy")
            logger.info("기본 토크나이저 사용 (공백 기반)")
            self.tokenizer = None
            self.use_python_mecab = False
    
    def tokenize(self, text: str) -> List[str]:
        """
        텍스트를 토큰으로 분리
        
        Args:
            text: 입력 텍스트
            
        Returns:
            토큰 리스트
        """
        if self.tokenizer is None:
            # 기본 토크나이저 (공백 + 기본 규칙)
            return self._basic_tokenize(text)
        
        try:
            # python-mecab-ko 사용
            if hasattr(self, 'use_python_mecab') and self.use_python_mecab:
                return self.tokenizer.morphs(text)
            # KoNLPy 토크나이저 사용
            elif hasattr(self.tokenizer, 'morphs'):
                return self.tokenizer.morphs(text)
            elif hasattr(self.tokenizer, 'pos'):
                # 형태소 분석 후 토큰만 추출
                return [word for word, _ in self.tokenizer.pos(text)]
        except Exception as e:
            logger.debug(f"토크나이저 오류: {e}")
            return self._basic_tokenize(text)
    
    def _basic_tokenize(self, text: str) -> List[str]:
        """
        기본 토크나이저 (KoNLPy 없을 때)
        
        Args:
            text: 입력 텍스트
            
        Returns:
            토큰 리스트
        """
        # 한글, 영어, 숫자 단위로 분리
        tokens = re.findall(r'[가-힣]+|[a-zA-Z]+|\d+', text)
        return tokens
    
    def count_tokens(self, text: str) -> int:
        """
        토큰 수 계산
        
        Args:
            text: 입력 텍스트
            
        Returns:
            토큰 수
        """
        tokens = self.tokenize(text)
        return len(tokens)
    
    def estimate_tokens(self, text: str) -> int:
        """
        빠른 토큰 수 추정 (정확도 낮지만 빠름)
        
        Args:
            text: 입력 텍스트
            
        Returns:
            추정 토큰 수
        """
        # 한국어 어절 기준 추정
        # 평균적으로 한국어 어절 = 1.5~2 토큰
        words = text.split()
        korean_words = len([w for w in words if any(ord('가') <= ord(c) <= ord('힣') for c in w)])
        english_words = len([w for w in words if w.isalpha() and not any(ord('가') <= ord(c) <= ord('힣') for c in w)])
        
        estimated = korean_words * 1.8 + english_words * 1.2
        return int(estimated)


class KoreanEmbedding:
    """한국어 특화 임베딩"""
    
    # 추천 한국어 임베딩 모델
    RECOMMENDED_MODELS = {
        'kcbert': 'beomi/kcbert-base',  # 한국어 BERT (추천)
        'kobert': 'skt/kobert-base-v1',  # SKT KoBERT
        'klue': 'klue/roberta-base',  # KLUE RoBERTa
        'korsimcse': 'BM-K/KoSimCSE-roberta',  # 한국어 SimCSE
        'multilingual': 'sentence-transformers/xlm-r-100langs-bert-base-nli-stsb-mean-tokens'  # 다국어
    }
    
    def __init__(self, model_type: str = "kcbert"):
        """
        초기화
        
        Args:
            model_type: 모델 타입
        """
        self.model_type = model_type
        self.model = None
        self.model_name = self.RECOMMENDED_MODELS.get(model_type, model_type)
        
        self._initialize_model()
    
    def _initialize_model(self):
        """모델 초기화"""
        try:
            from sentence_transformers import SentenceTransformer
            
            logger.info(f"임베딩 모델 로드: {self.model_name}")
            self.model = SentenceTransformer(self.model_name)
            logger.info("임베딩 모델 초기화 완료")
            
        except ImportError:
            logger.warning("sentence-transformers가 설치되지 않았습니다.")
            logger.warning("pip install sentence-transformers")
            self.model = None
        except Exception as e:
            logger.error(f"모델 로드 실패: {e}")
            # 대체 모델 시도
            if self.model_type != 'multilingual':
                logger.info("다국어 모델로 대체 시도...")
                self.model_name = self.RECOMMENDED_MODELS['multilingual']
                try:
                    self.model = SentenceTransformer(self.model_name)
                    logger.info("다국어 모델 로드 성공")
                except:
                    self.model = None
    
    def encode(self, texts: List[str], show_progress: bool = False) -> Optional[any]:
        """
        텍스트를 벡터로 인코딩
        
        Args:
            texts: 텍스트 리스트
            show_progress: 진행 표시 여부
            
        Returns:
            임베딩 벡터
        """
        if self.model is None:
            logger.error("임베딩 모델이 로드되지 않았습니다.")
            return None
        
        try:
            embeddings = self.model.encode(
                texts,
                show_progress_bar=show_progress,
                convert_to_numpy=True
            )
            return embeddings
        except Exception as e:
            logger.error(f"임베딩 생성 실패: {e}")
            return None
    
    def get_model_info(self) -> dict:
        """모델 정보 반환"""
        return {
            'type': self.model_type,
            'name': self.model_name,
            'loaded': self.model is not None
        }


def get_best_tokenizer() -> KoreanTokenizer:
    """
    사용 가능한 최적의 토크나이저 반환
    
    Returns:
        KoreanTokenizer 인스턴스
    """
    # MeCab > Okt > 기본 순서로 시도
    for tokenizer_type in ['mecab', 'okt']:
        tokenizer = KoreanTokenizer(tokenizer_type)
        if tokenizer.tokenizer is not None:
            return tokenizer
    
    # 모두 실패시 기본 토크나이저
    return KoreanTokenizer('basic')


def get_best_embedding() -> KoreanEmbedding:
    """
    사용 가능한 최적의 임베딩 모델 반환
    
    Returns:
        KoreanEmbedding 인스턴스
    """
    # kcbert > korsimcse > multilingual 순서로 시도
    for model_type in ['kcbert', 'korsimcse', 'multilingual']:
        embedding = KoreanEmbedding(model_type)
        if embedding.model is not None:
            return embedding
    
    # 모두 실패시 None
    return KoreanEmbedding('multilingual')


# 한국어 문장 분리 패턴
KOREAN_SENTENCE_ENDINGS = re.compile(
    r'([.!?。]+)\s*'  # 기본 문장 종결
    r'|([다요죠]+[.!?])\s*'  # 한국어 종결어미
    r'|(\n{2,})'  # 두 줄 이상 개행
)


def split_korean_sentences(text: str) -> List[str]:
    """
    한국어 문장 분리 (KSS 우선, 없으면 기본)
    
    Args:
        text: 입력 텍스트
        
    Returns:
        문장 리스트
    """
    try:
        # KSS (Korean Sentence Splitter) 사용 - 가장 정확함
        import kss
        sentences = kss.split_sentences(text)
        logger.debug("KSS를 사용한 문장 분리")
        return sentences
    except ImportError:
        logger.debug("KSS 미설치, 기본 문장 분리 사용")
        # 특수 케이스 보호 (예: 1.2, 3.가 등)
        protected = text
        protected = re.sub(r'(\d+)\.(\d+)', r'\1§§§\2', protected)  # 소수점
        protected = re.sub(r'(\d+)\.([가-힣])', r'\1§§§\2', protected)  # 번호
        
        # 문장 분리
        sentences = KOREAN_SENTENCE_ENDINGS.split(protected)
        
        # 빈 문장 제거 및 복원
        result = []
        for sent in sentences:
            if sent and sent.strip():
                restored = sent.replace('§§§', '.')
                result.append(restored.strip())
        
        return result


if __name__ == "__main__":
    # 테스트
    print("="*60)
    print("한국어 특화 유틸리티 테스트")
    print("="*60)
    
    # 토크나이저 테스트
    print("\n1. 토크나이저 테스트")
    tokenizer = get_best_tokenizer()
    test_text = "금융위원회는 전자금융거래법을 개정했습니다."
    tokens = tokenizer.tokenize(test_text)
    print(f"원문: {test_text}")
    print(f"토큰: {tokens}")
    print(f"토큰 수: {len(tokens)}")
    
    # 임베딩 테스트
    print("\n2. 임베딩 모델 테스트")
    embedding = get_best_embedding()
    info = embedding.get_model_info()
    print(f"모델: {info['name']}")
    print(f"로드 상태: {'성공' if info['loaded'] else '실패'}")
    
    # 문장 분리 테스트
    print("\n3. 문장 분리 테스트")
    test_paragraph = """제1조(목적) 이 법은 전자금융거래를 규정합니다.
    제2조(정의) 1. 전자금융거래란 금융회사가 제공하는 서비스입니다. 2. 금융회사는 은행을 포함합니다."""
    sentences = split_korean_sentences(test_paragraph)
    for i, sent in enumerate(sentences, 1):
        print(f"{i}. {sent}")