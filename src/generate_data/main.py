#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
FSKU 데이터 증강 시스템 - 개선된 버전
- 모델 사전 로딩 ✅
- 동작 검증 포함 ✅
- 에러 처리 강화 ✅
"""

import warnings
warnings.filterwarnings('ignore')

import os
import sys
import json
import time
import pickle
import traceback
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Optional, Any, Tuple
from collections import defaultdict
import random
import logging
import argparse

# 데이터 처리
import numpy as np
import pandas as pd
from tqdm import tqdm

# 딥러닝 및 NLP
import torch
from transformers import (
    AutoModelForCausalLM, 
    AutoTokenizer, 
    BitsAndBytesConfig,
    set_seed
)

# 벡터 검색 및 임베딩 (RAG용)
from sentence_transformers import SentenceTransformer
import faiss

# 문서 처리
import PyPDF2

# 로깅 설정
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('fsku_generation.log')
    ]
)
logger = logging.getLogger(__name__)

# 재현성을 위한 시드 설정
set_seed(42)
random.seed(42)
np.random.seed(42)

# 경로 설정
BASE_DIR = Path(__file__).parent.parent.parent  # ai-dacon 폴더
EXTERNAL_DIR = BASE_DIR / "data" / "external"
OUTPUT_DIR = BASE_DIR / "data" / "augmented"
CACHE_DIR = BASE_DIR / "data" / "cache"
MODEL_CACHE_DIR = BASE_DIR / "models" / "cache"

# 디렉토리 생성
for dir_path in [OUTPUT_DIR, CACHE_DIR, MODEL_CACHE_DIR]:
    dir_path.mkdir(parents=True, exist_ok=True)


class ModelManager:
    """
    모델 관리 클래스 - 사전 로딩 및 캐싱
    """
    _instance = None
    _models = {}
    _tokenizers = {}
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(ModelManager, cls).__new__(cls)
        return cls._instance
    
    def load_model(self, 
                   model_name: str, 
                   use_quantization: bool = False,
                   force_reload: bool = False) -> Tuple[Any, Any]:
        """
        모델과 토크나이저 로드 (캐싱 지원)
        
        Args:
            model_name: 모델 이름
            use_quantization: 양자화 사용 여부
            force_reload: 강제 재로드
            
        Returns:
            (model, tokenizer) 튜플
        """
        cache_key = f"{model_name}_{use_quantization}"
        
        # 캐시 확인
        if not force_reload and cache_key in self._models:
            logger.info(f"📦 캐시에서 모델 로드: {model_name}")
            return self._models[cache_key], self._tokenizers[cache_key]
        
        logger.info(f"🚀 새로운 모델 로드 시작: {model_name}")
        start_time = time.time()
        
        try:
            # 토크나이저 로드
            tokenizer = AutoTokenizer.from_pretrained(
                model_name,
                trust_remote_code=True,
                cache_dir=MODEL_CACHE_DIR
            )
            
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
                tokenizer.pad_token_id = tokenizer.eos_token_id
            
            # 모델 로드 설정
            model_kwargs = {
                "trust_remote_code": True,
                "device_map": "auto",
                "cache_dir": MODEL_CACHE_DIR
            }
            
            # GPU 최적화
            if torch.cuda.is_available():
                # H100/A100은 bfloat16, 나머지는 float16
                compute_capability = torch.cuda.get_device_capability()
                if compute_capability[0] >= 8:  # Ampere 이상
                    model_kwargs["torch_dtype"] = torch.bfloat16
                    logger.info("🔥 bfloat16 사용 (H100/A100)")
                else:
                    model_kwargs["torch_dtype"] = torch.float16
                    logger.info("🔥 float16 사용")
            
            # 양자화 설정
            if use_quantization:
                bnb_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_compute_dtype=model_kwargs.get("torch_dtype", torch.float16),
                    bnb_4bit_use_double_quant=True
                )
                model_kwargs["quantization_config"] = bnb_config
                logger.info("💎 4bit 양자화 활성화")
            
            # 모델 로드
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                **model_kwargs
            )
            
            # 캐시에 저장
            self._models[cache_key] = model
            self._tokenizers[cache_key] = tokenizer
            
            load_time = time.time() - start_time
            
            # 메모리 정보
            if torch.cuda.is_available():
                memory_gb = torch.cuda.memory_allocated() / 1024**3
                max_memory_gb = torch.cuda.max_memory_allocated() / 1024**3
                logger.info(f"✅ 로드 완료! (시간: {load_time:.1f}초)")
                logger.info(f"💾 현재 메모리: {memory_gb:.2f}GB / 최대: {max_memory_gb:.2f}GB")
            else:
                logger.info(f"✅ CPU 로드 완료! ({load_time:.1f}초)")
            
            return model, tokenizer
            
        except Exception as e:
            logger.error(f"❌ 모델 로드 실패: {e}")
            logger.error(traceback.format_exc())
            raise
    
    def clear_cache(self):
        """캐시 클리어"""
        self._models.clear()
        self._tokenizers.clear()
        torch.cuda.empty_cache()
        logger.info("🧹 모델 캐시 클리어 완료")


class SimpleDataGenerator:
    """
    개선된 FSKU 데이터 생성기
    - 모델 사전 로딩 ✅
    - 에러 처리 강화 ✅
    - 통계 추적 개선 ✅
    """
    
    def __init__(self, 
                 model_name: str = "microsoft/phi-2",
                 use_chaining: bool = False,
                 use_quantization: bool = False):
        """
        초기화
        
        Args:
            model_name: 사용할 모델명
            use_chaining: CoT 체이닝 사용 여부
            use_quantization: 4bit 양자화 사용 여부
        """
        self.model_name = model_name
        self.use_chaining = use_chaining
        self.use_quantization = use_quantization
        
        # 모델 매니저
        self.model_manager = ModelManager()
        self.model = None
        self.tokenizer = None
        
        # 프롬프트 템플릿
        self.simple_prompt = """주제: {question_type}

참고 내용:
{context}

위 내용을 바탕으로 {question_type} 문제를 1개 만드세요.

문제:
정답:"""
        
        self.chaining_prompt = """금융 전문가로서 FSKU 시험 문제를 생성하세요.

참고 문서:
{context}

요구사항:
- 문제 유형: {question_type}
- FSKU 실제 시험 수준
- 명확하고 정확한 표현

문제:
정답:
해설:"""
        
        # 통계
        self.stats = {
            'total': 0,
            'success': 0,
            'failed': 0,
            'errors': defaultdict(int),
            'generation_times': []
        }
        
    def initialize(self):
        """모델 초기화 (사전 로딩)"""
        logger.info("=" * 60)
        logger.info("🚀 데이터 생성기 초기화")
        logger.info(f"📋 설정: 모델={self.model_name}, 체이닝={self.use_chaining}, 양자화={self.use_quantization}")
        logger.info("=" * 60)
        
        try:
            # 모델 로드
            self.model, self.tokenizer = self.model_manager.load_model(
                self.model_name,
                self.use_quantization
            )
            
            # 동작 검증
            if not self.verify_model():
                raise RuntimeError("모델 동작 검증 실패")
            
            logger.info("✅ 데이터 생성기 초기화 완료!")
            return True
            
        except Exception as e:
            logger.error(f"❌ 초기화 실패: {e}")
            return False
    
    def verify_model(self) -> bool:
        """모델 동작 검증"""
        logger.info("🔍 모델 동작 검증 중...")
        
        try:
            # 간단한 테스트 생성
            test_prompt = "안녕하세요. 테스트입니다."
            test_output = self.generate_text(test_prompt, max_tokens=10)
            
            if test_output and len(test_output) > 0:
                logger.info(f"✅ 모델 검증 성공: '{test_output[:30]}...'")
                return True
            else:
                logger.error("❌ 모델 검증 실패: 출력이 비어있음")
                return False
                
        except Exception as e:
            logger.error(f"❌ 모델 검증 중 오류: {e}")
            return False
    
    def generate_text(self, prompt: str, max_tokens: int = 150) -> str:
        """텍스트 생성"""
        if not self.model or not self.tokenizer:
            raise ValueError("모델이 초기화되지 않았습니다. initialize()를 먼저 호출하세요.")
        
        try:
            # 토큰화
            inputs = self.tokenizer(
                prompt,
                return_tensors="pt",
                truncation=True,
                max_length=800
            )
            
            # GPU로 이동
            if torch.cuda.is_available():
                inputs = {k: v.cuda() for k, v in inputs.items()}
            
            # 생성
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=max_tokens,
                    temperature=0.7,
                    top_p=0.9,
                    do_sample=True,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id
                )
            
            # 디코딩
            generated = self.tokenizer.decode(
                outputs[0][inputs['input_ids'].shape[1]:],
                skip_special_tokens=True
            )
            
            return generated.strip()
            
        except Exception as e:
            logger.error(f"텍스트 생성 오류: {e}")
            self.stats['errors']['generation'] += 1
            return ""
    
    def generate_qa_pair(self, context: str, question_type: str = "객관식") -> Optional[Dict]:
        """QA 쌍 생성"""
        self.stats['total'] += 1
        start_time = time.time()
        
        try:
            if self.use_chaining:
                result = self._generate_with_chaining(context, question_type)
            else:
                result = self._generate_simple(context, question_type)
            
            if result:
                generation_time = time.time() - start_time
                result['generation_time'] = generation_time
                self.stats['generation_times'].append(generation_time)
                self.stats['success'] += 1
                return result
            else:
                self.stats['failed'] += 1
                return None
                
        except Exception as e:
            logger.error(f"QA 생성 오류: {e}")
            self.stats['failed'] += 1
            self.stats['errors']['qa_generation'] += 1
            return None
    
    def _generate_simple(self, context: str, question_type: str) -> Optional[Dict]:
        """단순 생성 (1회 호출)"""
        prompt = self.simple_prompt.format(
            context=context[:400],
            question_type=question_type
        )
        
        generated = self.generate_text(prompt)
        if not generated:
            return None
        
        qa_pair = self._parse_qa(generated)
        if not qa_pair:
            self.stats['errors']['parsing'] += 1
            return None
        
        return {
            'question': qa_pair['question'],
            'answer': qa_pair['answer'],
            'context': context,
            'question_type': question_type,
            'method': 'simple',
            'timestamp': datetime.now().isoformat()
        }
    
    def _generate_with_chaining(self, context: str, question_type: str) -> Optional[Dict]:
        """체이닝 생성 (다중 호출)"""
        # 1. 초기 생성
        prompt = self.chaining_prompt.format(
            context=context[:600],
            question_type=question_type
        )
        generated = self.generate_text(prompt, max_tokens=200)
        
        if not generated:
            return None
        
        qa_pair = self._parse_qa_detailed(generated)
        if not qa_pair:
            self.stats['errors']['parsing'] += 1
            return None
        
        # 2. 검증
        verification_prompt = f"""다음 문제를 검토하세요:

문제: {qa_pair['question']}
정답: {qa_pair['answer']}

문제점이 있으면 지적하고, 없으면 "적합"이라고 답하세요:"""
        
        feedback = self.generate_text(verification_prompt, max_tokens=100)
        
        return {
            'question': qa_pair['question'],
            'answer': qa_pair['answer'],
            'explanation': qa_pair.get('explanation', ''),
            'context': context,
            'question_type': question_type,
            'method': 'chaining',
            'feedback': feedback,
            'timestamp': datetime.now().isoformat()
        }
    
    def _parse_qa(self, text: str) -> Optional[Dict]:
        """간단한 QA 파싱"""
        try:
            # 다양한 구분자 시도
            for q_marker in ['문제:', '질문:', 'Q:', 'Question:']:
                if q_marker in text:
                    text = text.split(q_marker, 1)[1]
                    break
            
            for a_marker in ['정답:', '답:', 'A:', 'Answer:']:
                if a_marker in text:
                    parts = text.split(a_marker, 1)
                    question = parts[0].strip()
                    answer = parts[1].strip() if len(parts) > 1 else ""
                    
                    # 최소 길이 체크
                    if len(question) > 10 and len(answer) > 0:
                        return {'question': question, 'answer': answer}
            
            # 구분자가 없으면 줄바꿈으로 구분
            lines = text.strip().split('\n')
            if len(lines) >= 2:
                return {'question': lines[0].strip(), 'answer': lines[1].strip()}
            
            return None
            
        except Exception as e:
            logger.error(f"파싱 오류: {e}")
            return None
    
    def _parse_qa_detailed(self, text: str) -> Optional[Dict]:
        """상세 QA 파싱 (해설 포함)"""
        try:
            result = {'question': '', 'answer': '', 'explanation': ''}
            
            # 섹션별 추출
            sections = text.split('\n\n')
            for section in sections:
                section_lower = section.lower()
                if '문제' in section_lower or 'question' in section_lower:
                    result['question'] = section.split(':', 1)[-1].strip()
                elif '정답' in section_lower or 'answer' in section_lower:
                    result['answer'] = section.split(':', 1)[-1].strip()
                elif '해설' in section_lower or 'explanation' in section_lower:
                    result['explanation'] = section.split(':', 1)[-1].strip()
            
            # 최소 조건 확인
            if result['question'] and result['answer']:
                return result
            
            # 실패시 단순 파싱 시도
            return self._parse_qa(text)
            
        except Exception as e:
            logger.error(f"상세 파싱 오류: {e}")
            return self._parse_qa(text)
    
    def get_stats(self) -> Dict:
        """통계 반환"""
        success_rate = (self.stats['success'] / self.stats['total'] * 100) if self.stats['total'] > 0 else 0
        avg_time = np.mean(self.stats['generation_times']) if self.stats['generation_times'] else 0
        
        return {
            'total': self.stats['total'],
            'success': self.stats['success'],
            'failed': self.stats['failed'],
            'success_rate': round(success_rate, 1),
            'avg_generation_time': round(avg_time, 2),
            'mode': 'chaining' if self.use_chaining else 'simple',
            'errors': dict(self.stats['errors'])
        }


class SimpleRAGSystem:
    """경량화된 RAG 시스템"""
    
    def __init__(self, external_dir: Path = EXTERNAL_DIR):
        self.external_dir = external_dir
        self.documents = []
        self.embeddings = None
        self.index = None
        self.embedding_model = None
        
        # 캐시 파일 경로
        self.cache_file = CACHE_DIR / "rag_cache.pkl"
    
    def initialize(self):
        """RAG 시스템 초기화"""
        logger.info("📚 RAG 시스템 초기화...")
        
        # 캐시 확인
        if self.cache_file.exists():
            try:
                logger.info("📦 캐시에서 RAG 인덱스 로드...")
                with open(self.cache_file, 'rb') as f:
                    cache_data = pickle.load(f)
                    self.documents = cache_data['documents']
                    self.embeddings = cache_data['embeddings']
                    
                # 임베딩 모델은 항상 로드
                self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
                
                # FAISS 인덱스 재생성
                self._rebuild_index()
                
                logger.info(f"✅ 캐시에서 로드 완료! 문서: {len(self.documents)}개")
                return
                
            except Exception as e:
                logger.warning(f"캐시 로드 실패: {e}. 새로 생성합니다.")
        
        # 새로 생성
        self._load_documents()
        
        if not self.documents:
            logger.warning("⚠️ 외부 문서가 없습니다. RAG 비활성화")
            return
        
        # 임베딩 모델 로드
        logger.info("🔍 임베딩 모델 로드...")
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        
        # 인덱스 생성
        self._create_index()
        
        # 캐시 저장
        self._save_cache()
        
        logger.info(f"✅ RAG 초기화 완료! 문서: {len(self.documents)}개")
    
    def _load_documents(self):
        """문서 로드"""
        self.documents = []
        pdf_files = list(self.external_dir.glob("*.pdf"))
        
        if not pdf_files:
            logger.warning(f"⚠️ {self.external_dir}에 PDF 파일이 없습니다.")
            return
        
        logger.info(f"📄 {len(pdf_files)}개 PDF 파일 로드 중...")
        
        for file_path in tqdm(pdf_files, desc="문서 로드"):
            try:
                with open(file_path, 'rb') as file:
                    reader = PyPDF2.PdfReader(file)
                    text = ""
                    
                    # 처음 10페이지만 읽기
                    max_pages = min(10, len(reader.pages))
                    for page_num in range(max_pages):
                        text += reader.pages[page_num].extract_text()
                    
                    # 청킹
                    chunks = self._simple_chunk(text, chunk_size=300)
                    for chunk in chunks:
                        self.documents.append({
                            'text': chunk,
                            'source': file_path.name,
                            'chunk_id': len(self.documents)
                        })
                        
            except Exception as e:
                logger.warning(f"문서 로드 실패 {file_path.name}: {e}")
    
    def _simple_chunk(self, text: str, chunk_size: int = 300) -> List[str]:
        """간단한 청킹"""
        # 문장 단위로 분리
        sentences = text.replace('\n', ' ').split('. ')
        chunks = []
        current_chunk = ""
        
        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue
                
            if len(current_chunk) + len(sentence) < chunk_size:
                current_chunk += sentence + ". "
            else:
                if current_chunk and len(current_chunk) > 50:
                    chunks.append(current_chunk.strip())
                current_chunk = sentence + ". "
        
        if current_chunk and len(current_chunk) > 50:
            chunks.append(current_chunk.strip())
        
        return chunks
    
    def _create_index(self):
        """FAISS 인덱스 생성"""
        if not self.documents:
            return
        
        logger.info("🔍 임베딩 생성 중...")
        texts = [doc['text'] for doc in self.documents]
        
        # 배치 처리로 임베딩 생성
        batch_size = 32
        self.embeddings = []
        
        for i in tqdm(range(0, len(texts), batch_size), desc="임베딩 생성"):
            batch = texts[i:i+batch_size]
            batch_embeddings = self.embedding_model.encode(batch)
            self.embeddings.extend(batch_embeddings)
        
        self.embeddings = np.array(self.embeddings)
        
        # FAISS 인덱스 생성
        self._rebuild_index()
        
        logger.info(f"✅ 인덱스 생성 완료: {len(texts)}개 청크")
    
    def _rebuild_index(self):
        """FAISS 인덱스 재구축"""
        if self.embeddings is None or len(self.embeddings) == 0:
            return
            
        dimension = self.embeddings.shape[1]
        self.index = faiss.IndexFlatIP(dimension)
        self.index.add(self.embeddings.astype('float32'))
    
    def _save_cache(self):
        """캐시 저장"""
        try:
            cache_data = {
                'documents': self.documents,
                'embeddings': self.embeddings
            }
            with open(self.cache_file, 'wb') as f:
                pickle.dump(cache_data, f)
            logger.info(f"💾 캐시 저장 완료: {self.cache_file}")
        except Exception as e:
            logger.warning(f"캐시 저장 실패: {e}")
    
    def search(self, query: str, top_k: int = 3) -> List[str]:
        """문서 검색"""
        if not self.index or not self.embedding_model:
            return []
        
        try:
            # 쿼리 임베딩
            query_embedding = self.embedding_model.encode([query])
            
            # 검색
            scores, indices = self.index.search(query_embedding.astype('float32'), top_k)
            
            # 결과 반환
            results = []
            for idx in indices[0]:
                if 0 <= idx < len(self.documents):
                    results.append(self.documents[idx]['text'])
            
            return results
            
        except Exception as e:
            logger.error(f"검색 오류: {e}")
            return []
    
    def get_random_context(self, n: int = 2) -> List[str]:
        """랜덤 컨텍스트 반환"""
        if not self.documents:
            return []
        
        selected = random.sample(self.documents, min(n, len(self.documents)))
        return [doc['text'] for doc in selected]


class FSKUAugmentationSystem:
    """
    FSKU 데이터 증강 통합 시스템
    - 모델 사전 로딩 ✅
    - 에러 처리 강화 ✅
    - 동작 검증 포함 ✅
    """
    
    def __init__(self,
                 model_name: str = "microsoft/phi-2",
                 use_chaining: bool = False,
                 use_quantization: bool = False,
                 use_rag: bool = True):
        """초기화"""
        self.model_name = model_name
        self.use_chaining = use_chaining
        self.use_quantization = use_quantization
        self.use_rag = use_rag
        
        # 컴포넌트
        self.generator = None
        self.rag_system = None
        
        # 설정
        self.config = {
            'question_types': ['객관식', '주관식', '단답형', '서술형'],
            'default_contexts': [
                "개인정보 처리자는 개인정보를 처리할 목적을 명확히 하여야 하며, 그 목적에 필요한 범위에서 최소한으로 개인정보를 처리하여야 한다.",
                "금융기관은 전자금융거래 시 충분한 보안대책을 수립·시행하여야 하며, 이용자로부터 이용자를 식별할 수 있는 정보를 요구할 수 있다.",
                "금융회사는 자금세탁방지 및 테러자금조달금지에 관한 법률에 따라 고객확인의무를 이행하여야 한다.",
                "신용정보회사는 신용정보주체의 동의를 받지 아니하고는 개인신용정보를 제3자에게 제공하거나 목적 외의 용도로 이용할 수 없다.",
                "금융회사는 내부통제기준을 마련하여 이사회의 승인을 받고 이를 성실히 이행하여야 한다."
            ]
        }
        
        # 초기화 상태
        self.initialized = False
    
    def initialize(self) -> bool:
        """시스템 초기화 (모델 사전 로딩)"""
        logger.info("=" * 60)
        logger.info("🚀 FSKU 데이터 증강 시스템 초기화")
        logger.info("=" * 60)
        
        try:
            # 1. 데이터 생성기 초기화
            logger.info(f"\n[1/2] 데이터 생성기 초기화...")
            logger.info(f"  - 모델: {self.model_name}")
            logger.info(f"  - 체이닝: {'✅ ON' if self.use_chaining else '❌ OFF'}")
            logger.info(f"  - 양자화: {'✅ ON' if self.use_quantization else '❌ OFF'}")
            
            self.generator = SimpleDataGenerator(
                model_name=self.model_name,
                use_chaining=self.use_chaining,
                use_quantization=self.use_quantization
            )
            
            if not self.generator.initialize():
                raise RuntimeError("데이터 생성기 초기화 실패")
            
            # 2. RAG 시스템 초기화
            if self.use_rag:
                logger.info("\n[2/2] RAG 시스템 초기화...")
                self.rag_system = SimpleRAGSystem()
                self.rag_system.initialize()
            else:
                logger.info("\n[2/2] RAG 시스템 비활성화")
            
            self.initialized = True
            logger.info("\n✅ 모든 시스템 초기화 완료!")
            return True
            
        except Exception as e:
            logger.error(f"❌ 시스템 초기화 실패: {e}")
            logger.error(traceback.format_exc())
            self.initialized = False
            return False
    
    def verify_system(self) -> bool:
        """시스템 동작 검증"""
        if not self.initialized:
            logger.error("시스템이 초기화되지 않았습니다.")
            return False
        
        logger.info("\n🔍 시스템 동작 검증 시작...")
        
        try:
            # 1. 간단한 QA 생성 테스트
            test_context = "테스트를 위한 금융 관련 내용입니다."
            test_result = self.generator.generate_qa_pair(test_context, "객관식")
            
            if not test_result:
                logger.error("QA 생성 테스트 실패")
                return False
            
            logger.info(f"✅ QA 생성 테스트 성공")
            logger.info(f"  - 문제: {test_result['question'][:50]}...")
            logger.info(f"  - 답변: {test_result['answer'][:30]}...")
            
            # 2. RAG 검색 테스트 (활성화된 경우)
            if self.use_rag and self.rag_system:
                test_results = self.rag_system.search("개인정보", top_k=1)
                if test_results:
                    logger.info(f"✅ RAG 검색 테스트 성공: {len(test_results)}개 결과")
                else:
                    logger.warning("⚠️ RAG 검색 결과 없음")
            
            logger.info("✅ 시스템 검증 완료!")
            return True
            
        except Exception as e:
            logger.error(f"❌ 시스템 검증 실패: {e}")
            return False
    
    def run(self, target_count: int = 10, output_file: str = None) -> List[Dict]:
        """데이터 생성 실행"""
        if not self.initialized:
            logger.error("시스템이 초기화되지 않았습니다. initialize()를 먼저 호출하세요.")
            return []
        
        logger.info(f"\n🎯 데이터 생성 시작")
        logger.info(f"  - 목표: {target_count}개")
        logger.info(f"  - 모드: {'🔗 체이닝' if self.use_chaining else '⚡ 단순'}")
        logger.info(f"  - RAG: {'✅ 사용' if self.use_rag else '❌ 미사용'}")
        
        # 컨텍스트 준비
        contexts = self._prepare_contexts(target_count)
        
        # 생성 실행
        start_time = time.time()
        results = []
        
        with tqdm(total=target_count, desc="데이터 생성") as pbar:
            for i, context in enumerate(contexts[:target_count]):
                qtype = self.config['question_types'][i % len(self.config['question_types'])]
                
                result = self.generator.generate_qa_pair(context, qtype)
                
                if result:
                    results.append(result)
                    pbar.update(1)
                    pbar.set_postfix({'성공': len(results), '실패': i + 1 - len(results)})
                else:
                    pbar.update(1)
                    pbar.set_postfix({'성공': len(results), '실패': i + 1 - len(results)})
        
        total_time = time.time() - start_time
        
        # 결과 출력
        self._print_results(results, target_count, total_time)
        
        # 결과 저장
        if results:
            saved_file = self._save_results(results, output_file)
            logger.info(f"💾 결과 저장: {saved_file}")
        
        return results
    
    def _prepare_contexts(self, count: int) -> List[str]:
        """컨텍스트 준비"""
        contexts = []
        
        if self.use_rag and self.rag_system and self.rag_system.documents:
            # RAG 검색
            topics = ["개인정보보호", "전자금융거래", "금융보안", "자금세탁방지", 
                     "신용정보", "내부통제", "정보보안", "사이버보안"]
            
            for topic in topics:
                search_results = self.rag_system.search(topic, top_k=3)
                contexts.extend(search_results)
            
            # 랜덤 컨텍스트 추가
            random_contexts = self.rag_system.get_random_context(n=count // 2)
            contexts.extend(random_contexts)
        
        # 기본 컨텍스트 추가
        contexts.extend(self.config['default_contexts'])
        
        # 필요한 만큼 반복
        while len(contexts) < count:
            contexts.extend(contexts[:count - len(contexts)])
        
        # 셔플
        random.shuffle(contexts)
        
        return contexts
    
    def _print_results(self, results: List[Dict], target_count: int, total_time: float):
        """결과 출력"""
        if results:
            success_rate = len(results) / target_count * 100
            avg_time = total_time / len(results)
            
            logger.info(f"\n🎉 생성 완료!")
            logger.info(f"📊 결과:")
            logger.info(f"  - 성공: {len(results)}/{target_count}개 ({success_rate:.1f}%)")
            logger.info(f"  - 소요 시간: {total_time:.1f}초")
            logger.info(f"  - 평균 시간: {avg_time:.1f}초/개")
            
            # 통계
            stats = self.generator.get_stats()
            logger.info(f"\n📈 통계:")
            logger.info(f"  - 성공률: {stats['success_rate']}%")
            logger.info(f"  - 평균 생성 시간: {stats['avg_generation_time']}초")
            logger.info(f"  - 모드: {stats['mode']}")
            
            if stats['errors']:
                logger.info(f"  - 오류: {stats['errors']}")
            
            # 샘플 출력
            logger.info(f"\n📋 생성 샘플:")
            for i, result in enumerate(results[:3]):
                logger.info(f"\n[샘플 {i+1}] ({result.get('method', 'unknown')})")
                logger.info(f"  문제: {result['question'][:100]}...")
                logger.info(f"  답변: {result['answer'][:50]}...")
                if 'explanation' in result and result['explanation']:
                    logger.info(f"  해설: {result['explanation'][:50]}...")
        else:
            logger.error("❌ 생성된 데이터가 없습니다.")
    
    def _save_results(self, results: List[Dict], output_file: str = None) -> Path:
        """결과 저장"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        if output_file:
            output_path = Path(output_file)
        else:
            filename = f"fsku_data_{timestamp}.json"
            output_path = OUTPUT_DIR / filename
        
        # 메타데이터 추가
        output_data = {
            'metadata': {
                'timestamp': timestamp,
                'model': self.model_name,
                'use_chaining': self.use_chaining,
                'use_quantization': self.use_quantization,
                'use_rag': self.use_rag,
                'total_count': len(results),
                'stats': self.generator.get_stats()
            },
            'data': results
        }
        
        # 저장
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, ensure_ascii=False, indent=2)
        
        return output_path


def main():
    """메인 실행 함수"""
    parser = argparse.ArgumentParser(description="FSKU 데이터 증강 시스템")
    
    # 기본 옵션
    parser.add_argument('--model', type=str, default="microsoft/phi-2",
                       help="사용할 모델 (default: microsoft/phi-2)")
    parser.add_argument('--count', type=int, default=10,
                       help="생성할 데이터 개수 (default: 10)")
    parser.add_argument('--output', type=str, default=None,
                       help="출력 파일 경로")
    
    # 모드 옵션
    parser.add_argument('--chaining', action='store_true',
                       help="체이닝 모드 사용 (고품질)")
    parser.add_argument('--quantization', action='store_true',
                       help="4bit 양자화 사용")
    parser.add_argument('--no-rag', action='store_true',
                       help="RAG 시스템 비활성화")
    
    # 테스트 옵션
    parser.add_argument('--test', action='store_true',
                       help="테스트 모드 (5개만 생성)")
    parser.add_argument('--verify', action='store_true',
                       help="시스템 검증만 수행")
    
    args = parser.parse_args()
    
    # 테스트 모드
    if args.test:
        args.count = 5
        logger.info("⚠️ 테스트 모드: 5개만 생성합니다.")
    
    # 시스템 생성
    logger.info("🚀 FSKU 데이터 증강 시스템 시작")
    logger.info(f"설정: 모델={args.model}, 체이닝={args.chaining}, 양자화={args.quantization}")
    
    system = FSKUAugmentationSystem(
        model_name=args.model,
        use_chaining=args.chaining,
        use_quantization=args.quantization,
        use_rag=not args.no_rag
    )
    
    # 초기화 (모델 사전 로딩)
    if not system.initialize():
        logger.error("시스템 초기화 실패")
        return 1
    
    # 시스템 검증
    if not system.verify_system():
        logger.error("시스템 검증 실패")
        return 1
    
    # 검증만 수행
    if args.verify:
        logger.info("✅ 시스템 검증 완료. 종료합니다.")
        return 0
    
    # 데이터 생성
    results = system.run(
        target_count=args.count,
        output_file=args.output
    )
    
    if results:
        logger.info(f"✅ 성공적으로 {len(results)}개 데이터 생성 완료!")
        return 0
    else:
        logger.error("❌ 데이터 생성 실패")
        return 1


if __name__ == "__main__":
    exit(main())