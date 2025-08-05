#!/usr/bin/env python3
"""
개선된 RAG 시스템 - 성능 최적화 및 메모리 관리 강화
즉시 적용 가능한 개선사항과 단기 과제 구현
"""

import os
import gc
import json
import pickle
import asyncio
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
from dataclasses import dataclass
from functools import lru_cache
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor

import torch
import numpy as np
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel
import faiss

# ============================================================
# 1. 로깅 시스템 개선 (즉시 적용)
# ============================================================

# 구조화된 로깅 설정
def setup_logging(log_file: str = 'fsku_processing.log'):
    """개선된 로깅 시스템 설정"""
    
    # 로그 포맷 정의
    log_format = '%(asctime)s - %(name)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s'
    
    # 로거 설정
    logging.basicConfig(
        level=logging.INFO,
        format=log_format,
        handlers=[
            logging.FileHandler(log_file, encoding='utf-8'),
            logging.StreamHandler()
        ]
    )
    
    # 성능 로거 별도 설정
    perf_logger = logging.getLogger('performance')
    perf_handler = logging.FileHandler('performance.log', encoding='utf-8')
    perf_handler.setFormatter(logging.Formatter(log_format))
    perf_logger.addHandler(perf_handler)
    
    return logging.getLogger(__name__)

logger = setup_logging()

# ============================================================
# 2. 메모리 관리 클래스 (즉시 적용)
# ============================================================

class MemoryManager:
    """메모리 사용량 모니터링 및 자동 정리"""
    
    def __init__(self, threshold_gb: float = 20.0):
        self.threshold_gb = threshold_gb
        self.cleanup_count = 0
        
    def get_memory_usage(self) -> Dict[str, float]:
        """현재 메모리 사용량 조회"""
        import psutil
        
        process = psutil.Process()
        memory_info = process.memory_info()
        
        return {
            'rss_gb': memory_info.rss / (1024**3),  # 실제 메모리 사용량
            'vms_gb': memory_info.vms / (1024**3),  # 가상 메모리 사용량
            'percent': process.memory_percent(),
            'available_gb': psutil.virtual_memory().available / (1024**3)
        }
    
    def cleanup(self, force: bool = False):
        """메모리 정리"""
        memory = self.get_memory_usage()
        
        if force or memory['rss_gb'] > self.threshold_gb:
            logger.info(f"🧹 메모리 정리 시작 (현재: {memory['rss_gb']:.2f}GB)")
            
            # Python 가비지 컬렉션
            gc.collect()
            
            # PyTorch CUDA 메모리 정리
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
            
            # MPS 메모리 정리 (Mac)
            if torch.backends.mps.is_available():
                torch.mps.empty_cache()
                torch.mps.synchronize()
            
            self.cleanup_count += 1
            
            # 정리 후 메모리 상태
            memory_after = self.get_memory_usage()
            freed = memory['rss_gb'] - memory_after['rss_gb']
            logger.info(f"✅ 메모리 정리 완료 (해제: {freed:.2f}GB, 현재: {memory_after['rss_gb']:.2f}GB)")
            
            return freed
        return 0
    
    def monitor_decorator(self, func):
        """함수 실행 전후 메모리 모니터링 데코레이터"""
        def wrapper(*args, **kwargs):
            # 실행 전 메모리
            before = self.get_memory_usage()
            logger.debug(f"메모리 사용 전: {before['rss_gb']:.2f}GB")
            
            # 함수 실행
            result = func(*args, **kwargs)
            
            # 실행 후 메모리
            after = self.get_memory_usage()
            used = after['rss_gb'] - before['rss_gb']
            
            if used > 1.0:  # 1GB 이상 증가시 경고
                logger.warning(f"⚠️ 메모리 급증: {used:.2f}GB (함수: {func.__name__})")
            
            # 자동 정리
            self.cleanup()
            
            return result
        return wrapper

# 전역 메모리 매니저
memory_manager = MemoryManager(threshold_gb=20.0)

# ============================================================
# 3. 배치 처리 최적화 (단기 과제)
# ============================================================

@dataclass
class BatchConfig:
    """배치 처리 설정"""
    batch_size: int = 32
    max_batch_size: int = 128
    min_batch_size: int = 4
    adaptive: bool = True
    memory_threshold_gb: float = 18.0

class AdaptiveBatchProcessor:
    """메모리 기반 적응형 배치 처리"""
    
    def __init__(self, config: BatchConfig = None):
        self.config = config or BatchConfig()
        self.current_batch_size = self.config.batch_size
        self.performance_history = []
        
    def get_optimal_batch_size(self) -> int:
        """현재 메모리 상태 기반 최적 배치 크기 계산"""
        if not self.config.adaptive:
            return self.current_batch_size
        
        memory = memory_manager.get_memory_usage()
        available_gb = memory['available_gb']
        
        # 메모리 기반 배치 크기 조정
        if available_gb < 4:
            self.current_batch_size = self.config.min_batch_size
        elif available_gb < 8:
            self.current_batch_size = min(16, self.config.batch_size)
        elif available_gb < 16:
            self.current_batch_size = self.config.batch_size
        else:
            self.current_batch_size = min(
                self.config.max_batch_size,
                self.config.batch_size * 2
            )
        
        logger.info(f"📊 배치 크기 조정: {self.current_batch_size} (가용 메모리: {available_gb:.2f}GB)")
        return self.current_batch_size
    
    def process_batch(self, items: List[Any], process_func, **kwargs) -> List[Any]:
        """배치 단위 처리 with 메모리 관리"""
        batch_size = self.get_optimal_batch_size()
        results = []
        
        # 진행률 표시
        total_batches = (len(items) + batch_size - 1) // batch_size
        
        with tqdm(total=len(items), desc="배치 처리") as pbar:
            for i in range(0, len(items), batch_size):
                batch = items[i:i + batch_size]
                
                # 배치 처리
                try:
                    batch_results = process_func(batch, **kwargs)
                    results.extend(batch_results)
                    
                    # 성능 기록
                    self.performance_history.append({
                        'batch_size': len(batch),
                        'memory_used': memory_manager.get_memory_usage()['rss_gb'],
                        'timestamp': datetime.now()
                    })
                    
                except Exception as e:
                    logger.error(f"배치 처리 실패: {e}")
                    # 배치 크기 줄이고 재시도
                    if batch_size > self.config.min_batch_size:
                        self.current_batch_size = max(
                            self.config.min_batch_size,
                            batch_size // 2
                        )
                        logger.info(f"배치 크기 감소: {self.current_batch_size}")
                    raise
                
                # 주기적 메모리 정리 (10 배치마다)
                if (i // batch_size) % 10 == 0:
                    memory_manager.cleanup()
                
                pbar.update(len(batch))
        
        return results

# ============================================================
# 4. 비동기 PDF 처리 (단기 과제)
# ============================================================

class AsyncPDFProcessor:
    """비동기 PDF 처리 시스템"""
    
    def __init__(self, max_workers: int = 4):
        self.max_workers = max_workers
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
        
    async def process_pdf_async(self, pdf_path: Path) -> Dict[str, Any]:
        """단일 PDF 비동기 처리"""
        loop = asyncio.get_event_loop()
        
        try:
            # CPU 집약적 작업은 스레드풀에서 실행
            result = await loop.run_in_executor(
                self.executor,
                self._process_pdf_sync,
                pdf_path
            )
            return result
            
        except Exception as e:
            logger.error(f"PDF 처리 실패 ({pdf_path}): {e}")
            return {'error': str(e), 'path': str(pdf_path)}
    
    def _process_pdf_sync(self, pdf_path: Path) -> Dict[str, Any]:
        """동기 PDF 처리 (실제 로직)"""
        import PyPDF2
        
        logger.info(f"📄 PDF 처리 중: {pdf_path.name}")
        
        try:
            with open(pdf_path, 'rb') as file:
                reader = PyPDF2.PdfReader(file)
                
                # 텍스트 추출
                text_content = []
                for page_num in range(len(reader.pages)):
                    page = reader.pages[page_num]
                    text_content.append(page.extract_text())
                
                # 메타데이터 추출
                metadata = {
                    'title': reader.metadata.get('/Title', 'Unknown'),
                    'pages': len(reader.pages),
                    'file_size': pdf_path.stat().st_size,
                    'processed_at': datetime.now().isoformat()
                }
                
                return {
                    'path': str(pdf_path),
                    'text': '\n'.join(text_content),
                    'metadata': metadata
                }
                
        except Exception as e:
            logger.error(f"PDF 읽기 실패: {e}")
            raise
    
    async def process_pdfs_batch(self, pdf_paths: List[Path]) -> List[Dict]:
        """여러 PDF 동시 처리"""
        logger.info(f"🚀 {len(pdf_paths)}개 PDF 비동기 처리 시작")
        
        # 비동기 태스크 생성
        tasks = [self.process_pdf_async(path) for path in pdf_paths]
        
        # 동시 실행 및 결과 수집
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # 성공/실패 분리
        successful = [r for r in results if not isinstance(r, Exception) and 'error' not in r]
        failed = [r for r in results if isinstance(r, Exception) or 'error' in r]
        
        logger.info(f"✅ PDF 처리 완료: 성공 {len(successful)}개, 실패 {len(failed)}개")
        
        # 메모리 정리
        memory_manager.cleanup()
        
        return results

# ============================================================
# 5. 개선된 RAG 시스템 (통합)
# ============================================================

class ImprovedRAGSystem:
    """성능 최적화된 RAG 시스템"""
    
    def __init__(self, 
                 model_name: str = "jhgan/ko-sroberta-multitask",
                 cache_dir: str = "data/cache",
                 use_async: bool = True):
        
        self.model_name = model_name
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.use_async = use_async
        
        # 컴포넌트 초기화
        self.memory_manager = memory_manager
        self.batch_processor = AdaptiveBatchProcessor()
        self.pdf_processor = AsyncPDFProcessor()
        
        # 모델 및 토크나이저
        self.tokenizer = None
        self.model = None
        self.device = self._get_device()
        
        # 벡터 저장소
        self.index = None
        self.documents = []
        
        logger.info(f"✅ ImprovedRAGSystem 초기화 완료 (Device: {self.device})")
    
    def _get_device(self) -> str:
        """최적 디바이스 선택"""
        if torch.cuda.is_available():
            return "cuda"
        elif torch.backends.mps.is_available():
            return "mps"
        else:
            return "cpu"
    
    @memory_manager.monitor_decorator
    def load_model(self):
        """모델 로드 with 메모리 모니터링"""
        logger.info(f"🔄 모델 로딩: {self.model_name}")
        
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModel.from_pretrained(self.model_name)
            
            # 디바이스 이동
            if self.device != "cpu":
                self.model = self.model.to(self.device)
            
            # 평가 모드
            self.model.eval()
            
            logger.info("✅ 모델 로드 완료")
            
        except Exception as e:
            logger.error(f"모델 로드 실패: {e}")
            raise
    
    @lru_cache(maxsize=1000)
    def get_embedding(self, text: str) -> np.ndarray:
        """텍스트 임베딩 생성 (캐싱)"""
        with torch.no_grad():
            inputs = self.tokenizer(
                text,
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors="pt"
            )
            
            if self.device != "cpu":
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            outputs = self.model(**inputs)
            embeddings = outputs.last_hidden_state.mean(dim=1)
            
            return embeddings.cpu().numpy()[0]
    
    def process_documents_batch(self, texts: List[str]) -> np.ndarray:
        """배치 단위 문서 처리"""
        
        def batch_embed(batch_texts):
            embeddings = []
            for text in batch_texts:
                emb = self.get_embedding(text)
                embeddings.append(emb)
            return embeddings
        
        # 적응형 배치 처리
        all_embeddings = self.batch_processor.process_batch(
            texts,
            batch_embed
        )
        
        return np.array(all_embeddings)
    
    async def build_index_async(self, pdf_dir: Path):
        """비동기 인덱스 구축"""
        logger.info("🔨 비동기 인덱스 구축 시작")
        
        # PDF 파일 목록
        pdf_files = list(pdf_dir.glob("*.pdf"))
        
        if not pdf_files:
            logger.warning("PDF 파일이 없습니다")
            return
        
        # 비동기 PDF 처리
        pdf_results = await self.pdf_processor.process_pdfs_batch(pdf_files)
        
        # 텍스트 추출 및 청킹
        all_chunks = []
        for result in pdf_results:
            if 'text' in result:
                chunks = self._chunk_text(result['text'])
                all_chunks.extend(chunks)
        
        logger.info(f"총 {len(all_chunks)}개 청크 생성")
        
        # 임베딩 생성 (배치 처리)
        embeddings = self.process_documents_batch(all_chunks)
        
        # FAISS 인덱스 생성
        dimension = embeddings.shape[1]
        self.index = faiss.IndexFlatL2(dimension)
        self.index.add(embeddings.astype('float32'))
        
        # 문서 저장
        self.documents = all_chunks
        
        # 캐시 저장
        self._save_cache()
        
        logger.info(f"✅ 인덱스 구축 완료: {len(self.documents)}개 문서")
    
    def _chunk_text(self, text: str, chunk_size: int = 500, overlap: int = 50) -> List[str]:
        """텍스트 청킹"""
        chunks = []
        for i in range(0, len(text), chunk_size - overlap):
            chunk = text[i:i + chunk_size]
            if chunk:
                chunks.append(chunk)
        return chunks
    
    def _save_cache(self):
        """인덱스 캐시 저장"""
        cache_path = self.cache_dir / "index_cache.pkl"
        
        cache_data = {
            'index': faiss.serialize_index(self.index),
            'documents': self.documents,
            'metadata': {
                'created_at': datetime.now().isoformat(),
                'num_documents': len(self.documents),
                'model': self.model_name
            }
        }
        
        with open(cache_path, 'wb') as f:
            pickle.dump(cache_data, f)
        
        logger.info(f"💾 캐시 저장: {cache_path}")
    
    def search(self, query: str, k: int = 5) -> List[Dict]:
        """유사 문서 검색"""
        if self.index is None:
            logger.error("인덱스가 구축되지 않았습니다")
            return []
        
        # 쿼리 임베딩
        query_embedding = self.get_embedding(query)
        query_embedding = np.array([query_embedding]).astype('float32')
        
        # 검색
        distances, indices = self.index.search(query_embedding, k)
        
        # 결과 정리
        results = []
        for i, (dist, idx) in enumerate(zip(distances[0], indices[0])):
            if idx < len(self.documents):
                results.append({
                    'rank': i + 1,
                    'text': self.documents[idx],
                    'distance': float(dist),
                    'similarity': 1 / (1 + float(dist))
                })
        
        return results
    
    def cleanup(self):
        """리소스 정리"""
        logger.info("🧹 리소스 정리 중...")
        
        # 모델 정리
        if self.model is not None:
            del self.model
            self.model = None
        
        if self.tokenizer is not None:
            del self.tokenizer
            self.tokenizer = None
        
        # 인덱스 정리
        if self.index is not None:
            del self.index
            self.index = None
        
        # 메모리 강제 정리
        memory_manager.cleanup(force=True)
        
        logger.info("✅ 리소스 정리 완료")

# ============================================================
# 6. 성능 모니터링 유틸리티
# ============================================================

class PerformanceMonitor:
    """성능 모니터링 및 보고"""
    
    def __init__(self):
        self.metrics = []
        
    def record(self, operation: str, duration: float, memory_used: float):
        """성능 메트릭 기록"""
        self.metrics.append({
            'operation': operation,
            'duration': duration,
            'memory_gb': memory_used,
            'timestamp': datetime.now()
        })
    
    def report(self) -> Dict:
        """성능 보고서 생성"""
        if not self.metrics:
            return {}
        
        report = {
            'total_operations': len(self.metrics),
            'total_duration': sum(m['duration'] for m in self.metrics),
            'avg_duration': np.mean([m['duration'] for m in self.metrics]),
            'max_memory': max(m['memory_gb'] for m in self.metrics),
            'avg_memory': np.mean([m['memory_gb'] for m in self.metrics])
        }
        
        logger.info("📊 성능 보고서:")
        for key, value in report.items():
            logger.info(f"  {key}: {value:.2f}")
        
        return report

# ============================================================
# 7. 사용 예시
# ============================================================

async def main_example():
    """사용 예시"""
    
    # RAG 시스템 초기화
    rag = ImprovedRAGSystem()
    
    # 모델 로드
    rag.load_model()
    
    # PDF 디렉토리 설정
    pdf_dir = Path("data/external")
    
    # 비동기 인덱스 구축
    if rag.use_async:
        await rag.build_index_async(pdf_dir)
    else:
        # 동기 방식 폴백
        logger.info("동기 방식으로 전환")
    
    # 검색 테스트
    query = "금융 상품 가입 조건은?"
    results = rag.search(query, k=3)
    
    logger.info(f"검색 결과 ({len(results)}개):")
    for result in results:
        logger.info(f"  [{result['rank']}] 유사도: {result['similarity']:.4f}")
        logger.info(f"      {result['text'][:100]}...")
    
    # 정리
    rag.cleanup()
    
    # 성능 보고
    monitor = PerformanceMonitor()
    monitor.report()

if __name__ == "__main__":
    # 비동기 실행
    asyncio.run(main_example())