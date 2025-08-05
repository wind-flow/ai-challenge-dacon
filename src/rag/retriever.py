#!/usr/bin/env python3
"""
향상된 RAG (Retrieval Augmented Generation) 문서 검색 모듈

PDF 문서 처리, 청킹, 하이브리드 검색 지원
"""

import json
import re
import pickle
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Any
import logging
from collections import defaultdict, Counter
import numpy as np

logger = logging.getLogger(__name__)


class DocumentRetriever:
    """향상된 문서 검색 및 관리 클래스"""
    
    def __init__(self, data_dir: str = "data", use_embedding: bool = True, use_cache: bool = True):
        """
        초기화
        
        Args:
            data_dir: 데이터 디렉토리
            use_embedding: 벡터 임베딩 사용 여부 (기본: True)
            use_cache: 캐시된 인덱스 사용 여부
        """
        self.data_dir = Path(data_dir)
        self.documents = []
        self.chunks = []  # 청킹된 문서
        self.chunk_index = defaultdict(list)  # 키워드 -> 청크 인덱스
        self.use_embedding = use_embedding
        self.embeddings = None
        self.embedding_model = None
        self.faiss_index = None
        self.index_path = self.data_dir / "vectordb" / "index.pkl"
        
        # PDF 로더와 청커 초기화
        try:
            from pdf_loader import DocumentLoader
            from chunker import DocumentChunker
        except ImportError:
            from .pdf_loader import DocumentLoader
            from .chunker import DocumentChunker
        
        self.loader = DocumentLoader()
        # Colab 설정과 동일하게 최적화
        self.chunker = DocumentChunker(chunk_size=300, chunk_overlap=50)
        
        # 캐시 확인 및 로드
        if use_cache and self.index_path.exists():
            print("📂 캐시된 인덱스 발견...")
            if self.load_index():
                print("✅ 캐시에서 인덱스 로드 완료!")
                # 임베딩 초기화 (선택적)
                if use_embedding and self.embeddings is None:
                    self._initialize_embeddings()
                return
        
        # 캐시가 없거나 로드 실패시 새로 생성
        print("🔄 인덱스를 새로 생성합니다...")
        self._load_and_process_documents()
        self._build_index()
        
        # 인덱스 저장
        self.save_index()
        
        # 임베딩 초기화 (선택적)
        if use_embedding:
            self._initialize_embeddings()
    
    def _load_and_process_documents(self):
        """문서 로드 및 청킹 처리"""
        external_dir = self.data_dir / "external"
        
        if not external_dir.exists():
            logger.warning(f"외부 데이터 디렉토리 없음: {external_dir}")
            return
        
        print("📚 문서 로드 및 처리 시작...")
        
        # 모든 문서 로드
        self.documents = self.loader.load_directory(external_dir)
        
        # 각 문서를 청킹
        for doc in self.documents:
            doc_chunks = self.chunker.chunk_document(doc)
            
            # 소스 정보 추가
            for chunk in doc_chunks:
                chunk['source'] = doc['metadata']['source']
                chunk['doc_type'] = doc['metadata']['type']
            
            self.chunks.extend(doc_chunks)
        
        print(f"✅ {len(self.documents)}개 문서에서 {len(self.chunks)}개 청크 생성")
    
    def _build_index(self):
        """청크 인덱스 구축 (BM25용)"""
        print("🔨 인덱스 구축 중...")
        
        for idx, chunk in enumerate(self.chunks):
            content = chunk['content'].lower()
            
            # 단어 추출
            words = re.findall(r'[가-힣]+|[a-zA-Z]+|\d+', content)
            
            # 각 단어에 대해 청크 인덱스 저장
            for word in set(words):
                if len(word) >= 2:
                    self.chunk_index[word].append(idx)
            
            # 키워드가 있으면 추가
            if 'keywords' in chunk:
                for keyword in chunk['keywords']:
                    self.chunk_index[keyword.lower()].append(idx)
        
        print(f"✅ 인덱스 구축 완료: {len(self.chunk_index)}개 키워드")
    
    def _initialize_embeddings(self):
        """벡터 임베딩 초기화 - 한국어 특화 모델 우선"""
        try:
            from sentence_transformers import SentenceTransformer
            
            # Colab과 동일한 한국어 모델 사용
            model_name = "jhgan/ko-sbert-nli"
            print(f"🤖 한국어 임베딩 모델 로드 중: {model_name}")
            
            self.embedding_model = SentenceTransformer(model_name)
            print("✅ 한국어 임베딩 모델 로드 완료")
            
            # 모든 청크 임베딩 (정규화 포함)
            texts = [chunk['content'] for chunk in self.chunks]
            self.embeddings = self.embedding_model.encode(
                texts, 
                normalize_embeddings=True,  # Colab과 동일
                show_progress_bar=True
            )
            
            print(f"✅ {len(self.embeddings)}개 청크 임베딩 완료")
            
            # FAISS 인덱스 구축 (선택적)
            self._build_faiss_index()
            
        except ImportError:
            logger.warning("sentence-transformers가 설치되지 않았습니다.")
            logger.warning("pip install sentence-transformers")
            self.use_embedding = False
    
    def _build_faiss_index(self):
        """FAISS 인덱스 구축"""
        try:
            import faiss
            
            # 임베딩 차원
            d = self.embeddings.shape[1]
            
            # FAISS 인덱스 생성
            self.faiss_index = faiss.IndexFlatIP(d)  # 내적 유사도
            
            # 정규화
            faiss.normalize_L2(self.embeddings)
            
            # 인덱스에 추가
            self.faiss_index.add(self.embeddings)
            
            print(f"✅ FAISS 인덱스 구축 완료")
            
        except ImportError:
            logger.warning("faiss가 설치되지 않았습니다.")
            logger.warning("pip install faiss-cpu")
    
    def search(self, query: str, top_k: int = 3, method: str = "similarity") -> str:
        """
        문서 검색
        
        Args:
            query: 검색 쿼리
            top_k: 반환할 청크 수
            method: 검색 방법 (bm25, embedding, hybrid)
            
        Returns:
            검색된 컨텍스트
        """
        if not self.chunks:
            return ""
        
        if method == "similarity" and self.use_embedding:
            # 기본: 임베딩 유사도 검색 (Colab 방식)
            results = self._embedding_search(query, top_k * 2)
        elif method == "hybrid" and self.use_embedding:
            results = self._hybrid_search(query, top_k * 2)
        elif method == "bm25":
            results = self._bm25_search(query, top_k * 2)
        else:
            # 폴백: BM25
            results = self._bm25_search(query, top_k * 2)
        
        # 상위 k개 선택 및 중복 제거
        seen_content = set()
        final_results = []
        
        for chunk_idx, score in results[:top_k*2]:
            chunk = self.chunks[chunk_idx]
            content_hash = hash(chunk['content'][:100])  # 앞부분으로 중복 체크
            
            if content_hash not in seen_content:
                seen_content.add(content_hash)
                final_results.append((chunk, score))
                
                if len(final_results) >= top_k:
                    break
        
        # 컨텍스트 생성
        contexts = []
        for chunk, score in final_results:
            source = chunk.get('source', 'Unknown')
            content = chunk['content']
            
            # 소스 정보 포함
            context = f"[출처: {source}]\n{content}"
            contexts.append(context)
        
        return "\n\n---\n\n".join(contexts)
    
    def _bm25_search(self, query: str, top_k: int) -> List[Tuple[int, float]]:
        """BM25 기반 검색"""
        query_lower = query.lower()
        query_words = re.findall(r'[가-힣]+|[a-zA-Z]+|\d+', query_lower)
        
        # BM25 파라미터
        k1 = 1.2
        b = 0.75
        
        # 문서 길이 평균
        avg_len = np.mean([len(chunk['content']) for chunk in self.chunks])
        
        # 각 청크에 대한 점수 계산
        scores = defaultdict(float)
        
        for word in query_words:
            if word in self.chunk_index:
                # IDF 계산
                df = len(self.chunk_index[word])
                idf = np.log((len(self.chunks) - df + 0.5) / (df + 0.5) + 1)
                
                # 각 문서에 대한 TF 계산
                for chunk_idx in self.chunk_index[word]:
                    chunk = self.chunks[chunk_idx]
                    tf = chunk['content'].lower().count(word)
                    doc_len = len(chunk['content'])
                    
                    # BM25 점수
                    score = idf * (tf * (k1 + 1)) / (tf + k1 * (1 - b + b * doc_len / avg_len))
                    scores[chunk_idx] += score
        
        # 정렬
        sorted_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return sorted_scores[:top_k]
    
    def _embedding_search(self, query: str, top_k: int) -> List[Tuple[int, float]]:
        """임베딩 기반 의미 검색"""
        if not self.use_embedding or self.embedding_model is None:
            return []
        
        # 쿼리 임베딩 (정규화 포함)
        query_embedding = self.embedding_model.encode(
            [query],
            normalize_embeddings=True
        )
        
        if hasattr(self, 'faiss_index'):
            # FAISS 사용
            import faiss
            faiss.normalize_L2(query_embedding)
            scores, indices = self.faiss_index.search(query_embedding, top_k)
            results = [(int(idx), float(score)) for idx, score in zip(indices[0], scores[0])]
        else:
            # 코사인 유사도 계산
            from sklearn.metrics.pairwise import cosine_similarity
            similarities = cosine_similarity(query_embedding, self.embeddings)[0]
            top_indices = np.argsort(similarities)[::-1][:top_k]
            results = [(int(idx), float(similarities[idx])) for idx in top_indices]
        
        return results
    
    def _hybrid_search(self, query: str, top_k: int) -> List[Tuple[int, float]]:
        """하이브리드 검색 (BM25 + 임베딩)"""
        # BM25 검색
        bm25_results = self._bm25_search(query, top_k)
        bm25_scores = {idx: score for idx, score in bm25_results}
        
        # 임베딩 검색
        embedding_results = self._embedding_search(query, top_k)
        embedding_scores = {idx: score for idx, score in embedding_results}
        
        # 점수 정규화 및 결합
        all_indices = set(bm25_scores.keys()) | set(embedding_scores.keys())
        
        hybrid_scores = {}
        for idx in all_indices:
            # 가중 평균 (BM25: 0.2, 임베딩: 0.8) - 임베딩 비중 증가
            bm25_score = bm25_scores.get(idx, 0)
            embedding_score = embedding_scores.get(idx, 0)
            
            # 정규화
            max_bm25 = max(bm25_scores.values()) if bm25_scores else 1
            max_embedding = max(embedding_scores.values()) if embedding_scores else 1
            
            normalized_bm25 = bm25_score / max_bm25 if max_bm25 > 0 else 0
            normalized_embedding = embedding_score / max_embedding if max_embedding > 0 else 0
            
            # 가중 결합
            hybrid_scores[idx] = 0.2 * normalized_bm25 + 0.8 * normalized_embedding
        
        # 정렬
        sorted_scores = sorted(hybrid_scores.items(), key=lambda x: x[1], reverse=True)
        return sorted_scores[:top_k]
    
    def get_random_chunks(self, n: int = 3) -> str:
        """
        랜덤 청크 가져오기 (개념 추출 대체용)
        
        Args:
            n: 가져올 청크 수
            
        Returns:
            랜덤 청크들의 텍스트
        """
        if not self.chunks:
            return ""
        
        import random
        selected_chunks = random.sample(self.chunks, min(n, len(self.chunks)))
        
        contexts = []
        for chunk in selected_chunks:
            source = chunk.get('source', 'Unknown')
            content = chunk['content']
            context = f"[출처: {source}]\n{content}"
            contexts.append(context)
        
        return "\n\n---\n\n".join(contexts)
    
    def get_statistics(self) -> Dict:
        """통계 정보 반환"""
        stats = {
            'total_documents': len(self.documents),
            'total_chunks': len(self.chunks),
            'total_keywords': len(self.chunk_index),
            'sources': list(set(chunk.get('source', 'Unknown') for chunk in self.chunks)),
            'avg_chunk_size': np.mean([chunk.get('tokens', 0) for chunk in self.chunks]) if self.chunks else 0,
            'use_embedding': self.use_embedding
        }
        
        # 문서 타입별 통계
        doc_types = Counter(chunk.get('doc_type', 'Unknown') for chunk in self.chunks)
        stats['document_types'] = dict(doc_types)
        
        return stats
    
    def save_index(self, path: str = "data/vectordb/index.pkl"):
        """인덱스 저장"""
        save_path = Path(path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        
        save_data = {
            'chunks': self.chunks,
            'chunk_index': dict(self.chunk_index),
            'embeddings': self.embeddings
        }
        
        with open(save_path, 'wb') as f:
            pickle.dump(save_data, f)
        
        print(f"✅ 인덱스 저장 완료: {save_path}")
    
    def load_index(self, path: str = "data/vectordb/index.pkl"):
        """인덱스 로드"""
        load_path = Path(path)
        
        if not load_path.exists():
            logger.warning(f"인덱스 파일이 없습니다: {load_path}")
            return False
        
        with open(load_path, 'rb') as f:
            save_data = pickle.load(f)
        
        self.chunks = save_data['chunks']
        self.chunk_index = defaultdict(list, save_data['chunk_index'])
        self.embeddings = save_data['embeddings']
        
        print(f"✅ 인덱스 로드 완료: {len(self.chunks)}개 청크")
        return True


if __name__ == "__main__":
    # 테스트
    print("="*60)
    print("RAG 시스템 테스트")
    print("="*60)
    
    # RAG 초기화 (임베딩 없이)
    retriever = DocumentRetriever(use_embedding=False)
    
    # 통계 출력
    stats = retriever.get_statistics()
    print(f"\n📊 통계:")
    print(f"- 문서 수: {stats['total_documents']}")
    print(f"- 청크 수: {stats['total_chunks']}")
    print(f"- 키워드 수: {stats['total_keywords']}")
    print(f"- 문서 타입: {stats['document_types']}")
    
    # 테스트 검색
    test_queries = [
        "전자금융거래",
        "개인정보보호",
        "암호기술"
    ]
    
    for query in test_queries:
        print(f"\n🔍 검색: '{query}'")
        results = retriever.search(query, top_k=2)
        
        if results:
            print(f"결과:\n{results[:200]}...")
        else:
            print("결과 없음")
    
    # 인덱스 저장
    retriever.save_index()