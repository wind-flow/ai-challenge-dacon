#!/usr/bin/env python3
"""
RAG 인덱스 재구축 유틸리티

문서가 업데이트되었을 때 실행하세요.
"""

import sys
from pathlib import Path

sys.path.append("src")

def rebuild_index():
    """인덱스 재구축"""
    print("="*60)
    print("🔄 RAG 인덱스 재구축")
    print("="*60)
    
    # 기존 인덱스 삭제
    index_path = Path("data/vectordb/index.pkl")
    if index_path.exists():
        print(f"🗑️ 기존 인덱스 삭제: {index_path}")
        index_path.unlink()
    
    # 새 인덱스 생성
    print("\n📚 새 인덱스 생성 중...")
    from rag.retriever import DocumentRetriever
    
    retriever = DocumentRetriever(use_embedding=False, use_cache=False)
    
    # 통계 출력
    stats = retriever.get_statistics()
    print(f"\n✅ 인덱스 재구축 완료!")
    print(f"  - 문서: {stats['total_documents']}개")
    print(f"  - 청크: {stats['total_chunks']}개")
    print(f"  - 키워드: {stats['total_keywords']:,}개")
    
    print(f"\n💾 저장 위치: {index_path}")
    print("📌 이제 main.py를 실행하면 캐시된 인덱스를 사용합니다.")

if __name__ == "__main__":
    rebuild_index()