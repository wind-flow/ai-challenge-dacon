#!/usr/bin/env python3
"""
PDF 및 Excel 문서 로더

금융 규정 문서를 텍스트로 변환
"""

import os
import re
from pathlib import Path
from typing import List, Dict, Optional
import logging

logger = logging.getLogger(__name__)


class DocumentLoader:
    """문서 로드 및 전처리"""
    
    def __init__(self):
        self.supported_formats = ['.pdf', '.txt', '.xlsx', '.json']
        
    def load_document(self, file_path: Path) -> Optional[Dict]:
        """
        문서 로드
        
        Args:
            file_path: 문서 경로
            
        Returns:
            문서 딕셔너리 (content, metadata)
        """
        if not file_path.exists():
            logger.error(f"파일이 존재하지 않습니다: {file_path}")
            return None
            
        suffix = file_path.suffix.lower()
        
        try:
            if suffix == '.pdf':
                return self._load_pdf(file_path)
            elif suffix == '.txt':
                return self._load_text(file_path)
            elif suffix == '.xlsx':
                return self._load_excel(file_path)
            elif suffix == '.json':
                return self._load_json(file_path)
            else:
                logger.warning(f"지원하지 않는 형식: {suffix}")
                return None
        except Exception as e:
            logger.error(f"문서 로드 실패 {file_path}: {e}")
            return None
    
    def _load_pdf(self, file_path: Path) -> Dict:
        """PDF 파일 로드"""
        try:
            import PyPDF2
        except ImportError:
            logger.warning("PyPDF2가 설치되지 않았습니다. pip install PyPDF2")
            # 대체 방법 시도
            try:
                import pdfplumber
                return self._load_pdf_with_pdfplumber(file_path)
            except ImportError:
                logger.error("PDF 라이브러리가 없습니다. pip install PyPDF2 또는 pdfplumber")
                return None
        
        text_pages = []
        metadata = {
            'source': file_path.name,
            'type': 'pdf',
            'pages': 0
        }
        
        try:
            with open(file_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                metadata['pages'] = len(pdf_reader.pages)
                
                for page_num, page in enumerate(pdf_reader.pages):
                    try:
                        text = page.extract_text()
                        if text.strip():
                            text_pages.append({
                                'page': page_num + 1,
                                'content': text
                            })
                    except Exception as e:
                        logger.debug(f"페이지 {page_num+1} 추출 실패: {e}")
                        continue
        except Exception as e:
            logger.error(f"PDF 읽기 실패 {file_path}: {e}")
            return None
        
        # 전체 텍스트 결합
        full_text = "\n\n".join([f"[페이지 {p['page']}]\n{p['content']}" 
                                 for p in text_pages])
        
        return {
            'content': full_text,
            'metadata': metadata,
            'pages': text_pages
        }
    
    def _load_pdf_with_pdfplumber(self, file_path: Path) -> Dict:
        """pdfplumber로 PDF 로드 (대체 방법)"""
        import pdfplumber
        
        text_pages = []
        metadata = {
            'source': file_path.name,
            'type': 'pdf',
            'pages': 0
        }
        
        try:
            with pdfplumber.open(file_path) as pdf:
                metadata['pages'] = len(pdf.pages)
                
                for page_num, page in enumerate(pdf.pages):
                    text = page.extract_text()
                    if text:
                        text_pages.append({
                            'page': page_num + 1,
                            'content': text
                        })
        except Exception as e:
            logger.error(f"pdfplumber 로드 실패 {file_path}: {e}")
            return None
        
        full_text = "\n\n".join([f"[페이지 {p['page']}]\n{p['content']}" 
                                 for p in text_pages])
        
        return {
            'content': full_text,
            'metadata': metadata,
            'pages': text_pages
        }
    
    def _load_text(self, file_path: Path) -> Dict:
        """텍스트 파일 로드"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
        except UnicodeDecodeError:
            # 다른 인코딩 시도
            with open(file_path, 'r', encoding='cp949') as f:
                content = f.read()
        
        return {
            'content': content,
            'metadata': {
                'source': file_path.name,
                'type': 'text'
            }
        }
    
    def _load_excel(self, file_path: Path) -> Dict:
        """Excel 파일 로드"""
        try:
            import pandas as pd
        except ImportError:
            logger.error("pandas가 설치되지 않았습니다. pip install pandas openpyxl")
            return None
        
        try:
            # 모든 시트 읽기
            dfs = pd.read_excel(file_path, sheet_name=None)
            
            text_parts = []
            for sheet_name, df in dfs.items():
                text_parts.append(f"[시트: {sheet_name}]")
                text_parts.append(df.to_string())
            
            content = "\n\n".join(text_parts)
            
            return {
                'content': content,
                'metadata': {
                    'source': file_path.name,
                    'type': 'excel',
                    'sheets': list(dfs.keys())
                }
            }
        except Exception as e:
            logger.error(f"Excel 로드 실패 {file_path}: {e}")
            return None
    
    def _load_json(self, file_path: Path) -> Dict:
        """JSON 파일 로드"""
        import json
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # JSON을 텍스트로 변환
            content = json.dumps(data, ensure_ascii=False, indent=2)
            
            return {
                'content': content,
                'metadata': {
                    'source': file_path.name,
                    'type': 'json'
                }
            }
        except Exception as e:
            logger.error(f"JSON 로드 실패 {file_path}: {e}")
            return None
    
    def load_directory(self, dir_path: Path) -> List[Dict]:
        """
        디렉토리의 모든 문서 로드
        
        Args:
            dir_path: 디렉토리 경로
            
        Returns:
            문서 리스트
        """
        documents = []
        
        if not dir_path.exists():
            logger.error(f"디렉토리가 존재하지 않습니다: {dir_path}")
            return documents
        
        # 지원하는 형식의 파일들 찾기
        for ext in self.supported_formats:
            for file_path in dir_path.glob(f"*{ext}"):
                print(f"📄 로딩: {file_path.name}")
                doc = self.load_document(file_path)
                if doc:
                    documents.append(doc)
                    print(f"   ✅ 성공")
                else:
                    print(f"   ❌ 실패")
        
        print(f"\n총 {len(documents)}개 문서 로드 완료")
        return documents


if __name__ == "__main__":
    # 테스트
    loader = DocumentLoader()
    
    # 외부 데이터 폴더 로드
    external_dir = Path("data/external")
    documents = loader.load_directory(external_dir)
    
    # 통계 출력
    for doc in documents:
        meta = doc['metadata']
        content_len = len(doc['content'])
        print(f"- {meta['source']}: {meta['type']} ({content_len:,}자)")