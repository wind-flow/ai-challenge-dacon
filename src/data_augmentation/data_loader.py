"""
데이터 로더 모듈
외부 금융 데이터를 로드하고 전처리
"""

import os
import json
import logging
from pathlib import Path
from typing import List, Dict, Optional, Union
from datetime import datetime

import pandas as pd
import pdfplumber
from tqdm import tqdm

logger = logging.getLogger(__name__)


class FinancialDataLoader:
    """외부 금융 데이터를 로드하고 전처리하는 클래스"""
    
    def __init__(self, data_dir: str):
        """
        Args:
            data_dir: 외부 데이터 디렉토리 경로
        """
        self.data_dir = Path(data_dir)
        self.documents = []
        self.metadata = {}
        
        # 지원하는 파일 형식
        self.supported_formats = {
            'pdf': ['.pdf'],
            'text': ['.txt', '.md'],
            'data': ['.json', '.jsonl', '.csv', '.xlsx']
        }
        
        logger.info(f"DataLoader initialized with directory: {self.data_dir}")
    
    def load_pdf(self, file_path: Union[str, Path]) -> str:
        """
        PDF 파일에서 텍스트 추출
        
        Args:
            file_path: PDF 파일 경로
            
        Returns:
            추출된 텍스트
        """
        file_path = Path(file_path)
        text_content = []
        
        try:
            with pdfplumber.open(file_path) as pdf:
                for page_num, page in enumerate(pdf.pages, 1):
                    page_text = page.extract_text()
                    if page_text:
                        text_content.append(page_text)
                        
            logger.info(f"Successfully extracted text from {file_path.name} ({len(text_content)} pages)")
            return "\n\n".join(text_content)
            
        except Exception as e:
            logger.error(f"Error loading PDF {file_path}: {e}")
            return ""
    
    def load_text_file(self, file_path: Union[str, Path]) -> str:
        """
        텍스트 파일 로드
        
        Args:
            file_path: 텍스트 파일 경로
            
        Returns:
            파일 내용
        """
        file_path = Path(file_path)
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            logger.info(f"Successfully loaded text file: {file_path.name}")
            return content
            
        except Exception as e:
            logger.error(f"Error loading text file {file_path}: {e}")
            return ""
    
    def load_json_file(self, file_path: Union[str, Path]) -> Union[Dict, List]:
        """
        JSON/JSONL 파일 로드
        
        Args:
            file_path: JSON 파일 경로
            
        Returns:
            파싱된 JSON 데이터
        """
        file_path = Path(file_path)
        
        try:
            if file_path.suffix == '.jsonl':
                data = []
                with open(file_path, 'r', encoding='utf-8') as f:
                    for line in f:
                        data.append(json.loads(line.strip()))
                return data
            else:
                with open(file_path, 'r', encoding='utf-8') as f:
                    return json.load(f)
                    
        except Exception as e:
            logger.error(f"Error loading JSON file {file_path}: {e}")
            return {}
    
    def load_csv_file(self, file_path: Union[str, Path]) -> pd.DataFrame:
        """
        CSV 파일 로드
        
        Args:
            file_path: CSV 파일 경로
            
        Returns:
            DataFrame
        """
        try:
            df = pd.read_csv(file_path, encoding='utf-8')
            logger.info(f"Successfully loaded CSV: {file_path.name} ({len(df)} rows)")
            return df
            
        except Exception as e:
            logger.error(f"Error loading CSV file {file_path}: {e}")
            return pd.DataFrame()
    
    def parse_financial_documents(self) -> List[Dict]:
        """
        금융 문서 파싱 및 구조화
        
        Returns:
            구조화된 문서 리스트
        """
        documents = []
        
        # 모든 파일 탐색
        for file_path in tqdm(list(self.data_dir.rglob("*")), desc="Loading documents"):
            if not file_path.is_file():
                continue
                
            document = {
                "source": file_path.name,
                "path": str(file_path),
                "content": "",
                "metadata": {
                    "file_type": file_path.suffix,
                    "size_bytes": file_path.stat().st_size,
                    "modified_date": datetime.fromtimestamp(file_path.stat().st_mtime).isoformat(),
                    "license": self._get_license_info(file_path)
                }
            }
            
            # 파일 형식별 처리
            if file_path.suffix in self.supported_formats['pdf']:
                document["content"] = self.load_pdf(file_path)
                document["metadata"]["content_type"] = "pdf_text"
                
            elif file_path.suffix in self.supported_formats['text']:
                document["content"] = self.load_text_file(file_path)
                document["metadata"]["content_type"] = "plain_text"
                
            elif file_path.suffix == '.json' or file_path.suffix == '.jsonl':
                data = self.load_json_file(file_path)
                document["content"] = json.dumps(data, ensure_ascii=False)
                document["metadata"]["content_type"] = "structured_data"
                document["metadata"]["data_count"] = len(data) if isinstance(data, list) else 1
                
            elif file_path.suffix == '.csv':
                df = self.load_csv_file(file_path)
                document["content"] = df.to_json(orient='records', force_ascii=False)
                document["metadata"]["content_type"] = "tabular_data"
                document["metadata"]["row_count"] = len(df)
                document["metadata"]["columns"] = list(df.columns)
            
            if document["content"]:
                documents.append(document)
        
        self.documents = documents
        logger.info(f"Loaded {len(documents)} documents from {self.data_dir}")
        
        return documents
    
    def _get_license_info(self, file_path: Path) -> Dict:
        """
        파일의 라이선스 정보 추출 (메타데이터 파일이나 파일명에서)
        
        Args:
            file_path: 파일 경로
            
        Returns:
            라이선스 정보 딕셔너리
        """
        license_info = {
            "type": "unknown",
            "commercial_use": False,
            "date": None
        }
        
        # 파일명에서 라이선스 정보 추출 시도
        filename_lower = file_path.name.lower()
        
        # 공공누리 라이선스 체크
        if "공공누리" in filename_lower or "kogl" in filename_lower:
            if "1유형" in filename_lower or "type1" in filename_lower:
                license_info["type"] = "KOGL Type 1"
                license_info["commercial_use"] = True
            elif "2유형" in filename_lower or "type2" in filename_lower:
                license_info["type"] = "KOGL Type 2"
                license_info["commercial_use"] = False
            elif "3유형" in filename_lower or "type3" in filename_lower:
                license_info["type"] = "KOGL Type 3"
                license_info["commercial_use"] = True
            elif "4유형" in filename_lower or "type4" in filename_lower:
                license_info["type"] = "KOGL Type 4"
                license_info["commercial_use"] = False
        
        # Creative Commons 라이선스 체크
        elif "cc-by" in filename_lower:
            if "nc" in filename_lower:
                license_info["type"] = "CC BY-NC"
                license_info["commercial_use"] = False
            elif "sa" in filename_lower:
                license_info["type"] = "CC BY-SA"
                license_info["commercial_use"] = True
            else:
                license_info["type"] = "CC BY"
                license_info["commercial_use"] = True
        
        # 메타데이터 파일 확인
        metadata_file = file_path.parent / f"{file_path.stem}_metadata.json"
        if metadata_file.exists():
            try:
                with open(metadata_file, 'r', encoding='utf-8') as f:
                    metadata = json.load(f)
                    license_info.update(metadata.get("license", {}))
            except:
                pass
        
        return license_info
    
    def filter_by_license(self, commercial_use: bool = False) -> List[Dict]:
        """
        라이선스 기준으로 문서 필터링
        
        Args:
            commercial_use: 상업적 사용 가능 여부
            
        Returns:
            필터링된 문서 리스트
        """
        filtered = []
        
        for doc in self.documents:
            license_info = doc["metadata"].get("license", {})
            
            # 비상업적 라이선스만 허용 (대회 규칙)
            if not commercial_use and not license_info.get("commercial_use", True):
                filtered.append(doc)
            elif commercial_use and license_info.get("commercial_use", False):
                filtered.append(doc)
        
        logger.info(f"Filtered {len(filtered)} documents with commercial_use={commercial_use}")
        return filtered
    
    def save_processed_data(self, output_dir: str):
        """
        전처리된 데이터 저장
        
        Args:
            output_dir: 출력 디렉토리
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # 문서별로 저장
        for idx, doc in enumerate(self.documents):
            output_file = output_path / f"document_{idx:04d}.json"
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(doc, f, ensure_ascii=False, indent=2)
        
        # 메타데이터 요약 저장
        summary = {
            "total_documents": len(self.documents),
            "processed_date": datetime.now().isoformat(),
            "source_directory": str(self.data_dir),
            "file_types": {},
            "licenses": {}
        }
        
        for doc in self.documents:
            file_type = doc["metadata"]["file_type"]
            summary["file_types"][file_type] = summary["file_types"].get(file_type, 0) + 1
            
            license_type = doc["metadata"]["license"]["type"]
            summary["licenses"][license_type] = summary["licenses"].get(license_type, 0) + 1
        
        with open(output_path / "processing_summary.json", 'w', encoding='utf-8') as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
        
        logger.info(f"Saved {len(self.documents)} processed documents to {output_path}")
    
    def get_statistics(self) -> Dict:
        """
        로드된 데이터의 통계 정보 반환
        
        Returns:
            통계 정보 딕셔너리
        """
        stats = {
            "total_documents": len(self.documents),
            "total_characters": sum(len(doc["content"]) for doc in self.documents),
            "file_types": {},
            "licenses": {},
            "avg_document_length": 0
        }
        
        if self.documents:
            stats["avg_document_length"] = stats["total_characters"] / stats["total_documents"]
            
            for doc in self.documents:
                file_type = doc["metadata"]["file_type"]
                stats["file_types"][file_type] = stats["file_types"].get(file_type, 0) + 1
                
                license_type = doc["metadata"]["license"]["type"]
                stats["licenses"][license_type] = stats["licenses"].get(license_type, 0) + 1
        
        return stats


if __name__ == "__main__":
    # 테스트용 코드
    import sys
    sys.path.append(str(Path(__file__).parent.parent.parent))
    
    logging.basicConfig(level=logging.INFO)
    
    # 데이터 로더 초기화
    loader = FinancialDataLoader("data/external")
    
    # 문서 로드
    documents = loader.parse_financial_documents()
    
    # 통계 출력
    stats = loader.get_statistics()
    print(f"\n로드된 문서 통계:")
    print(f"- 총 문서 수: {stats['total_documents']}")
    print(f"- 총 문자 수: {stats['total_characters']:,}")
    print(f"- 평균 문서 길이: {stats['avg_document_length']:.0f}")
    print(f"- 파일 형식: {stats['file_types']}")
    print(f"- 라이선스: {stats['licenses']}")
    
    # 비상업적 라이선스 문서만 필터링
    non_commercial = loader.filter_by_license(commercial_use=False)
    print(f"\n비상업적 라이선스 문서: {len(non_commercial)}개")
    
    # 처리된 데이터 저장
    loader.save_processed_data("data/processed")