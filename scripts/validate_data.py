#!/usr/bin/env python3
"""
데이터 검증 스크립트
생성된 데이터의 품질과 규칙 준수를 검증
"""

import sys
import os
import json
import argparse
import logging
from pathlib import Path
from datetime import datetime
import pandas as pd
from typing import Dict, List

# 프로젝트 루트 추가
sys.path.append(str(Path(__file__).parent.parent))

from src.data_augmentation import DataQualityChecker


def setup_logging(log_level: str = 'INFO'):
    """로깅 설정"""
    logging.basicConfig(
        level=getattr(logging, log_level),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    return logging.getLogger(__name__)


def load_data(file_path: str) -> List[Dict]:
    """데이터 로드"""
    file_path = Path(file_path)
    
    if not file_path.exists():
        raise FileNotFoundError(f"파일을 찾을 수 없습니다: {file_path}")
    
    data = []
    
    if file_path.suffix == '.json':
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    
    elif file_path.suffix == '.jsonl':
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    data.append(json.loads(line))
    
    elif file_path.suffix == '.csv':
        df = pd.read_csv(file_path)
        data = df.to_dict('records')
    
    else:
        raise ValueError(f"지원하지 않는 파일 형식: {file_path.suffix}")
    
    return data


def check_competition_rules(data: List[Dict]) -> Dict:
    """대회 규칙 준수 여부 확인"""
    violations = {
        'test_data_usage': [],
        'api_usage': [],
        'commercial_license': [],
        'format_issues': [],
        'single_model_violation': []
    }
    
    for i, item in enumerate(data):
        # test.csv 사용 여부 체크
        if 'source' in item and 'test.csv' in str(item.get('source', '')):
            violations['test_data_usage'].append(i)
        
        # API 사용 여부 체크 (금지된 키워드)
        forbidden_apis = ['openai', 'gpt', 'claude', 'gemini', 'api_key']
        content = json.dumps(item).lower()
        for api in forbidden_apis:
            if api in content:
                violations['api_usage'].append(i)
                break
        
        # 라이선스 체크
        if 'license' in item:
            license_info = item['license']
            if isinstance(license_info, dict) and license_info.get('commercial_use', False):
                violations['commercial_license'].append(i)
        
        # 형식 체크
        if 'type' not in item or 'question' not in item:
            violations['format_issues'].append(i)
    
    return violations


def analyze_data_quality(data: List[Dict]) -> Dict:
    """데이터 품질 분석"""
    analysis = {
        'total': len(data),
        'types': {},
        'sources': {},
        'avg_question_length': 0,
        'avg_answer_length': 0,
        'missing_fields': {},
        'duplicates': 0
    }
    
    # 타입별 분포
    for item in data:
        item_type = item.get('type', 'unknown')
        analysis['types'][item_type] = analysis['types'].get(item_type, 0) + 1
    
    # 소스별 분포
    for item in data:
        source = item.get('source_document', 'unknown')
        analysis['sources'][source] = analysis['sources'].get(source, 0) + 1
    
    # 평균 길이
    question_lengths = [len(item.get('question', '')) for item in data]
    answer_lengths = [len(str(item.get('answer', ''))) for item in data if 'answer' in item]
    
    if question_lengths:
        analysis['avg_question_length'] = sum(question_lengths) / len(question_lengths)
    
    if answer_lengths:
        analysis['avg_answer_length'] = sum(answer_lengths) / len(answer_lengths)
    
    # 필드 누락 체크
    required_fields = ['type', 'question', 'answer']
    for field in required_fields:
        missing = sum(1 for item in data if field not in item)
        if missing > 0:
            analysis['missing_fields'][field] = missing
    
    # 중복 체크 (간단한 방법)
    questions = [item.get('question', '') for item in data]
    analysis['duplicates'] = len(questions) - len(set(questions))
    
    return analysis


def validate_fsku_format(data: List[Dict]) -> Dict:
    """FSKU 형식 검증"""
    issues = {
        'invalid_mc_options': [],
        'invalid_answer_index': [],
        'missing_keywords': [],
        'invalid_question_format': []
    }
    
    for i, item in enumerate(data):
        item_type = item.get('type', '')
        
        # 객관식 검증
        if 'multiple_choice' in item_type:
            # 선택지 개수 체크
            expected_options = int(item_type.split('_')[-1]) if '_' in item_type else 4
            actual_options = len(item.get('options', []))
            
            if actual_options != expected_options:
                issues['invalid_mc_options'].append({
                    'index': i,
                    'expected': expected_options,
                    'actual': actual_options
                })
            
            # 정답 인덱스 체크
            answer_idx = item.get('answer', 0)
            if answer_idx < 1 or answer_idx > actual_options:
                issues['invalid_answer_index'].append({
                    'index': i,
                    'answer_idx': answer_idx,
                    'options_count': actual_options
                })
        
        # 주관식 검증
        elif item_type == 'essay':
            if 'keywords' not in item or not item['keywords']:
                issues['missing_keywords'].append(i)
        
        # 질문 형식 체크
        question = item.get('question', '')
        if not question or len(question) < 10:
            issues['invalid_question_format'].append(i)
    
    return issues


def generate_report(data: List[Dict], output_file: str = None):
    """검증 보고서 생성"""
    print("\n" + "=" * 60)
    print("데이터 검증 보고서")
    print("=" * 60)
    
    # 대회 규칙 검증
    print("\n[대회 규칙 준수 여부]")
    violations = check_competition_rules(data)
    
    for rule, indices in violations.items():
        if indices:
            print(f"⚠️  {rule}: {len(indices)}건 위반")
        else:
            print(f"✅ {rule}: 위반 없음")
    
    # 데이터 품질 분석
    print("\n[데이터 품질 분석]")
    quality = analyze_data_quality(data)
    
    print(f"총 데이터 수: {quality['total']}")
    print(f"타입별 분포:")
    for t, count in quality['types'].items():
        print(f"  - {t}: {count}개 ({count/quality['total']*100:.1f}%)")
    
    print(f"평균 질문 길이: {quality['avg_question_length']:.1f}자")
    print(f"평균 답변 길이: {quality['avg_answer_length']:.1f}자")
    print(f"중복 문제: {quality['duplicates']}개")
    
    if quality['missing_fields']:
        print("필드 누락:")
        for field, count in quality['missing_fields'].items():
            print(f"  - {field}: {count}개")
    
    # FSKU 형식 검증
    print("\n[FSKU 형식 검증]")
    format_issues = validate_fsku_format(data)
    
    for issue_type, issues in format_issues.items():
        if issues:
            print(f"⚠️  {issue_type}: {len(issues)}건")
        else:
            print(f"✅ {issue_type}: 문제 없음")
    
    # 종합 평가
    print("\n[종합 평가]")
    
    total_violations = sum(len(v) for v in violations.values())
    total_format_issues = sum(len(v) for v in format_issues.values())
    
    if total_violations == 0 and total_format_issues == 0:
        print("✅ 모든 검증 통과!")
    else:
        print(f"⚠️  수정 필요: 규칙 위반 {total_violations}건, 형식 문제 {total_format_issues}건")
    
    # 보고서 파일 저장
    if output_file:
        report = {
            'timestamp': datetime.now().isoformat(),
            'total_data': quality['total'],
            'violations': violations,
            'quality_analysis': quality,
            'format_issues': format_issues
        }
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        
        print(f"\n📄 보고서 저장: {output_file}")
    
    print("=" * 60)


def main():
    """메인 함수"""
    parser = argparse.ArgumentParser(description='생성된 데이터 검증')
    
    parser.add_argument(
        'input_file',
        type=str,
        help='검증할 데이터 파일 (JSON/JSONL/CSV)'
    )
    
    parser.add_argument(
        '--output',
        type=str,
        help='검증 보고서 출력 파일'
    )
    
    parser.add_argument(
        '--log_level',
        type=str,
        default='INFO',
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
        help='로그 레벨'
    )
    
    parser.add_argument(
        '--check_quality',
        action='store_true',
        help='품질 검증 수행'
    )
    
    args = parser.parse_args()
    
    # 로깅 설정
    logger = setup_logging(args.log_level)
    
    try:
        # 데이터 로드
        logger.info(f"데이터 로드 중: {args.input_file}")
        data = load_data(args.input_file)
        logger.info(f"로드 완료: {len(data)}개 항목")
        
        # 품질 검증
        if args.check_quality:
            logger.info("품질 검증 수행 중...")
            checker = DataQualityChecker()
            validated, stats = checker.validate_batch(data)
            
            print(f"\n품질 검증 결과:")
            print(f"- 검사: {stats['total_checked']}개")
            print(f"- 통과: {stats['passed']}개")
            print(f"- 실패: {stats['failed']}개")
            print(f"- 중복 제거: {stats['duplicates_removed']}개")
        
        # 보고서 생성
        generate_report(data, args.output)
        
        return 0
        
    except Exception as e:
        logger.error(f"검증 중 오류 발생: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())