#!/usr/bin/env python3
"""
CoT(Chain of Thought) 기반 데이터 증강 실행 스크립트
고품질 금융 문제 생성을 위한 메인 실행 파일
"""

import os
import sys
import json
import logging
import argparse
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional

# 프로젝트 루트 경로 추가
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from src.data_augmentation.augmentation_pipeline import FSKUAugmentationPipeline
from src.data_augmentation.cot_generator import CoTQuestionGenerator
from src.data_augmentation.reasoning_templates import reasoning_templates


def setup_logging(log_level: str = "INFO") -> None:
    """로깅 설정"""
    log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format=log_format,
        handlers=[
            logging.FileHandler(f'augmentation_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'),
            logging.StreamHandler()
        ]
    )


def load_config(config_path: Optional[str] = None) -> Dict:
    """설정 파일 로드 또는 기본 설정 반환"""
    
    default_config = {
        # 데이터 경로
        'data_dir': 'data/external',
        'output_dir': 'data/augmented',
        
        # 모델 설정
        'model_name': 'beomi/SOLAR-10.7B-v1.0',  # 한국어 금융 모델
        'use_quantization': True,  # RTX 4090 24GB 대응
        
        # CoT 설정
        'use_cot': True,
        'use_cot_ratio': 0.3,  # 30% 문제에 CoT 적용
        'cot_confidence_threshold': 80.0,  # CoT 신뢰도 임계값
        
        # 문제 생성 설정
        'target_count': 5000,  # 목표 문제 수
        'multiple_choice_ratio': 0.7,  # 객관식 비율
        'mc4_ratio': 0.5,  # 4지선다 비율 (나머지는 5지선다)
        
        # 품질 관리
        'similarity_threshold': 0.85,  # 중복 제거 임계값
        'min_confidence_score': 70.0,  # 최소 신뢰도 점수
        
        # 증강 전략
        'augmentation_strategies': [
            'difficulty_variation',
            'paraphrase', 
            'scenario_application'
        ],
        'augmentation_ratio': 0.2,  # 원본 대비 증강 비율
        
        # 배치 처리
        'batch_size': 50,
        'save_interval': 500  # N개마다 중간 저장
    }
    
    if config_path and Path(config_path).exists():
        with open(config_path, 'r', encoding='utf-8') as f:
            loaded_config = json.load(f)
            default_config.update(loaded_config)
            logging.info(f"Loaded configuration from {config_path}")
    
    return default_config


def validate_environment() -> bool:
    """실행 환경 검증"""
    checks = {
        'CUDA Available': False,
        'Memory Sufficient': False,
        'Data Directory Exists': False,
        'Output Directory Writable': False
    }
    
    # CUDA 확인
    try:
        import torch
        checks['CUDA Available'] = torch.cuda.is_available()
        if checks['CUDA Available']:
            device_name = torch.cuda.get_device_name(0)
            memory_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3
            logging.info(f"GPU: {device_name} ({memory_gb:.1f}GB)")
            checks['Memory Sufficient'] = memory_gb >= 20  # 최소 20GB 권장
    except ImportError:
        logging.warning("PyTorch not properly installed")
    
    # 디렉토리 확인
    data_dir = Path('data/external')
    output_dir = Path('data/augmented')
    
    checks['Data Directory Exists'] = data_dir.exists()
    
    try:
        output_dir.mkdir(parents=True, exist_ok=True)
        test_file = output_dir / 'test_write.tmp'
        test_file.touch()
        test_file.unlink()
        checks['Output Directory Writable'] = True
    except:
        checks['Output Directory Writable'] = False
    
    # 결과 출력
    all_passed = all(checks.values())
    
    logging.info("=" * 60)
    logging.info("Environment Validation:")
    for check, passed in checks.items():
        status = "✓" if passed else "✗"
        logging.info(f"  {status} {check}: {passed}")
    logging.info("=" * 60)
    
    return all_passed


def run_augmentation(config: Dict) -> Dict:
    """데이터 증강 실행"""
    
    logging.info("Starting data augmentation with CoT...")
    logging.info(f"Target: {config['target_count']} questions")
    logging.info(f"CoT Usage: {config['use_cot_ratio']*100:.0f}%")
    
    # 파이프라인 초기화
    pipeline = FSKUAugmentationPipeline(config)
    
    # 증강 실행
    result = pipeline.run(target_count=config['target_count'])
    
    # 결과 저장
    output_dir = config['output_dir']
    pipeline.save_results(output_dir)
    
    return result


def analyze_results(result: Dict, config: Dict) -> None:
    """결과 분석 및 리포트 생성"""
    
    data = result['data']
    stats = result['statistics']
    metadata = result['metadata']
    
    # CoT 사용 통계
    cot_questions = [q for q in data if q.get('generation_method') == 'cot' 
                     or q.get('augmentation_method') == 'cot_based']
    
    # 문제 유형별 통계
    type_stats = {}
    for q in data:
        q_type = q.get('type', 'unknown')
        type_stats[q_type] = type_stats.get(q_type, 0) + 1
    
    # 신뢰도 분포 (CoT 생성 문제만)
    confidence_scores = [q.get('confidence_score', 0) for q in cot_questions 
                        if 'confidence_score' in q]
    
    avg_confidence = sum(confidence_scores) / len(confidence_scores) if confidence_scores else 0
    
    # 리포트 생성
    report = {
        'summary': {
            'total_questions': len(data),
            'target_questions': config['target_count'],
            'achievement_rate': (len(data) / config['target_count'] * 100),
            'processing_time_seconds': stats['time_elapsed'],
            'questions_per_minute': (len(data) / stats['time_elapsed'] * 60) if stats['time_elapsed'] > 0 else 0
        },
        'cot_statistics': {
            'total_cot_questions': len(cot_questions),
            'cot_percentage': (len(cot_questions) / len(data) * 100) if data else 0,
            'average_confidence': avg_confidence,
            'high_confidence_count': sum(1 for s in confidence_scores if s >= 80),
            'low_confidence_count': sum(1 for s in confidence_scores if s < 70)
        },
        'type_distribution': type_stats,
        'source_statistics': {
            'documents_processed': stats['documents_processed'],
            'concepts_extracted': stats['concepts_extracted'],
            'questions_generated': stats['questions_generated'],
            'questions_validated': stats['questions_validated']
        }
    }
    
    # 리포트 저장
    output_dir = Path(config['output_dir'])
    report_file = output_dir / f"augmentation_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    
    with open(report_file, 'w', encoding='utf-8') as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    
    # 콘솔 출력
    print("\n" + "=" * 60)
    print("AUGMENTATION RESULTS")
    print("=" * 60)
    print(f"Total Questions Generated: {report['summary']['total_questions']:,}")
    print(f"Achievement Rate: {report['summary']['achievement_rate']:.1f}%")
    print(f"Processing Time: {report['summary']['processing_time_seconds']:.1f} seconds")
    print(f"Generation Speed: {report['summary']['questions_per_minute']:.1f} questions/min")
    print("\nCoT Statistics:")
    print(f"  - CoT Questions: {report['cot_statistics']['total_cot_questions']:,} ({report['cot_statistics']['cot_percentage']:.1f}%)")
    print(f"  - Average Confidence: {report['cot_statistics']['average_confidence']:.1f}%")
    print(f"  - High Confidence (≥80%): {report['cot_statistics']['high_confidence_count']}")
    print(f"  - Low Confidence (<70%): {report['cot_statistics']['low_confidence_count']}")
    print("\nType Distribution:")
    for q_type, count in type_stats.items():
        percentage = (count / len(data) * 100) if data else 0
        print(f"  - {q_type}: {count:,} ({percentage:.1f}%)")
    print("=" * 60)
    print(f"Report saved to: {report_file}")
    
    # 샘플 출력
    print("\n" + "=" * 60)
    print("SAMPLE QUESTIONS (with CoT)")
    print("=" * 60)
    
    # CoT로 생성된 문제 중 상위 3개 출력
    cot_samples = sorted(cot_questions, 
                        key=lambda x: x.get('confidence_score', 0), 
                        reverse=True)[:3]
    
    for i, sample in enumerate(cot_samples, 1):
        print(f"\n--- Sample {i} (Confidence: {sample.get('confidence_score', 0):.1f}%) ---")
        print(f"Question: {sample.get('question', 'N/A')}")
        
        if 'options' in sample:
            print("Options:")
            for j, opt in enumerate(sample['options'], 1):
                prefix = "→" if j == sample.get('answer') else " "
                print(f"  {prefix} {j}) {opt}")
        elif 'answer' in sample:
            print(f"Answer: {sample['answer'][:200]}...")
        
        if 'reasoning_trace' in sample:
            print(f"Reasoning Steps: {len(sample['reasoning_trace'].split('Step'))-1}")


def main():
    """메인 실행 함수"""
    parser = argparse.ArgumentParser(
        description='Run FSKU data augmentation with Chain of Thought'
    )
    parser.add_argument(
        '--config', 
        type=str, 
        help='Path to configuration file (JSON)'
    )
    parser.add_argument(
        '--target', 
        type=int, 
        default=5000,
        help='Target number of questions to generate'
    )
    parser.add_argument(
        '--cot-ratio', 
        type=float, 
        default=0.3,
        help='Ratio of questions to generate with CoT (0.0-1.0)'
    )
    parser.add_argument(
        '--model', 
        type=str,
        help='Model name to use for generation'
    )
    parser.add_argument(
        '--log-level',
        type=str,
        default='INFO',
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
        help='Logging level'
    )
    parser.add_argument(
        '--validate-only',
        action='store_true',
        help='Only validate environment without running augmentation'
    )
    
    args = parser.parse_args()
    
    # 로깅 설정
    setup_logging(args.log_level)
    
    # 설정 로드
    config = load_config(args.config)
    
    # 명령줄 인자로 설정 오버라이드
    if args.target:
        config['target_count'] = args.target
    if args.cot_ratio is not None:
        config['use_cot_ratio'] = args.cot_ratio
    if args.model:
        config['model_name'] = args.model
    
    # 환경 검증
    if not validate_environment():
        logging.error("Environment validation failed!")
        if not args.validate_only:
            return 1
    
    if args.validate_only:
        logging.info("Environment validation completed.")
        return 0
    
    try:
        # 증강 실행
        result = run_augmentation(config)
        
        # 결과 분석
        analyze_results(result, config)
        
        logging.info("Augmentation completed successfully!")
        return 0
        
    except KeyboardInterrupt:
        logging.warning("Augmentation interrupted by user")
        return 2
        
    except Exception as e:
        logging.error(f"Augmentation failed: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())