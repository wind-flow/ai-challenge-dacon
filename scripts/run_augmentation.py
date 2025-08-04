#!/usr/bin/env python3
"""
데이터 증강 실행 스크립트
2025 금융 AI Challenge Track1을 위한 데이터 증강
"""

import sys
import os
import argparse
import logging
import json
from pathlib import Path
from datetime import datetime

# 프로젝트 루트 추가
sys.path.append(str(Path(__file__).parent.parent))

from src.data_augmentation import FSKUAugmentationPipeline


def setup_logging(log_level: str = 'INFO'):
    """로깅 설정"""
    log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    
    # 콘솔 핸들러
    console_handler = logging.StreamHandler()
    console_handler.setLevel(getattr(logging, log_level))
    
    # 파일 핸들러
    log_file = f"logs/augmentation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    os.makedirs('logs', exist_ok=True)
    file_handler = logging.FileHandler(log_file, encoding='utf-8')
    file_handler.setLevel(logging.DEBUG)
    
    # 로거 설정
    logging.basicConfig(
        level=logging.DEBUG,
        format=log_format,
        handlers=[console_handler, file_handler]
    )
    
    return log_file


def parse_arguments():
    """명령줄 인수 파싱"""
    parser = argparse.ArgumentParser(
        description='FSKU 데이터 증강 파이프라인 실행'
    )
    
    # 데이터 관련
    parser.add_argument(
        '--external_data_dir',
        type=str,
        default='./data/external',
        help='외부 데이터 디렉토리 경로'
    )
    
    parser.add_argument(
        '--output_dir',
        type=str,
        default='./data/augmented',
        help='출력 디렉토리 경로'
    )
    
    # 생성 관련
    parser.add_argument(
        '--target_count',
        type=int,
        default=5000,
        help='목표 문제 수 (기본값: 5000)'
    )
    
    parser.add_argument(
        '--mc_ratio',
        type=float,
        default=0.7,
        help='객관식 문제 비율 (기본값: 0.7)'
    )
    
    # 모델 관련
    parser.add_argument(
        '--model_name',
        type=str,
        default='beomi/SOLAR-10.7B-v1.0',
        help='사용할 LLM 모델 이름'
    )
    
    parser.add_argument(
        '--use_quantization',
        action='store_true',
        default=True,
        help='4bit 양자화 사용 여부'
    )
    
    parser.add_argument(
        '--no_quantization',
        action='store_false',
        dest='use_quantization',
        help='양자화 사용 안함'
    )
    
    # 품질 관련
    parser.add_argument(
        '--similarity_threshold',
        type=float,
        default=0.85,
        help='중복 판단 유사도 임계값 (기본값: 0.85)'
    )
    
    # 기타
    parser.add_argument(
        '--batch_size',
        type=int,
        default=32,
        help='배치 크기 (기본값: 32)'
    )
    
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='랜덤 시드 (기본값: 42)'
    )
    
    parser.add_argument(
        '--log_level',
        type=str,
        default='INFO',
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
        help='로그 레벨'
    )
    
    parser.add_argument(
        '--dry_run',
        action='store_true',
        help='테스트 실행 (10개만 생성)'
    )
    
    parser.add_argument(
        '--config_file',
        type=str,
        help='설정 파일 경로 (JSON)'
    )
    
    return parser.parse_args()


def load_config(args):
    """설정 로드"""
    config = {
        'data_dir': args.external_data_dir,
        'output_dir': args.output_dir,
        'model_name': args.model_name,
        'use_quantization': args.use_quantization,
        'similarity_threshold': args.similarity_threshold,
        'multiple_choice_ratio': args.mc_ratio,
        'batch_size': args.batch_size,
        'seed': args.seed
    }
    
    # 설정 파일이 있으면 로드
    if args.config_file and Path(args.config_file).exists():
        with open(args.config_file, 'r', encoding='utf-8') as f:
            file_config = json.load(f)
            config.update(file_config)
    
    return config


def validate_environment():
    """실행 환경 검증"""
    import torch
    
    print("\n=== 환경 검증 ===")
    print(f"Python 버전: {sys.version}")
    print(f"PyTorch 버전: {torch.__version__}")
    print(f"CUDA 사용 가능: {torch.cuda.is_available()}")
    
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU 메모리: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    
    # 필수 디렉토리 확인
    required_dirs = ['data/external', 'data/processed', 'data/augmented']
    for dir_path in required_dirs:
        Path(dir_path).mkdir(parents=True, exist_ok=True)
    
    print("\n환경 검증 완료!")
    return True


def print_banner():
    """배너 출력"""
    banner = """
    ╔═══════════════════════════════════════════════════════════════╗
    ║                                                               ║
    ║     2025 금융 AI Challenge Track1 - 데이터 증강 시스템       ║
    ║                                                               ║
    ║     FSKU (Financial Semantic Korean Understanding)           ║
    ║                                                               ║
    ╚═══════════════════════════════════════════════════════════════╝
    """
    print(banner)


def main():
    """메인 함수"""
    # 배너 출력
    print_banner()
    
    # 인수 파싱
    args = parse_arguments()
    
    # 로깅 설정
    log_file = setup_logging(args.log_level)
    logger = logging.getLogger(__name__)
    
    logger.info("=" * 60)
    logger.info("데이터 증강 파이프라인 시작")
    logger.info(f"로그 파일: {log_file}")
    logger.info("=" * 60)
    
    # 환경 검증
    if not validate_environment():
        logger.error("환경 검증 실패")
        return 1
    
    # 설정 로드
    config = load_config(args)
    
    # 테스트 실행인 경우
    if args.dry_run:
        logger.info("테스트 실행 모드 (10개만 생성)")
        target_count = 10
    else:
        target_count = args.target_count
    
    logger.info(f"설정:")
    logger.info(f"- 외부 데이터: {config['data_dir']}")
    logger.info(f"- 출력 디렉토리: {config['output_dir']}")
    logger.info(f"- 목표 문제 수: {target_count}")
    logger.info(f"- 모델: {config['model_name']}")
    logger.info(f"- 양자화: {config['use_quantization']}")
    logger.info(f"- 객관식 비율: {config['multiple_choice_ratio']}")
    
    try:
        # 파이프라인 초기화
        logger.info("\n파이프라인 초기화 중...")
        pipeline = FSKUAugmentationPipeline(config)
        
        # 파이프라인 실행
        logger.info("\n파이프라인 실행 중...")
        result = pipeline.run(target_count=target_count)
        
        # 결과 출력
        logger.info("\n" + "=" * 60)
        logger.info("파이프라인 실행 완료!")
        logger.info("=" * 60)
        
        stats = result['statistics']
        metadata = result['metadata']
        
        logger.info(f"통계:")
        logger.info(f"- 처리된 문서: {stats['documents_processed']}개")
        logger.info(f"- 추출된 개념: {stats['concepts_extracted']}개")
        logger.info(f"- 생성된 문제: {stats['questions_generated']}개")
        logger.info(f"- 검증 통과: {stats['questions_validated']}개")
        logger.info(f"- 소요 시간: {stats['time_elapsed']:.2f}초")
        logger.info(f"- 최종 문제 수: {metadata['actual_count']}개")
        
        # 결과 저장
        logger.info(f"\n결과 저장 중...")
        pipeline.save_results(config['output_dir'])
        
        # 샘플 출력
        samples = pipeline.get_sample_output(5)
        
        print("\n" + "=" * 60)
        print("생성된 문제 샘플")
        print("=" * 60)
        
        for i, sample in enumerate(samples, 1):
            print(f"\n[문제 {i}]")
            print(f"유형: {sample.get('type', 'N/A')}")
            print(f"질문: {sample.get('question', 'N/A')}")
            
            if 'options' in sample:
                print("선택지:")
                for j, opt in enumerate(sample['options'], 1):
                    if j == sample.get('answer', 0):
                        print(f"  {j}. {opt} ✓")
                    else:
                        print(f"  {j}. {opt}")
            elif 'answer' in sample:
                answer_preview = sample['answer'][:100] + "..." if len(sample['answer']) > 100 else sample['answer']
                print(f"답변: {answer_preview}")
            
            if 'keywords' in sample:
                print(f"핵심 키워드: {', '.join(sample['keywords'])}")
        
        # 품질 보고서
        quality_report = pipeline.checker.generate_quality_report(result['data'])
        
        print("\n" + "=" * 60)
        print("품질 보고서")
        print("=" * 60)
        print(f"총 문제 수: {quality_report['total_questions']}")
        print(f"타입별 분포: {quality_report['type_distribution']}")
        print(f"난이도 분포: {quality_report['difficulty_distribution']}")
        print(f"평균 질문 길이: {quality_report['avg_question_length']:.1f}자")
        print(f"평균 답변 길이: {quality_report['avg_answer_length']:.1f}자")
        print(f"품질 점수: {quality_report['quality_score']:.1f}%")
        
        print("\n" + "=" * 60)
        print("✅ 데이터 증강 완료!")
        print(f"📁 결과 파일: {config['output_dir']}")
        print("=" * 60)
        
        return 0
        
    except Exception as e:
        logger.error(f"파이프라인 실행 중 오류 발생: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    import random
    import numpy as np
    import torch
    
    # 시드 고정
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)
    
    # 메인 함수 실행
    sys.exit(main())