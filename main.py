#!/usr/bin/env python3
"""
FSKU 프로젝트 메인 실행 파일

깔끔하게 정리된 통합 인터페이스
"""

import sys
from pathlib import Path

# src 디렉토리를 경로에 추가
sys.path.append(str(Path(__file__).parent / "src"))

from generate_data.main import generate_data
from training.train import train_model
from infer.inference import run_inference


def print_menu():
    """메뉴 출력"""
    print("""
╔════════════════════════════════════════╗
║         FSKU 프로젝트 시스템           ║
╚════════════════════════════════════════╝

1. 데이터 생성 (Generate Data)
2. 모델 학습 (Train Model)
3. 추론 실행 (Run Inference)
4. 전체 파이프라인 (Full Pipeline)
5. 종료 (Exit)

선택: """, end="")


def data_generation_menu():
    """데이터 생성 설정"""
    print("\n=== 데이터 생성 설정 ===")
    
    # 기본값 제공
    config = {
        'model_name': 'beomi/SOLAR-10.7B-v1.0',
        'use_rag': True,
        'use_quantization': True,
        'num_questions': 100,
        'min_quality': 70,
        'temperature': 0.7,
        'prompt_template': 'prompts/cot.txt'
    }
    
    # 사용자 입력 (Enter 누르면 기본값)
    print("\n기본값을 사용하려면 Enter를 누르세요.")
    
    num = input(f"생성할 문제 수 [{config['num_questions']}]: ").strip()
    if num:
        config['num_questions'] = int(num)
    
    use_rag = input(f"RAG 사용 (y/n) [{'y' if config['use_rag'] else 'n'}]: ").strip().lower()
    if use_rag:
        config['use_rag'] = use_rag == 'y'
    
    # 실행
    print("\n데이터 생성 중...")
    output_path = generate_data(config)
    print(f"\n✅ 완료! 결과: {output_path}")
    
    return output_path


def training_menu():
    """모델 학습 설정"""
    print("\n=== 모델 학습 설정 ===")
    
    # 기본값
    config = {
        'base_model': 'beomi/SOLAR-10.7B-v1.0',
        'use_lora': True,
        'use_qlora': True,
        'lora_r': 16,
        'lora_alpha': 32,
        'num_epochs': 3,
        'batch_size': 4,
        'learning_rate': 2e-4,
        'output_dir': 'models'
    }
    
    print("\n기본값을 사용하려면 Enter를 누르세요.")
    
    # 학습 데이터 경로
    train_data = input("학습 데이터 경로 [data/augmented/train_data.jsonl]: ").strip()
    if train_data:
        config['train_data'] = train_data
    
    epochs = input(f"에폭 수 [{config['num_epochs']}]: ").strip()
    if epochs:
        config['num_epochs'] = int(epochs)
    
    # 실행
    print("\n모델 학습 중...")
    results = train_model(config)
    print(f"\n✅ 완료! 모델: {results['model_path']}")
    
    return results['model_path']


def inference_menu():
    """추론 실행 설정"""
    print("\n=== 추론 실행 설정 ===")
    
    # 모델 경로
    model_path = input("모델 경로: ").strip()
    if not model_path:
        print("모델 경로를 입력해주세요.")
        return
    
    # 테스트 파일
    test_file = input("테스트 파일 경로 [data/test.csv]: ").strip()
    if not test_file:
        test_file = "data/test.csv"
    
    # 출력 파일
    output_file = input("결과 저장 파일명 [inference_results.csv]: ").strip()
    if not output_file:
        output_file = "inference_results.csv"
    
    # 실행
    print("\n추론 실행 중...")
    output_path = run_inference(model_path, test_file, output_file)
    print(f"\n✅ 완료! 결과: {output_path}")


def full_pipeline():
    """전체 파이프라인 실행"""
    print("\n=== 전체 파이프라인 실행 ===")
    
    # 1. 데이터 생성
    print("\n[1/3] 데이터 생성")
    data_config = {
        'model_name': 'beomi/SOLAR-10.7B-v1.0',
        'use_rag': True,
        'num_questions': 100,
        'prompt_template': 'prompts/cot.txt'
    }
    data_path = generate_data(data_config)
    
    # 2. 모델 학습
    print("\n[2/3] 모델 학습")
    train_config = {
        'base_model': 'beomi/SOLAR-10.7B-v1.0',
        'use_lora': True,
        'use_qlora': True,
        'num_epochs': 3,
        'train_data': data_path
    }
    results = train_model(train_config)
    model_path = results['model_path']
    
    # 3. 추론 실행
    print("\n[3/3] 추론 실행")
    output_path = run_inference(model_path, "data/test.csv")
    
    print("\n" + "="*60)
    print("✅ 전체 파이프라인 완료!")
    print(f"  - 생성 데이터: {data_path}")
    print(f"  - 학습 모델: {model_path}")
    print(f"  - 추론 결과: {output_path}")
    print("="*60)


def main():
    """메인 함수"""
    while True:
        print_menu()
        choice = input().strip()
        
        if choice == '1':
            data_generation_menu()
        elif choice == '2':
            training_menu()
        elif choice == '3':
            inference_menu()
        elif choice == '4':
            full_pipeline()
        elif choice == '5':
            print("\n프로그램을 종료합니다.")
            break
        else:
            print("\n잘못된 선택입니다. 다시 선택해주세요.")
        
        input("\n계속하려면 Enter를 누르세요...")


if __name__ == "__main__":
    # 명령줄 인자 처리
    if len(sys.argv) > 1:
        command = sys.argv[1].lower()
        
        if command == "generate":
            # 빠른 데이터 생성
            config = {
                'model_name': 'beomi/SOLAR-10.7B-v1.0',
                'use_rag': True,
                'num_questions': 100
            }
            generate_data(config)
            
        elif command == "train":
            # 빠른 학습
            config = {
                'base_model': 'beomi/SOLAR-10.7B-v1.0',
                'use_lora': True,
                'use_qlora': True,
                'num_epochs': 3
            }
            train_model(config)
            
        elif command == "infer":
            # 빠른 추론
            if len(sys.argv) > 2:
                model_path = sys.argv[2]
                test_file = sys.argv[3] if len(sys.argv) > 3 else "data/test.csv"
                run_inference(model_path, test_file)
            else:
                print("사용법: python main.py infer <model_path> [test_file]")
        
        elif command == "pipeline":
            # 전체 실행
            full_pipeline()
        
        else:
            print(f"알 수 없는 명령: {command}")
            print("사용 가능: generate, train, infer, pipeline")
    else:
        # 대화형 메뉴
        main()