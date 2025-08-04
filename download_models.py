#!/usr/bin/env python3
"""
FSKU 프로젝트용 모델 다운로드 스크립트

주요 모델들을 사전에 다운로드하여 실행 시간 단축
"""

import os
import sys
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

def print_gpu_info():
    """GPU 정보 출력"""
    print("\n📊 시스템 정보:")
    print(f"  - PyTorch 버전: {torch.__version__}")
    print(f"  - CUDA 사용 가능: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"  - GPU 이름: {torch.cuda.get_device_name(0)}")
        print(f"  - GPU 메모리: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f}GB")
    print()

def download_model(model_name: str, use_4bit: bool = False):
    """
    모델 다운로드
    
    Args:
        model_name: 다운로드할 모델 이름
        use_4bit: 4비트 양자화 사용 여부
    """
    print(f"\n📥 다운로드 중: {model_name}")
    print("  (첫 다운로드는 시간이 걸립니다...)")
    
    try:
        # 토크나이저 다운로드
        print("  - 토크나이저 다운로드 중...")
        tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=True
        )
        
        # 모델 다운로드 설정
        if use_4bit and torch.cuda.is_available():
            from transformers import BitsAndBytesConfig
            
            print("  - 4비트 양자화 모델 다운로드 중...")
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16
            )
            
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                quantization_config=bnb_config,
                device_map="auto",
                trust_remote_code=True
            )
        else:
            print("  - 전체 모델 다운로드 중...")
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                device_map="auto" if torch.cuda.is_available() else None,
                trust_remote_code=True
            )
        
        # 캐시 경로 확인
        cache_dir = Path.home() / ".cache" / "huggingface" / "hub"
        model_dir = list(cache_dir.glob(f"models--{model_name.replace('/', '--')}*"))
        
        if model_dir:
            size_gb = sum(f.stat().st_size for f in model_dir[0].rglob("*") if f.is_file()) / (1024**3)
            print(f"  ✅ 완료! (크기: {size_gb:.1f}GB)")
        else:
            print(f"  ✅ 완료!")
            
        return True
        
    except Exception as e:
        print(f"  ❌ 실패: {str(e)}")
        return False

def main():
    """메인 함수"""
    print("╔════════════════════════════════════════╗")
    print("║     FSKU 모델 다운로드 스크립트        ║")
    print("╚════════════════════════════════════════╝")
    
    # GPU 정보 확인
    print_gpu_info()
    
    # 추천 모델 목록
    models = [
        ("beomi/SOLAR-10.7B-v1.0", "SOLAR 10.7B (추천)", True),
        ("LG-AI-EXAONE/EXAONE-3.0-7.8B-Instruct", "LG EXAONE 7.8B", True),
        ("Qwen/Qwen2.5-7B-Instruct", "Qwen 2.5 7B (다국어)", True),
        ("beomi/llama-2-ko-7b", "Llama2 한국어 7B", False),
    ]
    
    print("📋 다운로드 가능한 모델:")
    for i, (model_name, desc, _) in enumerate(models, 1):
        print(f"  {i}. {desc}")
        print(f"     모델명: {model_name}")
    print(f"  {len(models)+1}. 전체 다운로드")
    print(f"  0. 취소")
    
    # 사용자 선택
    choice = input("\n선택 (번호 입력): ").strip()
    
    if choice == "0":
        print("취소되었습니다.")
        return
    
    # 4비트 양자화 옵션
    use_4bit = False
    if torch.cuda.is_available():
        use_4bit_input = input("\n4비트 양자화 사용? (메모리 절약) [y/N]: ").strip().lower()
        use_4bit = use_4bit_input == 'y'
        if use_4bit:
            print("➡️ 4비트 양자화 모드 활성화")
    
    # 다운로드 실행
    if choice == str(len(models)+1):
        # 전체 다운로드
        print("\n전체 모델 다운로드를 시작합니다...")
        success_count = 0
        for model_name, desc, recommended in models:
            if recommended or not use_4bit:  # 추천 모델이거나 양자화 미사용시
                if download_model(model_name, use_4bit):
                    success_count += 1
        
        print(f"\n✅ 완료! {success_count}/{len(models)}개 모델 다운로드됨")
    
    elif choice.isdigit() and 1 <= int(choice) <= len(models):
        # 선택한 모델만 다운로드
        idx = int(choice) - 1
        model_name, desc, _ = models[idx]
        download_model(model_name, use_4bit)
    
    else:
        print("잘못된 선택입니다.")
        return
    
    # 완료 메시지
    print("\n" + "="*50)
    print("💡 다음 단계:")
    print("  1. python main.py 실행")
    print("  2. '데이터 생성' 선택")
    print("  3. 다운로드한 모델 사용")
    print("="*50)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n중단되었습니다.")
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ 오류 발생: {e}")
        sys.exit(1)