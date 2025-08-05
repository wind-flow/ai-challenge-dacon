#!/usr/bin/env python3
"""
추론용 모델 준비 스크립트

4가지 조합을 위한 모델 다운로드 및 최적화
"""

import os
import sys
import torch
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 추론 환경 스펙
RUNPOD_SPEC = {
    'gpu': 'RTX 4090',
    'vram': 24,  # GB
    'ram': 41,   # GB
    'time_limit': 270  # minutes
}

# 테스트할 조합
MODEL_COMBINATIONS = [
    {
        'id': 'combo1',
        'generation': 'Qwen/Qwen2.5-32B-Instruct',
        'inference': 'upstage/SOLAR-10.7B-v1.0',
        'description': 'Qwen-32B 생성 + SOLAR 추론'
    },
    {
        'id': 'combo2',
        'generation': 'Qwen/Qwen2.5-14B-Instruct',
        'inference': 'upstage/SOLAR-10.7B-v1.0',
        'description': 'Qwen-14B 생성 + SOLAR 추론'
    },
    {
        'id': 'combo3',
        'generation': 'Qwen/Qwen2.5-32B-Instruct',
        'inference': 'Qwen/Qwen2.5-7B-Instruct',
        'description': 'Qwen-32B 생성 + Qwen-7B 추론'
    },
    {
        'id': 'combo4',
        'generation': 'upstage/SOLAR-10.7B-v1.0',
        'inference': 'upstage/SOLAR-10.7B-v1.0',
        'description': 'SOLAR 생성 + SOLAR 추론 (베이스라인)'
    }
]

def check_model_size(model_name):
    """모델 크기 추정"""
    try:
        config = AutoModelForCausalLM.from_pretrained(
            model_name,
            trust_remote_code=True
        ).config
        
        # 파라미터 수 계산
        params = config.hidden_size * config.num_hidden_layers * 12  # 대략적 추정
        size_gb = params * 2 / 1024**3  # FP16 기준
        
        return size_gb
    except:
        # 모델 이름에서 추정
        if "32B" in model_name:
            return 64
        elif "14B" in model_name:
            return 28
        elif "10.7B" in model_name or "10B" in model_name:
            return 21
        elif "7B" in model_name:
            return 14
        else:
            return 14  # 기본값

def prepare_inference_model(model_name, output_dir):
    """추론용 모델 준비"""
    logger.info(f"\n🔄 {model_name} 준비 중...")
    
    # 모델 크기 확인
    size_gb = check_model_size(model_name)
    logger.info(f"📊 예상 크기: {size_gb:.1f}GB")
    
    # 메모리 체크
    if size_gb > RUNPOD_SPEC['vram']:
        logger.warning(f"⚠️ 모델이 VRAM보다 큽니다! 양자화 필요")
        quantization = "4bit"
    elif size_gb > RUNPOD_SPEC['vram'] * 0.8:
        quantization = "8bit"
    else:
        quantization = None
    
    # 출력 경로
    model_dir = Path(output_dir) / model_name.replace("/", "_")
    model_dir.mkdir(parents=True, exist_ok=True)
    
    # 다운로드 정보 저장
    info = {
        'model_name': model_name,
        'size_gb': size_gb,
        'quantization': quantization,
        'runpod_compatible': size_gb <= RUNPOD_SPEC['vram'],
        'recommended_batch_size': 1 if size_gb > 20 else 2 if size_gb > 14 else 4
    }
    
    import json
    with open(model_dir / "model_info.json", 'w') as f:
        json.dump(info, f, indent=2)
    
    logger.info(f"✅ 모델 정보 저장: {model_dir}")
    
    return info

def generate_inference_script(combo, model_info):
    """추론 스크립트 생성"""
    script = f'''#!/usr/bin/env python3
"""
자동 생성된 추론 스크립트
조합: {combo['description']}
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import pandas as pd
import json
from tqdm import tqdm

# 설정
MODEL_NAME = "{combo['inference']}"
QUANTIZATION = {model_info.get('quantization', None)}
BATCH_SIZE = {model_info['recommended_batch_size']}

def load_model():
    """모델 로드"""
    print(f"🔄 모델 로딩: {{MODEL_NAME}}")
    
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME,
        trust_remote_code=True
    )
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    model_kwargs = {{
        'trust_remote_code': True,
        'device_map': 'auto'
    }}
    
    if QUANTIZATION == "4bit":
        from transformers import BitsAndBytesConfig
        model_kwargs['quantization_config'] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16
        )
    elif QUANTIZATION == "8bit":
        from transformers import BitsAndBytesConfig
        model_kwargs['quantization_config'] = BitsAndBytesConfig(
            load_in_8bit=True
        )
    else:
        model_kwargs['torch_dtype'] = torch.float16
    
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        **model_kwargs
    )
    
    return model, tokenizer

def run_inference(test_file="data/test.csv", output_file="submission.csv"):
    """추론 실행"""
    # 모델 로드
    model, tokenizer = load_model()
    
    # 테스트 데이터 로드
    df = pd.read_csv(test_file)
    results = []
    
    # 배치 처리
    for i in tqdm(range(0, len(df), BATCH_SIZE)):
        batch = df.iloc[i:i+BATCH_SIZE]
        
        # 프롬프트 생성
        prompts = []
        for _, row in batch.iterrows():
            prompt = create_prompt(row)
            prompts.append(prompt)
        
        # 추론
        inputs = tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=1024
        ).to(model.device)
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=256,
                temperature=0.7,
                top_p=0.9,
                do_sample=True
            )
        
        # 디코딩
        for j, output in enumerate(outputs):
            text = tokenizer.decode(output, skip_special_tokens=True)
            answer = extract_answer(text, prompts[j])
            results.append({{
                'id': batch.iloc[j]['id'],
                'answer': answer
            }})
    
    # 저장
    result_df = pd.DataFrame(results)
    result_df.to_csv(output_file, index=False)
    print(f"✅ 결과 저장: {{output_file}}")

def create_prompt(row):
    """프롬프트 생성"""
    if row['type'] == 'multiple_choice':
        return f"""다음 금융 문제에 대한 정답을 선택하세요.

문제: {{row['question']}}

선택지:
{{row['choices']}}

정답 번호만 출력하세요: """
    else:
        return f"""다음 금융 문제에 대해 전문가의 관점에서 답변하세요.

문제: {{row['question']}}

답변: """

def extract_answer(text, prompt):
    """답변 추출"""
    answer = text.replace(prompt, "").strip()
    
    # 객관식인 경우 번호만 추출
    if "정답 번호만" in prompt:
        import re
        match = re.search(r'[1-4]', answer)
        if match:
            return match.group()
    
    return answer

if __name__ == "__main__":
    import time
    start = time.time()
    
    run_inference()
    
    elapsed = time.time() - start
    print(f"⏱️ 총 소요 시간: {{elapsed/60:.1f}}분")
    
    if elapsed > {RUNPOD_SPEC['time_limit']} * 60:
        print("❌ 시간 초과!")
    else:
        print("✅ 시간 내 완료!")
'''
    
    return script

def main():
    """메인 실행"""
    print("="*60)
    print("🚀 추론 모델 준비")
    print("="*60)
    
    output_dir = Path("models/inference_ready")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 각 조합 준비
    for combo in MODEL_COMBINATIONS:
        print(f"\n📋 {combo['description']}")
        print("-"*60)
        
        # 추론 모델 정보
        model_info = prepare_inference_model(
            combo['inference'],
            output_dir
        )
        
        # 스크립트 생성
        script_path = output_dir / f"{combo['id']}_inference.py"
        script = generate_inference_script(combo, model_info)
        
        with open(script_path, 'w') as f:
            f.write(script)
        
        print(f"✅ 스크립트 생성: {script_path}")
        
        # 요약
        print(f"\n📊 조합 요약:")
        print(f"- 생성 모델: {combo['generation']}")
        print(f"- 추론 모델: {combo['inference']}")
        print(f"- 양자화: {model_info.get('quantization', 'None')}")
        print(f"- 권장 배치: {model_info['recommended_batch_size']}")
        print(f"- RunPod 호환: {'✅' if model_info['runpod_compatible'] else '❌'}")
    
    # 최종 권장사항
    print("\n" + "="*60)
    print("💡 권장사항:")
    print("1. combo1 (Qwen-32B + SOLAR): 최고 품질")
    print("2. combo2 (Qwen-14B + SOLAR): 균형")
    print("3. combo4 (SOLAR + SOLAR): 베이스라인")
    print("\n다음 단계:")
    print("1. Colab에서 각 조합별 데이터 생성")
    print("2. 생성된 데이터로 파인튜닝")
    print("3. RunPod에서 추론 테스트")
    print("4. 최고 성능 조합 선택")

if __name__ == "__main__":
    main()