#!/usr/bin/env python3
"""
모델 접근 실패 시 대체 모델 자동 선택

EXAONE이 gated인 경우 자동으로 대체 모델 사용
"""

from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
import torch
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ModelLoader:
    """스마트 모델 로더"""
    
    # 모델 우선순위 (높은 순)
    MODEL_PRIORITY = [
        {
            'name': 'LGAI-EXAONE/EXAONE-3.0-7.8B-Instruct',
            'fallback': 'upstage/SOLAR-10.7B-v1.0',
            'type': 'korean-specialized'
        },
        {
            'name': 'upstage/SOLAR-10.7B-v1.0',
            'fallback': 'Qwen/Qwen2.5-7B-Instruct',
            'type': 'korean-verified'
        },
        {
            'name': 'Qwen/Qwen2.5-14B-Instruct',
            'fallback': 'Qwen/Qwen2.5-7B-Instruct',
            'type': 'large-scale'
        },
        {
            'name': 'Qwen/Qwen2.5-7B-Instruct',
            'fallback': 'NCSOFT/Llama-VARCO-8B-Instruct',
            'type': 'balanced'
        }
    ]
    
    @classmethod
    def load_model(cls, model_name=None, device="auto", quantization=None):
        """
        모델 로드 (실패 시 자동 대체)
        
        Args:
            model_name: 요청한 모델명
            device: 디바이스 설정
            quantization: 양자화 설정 (4bit, 8bit, None)
            
        Returns:
            model, tokenizer, actual_model_name
        """
        # 요청 모델이 없으면 첫 번째 우선순위
        if not model_name:
            model_name = cls.MODEL_PRIORITY[0]['name']
        
        # 우선순위에서 해당 모델 찾기
        model_info = None
        for info in cls.MODEL_PRIORITY:
            if info['name'] == model_name:
                model_info = info
                break
        
        if not model_info:
            # 우선순위에 없는 모델은 그대로 시도
            model_info = {'name': model_name, 'fallback': cls.MODEL_PRIORITY[0]['name']}
        
        # 로드 시도
        model, tokenizer = cls._try_load(model_info['name'], device, quantization)
        
        if model is not None:
            logger.info(f"✅ {model_info['name']} 로드 성공!")
            return model, tokenizer, model_info['name']
        
        # 실패 시 대체 모델
        logger.warning(f"⚠️ {model_info['name']} 로드 실패. 대체 모델 시도...")
        
        fallback_name = model_info['fallback']
        model, tokenizer = cls._try_load(fallback_name, device, quantization)
        
        if model is not None:
            logger.info(f"✅ 대체 모델 {fallback_name} 로드 성공!")
            return model, tokenizer, fallback_name
        
        # 모든 모델 순차 시도
        logger.warning("⚠️ 모든 우선순위 모델 시도 중...")
        
        for info in cls.MODEL_PRIORITY:
            if info['name'] != model_name and info['name'] != fallback_name:
                model, tokenizer = cls._try_load(info['name'], device, quantization)
                if model is not None:
                    logger.info(f"✅ {info['name']} 로드 성공!")
                    return model, tokenizer, info['name']
        
        raise RuntimeError("❌ 사용 가능한 모델이 없습니다!")
    
    @classmethod
    def _try_load(cls, model_name, device, quantization):
        """모델 로드 시도"""
        try:
            logger.info(f"🔄 {model_name} 로드 시도 중...")
            
            # 토크나이저
            tokenizer = AutoTokenizer.from_pretrained(
                model_name,
                trust_remote_code=True
            )
            
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
            
            # 양자화 설정
            model_kwargs = {
                'trust_remote_code': True,
                'device_map': device
            }
            
            if quantization == '4bit':
                from transformers import BitsAndBytesConfig
                model_kwargs['quantization_config'] = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.float16,
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_quant_type="nf4"
                )
            elif quantization == '8bit':
                from transformers import BitsAndBytesConfig
                model_kwargs['quantization_config'] = BitsAndBytesConfig(
                    load_in_8bit=True
                )
            else:
                model_kwargs['torch_dtype'] = torch.float16
            
            # 모델 로드
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                **model_kwargs
            )
            
            return model, tokenizer
            
        except Exception as e:
            logger.error(f"❌ {model_name} 로드 실패: {e}")
            return None, None
    
    @classmethod
    def get_recommended_settings(cls, model_name, gpu_memory_gb):
        """
        GPU 메모리에 따른 권장 설정
        
        Args:
            model_name: 모델명
            gpu_memory_gb: GPU 메모리 (GB)
            
        Returns:
            dict: 권장 설정
        """
        settings = {
            'batch_size': 1,
            'max_length': 2048,
            'quantization': None,
            'gradient_accumulation': 1
        }
        
        # 모델 크기 추정
        if "32B" in model_name:
            model_size = 32
        elif "14B" in model_name:
            model_size = 14
        elif "10.7B" in model_name or "10B" in model_name:
            model_size = 10.7
        elif "8B" in model_name:
            model_size = 8
        elif "7B" in model_name:
            model_size = 7
        elif "6B" in model_name:
            model_size = 6
        else:
            model_size = 7  # 기본값
        
        # GPU 메모리에 따른 설정
        if gpu_memory_gb >= 40:  # A100
            if model_size <= 14:
                settings['batch_size'] = 8
                settings['quantization'] = None
            else:
                settings['batch_size'] = 4
                settings['quantization'] = '8bit'
                
        elif gpu_memory_gb >= 24:  # RTX 4090
            if model_size <= 7:
                settings['batch_size'] = 4
                settings['quantization'] = None
            elif model_size <= 14:
                settings['batch_size'] = 2
                settings['quantization'] = '8bit'
            else:
                settings['batch_size'] = 1
                settings['quantization'] = '4bit'
                
        elif gpu_memory_gb >= 16:  # V100
            if model_size <= 7:
                settings['batch_size'] = 2
                settings['quantization'] = '8bit'
            else:
                settings['batch_size'] = 1
                settings['quantization'] = '4bit'
                
        else:  # 작은 GPU
            settings['batch_size'] = 1
            settings['quantization'] = '4bit'
            settings['max_length'] = 1024
        
        # Gradient accumulation 계산
        target_batch = 16
        settings['gradient_accumulation'] = max(1, target_batch // settings['batch_size'])
        
        return settings


def test_fallback():
    """대체 로직 테스트"""
    print("="*60)
    print("🔍 모델 대체 로직 테스트")
    print("="*60)
    
    # GPU 메모리 확인
    if torch.cuda.is_available():
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f"\nGPU: {torch.cuda.get_device_name(0)} ({gpu_memory:.1f}GB)")
    else:
        gpu_memory = 0
        print("\nGPU를 사용할 수 없습니다.")
    
    # EXAONE 로드 시도 (실패 예상)
    print("\n1️⃣ EXAONE 로드 테스트...")
    try:
        model, tokenizer, actual_name = ModelLoader.load_model(
            "LGAI-EXAONE/EXAONE-3.0-7.8B-Instruct",
            quantization='4bit' if gpu_memory < 24 else None
        )
        print(f"✅ 최종 로드된 모델: {actual_name}")
        
        # 권장 설정
        settings = ModelLoader.get_recommended_settings(actual_name, gpu_memory)
        print(f"\n📊 권장 설정:")
        print(f"- 배치 크기: {settings['batch_size']}")
        print(f"- 최대 길이: {settings['max_length']}")
        print(f"- 양자화: {settings['quantization'] or 'None'}")
        print(f"- Gradient Accumulation: {settings['gradient_accumulation']}")
        
    except Exception as e:
        print(f"❌ 오류: {e}")


if __name__ == "__main__":
    test_fallback()