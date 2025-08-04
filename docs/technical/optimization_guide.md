# 최적화 가이드

## 🚀 메모리 최적화 전략

### QLoRA 설정
```python
from transformers import BitsAndBytesConfig
import torch

quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4"
)
```

### Gradient Checkpointing
```python
model.gradient_checkpointing_enable()
model.config.use_cache = False  # 학습 시에만
```

### 메모리 효율적인 배치 처리
```python
def adaptive_batch_size(model, initial_batch=8):
    """GPU 메모리에 따라 배치 크기 자동 조정"""
    while initial_batch > 1:
        try:
            # 테스트 실행
            test_input = torch.randint(0, 32000, (initial_batch, 512))
            _ = model(test_input)
            torch.cuda.empty_cache()
            return initial_batch
        except torch.cuda.OutOfMemoryError:
            initial_batch //= 2
            torch.cuda.empty_cache()
    return 1
```

## ⏱️ 추론 속도 최적화

### 시간 관리 (270분 기준)
- 모델 로딩: ~5분
- 추론: ~260분 (약 30초/문항)
- 후처리 및 저장: ~5분

### vLLM 활용
```python
from vllm import LLM, SamplingParams

# vLLM으로 빠른 추론
llm = LLM(
    model="LG-AI-EXAONE/EXAONE-3.0-7.8B-Instruct",
    tensor_parallel_size=1,
    dtype="half",  # FP16
    max_model_len=2048
)

sampling_params = SamplingParams(
    temperature=0.7,
    top_p=0.9,
    max_tokens=512
)
```

### 배치 추론 최적화
```python
def batch_inference(model, questions, batch_size=4):
    """배치 단위로 추론하여 속도 향상"""
    results = []
    
    for i in range(0, len(questions), batch_size):
        batch = questions[i:i+batch_size]
        
        # 패딩 처리
        inputs = tokenizer(
            batch,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt"
        )
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=256,
                do_sample=False,  # 결정적 출력
                num_beams=1  # 빔 서치 비활성화로 속도 향상
            )
        
        results.extend(tokenizer.batch_decode(outputs, skip_special_tokens=True))
        
    return results
```

## 🔄 캐싱 전략

### KV 캐시 활용
```python
class CachedInference:
    def __init__(self, model):
        self.model = model
        self.cache = {}
    
    def get_embedding(self, text):
        if text not in self.cache:
            self.cache[text] = self.model.encode(text)
        return self.cache[text]
```

### 유사 질문 그룹화
```python
def group_similar_questions(questions):
    """유사한 질문을 그룹화하여 컨텍스트 재사용"""
    groups = {
        'multiple_choice_4': [],
        'multiple_choice_5': [],
        'subjective': []
    }
    
    for q in questions:
        if "1 " in q and "4 " in q:
            groups['multiple_choice_4'].append(q)
        elif "1 " in q and "5 " in q:
            groups['multiple_choice_5'].append(q)
        else:
            groups['subjective'].append(q)
    
    return groups
```

## 📊 프로파일링 및 모니터링

### 시간 측정
```python
import time
from contextlib import contextmanager

@contextmanager
def timer(name):
    start = time.time()
    yield
    end = time.time()
    print(f"{name}: {end - start:.2f}초")

# 사용 예시
with timer("모델 로딩"):
    model = load_model()

with timer("515문항 추론"):
    results = batch_inference(model, questions)
```

### GPU 메모리 모니터링
```python
def print_gpu_memory():
    """GPU 메모리 사용량 출력"""
    import torch
    if torch.cuda.is_available():
        print(f"GPU 메모리 사용: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
        print(f"GPU 메모리 예약: {torch.cuda.memory_reserved() / 1024**3:.2f} GB")
```

## 🎯 최종 체크리스트

### 추론 전 확인사항
- [ ] 모델 4bit 양자화 적용
- [ ] 배치 크기 최적화 (OOM 방지)
- [ ] 불필요한 로깅 제거
- [ ] torch.no_grad() 적용
- [ ] CUDA 캐시 정리

### 성능 목표
- [ ] 단일 문항 추론: < 30초
- [ ] 전체 515문항: < 260분
- [ ] GPU 메모리 사용: < 22GB
- [ ] 정확도 손실: < 2%

## 💡 추가 최적화 팁

1. **Flash Attention 2 사용**
```python
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    attn_implementation="flash_attention_2"
)
```

2. **Mixed Precision Training**
```python
from torch.cuda.amp import autocast

with autocast():
    outputs = model(**inputs)
```

3. **컴파일 최적화** (PyTorch 2.0+)
```python
import torch._dynamo
torch._dynamo.config.suppress_errors = True
model = torch.compile(model)
```