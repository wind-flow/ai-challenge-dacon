# 모델 선정 및 RAG 구축 가이드

## 🔍 RTX 4090 24GB에서 실행 가능한 한국어 금융/법률 모델

### 파라미터 수용 한계
- **4bit 양자화**: 최대 ~65B 파라미터
- **8bit 양자화**: 최대 ~30B 파라미터
- **16bit (FP16)**: 최대 ~13B 파라미터

## ✅ 대회 사용 가능 모델 (오픈소스 + 한국어 특화)

### 1. **LG EXAONE 3.0** ⭐ 최우선 추천
- **공개 여부**: ✅ 오픈소스 (2024년 8월 공개)
- **라이선스**: 비상업적 연구용 Apache 2.0
- **모델 크기**: 7.8B (RTX 4090에서 충분히 실행 가능)
- **한국어 특화**:
  - 8조 토큰 중 한국어 데이터 60% 이상
  - 한국 금융감독원, 금융위원회 문서 포함
  - 한국 법령, 판례 데이터 학습
- **성능**: KMMLU 64.9%, Ko-CommonGen v2 73.0%
- **Hugging Face**: `LG-AI-EXAONE/EXAONE-3.0-7.8B-Instruct`

### 2. **beomi/SOLAR-10.7B** ⭐ 강력 추천
- **공개 여부**: ✅ 완전 오픈소스
- **라이선스**: Apache 2.0
- **모델 크기**: 10.7B (4bit 양자화로 RTX 4090 실행 가능)
- **한국어 특화**:
  - Upstage에서 한국어 데이터로 추가 학습
  - 한국어 벤치마크 최상위권
  - 금융/법률 용어 이해도 높음
- **성능**: 한국어 태스크에서 GPT-3.5 수준

### 3. **maywell/EXAONE-3.0-7.8B-Instruct-Llamafied**
- **공개 여부**: ✅ 커뮤니티 변환 버전
- **모델 크기**: 7.8B
- **특징**: EXAONE을 Llama 형식으로 변환 (더 많은 도구 지원)

### 4. **Qwen2.5 시리즈** (한국어 지원)
- **공개 여부**: ✅ 오픈소스
- **라이선스**: Apache 2.0
- **추천 모델**:
  - `Qwen/Qwen2.5-14B-Instruct` (4bit 양자화 필요)
  - `Qwen/Qwen2.5-7B-Instruct` (여유있게 실행)
- **한국어 성능**: 우수 (다국어 학습)

## ❌ 사용 불가 모델 (비공개/상업용)
1. **HyperCLOVA X**: 네이버 독점, API만 제공
2. **KoGPT**: 카카오 상업용, 일부 구버전만 공개
3. **BloombergGPT**: 완전 비공개
4. **Claude/GPT 계열**: API 사용 금지

## 📊 증강용 vs 추론용 모델 전략

### 증강용 (품질 우선)
```python
# 로컬 GPU에서 가능한 최대 모델 사용
augmentation_model = "LG-AI-EXAONE/EXAONE-3.0-7.8B-Instruct"
# 또는
augmentation_model = "beomi/SOLAR-10.7B-v1.0"
```

### 추론용 (속도 우선)
```python
# 4bit 양자화로 빠른 추론
from transformers import AutoModelForCausalLM, BitsAndBytesConfig

quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4"
)

model = AutoModelForCausalLM.from_pretrained(
    "LG-AI-EXAONE/EXAONE-3.0-7.8B-Instruct",
    quantization_config=quantization_config,
    device_map="auto"
)
```

## 🎯 금융/법률 특화를 위한 한국어 RAG 구성

### 한국어 특화 임베딩 모델
```python
# 1. 한국어 금융 특화 (추천)
"upskyy/bge-m3-korean"  # 한국어 최적화
"snunlp/KR-SBERT-V40K-klueNLI-augSTS"  # 서울대 개발

# 2. 다국어 (한국어 포함)
"BAAI/bge-m3"  # 강력한 한국어 지원
"intfloat/multilingual-e5-large"  # 한국어 성능 우수
```

### 한국 금융 데이터로 파인튜닝
```python
# EXAONE으로 금융 데이터 파인튜닝 예시
from peft import LoraConfig, get_peft_model

lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
    lora_dropout=0.1,
    task_type="CAUSAL_LM"
)

# 수집한 한국 금융 데이터로 학습
model = get_peft_model(base_model, lora_config)
```

## 💡 실전 구현 권장사항
1. **증강 단계**: EXAONE 3.0 또는 SOLAR-10.7B 사용
2. **추론 단계**: 동일 모델을 4bit 양자화하여 사용
3. **한국어 RAG**: BGE-M3 한국어 버전 + Weaviate
4. **메모리 관리**: gradient checkpointing 활용

## 🚀 RAG 시스템 구성 세부사항

### 핵심 컴포넌트
- **벡터DB**: Weaviate (한국어 하이브리드 검색)
- **임베딩**: upskyy/bge-m3-korean 또는 BAAI/bge-m3
- **청킹**: 512-1024 토큰 (문서 유형별 차별화)
- **검색**: 하이브리드(의미+키워드) + 리랭킹

### 구현 코드
```python
# 하이브리드 검색 설정
class HybridRetriever:
    def __init__(self, alpha=0.7):  # 의미:키워드 = 7:3
        self.vectorstore = Weaviate(...)
        self.bm25 = BM25Okapi(...)
    
    def retrieve(self, query, k=10):
        # 1차: 하이브리드 검색 (상위 25개)
        candidates = self._hybrid_search(query, 25)
        # 2차: 리랭킹 (최종 3개)
        return self._rerank(query, candidates, 3)
```

## 📈 증강 전략 세부사항

### 단계별 접근
- **증강용**: EXAONE 3.0 또는 SOLAR-10.7B (품질 우선)
- **추론용**: 동일 모델 4bit 양자화 (속도 우선)

### 메모리 최적화
- Gradient checkpointing 사용
- 배치 크기 동적 조정
- 불필요한 텐서 즉시 삭제

## ⚠️ 주의사항
- ❌ API 모델 사용 금지 (OpenAI, Claude 등)
- ❌ 2025.07.31 이후 공개 모델 사용 불가
- ✅ 270분 내 515문항 처리 필수
- ✅ 오프라인 환경 실행 가능