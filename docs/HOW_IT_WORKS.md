# 📚 FSKU 프로젝트 동작 원리

## 🎯 핵심: 외부 데이터 기반 자동 생성

**대회 규칙 준수**: 수기 작성 ❌ → 외부 데이터 기반 ✅

## 🔄 전체 파이프라인

```
1. 외부 데이터 로드 → 2. 개념 추출 → 3. RAG 검색 
→ 4. LLM 생성 → 5. 품질 평가 → 6. 모델 학습 → 7. 추론
```

## 📁 프로젝트 구조

```
ai-dacon/
├── main.py                      # 진입점
└── src/
    ├── generate_data/           # 데이터 생성
    │   ├── main.py             # 생성 엔진
    │   ├── concept_extractor.py # 개념 추출
    │   └── quality_checker.py  # 품질 검증
    ├── training/                # 학습
    │   └── train.py            # LoRA/QLoRA
    ├── infer/                   # 추론
    │   └── inference.py        # 모델 추론
    └── rag/                     # RAG
        └── retriever.py        # 문서 검색
```

## 🚀 각 모듈 동작

### 1. 데이터 생성 (`src/generate_data/`)

#### ConceptExtractor
```python
# 외부 데이터에서 금융 개념 자동 추출
1. data/external/ 폴더의 문서 로드
2. 정규식 패턴으로 금융 용어 추출
3. 빈도수 기반 가중치 계산
```

#### DocumentRetriever (RAG)
```python
# 관련 문서 검색
1. 역인덱스 구축
2. 쿼리와 문서 매칭
3. 관련 컨텍스트 추출
```

#### DataGenerator
```python
# LLM으로 문제 생성
1. 개념 선택 (빈도 기반)
2. RAG로 컨텍스트 검색
3. 프롬프트 구성
4. LLM 생성
5. 품질 평가
```

### 2. 모델 학습 (`src/training/`)

```python
# QLoRA 학습
1. 4bit 양자화 설정
2. LoRA 어댑터 추가
3. 학습 실행
4. 모델 저장
```

### 3. 추론 (`src/infer/`)

```python
# 효율적 추론
1. 모델 로드
2. 배치 처리
3. 결과 저장
```

## 💡 핵심 기법

### 외부 데이터 기반 추출
```python
# ❌ 이전 (규칙 위반)
concepts = ["금리", "환율"]  # 수기 작성

# ✅ 현재 (규칙 준수)
extractor.extract_from_external_data()
# → 자동으로 "기준금리", "콜금리" 등 추출
```

### QLoRA 메모리 최적화
```python
BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16
)
# 24GB → 8GB로 메모리 사용량 감소
```

### 배치 추론
```python
# 515문항을 배치로 처리
batch_size = 8
for batch in batches:
    results = model.generate(batch)
```

## 🎮 실행 방법

### 1. 외부 데이터 준비 (필수!)
```
data/external/
├── 금융_용어.txt
├── 규제_가이드.txt
└── ...
```

### 2. 실행
```bash
python main.py
```

### 3. 파이프라인
```
데이터 생성 → 학습 → 추론
```

## 📊 성능 지표

| 단계 | 시간 | 메모리 |
|-----|------|--------|
| 데이터 생성 | 3초/문제 | 8GB |
| 학습 | 2시간/epoch | 16GB |
| 추론 | 0.5초/문제 | 8GB |

## ⚠️ 주의사항

1. **외부 데이터 필수**: data/external/ 비어있으면 동작 안 함
2. **test.csv 금지**: 학습에 사용 시 실격
3. **GPU 필수**: CPU로는 너무 느림

## 🔍 디버깅

```bash
# 개념 추출 테스트
python -c "from src.generate_data.concept_extractor import ConceptExtractor; e=ConceptExtractor(); print(e.extract_concepts())"

# GPU 확인
nvidia-smi
```

---
**최종 업데이트**: 2025-01-04