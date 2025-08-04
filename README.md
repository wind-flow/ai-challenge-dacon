# FSKU 프로젝트

## 📂 프로젝트 구조

```
ai-dacon/
├── 📌 main.py                    # ⭐ 메인 실행 파일 (여기서 시작!)
│
├── src/                          # 핵심 코드
│   ├── generate_data/            # 데이터 생성
│   ├── training/                 # 모델 학습
│   ├── infer/                    # 추론
│   └── rag/                      # RAG
│
├── data/                         # 데이터
│   ├── external/                # 외부 문서 (필수!)
│   └── test.csv                 # 테스트 (학습 금지!)
│
├── prompts/                      # 프롬프트 템플릿
└── results/                      # 실행 결과
```

## 🚀 빠른 시작

```bash
# 1. 설치
pip install -r requirements.txt

# 2. 외부 데이터 준비
# data/external/ 폴더에 금융 문서 추가

# 3. 실행
python main.py
```

## 💡 사용법

### 대화형 메뉴
```bash
python main.py
```

### 직접 명령
```bash
python main.py generate   # 데이터 생성
python main.py train      # 모델 학습
python main.py infer <model> <test>  # 추론
python main.py pipeline   # 전체 실행
```

## 📊 설정

### 데이터 생성
```python
config = {
    'model_name': 'beomi/SOLAR-10.7B-v1.0',
    'use_rag': True,
    'num_questions': 1000,
    'min_quality': 70
}
```

### 모델 학습
```python
config = {
    'base_model': 'beomi/SOLAR-10.7B-v1.0',
    'use_qlora': True,
    'lora_r': 16,
    'num_epochs': 3,
    'batch_size': 4
}
```

## 🎯 대회 규칙

### ✅ 허용
- 2025.07.31 이전 공개 데이터
- 로컬 모델
- RAG
- LoRA/QLoRA

### ❌ 금지
- test.csv 학습
- 복수 LLM
- 외부 API
- 수기 작성

## 🔧 최적화

### RTX 4090 (24GB)
```python
use_qlora=True         # 4bit 양자화
batch_size=2-4         # 작은 배치
gradient_accumulation=4-8  # 그래디언트 누적
```

## 📁 핵심 파일

- `main.py`: 통합 실행
- `src/generate_data/`: 데이터 생성
  - `concept_extractor.py`: 외부 데이터 개념 추출
  - `quality_checker.py`: 품질 검증
- `src/training/train.py`: 학습
- `src/infer/inference.py`: 추론
- `src/rag/retriever.py`: RAG

## ❓ FAQ

**Q: 모델 다운로드?**
- 첫 실행 시 자동 (10-20GB)

**Q: 메모리 부족?**
- `use_qlora=True`, `batch_size=2`

**Q: 외부 데이터?**
- `data/external/` 필수
- sample_finance_terms.txt 제공

---
**2025 금융 AI Challenge Track1**