# 2025 금융 AI Challenge Track1 - 데이터 증강 시스템

## 📌 프로젝트 개요

2025 금융 AI Challenge Track1을 위한 FSKU(Financial Semantic Korean Understanding) 데이터 증강 시스템입니다.

### 주요 기능
- 외부 금융 데이터 수집 및 전처리
- 금융 지식 추출 및 구조화
- FSKU 형식 문제 자동 생성 (객관식/주관식)
- 품질 검증 및 중복 제거
- 대회 규칙 준수 검증

## 🚀 Quick Start

### 1. 환경 설정

```bash
# Python 3.10 권장
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 의존성 설치
pip install -r requirements.txt
```

### 2. 데이터 준비

```bash
# 외부 데이터를 data/external 폴더에 배치
mkdir -p data/external
# PDF, TXT, JSON 등 금융 관련 문서 추가
```

### 3. 데이터 증강 실행

```bash
# 기본 실행 (5000개 문제 생성)
python scripts/run_augmentation.py

# 옵션 지정
python scripts/run_augmentation.py \
    --target_count 5000 \
    --model_name "beomi/SOLAR-10.7B-v1.0" \
    --mc_ratio 0.7 \
    --output_dir ./data/augmented

# 테스트 실행 (10개만 생성)
python scripts/run_augmentation.py --dry_run
```

### 4. 데이터 검증

```bash
# 생성된 데이터 검증
python scripts/validate_data.py data/augmented/augmented_data_*.json \
    --output validation_report.json \
    --check_quality
```

## 📁 프로젝트 구조

```
ai-dacon/
├── src/
│   ├── data_augmentation/
│   │   ├── data_loader.py         # 데이터 로딩
│   │   ├── knowledge_extractor.py # 지식 추출
│   │   ├── question_generator.py  # 문제 생성
│   │   ├── quality_checker.py     # 품질 검증
│   │   └── augmentation_pipeline.py # 파이프라인
│   └── config/
│       └── settings.py            # 설정 관리
├── data/
│   ├── external/                  # 외부 데이터
│   ├── processed/                 # 전처리 데이터
│   └── augmented/                 # 생성된 데이터
├── scripts/
│   ├── run_augmentation.py        # 실행 스크립트
│   └── validate_data.py          # 검증 스크립트
├── docs/                          # 문서
├── logs/                          # 로그
└── requirements.txt
```

## ⚙️ 설정 옵션

### 명령줄 옵션

| 옵션 | 설명 | 기본값 |
|------|------|--------|
| `--target_count` | 생성할 문제 수 | 5000 |
| `--model_name` | 사용할 LLM 모델 | beomi/SOLAR-10.7B-v1.0 |
| `--mc_ratio` | 객관식 비율 | 0.7 |
| `--similarity_threshold` | 중복 판단 임계값 | 0.85 |
| `--batch_size` | 배치 크기 | 32 |
| `--use_quantization` | 4bit 양자화 사용 | True |

### 환경 변수

```bash
export FSKU_MODEL_NAME="LG-AI-EXAONE/EXAONE-3.0-7.8B-Instruct"
export FSKU_TARGET_COUNT=10000
export FSKU_USE_GPU=true
```

## 📊 출력 형식

### 객관식 문제
```json
{
    "id": "AUG_00001",
    "type": "multiple_choice_4",
    "question": "금리란 무엇인가?",
    "options": [
        "자금 대여의 대가",
        "원금",
        "투자 수익",
        "예금 잔액"
    ],
    "answer": 1,
    "source_document": "금융감독원_교육자료.pdf",
    "created_at": "2025-01-15T10:00:00Z"
}
```

### 주관식 문제
```json
{
    "id": "AUG_00002",
    "type": "essay",
    "question": "개인정보보호법의 주요 내용을 설명하시오.",
    "answer": "개인정보보호법은...",
    "keywords": ["개인정보", "보호", "처리"],
    "evaluation_criteria": "...",
    "created_at": "2025-01-15T10:00:00Z"
}
```

## ⚠️ 대회 규칙 준수

### ✅ 허용
- 2025.07.31 이전 공개된 오픈소스 모델/데이터
- 로컬 실행 가능한 모델
- RAG (생성 모델의 요약/재구성 필수)
- 데이터 증강 (로컬 모델 활용)

### ❌ 금지
- test.csv 학습/검증 사용
- 복수 LLM 앙상블
- 외부 API (OpenAI, Gemini 등)
- 직접 수집 데이터 (크롤링, 수기작성)
- 상업적 라이선스

## 🛠️ 트러블슈팅

### GPU 메모리 부족
```bash
# 양자화 사용
--use_quantization

# 배치 크기 감소
--batch_size 8
```

### 모델 로딩 실패
```bash
# 대체 모델 사용
--model_name "Qwen/Qwen2.5-7B-Instruct"
```

## 📝 라이선스

MIT License

## 🤝 기여

이슈 및 PR은 환영합니다!

## 📧 문의

DACON 대회 관련 문의는 공식 토론 게시판을 이용해주세요.