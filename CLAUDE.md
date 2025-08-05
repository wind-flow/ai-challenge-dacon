# FSKU 프로젝트 가이드 (Claude Code용)

## 🎯 프로젝트 개요
- **대회명**: 2025 금융 AI Challenge Track1
- **목표**: FSKU 평가 문항에 정확히 응답하는 단일 LLM 개발
- **핵심 제약**: 
  - 단일 LLM만 사용
  - 오프라인 환경
  - RTX 4090 24GB
  - 270분 내 515문항 처리

## 📂 현재 프로젝트 구조 (정리 완료!)

```
ai-dacon/
├── 📌 main.py                    # ⭐ 메인 실행 파일
│
├── src/                          # 핵심 코드 (기능별 정리)
│   ├── generate_data/            # 데이터 생성
│   │   ├── main.py              # 생성 메인
│   │   ├── concept_extractor.py # 외부 데이터 기반 추출
│   │   └── quality_checker.py   # 품질 검증
│   │
│   ├── training/                 # 모델 학습
│   │   └── train.py             # LoRA/QLoRA 학습
│   │
│   ├── infer/                    # 추론
│   │   └── inference.py         # 모델 추론
│   │
│   └── rag/                      # RAG 시스템
│       ├── retriever.py         # 문서 검색 (캐싱 지원)
│       ├── pdf_loader.py        # PDF/Excel 문서 로더
│       └── chunker.py           # 문서 청킹
│
├── data/                         # 데이터
│   ├── external/                # 외부 문서 (필수!)
│   ├── augmented/               # 생성된 데이터
│   └── test.csv                 # 테스트 (학습 금지!)
│
├── prompts/                      # 프롬프트 템플릿
└── results/                      # 실행 결과
```

## 📋 핵심 규칙

### ✅ 허용
- 2025.07.31 이전 공개 오픈소스
- 로컬 실행 가능 모델
- RAG (재구성 필수)
- 데이터 증강 (외부 데이터 기반)
- LoRA/QLoRA

### ❌ 금지
- test.csv 학습/검증 사용
- 복수 LLM 앙상블
- 외부 API (OpenAI 등)
- **수기 작성 데이터**
- 상업적 라이선스

## 🚀 Quick Start

```bash
# 1. 설치
pip install -r requirements.txt

# 2. 외부 데이터 준비 (필수!)
# data/external/ 폴더에 금융 문서 추가

# 3. 실행
python main.py
```

## 💡 핵심 워크플로우

### 1️⃣ 데이터 생성 (`src/generate_data/`)
```python
# 외부 데이터 기반 자동 추출 (수기 작성 X)
1. ConceptExtractor: PDF/Excel 문서에서 개념 추출
2. DocumentRetriever: RAG로 관련 문서 검색 (캐싱 지원)
3. DataGenerator: LLM으로 문제 생성
4. QualityChecker: 품질 평가 (70점 이상만)
```

#### 🚀 RAG 캐싱 시스템
- **첫 실행**: PDF 문서 처리 → 인덱스 생성 (약 46초)
- **이후 실행**: 캐시 자동 로드 (0.02초) - **2,300배 빠름!**
- **인덱스 위치**: `data/vectordb/index.pkl`
- **재구축 필요시**: `python rebuild_index.py`

### 2️⃣ 모델 학습 (`src/training/`)
```python
# QLoRA로 메모리 효율적 학습
1. 데이터 로드 (생성된 데이터)
2. QLoRA 4bit 설정
3. LoRA 파인튜닝
4. 모델 저장
```

### 3️⃣ 추론 (`src/infer/`)
```python
# 270분 내 515문항 처리
1. 모델 로드
2. 배치 처리
3. 결과 저장
```

## 🔧 최적화 전략

### 메모리 (RTX 4090 24GB)
```python
# QLoRA 필수
use_qlora=True
batch_size=2-4
gradient_accumulation=4-8
```

### 속도 (270분 제한)
- 배치 처리 최대화
- 캐싱 활용
- vLLM 검토

## 📊 평가 지표
```
Score = 0.5 × 객관식 + 0.5 × 주관식
주관식 = 0.6 × 의미유사도 + 0.4 × 키워드재현율
```

## ⚠️ 주의사항

1. **test.csv 절대 학습 금지** - 실격 사유
2. **외부 데이터 필수** - data/external/ 확인
3. **수기 작성 금지** - 자동 추출만 사용
4. **단일 모델** - 앙상블 불가

## 🎮 주요 명령

```bash
# 데이터 생성
python main.py generate

# 모델 학습  
python main.py train

# 추론
python main.py infer <model> <test>

# 전체 파이프라인
python main.py pipeline
```

## 📁 핵심 파일 역할

- `main.py`: 통합 실행 인터페이스
- `src/generate_data/main.py`: 데이터 생성 엔진
- `src/generate_data/concept_extractor.py`: 외부 데이터 개념 추출
- `src/training/train.py`: LoRA/QLoRA 학습
- `src/infer/inference.py`: 모델 추론
- `src/rag/retriever.py`: RAG 문서 검색 (캐싱 지원)

## 🏆 추천 모델

1. **upstage/SOLAR-10.7B-v1.0** (추천) - 한국어 최적화
2. **LG-AI-EXAONE/EXAONE-3.0-7.8B-Instruct** - LG AI 연구소
3. **Qwen/Qwen2.5-7B-Instruct** - 다국어 성능 우수
4. beomi/llama-2-ko-7b - 한국어 파인튜닝

## 📝 개발 원칙 (중요!)

### 파일 관리 원칙
1. **레거시 파일 제거**: 불필요한 파일은 즉시 삭제
2. **기존 파일 우선 수정**: 
   - 새 파일 생성 ❌ → 기존 파일 수정 ✅
   - 기능 추가시 관련 파일에 통합
   - 파일 수는 최소한으로 유지 (현재 7개)
3. **명확한 파일 구조**:
   - 기능별 폴더 구분 유지
   - 중복 코드 제거

### 코드 작성 규칙
```python
# ✅ 좋은 예시 - 상세한 한글 주석
def extract_concepts(self, text: str) -> List[str]:
    """
    외부 데이터에서 금융 개념 추출
    
    Args:
        text: 분석할 텍스트
    
    Returns:
        추출된 개념 리스트
    """
    # 1. 금융 용어 패턴 매칭
    patterns = [...]
    
    # 2. 빈도수 기반 가중치 계산
    weights = self._calculate_weights(...)
    
    # 3. 상위 N개 개념 선택
    return top_concepts

# ❌ 나쁜 예시 - 주석 없음
def extract_concepts(self, text: str) -> List[str]:
    patterns = [...]
    weights = self._calculate_weights(...)
    return top_concepts
```

### 주석 작성 규칙
1. **모든 함수에 docstring 필수**
2. **복잡한 로직은 단계별 주석**
3. **변수명이 불명확하면 주석 추가**
4. **한글 주석 사용** (이해도 향상)
5. **TODO, FIXME 명시적 표시**

### 언어 사용 규칙
1. **모든 출력 메시지 한글 사용**
   - print문, 로그, 에러 메시지 모두 한글
   - 진행 상황, 결과 보고 한글
2. **변수명/함수명은 영어** (Python 관례)
3. **주석과 문서는 한글**

### 코드 수정시 체크리스트
- [ ] 레거시 코드 제거했는가?
- [ ] 기존 파일에 통합 가능한가?
- [ ] 충분한 주석을 달았는가?
- [ ] 중복 코드는 없는가?
- [ ] 파일 수가 증가하지 않았는가?

## 📅 마일스톤

- [x] 프로젝트 구조 정리
- [ ] 외부 데이터 수집
- [ ] 데이터 증강 (5000개+)
- [ ] 모델 학습
- [ ] 추론 최적화 (270분)
- [ ] 최종 제출

### 출력 예시
```python
# ✅ 좋은 예시
print("데이터 생성 중...")
print(f"✅ 완료! 총 {count}개 생성됨")
print(f"⚠️ 경고: 메모리 부족 (현재: {mem}GB)")

# ❌ 나쁜 예시  
print("Loading data...")
print(f"Done! Generated {count} items")
```

---
**Last Updated**: 2025-01-04
**Version**: 5.1 (개발 원칙 추가)