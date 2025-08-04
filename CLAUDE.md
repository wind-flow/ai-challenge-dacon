# 2025 금융 AI Challenge Track1 - 프로젝트 지침

## 🎯 프로젝트 개요
- **대회명**: 2025 금융 AI Challenge Track1 - 금융보안 특화 AI 모델 경쟁
- **목표**: FSKU 평가지표 문항에 대해 정확한 응답을 생성하는 단일 LLM 모델 개발
- **핵심 제약**: 단일 LLM, 오프라인 환경, RTX 4090 24GB, 270분 내 515문항 처리

## 📌 전체 응답 원칙
1. **규칙 준수 최우선**: 모든 제안과 구현은 대회 규칙을 엄격히 준수
2. **실용적 접근**: RTX 4090 24GB와 270분 제한을 항상 고려
3. **품질 > 양**: 데이터 증강 시 무작정 늘리기보다 고품질 데이터 생성
4. **문서화 철저**: 모든 결정사항과 코드는 재현 가능하도록 문서화

## 📁 프로젝트 구조
```
ai-dacon/
├── data/
│   ├── test.csv (515문항) - 추론용만, 학습 금지 ⚠️
│   ├── sample_submission.csv - 제출 형식
│   ├── external/ - 외부 수집 데이터
│   ├── augmented/ - 증강된 데이터
│   └── processed/ - 전처리된 데이터
├── docs/ - 상세 문서
│   ├── rules/ - 대회 규칙
│   │   └── competition_rules.md
│   ├── data/ - 데이터 관련
│   │   └── external_data_sources.md
│   ├── models/ - 모델 선정
│   │   └── model_selection_guide.md
│   ├── technical/ - 기술 가이드
│   │   └── optimization_guide.md
│   └── reports/ - 프로젝트 리포트
│       └── cot_integration_status.md
├── src/
│   ├── config/
│   │   ├── __init__.py
│   │   └── settings.py - 설정 관리
│   ├── data_augmentation/ - 데이터 증강 모듈
│   │   ├── __init__.py
│   │   ├── augmentation_pipeline.py - 통합 파이프라인
│   │   ├── cot_generator.py - CoT 생성기
│   │   ├── data_loader.py - 데이터 로더
│   │   ├── knowledge_extractor.py - 지식 추출
│   │   ├── quality_checker.py - 품질 검증
│   │   ├── question_generator.py - 문제 생성
│   │   └── reasoning_templates.py - 추론 템플릿
│   ├── run_augmentation_with_cot.py - CoT 증강 실행 ⭐
│   └── utils/ - 유틸리티 함수
├── scripts/ - 실행 스크립트
│   ├── run_augmentation.py - 기본 증강 실행
│   └── validate_data.py - 데이터 검증
├── notebooks/ - 실험 노트북
├── tests/ - 테스트 코드
├── requirements.txt - 의존성 패키지
├── README.md - 프로젝트 소개
└── CLAUDE.md (현재 파일)
```

## 📋 핵심 규칙 요약

### ✅ 허용
- 2025.07.31 이전 공개된 오픈소스 모델/데이터
- 로컬 실행 가능한 모델
- RAG (생성 모델의 요약/재구성 필수)
- 데이터 증강 (로컬 모델 활용)
- LoRA/QLoRA 파인튜닝

### ❌ 금지
- test.csv 학습/검증 사용
- 복수 LLM 앙상블
- 외부 API (OpenAI, Gemini 등)
- 직접 수집 데이터 (크롤링, 수기작성)
- 상업적 라이선스

## 🚀 Quick Start

### 1. 환경 설정
```bash
# Python 3.10, CUDA 11.8, PyTorch 2.1.0
pip install -r requirements.txt
```

### 2. 모델 선정
- **최우선**: LG EXAONE 3.0 (7.8B)
- **대안**: beomi/SOLAR-10.7B
- 상세 내용: [모델 선정 가이드](docs/models/model_selection_guide.md)

### 3. 데이터 수집
- 우선순위별 외부 데이터 목록: [데이터 소스](docs/data/external_data_sources.md)
- 라이선스 확인 필수

### 4. 최적화
- 4bit 양자화, vLLM, 배치 처리
- 상세 내용: [최적화 가이드](docs/technical/optimization_guide.md)

### 5. 데이터 증강 실행 (CoT 포함)
```bash
# Chain-of-Thought 추론을 포함한 데이터 증강
python src/run_augmentation_with_cot.py --num_variations 5
```

## 📊 평가 지표
```
Score = 0.5 × 객관식 정확도 + 0.5 × 주관식 점수
주관식 = 0.6 × 의미 유사도 + 0.4 × 키워드 재현율
```

## 📅 마일스톤
- [ ] Week 1: 데이터 수집 및 라이선스 확인
- [ ] Week 2: 모델 학습 (LoRA/QLoRA)
- [ ] Week 3: 추론 최적화 (270분 내)
- [ ] Week 4: 최종 검증 및 제출

## 🔗 상세 문서
- [대회 규칙 전문](docs/rules/competition_rules.md)
- [외부 데이터 소스 목록](docs/data/external_data_sources.md)
- [모델 선정 가이드](docs/models/model_selection_guide.md)
- [최적화 가이드](docs/technical/optimization_guide.md)

## ⚠️ 중요 체크포인트
1. **RunPod 사용 가능 여부** - 공식 문의 필요
2. **test.csv는 절대 학습에 사용 금지**
3. **단일 LLM 모델로만 추론**
4. **270분 시간 제한 엄수**
5. **모든 외부 데이터 출처 증빙**

## 📝 작업별 세부 지침

### 1. 데이터 수집 및 증강 작업 시
- **test.csv 절대 사용 금지**: Data Leakage 규칙 위반 시 실격
- **외부 데이터 우선**: 공개된 금융 교육자료, 자격증 문제 등 활용
- **라이선스 철저 확인**: 2025.07.31 이전, 비상업적 라이선스만
- **금융 보안 도메인 특화**: 금융 보안에 관련된 용어와 맥락을 활용한 창의적 변형
- **다양성 확보**: 동일 개념에 대해 3-5개 변형 생성
- **변형 전략**:
  - 난이도 변경 (쉽게/어렵게)
  - 문제 유형 변경 (객관식↔주관식)
  - 상황 응용 (다른 금융 시나리오 적용)
  - 심화 문제 (원본 개념을 확장)
- **품질 검증**: 금융 개념의 정확성과 문제의 타당성 확인
- **메타데이터 기록**: 데이터 출처, 라이선스, 변형 방법 필수 기록

#### run_augmentation_with_cot.py 사용법
**목적**: Chain-of-Thought (CoT) 추론을 활용한 고품질 데이터 증강

**핵심 기능**:
1. **지식 추출**: 원본 데이터에서 핵심 개념과 관계 추출
2. **CoT 생성**: 단계별 추론 과정 자동 생성
3. **문제 변형**: 다양한 난이도와 유형으로 변형
4. **품질 검증**: 생성된 데이터의 정확성과 일관성 검증

**실행 명령**:
```bash
# 기본 실행 (5개 변형 생성)
python src/run_augmentation_with_cot.py --num_variations 5

# 고급 옵션
python src/run_augmentation_with_cot.py \
  --num_variations 3 \
  --difficulty_levels "easy,medium,hard" \
  --include_explanations \
  --validate_answers
```

**생성 데이터 구조**:
- 원본 문제 + CoT 추론 과정
- 변형 문제 (난이도별)
- 정답 및 상세 해설
- 메타데이터 (출처, 변형 방법, 검증 결과)

### 2. 모델 학습 작업 시
- **효율성 우선**: LoRA/QLoRA 활용하여 메모리 효율적 학습
- **체크포인트 관리**: 주기적 저장 및 최적 체크포인트 선택
- **학습 로그**: 손실값, 검증 지표 등 상세 기록
- **하이퍼파라미터**: 작은 값부터 시작하여 점진적 조정

### 3. 추론 최적화 작업 시
- **시간 측정**: 각 단계별 소요 시간 프로파일링
- **배치 처리**: 가능한 최대 배치 크기 활용
- **캐싱 전략**: 반복 계산 최소화
- **양자화 검토**: 4bit/8bit 양자화로 속도 향상
- **vLLM 고려**: 추론 속도 향상을 위한 도구 검토

### 4. 코드 작성 시
```python
# 모든 코드는 다음 구조를 따름:
# 1. 명확한 docstring
# 2. 타입 힌트 사용
# 3. 에러 처리 포함
# 4. 로깅 구현
# 5. 상대 경로 사용

def function_name(param: type) -> return_type:
    """
    함수 설명
    
    Args:
        param: 파라미터 설명
    
    Returns:
        반환값 설명
    """
    try:
        # 구현
        logging.info("작업 내용")
        return result
    except Exception as e:
        logging.error(f"에러 발생: {e}")
        raise
```

### 5. 문제 해결 접근법
1. **문제 발생 시**:
   - 먼저 대회 규칙 재확인
   - 유사 사례 검색
   - 단계별 디버깅
   - 대안 솔루션 준비

2. **불확실한 사항**:
   - RunPod 같은 애매한 케이스는 즉시 문의
   - 보수적 해석 적용 (허용 안 됨으로 가정)
   - Plan B 준비

### 6. 일일 작업 우선순위
1. **최우선**: 외부 금융 데이터 소스 발굴 및 라이선스 확인
2. **긴급**: RunPod 문의, 규칙 확인 사항
3. **중요**: 수집 데이터 기반 문제 생성
4. **필수**: 추론 시간 테스트
5. **추가**: 성능 최적화, 문서화

## 🚨 위험 관리
- **데이터 누수**: 테스트 데이터 절대 학습에 사용 금지
- **시간 초과**: 매 구현마다 추론 시간 체크
- **메모리 부족**: OOM 방지를 위한 배치 크기 조정
- **재현성**: 모든 랜덤 시드 고정

## 💡 전략적 조언
1. **경쟁자 대비 차별화**:
   - 금융 도메인 지식 최대한 활용
   - 프롬프트 엔지니어링으로 하드웨어 한계 극복
   - 데이터 품질로 승부

2. **시간 관리**:
   - 완벽보다는 작동하는 버전 먼저
   - 점진적 개선 approach
   - 제출 횟수 전략적 활용 (1일 3회)

3. **협업 시**:
   - 모든 공유는 DACON 플랫폼 내에서만
   - Private sharing 절대 금지
   - 코드 버전 관리 철저

## 📍 항상 기억할 것
- "단일 LLM 모델"이 핵심 제약
- 오프라인 환경에서 작동 필수
- 270분 내 515문항 처리
- 모든 외부 자원은 2025.07.31 이전 공개

---
**Last Updated**: 2025-08-04
**Version**: 4.1 (프로젝트 구조 및 CoT 증강 추가)
**Author**: AI Assistant for DACON Competition