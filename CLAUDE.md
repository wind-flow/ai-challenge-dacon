# 2025 금융 AI Challenge Track1 - 프로젝트 매니저

## 🎯 프로젝트 개요
- **대회명**: 2025 금융 AI Challenge Track1 - 금융보안 특화 AI 모델 경쟁
- **목표**: FSKU 평가지표 문항에 대해 정확한 응답을 생성하는 단일 LLM 모델 개발
- **핵심 제약**: 단일 LLM, 오프라인 환경, RTX 4090 24GB, 270분 내 515문항 처리

## 📁 프로젝트 구조
```
ai-dacon/
├── data/
│   ├── test.csv (515문항) - 추론용만, 학습 금지 ⚠️
│   ├── sample_submission.csv - 제출 형식
│   └── external/ - 외부 수집 데이터
├── docs/ - 상세 문서
│   ├── rules/ - 대회 규칙
│   │   └── competition_rules.md
│   ├── data/ - 데이터 관련
│   │   └── external_data_sources.md
│   ├── models/ - 모델 선정
│   │   └── model_selection_guide.md
│   └── technical/ - 기술 가이드
│       └── optimization_guide.md
├── src/
│   ├── generate_questions.py - Ollama 기반 문제 생성
│   ├── train.py - 모델 학습 (예정)
│   └── inference.py - 추론 코드 (예정)
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

---
**Last Updated**: 2025-08-03
**Version**: 3.0 (문서 구조화)
**Author**: AI Assistant for DACON Competition