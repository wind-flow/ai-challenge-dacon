# 프롬프트 템플릿 가이드

## 📁 프롬프트 파일 설명

### 기본 프롬프트
- `default.txt`: 기본 템플릿
- `basic.txt`: 간단한 문제 생성
- `cot.txt`: Chain-of-Thought 추론 포함

### 고품질 프롬프트 (추천) ⭐
- `training_data.txt`: 구조화된 학습 데이터 생성
- `high_quality.txt`: 품질 기준 중심 생성
- `diverse_types.txt`: 다양한 문제 유형 생성

### 특화 프롬프트
- `financial.txt`: 금융 도메인 특화
- `validation.txt`: 검증용 문제 생성

## 🎯 사용 전략

### 1. 초기 데이터 생성
```python
# 다양한 유형으로 초기 데이터 확보
config = {
    'prompt_template': 'prompts/diverse_types.txt',
    'num_questions': 1000
}
```

### 2. 고품질 데이터 추가
```python
# 품질 중심으로 추가 생성
config = {
    'prompt_template': 'prompts/high_quality.txt',
    'num_questions': 500,
    'min_quality': 80  # 높은 품질 기준
}
```

### 3. 학습용 최종 데이터
```python
# 구조화된 형식으로 최종 생성
config = {
    'prompt_template': 'prompts/training_data.txt',
    'num_questions': 2000
}
```

## 💡 프롬프트 선택 기준

| 목적 | 추천 프롬프트 | 특징 |
|------|--------------|------|
| 초기 테스트 | `basic.txt` | 빠른 생성 |
| 다양성 확보 | `diverse_types.txt` | 5가지 유형 |
| 품질 우선 | `high_quality.txt` | 엄격한 기준 |
| 학습 데이터 | `training_data.txt` | 구조화된 출력 |
| 추론 과정 | `cot.txt` | 단계별 설명 |

## 🔧 커스터마이징

### 프롬프트 수정 시 체크리스트
- [ ] 명확한 출력 형식 지정 ([TAG] 사용)
- [ ] 구체적인 예시 포함
- [ ] 평가 기준 명시
- [ ] 메타데이터 요구
- [ ] 한국 금융 맥락 강조

### 효과적인 변수 사용
- `{concept}`: 핵심 개념
- `{context}`: RAG 검색 결과
- `{difficulty}`: 난이도 지정
- `{type}`: 문제 유형 지정

## 📊 성능 비교

| 프롬프트 | 품질점수 | 다양성 | 생성속도 |
|---------|---------|--------|----------|
| basic | 70 | 낮음 | 빠름 |
| cot | 75 | 중간 | 보통 |
| training_data | 85 | 높음 | 보통 |
| high_quality | 90 | 중간 | 느림 |
| diverse_types | 80 | 매우높음 | 느림 |

## ✅ 추천 워크플로우

1. **diverse_types.txt**로 다양한 유형 생성 (1000개)
2. **high_quality.txt**로 고품질 데이터 추가 (500개)  
3. **training_data.txt**로 최종 학습 데이터 생성 (1500개)
4. 총 3000개 고품질 데이터 확보

## 🚨 주의사항

- 프롬프트가 길수록 토큰 소비 증가
- 복잡한 프롬프트는 생성 속도 저하
- 너무 엄격한 기준은 생성 실패율 증가
- 정기적으로 생성 품질 모니터링 필요