"""
Chain of Thought를 위한 추론 템플릿 모음
금융 도메인 특화 프롬프트 템플릿
"""

from typing import Dict, List, Optional
from dataclasses import dataclass


@dataclass
class PromptTemplate:
    """프롬프트 템플릿 데이터 클래스"""
    name: str
    template: str
    variables: List[str]
    description: str
    
    def format(self, **kwargs) -> str:
        """템플릿 포맷팅"""
        return self.template.format(**kwargs)


class ReasoningTemplates:
    """금융 도메인 특화 추론 템플릿"""
    
    def __init__(self):
        self.templates = self._initialize_templates()
        self.few_shot_examples = self._initialize_few_shot_examples()
    
    def _initialize_templates(self) -> Dict[str, PromptTemplate]:
        """템플릿 초기화"""
        return {
            # ===== 개념 분석 템플릿 =====
            'concept_analysis_detailed': PromptTemplate(
                name='concept_analysis_detailed',
                template="""당신은 한국 금융 전문가입니다. 다음 금융 개념을 상세히 분석해주세요.

개념: {concept}
분야: {domain}

다음 단계에 따라 분석하세요:

1. 핵심 정의 (Definition):
   - 학술적 정의:
   - 실무적 정의:
   - 법적 정의 (있다면):

2. 구성 요소 (Components):
   - 주요 구성요소:
   - 각 요소의 역할:

3. 금융 시스템 내 위치 (Position):
   - 관련 기관:
   - 관련 법규:
   - 시장에서의 역할:

4. 실무 적용 (Application):
   - 일반적 사용 사례:
   - 주의사항:
   - 모범 사례:

5. 위험 요소 (Risks):
   - 잠재적 위험:
   - 위험 완화 방안:

분석 결과를 한국어로 작성하세요:""",
                variables=['concept', 'domain'],
                description='금융 개념의 상세 분석'
            ),
            
            # ===== 문제 생성 템플릿 =====
            'mc_generation_cot': PromptTemplate(
                name='mc_generation_cot',
                template="""금융 교육 전문가로서 다음 개념에 대한 객관식 문제를 생성하세요.

개념: {concept}
설명: {description}
난이도: {difficulty}
문제 유형: {question_type}

Chain of Thought 추론 과정:

Step 1. 핵심 학습 목표 설정
- 이 문제를 통해 평가하고자 하는 것:
- 필요한 선행 지식:

Step 2. 질문 구성
- 주요 질문:
- 보조 정보 (필요시):

Step 3. 정답 도출
- 정답:
- 정답인 이유:

Step 4. 오답 설계 (Distractor Reasoning)
- 오답 1: [내용] - 선택 이유: [흔한 오해/유사 개념]
- 오답 2: [내용] - 선택 이유: [부분적 정답/불완전한 이해]
- 오답 3: [내용] - 선택 이유: [관련 없지만 그럴듯한 답]
{extra_distractor}

Step 5. 검증
- 문제의 명확성: 
- 정답의 유일성:
- 난이도 적절성:

최종 문제:
질문: 
선택지:
1) 
2) 
3) 
4) 
{extra_option}

정답: 
해설: """,
                variables=['concept', 'description', 'difficulty', 'question_type', 'extra_distractor', 'extra_option'],
                description='CoT 기반 객관식 문제 생성'
            ),
            
            # ===== 주관식 문제 생성 템플릿 =====
            'essay_generation_cot': PromptTemplate(
                name='essay_generation_cot',
                template="""금융 전문가 평가를 위한 주관식 문제를 생성하세요.

주제: {topic}
세부 개념: {concepts}
난이도: {difficulty}
평가 목적: {evaluation_purpose}

추론 과정:

1. 평가 목표 분석
   - 측정하려는 역량:
   - 필요한 지식 수준:
   - 기대되는 답변 깊이:

2. 문제 설계
   - 핵심 질문:
   - 하위 질문들 (있다면):
   - 답변 범위 제한:

3. 모범 답안 구조
   - 서론 (도입부):
   - 본론 (핵심 내용):
     * 첫 번째 논점:
     * 두 번째 논점:
     * 세 번째 논점:
   - 결론 (요약 및 시사점):

4. 평가 기준 설정
   - 필수 포함 요소:
   - 핵심 키워드:
   - 논리 전개:
   - 실무 적용성:

5. 채점 기준
   - 우수 (90-100점):
   - 양호 (70-89점):
   - 보통 (50-69점):
   - 미흡 (50점 미만):

최종 문제:
질문: 

모범 답안:

핵심 키워드: 

평가 기준:""",
                variables=['topic', 'concepts', 'difficulty', 'evaluation_purpose'],
                description='CoT 기반 주관식 문제 생성'
            ),
            
            # ===== 시나리오 기반 문제 템플릿 =====
            'scenario_based_problem': PromptTemplate(
                name='scenario_based_problem',
                template="""실무 시나리오 기반 문제를 생성하세요.

업무 상황: {scenario}
관련 규정: {regulations}
핵심 이슈: {key_issues}

시나리오 구성:

1. 배경 설정
   - 회사/기관: {company_type}
   - 상황 설명:
   - 등장 인물/부서:

2. 문제 상황
   - 발생한 이슈:
   - 제약 조건:
   - 의사결정 필요 사항:

3. 질문 구성
   주요 질문: 이 상황에서 {role}로서 어떻게 대응하시겠습니까?
   
   고려사항:
   - 법규 준수
   - 리스크 관리
   - 이해관계자 관리
   - 실행 가능성

4. 평가 포인트
   - 문제 인식 능력:
   - 해결 방안의 적절성:
   - 규정 이해도:
   - 실무 적용 능력:

최종 시나리오 문제:

답안 가이드:""",
                variables=['scenario', 'regulations', 'key_issues', 'company_type', 'role'],
                description='실무 시나리오 기반 문제'
            ),
            
            # ===== 오답 생성 전문 템플릿 =====
            'distractor_generation': PromptTemplate(
                name='distractor_generation',
                template="""효과적인 오답(distractor)을 생성하세요.

정답: {correct_answer}
개념: {concept}
문제 맥락: {context}

오답 생성 전략:

1. 유형 A - 개념 혼동형
   - 유사 개념: {similar_concepts}
   - 혼동 포인트:
   - 생성된 오답:

2. 유형 B - 부분 정답형
   - 정답의 일부만 포함:
   - 누락된 핵심 요소:
   - 생성된 오답:

3. 유형 C - 과도한 일반화형
   - 지나치게 넓은 범위:
   - 구체성 부족:
   - 생성된 오답:

4. 유형 D - 맥락 불일치형
   - 다른 상황에서는 맞지만:
   - 현재 맥락에서 틀린 이유:
   - 생성된 오답:

5. 유형 E - 순서/절차 오류형
   - 올바른 요소지만 잘못된 순서:
   - 생성된 오답:

검증:
- 각 오답이 그럴듯한가?
- 정답과 명확히 구분되는가?
- 학습자의 일반적 오해를 반영하는가?

최종 오답 목록:
1) 
2) 
3) 
4) """,
                variables=['correct_answer', 'concept', 'context', 'similar_concepts'],
                description='체계적인 오답 생성'
            ),
            
            # ===== 난이도 조정 템플릿 =====
            'difficulty_adjustment': PromptTemplate(
                name='difficulty_adjustment',
                template="""문제의 난이도를 조정하세요.

원본 문제: {original_question}
현재 난이도: {current_difficulty}
목표 난이도: {target_difficulty}

난이도 조정 방법:

{difficulty_up_section}

{difficulty_down_section}

조정된 문제:
질문: 
난이도 변경 사항:
- 변경 전: 
- 변경 후: 
- 조정 근거: """,
                variables=['original_question', 'current_difficulty', 'target_difficulty', 'difficulty_up_section', 'difficulty_down_section'],
                description='문제 난이도 조정'
            ),
            
            # ===== 검증 템플릿 =====
            'quality_verification': PromptTemplate(
                name='quality_verification',
                template="""생성된 문제의 품질을 검증하세요.

문제: {question}
답안: {answer}
문제 유형: {question_type}

검증 체크리스트:

1. 내용 정확성 (Content Accuracy)
   □ 금융 용어가 정확한가?
   □ 법규/규정 인용이 올바른가?
   □ 수치/통계가 최신인가?
   점수: /25

2. 문제 명확성 (Question Clarity)
   □ 질문이 명확한가?
   □ 중의적 해석 가능성이 없는가?
   □ 필요한 정보가 모두 제공되었는가?
   점수: /25

3. 답안 타당성 (Answer Validity)
   □ 정답이 유일한가?
   □ 답안이 질문에 정확히 대응하는가?
   □ 채점 기준이 명확한가?
   점수: /25

4. 난이도 적절성 (Difficulty Appropriateness)
   □ 목표 수준에 맞는가?
   □ 선행 지식 요구가 적절한가?
   □ 시간 내 해결 가능한가?
   점수: /25

총점: /100

개선 필요 사항:

최종 판정: [통과/수정필요/재생성필요]""",
                variables=['question', 'answer', 'question_type'],
                description='문제 품질 검증'
            )
        }
    
    def _initialize_few_shot_examples(self) -> Dict[str, List[Dict]]:
        """Few-shot 예제 초기화"""
        return {
            'financial_security': [
                {
                    'input': '개인정보보호와 정보보안의 차이',
                    'cot_process': """
                    Step 1: 개념 구분
                    - 개인정보보호: 개인을 식별할 수 있는 정보의 보호
                    - 정보보안: 모든 정보 자산의 기밀성, 무결성, 가용성 보호
                    
                    Step 2: 적용 범위 분석
                    - 개인정보보호: 개인정보보호법, 정보통신망법 등
                    - 정보보안: 전자금융거래법, 정보통신기반보호법 등
                    
                    Step 3: 문제 구성
                    두 개념의 차이를 명확히 구분할 수 있는 사례 중심 문제
                    """,
                    'output': {
                        'question': 'A은행에서 발생한 다음 사건 중 개인정보보호 위반이 아닌 정보보안 사고에 해당하는 것은?',
                        'options': [
                            '고객 동의 없이 마케팅 목적으로 고객 정보를 제3자에게 제공',
                            '직원이 고객 주민등록번호를 포함한 엑셀 파일을 이메일로 전송',
                            '해커의 DDoS 공격으로 인한 온라인 뱅킹 서비스 중단',
                            '퇴사 직원이 고객 데이터베이스를 무단 반출'
                        ],
                        'answer': 3,
                        'explanation': 'DDoS 공격은 서비스 가용성을 침해하는 정보보안 사고이며, 개인정보 침해와는 직접적 관련이 없습니다.'
                    }
                }
            ],
            'risk_management': [
                {
                    'input': '운영리스크와 신용리스크',
                    'cot_process': """
                    Step 1: 리스크 유형 정의
                    - 운영리스크: 내부 프로세스, 인력, 시스템 실패로 인한 손실
                    - 신용리스크: 거래상대방의 채무불이행으로 인한 손실
                    
                    Step 2: 실무 사례 도출
                    - 운영리스크: 시스템 오류, 직원 실수, 내부 부정
                    - 신용리스크: 대출 부실, 거래처 파산
                    
                    Step 3: 평가 방법 차별화
                    측정 방법과 관리 체계의 차이를 묻는 문제 구성
                    """,
                    'output': {
                        'question': '다음 중 Basel III 기준 운영리스크 측정 방법이 아닌 것은?',
                        'options': [
                            '기초지표법(Basic Indicator Approach)',
                            '표준방법(Standardized Approach)', 
                            '고급측정법(Advanced Measurement Approach)',
                            '내부등급법(Internal Ratings-Based Approach)'
                        ],
                        'answer': 4,
                        'explanation': '내부등급법(IRB)은 신용리스크 측정 방법입니다. 운영리스크는 BIA, SA, AMA를 사용합니다.'
                    }
                }
            ],
            'fintech': [
                {
                    'input': '오픈뱅킹과 마이데이터',
                    'cot_process': """
                    Step 1: 서비스 개념 정리
                    - 오픈뱅킹: 표준 API를 통한 금융 서비스 연계
                    - 마이데이터: 개인 데이터 통합 관리 및 활용
                    
                    Step 2: 법적 근거 확인
                    - 오픈뱅킹: 전자금융거래법
                    - 마이데이터: 신용정보법 (개정)
                    
                    Step 3: 차이점 도출
                    데이터 주권과 서비스 범위의 차이를 중심으로 문제 구성
                    """,
                    'output': {
                        'question': '마이데이터 사업과 오픈뱅킹의 주요 차이점을 설명하시오.',
                        'answer': """
                        마이데이터는 개인이 자신의 데이터에 대한 주권을 행사하여 다양한 금융회사에 흩어진 본인 정보를 
                        통합 조회하고 제3자 제공을 통해 맞춤형 서비스를 받는 것입니다. 
                        
                        반면 오픈뱅킹은 표준화된 API를 통해 하나의 앱에서 여러 은행의 계좌조회, 이체 등 
                        금융거래를 할 수 있게 하는 인프라 서비스입니다.
                        
                        주요 차이점:
                        1. 법적 근거: 마이데이터(신용정보법), 오픈뱅킹(전자금융거래법)
                        2. 데이터 범위: 마이데이터(全 금융 데이터), 오픈뱅킹(은행 거래 정보)
                        3. 서비스: 마이데이터(조회+분석+추천), 오픈뱅킹(조회+이체)
                        """,
                        'keywords': ['데이터 주권', '신용정보법', 'API', '통합조회', '맞춤형 서비스']
                    }
                }
            ]
        }
    
    def get_template(self, name: str) -> Optional[PromptTemplate]:
        """템플릿 반환"""
        return self.templates.get(name)
    
    def get_few_shot_examples(self, domain: str, n: int = 2) -> List[Dict]:
        """도메인별 Few-shot 예제 반환"""
        examples = self.few_shot_examples.get(domain, [])
        return examples[:n]
    
    def format_few_shot_prompt(self, domain: str, task: str, n_examples: int = 2) -> str:
        """Few-shot 프롬프트 생성"""
        examples = self.get_few_shot_examples(domain, n_examples)
        
        prompt = f"다음은 {domain} 분야의 문제 생성 예제입니다.\n\n"
        
        for i, example in enumerate(examples, 1):
            prompt += f"예제 {i}:\n"
            prompt += f"입력: {example['input']}\n"
            prompt += f"추론 과정:\n{example['cot_process']}\n"
            prompt += f"결과:\n"
            
            output = example['output']
            if isinstance(output, dict):
                if 'question' in output:
                    prompt += f"  문제: {output['question']}\n"
                if 'options' in output:
                    prompt += "  선택지:\n"
                    for j, opt in enumerate(output['options'], 1):
                        prompt += f"    {j}) {opt}\n"
                if 'answer' in output:
                    if isinstance(output['answer'], int):
                        prompt += f"  정답: {output['answer']}번\n"
                    else:
                        prompt += f"  답안: {output['answer'][:200]}...\n"
                if 'explanation' in output:
                    prompt += f"  설명: {output['explanation']}\n"
            prompt += "\n"
        
        prompt += f"\n이제 위 예제를 참고하여 {task}을(를) 수행하세요.\n"
        
        return prompt
    
    def get_difficulty_adjustment_prompt(self, current_diff: str, target_diff: str) -> Dict:
        """난이도 조정 프롬프트 생성"""
        difficulty_up = """난이도 상향 방법:
1. 복합 개념 추가
   - 여러 개념을 연결
   - 상호 관계 분석 요구
   
2. 응용 상황 제시
   - 실무 시나리오 추가
   - 예외 상황 포함
   
3. 심화 분석 요구
   - "왜"와 "어떻게" 질문
   - 비판적 사고 요구
   
4. 계산/추론 추가
   - 수치 계산 포함
   - 다단계 추론 필요"""
        
        difficulty_down = """난이도 하향 방법:
1. 개념 단순화
   - 기본 정의 중심
   - 핵심만 다루기
   
2. 명확한 힌트 제공
   - 맥락 정보 추가
   - 키워드 강조
   
3. 선택지 차별화
   - 오답을 명확히 구분
   - 관련성 낮은 오답
   
4. 직접적 질문
   - 단순 사실 확인
   - Yes/No 형태"""
        
        return {
            'difficulty_up_section': difficulty_up if target_diff > current_diff else "",
            'difficulty_down_section': difficulty_down if target_diff < current_diff else ""
        }
    
    def create_custom_template(self, name: str, template: str, 
                              variables: List[str], description: str) -> PromptTemplate:
        """커스텀 템플릿 생성"""
        custom = PromptTemplate(
            name=name,
            template=template,
            variables=variables,
            description=description
        )
        self.templates[name] = custom
        return custom


# 싱글톤 인스턴스
reasoning_templates = ReasoningTemplates()


if __name__ == "__main__":
    # 테스트
    templates = ReasoningTemplates()
    
    # 템플릿 테스트
    print("=" * 60)
    print("템플릿 목록:")
    for name, template in templates.templates.items():
        print(f"- {name}: {template.description}")
    
    # Few-shot 프롬프트 테스트
    print("\n" + "=" * 60)
    print("Few-shot 프롬프트 예시:")
    prompt = templates.format_few_shot_prompt(
        domain='financial_security',
        task='개인정보보호 관련 문제 생성',
        n_examples=1
    )
    print(prompt)
    
    # 템플릿 포맷팅 테스트
    print("\n" + "=" * 60)
    print("템플릿 포맷팅 테스트:")
    mc_template = templates.get_template('mc_generation_cot')
    if mc_template:
        formatted = mc_template.format(
            concept='금리',
            description='자금을 빌려주는 대가로 받는 이자의 비율',
            difficulty='중급',
            question_type='응용문제',
            extra_distractor='',
            extra_option=''
        )
        print(formatted[:500] + "...")