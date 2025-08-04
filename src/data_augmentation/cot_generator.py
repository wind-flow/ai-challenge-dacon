"""
Chain of Thought (CoT) 기반 문제 생성 모듈
단계별 추론을 통한 고품질 금융 문제 생성
"""

import json
import logging
from typing import List, Dict, Optional, Tuple, Any
from dataclasses import dataclass
from enum import Enum
import random

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

logger = logging.getLogger(__name__)


class ReasoningStep(Enum):
    """추론 단계 정의"""
    CONCEPT_ANALYSIS = "concept_analysis"  # 개념 분석
    CONTEXT_BUILDING = "context_building"  # 맥락 구성
    DIFFICULTY_ASSESSMENT = "difficulty_assessment"  # 난이도 평가
    DISTRACTOR_REASONING = "distractor_reasoning"  # 오답 추론
    QUESTION_FORMULATION = "question_formulation"  # 질문 구성
    ANSWER_GENERATION = "answer_generation"  # 답변 생성
    SELF_VERIFICATION = "self_verification"  # 자체 검증


@dataclass
class ReasoningChain:
    """추론 체인 데이터 클래스"""
    concept: str
    problem_type: str
    steps: List[Dict[str, Any]]
    final_output: Optional[Dict] = None
    confidence_score: float = 0.0
    
    def add_step(self, step_type: ReasoningStep, content: str, metadata: Dict = None):
        """추론 단계 추가"""
        self.steps.append({
            'type': step_type.value,
            'content': content,
            'metadata': metadata or {}
        })
    
    def get_reasoning_trace(self) -> str:
        """추론 과정을 문자열로 반환"""
        trace = []
        for i, step in enumerate(self.steps, 1):
            trace.append(f"Step {i} ({step['type']}): {step['content']}")
        return "\n".join(trace)


class CoTQuestionGenerator:
    """Chain of Thought 기반 문제 생성기"""
    
    def __init__(self, model_name: str = None, use_model: bool = True):
        """
        초기화
        
        Args:
            model_name: 사용할 모델 이름
            use_model: 실제 모델 사용 여부
        """
        self.model = None
        self.tokenizer = None
        self.use_model = use_model
        
        if use_model and model_name:
            try:
                self.model = AutoModelForCausalLM.from_pretrained(
                    model_name,
                    torch_dtype=torch.float16,
                    device_map="auto"
                )
                self.tokenizer = AutoTokenizer.from_pretrained(model_name)
                logger.info(f"CoT Generator initialized with model: {model_name}")
            except Exception as e:
                logger.warning(f"Failed to load model, using template mode: {e}")
                self.use_model = False
        
        # Few-shot 예제 로드
        self.few_shot_examples = self._load_few_shot_examples()
        
        # 추론 템플릿
        self.reasoning_templates = self._load_reasoning_templates()
    
    def _load_few_shot_examples(self) -> Dict[str, List[Dict]]:
        """Few-shot 학습용 예제 로드"""
        return {
            'multiple_choice': [
                {
                    'concept': '금리',
                    'reasoning_chain': [
                        "개념 분석: 금리는 자금을 빌려주는 대가로 받는 이자의 비율",
                        "맥락 구성: 중앙은행 기준금리, 시장금리, 대출금리 등 다양한 맥락 존재",
                        "난이도 평가: 중급 - 금리 인상의 경제적 영향 이해 필요",
                        "오답 추론: 금리와 물가, 환율, 주가의 관계에서 흔한 오해들",
                        "질문 구성: 금리 인상이 경제에 미치는 영향을 묻는 응용 문제"
                    ],
                    'question': "중앙은행이 기준금리를 인상할 때 나타나는 경제적 효과로 적절하지 않은 것은?",
                    'options': [
                        "시중 유동성이 감소한다",
                        "대출 이자 부담이 증가한다",
                        "물가 상승 압력이 증가한다",  # 정답 (실제로는 감소)
                        "예금 금리가 상승한다"
                    ],
                    'answer': 3,
                    'explanation': "금리 인상은 통화 긴축 정책으로 물가 상승 압력을 감소시킵니다."
                },
                {
                    'concept': '개인정보보호',
                    'reasoning_chain': [
                        "개념 분석: 개인정보는 특정 개인을 식별할 수 있는 정보",
                        "맥락 구성: 금융거래에서 수집되는 다양한 개인정보와 보호 의무",
                        "난이도 평가: 상급 - 법령의 구체적 조항과 실무 적용",
                        "오답 추론: 유사 법령 간 혼동, 보호조치 범위의 오해",
                        "질문 구성: 개인정보보호법상 안전조치 의무를 묻는 문제"
                    ],
                    'question': "개인정보보호법 제29조에 따른 안전성 확보조치에 포함되지 않는 것은?",
                    'options': [
                        "접근 권한의 관리",
                        "접속기록의 보관 및 점검",
                        "개인정보의 암호화",
                        "개인정보의 판매 금지"  # 정답 (안전조치가 아닌 별도 금지사항)
                    ],
                    'answer': 4,
                    'explanation': "개인정보 판매 금지는 안전조치가 아닌 별도의 금지 규정입니다."
                }
            ],
            'essay': [
                {
                    'concept': '재해복구체계',
                    'reasoning_chain': [
                        "개념 분석: 재해로 인한 시스템 중단 시 복구 체계",
                        "맥락 구성: 금융회사의 업무 연속성 계획(BCP)과 연계",
                        "난이도 평가: 상급 - 실무적 구현 방안 포함",
                        "답변 구조: 정의 → 구성요소 → 구축 기준 → 테스트",
                        "키워드 도출: RTO, RPO, 백업센터, 모의훈련"
                    ],
                    'question': "금융회사의 재해복구체계 구축 요건과 운영 방안을 설명하시오.",
                    'answer': "재해복구체계는 재해 발생 시 핵심 업무를 지속하기 위한 체계로, RTO(복구목표시간)와 RPO(복구목표시점)를 설정하고, 백업센터를 구축하며, 정기적인 모의훈련을 실시해야 합니다.",
                    'keywords': ["RTO", "RPO", "백업센터", "모의훈련", "BCP"]
                }
            ]
        }
    
    def _load_reasoning_templates(self) -> Dict[str, str]:
        """추론 템플릿 로드"""
        return {
            'concept_analysis': """
개념: {concept}

이 개념의 핵심 특징을 분석하세요:
1. 정의: 
2. 주요 구성요소:
3. 금융 실무에서의 중요성:
4. 관련 규제나 법령:

분석 결과:
""",
            'context_building': """
개념: {concept}
이전 분석: {previous_analysis}

이 개념이 활용되는 실무 맥락을 구성하세요:
1. 실제 적용 사례:
2. 관련 업무 프로세스:
3. 이해관계자:
4. 주요 리스크:

맥락 구성:
""",
            'difficulty_assessment': """
개념: {concept}
맥락: {context}

적절한 난이도를 평가하세요:
- 초급: 기본 정의와 개념 이해
- 중급: 응용과 적용 능력
- 고급: 심화 분석과 종합적 판단

난이도 선택 및 근거:
""",
            'distractor_reasoning': """
개념: {concept}
정답: {correct_answer}

효과적인 오답(distractor)을 추론하세요:
1. 유사 개념으로 혼동 가능한 것:
2. 부분적으로만 맞는 것:
3. 일반적인 오해나 착각:
4. 맥락상 그럴듯하지만 틀린 것:

오답 목록 및 선택 이유:
""",
            'self_verification': """
생성된 문제: {question}
답변: {answer}

다음 기준으로 검증하세요:
1. 금융 개념의 정확성: 
2. 문제의 명확성:
3. 난이도의 적절성:
4. 답변의 타당성:
5. FSKU 평가 기준 부합:

검증 결과 및 신뢰도 점수 (0-100):
"""
        }
    
    def generate_with_reasoning(self, 
                               concept: Dict, 
                               problem_type: str = "multiple_choice",
                               difficulty: str = "medium") -> Dict:
        """
        CoT 기반 문제 생성
        
        Args:
            concept: 개념 정보
            problem_type: 문제 유형
            difficulty: 난이도
            
        Returns:
            생성된 문제와 추론 과정
        """
        # 추론 체인 초기화
        chain = ReasoningChain(
            concept=concept.get('term', concept.get('context', '')),
            problem_type=problem_type,
            steps=[]
        )
        
        # Step 1: 개념 분석
        concept_analysis = self._analyze_concept(concept, chain)
        
        # Step 2: 맥락 구성
        context = self._build_context(concept, concept_analysis, chain)
        
        # Step 3: 난이도 평가
        assessed_difficulty = self._assess_difficulty(concept, context, difficulty, chain)
        
        # Step 4: 문제 생성
        if problem_type == "multiple_choice":
            question_data = self._generate_mc_with_reasoning(
                concept, context, assessed_difficulty, chain
            )
        else:
            question_data = self._generate_essay_with_reasoning(
                concept, context, assessed_difficulty, chain
            )
        
        # Step 5: 자체 검증
        confidence = self._self_verify(question_data, chain)
        
        # 최종 결과 구성
        chain.final_output = question_data
        chain.confidence_score = confidence
        
        return {
            'question_data': question_data,
            'reasoning_chain': chain.get_reasoning_trace(),
            'confidence_score': confidence,
            'metadata': {
                'concept': concept,
                'problem_type': problem_type,
                'difficulty': assessed_difficulty,
                'cot_steps': len(chain.steps)
            }
        }
    
    def _analyze_concept(self, concept: Dict, chain: ReasoningChain) -> str:
        """Step 1: 개념 분석"""
        concept_text = concept.get('term', concept.get('context', ''))
        
        if self.use_model and self.model:
            # 실제 모델 사용
            prompt = self.reasoning_templates['concept_analysis'].format(
                concept=concept_text
            )
            analysis = self._generate_with_model(prompt)
        else:
            # 템플릿 기반 분석
            analysis = f"""
            개념 '{concept_text}'은(는) 금융 분야의 핵심 개념입니다.
            주요 특징: {concept.get('definition', '금융 거래와 관련된 중요한 요소')}
            실무 중요성: 금융 거래의 기본이 되는 개념으로 정확한 이해가 필요합니다.
            """
        
        chain.add_step(ReasoningStep.CONCEPT_ANALYSIS, analysis, {'concept': concept})
        return analysis
    
    def _build_context(self, concept: Dict, previous_analysis: str, 
                      chain: ReasoningChain) -> str:
        """Step 2: 맥락 구성"""
        if self.use_model and self.model:
            prompt = self.reasoning_templates['context_building'].format(
                concept=concept.get('term', ''),
                previous_analysis=previous_analysis
            )
            context = self._generate_with_model(prompt)
        else:
            # 템플릿 기반 맥락 구성
            context = f"""
            실무 적용: 금융회사에서 {concept.get('term', '개념')}을(를) 활용하는 상황
            관련 프로세스: 심사, 평가, 모니터링 등의 업무에서 활용
            주요 리스크: 잘못된 이해로 인한 업무 오류 및 규제 위반 가능성
            """
        
        chain.add_step(ReasoningStep.CONTEXT_BUILDING, context)
        return context
    
    def _assess_difficulty(self, concept: Dict, context: str, 
                          target_difficulty: str, chain: ReasoningChain) -> str:
        """Step 3: 난이도 평가"""
        # 개념의 복잡도 분석
        complexity_factors = {
            'term_length': len(concept.get('term', '')),
            'has_legal': '법' in str(concept) or '규정' in str(concept),
            'has_numeric': any(c.isdigit() for c in str(concept)),
            'context_complexity': len(context.split())
        }
        
        # 난이도 결정
        if complexity_factors['has_legal'] and complexity_factors['has_numeric']:
            assessed = "high"
        elif complexity_factors['has_legal'] or complexity_factors['context_complexity'] > 50:
            assessed = "medium"
        else:
            assessed = "low"
        
        # 타겟 난이도와 조정
        if target_difficulty:
            assessed = target_difficulty
        
        reasoning = f"난이도 평가: {assessed} (법령: {complexity_factors['has_legal']}, 수치: {complexity_factors['has_numeric']})"
        chain.add_step(ReasoningStep.DIFFICULTY_ASSESSMENT, reasoning, {'difficulty': assessed})
        
        return assessed
    
    def _generate_mc_with_reasoning(self, concept: Dict, context: str, 
                                   difficulty: str, chain: ReasoningChain) -> Dict:
        """객관식 문제 생성 with CoT"""
        
        # Few-shot 예제 선택
        examples = self.few_shot_examples.get('multiple_choice', [])
        if examples:
            example = random.choice(examples)
            # 예제 기반 생성
            question_template = example['question']
            
            # 개념에 맞게 수정
            question = question_template.replace(
                example['concept'], 
                concept.get('term', '개념')
            )
        else:
            question = f"{concept.get('term', '개념')}에 대한 설명으로 옳은 것은?"
        
        # 정답 생성
        correct_answer = concept.get('definition', f"{concept.get('term', '')}의 올바른 설명")
        
        # 오답 추론
        distractors = self._reason_distractors(correct_answer, chain)
        
        # 선택지 구성
        options = [correct_answer] + distractors[:3]
        random.shuffle(options)
        answer_idx = options.index(correct_answer) + 1
        
        chain.add_step(
            ReasoningStep.QUESTION_FORMULATION, 
            f"질문: {question}\n선택지: {options}"
        )
        
        return {
            'type': 'multiple_choice_4',
            'question': question,
            'options': options,
            'answer': answer_idx,
            'explanation': f"정답은 '{correct_answer}'입니다."
        }
    
    def _generate_essay_with_reasoning(self, concept: Dict, context: str,
                                      difficulty: str, chain: ReasoningChain) -> Dict:
        """주관식 문제 생성 with CoT"""
        
        # 난이도별 질문 구성
        if difficulty == "high":
            question = f"{concept.get('term', '')}의 실무 적용 방안과 주의사항을 상세히 설명하시오."
        elif difficulty == "medium":
            question = f"{concept.get('term', '')}의 주요 특징과 금융 실무에서의 활용을 설명하시오."
        else:
            question = f"{concept.get('term', '')}(이)란 무엇인지 설명하시오."
        
        # 답변 생성
        answer = self._generate_essay_answer(concept, context, difficulty)
        
        # 키워드 추출
        keywords = self._extract_keywords_with_reasoning(answer, chain)
        
        chain.add_step(
            ReasoningStep.ANSWER_GENERATION,
            f"질문: {question}\n답변: {answer[:100]}..."
        )
        
        return {
            'type': 'essay',
            'question': question,
            'answer': answer,
            'keywords': keywords,
            'evaluation_criteria': self._generate_evaluation_criteria(keywords, difficulty)
        }
    
    def _reason_distractors(self, correct_answer: str, chain: ReasoningChain) -> List[str]:
        """오답 추론"""
        distractors = []
        
        # 전략 1: 유사하지만 다른 개념
        similar_wrong = correct_answer.replace("해야 한다", "할 수 있다")
        if similar_wrong != correct_answer:
            distractors.append(similar_wrong)
        
        # 전략 2: 부분적으로만 맞는 답
        if len(correct_answer) > 20:
            partial = correct_answer[:len(correct_answer)//2] + "는 경우가 있다"
            distractors.append(partial)
        
        # 전략 3: 반대 개념
        opposite = correct_answer.replace("증가", "감소").replace("상승", "하락")
        if opposite != correct_answer:
            distractors.append(opposite)
        
        # 전략 4: 일반적 오해
        misconception = "일반적으로 " + correct_answer + "라고 알려져 있으나 예외가 있다"
        distractors.append(misconception)
        
        chain.add_step(
            ReasoningStep.DISTRACTOR_REASONING,
            f"오답 생성 전략: 유사 개념, 부분 정답, 반대 개념, 일반적 오해"
        )
        
        return distractors[:4]  # 최대 4개 반환
    
    def _generate_essay_answer(self, concept: Dict, context: str, difficulty: str) -> str:
        """주관식 답변 생성"""
        base_answer = concept.get('definition', concept.get('context', ''))
        
        if difficulty == "high":
            answer = f"""
            {base_answer}
            
            실무 적용 시에는 다음과 같은 점들을 고려해야 합니다:
            1. 규제 준수: 관련 법령과 지침을 철저히 준수
            2. 리스크 관리: 잠재적 위험 요소 사전 식별 및 대응
            3. 프로세스 최적화: 효율적인 업무 프로세스 구축
            
            주의사항:
            - 정기적인 모니터링과 개선
            - 이해관계자와의 원활한 소통
            - 변화하는 규제 환경에 대한 지속적 업데이트
            """
        elif difficulty == "medium":
            answer = f"""
            {base_answer}
            
            주요 특징:
            - {concept.get('term', '')}의 핵심 요소
            - 금융 거래에서의 역할
            - 다른 개념과의 관계
            
            실무 활용:
            - 일상 업무에서의 적용
            - 의사결정 시 고려사항
            """
        else:
            answer = base_answer
        
        return answer.strip()
    
    def _extract_keywords_with_reasoning(self, text: str, chain: ReasoningChain) -> List[str]:
        """추론 과정을 거친 키워드 추출"""
        keywords = []
        
        # 금융 전문 용어
        financial_terms = ['금리', '대출', '예금', '투자', '리스크', '규제', '감독', 
                          '보안', '개인정보', '암호화', '인증', '모니터링']
        
        for term in financial_terms:
            if term in text:
                keywords.append(term)
        
        # 숫자/수치 정보
        import re
        numbers = re.findall(r'\d+[조원%일년개]', text)
        keywords.extend(numbers[:2])
        
        # 법령 관련
        laws = re.findall(r'[가-힣]+법|[가-힣]+규정', text)
        keywords.extend(laws[:2])
        
        chain.add_step(
            ReasoningStep.ANSWER_GENERATION,
            f"추출된 키워드: {', '.join(keywords[:5])}"
        )
        
        return keywords[:5]
    
    def _generate_evaluation_criteria(self, keywords: List[str], difficulty: str) -> str:
        """평가 기준 생성"""
        criteria = f"""
        평가 기준 (난이도: {difficulty}):
        
        1. 핵심 키워드 포함 (40%):
           - 필수 키워드: {', '.join(keywords[:3])}
           - 추가 키워드: {', '.join(keywords[3:])} 
        
        2. 논리적 구성 (30%):
           - 서론-본론-결론 구조
           - 일관된 논리 전개
        
        3. 금융 실무 적합성 (20%):
           - 실무 사례 제시
           - 현실적 적용 가능성
        
        4. 정확성 (10%):
           - 금융 용어의 정확한 사용
           - 법령/규정 인용의 정확성
        """
        
        return criteria.strip()
    
    def _self_verify(self, question_data: Dict, chain: ReasoningChain) -> float:
        """Step 5: 자체 검증"""
        confidence = 85.0  # 기본 신뢰도
        
        # 검증 항목
        checks = {
            'has_question': 'question' in question_data and len(question_data['question']) > 10,
            'has_answer': 'answer' in question_data or 'options' in question_data,
            'proper_type': question_data.get('type') in ['multiple_choice_4', 'multiple_choice_5', 'essay'],
            'reasonable_length': len(str(question_data.get('question', ''))) < 500
        }
        
        # 각 체크 항목에 대한 점수 조정
        for check, passed in checks.items():
            if passed:
                confidence += 2.5
            else:
                confidence -= 10
                logger.warning(f"Verification failed for: {check}")
        
        # 추론 단계 수에 따른 보너스
        if len(chain.steps) >= 4:
            confidence += 5
        
        # 신뢰도 범위 제한
        confidence = max(0, min(100, confidence))
        
        verification_result = f"""
        검증 완료:
        - 문제 구조 완성도: {'통과' if checks['has_question'] else '실패'}
        - 답변 완성도: {'통과' if checks['has_answer'] else '실패'}
        - 형식 적합성: {'통과' if checks['proper_type'] else '실패'}
        - 전체 신뢰도: {confidence:.1f}%
        """
        
        chain.add_step(ReasoningStep.SELF_VERIFICATION, verification_result, {'confidence': confidence})
        
        return confidence
    
    def _generate_with_model(self, prompt: str, max_length: int = 512) -> str:
        """실제 모델을 사용한 텍스트 생성"""
        if not self.model or not self.tokenizer:
            return "모델이 로드되지 않아 템플릿 기반 생성을 사용합니다."
        
        try:
            inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=max_length)
            
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=256,
                    temperature=0.7,
                    do_sample=True,
                    top_p=0.9
                )
            
            generated = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            # 프롬프트 제거
            if prompt in generated:
                generated = generated.replace(prompt, "").strip()
            
            return generated
            
        except Exception as e:
            logger.error(f"Model generation failed: {e}")
            return "생성 실패 - 템플릿 사용"
    
    def get_few_shot_prompt(self, problem_type: str, n_examples: int = 2) -> str:
        """Few-shot 프롬프트 생성"""
        examples = self.few_shot_examples.get(problem_type, [])[:n_examples]
        
        prompt = "다음 예제들을 참고하여 새로운 문제를 생성하세요:\n\n"
        
        for i, example in enumerate(examples, 1):
            prompt += f"예제 {i}:\n"
            prompt += f"개념: {example['concept']}\n"
            prompt += f"추론 과정:\n"
            for step in example['reasoning_chain']:
                prompt += f"  - {step}\n"
            prompt += f"생성된 문제: {example['question']}\n"
            if 'options' in example:
                prompt += f"선택지: {example['options']}\n"
                prompt += f"정답: {example['answer']}번\n"
            prompt += f"설명: {example.get('explanation', '')}\n\n"
        
        prompt += "이제 위 예제들을 참고하여 새로운 문제를 생성하세요.\n"
        
        return prompt


if __name__ == "__main__":
    # 테스트
    import sys
    from pathlib import Path
    sys.path.append(str(Path(__file__).parent.parent.parent))
    
    logging.basicConfig(level=logging.INFO)
    
    # CoT 생성기 초기화 (모델 없이 테스트)
    cot_generator = CoTQuestionGenerator(use_model=False)
    
    # 테스트 개념
    test_concept = {
        'type': 'definition',
        'term': '개인정보보호',
        'definition': '개인을 식별할 수 있는 정보를 보호하는 것',
        'context': '금융회사는 고객의 개인정보를 안전하게 보호해야 한다.'
    }
    
    print("=" * 60)
    print("Chain of Thought 기반 문제 생성 테스트")
    print("=" * 60)
    
    # 객관식 생성
    result = cot_generator.generate_with_reasoning(
        test_concept,
        problem_type="multiple_choice",
        difficulty="medium"
    )
    
    print("\n[생성된 객관식 문제]")
    print(f"질문: {result['question_data']['question']}")
    print("선택지:")
    for i, opt in enumerate(result['question_data']['options'], 1):
        print(f"  {i}. {opt}")
    print(f"정답: {result['question_data']['answer']}번")
    print(f"\n신뢰도: {result['confidence_score']:.1f}%")
    
    print("\n[추론 과정]")
    print(result['reasoning_chain'])
    
    # 주관식 생성
    print("\n" + "=" * 60)
    essay_result = cot_generator.generate_with_reasoning(
        test_concept,
        problem_type="essay",
        difficulty="high"
    )
    
    print("\n[생성된 주관식 문제]")
    print(f"질문: {essay_result['question_data']['question']}")
    print(f"답변: {essay_result['question_data']['answer'][:200]}...")
    print(f"키워드: {', '.join(essay_result['question_data']['keywords'])}")
    print(f"\n신뢰도: {essay_result['confidence_score']:.1f}%")