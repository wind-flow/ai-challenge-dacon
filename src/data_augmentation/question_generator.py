"""
문제 생성 모듈
FSKU 형식의 금융 문제를 생성
"""

import re
import random
import logging
from typing import List, Dict, Optional, Tuple
from datetime import datetime
import json

import torch
from transformers import (
    AutoModelForCausalLM, 
    AutoTokenizer,
    BitsAndBytesConfig
)

# CoT 관련 모듈 임포트
try:
    from .cot_generator import CoTQuestionGenerator, ReasoningChain
    from .reasoning_templates import reasoning_templates
    COT_AVAILABLE = True
except ImportError:
    COT_AVAILABLE = False
    logger.warning("CoT modules not available, using basic generation")

logger = logging.getLogger(__name__)


class FSKUQuestionGenerator:
    """FSKU 형식 문제 생성 클래스 (CoT 통합)"""
    
    def __init__(self, model_name: str = "beomi/SOLAR-10.7B-v1.0", 
                 use_quantization: bool = True,
                 use_cot: bool = True):
        """
        로컬 LLM 모델 초기화
        
        Args:
            model_name: 사용할 모델 이름
            use_quantization: 양자화 사용 여부
            use_cot: Chain of Thought 사용 여부
        """
        self.model_name = model_name
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.use_cot = use_cot and COT_AVAILABLE
        
        # CoT 생성기 초기화
        if self.use_cot:
            self.cot_generator = CoTQuestionGenerator(
                model_name=model_name if model_name != "dummy" else None,
                use_model=model_name != "dummy"
            )
            self.reasoning_templates = reasoning_templates
            logger.info("CoT generation enabled")
        
        # 모델과 토크나이저 로드
        try:
            if use_quantization and self.device == "cuda":
                # 4bit 양자화 설정
                quantization_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.float16,
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_quant_type="nf4"
                )
                
                self.model = AutoModelForCausalLM.from_pretrained(
                    model_name,
                    quantization_config=quantization_config,
                    device_map="auto",
                    trust_remote_code=True
                )
            else:
                self.model = AutoModelForCausalLM.from_pretrained(
                    model_name,
                    torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                    device_map="auto" if self.device == "cuda" else None,
                    trust_remote_code=True
                )
                
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_name,
                trust_remote_code=True
            )
            
            # 패딩 토큰 설정
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
                
            logger.info(f"Successfully loaded model: {model_name}")
            
        except Exception as e:
            logger.warning(f"Failed to load model {model_name}: {e}")
            logger.info("Using template-based generation as fallback")
            self.model = None
            self.tokenizer = None
        
        # 문제 생성 템플릿
        self.question_templates = self._load_question_templates()
        
        # 오답 생성용 변형 규칙
        self.distractor_rules = self._load_distractor_rules()
    
    def _load_question_templates(self) -> Dict:
        """
        문제 생성 템플릿 로드
        
        Returns:
            템플릿 딕셔너리
        """
        templates = {
            'definition': [
                "{term}(이)란 무엇인가?",
                "{term}의 정의는?",
                "다음 중 {term}에 대한 설명으로 옳은 것은?",
                "{term}의 의미로 가장 적절한 것은?"
            ],
            'legal_article': [
                "{law} {article}의 내용으로 옳은 것은?",
                "{law}에서 규정하는 {topic}은?",
                "다음 중 {law} {article}에 해당하는 것은?",
                "{law}상 {topic}의 기준은?"
            ],
            'numeric': [
                "{context}의 {type}은?",
                "{subject}의 {type} 기준은?",
                "다음 중 {context}에 해당하는 {type}은?",
                "{regulation}에서 정한 {subject}의 {type}은?"
            ],
            'process': [
                "{process} 절차의 순서로 옳은 것은?",
                "{process}의 단계별 진행 순서는?",
                "다음 중 {process} 과정에 포함되지 않는 것은?",
                "{process} 수행 시 필요한 요건은?"
            ],
            'comparison': [
                "{item1}와(과) {item2}의 차이점은?",
                "다음 중 {item1}의 특징으로 {item2}와 구별되는 것은?",
                "{item1}과 {item2}를 비교한 설명으로 옳은 것은?",
                "{category}에서 {item1}과 {item2}의 주요 차이는?"
            ]
        }
        
        return templates
    
    def _load_distractor_rules(self) -> Dict:
        """
        오답 생성 규칙 로드
        
        Returns:
            오답 생성 규칙 딕셔너리
        """
        rules = {
            'numeric': {
                'percentage': lambda x: [x*0.5, x*0.75, x*1.25, x*1.5, x*2],
                'amount': lambda x: [x*0.1, x*0.5, x*2, x*10],
                'period': lambda x: [x//2, x*2, x*3, x*12] if x < 12 else [x//12, x//2, x*2]
            },
            'term_confusion': {
                '예금': ['적금', '예탁금', '공탁금', '보증금'],
                '대출': ['융자', '차입', '여신', '투자'],
                '이자': ['이율', '금리', '수익률', '할인율'],
                '자본': ['자산', '자금', '자본금', '출자금']
            },
            'legal_similar': {
                '금융위원회': ['금융감독원', '한국은행', '예금보험공사'],
                '개인정보보호법': ['정보통신망법', '신용정보법', '전자금융거래법'],
                '안전조치': ['보호조치', '보안조치', '기술적조치', '관리적조치']
            }
        }
        
        return rules
    
    def generate_multiple_choice(self, 
                                concept: Dict, 
                                num_options: int = 4,
                                use_cot: Optional[bool] = None) -> Dict:
        """
        객관식 문제 생성
        
        Args:
            concept: 개념 정보
            num_options: 선택지 개수 (4 또는 5)
            use_cot: CoT 사용 여부 (None이면 기본 설정 사용)
            
        Returns:
            생성된 객관식 문제
        """
        # CoT 사용 여부 결정
        should_use_cot = use_cot if use_cot is not None else self.use_cot
        
        # CoT 기반 생성 (복잡한 개념이거나 명시적 요청시)
        if should_use_cot and self._should_use_cot_for_concept(concept):
            return self._generate_mc_with_cot(concept, num_options)
        
        # 기본 템플릿 기반 생성
        question_data = {
            'type': f'multiple_choice_{num_options}',
            'question': '',
            'options': [],
            'answer': 1,  # 1-based index
            'explanation': '',
            'source_concept': concept
        }
        
        # 개념 유형별 문제 생성
        if concept['type'] == 'definition':
            question_data = self._generate_definition_question(concept, num_options)
            
        elif concept['type'] == 'legal_article':
            question_data = self._generate_legal_question(concept, num_options)
            
        elif concept['type'] == 'numeric_info':
            question_data = self._generate_numeric_question(concept, num_options)
            
        elif concept['type'] == 'process':
            question_data = self._generate_process_question(concept, num_options)
            
        else:
            # 기본 템플릿 기반 생성
            question_data = self._generate_template_question(concept, num_options)
        
        # LLM을 사용한 문제 개선 (모델이 로드된 경우)
        if self.model and self.tokenizer:
            question_data = self._enhance_with_llm(question_data)
        
        return question_data
    
    def _generate_definition_question(self, concept: Dict, num_options: int) -> Dict:
        """정의 관련 객관식 문제 생성"""
        
        term = concept.get('term', '')
        definition = concept.get('definition', '')
        
        # 질문 생성
        template = random.choice(self.question_templates['definition'])
        question = template.format(term=term)
        
        # 정답 생성
        correct_answer = definition if definition else f"{term}에 대한 올바른 정의"
        
        # 오답 생성
        distractors = self._generate_distractors_for_term(term, num_options - 1)
        
        # 선택지 구성
        options = [correct_answer] + distractors
        random.shuffle(options)
        
        # 정답 인덱스 찾기
        answer_idx = options.index(correct_answer) + 1
        
        return {
            'type': f'multiple_choice_{num_options}',
            'question': question,
            'options': options,
            'answer': answer_idx,
            'explanation': f"{term}의 정의는 '{correct_answer}'입니다.",
            'source_concept': concept
        }
    
    def _generate_legal_question(self, concept: Dict, num_options: int) -> Dict:
        """법령 관련 객관식 문제 생성"""
        
        article = concept.get('article', '')
        context = concept.get('context', '')
        
        # 질문 생성
        question = f"{article}의 내용으로 옳은 것은?"
        
        # 정답 생성 (컨텍스트에서 핵심 내용 추출)
        correct_answer = self._extract_key_phrase(context)
        
        # 오답 생성 (유사 법령 조항으로 혼동 유발)
        distractors = self._generate_legal_distractors(article, context, num_options - 1)
        
        # 선택지 구성
        options = [correct_answer] + distractors
        random.shuffle(options)
        
        # 정답 인덱스
        answer_idx = options.index(correct_answer) + 1
        
        return {
            'type': f'multiple_choice_{num_options}',
            'question': question,
            'options': options,
            'answer': answer_idx,
            'explanation': f"{article}은 {correct_answer}",
            'source_concept': concept
        }
    
    def _generate_numeric_question(self, concept: Dict, num_options: int) -> Dict:
        """수치 관련 객관식 문제 생성"""
        
        value = concept.get('value', '')
        unit = concept.get('unit', '')
        context = concept.get('context', '')
        
        # 질문 생성
        question_context = self._extract_question_context(context, value)
        question = f"{question_context}의 기준은?"
        
        # 정답
        correct_answer = f"{value}{unit}"
        
        # 오답 생성 (숫자 변형)
        distractors = self._generate_numeric_distractors(value, unit, num_options - 1)
        
        # 선택지 구성
        options = [correct_answer] + distractors
        random.shuffle(options)
        
        # 정답 인덱스
        answer_idx = options.index(correct_answer) + 1
        
        return {
            'type': f'multiple_choice_{num_options}',
            'question': question,
            'options': options,
            'answer': answer_idx,
            'explanation': f"정답은 {correct_answer}입니다.",
            'source_concept': concept
        }
    
    def _generate_process_question(self, concept: Dict, num_options: int) -> Dict:
        """프로세스 관련 객관식 문제 생성"""
        
        keyword = concept.get('keyword', '')
        sentence = concept.get('sentence', '')
        steps = concept.get('steps', [])
        
        # 질문 생성
        if steps:
            question = f"{keyword} 절차는 몇 단계로 구성되는가?"
            correct_answer = f"{len(steps)}단계"
            
            # 오답 생성
            distractors = [f"{len(steps)+i}단계" for i in [-2, -1, 1, 2] if len(steps)+i > 0][:num_options-1]
        else:
            question = f"{keyword} 과정에 포함되는 것은?"
            correct_answer = self._extract_key_phrase(sentence)
            distractors = self._generate_process_distractors(keyword, num_options - 1)
        
        # 선택지 구성
        options = [correct_answer] + distractors
        random.shuffle(options)
        
        # 정답 인덱스
        answer_idx = options.index(correct_answer) + 1
        
        return {
            'type': f'multiple_choice_{num_options}',
            'question': question,
            'options': options,
            'answer': answer_idx,
            'explanation': f"{keyword} 관련 정답은 {correct_answer}",
            'source_concept': concept
        }
    
    def _generate_template_question(self, concept: Dict, num_options: int) -> Dict:
        """템플릿 기반 기본 문제 생성"""
        
        # 개념에서 핵심 정보 추출
        context = concept.get('context', '')
        concept_type = concept.get('type', '')
        
        # 기본 질문
        question = f"다음 중 올바른 설명은?"
        
        # 정답 생성
        correct_answer = self._extract_key_phrase(context)
        
        # 오답 생성
        distractors = []
        for i in range(num_options - 1):
            distractor = self._modify_statement(correct_answer, i)
            distractors.append(distractor)
        
        # 선택지 구성
        options = [correct_answer] + distractors
        random.shuffle(options)
        
        # 정답 인덱스
        answer_idx = options.index(correct_answer) + 1
        
        return {
            'type': f'multiple_choice_{num_options}',
            'question': question,
            'options': options,
            'answer': answer_idx,
            'explanation': f"정답: {correct_answer}",
            'source_concept': concept
        }
    
    def generate_essay_question(self, concept: Dict, use_cot: Optional[bool] = None) -> Dict:
        """
        주관식 문제 생성
        
        Args:
            concept: 개념 정보
            use_cot: CoT 사용 여부 (None이면 기본 설정 사용)
            
        Returns:
            생성된 주관식 문제
        """
        # CoT 사용 여부 결정
        should_use_cot = use_cot if use_cot is not None else self.use_cot
        
        # CoT 기반 생성 (복잡한 주제나 명시적 요청시)
        if should_use_cot and self._should_use_cot_for_concept(concept):
            return self._generate_essay_with_cot(concept)
        
        # 기본 템플릿 기반 생성
        question_data = {
            'type': 'essay',
            'question': '',
            'answer': '',
            'keywords': [],
            'evaluation_criteria': '',
            'source_concept': concept
        }
        
        # 개념 유형별 주관식 문제 생성
        if concept['type'] == 'definition':
            term = concept.get('term', '')
            definition = concept.get('definition', '')
            
            question_data['question'] = f"{term}의 정의와 금융 실무에서의 중요성을 설명하시오."
            question_data['answer'] = definition if definition else f"{term}은 금융 분야에서 중요한 개념입니다."
            question_data['keywords'] = self._extract_keywords(definition)
            
        elif concept['type'] == 'legal_article':
            article = concept.get('article', '')
            context = concept.get('context', '')
            
            question_data['question'] = f"{article}의 주요 내용과 적용 사례를 설명하시오."
            question_data['answer'] = context
            question_data['keywords'] = self._extract_keywords(context)
            
        elif concept['type'] == 'process':
            keyword = concept.get('keyword', '')
            sentence = concept.get('sentence', '')
            
            question_data['question'] = f"{keyword}의 절차와 각 단계별 주의사항을 설명하시오."
            question_data['answer'] = sentence
            question_data['keywords'] = [keyword] + self._extract_keywords(sentence)
            
        else:
            # 기본 주관식 문제
            context = concept.get('context', '')
            question_data['question'] = "다음 내용을 금융 실무 관점에서 설명하시오."
            question_data['answer'] = context
            question_data['keywords'] = self._extract_keywords(context)
        
        # 평가 기준 생성
        question_data['evaluation_criteria'] = self._generate_evaluation_criteria(question_data['keywords'])
        
        # LLM을 사용한 문제 개선
        if self.model and self.tokenizer:
            question_data = self._enhance_essay_with_llm(question_data)
        
        return question_data
    
    def _should_use_cot_for_concept(self, concept: Dict) -> bool:
        """
        개념의 복잡도를 평가하여 CoT 사용 여부 결정
        
        Args:
            concept: 개념 정보
            
        Returns:
            CoT 사용 여부
        """
        if not self.use_cot:
            return False
        
        # 복잡도 점수 계산
        complexity_score = 0
        
        # 법령 관련 개념
        if concept.get('type') == 'legal_article':
            complexity_score += 3
        
        # 프로세스 관련
        if concept.get('type') == 'process':
            complexity_score += 2
        
        # 긴 설명
        context = concept.get('context', '')
        if len(context) > 200:
            complexity_score += 2
        
        # 숫자 정보 포함
        if concept.get('type') == 'numeric_info':
            complexity_score += 1
        
        # 여러 개념 결합
        if '그리고' in context or '또한' in context:
            complexity_score += 2
        
        # 복잡도 임계값 (3점 이상이면 CoT 사용)
        return complexity_score >= 3
    
    def _generate_mc_with_cot(self, concept: Dict, num_options: int) -> Dict:
        """
        CoT를 사용한 객관식 문제 생성
        
        Args:
            concept: 개념 정보
            num_options: 선택지 개수
            
        Returns:
            생성된 문제
        """
        if not self.cot_generator:
            # Fallback to basic generation
            return self._generate_template_question(concept, num_options)
        
        # 난이도 결정
        difficulty = self._assess_difficulty(concept)
        
        # CoT 기반 생성
        result = self.cot_generator.generate_with_reasoning(
            concept=concept,
            problem_type="multiple_choice",
            difficulty=difficulty
        )
        
        # 생성된 문제 추출
        question_data = result['question_data']
        
        # 메타데이터 추가
        question_data['generation_method'] = 'cot'
        question_data['confidence_score'] = result['confidence_score']
        question_data['reasoning_trace'] = result['reasoning_chain']
        
        # 선택지 수 조정 (필요시)
        if len(question_data.get('options', [])) != num_options:
            question_data = self._adjust_options_count(question_data, num_options)
        
        return question_data
    
    def _generate_essay_with_cot(self, concept: Dict) -> Dict:
        """
        CoT를 사용한 주관식 문제 생성
        
        Args:
            concept: 개념 정보
            
        Returns:
            생성된 문제
        """
        if not self.cot_generator:
            # Fallback to basic generation
            return self.generate_essay_question(concept, use_cot=False)
        
        # 난이도 결정
        difficulty = self._assess_difficulty(concept)
        
        # CoT 기반 생성
        result = self.cot_generator.generate_with_reasoning(
            concept=concept,
            problem_type="essay",
            difficulty=difficulty
        )
        
        # 생성된 문제 추출
        question_data = result['question_data']
        
        # 메타데이터 추가
        question_data['generation_method'] = 'cot'
        question_data['confidence_score'] = result['confidence_score']
        question_data['reasoning_trace'] = result['reasoning_chain']
        
        return question_data
    
    def _assess_difficulty(self, concept: Dict) -> str:
        """
        개념의 난이도 평가
        
        Args:
            concept: 개념 정보
            
        Returns:
            난이도 (low, medium, high)
        """
        context = concept.get('context', '')
        concept_type = concept.get('type', '')
        
        # 난이도 점수
        score = 0
        
        # 법령 관련
        if 'legal' in concept_type or '법' in context or '규정' in context:
            score += 3
        
        # 수치 정보
        if 'numeric' in concept_type or any(c.isdigit() for c in context):
            score += 2
        
        # 복잡한 프로세스
        if 'process' in concept_type or '절차' in context or '단계' in context:
            score += 2
        
        # 긴 설명
        if len(context) > 300:
            score += 2
        elif len(context) > 150:
            score += 1
        
        # 난이도 매핑
        if score >= 5:
            return 'high'
        elif score >= 3:
            return 'medium'
        else:
            return 'low'
    
    def _adjust_options_count(self, question_data: Dict, target_count: int) -> Dict:
        """
        선택지 개수 조정
        
        Args:
            question_data: 문제 데이터
            target_count: 목표 선택지 개수
            
        Returns:
            조정된 문제 데이터
        """
        current_options = question_data.get('options', [])
        current_count = len(current_options)
        
        if current_count == target_count:
            return question_data
        
        if current_count < target_count:
            # 선택지 추가
            correct_answer = current_options[question_data.get('answer', 1) - 1]
            additional_distractors = self._generate_distractors_for_term(
                question_data.get('source_concept', {}).get('term', ''),
                target_count - current_count
            )
            current_options.extend(additional_distractors)
        else:
            # 선택지 제거 (정답 제외)
            correct_idx = question_data.get('answer', 1) - 1
            correct_answer = current_options[correct_idx]
            distractors = [opt for i, opt in enumerate(current_options) if i != correct_idx]
            random.shuffle(distractors)
            current_options = [correct_answer] + distractors[:target_count-1]
            random.shuffle(current_options)
            question_data['answer'] = current_options.index(correct_answer) + 1
        
        question_data['options'] = current_options[:target_count]
        question_data['type'] = f'multiple_choice_{target_count}'
        
        return question_data
    
    def generate_with_cot_batch(self, concepts: List[Dict], 
                               problem_types: Optional[List[str]] = None,
                               batch_size: int = 10) -> List[Dict]:
        """
        배치 단위로 CoT 기반 문제 생성
        
        Args:
            concepts: 개념 리스트
            problem_types: 문제 유형 리스트 (None이면 자동 결정)
            batch_size: 배치 크기
            
        Returns:
            생성된 문제 리스트
        """
        if not self.use_cot:
            logger.warning("CoT is not enabled, using basic generation")
            return [self.generate_multiple_choice(c) for c in concepts]
        
        generated_questions = []
        
        for i in range(0, len(concepts), batch_size):
            batch_concepts = concepts[i:i+batch_size]
            
            for j, concept in enumerate(batch_concepts):
                # 문제 유형 결정
                if problem_types and i+j < len(problem_types):
                    problem_type = problem_types[i+j]
                else:
                    # 자동 결정 (70% 객관식, 30% 주관식)
                    problem_type = 'multiple_choice' if random.random() < 0.7 else 'essay'
                
                try:
                    if problem_type == 'essay':
                        question = self.generate_essay_question(concept, use_cot=True)
                    else:
                        num_options = 4 if random.random() < 0.5 else 5
                        question = self.generate_multiple_choice(concept, num_options, use_cot=True)
                    
                    generated_questions.append(question)
                    
                except Exception as e:
                    logger.error(f"Failed to generate question with CoT: {e}")
                    # Fallback to basic generation
                    if problem_type == 'essay':
                        question = self.generate_essay_question(concept, use_cot=False)
                    else:
                        question = self.generate_multiple_choice(concept, 4, use_cot=False)
                    generated_questions.append(question)
            
            logger.info(f"Generated {len(generated_questions)} questions so far")
        
        return generated_questions
    
    def apply_augmentation_strategy(self, 
                                   base_question: Dict, 
                                   strategy: str) -> List[Dict]:
        """
        증강 전략 적용
        
        Args:
            base_question: 기본 문제
            strategy: 증강 전략
            
        Returns:
            증강된 문제 리스트
        """
        augmented_questions = []
        
        if strategy == 'difficulty_variation':
            # 난이도 변형
            augmented_questions.extend(self._vary_difficulty(base_question))
            
        elif strategy == 'paraphrase':
            # 문장 재구성
            augmented_questions.extend(self._paraphrase_question(base_question))
            
        elif strategy == 'scenario_application':
            # 시나리오 적용
            augmented_questions.extend(self._apply_scenario(base_question))
            
        elif strategy == 'concept_combination':
            # 개념 결합
            augmented_questions.extend(self._combine_concepts(base_question))
            
        elif strategy == 'all':
            # 모든 전략 적용
            augmented_questions.extend(self._vary_difficulty(base_question))
            augmented_questions.extend(self._paraphrase_question(base_question))
            augmented_questions.extend(self._apply_scenario(base_question))
            
        return augmented_questions
    
    def _vary_difficulty(self, question: Dict) -> List[Dict]:
        """난이도 변형"""
        variations = []
        
        # 쉬운 버전
        easy_q = question.copy()
        easy_q['question'] = f"다음 중 {question['question'].replace('?', '')}에 대한 기본적인 이해로 옳은 것은?"
        easy_q['difficulty'] = 'easy'
        variations.append(easy_q)
        
        # 어려운 버전
        hard_q = question.copy()
        hard_q['question'] = f"{question['question'].replace('?', '')}를 실무에 적용할 때 고려해야 할 사항은?"
        hard_q['difficulty'] = 'hard'
        variations.append(hard_q)
        
        return variations
    
    def _paraphrase_question(self, question: Dict) -> List[Dict]:
        """문장 재구성"""
        variations = []
        
        # 다른 표현으로 변경
        paraphrased = question.copy()
        original_q = question['question']
        
        # 간단한 패러프레이징 규칙
        replacements = {
            '무엇인가?': '어떻게 정의되는가?',
            '옳은 것은?': '적절한 것은?',
            '설명하시오': '서술하시오',
            '기준은?': '규정은?'
        }
        
        for old, new in replacements.items():
            if old in original_q:
                paraphrased['question'] = original_q.replace(old, new)
                variations.append(paraphrased)
                break
        
        return variations
    
    def _apply_scenario(self, question: Dict) -> List[Dict]:
        """시나리오 적용"""
        variations = []
        
        # 금융 시나리오 템플릿
        scenarios = [
            "A은행에서 ",
            "금융회사 B가 ",
            "투자자 C씨가 ",
            "온라인 금융 서비스 제공 시 "
        ]
        
        for scenario in scenarios[:2]:  # 2개만 생성
            scenario_q = question.copy()
            scenario_q['question'] = scenario + question['question'].lower()
            scenario_q['scenario'] = scenario.strip()
            variations.append(scenario_q)
        
        return variations
    
    def _combine_concepts(self, question: Dict) -> List[Dict]:
        """개념 결합 (구현 예정)"""
        # 다른 개념과 결합하여 복합 문제 생성
        return []
    
    # 헬퍼 메서드들
    def _extract_key_phrase(self, text: str, max_length: int = 100) -> str:
        """텍스트에서 핵심 구문 추출"""
        sentences = text.split('.')
        if sentences:
            key_phrase = sentences[0].strip()
            if len(key_phrase) > max_length:
                key_phrase = key_phrase[:max_length] + "..."
            return key_phrase
        return text[:max_length]
    
    def _extract_keywords(self, text: str) -> List[str]:
        """텍스트에서 키워드 추출"""
        # 간단한 키워드 추출 (향후 개선 가능)
        keywords = []
        
        # 금융 관련 주요 단어들
        important_terms = ['금융', '은행', '투자', '대출', '이자', '금리', '규정', '법', '조항', 
                          '보호', '관리', '감독', '위원회', '정보', '개인정보', '안전', '조치']
        
        for term in important_terms:
            if term in text:
                keywords.append(term)
        
        # 숫자 정보도 키워드로
        numbers = re.findall(r'\d+(?:\.\d+)?(?:%|퍼센트|원|년|개월|일)', text)
        keywords.extend(numbers[:3])  # 최대 3개
        
        return keywords[:5]  # 최대 5개 키워드
    
    def _extract_question_context(self, context: str, value: str) -> str:
        """수치 주변의 질문 컨텍스트 추출"""
        # value 앞의 의미있는 구문 찾기
        idx = context.find(value)
        if idx > 0:
            before_text = context[:idx].strip()
            words = before_text.split()[-5:]  # 마지막 5단어
            return ' '.join(words)
        return "해당 기준"
    
    def _generate_distractors_for_term(self, term: str, num: int) -> List[str]:
        """용어에 대한 오답 생성"""
        distractors = []
        
        # 유사 용어로 오답 생성
        if term in self.distractor_rules.get('term_confusion', {}):
            similar_terms = self.distractor_rules['term_confusion'][term]
            distractors.extend(similar_terms[:num])
        
        # 부족하면 일반적인 오답 추가
        generic_distractors = [
            f"{term}와 유사하지만 다른 개념",
            f"{term}의 반대 개념",
            f"{term}보다 넓은 개념",
            f"{term}보다 좁은 개념"
        ]
        
        while len(distractors) < num:
            distractors.append(generic_distractors[len(distractors) % len(generic_distractors)])
        
        return distractors[:num]
    
    def _generate_legal_distractors(self, article: str, context: str, num: int) -> List[str]:
        """법령 관련 오답 생성"""
        distractors = []
        
        # 조항 번호 변경
        article_num = re.search(r'\d+', article)
        if article_num:
            base_num = int(article_num.group())
            for offset in [-2, -1, 1, 2]:
                if base_num + offset > 0:
                    modified_article = article.replace(str(base_num), str(base_num + offset))
                    distractors.append(f"{modified_article}의 내용")
        
        # 유사 법령으로 변경
        law_variations = ['개인정보보호법', '정보통신망법', '전자금융거래법', '신용정보법']
        for law in law_variations:
            if law not in context:
                distractors.append(f"{law}의 관련 조항")
        
        return distractors[:num]
    
    def _generate_numeric_distractors(self, value: str, unit: str, num: int) -> List[str]:
        """수치 관련 오답 생성"""
        distractors = []
        
        try:
            # 숫자 추출 및 변형
            numeric_value = float(value.replace(',', ''))
            
            # 단위별 변형 규칙 적용
            if '%' in unit or '퍼센트' in unit:
                variations = [numeric_value * 0.5, numeric_value * 0.75, 
                            numeric_value * 1.25, numeric_value * 1.5, numeric_value * 2]
            elif '원' in unit:
                variations = [numeric_value * 0.1, numeric_value * 0.5, 
                            numeric_value * 2, numeric_value * 10]
            elif '년' in unit or '개월' in unit or '일' in unit:
                if numeric_value < 12:
                    variations = [numeric_value * 2, numeric_value * 3, numeric_value * 6]
                else:
                    variations = [numeric_value / 2, numeric_value * 2, numeric_value * 1.5]
            else:
                # 기본 변형
                variations = [numeric_value * 0.8, numeric_value * 1.2, 
                            numeric_value * 1.5, numeric_value * 2]
            
            for var in variations[:num]:
                # 원래 형식 유지
                if ',' in value:
                    formatted = f"{var:,.0f}"
                elif '.' in value:
                    formatted = f"{var:.1f}"
                else:
                    formatted = f"{int(var)}"
                
                distractors.append(f"{formatted}{unit}")
                
        except:
            # 숫자 변환 실패 시 기본 오답
            distractors = [f"약 {value}{unit}", f"{value}{unit} 이상", 
                          f"{value}{unit} 이하", f"{value}{unit} 내외"]
        
        return distractors[:num]
    
    def _generate_process_distractors(self, keyword: str, num: int) -> List[str]:
        """프로세스 관련 오답 생성"""
        distractors = []
        
        # 관련 없는 프로세스 단계들
        unrelated_steps = [
            "사전 검토 단계",
            "내부 승인 절차",
            "외부 감사 과정",
            "최종 검증 단계",
            "사후 모니터링",
            "정기 점검 절차"
        ]
        
        for step in unrelated_steps:
            if keyword not in step:
                distractors.append(step)
        
        return distractors[:num]
    
    def _modify_statement(self, statement: str, variation_idx: int) -> str:
        """문장 변형하여 오답 생성"""
        modifications = [
            lambda s: s.replace('해야 한다', '할 수 있다'),
            lambda s: s.replace('이상', '이하'),
            lambda s: s.replace('포함', '제외'),
            lambda s: s.replace('필수', '선택'),
            lambda s: s.replace('금지', '허용')
        ]
        
        if variation_idx < len(modifications):
            return modifications[variation_idx](statement)
        
        return statement + " (예외 조항 적용)"
    
    def _generate_evaluation_criteria(self, keywords: List[str]) -> str:
        """주관식 평가 기준 생성"""
        criteria = f"평가 기준:\n"
        criteria += f"1. 핵심 키워드 포함 여부: {', '.join(keywords)}\n"
        criteria += f"2. 논리적 설명의 일관성\n"
        criteria += f"3. 금융 실무 관점의 적절성\n"
        criteria += f"4. 구체적 사례 또는 근거 제시"
        
        return criteria
    
    def _enhance_with_llm(self, question_data: Dict) -> Dict:
        """LLM을 사용한 문제 개선"""
        if not self.model or not self.tokenizer:
            return question_data
        
        try:
            # 프롬프트 생성
            prompt = self._create_enhancement_prompt(question_data)
            
            # 토크나이징
            inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024)
            
            if self.device == "cuda":
                inputs = inputs.to(self.device)
            
            # 생성
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=256,
                    temperature=0.7,
                    do_sample=True,
                    top_p=0.9
                )
            
            # 디코딩
            enhanced_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # 개선된 내용 파싱 및 적용
            question_data = self._parse_enhanced_question(enhanced_text, question_data)
            
        except Exception as e:
            logger.warning(f"LLM enhancement failed: {e}")
        
        return question_data
    
    def _enhance_essay_with_llm(self, question_data: Dict) -> Dict:
        """LLM을 사용한 주관식 문제 개선"""
        if not self.model or not self.tokenizer:
            return question_data
        
        try:
            # 주관식 개선 프롬프트
            prompt = f"""
            다음 주관식 문제를 개선해주세요:
            
            원본 질문: {question_data['question']}
            
            개선 요구사항:
            1. 금융 실무 관점 강조
            2. 구체적인 답변 유도
            3. 평가 가능한 명확한 질문
            
            개선된 질문:
            """
            
            inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
            
            if self.device == "cuda":
                inputs = inputs.to(self.device)
            
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=128,
                    temperature=0.7,
                    do_sample=True
                )
            
            enhanced_question = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # 개선된 질문 추출
            if "개선된 질문:" in enhanced_question:
                question_data['question'] = enhanced_question.split("개선된 질문:")[-1].strip()
            
        except Exception as e:
            logger.warning(f"Essay enhancement failed: {e}")
        
        return question_data
    
    def _create_enhancement_prompt(self, question_data: Dict) -> str:
        """문제 개선을 위한 프롬프트 생성"""
        prompt = f"""
        금융 전문가로서 다음 문제를 검토하고 개선해주세요:
        
        문제: {question_data['question']}
        선택지:
        """
        
        for i, option in enumerate(question_data.get('options', []), 1):
            prompt += f"{i}. {option}\n"
        
        prompt += f"""
        정답: {question_data.get('answer', 1)}번
        
        개선 사항:
        1. 문제가 명확한가?
        2. 선택지가 적절한가?
        3. 금융 용어가 정확한가?
        
        개선된 문제를 제시해주세요.
        """
        
        return prompt
    
    def _parse_enhanced_question(self, enhanced_text: str, original_question: Dict) -> Dict:
        """LLM 출력에서 개선된 문제 파싱"""
        # 간단한 파싱 로직 (향후 개선 가능)
        if "개선된 문제:" in enhanced_text:
            parts = enhanced_text.split("개선된 문제:")[-1].strip()
            # 질문 부분 추출
            lines = parts.split('\n')
            if lines:
                original_question['question'] = lines[0].strip()
        
        return original_question


if __name__ == "__main__":
    # 테스트용 코드
    import sys
    from pathlib import Path
    sys.path.append(str(Path(__file__).parent.parent.parent))
    
    logging.basicConfig(level=logging.INFO)
    
    # 문제 생성기 초기화 (모델 없이 템플릿 기반으로 테스트)
    generator = FSKUQuestionGenerator(model_name="dummy", use_quantization=False)
    
    # 테스트 개념
    test_concepts = [
        {
            'type': 'definition',
            'term': '금리',
            'definition': '자금을 빌려주는 대가로 받는 이자의 비율',
            'context': '금리란 자금을 빌려주는 대가로 받는 이자의 비율을 의미한다.'
        },
        {
            'type': 'numeric_info',
            'value': '10',
            'unit': '조원',
            'context': '자산총액이 10조원 이상인 은행은 재해복구센터를 구축해야 한다.'
        },
        {
            'type': 'legal_article',
            'article': '제21조',
            'context': '제21조에서는 금융회사의 정보보호 의무를 규정한다.'
        }
    ]
    
    print("=== 객관식 문제 생성 테스트 ===\n")
    
    for concept in test_concepts:
        # 4지선다 생성
        mc4 = generator.generate_multiple_choice(concept, num_options=4)
        print(f"문제: {mc4['question']}")
        print("선택지:")
        for i, opt in enumerate(mc4['options'], 1):
            print(f"  {i}. {opt}")
        print(f"정답: {mc4['answer']}번")
        print(f"설명: {mc4['explanation']}\n")
    
    print("\n=== 주관식 문제 생성 테스트 ===\n")
    
    for concept in test_concepts[:1]:
        essay = generator.generate_essay_question(concept)
        print(f"문제: {essay['question']}")
        print(f"예시 답안: {essay['answer'][:100]}...")
        print(f"핵심 키워드: {', '.join(essay['keywords'])}")
        print(f"평가 기준:\n{essay['evaluation_criteria']}\n")
    
    print("\n=== 증강 전략 테스트 ===\n")
    
    base_q = mc4  # 마지막 생성 문제 사용
    augmented = generator.apply_augmentation_strategy(base_q, 'difficulty_variation')
    
    for aug_q in augmented:
        print(f"난이도: {aug_q.get('difficulty', 'normal')}")
        print(f"문제: {aug_q['question']}\n")