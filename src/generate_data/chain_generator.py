#!/usr/bin/env python3
"""
체이닝 기능이 추가된 데이터 생성기

기존 DataGenerator를 확장하여 다단계 생성/검증/개선 프로세스 구현
"""

import json
import time
import re
from typing import Dict, List, Optional, Any
import logging
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# 기존 DataGenerator 상속
try:
    from .main import DataGenerator
except ImportError:
    from main import DataGenerator

logger = logging.getLogger(__name__)


class ChainDataGenerator(DataGenerator):
    """체이닝 기능이 추가된 데이터 생성기"""
    
    def __init__(self, config: Dict[str, Any]):
        """초기화"""
        super().__init__(config)
        self.chain_steps = config.get('chain_steps', ['generate'])
        self.validation_model = None
        self.validation_tokenizer = None
        
        # 검증용 모델 로드
        if config.get('validation_model') and 'validate' in self.chain_steps:
            self._load_validation_model()
    
    def _load_validation_model(self):
        """검증용 모델 로드"""
        logger.info(f"검증 모델 로딩: {self.config['validation_model']}")
        
        self.validation_tokenizer = AutoTokenizer.from_pretrained(
            self.config['validation_model'],
            trust_remote_code=True
        )
        
        if self.validation_tokenizer.pad_token is None:
            self.validation_tokenizer.pad_token = self.validation_tokenizer.eos_token
        
        # 리소스 제약 없는 환경이므로 전체 정밀도 사용
        self.validation_model = AutoModelForCausalLM.from_pretrained(
            self.config['validation_model'],
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True
        )
        
        logger.info("검증 모델 로드 완료")
    
    def generate_with_chain(self, prompt: str, context: str) -> Dict[str, Any]:
        """체이닝을 통한 문제 생성"""
        result = {
            'prompt': prompt,
            'context': context,
            'chain_history': []
        }
        
        current_output = None
        
        for step in self.chain_steps:
            if step == 'generate':
                # 1단계: 초기 생성
                current_output = self._generate_with_llm(prompt, self.config['temperature'])
                result['initial_output'] = current_output
                
            elif step == 'validate' and self.validation_model:
                # 2단계: 검증
                validation_result = self._validate_output(current_output)
                result['validation_score'] = validation_result['score']
                result['validation_feedback'] = validation_result['feedback']
                
                if validation_result['score'] < self.config.get('min_quality_score', 80):
                    current_output = validation_result['suggestion']
                    
            elif step == 'refine':
                # 3단계: 개선
                refined_output = self._refine_output(
                    current_output, 
                    result.get('validation_feedback', '')
                )
                current_output = refined_output
                
            elif step == 'final_check':
                # 4단계: 최종 검증
                final_score = self.quality_checker.evaluate(current_output)
                result['final_score'] = final_score
            
            result['chain_history'].append({
                'step': step,
                'output': current_output[:200] + '...' if len(current_output) > 200 else current_output
            })
        
        result['final_output'] = current_output
        return result
    
    def _validate_output(self, output: str) -> Dict[str, Any]:
        """검증 모델을 사용한 출력 평가"""
        validation_prompt = f"""다음 금융 시험 문제를 평가하고 개선점을 제시하세요.

문제:
{output}

평가 기준:
1. 명확성 (1-10): 문제가 명확하고 모호하지 않은가?
2. 정확성 (1-10): 금융 지식이 정확한가?
3. 난이도 (1-10): 적절한 난이도인가?
4. 형식 (1-10): 문제 형식이 올바른가?

출력 형식:
[점수] 총점: XX/40
[피드백] 개선이 필요한 부분
[제안] 개선된 문제
"""
        
        # 검증 모델로 평가
        inputs = self.validation_tokenizer(
            validation_prompt,
            return_tensors="pt",
            truncation=True,
            max_length=2048
        ).to(self.validation_model.device)
        
        with torch.no_grad():
            outputs = self.validation_model.generate(
                **inputs,
                max_new_tokens=512,
                temperature=0.7,
                do_sample=True
            )
        
        validation_output = self.validation_tokenizer.decode(outputs[0], skip_special_tokens=True)
        validation_output = validation_output.replace(validation_prompt, '').strip()
        
        # 결과 파싱
        score_match = re.search(r'총점[:\s]*(\d+)', validation_output)
        score = int(score_match.group(1)) if score_match else 20
        
        feedback_match = re.search(r'\[피드백\](.*?)\[제안\]', validation_output, re.DOTALL)
        feedback = feedback_match.group(1).strip() if feedback_match else ''
        
        suggestion_match = re.search(r'\[제안\](.*)', validation_output, re.DOTALL)
        suggestion = suggestion_match.group(1).strip() if suggestion_match else output
        
        return {
            'score': (score / 40) * 100,  # 100점 만점으로 변환
            'feedback': feedback,
            'suggestion': suggestion
        }
    
    def _refine_output(self, output: str, feedback: str) -> str:
        """피드백을 반영한 출력 개선"""
        refine_prompt = f"""다음 금융 문제를 피드백을 참고하여 개선하세요.

원본 문제:
{output}

피드백:
{feedback}

개선된 문제 (더 명확하고 전문적으로):
"""
        
        refined = self._generate_with_llm(refine_prompt, temperature=0.7)
        return refined
    
    def generate_questions_batch(self, chunks: List[Dict], batch_size: int = 32) -> List[Dict]:
        """배치 단위로 문제 생성"""
        questions = []
        
        for i in range(0, len(chunks), batch_size):
            batch_chunks = chunks[i:i+batch_size]
            batch_prompts = []
            
            for chunk in batch_chunks:
                prompt = self.prompt_template.format(
                    concept=chunk.get('keywords', ['금융'])[0] if chunk.get('keywords') else '금융',
                    context=chunk['content']
                )
                batch_prompts.append(prompt)
            
            # 배치 처리 (향후 병렬화 가능)
            for prompt, chunk in zip(batch_prompts, batch_chunks):
                try:
                    if self.config.get('use_chaining'):
                        result = self.generate_with_chain(prompt, chunk['content'])
                        if result.get('final_score', 0) >= self.config.get('min_quality_score', 80):
                            question = self._parse_question_result(result)
                            questions.append(question)
                    else:
                        # 기존 방식
                        generated = self._generate_with_llm(prompt, self.config['temperature'])
                        question = self._extract_question_answer(generated)
                        questions.append(question)
                except Exception as e:
                    logger.error(f"생성 오류: {e}")
                    continue
        
        return questions
    
    def _parse_question_result(self, result: Dict) -> Dict:
        """체이닝 결과를 문제 형식으로 파싱"""
        final_output = result['final_output']
        
        # 기존 파싱 메서드 사용
        question = self._clean_question(final_output)
        answer = self._extract_answer(final_output)
        
        return {
            'id': f"CHAIN_{int(time.time()*1000) % 1000000:06d}",
            'question': question,
            'answer': answer,
            'metadata': {
                'chain_steps': len(result['chain_history']),
                'initial_score': result.get('validation_score', 0),
                'final_score': result.get('final_score', 0),
                'context_used': bool(result.get('context'))
            }
        }
    
    def _extract_question_answer(self, generated: str) -> Dict:
        """기존 방식의 질문/답변 추출"""
        question = self._clean_question(generated)
        answer = self._extract_answer(generated)
        
        return {
            'id': f"GEN_{int(time.time()*1000) % 1000000:06d}",
            'question': question,
            'answer': answer
        }


def generate_chain_data(config: Dict[str, Any]) -> str:
    """체이닝 데이터 생성 실행"""
    print("="*60)
    print("🔗 체이닝 데이터 생성 시작")
    print(f"모델: {config.get('model_name')}")
    print(f"검증 모델: {config.get('validation_model', 'None')}")
    print(f"체인 단계: {config.get('chain_steps', ['generate'])}")
    print("="*60)
    
    # 생성기 초기화
    generator = ChainDataGenerator(config)
    
    # 문제 생성
    start_time = time.time()
    questions = generator.generate_questions(
        num_questions=config.get('num_questions', 100),
        min_quality_score=config.get('min_quality_score', 80),
        temperature=config.get('temperature', 0.7)
    )
    elapsed_time = time.time() - start_time
    
    # 결과 저장
    output_file = generator.save_results(questions)
    
    print(f"\n✅ 체이닝 생성 완료!")
    print(f"📁 저장 위치: {output_file}")
    print(f"📊 생성된 문제: {len(questions)}개")
    print(f"⏱️ 소요 시간: {elapsed_time:.1f}초")
    
    # 체이닝 통계
    chain_questions = [q for q in questions if 'metadata' in q and 'chain_steps' in q['metadata']]
    if chain_questions:
        avg_improvement = sum(
            q['metadata'].get('final_score', 0) - q['metadata'].get('initial_score', 0)
            for q in chain_questions
        ) / len(chain_questions)
        print(f"📈 평균 품질 개선: +{avg_improvement:.1f}점")
    
    return output_file


if __name__ == "__main__":
    # 테스트 실행
    test_config = {
        'model_name': 'Qwen/Qwen2.5-7B-Instruct',
        'validation_model': 'meta-llama/Llama-3-7B-Instruct',
        'use_rag': True,
        'use_chaining': True,
        'chain_steps': ['generate', 'validate', 'refine', 'final_check'],
        'num_questions': 10,
        'min_quality_score': 80
    }
    
    generate_chain_data(test_config)