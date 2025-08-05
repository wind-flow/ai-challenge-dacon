#!/usr/bin/env python3
"""
데이터 생성 메인 모듈

학습 데이터 증강을 위한 통합 시스템
RAG를 활용한 고품질 금융 문제 생성
"""

import json
import time
import re
from pathlib import Path
from typing import Dict, List, Optional, Any
from datetime import datetime
import logging

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from tqdm import tqdm

# 로컬 모듈
try:
    from generate_data.quality_checker import QualityChecker
    from rag.retriever import DocumentRetriever
except ImportError:
    # 직접 실행시
    try:
        from .quality_checker import QualityChecker
    except ImportError:
        from quality_checker import QualityChecker
    
    import sys
    sys.path.append(str(Path(__file__).parent.parent))
    from rag.retriever import DocumentRetriever

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DataGenerator:
    """
    학습 데이터 생성 클래스
    
    외부 데이터 기반 개념 추출 → RAG 검색 → LLM 생성 → 품질 검증
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        초기화
        
        Args:
            config: 설정 딕셔너리
                - model_name: 사용할 LLM 모델
                - use_rag: RAG 사용 여부
                - use_quantization: 4bit 양자화 사용
                - prompt_template: 프롬프트 템플릿 경로
        """
        self.config = config
        self.model = None
        self.tokenizer = None
        
        # 디바이스 설정 (CUDA > MPS > CPU)
        if torch.cuda.is_available():
            self.device = "cuda"
        elif torch.backends.mps.is_available():
            self.device = "mps"  # Mac GPU
        else:
            self.device = "cpu"
        
        # 컴포넌트 초기화
        self.quality_checker = QualityChecker()
        
        if config.get('use_rag', False):
            self.retriever = DocumentRetriever()
        else:
            self.retriever = None
        
        # 프롬프트 템플릿 로드
        self.prompt_template = self._load_prompt_template()
        
        logger.info(f"데이터 생성기 초기화 완료 (Device: {self.device})")
    
    def _load_prompt_template(self) -> str:
        """프롬프트 템플릿 로드"""
        template_path = Path(self.config.get('prompt_template', 'prompts/default.txt'))
        
        if template_path.exists():
            with open(template_path, 'r', encoding='utf-8') as f:
                return f.read()
        else:
            # 기본 템플릿
            return """당신은 한국 금융 전문가입니다.

주제: {concept}
참고자료: {context}

위 내용을 바탕으로 FSKU 평가용 문제를 생성하세요.

요구사항:
- 한국 금융 실무와 관련된 내용
- 명확하고 모호하지 않은 문제
- 객관식 또는 주관식 형태

문제:"""
    
    def load_model(self):
        """LLM 모델 로드 (에러 처리 포함)"""
        model_name = self.config['model_name']
        print(f"🔄 모델 로딩 중: {model_name}")
        print("   (첫 실행시 다운로드로 시간이 걸릴 수 있습니다...)")
        
        try:
            # 토크나이저
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_name,
                trust_remote_code=True
            )
            
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            # 모델 설정
            if self.config.get('use_quantization', True) and self.device == "cuda":  # 양자화는 CUDA만 지원
                # 4bit 양자화 (메모리 절약)
                bnb_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.float16,
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_quant_type="nf4"
                )
                
                self.model = AutoModelForCausalLM.from_pretrained(
                    model_name,
                    quantization_config=bnb_config,
                    device_map="auto",
                    trust_remote_code=True
                )
                print("✅ 4bit 양자화 모델 로드 완료")
            else:
                # MPS나 CPU 사용
                if self.device == "mps":
                    self.model = AutoModelForCausalLM.from_pretrained(
                        model_name,
                        torch_dtype=torch.float16,
                        trust_remote_code=True
                    )
                    self.model = self.model.to(self.device)
                    print(f"✅ MPS(Mac GPU) 모델 로드 완료")
                else:
                    self.model = AutoModelForCausalLM.from_pretrained(
                        model_name,
                        torch_dtype=torch.float32,
                        trust_remote_code=True
                    )
                    print(f"✅ CPU 모델 로드 완료")
                
        except Exception as e:
            print(f"❌ 모델 로드 실패: {e}")
            print("💡 해결 방법:")
            print("   1. python download_models.py 실행")
            print("   2. 인터넷 연결 확인")
            print("   3. GPU 메모리 확인 (nvidia-smi)")
            raise
    
    def generate_questions(self, num_questions: int = 100, min_quality_score: int = 70, temperature: float = 0.7) -> List[Dict]:
        """
        문제 생성
        
        Args:
            num_questions: 생성할 문제 수
            min_quality_score: 최소 품질 점수
            temperature: 생성 온도
            
        Returns:
            생성된 문제 리스트
        """
        # 모델 로드
        if self.model is None:
            self.load_model()
        
        # RAG 준비 확인
        if self.retriever:
            print(f"✅ RAG 시스템 준비 완료")
        else:
            print("⚠️ RAG 사용 안 함")
        
        # 문제 생성
        generated_questions = []
        
        failed_attempts = 0
        max_retries = 3
        
        with tqdm(total=num_questions, desc="문제 생성 중") as pbar:
            while len(generated_questions) < num_questions and failed_attempts < num_questions * 2:
                try:
                    # RAG에서 랜덤 컨텍스트 가져오기
                    context = ""
                    concept = ""
                    
                    if self.retriever:
                        try:
                            # 랜덤 청크에서 컨텍스트 가져오기
                            context = self.retriever.get_random_chunks(n=2)
                            
                            # 컨텍스트에서 첫 번째 명사 찾기 (참고용)
                            nouns = re.findall(r'[\uac00-\ud7a3]{2,10}', context)
                            if nouns:
                                concept = nouns[0]  # 첫 번째 명사를 참고로 사용
                        except Exception as e:
                            logger.debug(f"RAG 컨텍스트 가져오기 실패: {e}")
                    
                    if not context:
                        # RAG가 없거나 실패한 경우
                        context = "금융 관련 일반 지식"
                        concept = "금융"
                    
                    # 프롬프트 생성
                    prompt = self.prompt_template.format(
                        concept=concept,
                        context=context if context else "없음"
                    )
                    
                    # LLM으로 생성 (재시도 로직 포함)
                    question = None
                    for retry in range(max_retries):
                        try:
                            question = self._generate_with_llm(prompt, temperature)
                            break
                        except Exception as e:
                            if retry < max_retries - 1:
                                logger.debug(f"생성 재시도 {retry+1}/{max_retries}: {e}")
                                time.sleep(1)
                            else:
                                raise
                    
                    if not question:
                        failed_attempts += 1
                        continue
                    
                    # 품질 평가 (개념 없이)
                    quality_score = self.quality_checker.evaluate(question)
                    
                    # 결과 저장 (질문과 답만)
                    # 질문에서 답 추출 시도
                    answer = self._extract_answer(question)
                    clean_question = self._clean_question(question)
                    
                    result = {
                        'id': f"GEN_{len(generated_questions)+1:05d}",
                        'question': clean_question,
                        'answer': answer
                    }
                    
                    # 메타데이터는 별도 저장
                    metadata = {
                        'id': result['id'],
                        'concept': concept,
                        'quality_score': quality_score,
                        'context_used': bool(context),
                        'context': context[:500] if context else "",
                        'timestamp': datetime.now().isoformat(),
                        'model': self.config['model_name'],
                        'full_generated': question
                    }
                    
                    # 메타데이터 리스트에 추가 (클래스 레벨 변수 필요)
                    if not hasattr(self, 'metadata_list'):
                        self.metadata_list = []
                    self.metadata_list.append(metadata)
                    
                    # 품질 기준 통과한 것만 추가
                    if quality_score >= min_quality_score:
                        generated_questions.append(result)
                        pbar.update(1)
                    else:
                        logger.debug(f"품질 미달: {quality_score:.1f}점")
                        failed_attempts += 1
                        
                except Exception as e:
                    logger.error(f"문제 생성 오류: {e}")
                    failed_attempts += 1
                    continue
        
        logger.info(f"총 {len(generated_questions)}개 문제 생성 완료")
        return generated_questions
    
    def _generate_with_llm(self, prompt: str, temperature: float = 0.7) -> str:
        """LLM으로 텍스트 생성"""
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=2048
        )
        
        if self.device in ["cuda", "mps"]:
            inputs = inputs.to(self.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=512,
                temperature=temperature,
                top_p=self.config.get('top_p', 0.9),
                do_sample=True,
                pad_token_id=self.tokenizer.pad_token_id
            )
        
        generated = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # 프롬프트 제거
        if prompt in generated:
            generated = generated.replace(prompt, "").strip()
        
        return generated
    
    def _extract_answer(self, generated_text: str) -> str:
        """생성된 텍스트에서 답변 추출"""
        # 다양한 패턴으로 답변 찾기
        patterns = [
            r'\[ANSWER\]([\s\S]*?)\[',
            r'\[답변\]([\s\S]*?)\[',
            r'\[답\]([\s\S]*?)\[', 
            r'정답[:：]\s*([^\n]+)',
            r'답[:：]\s*([^\n]+)'
        ]
        
        for pattern in patterns:
            match = re.search(pattern, generated_text, re.IGNORECASE | re.MULTILINE)
            if match:
                return match.group(1).strip()
        
        # 객관식 패턴 (1번, 2번 등)
        if any(num in generated_text for num in ['1)', '2)', '3)', '4)', '5)']):
            answer_match = re.search(r'정답.*?([1-5])[번\)]', generated_text)
            if answer_match:
                return answer_match.group(1) + "번"
        
        return "답변 추출 실패"
    
    def _clean_question(self, generated_text: str) -> str:
        """생성된 텍스트에서 질문만 추출"""
        # 질문 패턴
        patterns = [
            r'\[QUESTION\]([\s\S]*?)\[',
            r'\[문제\]([\s\S]*?)\[',
            r'문제[:：]\s*([^\[]+)',
        ]
        
        for pattern in patterns:
            match = re.search(pattern, generated_text, re.IGNORECASE | re.MULTILINE)
            if match:
                question = match.group(1).strip()
                # 선택지 제거
                if '\n1)' in question:
                    question = question[:question.find('\n1)')].strip()
                return question
        
        # 처음부터 물음표까지 찾기
        question_end = generated_text.find('?')
        if question_end > 0:
            return generated_text[:question_end+1].strip()
        
        # 처음 200자만 반환 (폴백)
        return generated_text[:200].strip()
    
    def save_results(self, questions: List[Dict], output_file: str = None):
        """결과 저장"""
        if not output_file:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_file = f"generated_data_{timestamp}.json"
        
        output_path = Path("results") / output_file
        output_path.parent.mkdir(exist_ok=True)
        
        result_data = {
            'metadata': {
                'total_questions': len(questions),
                'generation_time': datetime.now().isoformat(),
                'config': self.config,
                'quality_stats': self._calculate_quality_stats(questions)
            },
            'questions': questions
        }
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(result_data, f, ensure_ascii=False, indent=2)
        
        logger.info(f"결과 저장: {output_path}")
        return str(output_path)
    
    def _calculate_quality_stats(self, questions: List[Dict]) -> Dict:
        """품질 통계 계산"""
        if not questions or not hasattr(self, 'metadata_list'):
            return {'avg_score': 0, 'min_score': 0, 'max_score': 0, 'total': 0}
        
        scores = [m['quality_score'] for m in self.metadata_list if 'quality_score' in m]
        
        if not scores:
            return {'avg_score': 0, 'min_score': 0, 'max_score': 0, 'total': 0}
        
        return {
            'avg_score': sum(scores) / len(scores),
            'min_score': min(scores),
            'max_score': max(scores),
            'total': len(scores)
        }
    
    def cleanup(self):
        """메모리 정리"""
        if self.model is not None:
            del self.model
            self.model = None
        if self.tokenizer is not None:
            del self.tokenizer
            self.tokenizer = None
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        print("🧹 메모리 정리 완료")
    
    def generate_batch(self, prompts: List[str], temperature: float = 0.7) -> List[str]:
        """배치 생성 (향후 구현)"""
        # TODO: 배치 처리 최적화
        results = []
        for prompt in prompts:
            result = self._generate_with_llm(prompt, temperature)
            results.append(result)
        return results


def generate_data(config: Dict[str, Any] = None) -> str:
    """
    데이터 생성 실행 함수
    
    Args:
        config: 설정 딕셔너리
        
    Returns:
        저장된 파일 경로
    """
    if config is None:
        # 기본 설정
        config = {
            'model_name': 'upstage/SOLAR-10.7B-v1.0',
            'use_rag': True,
            'use_quantization': True,
            'num_questions': 100,
            'min_quality': 70,
            'temperature': 0.7,
            'prompt_template': 'prompts/cot.txt'
        }
    
    print("="*60)
    print("📊 데이터 생성 시작")
    print(f"모델: {config.get('model_name', 'upstage/SOLAR-10.7B-v1.0')}")
    print(f"문제 수: {config.get('num_questions', 100)}개")
    print(f"RAG 사용: {'예' if config.get('use_rag', True) else '아니오'}")
    print("="*60)
    
    try:
        # 생성기 초기화
        generator_config = {
            'model_name': config.get('model_name', 'upstage/SOLAR-10.7B-v1.0'),
            'use_rag': config.get('use_rag', True),
            'use_quantization': config.get('use_quantization', True),
            'prompt_template': config.get('prompt_template', 'prompts/cot.txt'),
            'top_p': config.get('top_p', 0.9),
            'temperature': config.get('temperature', 0.7)
        }
        generator = DataGenerator(generator_config)
        
        # 문제 생성
        start_time = time.time()
        questions = generator.generate_questions(
            num_questions=config.get('num_questions', 100),
            min_quality_score=config.get('min_quality', 70),
            temperature=config.get('temperature', 0.7)
        )
        elapsed_time = time.time() - start_time
        
        # 메모리 정리
        generator.cleanup()
        
    except Exception as e:
        print(f"❌ 오류 발생: {e}")
        if 'generator' in locals():
            generator.cleanup()
        raise
    
    # 결과 저장 (JSONL 형식)
    output_dir = Path("data/augmented")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = output_dir / f"train_data_{timestamp}.jsonl"
    metadata_file = output_dir / f"metadata_{timestamp}.json"
    
    # 질문과 답만 JSONL로 저장
    with open(output_file, 'w', encoding='utf-8') as f:
        for question in questions:
            f.write(json.dumps(question, ensure_ascii=False) + '\n')
    
    # 메타데이터는 별도 JSON 파일로 저장
    if hasattr(generator, 'metadata_list'):
        with open(metadata_file, 'w', encoding='utf-8') as f:
            json.dump({
                'generation_info': {
                    'timestamp': timestamp,
                    'model': config.get('model_name'),
                    'total_questions': len(questions),
                    'config': config
                },
                'metadata': generator.metadata_list
            }, f, ensure_ascii=False, indent=2)
        print(f"📄 메타데이터: {metadata_file}")
    
    # 결과 출력
    print("="*60)
    print("✅ 데이터 생성 완료!")
    print(f"📁 훈련 데이터: {output_file}")
    if 'metadata_file' in locals():
        print(f"📄 메타데이터: {metadata_file}")
    print(f"📊 생성된 문제: {len(questions)}개")
    print(f"⏱️ 소요 시간: {elapsed_time:.1f}초")
    print("="*60)
    
    # 샘플 출력
    if questions and len(questions) > 0:
        print("\n🔍 생성 예시 (3개):")
        for i, q in enumerate(questions[:3], 1):
            print(f"\n[{i}] 질문: {q['question'][:100]}..." if len(q['question']) > 100 else f"\n[{i}] 질문: {q['question']}")
            print(f"    답변: {q['answer'][:50]}..." if len(q['answer']) > 50 else f"    답변: {q['answer']}")
    
    return str(output_file)




if __name__ == "__main__":
    # 테스트 실행
    config = {
        'model_name': 'beomi/SOLAR-10.7B-v1.0',
        'use_rag': True,
        'num_questions': 10,  # 테스트용 적은 수
        'prompt_template': 'prompts/cot.txt'
    }
    
    generate_data(config)