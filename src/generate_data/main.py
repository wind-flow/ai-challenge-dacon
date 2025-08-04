#!/usr/bin/env python3
"""
데이터 생성 메인 모듈

학습 데이터 증강을 위한 통합 시스템
RAG를 활용한 고품질 금융 문제 생성
"""

import json
import time
from pathlib import Path
from typing import Dict, List, Optional, Any
from datetime import datetime
import logging

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from tqdm import tqdm

# 로컬 모듈
from .concept_extractor import ConceptExtractor
from .quality_checker import QualityChecker
from ..rag.retriever import DocumentRetriever

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
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # 컴포넌트 초기화
        self.concept_extractor = ConceptExtractor()
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
        """LLM 모델 로드"""
        model_name = self.config['model_name']
        logger.info(f"모델 로딩 중: {model_name}")
        
        # 토크나이저
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=True
        )
        
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # 모델 설정
        if self.config.get('use_quantization', True) and self.device == "cuda":
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
            logger.info("4bit 양자화 모델 로드 완료")
        else:
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                device_map="auto" if self.device == "cuda" else None,
                trust_remote_code=True
            )
            logger.info("일반 모델 로드 완료")
    
    def generate_questions(self, num_questions: int = 100) -> List[Dict]:
        """
        문제 생성
        
        Args:
            num_questions: 생성할 문제 수
            
        Returns:
            생성된 문제 리스트
        """
        # 모델 로드
        if self.model is None:
            self.load_model()
        
        # 외부 데이터에서 개념 추출
        concepts = self.concept_extractor.extract_concepts()
        if not concepts:
            logger.error("추출된 개념이 없습니다. data/external/ 폴더를 확인하세요.")
            return []
        
        logger.info(f"{len(concepts)}개 개념 추출 완료")
        
        # 문제 생성
        generated_questions = []
        
        for i in tqdm(range(num_questions), desc="문제 생성"):
            # 개념 선택 (빈도 기반 가중치)
            concept_info = self.concept_extractor.get_weighted_concept()
            concept = concept_info['concept']
            
            # RAG로 관련 문서 검색
            context = ""
            if self.retriever:
                context = self.retriever.search(concept, top_k=3)
            
            # 프롬프트 생성
            prompt = self.prompt_template.format(
                concept=concept,
                context=context if context else "없음"
            )
            
            # LLM으로 생성
            question = self._generate_with_llm(prompt)
            
            # 품질 평가
            quality_score = self.quality_checker.evaluate(question, concept)
            
            # 결과 저장
            result = {
                'id': f"GEN_{i+1:05d}",
                'concept': concept,
                'question': question,
                'quality_score': quality_score,
                'context_used': bool(context),
                'timestamp': datetime.now().isoformat(),
                'model': self.config['model_name']
            }
            
            # 품질 기준 통과한 것만 추가
            if quality_score >= self.config.get('min_quality', 70):
                generated_questions.append(result)
            else:
                logger.debug(f"품질 미달: {quality_score:.1f}점")
        
        logger.info(f"총 {len(generated_questions)}개 문제 생성 완료")
        return generated_questions
    
    def _generate_with_llm(self, prompt: str) -> str:
        """LLM으로 텍스트 생성"""
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=2048
        )
        
        if self.device == "cuda":
            inputs = inputs.to(self.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=512,
                temperature=self.config.get('temperature', 0.7),
                top_p=self.config.get('top_p', 0.9),
                do_sample=True,
                pad_token_id=self.tokenizer.pad_token_id
            )
        
        generated = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # 프롬프트 제거
        if prompt in generated:
            generated = generated.replace(prompt, "").strip()
        
        return generated
    
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
        if not questions:
            return {}
        
        scores = [q['quality_score'] for q in questions]
        
        return {
            'avg_score': sum(scores) / len(scores),
            'min_score': min(scores),
            'max_score': max(scores),
            'total': len(scores)
        }


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
            'model_name': 'beomi/SOLAR-10.7B-v1.0',
            'use_rag': True,
            'use_quantization': True,
            'num_questions': 100,
            'min_quality': 70,
            'temperature': 0.7,
            'prompt_template': 'prompts/cot.txt'
        }
    
    logger.info("="*60)
    logger.info("데이터 생성 시작")
    logger.info(f"설정: {json.dumps(config, indent=2, ensure_ascii=False)}")
    logger.info("="*60)
    
    # 생성기 초기화
    generator = DataGenerator(config)
    
    # 문제 생성
    start_time = time.time()
    questions = generator.generate_questions(
        num_questions=config.get('num_questions', 100)
    )
    elapsed_time = time.time() - start_time
    
    # 결과 저장
    output_path = generator.save_results(questions)
    
    # 결과 출력
    logger.info("="*60)
    logger.info("데이터 생성 완료!")
    logger.info(f"생성된 문제: {len(questions)}개")
    logger.info(f"소요 시간: {elapsed_time:.1f}초")
    logger.info(f"평균 품질: {generator._calculate_quality_stats(questions)['avg_score']:.1f}")
    logger.info(f"저장 위치: {output_path}")
    logger.info("="*60)
    
    return output_path


if __name__ == "__main__":
    # 테스트 실행
    config = {
        'model_name': 'beomi/SOLAR-10.7B-v1.0',
        'use_rag': True,
        'num_questions': 10,  # 테스트용 적은 수
        'prompt_template': 'prompts/cot.txt'
    }
    
    generate_data(config)