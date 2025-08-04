#!/usr/bin/env python3
"""
모델 추론 모듈

학습된 모델로 문제 해결 및 평가
"""

import json
import time
from pathlib import Path
from typing import Dict, List, Optional, Any
import logging

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import pandas as pd
from tqdm import tqdm

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ModelInference:
    """모델 추론 클래스"""
    
    def __init__(self, model_path: str, base_model: Optional[str] = None):
        """
        초기화
        
        Args:
            model_path: 학습된 모델 경로
            base_model: 베이스 모델 (LoRA의 경우 필요)
        """
        self.model_path = Path(model_path)
        self.base_model = base_model
        self.model = None
        self.tokenizer = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # 모델 로드
        self._load_model()
    
    def _load_model(self):
        """모델 로드"""
        logger.info(f"모델 로딩: {self.model_path}")
        
        # 토크나이저 로드
        self.tokenizer = AutoTokenizer.from_pretrained(
            str(self.model_path),
            trust_remote_code=True
        )
        
        # LoRA 모델인지 확인
        adapter_config_path = self.model_path / "adapter_config.json"
        
        if adapter_config_path.exists():
            # LoRA 모델
            if not self.base_model:
                # adapter_config에서 base_model 읽기
                with open(adapter_config_path, 'r') as f:
                    config = json.load(f)
                    self.base_model = config.get('base_model_name_or_path')
            
            logger.info(f"LoRA 모델 로드 (베이스: {self.base_model})")
            
            # 베이스 모델 로드
            base_model = AutoModelForCausalLM.from_pretrained(
                self.base_model,
                torch_dtype=torch.float16,
                device_map="auto",
                trust_remote_code=True
            )
            
            # LoRA 어댑터 로드
            self.model = PeftModel.from_pretrained(
                base_model,
                str(self.model_path)
            )
        else:
            # 일반 모델
            self.model = AutoModelForCausalLM.from_pretrained(
                str(self.model_path),
                torch_dtype=torch.float16,
                device_map="auto",
                trust_remote_code=True
            )
        
        logger.info("모델 로드 완료")
    
    def generate(self, prompt: str, **kwargs) -> str:
        """
        텍스트 생성
        
        Args:
            prompt: 입력 프롬프트
            **kwargs: 생성 파라미터
            
        Returns:
            생성된 텍스트
        """
        # 토크나이징
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=kwargs.get('max_length', 2048)
        )
        
        if self.device == "cuda":
            inputs = inputs.to(self.device)
        
        # 생성
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=kwargs.get('max_new_tokens', 256),
                temperature=kwargs.get('temperature', 0.7),
                top_p=kwargs.get('top_p', 0.9),
                do_sample=kwargs.get('do_sample', True),
                pad_token_id=self.tokenizer.pad_token_id
            )
        
        # 디코딩
        generated = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # 프롬프트 제거
        if prompt in generated:
            generated = generated.replace(prompt, "").strip()
        
        return generated
    
    def solve_questions(self, questions: List[Dict]) -> List[Dict]:
        """
        문제 일괄 해결
        
        Args:
            questions: 문제 리스트
            
        Returns:
            답변이 포함된 결과 리스트
        """
        results = []
        
        for q in tqdm(questions, desc="문제 해결"):
            # 프롬프트 생성
            if 'question' in q:
                prompt = f"### 질문:\n{q['question']}\n\n### 답변:\n"
            else:
                prompt = str(q)
            
            # 답변 생성
            answer = self.generate(prompt)
            
            # 결과 저장
            result = q.copy()
            result['generated_answer'] = answer
            results.append(result)
        
        return results
    
    def evaluate_test_data(self, test_file: str) -> pd.DataFrame:
        """
        테스트 데이터 평가
        
        Args:
            test_file: 테스트 파일 경로
            
        Returns:
            평가 결과 DataFrame
        """
        # 테스트 데이터 로드
        test_path = Path(test_file)
        
        if test_path.suffix == '.csv':
            df = pd.read_csv(test_path)
        elif test_path.suffix == '.json':
            with open(test_path, 'r') as f:
                data = json.load(f)
            df = pd.DataFrame(data)
        else:
            logger.error(f"지원하지 않는 파일 형식: {test_path.suffix}")
            return pd.DataFrame()
        
        # 답변 생성
        answers = []
        
        for idx, row in tqdm(df.iterrows(), total=len(df), desc="평가"):
            question = row.get('Question', row.get('question', ''))
            
            if question:
                prompt = f"### 질문:\n{question}\n\n### 답변:\n"
                answer = self.generate(prompt, temperature=0.3)  # 낮은 temperature로 일관성
                answers.append(answer)
            else:
                answers.append("")
        
        # 결과 추가
        df['generated_answer'] = answers
        
        return df
    
    def batch_inference(self, prompts: List[str], batch_size: int = 8) -> List[str]:
        """
        배치 추론 (메모리 효율적)
        
        Args:
            prompts: 프롬프트 리스트
            batch_size: 배치 크기
            
        Returns:
            생성된 텍스트 리스트
        """
        results = []
        
        for i in range(0, len(prompts), batch_size):
            batch = prompts[i:i+batch_size]
            
            # 배치 토크나이징
            inputs = self.tokenizer(
                batch,
                return_tensors="pt",
                truncation=True,
                padding=True,
                max_length=2048
            )
            
            if self.device == "cuda":
                inputs = inputs.to(self.device)
            
            # 배치 생성
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=256,
                    temperature=0.7,
                    pad_token_id=self.tokenizer.pad_token_id
                )
            
            # 디코딩
            for output in outputs:
                generated = self.tokenizer.decode(output, skip_special_tokens=True)
                results.append(generated)
        
        return results


def run_inference(
    model_path: str,
    test_file: str,
    output_file: str = "inference_results.csv"
):
    """
    추론 실행
    
    Args:
        model_path: 모델 경로
        test_file: 테스트 파일
        output_file: 결과 저장 파일
    """
    logger.info("="*60)
    logger.info("추론 시작")
    logger.info(f"모델: {model_path}")
    logger.info(f"테스트: {test_file}")
    logger.info("="*60)
    
    # 추론기 초기화
    inferencer = ModelInference(model_path)
    
    # 평가 실행
    start_time = time.time()
    results_df = inferencer.evaluate_test_data(test_file)
    elapsed_time = time.time() - start_time
    
    # 결과 저장
    output_path = Path("results") / output_file
    output_path.parent.mkdir(exist_ok=True)
    results_df.to_csv(output_path, index=False, encoding='utf-8-sig')
    
    logger.info("="*60)
    logger.info(f"추론 완료!")
    logger.info(f"처리 문제: {len(results_df)}개")
    logger.info(f"소요 시간: {elapsed_time:.1f}초")
    logger.info(f"결과 저장: {output_path}")
    logger.info("="*60)
    
    return str(output_path)


if __name__ == "__main__":
    # 테스트
    import sys
    
    if len(sys.argv) > 2:
        model_path = sys.argv[1]
        test_file = sys.argv[2]
        run_inference(model_path, test_file)
    else:
        print("사용법: python inference.py <model_path> <test_file>")
        
        # 예시 실행
        # run_inference("models/model_20250104", "data/test.csv")