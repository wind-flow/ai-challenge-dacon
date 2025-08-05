#!/usr/bin/env python3
"""
모델 학습 메인 모듈

LoRA/QLoRA를 사용한 효율적인 파인튜닝
"""

import json
import time
from pathlib import Path
from typing import Dict, List, Optional, Any
from datetime import datetime
import logging

import torch
from torch.utils.data import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
    TaskType
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FSKUDataset(Dataset):
    """FSKU 학습 데이터셋"""
    
    def __init__(self, data_path: str, tokenizer, max_length: int = 2048):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.data = self._load_data(data_path)
    
    def _load_data(self, data_path: str) -> List[Dict]:
        """데이터 로드"""
        path = Path(data_path)
        data = []
        
        if not path.exists():
            logger.warning(f"데이터 파일 없음: {data_path}")
            return []
        
        # JSONL 형식
        if path.suffix == '.jsonl':
            with open(path, 'r', encoding='utf-8') as f:
                for line in f:
                    if line.strip():
                        data.append(json.loads(line))
        
        # JSON 형식
        elif path.suffix == '.json':
            with open(path, 'r', encoding='utf-8') as f:
                loaded = json.load(f)
                if isinstance(loaded, list):
                    data = loaded
                elif 'questions' in loaded:
                    data = loaded['questions']
        
        logger.info(f"{len(data)}개 샘플 로드")
        return data
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        
        # 프롬프트 생성
        if 'question' in item:
            text = f"### 질문:\n{item['question']}\n\n### 답변:\n{item.get('answer', '')}"
        else:
            text = str(item)
        
        # 토크나이징
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].squeeze(),
            'attention_mask': encoding['attention_mask'].squeeze(),
            'labels': encoding['input_ids'].squeeze()
        }


class ModelTrainer:
    """모델 학습 클래스"""
    
    def __init__(self, config: Dict[str, Any]):
        """
        초기화
        
        Args:
            config: 학습 설정
                - base_model: 베이스 모델
                - use_lora: LoRA 사용 여부
                - use_qlora: QLoRA (4bit) 사용 여부
                - output_dir: 모델 저장 경로
        """
        self.config = config
        self.model = None
        self.tokenizer = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # 출력 디렉토리
        self.output_dir = Path(config.get('output_dir', 'models'))
        self.output_dir.mkdir(exist_ok=True)
    
    def setup_model(self):
        """모델 설정"""
        model_name = self.config['base_model']
        logger.info(f"모델 설정: {model_name}")
        
        # 토크나이저
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=True
        )
        
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # QLoRA 설정
        bnb_config = None
        if self.config.get('use_qlora', False):
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16
            )
            logger.info("QLoRA 4bit 양자화 활성화")
        
        # 모델 로드
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=bnb_config,
            torch_dtype=torch.float16 if not bnb_config else None,
            device_map="auto",
            trust_remote_code=True
        )
        
        # LoRA 설정
        if self.config.get('use_lora', False):
            if bnb_config:
                self.model = prepare_model_for_kbit_training(self.model)
            
            lora_config = LoraConfig(
                r=self.config.get('lora_r', 16),
                lora_alpha=self.config.get('lora_alpha', 32),
                target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
                lora_dropout=self.config.get('lora_dropout', 0.1),
                bias="none",
                task_type=TaskType.CAUSAL_LM
            )
            
            self.model = get_peft_model(self.model, lora_config)
            self.model.print_trainable_parameters()
            logger.info("LoRA 활성화")
    
    def train(self, train_data_path: str, valid_data_path: Optional[str] = None):
        """
        모델 학습
        
        Args:
            train_data_path: 학습 데이터 경로
            valid_data_path: 검증 데이터 경로 (선택)
        """
        logger.info("="*60)
        logger.info("학습 시작")
        logger.info("="*60)
        
        start_time = time.time()
        
        # 모델 설정
        self.setup_model()
        
        # 데이터셋 준비
        train_dataset = FSKUDataset(train_data_path, self.tokenizer)
        valid_dataset = None
        if valid_data_path:
            valid_dataset = FSKUDataset(valid_data_path, self.tokenizer)
        
        # 학습 인자
        training_args = TrainingArguments(
            output_dir=str(self.output_dir),
            num_train_epochs=self.config.get('num_epochs', 3),
            per_device_train_batch_size=self.config.get('batch_size', 4),
            gradient_accumulation_steps=self.config.get('gradient_accumulation', 4),
            warmup_steps=self.config.get('warmup_steps', 100),
            learning_rate=self.config.get('learning_rate', 2e-4),
            fp16=True,
            logging_steps=10,
            save_steps=500,
            eval_steps=100 if valid_dataset else None,
            evaluation_strategy="steps" if valid_dataset else "no",
            save_strategy="steps",
            load_best_model_at_end=True if valid_dataset else False,
            push_to_hub=False,
            report_to="none"
        )
        
        # 데이터 콜레이터
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer,
            mlm=False
        )
        
        # 트레이너
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=valid_dataset,
            data_collator=data_collator
        )
        
        # 학습
        train_result = trainer.train()
        
        # 모델 저장
        model_path = self.output_dir / f"model_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        trainer.save_model(str(model_path))
        self.tokenizer.save_pretrained(str(model_path))
        
        # 결과 저장
        elapsed_time = time.time() - start_time
        results = {
            'model_path': str(model_path),
            'train_loss': train_result.training_loss,
            'total_time': elapsed_time,
            'config': self.config,
            'timestamp': datetime.now().isoformat()
        }
        
        results_file = model_path / "training_results.json"
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        
        logger.info("="*60)
        logger.info(f"학습 완료!")
        logger.info(f"Loss: {train_result.training_loss:.4f}")
        logger.info(f"시간: {elapsed_time/60:.1f}분")
        logger.info(f"모델: {model_path}")
        logger.info("="*60)
        
        return results


def train_model(config: Dict[str, Any] = None):
    """
    모델 학습 실행 함수
    
    Args:
        config: 학습 설정
    """
    if config is None:
        # 기본 설정
        config = {
            'base_model': 'upstage/SOLAR-10.7B-v1.0',
            'use_lora': True,
            'use_qlora': True,
            'lora_r': 16,
            'lora_alpha': 32,
            'num_epochs': 3,
            'batch_size': 4,
            'learning_rate': 2e-4,
            'output_dir': 'models'
        }
    
    # 학습 데이터 경로
    train_data = config.get('train_data', 'data/augmented/train_data.jsonl')
    valid_data = config.get('valid_data', None)
    
    # 트레이너 초기화
    trainer = ModelTrainer(config)
    
    # 학습 실행
    results = trainer.train(train_data, valid_data)
    
    return results


def train_model(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    모델 학습 메인 함수 (외부 호출용)
    
    Args:
        config: 학습 설정
        
    Returns:
        학습 결과 (모델 경로 등)
    """
    trainer = ModelTrainer(
        base_model_name=config.get('base_model', 'beomi/SOLAR-10.7B-v1.0'),
        use_lora=config.get('use_lora', True),
        use_qlora=config.get('use_qlora', True),
        lora_r=config.get('lora_r', 16),
        lora_alpha=config.get('lora_alpha', 32),
        output_dir=config.get('output_dir', 'models')
    )
    
    results = trainer.train(
        train_data_path=config.get('train_data', 'data/augmented/train_data.jsonl'),
        valid_data_path=config.get('valid_data'),
        num_epochs=config.get('num_epochs', 3),
        batch_size=config.get('batch_size', 4),
        learning_rate=config.get('learning_rate', 2e-4)
    )
    
    return results


if __name__ == "__main__":
    # 테스트 실행
    config = {
        'base_model': 'upstage/SOLAR-10.7B-v1.0',
        'use_lora': True,
        'use_qlora': True,
        'num_epochs': 1,  # 테스트용
        'batch_size': 2
    }
    
    train_model(config)