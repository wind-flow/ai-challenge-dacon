#!/usr/bin/env python3
"""
Colab용 대량 데이터 생성 스크립트

고성능 GPU에서 최고 품질의 데이터를 대량으로 생성합니다.
"""

# Colab 환경 체크
import sys
try:
    import google.colab
    IN_COLAB = True
    print("✅ Google Colab 환경 감지")
except:
    IN_COLAB = False
    print("⚠️ 로컬 환경에서 실행 중")

# 필수 패키지 설치
if IN_COLAB:
    print("\n📦 필수 패키지 설치 중...")
    !pip install -q transformers accelerate sentencepiece
    !pip install -q torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
    !pip install -q datasets tqdm

import torch
import json
import time
from pathlib import Path
from datetime import datetime
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

# GPU 정보 출력
if torch.cuda.is_available():
    gpu_name = torch.cuda.get_device_name(0)
    gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
    print(f"\n🚀 GPU 감지: {gpu_name} ({gpu_memory:.1f}GB)")
else:
    print("\n⚠️ GPU를 사용할 수 없습니다!")

class ColabDataGenerator:
    """Colab에서 대량 데이터 생성"""
    
    def __init__(self, model_name="Qwen/Qwen2.5-32B-Instruct"):
        """
        초기화
        
        Args:
            model_name: 사용할 모델 (기본: Qwen 32B)
        """
        self.model_name = model_name
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = None
        self.tokenizer = None
        
    def load_model(self):
        """모델 로드 (4-bit 양자화)"""
        print(f"\n🔄 모델 로딩: {self.model_name}")
        
        # 토크나이저
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            trust_remote_code=True
        )
        
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # 큰 모델은 4-bit 양자화
        if "32B" in self.model_name or "14B" in self.model_name:
            from transformers import BitsAndBytesConfig
            
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4"
            )
            
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                quantization_config=bnb_config,
                device_map="auto",
                trust_remote_code=True
            )
            print("✅ 4-bit 양자화 모델 로드 완료")
        else:
            # 작은 모델은 FP16
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                torch_dtype=torch.float16,
                device_map="auto",
                trust_remote_code=True
            )
            print("✅ FP16 모델 로드 완료")
    
    def generate_questions(self, num_questions=10000, batch_size=4):
        """
        대량 문제 생성
        
        Args:
            num_questions: 생성할 문제 수
            batch_size: 배치 크기
            
        Returns:
            생성된 문제 리스트
        """
        if self.model is None:
            self.load_model()
        
        questions = []
        concepts = self._get_finance_concepts()
        
        print(f"\n📊 {num_questions}개 문제 생성 시작...")
        print(f"배치 크기: {batch_size}")
        
        with tqdm(total=num_questions) as pbar:
            while len(questions) < num_questions:
                # 배치 생성
                batch_prompts = []
                for _ in range(min(batch_size, num_questions - len(questions))):
                    concept = concepts[len(questions) % len(concepts)]
                    prompt = self._create_prompt(concept)
                    batch_prompts.append(prompt)
                
                # 배치 처리
                try:
                    generated = self._generate_batch(batch_prompts)
                    
                    for i, text in enumerate(generated):
                        parsed = self._parse_question(text)
                        if parsed and self._check_quality(parsed):
                            questions.append({
                                'id': f"COLAB_{len(questions)+1:05d}",
                                'concept': concepts[len(questions) % len(concepts)],
                                'question': parsed['question'],
                                'choices': parsed.get('choices', []),
                                'answer': parsed['answer'],
                                'explanation': parsed.get('explanation', ''),
                                'quality_score': self._calculate_quality(parsed),
                                'timestamp': datetime.now().isoformat()
                            })
                            pbar.update(1)
                
                except Exception as e:
                    print(f"\n⚠️ 생성 오류: {e}")
                    continue
        
        return questions
    
    def _create_prompt(self, concept):
        """프롬프트 생성"""
        return f"""당신은 한국 금융 전문가입니다. 다음 개념에 대한 고품질 FSKU 문제를 생성하세요.

금융 개념: {concept}

요구사항:
1. 한국 금융 실무와 직접 관련된 내용
2. 명확하고 모호하지 않은 문제
3. 4지선다 객관식 또는 서술형
4. 실제 금융 전문가가 알아야 할 내용

[문제]
(여기에 문제 작성)

[선택지] (객관식인 경우)
1) 
2) 
3) 
4) 

[정답]
(정답 번호 또는 서술형 답변)

[해설]
(왜 이것이 정답인지 설명)"""

    def _generate_batch(self, prompts):
        """배치 생성"""
        inputs = self.tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=1024
        ).to(self.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=512,
                temperature=0.8,
                top_p=0.9,
                do_sample=True,
                pad_token_id=self.tokenizer.pad_token_id
            )
        
        generated_texts = []
        for output in outputs:
            text = self.tokenizer.decode(output, skip_special_tokens=True)
            # 프롬프트 제거
            for prompt in prompts:
                if prompt in text:
                    text = text.replace(prompt, "").strip()
                    break
            generated_texts.append(text)
        
        return generated_texts
    
    def _parse_question(self, text):
        """생성된 텍스트 파싱"""
        import re
        
        result = {
            'question': '',
            'choices': [],
            'answer': '',
            'explanation': ''
        }
        
        # 문제 추출
        question_match = re.search(r'\[문제\](.*?)(?:\[|$)', text, re.DOTALL)
        if question_match:
            result['question'] = question_match.group(1).strip()
        
        # 선택지 추출
        choices_match = re.search(r'\[선택지\](.*?)(?:\[|$)', text, re.DOTALL)
        if choices_match:
            choices_text = choices_match.group(1).strip()
            choice_pattern = r'([1-4])\)\s*(.+?)(?=(?:[1-4]\)|$))'
            choices = re.findall(choice_pattern, choices_text, re.DOTALL)
            result['choices'] = [f"{num}) {text.strip()}" for num, text in choices]
        
        # 정답 추출
        answer_match = re.search(r'\[정답\](.*?)(?:\[|$)', text, re.DOTALL)
        if answer_match:
            result['answer'] = answer_match.group(1).strip()
        
        # 해설 추출
        explanation_match = re.search(r'\[해설\](.*?)(?:\[|$)', text, re.DOTALL)
        if explanation_match:
            result['explanation'] = explanation_match.group(1).strip()
        
        return result if result['question'] and result['answer'] else None
    
    def _check_quality(self, parsed):
        """품질 검증"""
        # 기본 품질 체크
        if len(parsed['question']) < 20:
            return False
        
        if parsed['choices'] and len(parsed['choices']) < 4:
            return False
        
        if not parsed['answer']:
            return False
        
        return True
    
    def _calculate_quality(self, parsed):
        """품질 점수 계산"""
        score = 70  # 기본 점수
        
        # 문제 길이
        if len(parsed['question']) > 100:
            score += 10
        
        # 해설 유무
        if parsed['explanation'] and len(parsed['explanation']) > 50:
            score += 10
        
        # 선택지 품질
        if parsed['choices'] and all(len(c) > 20 for c in parsed['choices']):
            score += 10
        
        return min(score, 100)
    
    def _get_finance_concepts(self):
        """금융 개념 리스트"""
        return [
            "전자금융거래", "금융보안", "개인정보보호", "암호기술",
            "블록체인", "핀테크", "오픈뱅킹", "마이데이터",
            "인증서", "바이오인증", "이상거래탐지", "자금세탁방지",
            "신용평가", "리스크관리", "내부통제", "컴플라이언스",
            "금융사고", "보안취약점", "침해사고대응", "재해복구",
            "클라우드보안", "API보안", "모바일보안", "AI보안"
        ]
    
    def save_results(self, questions, output_path="generated_data_colab.jsonl"):
        """결과 저장"""
        # 품질순 정렬
        questions.sort(key=lambda x: x['quality_score'], reverse=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            for q in questions:
                f.write(json.dumps(q, ensure_ascii=False) + '\n')
        
        print(f"\n✅ {len(questions)}개 문제 저장 완료: {output_path}")
        
        # 통계 출력
        avg_quality = sum(q['quality_score'] for q in questions) / len(questions)
        print(f"📊 평균 품질 점수: {avg_quality:.1f}")
        print(f"📊 90점 이상: {sum(1 for q in questions if q['quality_score'] >= 90)}개")
        print(f"📊 80점 이상: {sum(1 for q in questions if q['quality_score'] >= 80)}개")

def main():
    """메인 실행"""
    print("="*60)
    print("🚀 Colab 대량 데이터 생성")
    print("="*60)
    
    # 모델 선택
    print("\n모델 선택:")
    print("1. Qwen2.5-32B (최고 품질) ⭐")
    print("2. Qwen2.5-14B (균형)")
    print("3. EXAONE-3.0-7.8B (한국 특화)")
    print("4. SOLAR-10.7B (검증된 성능)")
    
    choice = input("\n선택 [1]: ").strip() or "1"
    
    model_map = {
        "1": "Qwen/Qwen2.5-32B-Instruct",
        "2": "Qwen/Qwen2.5-14B-Instruct", 
        "3": "LG-AI-EXAONE/EXAONE-3.0-7.8B-Instruct",
        "4": "upstage/SOLAR-10.7B-v1.0"
    }
    
    model_name = model_map.get(choice, model_map["1"])
    
    # 생성 개수
    num_questions = int(input("생성할 문제 수 [10000]: ").strip() or "10000")
    
    # 생성기 초기화
    generator = ColabDataGenerator(model_name)
    
    # 생성 시작
    start_time = time.time()
    questions = generator.generate_questions(num_questions)
    elapsed = time.time() - start_time
    
    # 결과 저장
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = f"colab_data_{timestamp}.jsonl"
    generator.save_results(questions, output_file)
    
    print(f"\n⏱️ 총 소요 시간: {elapsed/60:.1f}분")
    print(f"⚡ 평균 생성 속도: {len(questions)/elapsed:.1f} 문제/초")
    
    # Google Drive 저장 (Colab인 경우)
    if IN_COLAB:
        from google.colab import drive
        drive.mount('/content/drive')
        
        save_path = f"/content/drive/MyDrive/FSKU/{output_file}"
        Path("/content/drive/MyDrive/FSKU").mkdir(exist_ok=True)
        
        import shutil
        shutil.copy(output_file, save_path)
        print(f"\n💾 Google Drive 저장 완료: {save_path}")

if __name__ == "__main__":
    main()