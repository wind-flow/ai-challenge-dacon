"""
데이터 증강 파이프라인
전체 증강 프로세스를 관리하는 통합 파이프라인
"""

import os
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional
from datetime import datetime
from tqdm import tqdm
import random
import pandas as pd

from .data_loader import FinancialDataLoader
from .knowledge_extractor import FinancialKnowledgeExtractor
from .question_generator import FSKUQuestionGenerator
from .quality_checker import DataQualityChecker

logger = logging.getLogger(__name__)


class FSKUAugmentationPipeline:
    """전체 증강 파이프라인 관리 클래스"""
    
    def __init__(self, config: Dict):
        """
        파이프라인 초기화
        
        Args:
            config: 설정 딕셔너리
        """
        self.config = config
        
        # 컴포넌트 초기화
        self.loader = FinancialDataLoader(config.get('data_dir', 'data/external'))
        self.extractor = FinancialKnowledgeExtractor()
        self.generator = FSKUQuestionGenerator(
            model_name=config.get('model_name', 'beomi/SOLAR-10.7B-v1.0'),
            use_quantization=config.get('use_quantization', True),
            use_cot=config.get('use_cot', True)  # CoT 기능 활성화
        )
        self.checker = DataQualityChecker(
            similarity_threshold=config.get('similarity_threshold', 0.85)
        )
        
        # 통계 정보
        self.statistics = {
            'documents_processed': 0,
            'concepts_extracted': 0,
            'questions_generated': 0,
            'questions_validated': 0,
            'time_elapsed': 0
        }
        
        # 생성된 데이터
        self.generated_data = []
        
        logger.info("Augmentation pipeline initialized")
    
    def run(self, target_count: int = 5000) -> Dict:
        """
        전체 파이프라인 실행
        
        Args:
            target_count: 목표 문제 수
            
        Returns:
            생성 결과 딕셔너리
        """
        start_time = datetime.now()
        logger.info(f"Starting augmentation pipeline with target: {target_count} questions")
        
        # 1단계: 데이터 로드
        logger.info("Step 1: Loading external data...")
        documents = self._load_data()
        
        # 2단계: 지식 추출
        logger.info("Step 2: Extracting knowledge...")
        knowledge_base = self._extract_knowledge(documents)
        
        # 3단계: 문제 생성
        logger.info("Step 3: Generating questions...")
        raw_questions = self._generate_questions(knowledge_base, target_count)
        
        # 4단계: 품질 검증
        logger.info("Step 4: Validating quality...")
        validated_questions = self._validate_questions(raw_questions)
        
        # 5단계: 증강 적용
        logger.info("Step 5: Applying augmentation strategies...")
        augmented_questions = self._apply_augmentation(validated_questions, target_count)
        
        # 6단계: 최종 검증
        logger.info("Step 6: Final validation...")
        final_questions = self._final_validation(augmented_questions)
        
        # 통계 업데이트
        self.statistics['time_elapsed'] = (datetime.now() - start_time).total_seconds()
        self.statistics['questions_validated'] = len(final_questions)
        
        # 결과 반환
        return {
            'data': final_questions,
            'statistics': self.statistics,
            'metadata': {
                'generated_at': datetime.now().isoformat(),
                'config': self.config,
                'target_count': target_count,
                'actual_count': len(final_questions)
            }
        }
    
    def _load_data(self) -> List[Dict]:
        """데이터 로드 단계"""
        documents = self.loader.parse_financial_documents()
        
        # 비상업적 라이선스만 필터링
        documents = self.loader.filter_by_license(commercial_use=False)
        
        self.statistics['documents_processed'] = len(documents)
        logger.info(f"Loaded {len(documents)} documents")
        
        return documents
    
    def _extract_knowledge(self, documents: List[Dict]) -> Dict:
        """지식 추출 단계"""
        all_concepts = []
        
        for doc in tqdm(documents, desc="Extracting concepts"):
            # 문서별 개념 추출
            concepts = self.extractor.extract_concepts(doc.get('content', ''))
            
            # 소스 정보 추가
            for concept in concepts:
                concept['source_document'] = doc.get('source', 'unknown')
                concept['license'] = doc.get('metadata', {}).get('license', {})
            
            all_concepts.extend(concepts)
        
        # 지식 그래프 구축
        knowledge_graph = self.extractor.build_knowledge_graph(documents)
        
        # QA 쌍 추출
        qa_pairs = []
        for doc in documents:
            pairs = self.extractor.identify_qa_pairs(doc.get('content', ''))
            qa_pairs.extend(pairs)
        
        self.statistics['concepts_extracted'] = len(all_concepts)
        logger.info(f"Extracted {len(all_concepts)} concepts and {len(qa_pairs)} QA pairs")
        
        return {
            'concepts': all_concepts,
            'knowledge_graph': knowledge_graph,
            'qa_pairs': qa_pairs
        }
    
    def _generate_questions(self, knowledge_base: Dict, target_count: int) -> List[Dict]:
        """문제 생성 단계"""
        questions = []
        concepts = knowledge_base['concepts']
        qa_pairs = knowledge_base['qa_pairs']
        
        # 목표 비율 설정
        multiple_choice_ratio = self.config.get('multiple_choice_ratio', 0.7)
        mc4_ratio = 0.5  # 4지선다 비율
        use_cot_ratio = self.config.get('use_cot_ratio', 0.3)  # CoT 사용 비율
        
        target_mc = int(target_count * multiple_choice_ratio)
        target_essay = target_count - target_mc
        target_mc4 = int(target_mc * mc4_ratio)
        target_mc5 = target_mc - target_mc4
        
        # CoT 사용 여부를 랜덤하게 결정
        use_cot_for_complex = lambda c: random.random() < use_cot_ratio or self._is_complex_concept(c)
        
        # 개념별로 문제 생성
        random.shuffle(concepts)
        
        # 객관식 4지선다 생성
        for concept in tqdm(concepts[:target_mc4], desc="Generating MC4"):
            try:
                # 복잡한 개념은 CoT 사용
                use_cot = use_cot_for_complex(concept)
                question = self.generator.generate_multiple_choice(
                    concept, 
                    num_options=4,
                    use_cot=use_cot
                )
                question['augmentation_method'] = 'cot_based' if use_cot else 'concept_based'
                question['created_at'] = datetime.now().isoformat()
                questions.append(question)
            except Exception as e:
                logger.warning(f"Failed to generate MC4: {e}")
        
        # 객관식 5지선다 생성
        for concept in tqdm(concepts[target_mc4:target_mc4+target_mc5], desc="Generating MC5"):
            try:
                # 복잡한 개념은 CoT 사용
                use_cot = use_cot_for_complex(concept)
                question = self.generator.generate_multiple_choice(
                    concept, 
                    num_options=5,
                    use_cot=use_cot
                )
                question['augmentation_method'] = 'cot_based' if use_cot else 'concept_based'
                question['created_at'] = datetime.now().isoformat()
                questions.append(question)
            except Exception as e:
                logger.warning(f"Failed to generate MC5: {e}")
        
        # 주관식 생성
        for concept in tqdm(concepts[target_mc:target_mc+target_essay], desc="Generating Essay"):
            try:
                # 주관식은 더 높은 확률로 CoT 사용
                use_cot = use_cot_for_complex(concept) or random.random() < 0.5
                question = self.generator.generate_essay_question(
                    concept,
                    use_cot=use_cot
                )
                question['augmentation_method'] = 'cot_based' if use_cot else 'concept_based'
                question['created_at'] = datetime.now().isoformat()
                questions.append(question)
            except Exception as e:
                logger.warning(f"Failed to generate essay: {e}")
        
        # QA 쌍에서 추가 문제 생성
        for q, a in qa_pairs[:min(len(qa_pairs), target_count//10)]:
            questions.append({
                'type': 'essay',
                'question': q,
                'answer': a,
                'keywords': self.extractor._extract_keywords(a),
                'augmentation_method': 'qa_pair',
                'created_at': datetime.now().isoformat()
            })
        
        self.statistics['questions_generated'] = len(questions)
        logger.info(f"Generated {len(questions)} raw questions")
        
        return questions
    
    def _validate_questions(self, questions: List[Dict]) -> List[Dict]:
        """품질 검증 단계"""
        validated, stats = self.checker.validate_batch(questions)
        
        logger.info(f"Validation stats: {stats}")
        
        return validated
    
    def _apply_augmentation(self, questions: List[Dict], target_count: int) -> List[Dict]:
        """증강 전략 적용"""
        augmented = questions.copy()
        
        # 목표 수에 도달할 때까지 증강
        strategies = ['difficulty_variation', 'paraphrase', 'scenario_application']
        
        while len(augmented) < target_count:
            # 랜덤하게 문제와 전략 선택
            base_question = random.choice(questions)
            strategy = random.choice(strategies)
            
            try:
                variations = self.generator.apply_augmentation_strategy(base_question, strategy)
                
                for var in variations:
                    var['augmentation_method'] = strategy
                    var['base_question_id'] = questions.index(base_question)
                    augmented.append(var)
                    
                    if len(augmented) >= target_count:
                        break
                        
            except Exception as e:
                logger.warning(f"Augmentation failed: {e}")
        
        logger.info(f"Augmented to {len(augmented)} questions")
        
        return augmented[:target_count]
    
    def _final_validation(self, questions: List[Dict]) -> List[Dict]:
        """최종 검증"""
        # 중복 제거
        unique_questions = self.checker.remove_duplicates(questions)
        
        # 최종 형식 검증
        final = []
        for q in unique_questions:
            if self.checker.validate_format(q):
                # ID 부여
                q['id'] = f"AUG_{len(final):05d}"
                final.append(q)
        
        self.generated_data = final
        
        return final
    
    def _is_complex_concept(self, concept: Dict) -> bool:
        """
        개념의 복잡도를 평가하여 CoT 사용 여부 결정
        
        Args:
            concept: 개념 정보
            
        Returns:
            복잡한 개념 여부
        """
        # 법령 관련
        if concept.get('type') == 'legal_article':
            return True
        
        # 프로세스 관련
        if concept.get('type') == 'process':
            return True
        
        # 긴 설명
        context = concept.get('context', '')
        if len(context) > 200:
            return True
        
        # 복잡한 키워드
        complex_keywords = ['규정', '법', '절차', '의무', '책임', '요건', '기준']
        for keyword in complex_keywords:
            if keyword in context:
                return True
        
        return False
    
    def save_results(self, output_dir: str):
        """
        결과 저장
        
        Args:
            output_dir: 출력 디렉토리
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # JSON 형식으로 저장
        json_file = output_path / f"augmented_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(json_file, 'w', encoding='utf-8') as f:
            json.dump(self.generated_data, f, ensure_ascii=False, indent=2)
        
        # JSONL 형식으로 저장
        jsonl_file = output_path / f"augmented_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jsonl"
        with open(jsonl_file, 'w', encoding='utf-8') as f:
            for item in self.generated_data:
                f.write(json.dumps(item, ensure_ascii=False) + '\n')
        
        # CSV 형식으로 저장 (객관식만)
        mc_questions = [q for q in self.generated_data if 'multiple_choice' in q.get('type', '')]
        if mc_questions:
            df = pd.DataFrame(mc_questions)
            csv_file = output_path / f"mc_questions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
            df.to_csv(csv_file, index=False, encoding='utf-8')
        
        # 통계 저장 (CoT 사용 통계 포함)
        cot_count = sum(1 for q in self.generated_data 
                       if q.get('generation_method') == 'cot' or 
                       q.get('augmentation_method') == 'cot_based')
        
        stats_file = output_path / f"statistics_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(stats_file, 'w', encoding='utf-8') as f:
            json.dump({
                'statistics': self.statistics,
                'cot_usage': {
                    'total_cot_generated': cot_count,
                    'cot_percentage': (cot_count / len(self.generated_data) * 100) if self.generated_data else 0
                },
                'quality_report': self.checker.generate_quality_report(self.generated_data)
            }, f, ensure_ascii=False, indent=2)
        
        logger.info(f"Results saved to {output_path}")
        logger.info(f"- Total questions: {len(self.generated_data)}")
        logger.info(f"- JSON file: {json_file.name}")
        logger.info(f"- JSONL file: {jsonl_file.name}")
        if mc_questions:
            logger.info(f"- CSV file: {csv_file.name}")
        logger.info(f"- Statistics: {stats_file.name}")
    
    def get_sample_output(self, n: int = 5) -> List[Dict]:
        """
        샘플 출력 반환
        
        Args:
            n: 샘플 수
            
        Returns:
            샘플 문제 리스트
        """
        if not self.generated_data:
            return []
        
        return random.sample(self.generated_data, min(n, len(self.generated_data)))


if __name__ == "__main__":
    # 테스트용 코드
    import sys
    from pathlib import Path
    sys.path.append(str(Path(__file__).parent.parent.parent))
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # 설정
    config = {
        'data_dir': 'data/external',
        'model_name': 'dummy',  # 테스트용
        'use_quantization': True,
        'use_cot': True,  # CoT 기능 활성화
        'use_cot_ratio': 0.3,  # 30% 확률로 CoT 사용
        'similarity_threshold': 0.85,
        'multiple_choice_ratio': 0.7
    }
    
    # 파이프라인 실행
    pipeline = FSKUAugmentationPipeline(config)
    
    # 작은 수로 테스트
    result = pipeline.run(target_count=10)
    
    print(f"\n파이프라인 실행 완료:")
    print(f"- 생성된 문제: {result['metadata']['actual_count']}개")
    print(f"- 소요 시간: {result['statistics']['time_elapsed']:.2f}초")
    print(f"- 처리된 문서: {result['statistics']['documents_processed']}개")
    print(f"- 추출된 개념: {result['statistics']['concepts_extracted']}개")
    
    # 샘플 출력
    samples = pipeline.get_sample_output(3)
    print(f"\n샘플 문제:")
    for i, sample in enumerate(samples, 1):
        print(f"\n{i}. {sample.get('question', 'N/A')}")
        if 'options' in sample:
            for j, opt in enumerate(sample['options'], 1):
                print(f"   {j}) {opt}")
            print(f"   정답: {sample.get('answer', 'N/A')}번")
    
    # 결과 저장
    pipeline.save_results('data/augmented')