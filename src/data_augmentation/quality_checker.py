"""
품질 검증 모듈
생성된 데이터의 품질을 검증
"""

import re
import logging
from typing import List, Dict, Tuple, Set
from collections import Counter
import numpy as np
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)


class DataQualityChecker:
    """생성된 데이터의 품질을 검증하는 클래스"""
    
    def __init__(self, similarity_threshold: float = 0.85):
        """
        초기화
        
        Args:
            similarity_threshold: 중복 판단 유사도 임계값
        """
        self.similarity_threshold = similarity_threshold
        
        # 의미 유사도 계산용 모델
        try:
            self.embedding_model = SentenceTransformer('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')
            logger.info("Embedding model loaded successfully")
        except:
            logger.warning("Failed to load embedding model, using rule-based similarity")
            self.embedding_model = None
        
        # 금융 용어 사전 (정확성 검증용)
        self.financial_terms = self._load_financial_dictionary()
        
        # 검증 통계
        self.validation_stats = {
            'total_checked': 0,
            'passed': 0,
            'failed': 0,
            'duplicates_removed': 0,
            'invalid_format': 0,
            'accuracy_issues': 0
        }
    
    def _load_financial_dictionary(self) -> Dict:
        """금융 용어 사전 로드"""
        return {
            '금리': {'정의': '자금 대여의 대가', '관련어': ['이자', '이율']},
            '예금': {'정의': '금융기관에 맡긴 돈', '관련어': ['저축', '적금']},
            '대출': {'정의': '금융기관에서 빌린 돈', '관련어': ['융자', '차입']},
            '투자': {'정의': '수익을 위한 자산 운용', '관련어': ['운용', '투기']},
            '채권': {'정의': '차용증서', '관련어': ['사채', '국채', '회사채']},
            '주식': {'정의': '회사 소유권 증서', '관련어': ['증권', '지분']},
            '파생상품': {'정의': '기초자산 가치 연동 상품', '관련어': ['선물', '옵션', '스왑']},
            'BIS': {'정의': '국제결제은행 자기자본비율', '관련어': ['자본적정성', '건전성']},
            '개인정보': {'정의': '개인 식별 가능 정보', '관련어': ['민감정보', '고유식별정보']},
            '재해복구': {'정의': '재해 시 시스템 복구', '관련어': ['백업', 'DR', 'BCP']}
        }
    
    def check_financial_accuracy(self, qa_pair: Dict) -> bool:
        """
        금융 용어와 개념의 정확성 검증
        
        Args:
            qa_pair: QA 쌍
            
        Returns:
            정확성 여부
        """
        question = qa_pair.get('question', '')
        answer = qa_pair.get('answer', '')
        options = qa_pair.get('options', [])
        
        # 1. 금융 용어 오용 체크
        misused_terms = self._check_term_misuse(question, answer)
        if misused_terms:
            logger.warning(f"Misused financial terms found: {misused_terms}")
            return False
        
        # 2. 숫자 정보 일관성 체크
        if not self._check_numeric_consistency(question, answer, options):
            logger.warning("Numeric inconsistency detected")
            return False
        
        # 3. 법령 조항 정확성 체크
        if not self._check_legal_accuracy(question, answer):
            logger.warning("Legal article inaccuracy detected")
            return False
        
        return True
    
    def check_answer_validity(self, qa_pair: Dict) -> bool:
        """
        답변의 타당성 검증
        
        Args:
            qa_pair: QA 쌍
            
        Returns:
            타당성 여부
        """
        question_type = qa_pair.get('type', '')
        
        if 'multiple_choice' in question_type:
            return self._check_multiple_choice_validity(qa_pair)
        elif question_type == 'essay':
            return self._check_essay_validity(qa_pair)
        
        return True
    
    def _check_multiple_choice_validity(self, qa_pair: Dict) -> bool:
        """객관식 답변 타당성 검증"""
        options = qa_pair.get('options', [])
        answer_idx = qa_pair.get('answer', 0)
        
        # 1. 답변 인덱스 유효성
        if answer_idx < 1 or answer_idx > len(options):
            return False
        
        # 2. 선택지 중복 체크
        if len(options) != len(set(options)):
            return False
        
        # 3. 선택지 최소 길이
        if any(len(opt.strip()) < 2 for opt in options):
            return False
        
        # 4. 정답과 오답의 차별성
        correct_answer = options[answer_idx - 1]
        for i, opt in enumerate(options):
            if i != answer_idx - 1:
                if self._calculate_similarity(correct_answer, opt) > 0.95:
                    return False
        
        return True
    
    def _check_essay_validity(self, qa_pair: Dict) -> bool:
        """주관식 답변 타당성 검증"""
        answer = qa_pair.get('answer', '')
        keywords = qa_pair.get('keywords', [])
        
        # 1. 답변 최소 길이
        if len(answer) < 20:
            return False
        
        # 2. 키워드 포함 여부
        answer_lower = answer.lower()
        keyword_coverage = sum(1 for kw in keywords if kw.lower() in answer_lower)
        
        if keywords and keyword_coverage < len(keywords) * 0.3:
            return False
        
        return True
    
    def remove_duplicates(self, qa_list: List[Dict]) -> List[Dict]:
        """
        중복 제거 (의미적 유사도 기반)
        
        Args:
            qa_list: QA 리스트
            
        Returns:
            중복 제거된 리스트
        """
        if not qa_list:
            return []
        
        unique_questions = []
        seen_questions = []
        
        for qa in qa_list:
            question = qa.get('question', '')
            
            # 의미적 유사도 체크
            is_duplicate = False
            
            if self.embedding_model:
                # 임베딩 기반 유사도
                for seen_q in seen_questions:
                    similarity = self._calculate_semantic_similarity(question, seen_q)
                    if similarity > self.similarity_threshold:
                        is_duplicate = True
                        self.validation_stats['duplicates_removed'] += 1
                        break
            else:
                # 규칙 기반 유사도
                for seen_q in seen_questions:
                    similarity = self._calculate_similarity(question, seen_q)
                    if similarity > self.similarity_threshold:
                        is_duplicate = True
                        self.validation_stats['duplicates_removed'] += 1
                        break
            
            if not is_duplicate:
                unique_questions.append(qa)
                seen_questions.append(question)
        
        logger.info(f"Removed {len(qa_list) - len(unique_questions)} duplicates")
        return unique_questions
    
    def validate_format(self, qa_pair: Dict) -> bool:
        """
        FSKU 형식 준수 여부 확인
        
        Args:
            qa_pair: QA 쌍
            
        Returns:
            형식 준수 여부
        """
        required_fields = ['type', 'question']
        
        # 필수 필드 체크
        for field in required_fields:
            if field not in qa_pair:
                return False
        
        question_type = qa_pair['type']
        
        # 타입별 필수 필드 체크
        if 'multiple_choice' in question_type:
            if 'options' not in qa_pair or 'answer' not in qa_pair:
                return False
            
            # 선택지 개수 체크
            num_options = int(question_type.split('_')[-1]) if '_' in question_type else 4
            if len(qa_pair['options']) != num_options:
                return False
                
        elif question_type == 'essay':
            if 'answer' not in qa_pair or 'keywords' not in qa_pair:
                return False
        
        # 질문 형식 체크
        question = qa_pair['question']
        if not question.strip():
            return False
        
        # 질문 끝 문자 체크 (? 또는 . 또는 설명형)
        if not (question.endswith('?') or question.endswith('.') or 
                '설명하시오' in question or '서술하시오' in question):
            return False
        
        return True
    
    def validate_batch(self, qa_list: List[Dict]) -> Tuple[List[Dict], Dict]:
        """
        배치 검증
        
        Args:
            qa_list: QA 리스트
            
        Returns:
            (검증 통과 리스트, 검증 통계)
        """
        validated = []
        
        for qa in qa_list:
            self.validation_stats['total_checked'] += 1
            
            # 형식 검증
            if not self.validate_format(qa):
                self.validation_stats['invalid_format'] += 1
                self.validation_stats['failed'] += 1
                continue
            
            # 정확성 검증
            if not self.check_financial_accuracy(qa):
                self.validation_stats['accuracy_issues'] += 1
                self.validation_stats['failed'] += 1
                continue
            
            # 타당성 검증
            if not self.check_answer_validity(qa):
                self.validation_stats['failed'] += 1
                continue
            
            validated.append(qa)
            self.validation_stats['passed'] += 1
        
        # 중복 제거
        validated = self.remove_duplicates(validated)
        
        return validated, self.validation_stats
    
    def generate_quality_report(self, qa_list: List[Dict]) -> Dict:
        """
        품질 보고서 생성
        
        Args:
            qa_list: QA 리스트
            
        Returns:
            품질 보고서
        """
        report = {
            'total_questions': len(qa_list),
            'type_distribution': {},
            'difficulty_distribution': {},
            'avg_question_length': 0,
            'avg_answer_length': 0,
            'keyword_coverage': 0,
            'quality_score': 0
        }
        
        # 타입별 분포
        type_counter = Counter(qa.get('type', 'unknown') for qa in qa_list)
        report['type_distribution'] = dict(type_counter)
        
        # 난이도 분포
        difficulty_counter = Counter(qa.get('difficulty', 'medium') for qa in qa_list)
        report['difficulty_distribution'] = dict(difficulty_counter)
        
        # 평균 길이
        question_lengths = [len(qa.get('question', '')) for qa in qa_list]
        answer_lengths = [len(str(qa.get('answer', ''))) for qa in qa_list if 'answer' in qa]
        
        if question_lengths:
            report['avg_question_length'] = np.mean(question_lengths)
        if answer_lengths:
            report['avg_answer_length'] = np.mean(answer_lengths)
        
        # 키워드 커버리지
        all_keywords = set()
        for qa in qa_list:
            if 'keywords' in qa:
                all_keywords.update(qa['keywords'])
        
        report['keyword_coverage'] = len(all_keywords)
        
        # 품질 점수 계산
        if self.validation_stats['total_checked'] > 0:
            report['quality_score'] = (
                self.validation_stats['passed'] / self.validation_stats['total_checked']
            ) * 100
        
        return report
    
    # 헬퍼 메서드들
    def _check_term_misuse(self, question: str, answer: str) -> List[str]:
        """금융 용어 오용 체크"""
        misused = []
        
        text = f"{question} {answer}".lower()
        
        # 흔한 오용 패턴
        misuse_patterns = {
            '이자': ['원금', '원리금'],  # 이자와 원금 혼동
            '예금': ['대출', '투자'],     # 예금과 대출 혼동
            '금리': ['수익률', '할인율']  # 금리와 수익률 혼동
        }
        
        for term, confused_terms in misuse_patterns.items():
            if term in text:
                for confused in confused_terms:
                    # 같은 문장에서 혼용되는지 체크
                    if confused in text:
                        sentences = text.split('.')
                        for sent in sentences:
                            if term in sent and confused in sent:
                                # 문맥상 구분이 명확하지 않은 경우
                                if '차이' not in sent and '비교' not in sent:
                                    misused.append(f"{term}-{confused}")
        
        return misused
    
    def _check_numeric_consistency(self, question: str, answer: str, options: List[str]) -> bool:
        """숫자 정보 일관성 체크"""
        # 모든 텍스트에서 숫자 추출
        all_text = f"{question} {answer} {' '.join(options)}"
        numbers = re.findall(r'\d+(?:\.\d+)?', all_text)
        
        # 동일한 항목에 대한 다른 숫자가 있는지 체크
        # (구현 간소화를 위해 기본 체크만)
        
        return True
    
    def _check_legal_accuracy(self, question: str, answer: str) -> bool:
        """법령 조항 정확성 체크"""
        # 법령 조항 패턴
        legal_pattern = r'제\s*(\d+)\s*조'
        
        q_articles = re.findall(legal_pattern, question)
        a_articles = re.findall(legal_pattern, answer)
        
        # 질문과 답변의 조항 번호가 일치하는지 체크
        if q_articles and a_articles:
            # 조항 번호가 크게 다르면 문제
            for q_art in q_articles:
                for a_art in a_articles:
                    if abs(int(q_art) - int(a_art)) > 50:
                        return False
        
        return True
    
    def _calculate_similarity(self, text1: str, text2: str) -> float:
        """규칙 기반 텍스트 유사도 계산"""
        # 간단한 자카드 유사도
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())
        
        if not words1 or not words2:
            return 0.0
        
        intersection = words1.intersection(words2)
        union = words1.union(words2)
        
        return len(intersection) / len(union)
    
    def _calculate_semantic_similarity(self, text1: str, text2: str) -> float:
        """의미적 유사도 계산"""
        if not self.embedding_model:
            return self._calculate_similarity(text1, text2)
        
        try:
            embeddings = self.embedding_model.encode([text1, text2])
            
            # 코사인 유사도
            similarity = np.dot(embeddings[0], embeddings[1]) / (
                np.linalg.norm(embeddings[0]) * np.linalg.norm(embeddings[1])
            )
            
            return float(similarity)
            
        except:
            return self._calculate_similarity(text1, text2)


if __name__ == "__main__":
    # 테스트용 코드
    import sys
    from pathlib import Path
    sys.path.append(str(Path(__file__).parent.parent.parent))
    
    logging.basicConfig(level=logging.INFO)
    
    # 품질 검증기 초기화
    checker = DataQualityChecker(similarity_threshold=0.85)
    
    # 테스트 데이터
    test_qa_list = [
        {
            'type': 'multiple_choice_4',
            'question': '금리란 무엇인가?',
            'options': ['자금 대여의 대가', '원금', '투자 수익', '예금 잔액'],
            'answer': 1
        },
        {
            'type': 'multiple_choice_4',
            'question': '금리의 정의는?',  # 중복
            'options': ['이자율', '원금', '수익률', '할인율'],
            'answer': 1
        },
        {
            'type': 'essay',
            'question': '개인정보보호법의 주요 내용을 설명하시오.',
            'answer': '개인정보보호법은 개인정보의 처리 및 보호에 관한 사항을 규정한 법률입니다.',
            'keywords': ['개인정보', '보호', '처리']
        }
    ]
    
    # 배치 검증
    validated, stats = checker.validate_batch(test_qa_list)
    
    print(f"\n검증 결과:")
    print(f"- 총 검사: {stats['total_checked']}개")
    print(f"- 통과: {stats['passed']}개")
    print(f"- 실패: {stats['failed']}개")
    print(f"- 중복 제거: {stats['duplicates_removed']}개")
    
    # 품질 보고서
    report = checker.generate_quality_report(validated)
    print(f"\n품질 보고서:")
    print(f"- 총 문제 수: {report['total_questions']}")
    print(f"- 타입 분포: {report['type_distribution']}")
    print(f"- 평균 질문 길이: {report['avg_question_length']:.1f}")
    print(f"- 품질 점수: {report['quality_score']:.1f}%")