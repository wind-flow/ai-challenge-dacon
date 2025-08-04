"""
지식 추출 모듈
금융 문서에서 핵심 개념과 지식을 추출
"""

import re
import logging
from typing import List, Dict, Tuple, Optional
from collections import defaultdict

logger = logging.getLogger(__name__)


class FinancialKnowledgeExtractor:
    """금융 문서에서 핵심 개념과 지식을 추출하는 클래스"""
    
    def __init__(self):
        """초기화"""
        # 금융 용어 패턴
        self.financial_terms = self._load_financial_terms()
        
        # 법령 조항 패턴
        self.law_patterns = [
            r'제\s*(\d+)\s*조\s*[\(（]?([^)）]*?)[\)）]?',  # 제N조
            r'제\s*(\d+)\s*항',  # 제N항
            r'제\s*(\d+)\s*호',  # 제N호
            r'별표\s*(\d+)',  # 별표N
        ]
        
        # 숫자/수치 정보 패턴
        self.numeric_patterns = [
            r'(\d+(?:\.\d+)?)\s*(%|퍼센트|프로)',  # 백분율
            r'(\d+(?:,\d{3})*(?:\.\d+)?)\s*(원|달러|유로|엔)',  # 금액
            r'(\d+)\s*(일|개월|년|년간|연간)',  # 기간
            r'(\d+)\s*(명|개|건|회)',  # 수량
        ]
        
        # 프로세스/절차 키워드
        self.process_keywords = [
            '절차', '단계', '프로세스', '과정', '방법',
            '신청', '심사', '승인', '평가', '검토',
            '제출', '보고', '통지', '공시', '신고'
        ]
    
    def _load_financial_terms(self) -> List[str]:
        """
        금융 전문 용어 목록 로드
        
        Returns:
            금융 용어 리스트
        """
        # 주요 금융 용어 (확장 가능)
        terms = [
            # 금융 일반
            '금리', '이자', '원금', '대출', '예금', '적금', '투자', '수익률',
            '포트폴리오', '자산', '부채', '자본', '유동성', '신용', '담보',
            
            # 증권/투자
            '주식', '채권', '펀드', '파생상품', '선물', '옵션', '스왑',
            'ETF', 'ELS', 'DLS', '워런트', '전환사채', '신주인수권',
            
            # 은행/보험
            '예금보험', '예금자보호', 'BIS', '자기자본비율', '건전성',
            '여신', '수신', '신용등급', '연체', '부실채권', 'NPL',
            
            # 금융규제
            '금융감독원', '금융위원회', '한국은행', '예금보험공사',
            '자본시장법', '은행법', '보험업법', '전자금융거래법',
            
            # 정보보호/보안
            '개인정보', '개인정보보호', '정보보호', '사이버보안',
            '재해복구', '백업', '암호화', '인증', '접근통제', '침해사고',
            
            # 핀테크/디지털
            '핀테크', '블록체인', '가상자산', '암호화폐', '디지털자산',
            '오픈뱅킹', 'API', '마이데이터', '인터넷뱅킹', '모바일뱅킹',
            
            # 리스크 관리
            '리스크', '위험관리', '신용위험', '시장위험', '운영위험',
            '유동성위험', 'VaR', '스트레스테스트', '내부통제', '준법감시',
            
            # 회계/재무
            'IFRS', 'K-IFRS', '재무제표', '손익계산서', '대차대조표',
            '현금흐름표', '감사', '회계감사', '내부회계', '공시'
        ]
        
        return terms
    
    def extract_concepts(self, text: str) -> List[Dict]:
        """
        금융 문서에서 핵심 개념 추출
        
        Args:
            text: 분석할 텍스트
            
        Returns:
            추출된 개념 리스트
        """
        concepts = []
        
        # 1. 금융 용어 추출
        financial_concepts = self._extract_financial_terms(text)
        concepts.extend(financial_concepts)
        
        # 2. 법령 조항 추출
        legal_concepts = self._extract_legal_articles(text)
        concepts.extend(legal_concepts)
        
        # 3. 수치 정보 추출
        numeric_concepts = self._extract_numeric_info(text)
        concepts.extend(numeric_concepts)
        
        # 4. 프로세스/절차 추출
        process_concepts = self._extract_processes(text)
        concepts.extend(process_concepts)
        
        # 5. 정의문 추출
        definition_concepts = self._extract_definitions(text)
        concepts.extend(definition_concepts)
        
        logger.info(f"Extracted {len(concepts)} concepts from text")
        return concepts
    
    def _extract_financial_terms(self, text: str) -> List[Dict]:
        """
        금융 용어 추출
        
        Args:
            text: 분석할 텍스트
            
        Returns:
            금융 용어 개념 리스트
        """
        concepts = []
        text_lower = text.lower()
        
        for term in self.financial_terms:
            if term.lower() in text_lower:
                # 용어 주변 컨텍스트 추출
                pattern = rf'.{{0,50}}{re.escape(term)}.{{0,50}}'
                matches = re.finditer(pattern, text, re.IGNORECASE)
                
                for match in matches:
                    context = match.group(0).strip()
                    concepts.append({
                        'type': 'financial_term',
                        'term': term,
                        'context': context,
                        'position': match.start()
                    })
        
        return concepts
    
    def _extract_legal_articles(self, text: str) -> List[Dict]:
        """
        법령 조항 추출
        
        Args:
            text: 분석할 텍스트
            
        Returns:
            법령 조항 개념 리스트
        """
        concepts = []
        
        for pattern in self.law_patterns:
            matches = re.finditer(pattern, text)
            
            for match in matches:
                # 조항 주변 100자 컨텍스트 추출
                start = max(0, match.start() - 50)
                end = min(len(text), match.end() + 50)
                context = text[start:end].strip()
                
                concepts.append({
                    'type': 'legal_article',
                    'article': match.group(0),
                    'number': match.group(1),
                    'context': context,
                    'position': match.start()
                })
        
        return concepts
    
    def _extract_numeric_info(self, text: str) -> List[Dict]:
        """
        수치 정보 추출
        
        Args:
            text: 분석할 텍스트
            
        Returns:
            수치 정보 개념 리스트
        """
        concepts = []
        
        for pattern in self.numeric_patterns:
            matches = re.finditer(pattern, text)
            
            for match in matches:
                # 수치 주변 컨텍스트 추출
                start = max(0, match.start() - 30)
                end = min(len(text), match.end() + 30)
                context = text[start:end].strip()
                
                concepts.append({
                    'type': 'numeric_info',
                    'value': match.group(1),
                    'unit': match.group(2) if len(match.groups()) > 1 else '',
                    'full_match': match.group(0),
                    'context': context,
                    'position': match.start()
                })
        
        return concepts
    
    def _extract_processes(self, text: str) -> List[Dict]:
        """
        프로세스/절차 추출
        
        Args:
            text: 분석할 텍스트
            
        Returns:
            프로세스 개념 리스트
        """
        concepts = []
        
        for keyword in self.process_keywords:
            pattern = rf'([^.!?\n]*{keyword}[^.!?\n]*[.!?])'
            matches = re.finditer(pattern, text, re.IGNORECASE)
            
            for match in matches:
                sentence = match.group(1).strip()
                
                # 단계별 프로세스 찾기
                step_pattern = r'(\d+)\s*[단계|차|번째]'
                steps = re.findall(step_pattern, sentence)
                
                concepts.append({
                    'type': 'process',
                    'keyword': keyword,
                    'sentence': sentence,
                    'steps': steps if steps else None,
                    'position': match.start()
                })
        
        return concepts
    
    def _extract_definitions(self, text: str) -> List[Dict]:
        """
        정의문 추출 (용어의 정의를 담은 문장)
        
        Args:
            text: 분석할 텍스트
            
        Returns:
            정의 개념 리스트
        """
        concepts = []
        
        # 정의 패턴들
        definition_patterns = [
            r'"([^"]+)"?\s*(?:이)?란\s+([^.!?\n]+[.!?])',  # "용어"란 ~
            r'([가-힣]+)\s*(?:이)?란\s+([^.!?\n]+[.!?])',  # 용어란 ~
            r'([가-힣]+)(?:을|를)?\s+([^.!?\n]*(?:의미|뜻|말)한다[^.!?\n]*[.!?])',  # ~을 의미한다
            r'([가-힣]+)(?:은|는)\s+([^.!?\n]*(?:이다|이며|이고)[^.!?\n]*[.!?])',  # ~는 ~이다
        ]
        
        for pattern in definition_patterns:
            matches = re.finditer(pattern, text)
            
            for match in matches:
                term = match.group(1).strip()
                definition = match.group(2).strip() if len(match.groups()) > 1 else ''
                
                # 금융 용어인 경우만 추출
                if any(ft in term for ft in self.financial_terms) or len(term) <= 20:
                    concepts.append({
                        'type': 'definition',
                        'term': term,
                        'definition': definition,
                        'full_sentence': match.group(0),
                        'position': match.start()
                    })
        
        return concepts
    
    def build_knowledge_graph(self, documents: List[Dict]) -> Dict:
        """
        문서들로부터 지식 그래프 구축
        
        Args:
            documents: 문서 리스트
            
        Returns:
            지식 그래프 딕셔너리
        """
        knowledge_graph = {
            'nodes': [],  # 개념 노드
            'edges': [],  # 관계 엣지
            'categories': defaultdict(list)  # 카테고리별 분류
        }
        
        all_concepts = []
        
        # 모든 문서에서 개념 추출
        for doc in documents:
            concepts = self.extract_concepts(doc.get('content', ''))
            
            for concept in concepts:
                concept['source'] = doc.get('source', 'unknown')
                all_concepts.append(concept)
        
        # 노드 생성
        seen_nodes = set()
        for concept in all_concepts:
            node_id = f"{concept['type']}_{concept.get('term', concept.get('article', concept.get('value', '')))}"
            
            if node_id not in seen_nodes:
                knowledge_graph['nodes'].append({
                    'id': node_id,
                    'type': concept['type'],
                    'data': concept
                })
                seen_nodes.add(node_id)
                
                # 카테고리별 분류
                knowledge_graph['categories'][concept['type']].append(node_id)
        
        # 관계 추출 (같은 문서에서 나타난 개념들 연결)
        for i, concept1 in enumerate(all_concepts):
            for concept2 in all_concepts[i+1:]:
                if concept1['source'] == concept2['source']:
                    # 위치가 가까운 개념들 연결 (100자 이내)
                    if abs(concept1.get('position', 0) - concept2.get('position', 0)) < 100:
                        edge = {
                            'source': f"{concept1['type']}_{concept1.get('term', '')}",
                            'target': f"{concept2['type']}_{concept2.get('term', '')}",
                            'relation': 'co-occurrence'
                        }
                        knowledge_graph['edges'].append(edge)
        
        logger.info(f"Built knowledge graph with {len(knowledge_graph['nodes'])} nodes and {len(knowledge_graph['edges'])} edges")
        return knowledge_graph
    
    def identify_qa_pairs(self, text: str) -> List[Tuple[str, str]]:
        """
        텍스트에서 잠재적 QA 쌍 식별
        
        Args:
            text: 분석할 텍스트
            
        Returns:
            (질문, 답변) 튜플 리스트
        """
        qa_pairs = []
        
        # 1. 정의문에서 QA 생성
        definitions = self._extract_definitions(text)
        for def_concept in definitions:
            question = f"{def_concept['term']}(이)란 무엇인가?"
            answer = def_concept['definition']
            if answer:
                qa_pairs.append((question, answer))
        
        # 2. 법령 조항에서 QA 생성
        legal_articles = self._extract_legal_articles(text)
        for article in legal_articles:
            if '조' in article['article']:
                question = f"{article['article']}의 내용은?"
                answer = article['context']
                qa_pairs.append((question, answer))
        
        # 3. 수치 정보에서 QA 생성
        numeric_info = self._extract_numeric_info(text)
        for num_info in numeric_info:
            context_words = num_info['context'].split()
            if len(context_words) > 5:
                # 수치 앞뒤 문맥으로 질문 생성
                question = f"{' '.join(context_words[:5])}의 기준은?"
                answer = num_info['full_match']
                qa_pairs.append((question, answer))
        
        # 4. 프로세스에서 QA 생성
        processes = self._extract_processes(text)
        for process in processes:
            if process.get('steps'):
                question = f"{process['keyword']} 절차는 몇 단계로 구성되는가?"
                answer = f"{len(process['steps'])}단계"
                qa_pairs.append((question, answer))
        
        logger.info(f"Identified {len(qa_pairs)} QA pairs from text")
        return qa_pairs
    
    def extract_key_sentences(self, text: str, max_sentences: int = 10) -> List[str]:
        """
        핵심 문장 추출 (중요도가 높은 문장)
        
        Args:
            text: 분석할 텍스트
            max_sentences: 추출할 최대 문장 수
            
        Returns:
            핵심 문장 리스트
        """
        # 문장 분리
        sentences = re.split(r'[.!?]\s+', text)
        
        # 문장별 점수 계산
        sentence_scores = []
        
        for sentence in sentences:
            score = 0
            
            # 금융 용어 포함 여부
            for term in self.financial_terms:
                if term in sentence:
                    score += 2
            
            # 숫자 정보 포함 여부
            if re.search(r'\d+', sentence):
                score += 1
            
            # 법령 조항 포함 여부
            if re.search(r'제\s*\d+\s*조', sentence):
                score += 3
            
            # 정의문 패턴 포함 여부
            if '란' in sentence or '의미한다' in sentence or '이다' in sentence:
                score += 2
            
            # 프로세스 키워드 포함 여부
            for keyword in self.process_keywords:
                if keyword in sentence:
                    score += 1
            
            if score > 0 and len(sentence) > 20:  # 너무 짧은 문장 제외
                sentence_scores.append((sentence, score))
        
        # 점수 기준 정렬
        sentence_scores.sort(key=lambda x: x[1], reverse=True)
        
        # 상위 N개 문장 반환
        key_sentences = [s[0] for s in sentence_scores[:max_sentences]]
        
        return key_sentences


if __name__ == "__main__":
    # 테스트용 코드
    import sys
    from pathlib import Path
    sys.path.append(str(Path(__file__).parent.parent.parent))
    
    logging.basicConfig(level=logging.INFO)
    
    # 샘플 텍스트
    sample_text = """
    전자금융감독규정 제21조의6(재해복구센터의 구축)에 따르면, 
    자산총액이 10조원 이상인 은행은 재해복구센터를 구축해야 한다.
    재해복구센터란 주전산센터가 재해로 인해 가동이 불가능한 경우를 대비하여 
    구축한 백업 시설을 의미한다. 모의훈련은 연 1회 이상 실시해야 하며,
    복구목표시간(RTO)은 24시간 이내로 설정해야 한다.
    
    개인정보보호법 제29조에서는 개인정보처리자가 개인정보가 분실·도난·유출·위조·변조 또는 
    훼손되지 아니하도록 안전성 확보에 필요한 기술적·관리적 및 물리적 조치를 하여야 한다고 규정한다.
    이는 정보주체의 권리를 보호하기 위한 필수적인 조치이다.
    
    금융투자상품의 수익률은 연 5.5%이며, 최소 투자금액은 1,000,000원이다.
    투자 절차는 다음과 같다: 1단계 상담, 2단계 위험성향 분석, 3단계 상품 선택, 4단계 계약 체결.
    """
    
    # 지식 추출기 초기화
    extractor = FinancialKnowledgeExtractor()
    
    # 개념 추출
    concepts = extractor.extract_concepts(sample_text)
    print(f"\n추출된 개념 수: {len(concepts)}")
    
    # 개념 유형별 출력
    concept_types = defaultdict(int)
    for concept in concepts:
        concept_types[concept['type']] += 1
    
    print("\n개념 유형별 분포:")
    for ctype, count in concept_types.items():
        print(f"- {ctype}: {count}개")
    
    # QA 쌍 추출
    qa_pairs = extractor.identify_qa_pairs(sample_text)
    print(f"\n추출된 QA 쌍: {len(qa_pairs)}개")
    
    if qa_pairs:
        print("\n샘플 QA:")
        for q, a in qa_pairs[:3]:
            print(f"Q: {q}")
            print(f"A: {a[:100]}...")
            print()
    
    # 핵심 문장 추출
    key_sentences = extractor.extract_key_sentences(sample_text, max_sentences=5)
    print("\n핵심 문장:")
    for i, sentence in enumerate(key_sentences, 1):
        print(f"{i}. {sentence}")