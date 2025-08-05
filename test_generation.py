#!/usr/bin/env python3
"""
데이터 생성 테스트 스크립트
EXAONE 또는 Qwen 모델로 질문-답변 생성 품질 테스트
"""

import sys
import json
import time
from pathlib import Path
from datetime import datetime

# src 디렉토리를 경로에 추가
sys.path.append(str(Path(__file__).parent / "src"))

from generate_data.main import DataGenerator
from rag.retriever import DocumentRetriever


def test_generation_with_prompts():
    """다양한 프롬프트로 생성 테스트"""
    
    print("="*80)
    print("🧪 EXAONE 모델을 이용한 데이터 생성 테스트")
    print("="*80)
    
    # 테스트할 프롬프트 템플릿들
    prompt_templates = [
        'prompts/diverse_types.txt',
        'prompts/high_quality.txt', 
        'prompts/training_data.txt'
    ]
    
    # 테스트 설정 (Qwen 또는 SOLAR 사용)
    base_config = {
        'model_name': 'Qwen/Qwen2.5-7B-Instruct',  # EXAONE 대신 Qwen 사용
        'use_rag': True,
        'use_quantization': False,  # Mac이므로 비활성화
        'temperature': 0.7,
        'top_p': 0.9
    }
    
    all_results = []
    
    for template_path in prompt_templates:
        print(f"\n📝 테스트 중: {template_path}")
        print("-"*60)
        
        config = base_config.copy()
        config['prompt_template'] = template_path
        
        try:
            # 생성기 초기화
            generator = DataGenerator(config)
            
            # 모델 로드
            print("🔄 모델 로딩 중...")
            generator.load_model()
            
            # RAG 테스트
            if generator.retriever:
                print("\n📚 RAG 검색 테스트:")
                test_query = "금융보안"
                context = generator.retriever.search(test_query, top_k=2)
                if context:
                    print(f"✅ RAG 작동 확인 (검색어: {test_query})")
                    print(f"   검색 결과 길이: {len(context)} 글자")
                else:
                    print("⚠️ RAG 검색 결과 없음")
            
            # 테스트 문제 생성 (3개만)
            print("\n🎯 문제 생성 중...")
            start_time = time.time()
            
            # 금융 관련 주제들
            finance_topics = ["자금세탁방지", "개인정보보호", "내부통제", "리스크관리", "투자자보호"]
            
            generated_items = []
            for i, topic in enumerate(finance_topics[:3], 1):
                print(f"\n[{i}/3] 주제: {topic}")
                
                # RAG로 관련 컨텍스트 검색
                context = ""
                if generator.retriever:
                    context = generator.retriever.search(topic, top_k=2)
                
                # 프롬프트 생성
                prompt = generator.prompt_template.format(
                    concept=topic,
                    context=context if context else "금융보안 관련 일반 지식"
                )
                
                # LLM으로 생성
                try:
                    generated_text = generator._generate_with_llm(prompt, temperature=0.7)
                    
                    result = {
                        'template': Path(template_path).stem,
                        'topic': topic,
                        'context_used': bool(context),
                        'generated': generated_text[:500] + "..." if len(generated_text) > 500 else generated_text,
                        'full_text': generated_text,
                        'length': len(generated_text)
                    }
                    
                    generated_items.append(result)
                    all_results.append(result)
                    
                    # 간단한 출력
                    print(f"✅ 생성 완료 ({len(generated_text)} 글자)")
                    
                except Exception as e:
                    print(f"❌ 생성 실패: {e}")
                    continue
            
            elapsed = time.time() - start_time
            print(f"\n⏱️ 소요 시간: {elapsed:.1f}초")
            
            # 메모리 정리
            generator.cleanup()
            
        except Exception as e:
            print(f"❌ 오류 발생: {e}")
            continue
    
    # 결과 저장
    output_file = Path("test_results") / f"generation_test_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    output_file.parent.mkdir(exist_ok=True)
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump({
            'test_time': datetime.now().isoformat(),
            'model': base_config['model_name'],
            'templates_tested': prompt_templates,
            'results': all_results
        }, f, ensure_ascii=False, indent=2)
    
    print("\n" + "="*80)
    print("📊 테스트 결과 요약")
    print("="*80)
    
    for template in prompt_templates:
        template_name = Path(template).stem
        template_results = [r for r in all_results if r['template'] == template_name]
        
        if template_results:
            print(f"\n📝 {template_name}:")
            avg_length = sum(r['length'] for r in template_results) / len(template_results)
            print(f"   - 생성 개수: {len(template_results)}")
            print(f"   - 평균 길이: {avg_length:.0f} 글자")
            print(f"   - RAG 사용률: {sum(1 for r in template_results if r['context_used']) / len(template_results) * 100:.0f}%")
    
    print(f"\n💾 상세 결과 저장: {output_file}")
    print("\n🎉 테스트 완료!")
    
    # 생성된 내용 샘플 출력
    print("\n" + "="*80)
    print("📄 생성 샘플 (첫 번째 결과)")
    print("="*80)
    
    if all_results:
        first_result = all_results[0]
        print(f"템플릿: {first_result['template']}")
        print(f"주제: {first_result['topic']}")
        print(f"RAG 사용: {'예' if first_result['context_used'] else '아니오'}")
        print(f"\n생성된 내용:\n{'-'*60}")
        print(first_result['full_text'][:1000])
        if len(first_result['full_text']) > 1000:
            print("\n... (이하 생략)")


def test_simple_generation():
    """간단한 생성 테스트 (빠른 확인용)"""
    
    print("\n🚀 간단한 테스트 모드")
    print("-"*60)
    
    # Qwen 사용 (작고 빠름)
    config = {
        'model_name': 'Qwen/Qwen2.5-1.5B-Instruct',  # 작은 모델
        'use_rag': True,
        'use_quantization': False,
        'prompt_template': 'prompts/training_data.txt'
    }
    
    try:
        generator = DataGenerator(config)
        generator.load_model()
        
        # 한 개만 테스트
        topic = "금융보안"
        
        # RAG 테스트
        context = ""
        if generator.retriever:
            # 랜덤 청크 가져오기
            context = generator.retriever.get_random_chunks(n=2)
            if context:
                print(f"✅ RAG 랜덤 청크 가져오기 성공")
        
        # 프롬프트
        prompt = generator.prompt_template.format(
            concept=topic,
            context=context if context else "금융 관련 일반 지식"
        )
        
        print("\n생성 중...")
        result = generator._generate_with_llm(prompt, temperature=0.7)
        
        print("\n" + "="*60)
        print("생성 결과:")
        print("="*60)
        print(result)
        
        generator.cleanup()
        
    except Exception as e:
        print(f"❌ 오류: {e}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="데이터 생성 테스트")
    parser.add_argument('--simple', action='store_true', help='간단한 테스트 실행')
    parser.add_argument('--model', type=str, help='사용할 모델 (exaone/qwen)')
    
    args = parser.parse_args()
    
    if args.simple:
        test_simple_generation()
    else:
        if args.model and args.model.lower() == 'qwen':
            # Qwen으로 테스트하려면 config 수정
            print("Qwen 모델로 테스트합니다...")
        test_generation_with_prompts()