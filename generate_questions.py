import pandas as pd
import requests
import json
import time
from datetime import datetime, timedelta
import os

OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL = "mistral"  # 또는 'phi', 'gemma'

INPUT_PATH = "./data/test.csv"
OUTPUT_PATH = "./output/generated_questions.jsonl"

def generate(prompt):
    headers = {"Content-Type": "application/json"}
    payload = {
        "model": MODEL,
        "prompt": prompt,
        "stream": False
    }
    try:
        response = requests.post(OLLAMA_URL, headers=headers, json=payload)
        response.raise_for_status()
        return response.json()["response"].strip()
    except Exception as e:
        print(f"❌ [오류] 모델 응답 실패: {e}")
        return ""

def build_question_prompt(original_question):
    return f"""
다음 질문은 한국 금융보안 실무자가 자주 묻는 질문입니다.

[질문]: {original_question}

이 질문과 의미는 유사하지만 표현 방식이 다른 질문 3개를 **한국어로** 생성해주세요.
FSKU 기준(정확성, 이해도, 일관성)에 부합하도록 실무적 맥락에서 다양하게 작성해주세요.
각 질문은 줄바꿈으로 구분해주세요.
"""

def build_answer_prompt(question):
    return f"""
[질문]: {question}

이 질문에 대해 **한국어로**, 금융보안 실무자가 납득할 수 있는 수준의 정답을 작성해주세요.
핵심 키워드를 포함하고 실무적인 답변이어야 합니다.
"""

def build_explanation_prompt(question, answer):
    return f"""
[질문]: {question}
[답변]: {answer}

위의 답변이 왜 적절한지 **한국어로**, 실무자 입장에서 핵심 이유만 간단히 설명해주세요.
불필요한 장황한 설명은 피해주세요.
"""

def load_completed_ids(output_path):
    if not os.path.exists(output_path):
        return set()
    with open(output_path, "r", encoding="utf-8") as f:
        return set(json.loads(line)["original_id"] for line in f if line.strip())

def main():
    df = pd.read_csv(INPUT_PATH)
    total = len(df)
    start_time = time.time()

    completed_ids = load_completed_ids(OUTPUT_PATH)
    df = df[~df["ID"].isin(completed_ids)]  # 처리되지 않은 질문만 남김

    print(f"\n🟢 총 {len(df)}개 질문 처리 시작 (이미 처리된 {total - len(df)}개는 스킵됨)\n")

    with open(OUTPUT_PATH, "a", encoding="utf-8") as f_out:
        for idx, row in df.iterrows():
            qid = row["ID"]
            original = row["Question"]
            progress_idx = idx + 1 + len(completed_ids)

            # ETA
            elapsed = time.time() - start_time
            avg_time = elapsed / max(progress_idx, 1)
            eta = timedelta(seconds=int(avg_time * (len(df) - idx - 1)))

            print(f"🧠 [{progress_idx}/{total}] 질문 {qid[:12]}... ⏳ ETA: {eta}")

            similar_raw = generate(build_question_prompt(original))
            similar_questions = [q.strip(" -0123.").strip() for q in similar_raw.split("\n") if q.strip()]

            for i, q in enumerate(similar_questions):
                print(f"   ↳ 질문 {i+1}: {q}")

                answer = generate(build_answer_prompt(q))
                explanation = generate(build_explanation_prompt(q, answer))

                row_json = {
                    "original_id": qid,
                    "original_question": original,
                    "generated_question": q,
                    "answer": answer,
                    "explanation": explanation
                }

                f_out.write(json.dumps(row_json, ensure_ascii=False) + "\n")
                f_out.flush()

                time.sleep(1.5)  # 요청 간격 조절

    total_time = timedelta(seconds=int(time.time() - start_time))
    print(f"\n✅ 생성 완료! 총 소요 시간: {total_time} ➤ 결과: {OUTPUT_PATH}")

if __name__ == "__main__":
    main()
