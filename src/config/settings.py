"""
프로젝트 설정 관리
"""

import os
from pathlib import Path
from typing import Dict, Any

# 프로젝트 루트 디렉토리
PROJECT_ROOT = Path(__file__).parent.parent.parent

# 디렉토리 경로
DATA_DIR = PROJECT_ROOT / "data"
EXTERNAL_DATA_DIR = DATA_DIR / "external"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
AUGMENTED_DATA_DIR = DATA_DIR / "augmented"
LOGS_DIR = PROJECT_ROOT / "logs"

# 모델 설정
MODEL_CONFIG = {
    "default_model": "beomi/SOLAR-10.7B-v1.0",
    "fallback_models": [
        "LG-AI-EXAONE/EXAONE-3.0-7.8B-Instruct",
        "Qwen/Qwen2.5-7B-Instruct"
    ],
    "use_quantization": True,
    "quantization_bits": 4,
    "max_length": 2048,
    "temperature": 0.7,
    "top_p": 0.9
}

# 데이터 증강 설정
AUGMENTATION_CONFIG = {
    "target_count": 5000,
    "multiple_choice_ratio": 0.7,  # 70% 객관식, 30% 주관식
    "mc4_ratio": 0.5,  # 객관식 중 50%는 4지선다
    "batch_size": 32,
    "similarity_threshold": 0.85,
    "min_question_length": 10,
    "max_question_length": 500,
    "min_answer_length": 5,
    "max_answer_length": 1000
}

# 품질 검증 설정
QUALITY_CONFIG = {
    "check_financial_accuracy": True,
    "check_answer_validity": True,
    "remove_duplicates": True,
    "validate_format": True,
    "min_quality_score": 80.0
}

# 증강 전략
AUGMENTATION_STRATEGIES = [
    "difficulty_variation",
    "paraphrase",
    "scenario_application",
    "concept_combination"
]

# 금융 도메인 카테고리
FINANCIAL_CATEGORIES = [
    "금융일반",
    "은행업무",
    "증권투자",
    "보험",
    "금융규제",
    "정보보호",
    "핀테크",
    "리스크관리",
    "회계재무"
]

# 라이선스 허용 목록
ALLOWED_LICENSES = [
    "공공누리 1유형",
    "공공누리 2유형",
    "공공누리 3유형",
    "공공누리 4유형",
    "CC0",
    "CC BY",
    "CC BY-SA",
    "CC BY-NC",
    "CC BY-NC-SA",
    "MIT",
    "Apache 2.0",
    "교육용",
    "비상업적"
]

# 실행 환경 설정
RUNTIME_CONFIG = {
    "device": "cuda",  # cuda, cpu, mps
    "num_workers": 4,
    "seed": 42,
    "deterministic": True,
    "log_level": "INFO"
}

# 파일 형식 설정
FILE_FORMATS = {
    "input": [".pdf", ".txt", ".md", ".json", ".jsonl", ".csv", ".xlsx"],
    "output": [".json", ".jsonl", ".csv"],
    "encoding": "utf-8"
}

# API 설정 (임베딩용 - 로컬만)
EMBEDDING_CONFIG = {
    "model_name": "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
    "max_seq_length": 512,
    "batch_size": 32
}

# 검증 규칙
VALIDATION_RULES = {
    "no_test_data": True,  # test.csv 사용 금지
    "no_api_calls": True,  # 외부 API 호출 금지
    "no_commercial_license": True,  # 상업적 라이선스 금지
    "single_model_only": True,  # 단일 모델만 사용
    "offline_only": True,  # 오프라인 환경
    "time_limit_minutes": 270  # 270분 제한
}

# 로깅 설정
LOGGING_CONFIG = {
    "version": 1,
    "disable_existing_loggers": False,
    "formatters": {
        "standard": {
            "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        },
        "detailed": {
            "format": "%(asctime)s - %(name)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s"
        }
    },
    "handlers": {
        "console": {
            "class": "logging.StreamHandler",
            "level": "INFO",
            "formatter": "standard",
            "stream": "ext://sys.stdout"
        },
        "file": {
            "class": "logging.FileHandler",
            "level": "DEBUG",
            "formatter": "detailed",
            "filename": str(LOGS_DIR / "augmentation.log"),
            "encoding": "utf-8"
        }
    },
    "loggers": {
        "": {
            "level": "DEBUG",
            "handlers": ["console", "file"]
        }
    }
}


def get_config(section: str = None) -> Dict[str, Any]:
    """
    설정 반환
    
    Args:
        section: 특정 섹션 이름
        
    Returns:
        설정 딕셔너리
    """
    config = {
        "model": MODEL_CONFIG,
        "augmentation": AUGMENTATION_CONFIG,
        "quality": QUALITY_CONFIG,
        "runtime": RUNTIME_CONFIG,
        "validation": VALIDATION_RULES,
        "logging": LOGGING_CONFIG
    }
    
    if section:
        return config.get(section, {})
    
    return config


def update_config(section: str, updates: Dict[str, Any]):
    """
    설정 업데이트
    
    Args:
        section: 섹션 이름
        updates: 업데이트할 내용
    """
    if section == "model":
        MODEL_CONFIG.update(updates)
    elif section == "augmentation":
        AUGMENTATION_CONFIG.update(updates)
    elif section == "quality":
        QUALITY_CONFIG.update(updates)
    elif section == "runtime":
        RUNTIME_CONFIG.update(updates)
    elif section == "validation":
        VALIDATION_RULES.update(updates)


# 환경 변수 오버라이드
if os.getenv("FSKU_MODEL_NAME"):
    MODEL_CONFIG["default_model"] = os.getenv("FSKU_MODEL_NAME")

if os.getenv("FSKU_TARGET_COUNT"):
    AUGMENTATION_CONFIG["target_count"] = int(os.getenv("FSKU_TARGET_COUNT"))

if os.getenv("FSKU_USE_GPU"):
    RUNTIME_CONFIG["device"] = "cuda" if os.getenv("FSKU_USE_GPU").lower() == "true" else "cpu"