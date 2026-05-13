import os

DASHSCOPE_API_KEY = os.getenv("DASHSCOPE_API_KEY")
if not DASHSCOPE_API_KEY:
    raise ValueError("错误：请设置 'DASHSCOPE_API_KEY' 环境变量。")

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
KNOWLEDGE_BASE_DIR = os.path.join(PROJECT_DIR, "knowledge_base")
CHROMA_DB_DIR = os.path.join(PROJECT_DIR, "chroma_db")

os.makedirs(KNOWLEDGE_BASE_DIR, exist_ok=True)
os.makedirs(os.path.join(KNOWLEDGE_BASE_DIR, "books"), exist_ok=True)
os.makedirs(os.path.join(KNOWLEDGE_BASE_DIR, "notes"), exist_ok=True)
os.makedirs(os.path.join(KNOWLEDGE_BASE_DIR, "exercises"), exist_ok=True)
os.makedirs(os.path.join(KNOWLEDGE_BASE_DIR, "images"), exist_ok=True)
os.makedirs(CHROMA_DB_DIR, exist_ok=True)

MULTIMODAL_EMBEDDING_MODEL = "multimodal-embedding-v1"
TEXT_EMBEDDING_MODEL = "text-embedding-v4"
LLM_MODEL = "qwen-flash"

CHUNK_SIZE = 600
CHUNK_OVERLAP = 100

BOOK_TYPES = ["初级会计实务", "经济法基础"]
KNOWLEDGE_TYPES = ["概念定义", "会计分录", "税法法条", "计算公式", "易错辨析", "真题习题"]
EXAM_LEVELS = ["必考", "高频", "了解"]
SOURCE_TYPES = ["2025官方教材", "2026教材", "三色笔记", "历年真题"]

IMAGE_KEYWORDS = ["图片", "图表", "图示", "表格", "展示"]
VIDEO_KEYWORDS = ["视频", "录像", "影片"]
MEDIA_DISTANCE_THRESHOLD = 3.0

RETRIEVAL_K = 5

EMBEDDING_DIM = 1024

# 混合检索配置
BM25_WEIGHT = 0.15  # 大幅降低BM25权重
VECTOR_WEIGHT = 0.85

# 动态分块规则
CHUNK_RULES = {
    "概念定义": {"chunk_size": 300, "chunk_overlap": 60},
    "会计分录": {"chunk_size": 200, "chunk_overlap": 40},
    "税法法条": {"chunk_size": 350, "chunk_overlap": 70},
    "计算公式": {"chunk_size": 250, "chunk_overlap": 50},
    "易错辨析": {"chunk_size": 350, "chunk_overlap": 70},
    "真题习题": {"chunk_size": 500, "chunk_overlap": 100},
    "default": {"chunk_size": 350, "chunk_overlap": 70}
}

# 难度等级
DIFFICULTY_LEVELS = ["简单", "中等", "困难"]