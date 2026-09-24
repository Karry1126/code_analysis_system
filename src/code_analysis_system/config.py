from pathlib import Path
from dotenv import load_dotenv
import os

# 路径
BASE_DIR = Path(__file__).resolve().parent.parent.parent
load_dotenv(BASE_DIR / ".env")
COMPANY_CODE_RELATIVE_PATH = os.getenv("COMPANY_CODE_RELATIVE_PATH")
if not COMPANY_CODE_RELATIVE_PATH:
    raise ValueError("COMPANY_CODE_RELATIVE_PATH is not set")
COMPANY_CODE_REPO_PATH = (BASE_DIR / COMPANY_CODE_RELATIVE_PATH).resolve()
RAG_INDEX_DIR = BASE_DIR / ".rag_index"
MODULE_CARDS_PATH = (BASE_DIR / "data" / "module_cards.json").resolve()
INCLUDE_INDEX_PATH = (BASE_DIR / "data" / "include_index.json").resolve()

# 参数
# DeepSeek 上下文约 1M。这里只约束 MessageStore 里的历史，
# system / tools / 运行时调度消息 / completion 还要另外占额度。
MAX_HISTORY_INPUT_TOKENS = 200_000
MAX_AGENT_LOOPS = 10

# 单条 tool 结果写入 MessageStore 前的硬顶（字符）。
# MessageStore.trim 不负责吞掉「当前这一块」超大结果。
MAX_TOOL_RESULT_CHARS = 12_000
MAX_SEARCH_MATCHES = 50
MAX_LIST_ITEMS = 80
MAX_FILENAME_MATCHES = 40
MAX_READ_LINES = 200
READ_PREVIEW_LINES = 40
RETRIEVE_TOP_K = 8
RETRIEVE_TOP_K_MAX = 12
SNIPPET_MAX_CHARS = 800

def get_api_key() -> str:
    """读取 API Key。"""
    api_key = os.getenv("DEEPSEEK_API_KEY")
    if not api_key:
        raise RuntimeError(
            "缺少环境变量 DEEPSEEK_API_KEY，请先在 PowerShell 中执行："
            ' $env:DEEPSEEK_API_KEY="你的 API Key"'
        )
    return api_key