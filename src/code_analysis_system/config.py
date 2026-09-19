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

# 参数
MAX_HISTORY_TURNS = 6
MAX_AGENT_LOOPS = 10

def get_api_key() -> str:
    """读取 API Key。"""
    api_key = os.getenv("DEEPSEEK_API_KEY")
    if not api_key:
        raise RuntimeError(
            "缺少环境变量 DEEPSEEK_API_KEY，请先在 PowerShell 中执行："
            ' $env:DEEPSEEK_API_KEY="你的 API Key"'
        )
    return api_key