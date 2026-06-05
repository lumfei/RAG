# ============================================================
# config_data.py — 项目配置文件
# ============================================================
# 这个文件存放所有可调的参数，方便统一管理，不用到各模块里找。
# 如果你想把向量库换个位置、调整切片大小，改这里就行。

# .md5.text 是去重记录文件（存已入库内容的 MD5 指纹）
# 同名文件再次上传时，跳过不重复处理
md5_path = ".md5.text"

# ChromaDB 向量库的"集合"名称（相当于数据库里的"表"）
# 同一个 collection 里的数据才能互相检索
collection_name = "rag"

# 向量库持久化存储目录（数据存到本地 chroma_db 文件夹）
# 这样程序关了再开，之前入库的内容还在
persist_directory = "./chroma_db"

# ---------- 文本切片参数 ----------
# 为什么要切片？大段文本塞给 embedding 模型效果差、开销大，
# 切成小段后每段生成一个向量，检索时更精准。

# 每个切片最多 1000 个字符
chunk_size = 1000
# 相邻切片之间重叠 100 个字符（防止关键信息刚好被切断）
chunk_overlap = 100

# 切片优先级：先按段落切 → 换行 → 句号/感叹号/问号 → 空格 → 逐字符
# 为什么这么排？尽量在语义边界切，避免把一句话劈两半
separators = ["\n\n","\n",".","!","?","。", "！","？"," ", ""]

# 超过这个字符数的文本才会被切片，
# 短内容直接整段入库，不用切
max_split_char_number = 1000


similarity_threshold = 2

import os

# ---------- Embedding 模型配置 ----------
embedding_model_name = "text-embedding-v4"
# DashScope（阿里云百炼）API Key：优先读环境变量，否则用这里配置的值
dashscope_api_key = os.environ.get("DASHSCOPE_API_KEY", "your-dashscope-api-key")

# ---------- DeepSeek 大模型配置 ----------
deepseek_model_name = "deepseek-chat"
deepseek_api_key = os.environ.get("DEEPSEEK_API_KEY", "your-deepseek-api-key")
deepseek_base_url = "https://api.deepseek.com/v1"

session_config = {
    "configurable": {
        "session_id": "user_001",
    }
}