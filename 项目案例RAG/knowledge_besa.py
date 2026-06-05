# ============================================================
# knowledge_besa.py — RAG 知识库核心服务
# ============================================================
# 这个模块是整个项目的"大脑"，负责两件事：
# 1. 把文本转成向量存入 ChromaDB（向量数据库）
# 2. 用 MD5 去重，避免重复内容反复入库
#
# 工作流程：
#   用户上传文本 → MD5 去重检查 → 切片 → 调 embedding 模型生成向量 → 存入 ChromaDB
#
# 依赖的"外部服务"：
#   - DashScope (阿里云百炼)：提供 text-embedding-v4 模型把文本变成向量
#   - ChromaDB：开源向量数据库，存向量 + 元数据，支持相似度检索

import os
from time import strftime

import config_data as config    # 导入上面的配置文件，所有可调参数都在那里
import hashlib                   # Python 内置的哈希库，用于计算 MD5
from langchain_chroma import Chroma                         # ChromaDB 的 LangChain 封装
from langchain_community.embeddings import DashScopeEmbeddings  # 阿里云的 embedding 模型
from langchain_text_splitters import RecursiveCharacterTextSplitter  # 文本切片工具
from datetime import datetime


# ------------------------------------------------------------------
# 工具函数：MD5 去重
# ------------------------------------------------------------------
# 思路：给每段文本算一个 MD5 指纹，
# 入库前先查这个指纹有没有出现过，出现过就跳过。

def check_md5(md5_str: str) -> bool:
    """检查一个 MD5 是否已经入库过。返回 True 表示已存在，应跳过。"""
    # 如果 .md5.text 文件还不存在，创建空文件，说明没有任何入库记录
    if not os.path.exists(config.md5_path):
        open(config.md5_path, 'w', encoding='utf-8').close()
        return False
    else:
        # 逐行读取 MD5 记录，看有没有匹配的
        for line in open(config.md5_path, 'r', encoding='utf-8').readlines():
            line = line.strip()
            if line == md5_str:
                return True
        return False


def save_md5(md5_str: str):
    """把一个新的 MD5 指纹追加写入记录文件（表示该内容已入库）"""
    with open(config.md5_path, 'a', encoding='utf-8') as f:
        f.write(md5_str + '\n')


def get_string_md5(input_str: str, encoding='utf-8') -> str:
    """计算一段文本的 MD5 哈希值（不管文本多长，输出固定 32 位）"""
    str_bytes = input_str.encode(encoding=encoding)   # 字符串 → 字节
    md5_obj = hashlib.md5()
    md5_obj.update(str_bytes)                          # 喂数据
    md5_hex = md5_obj.hexdigest()                      # 取出 32 位十六进制结果
    return md5_hex


# ------------------------------------------------------------------
# 核心类：知识库服务
# ------------------------------------------------------------------
class KnowledgeBesaService(object):
    """
    知识库服务 —— 把文本变成可检索的向量知识库。

    怎么用：
        service = KnowledgeBesaService()
        service.upload_by_str("你的文本内容", "文件名（用于标记来源）")
    """

    def __init__(self):
        # 确保向量库存储目录存在（exist_ok=True 表示已存在就不报错）
        os.makedirs(config.persist_directory, exist_ok=True)

        # 初始化 ChromaDB 向量库
        # Chroma 是一个开源的向量数据库，存"向量 + 元数据"
        # 用 LangChain 的 Chroma 封装来操作它
        self.chroma = Chroma(
            collection_name=config.collection_name,     # 表名
            embedding_function=DashScopeEmbeddings(     # 用阿里云的模型把文本变成向量
                model="text-embedding-v4"               # v4 是最新版本
            ),
            persist_directory=config.persist_directory  # 存到哪个文件夹
        )

        # 初始化文本切片器
        # RecursiveCharacterTextSplitter:
        #   从 separators 列表里优先级最高的分隔符（如 \n\n）开始切，
        #   切出来的段还是太长就换下一个分隔符，递归直到每段都 ≤ chunk_size
        self.spliter = RecursiveCharacterTextSplitter(
            chunk_size=config.chunk_size,           # 每段最大 1000 字符
            chunk_overlap=config.chunk_overlap,     # 段与段重叠 100 字符
            separators=config.separators,           # 按优先级尝试分隔符
            length_function=len                     # 用 Python 内置的 len 来算长度
        )

    def upload_by_str(self, data: str, filename: str) -> str:
        """
        把一段文本上传到向量库。

        参数:
            data: 要入库的文本内容
            filename: 来源文件名（记录在元数据里，方便追溯）

        返回:
            状态描述字符串
        """

        # 第一步：MD5 去重 —— 相同内容不重复入库
        md5_hex = get_string_md5(data)
        if check_md5(md5_hex):
            return "跳过，内容已经存在"

        # 第二步：文本切片 —— 长文本切成小段
        # 超过阈值的才切，短的整段直接用
        if len(data) > config.max_split_char_number:
            knowledge_chunks: list[str] = self.spliter.split_text(data)
        else:
            knowledge_chunks = [data]

        # 第三步：准备元数据 —— 记录这段文本的"档案信息"
        # 元数据不参与向量计算，但可以在检索时用来过滤和溯源
        metadata = {
            "source": filename,                                    # 来源文件名
            "create_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),  # 入库时间
            "operator": "小明",                                    # 操作人
        }

        # 第四步：写入向量库 —— 每段文本 → 调 embedding → 生成向量 → 存入 ChromaDB
        self.chroma.add_texts(
            knowledge_chunks,
            metadatas=[metadata for _ in knowledge_chunks]  # 每个切片都带上同样的元数据
        )

        # 第五步：记录 MD5 —— 下次再遇到同样内容直接跳过
        save_md5(md5_hex)
        return "内容成功载入向量库"


# ------------------------------------------------------------------
# 程序入口：直接运行本文件时执行下面的测试代码
# ------------------------------------------------------------------
if __name__ == '__main__':
    # 创建一个知识库服务实例
    service = KnowledgeBesaService()
    # 把"周杰伦"这条文本以 testfile 为来源存入向量库
    result = service.upload_by_str("周杰伦", "testfile")
    print(result)













