from langchain_core.documents import Document
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

from vector_stores import VectorStoreService
from langchain_community.embeddings import DashScopeEmbeddings
import config_data as config
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI



class RagService(object):
    def __init__(self):
        self.vector_service = VectorStoreService(
            embedding=DashScopeEmbeddings(model = config.embedding_model_name)
        )
        self.prompt_template = ChatPromptTemplate.from_messages(
            [
                ("system", "以我提供的已知参考资料为主，"
                           "简洁和专业的回答用户的问题。参考资料:{context}。"),
                ("user", "请回答用户提问：{input}")
            ]
        )
        self.chat_model = ChatOpenAI(
            model=config.deepseek_model_name,       # DeepSeek 模型名，如 deepseek-chat
            api_key=config.deepseek_api_key,        # DeepSeek API Key（需替换为真实 key）
            base_url=config.deepseek_base_url,      # DeepSeek 兼容 OpenAI 的接口地址
        )
        self.chain = self.__get_chain()


    def __get_chain(self):
        retriever = self.vector_service.get_retriever()


        def format_document(docs:list[Document]):
            if not docs:
                return "无相关参考资料"

            formatted_str = ""
            for doc in docs:
                formatted_str += f"文档片段：{doc.page_content}\n文档元数据：{doc.metadata}\n\n"

            return formatted_str

        chain = (
            {
                "context": retriever | format_document,   # 检索到的文档 → 格式化
                "input": RunnablePassthrough()            # 用户原始问题原文传递
            }
            | self.prompt_template                        # 填入提示词模板
            | self.chat_model                             # 发给 DeepSeek 大模型
            | StrOutputParser()                           # 解析输出为纯文本
        )
        return chain


if __name__ == '__main__':
    res = RagService().chain.invoke("我体重180斤，尺码推荐")
    print(res)




