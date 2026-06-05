# ============================================================
# app_file_uploader.py — 知识库前端页面（文件上传）
# ============================================================
# 这个模块用 Streamlit 框架搭建了一个 Web 界面，
# 用户可以上传 .txt 文件，预览内容，后续可以对接知识库入库。
#
# 运行方式：
#   终端执行：streamlit run app_file_uploader.py
#   浏览器会自动打开 http://localhost:8501
#
# Streamlit 是什么？
#   一个纯 Python 的 Web 框架，专为数据/AI 应用设计。
#   不需要写 HTML/CSS/JS，写 Python 就能生成网页。

import time                # 用于 sleep 延时
import streamlit as st      # Streamlit 框架，缩写 st 是官方约定
from knowledge_besa import KnowledgeBesaService

# ---------- 页面标题 ----------
st.title("知识库更新")

# ---------- 文件上传组件 ----------
# file_uploader 是 Streamlit 内置的上传组件
# 用户点击"Browse files"选文件，或者拖拽文件到页面上
uploader_file = st.file_uploader(
    label="上传txt文件",          # 上传按钮上显示的文字
    type=['txt'],                 # 只允许上传 .txt 文件
    accept_multiple_files=False,  # False = 一次只能上传一个文件
)

# ---------- 文件已上传：展示内容 ----------
# 当用户选了文件后，uploader_file 不再是 None
if "service" not in st.session_state:
    st.session_state["service"] = KnowledgeBesaService()


if uploader_file is not None:
    # 获取文件基本信息
    file_name = uploader_file.name        # 文件名，如 "测试.txt"
    file_type = uploader_file.type        # MIME 类型，如 "text/plain"
    file_size = uploader_file.size / 1024  # 文件大小，字节 → KB

    # 在页面上展示文件信息
    st.subheader(f"文件名：{file_name}")
    st.write(f"格式：{file_type} | 大小 {file_size:.2f} KB ")

    # 读取文件内容（.getvalue() 拿到原始字节，.decode() 转成字符串）
    texts = uploader_file.getvalue().decode("utf-8")
    # 把文件内容展示在页面上
    with st.spinner("载入数据库中"):
        time.sleep(1)
        result = st.session_state["service"].upload_by_str(texts, file_name)
        st.write(result)
