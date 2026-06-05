


import streamlit as st
from streamlit import session_state
from rag import RagService

import config_data as config

st.title("智能客服")
st.divider()




if "message" not in st.session_state:
    st.session_state["message"] = [{"role":"assistant","content":"你好，有什么可以帮助你的吗？"}]

if "rag" not in st.session_state:
    st.session_state["rag"] = RagService()


for message in st.session_state["message"]:
    st.chat_message(message["role"]).write(message["content"])
prompt = st.chat_input()


if prompt:
    # 显示用户消息
    st.chat_message("user").write(prompt)
    st.session_state["message"].append({"role": "user", "content": prompt})


    ai_res_list = []
    with st.spinner("思考中....."):
        # 调用 RAG 链：传入问题和会话配置
        res_stream = st.session_state["rag"].chain.stream(
            {"input": prompt},              # 第1个参数：问题（dict 格式）
            config.session_config           # 第2个参数：会话配置（来自 config_data.py）
        )

        def capture(generator,cache_list):
            for chunk in generator:
                cache_list.append(chunk)
                yield chunk

        st.chat_message("assistant").write_stream(capture(res_stream,ai_res_list))
        st.session_state["message"].append({"role": "assistant", "content": "".join(ai_res_list)})









