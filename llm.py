import streamlit as st
from openai import OpenAI
from st_pages import add_page_title, hide_pages
# import tiktoken


st.set_page_config(
    page_title="微风轻语BreeCho",  # 自定义页面标题
    page_icon="💭",  # 可以是一个URL链接，或者是emoji表情符号
    layout="wide",  # 页面布局: "centered" 或 "wide"
    initial_sidebar_state="collapsed",  # 初始侧边栏状态: "auto", "expanded", "collapsed"
)


# streamlit run GB500142021AO.py
hide_streamlit_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            .css-1lsmgbg.egzxvld1 {visibility: hidden;} /* This targets the Deploy button */
            </style>
            """

st.markdown(hide_streamlit_style, unsafe_allow_html=True)

# # 自定义CSS样式，使用自定义字体
# custom_css = """
# <style>
# @font-face {
#     font-family: 'LXGW WenKai GB';
#     src: url('LXGWWenKaiGB.ttf') format('truetype');
# }
# body {
#     font-family: 'LXGW WenKai GB', serif;
# }
# </style>
# """

# st.markdown(custom_css, unsafe_allow_html=True)

add_page_title(layout="wide")


API_BASE = "这里是base_url"
API_KEY = "这里是key"
client = OpenAI(
    api_key=API_KEY,
    base_url=API_BASE
)

# 设置输入和输出的 tokens 限制
INPUT_TOKENS_LIMIT = 512
OUTPUT_TOKENS_LIMIT = 2048
# st.title("💬 Chatbot")

if "messages" not in st.session_state:
    st.session_state["messages"] = [{"role": "assistant", "content": "有什么可以帮你的吗?"}]

for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

# 计算消息的 tokens 数量的函数
# def count_tokens(message, model="yi-medium"):
#     encoding = tiktoken.encoding_for_model(model)
#     return len(encoding.encode(message))

if prompt := st.chat_input():
    # if not openai_api_key:
    #     st.info("Please add your OpenAI API key to continue.")
    #     st.stop()

    # client = OpenAI(api_key=openai_api_key)
        # 计算输入消息的 tokens 数量
    # input_tokens = count_tokens(prompt)
    
    # if input_tokens > INPUT_TOKENS_LIMIT:
    #     st.warning(f"输入消息过长，请限制在 {INPUT_TOKENS_LIMIT} 个 tokens 以内。")

    st.session_state.messages.append({"role": "user", "content": prompt})
    st.chat_message("user").write(prompt)
    response = client.chat.completions.create(model="yi-medium", messages=st.session_state.messages, max_tokens=OUTPUT_TOKENS_LIMIT)
    msg = response.choices[0].message.content
    st.session_state.messages.append({"role": "assistant", "content": msg})
    st.chat_message("assistant").write(msg)