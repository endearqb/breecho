import streamlit as st
from openai import OpenAI
from st_pages import add_page_title, hide_pages
import time
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

# # Streamed response emulator
# def get_completion(prompt, model="yi-medium"):
#     messages = [
#         # {"role": "system", "content": "你是一个专业的翻译，你会根据用户的输入直接翻译成英文。"},
#         {"role": "user", "content": prompt}]
#     response = client.chat.completions.create(
#         model=model,
#         messages=messages,
#         temperature=0, # this is the degree of randomness of the model's output
#     )
#     return response.choices[0].message.content


def derect_translate(content):
    prompt = f"""你是一位精通简体中文的专业翻译，尤其擅长将专业学术论文翻译成浅显易懂的科普文章。请你帮我将英文翻译成中文，风格与中文科普读物相似。
规则：
- 翻译时要准确传达原文的事实和背景。
- 即使上意译也要保留原始段落格式，以及保留术语，例如 FLAC，JPEG 等。保留公司缩写，例如 Microsoft, Amazon, OpenAI 等。
- 人名不翻译
- 同时要保留引用的论文，例如 [20] 这样的引用。
- 对于 Figure 和 Table，翻译的同时保留原有格式，例如：“Figure 1: ”翻译为“图 1: ”，“Table 1: ”翻译为：“表 1: ”。
- 全角括号换成半角括号，并在左括号前面加半角空格，右括号后面加半角空格。
- 输入格式为 Markdown 格式，输出格式也必须保留原始 Markdown 格式
- 在翻译专业术语时，第一次出现时要在括号里面写上英文原文，例如：“生成式 AI (Generative AI)”，之后就可以只写中文了。
- 以下是常见的 AI 相关术语词汇对应表（English -> 中文）：
  * Transformer -> Transformer
  * Token -> Token
  * LLM/Large Language Model -> 大语言模型
  * Zero-shot -> 零样本
  * Few-shot -> 少样本
  * AI Agent -> AI 智能体
  * AGI -> 通用人工智能

根据以上要求将英文内容{content}直译，保持原有格式，不要遗漏任何信息

    """
    messages = [
        {"role": "user", "content": prompt}]
    response = client.chat.completions.create(
        model="yi-large-turbo",
        messages=messages,
        temperature=0, # this is the degree of randomness of the model's output
    )
    return response.choices[0].message.content

def thought_translate(content,translate):
    prompt = f"""根据直译的结果{translate}，结合原文{content}指出其中存在的具体问题，要准确描述，不宜笼统的表示，也不需要增加原文不存在的内容或格式，包括不仅限于：
  - 不符合中文表达习惯，明确指出不符合的地方
  - 语句不通顺，指出位置，不需要给出修改意见，意译时修复
  - 晦涩难懂，不易理解，可以尝试给出解释"""
    messages = [
        {"role": "user", "content": prompt}]
    response = client.chat.completions.create(
        model="yi-large-turbo",
        messages=messages,
        temperature=0.3, # this is the degree of randomness of the model's output
    )
    return response.choices[0].message.content

def final_translate(content,translate,thought):
    prompt = f"""根据第一步直译的结果{translate}和第二步指出的问题{thought}，将{content}重新进行意译，保证内容的原意的基础上，使其更易于理解，更符合中文的表达习惯，同时保持原有的格式不变"""
    messages = [
        {"role": "user", "content": prompt}]
    response = client.chat.completions.create(
        model="yi-large-turbo",
        messages=messages,
        temperature=0.3, # this is the degree of randomness of the model's output
    )
    return response.choices[0].message.content

def translator(content):
    translate = derect_translate(content)
    thought = thought_translate(content,translate)
    final = final_translate(content,translate,thought)
    return translate,thought,final

if "messages" not in st.session_state:
    st.session_state["messages"] = [
        {"role": "assistant", "content": "请输入需要翻译的英文段落"}]
    


for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

# if "chat_history" not in st.session_state:
#     st.session_state.chat_history = []

if prompt := st.chat_input():
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.chat_message("user").write(prompt)
    msg1 = derect_translate(prompt)
    st.chat_message("assistant").write(msg1)
    st.session_state.messages.append({"role": "assistant", "content": msg1})
    msg2 = thought_translate(prompt,msg1)
    st.chat_message("assistant").write(msg2)
    st.session_state.messages.append({"role": "assistant", "content": msg2})
    msg3 = final_translate(prompt,msg1,msg2)
    st.chat_message("assistant").write(msg3)
    st.session_state.messages.append({"role": "assistant", "content": msg3})
    # msg = translator(prompt)
    # # responses = st.write_stream(msg)
    # # Add assistant response to chat history
    # # st.session_state.messages.append({"role": "assistant", "content": responses})
    # st.session_state.messages.append({"role": "assistant", "content": msg})
    # st.chat_message("assistant").write(msg)