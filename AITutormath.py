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

def nicereader(prompt):
    prompt = f"""
    
    你是一个数学题读题大师，在判断问题跟逻辑推理和数学计算相关后，能够将数学题中的条件分段，并一个一个描述为更清晰的数字信息，反复思考，尽可能补充完整的信息，使数学题拥有更明确的数值和数值关系,只需要转写和做简单的一步计算，不需要解答数学题。
    转写举例：
    1.现在是9时，一刻钟后我将出发-->现在是9时，9时15分(9时+一刻=9时15分)我将出发
    2.我今年35岁，是我女儿年龄的5倍-->我今年35岁，我女儿年龄7岁(35/5=7)
    3.我和小王一起走了40米--> 我和小王一起走了40米(我走了40米，小王走了40米)
    4.小王先出发，我后出发，我在100米处追上了小王-->小王先出发，我后出发，我在100米处追上了小王(我走了100米，小王走了100米)
    5.我和小王速度一样，从家出发一起走了40米，然后我立刻回家，小王继续往前走了40米-->我和小王速度一样，从家出发一起走了40米(我走了40米，小王走了40米)，然后我立刻回家(我往回走了40米)，小王继续往前走了40米
    
    返回要求：只返回print()中的具体内容
    # 步骤
    if {prompt} is not 数学或者逻辑相关问题:
        print("请告诉我想和我一起做的数学题")
    else:
        content=[] # 转写后的内容
        print(content)

    
    """
    messages = [{"role": "user", "content": prompt}]
    response = client.chat.completions.create(
        model="deepseek-coder",
        messages=messages,
        temperature=0, 
    )
    return response.choices[0].message.content

def problem_solver(prompt):
    prompt = f"""
    # 你是一个数学题学习伙伴，你不直接计算数学题的答案，而是一步一步帮助我解出数学题{prompt}

    """
    messages = [{"role": "user", "content": prompt}]
    response = client.chat.completions.create(
        model="deepseek-coder",
        messages=messages,
        temperature=0, 
    )
    
    return response.choices[0].message.content    


if "messages" not in st.session_state:
    st.session_state["messages"] = [
        {"role": "assistant", "content": "我是一个小学数学题应用题学习伙伴，请告诉我你想学习的数学题吧"}]
    


for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

# if "chat_history" not in st.session_state:
#     st.session_state.chat_history = []

if prompt := st.chat_input():
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.chat_message("user").write(prompt)
    # response = get_completion(prompt)
    msg = nicereader(prompt)
    st.session_state.messages.append({"role": "assistant", "content": msg})
    st.chat_message("assistant").write(msg)
    wait_msg = "请稍等片刻，我正在为您解答数学题"
    st.session_state.messages.append({"role": "assistant", "content": wait_msg})
    st.chat_message("assistant").write(wait_msg)
    answer_path=problem_solver(msg)
    # responses = st.write_stream(msg)
    # Add assistant response to chat history
    # st.session_state.messages.append({"role": "assistant", "content": responses})
    st.session_state.messages.append({"role": "assistant", "content": answer_path})
    st.chat_message("assistant").write(answer_path)