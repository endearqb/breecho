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

def get_completion(history,prompt, model="yi-large"):
    prompt = f"""你是一个总是以苏格拉底式的风格回应的AI家庭教师,根据和学生的聊天记录{history},根据学生最后的回复{prompt}，引导学生解决问题
            - 如果学生没有提出问题，询问学生想练习什么科目,然后一起逐步练习
            - 以二年级阅读水平或学生语言水平进行沟通 
            - 不直接给出答案,而是引导学生独立思考 
            - 根据学生知识水平调整问题难度
            - 检查学生是否理解,提醒错误并鼓励练习
            - 警惕学生反复寻求提示而不付出努力
            - 如果连续3次无所作为,缩小提示范围并询问学生卡在哪里
            - 始终使用示例问题而非学生实际问题
            - 每次用不同的方式鼓励学生
            记住:一步一步引导，每次只给一步提示，一次回复不要超过100个字，直到学生自己得出答案
            """
    messages = [
        {"role": "user", "content": prompt}
    ]
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=0, # this is the degree of randomness of the model's output
    )
    return response.choices[0].message.content

if "messages" not in st.session_state:
    st.session_state["messages"] = [
        {"role": "assistant", "content": "有什么可以帮你的吗?"}]
    


for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

# if "chat_history" not in st.session_state:
#     st.session_state.chat_history = []

if prompt := st.chat_input():
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.chat_message("user").write(prompt)
    # response = get_completion(prompt)
    msg = get_completion(st.session_state.messages,prompt)
    # responses = st.write_stream(msg)
    # Add assistant response to chat history
    # st.session_state.messages.append({"role": "assistant", "content": responses})
    st.session_state.messages.append({"role": "assistant", "content": msg})
    st.chat_message("assistant").write(msg)