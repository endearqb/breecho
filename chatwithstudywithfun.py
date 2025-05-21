import streamlit as st
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, StorageContext, load_index_from_storage
from llama_index.core.node_parser import SimpleNodeParser
from llama_index.llms.openai import OpenAI
from llama_index.core import Settings
from llama_index.embeddings.openai import OpenAIEmbedding
# from llama_index.core.node_parser import SentenceSplitter
# from llama_index.core.postprocessor import SentenceTransformerRerank
import json
from datetime import datetime
import os


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
# 设置OpenAI API密钥
os.environ['OPENAI_API_KEY'] = '这里是key'
os.environ['OPENAI_API_BASE'] = '这里是base_url'

# 指定保存聊天历史的目录
SAVE_DIR = "/opt/stapp/data"

# 确保保存目录存在
os.makedirs(SAVE_DIR, exist_ok=True)

# 生成唯一的文件名
if "file_name" not in st.session_state:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    st.session_state.file_name = os.path.join(SAVE_DIR, f"chat_history_{timestamp}.json")

def save_chat_history():
    """将整个聊天历史保存到单个文件中"""
    with open(st.session_state.file_name, "w", encoding="utf-8") as f:
        json.dump(st.session_state.messages, f, ensure_ascii=False, indent=2)

st.header("Chat with the 微风轻语耳边风 💬 📚")

# 定义知识库选项
knowledge_base_options = ["微风轻语公众号", "ASM与二沉池建模设计"]

# 使用 st.radio 创建一个类似开关的选择界面
knowledge_base_name = st.radio(
    "请选择知识库",
    knowledge_base_options,
    index=0,
    key="knowledge_base",
    horizontal=True  # 这会使选项水平排列，更像一个开关
    )


if knowledge_base_name == "微风轻语公众号":
    storage_context = StorageContext.from_defaults(persist_dir="/opt/stapp/storagefun")
    index = load_index_from_storage(storage_context)
elif knowledge_base_name == "ASM与二沉池建模设计":
    storage_context = StorageContext.from_defaults(persist_dir="/opt/stapp/asmstorage")
    index = load_index_from_storage(storage_context)


cols_0 = st.columns([3, 1])


if "messages" not in st.session_state.keys(): # Initialize the chat message history
    st.session_state.messages = [
        {"role": "assistant", "content": "你可以问我一些关于微风轻语耳边风公众号或ASM与二沉池建模设计知识库中的相关内容与问题!"}
    ]

Settings.llm = OpenAI(temperature=0.1, model="gpt-4o-mini")
Settings.embed_model = OpenAIEmbedding(
    model="text-embedding-ada-002", embed_batch_size=100
)

# Settings.text_splitter = SentenceSplitter(chunk_size=1024)
# Settings.chunk_size = 512
# Settings.chunk_overlap = 20

# maximum input size to the LLM
Settings.context_window = 4096

# number of tokens reserved for text generation.
Settings.num_output = 2048


# documents = SimpleDirectoryReader("data").load_data()
# index = VectorStoreIndex.from_documents(
#     documents
# )


# storage_context = StorageContext.from_defaults(persist_dir="/opt/stapp/storagefun")
# index = load_index_from_storage(storage_context)
# rerank = SentenceTransformerRerank(top_n=5)

# 创建chat engine，设置初始召回数量并应用重排序器
chat_engine = index.as_chat_engine(
    similarity_top_k= 3,  # 初始召回10个节点
    # node_postprocessors=[rerank],  # 应用重排序
    chat_mode="condense_question",  # 或其他适合你用例的chat mode
    verbose=True
)
# chat_engine = index.as_chat_engine(chat_mode="condense_question", similarity_top_k=6, verbose=True)



if prompt := st.chat_input("Your question"): # Prompt for user input and save to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})

    # 自动保存更新后的聊天历史
    save_chat_history()

with cols_0[0]:
    for message in st.session_state.messages: # Display the prior chat messages
        if message["role"] != "node":  # 跳过role为node的消息
            with st.chat_message(message["role"]):
                st.write(message["content"])

    # If last message is not from assistant, generate a new response
    if st.session_state.messages[-1]["role"] != "assistant":
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                response = chat_engine.chat(prompt)
                st.write(response.response)
                message = {"role": "assistant", "content": response.response}
                st.session_state.messages.append(message) # Add response to message history
                save_chat_history()
                
        with cols_0[1]:
            source_content_list = []  # 创建一个列表来存储每个node的内容
            for node in response.source_nodes:
                    with st.expander("More Content"):
                        node_content = node.node.get_content()
                        st.markdown(node_content)
                        source_content_list.append(node_content)  # 将node内容添加到列表中

            # 将所有node的内容作为一个消息添加到聊天历史中
            if source_content_list:
                message = {"role": "node", "content": source_content_list}
                st.session_state.messages.append(message)
                save_chat_history()  # 保存更新后的聊天历史到JSON文件