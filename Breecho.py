import streamlit as st
import streamlit_shadcn_ui as ui
import pandas as pd
import numpy as np
# from local_components import card_container
# from streamlit_shadcn_ui import slider, input, textarea, radio_group, switch
from st_pages import Page, Section, show_pages, add_page_title, hide_pages
import base64
import os
import streamlit.components.v1 as components
# from st_aggrid import AgGrid
# import importlib
# 设置全局字体
# mpl.rc('font', family='Times New Roman', size=12)  # 可以选择你系统支持的字体

# 设置页面配置
st.set_page_config(
    page_title="微风轻语耳边风-Breecho.cn",  # 自定义页面标题
    page_icon="🌬️",  # 可以是一个URL链接，或者是emoji表情符号
    layout="wide",  # 页面布局: "centered" 或 "wide"
    initial_sidebar_state="collapsed",  # 初始侧边栏状态: "auto", "expanded", "collapsed"
)

components.html(
    """
<script type="text/javascript">
(function(c,l,a,r,i,t,y){
    c[a]=c[a]||function(){(c[a].q=c[a].q||[]).push(arguments)};
    t=l.createElement(r);t.async=1;t.src="https://www.clarity.ms/tag/"+i;
    y=l.getElementsByTagName(r)[0];y.parentNode.insertBefore(t,y);
})(window, document, "clarity", "script", "mitxe0k4zb");
</script>
    """,
    height=0
)

hide_streamlit_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            .css-1lsmgbg.egzxvld1 {visibility: hidden;} /* This targets the Deploy button */
            </style>
            """
st.markdown(hide_streamlit_style, unsafe_allow_html=True)

# 读取本地图片并进行 Base64 编码
def get_image_base64(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode()

gb_img = get_image_base64("/opt/stapp/gb.png")

dwa_img = get_image_base64("/opt/stapp/dwa.png")

takacs_img = get_image_base64("/opt/stapp/takacs.png")

asm1_img = get_image_base64("/opt/stapp/asm1.png")

qa_img = get_image_base64("/opt/stapp/qa.png")

twentyfour_img = get_image_base64("/opt/stapp/24.png")

epanet_img = get_image_base64("/opt/stapp/epanet.png")

asm2_img = get_image_base64("/opt/stapp/asm2.png")

app_img = get_image_base64("/opt/stapp/app.png")

add_page_title(layout="wide",initial_sidebar_state="collapsed")

show_pages(
    [   
        Page("/opt/stapp/Breecho.py", "微风轻语耳边风-Breecho.cn", "🌬️"),

        # # 2024 Content
        Section("Calcmodels 2024", "🧙‍♂️"),
        Page("/opt/stapp/gb.py", "室外排水设计标准AO计算", "📚", in_section=True),
        Page("/opt/stapp/dwa.py", "德国DWA-131-A AO计算", "📚", in_section=True),
        Page("/opt/stapp/takacs.py", "二沉池一维建模计算", "📚", in_section=True),
        Page("/opt/stapp/mix1.py", "New!储水池余氯衰减高级模拟系统(源自EPANET)", "✨", in_section=True),
        # Page("/opt/stapp/ASM1slimdemo.py", "New!新ASM简易不简单-易用又准确模型", "✨", in_section=True),
        # Page("/opt/stapp/ASMflow.py", "ASM1活性污泥模型计算", "📚", in_section=True),
        Page("/opt/stapp/calculate24.py", "24点计算", "📚", in_section=True),
        
        Section("AI Agents", "🧙‍♂️"),
        # Page("/opt/stapp/chatwithstudywithfun.py","微风轻语问答","💬",in_section=True),
        # Page("/opt/stapp/translator.py","英译中伙伴","👩‍🏫",in_section=True),
        # Page("/opt/stapp/AITutormath.py","AI小学数学题解题助手","🤖",in_section=True),
        Page("/opt/stapp/DocQA.py","QA处理","📚",in_section=True),
        Page("/opt/stapp/app.py","excel分析助手demo","🤖",in_section=True),

        # Page("/opt/stapp/llm.py","Chatbot","🤖",in_section=True),
    ]
)


# hide_pages(["Thank you"])

ui.badges(badge_list=[("Courses", "default"), ("Calculators", "secondary"), ("For Free", "destructive")], class_name="flex gap-2", key="main_badges1")
st.caption("欢迎来到微风轻语耳边风-Breecho.cn，这里记录着一些的学习笔记的内容，也有一些基础的水处理计算器")
st.caption("Welcome to Breecho.cn, here you can find some Study Records and Eviromental Calcmodels.")


# cols = st.columns(3)
# with cols[0]:
#     # with ui.card():
#     #     ui.element()
#     ui.card(title="💭微风轻语", content="AI相关内容", description="Last update:跟我一起从头开始学AI-如何...", key="card1").render()
#     with st.expander("More Notes"):
#         st.markdown("""
# ##### ⭐ AI将如何革命教育
# * AI将如何革命教育-The Global Classroom-理解和解决教育不平等[Go](https://mp.weixin.qq.com/s?__biz=MzkzMzYzMDQ3NQ==&mid=2247484000&idx=1&sn=348d5778fbf423ac60ebda2b18efe92f&chksm=c248c504f53f4c123dad8830783ff77c3efdd283a18fbd965209d4b428682db8cb6b11a85d70&token=209354273&lang=zh_CN#rd)
# * [新书导读]AI将如何革命教育-来自可汗学院Salman Khan[Go](https://mp.weixin.qq.com/s?__biz=MzkzMzYzMDQ3NQ==&mid=2247483956&idx=1&sn=e320f2918f071ff0d21b0edfcf66bbd9&chksm=c248c550f53f4c460c2e38e50cca4ab0ef9a4823bcc31c598de5a349226caf57790e17a5c1e5#rd)           
# * AI如何革命教育3核心-来自AI的高剂量辅导[Go](https://mp.weixin.qq.com/s?__biz=MzkzMzYzMDQ3NQ==&mid=2247484020&idx=1&sn=7f9352815c0cc9d86620b08092e1f828&chksm=c248c510f53f4c06426356cc9c88329ab64d73a22ff88a5b513c82acb250f7816ae6ae81180a&token=170394764&lang=zh_CN#rd)
# * AI将如何革命教育-创建AI家庭教师智能体(上)[Go](https://mp.weixin.qq.com/s/UJGJx_MM2Pb8ILNx8CKrXg)
                    
# ##### 🧙‍♂️ 跟我一起学从头开始学AI
# * 跟我一起从头开始学AI-试用不同的大语言模型[Go](https://mp.weixin.qq.com/s?__biz=MzkzMzYzMDQ3NQ==&mid=2247484066&idx=1&sn=fc955875087547d3b2eef08e5a938ae8&chksm=c248c5c6f53f4cd0c22d1aba62566a6c7d9143605c322edb8b2b5d00fcd3c225943bc1737b17&token=170394764&lang=zh_CN#rd)
# * 跟我一起从头开始学AI-如何与大语言模型对话[Go](https://mp.weixin.qq.com/s?__biz=MzkzMzYzMDQ3NQ==&mid=2247484089&idx=1&sn=14723cbbe3f2574cdcba053ac371708f&chksm=c248c5ddf53f4ccb0e7ff5c61eb61e74c3d51c1104bc95c7adc75b1df5b1f75be5a694effb08&token=34939858&lang=zh_CN#rd)
# * 跟我一起从头开始学AI-创建简单的AI(CoT)智能体[Go](https://mp.weixin.qq.com/s/qA1CyIqYq6MxPMkONE5tqw)
                    
#                     """, unsafe_allow_html=True)

# with cols[1]:
#     ui.card(title="⭐Breecho", content="随便写写", description="Last update:在A池大摇大摆做硝化...", key="card2").render()
#     with st.expander("More Notes"):
#         st.markdown("""
# ##### ⭐ Breecho
# * 在A池大摇大摆做硝化-MABR技术的原理与应用案例[Go](https://mp.weixin.qq.com/s/VY1p27hiE8mpQPqK24Jrlg)
# * 如何阅读一篇学术论文 [译][转][Go](https://mp.weixin.qq.com/s?__biz=MzkzMzYzMDQ3NQ==&mid=2247483831&idx=1&sn=e9e98b01fe30f98e2ad5281713b5122b&chksm=c248c6d3f53f4fc5553b944d5e83676cadfa156b614961727736095c0ff460ec172b9e891d76&token=571155477&lang=zh_CN#rd)
# * 关于三遍阅读法用于GPT的提示词的尝试[Go](https://mp.weixin.qq.com/s?__biz=MzkzMzYzMDQ3NQ==&mid=2247483836&idx=1&sn=f7098f875c3a2f671fc3ae8b4ad427dd&chksm=c248c6d8f53f4fce271ace88e41853b192f992a91286852649057f21b30bffaffc56f5e05e8b&token=571155477&lang=zh_CN#rd)
# * Data Analysis(GPT-4)教我使用python进行数据清洗与分析1[Go](https://mp.weixin.qq.com/s?__biz=MzkzMzYzMDQ3NQ==&mid=2247483746&idx=1&sn=efb8026ae7c3c04a7048f4399110682f&chksm=c248c606f53f4f10a906ee8f9df0fab8336c582e28ec56861c571bbb19e57143944993ad00fe&token=571155477&lang=zh_CN#rd)
             
# """, unsafe_allow_html=True)
# with cols[2]:
#     ui.card(title="📢耳边风", content="环境专业相关内容", description="Last update:breecho已上线ASM1模型...", key="card3").render()
#     with st.expander("More Notes"):
#         st.markdown("""
# ##### 📢 2024 重写ASM1学习笔记
# * 活性污泥模型ASM1基础-Monod方程[Go](https://mp.weixin.qq.com/s/Blf04VvC3mrycVac4cknfw)
# * 活性污泥模型ASM1基础-底物与过程[Go](https://mp.weixin.qq.com/s/LWGZY64qdOkscYQX5G3LRg)
# * 说一些关于活性污泥模型ASM1的计算应用[go](https://mp.weixin.qq.com/s/aRSdppMESvgmwgvl24ylKQ)
                    
# ##### 👨‍👦‍👦 2024 ASM1污水生物模型笔记

# * ASM (Activated Sludge Models) 活性污泥模型学习笔记#1模型与微生物角色的理解[Go](https://mp.weixin.qq.com/s?__biz=MzkzMzYzMDQ3NQ==&mid=2247483660&idx=1&sn=b28b1f7c2e17a1e6f4477adb0ec56000&chksm=c248c668f53f4f7e0e1abc0a062d7523ba3115670fdd18dfa3d60d6973f3d905e52e131c622e#rd)
# * ASM污水生物模型学习笔记#2模型参数和速率的计算[Go](https://mp.weixin.qq.com/s?__biz=MzkzMzYzMDQ3NQ==&mid=2247483671&idx=1&sn=037c5222d8acdee122e771905c0d58f9&chksm=c248c673f53f4f6587f8805ae6d08ab5471453c4a9878f44c83f1d995636041a51a246ff28a8#rd)
# * ASM污水生物模型学习笔记#3水平衡与物料平衡（上）[Go](https://mp.weixin.qq.com/s?__biz=MzkzMzYzMDQ3NQ==&mid=2247483682&idx=1&sn=80a6ae927261d1fc8a5ffac64f673875&chksm=c248c646f53f4f50aed55aa780bc3a3dec0c46d2a6eb8ac20b22930212820c60f17d020ce7dd&token=571155477&lang=zh_CN#rd)
# * Activated Sludge Model Study Notes #4: Water and Mass Balance 2[Go](https://mp.weixin.qq.com/s?__biz=MzkzMzYzMDQ3NQ==&mid=2247483691&idx=1&sn=898ba5ff3468e2f988168908dafd01af&chksm=c248c64ff53f4f59d969068fe3d7c57ae5261cce131b57c72c2952e5e9d4cd2ef3fcaf0a63f8&token=571155477&lang=zh_CN#rd)
# * ASM活性污泥模型学习笔记#5[Go](https://mp.weixin.qq.com/s?__biz=MzkzMzYzMDQ3NQ==&mid=2247483704&idx=1&sn=013f6887ed89d103348bb889e3e14741&chksm=c248c65cf53f4f4a53b6f0879887916ed6533970734c53af267c2fce9868933738f99789ff5e&token=571155477&lang=zh_CN#rd)

# ##### 👨‍🔧 2024 二沉池学习笔记
            
# * 二沉池（Secondary Settling Tank）的功能[Go](https://mp.weixin.qq.com/s?__biz=MzkzMzYzMDQ3NQ==&mid=2247483715&idx=1&sn=b1898d79e7d3fd0fe2738a0feffac312&chksm=c248c627f53f4f31c65ed8fed9e67c6d07084d3de579062156d9d92694ad42d93420377497c8&token=571155477&lang=zh_CN#rd)
# * 二沉池的一维建模基础[Go](https://mp.weixin.qq.com/s?__biz=MzkzMzYzMDQ3NQ==&mid=2247483728&idx=1&sn=5b13ca4d8cf5c00eb421b29955bc36a4&chksm=c248c634f53f4f22696eb9980042c4c777234b3274d28e833efb837b5fa85e6d9a6ff4900c0f&token=571155477&lang=zh_CN#rd)
# * 二沉池的一维建模[Go](https://mp.weixin.qq.com/s?__biz=MzkzMzYzMDQ3NQ==&mid=2247483785&idx=1&sn=4d818db5567fa1034aad775c9391a009&chksm=c248c6edf53f4ffbc5a942f988f83f35b8c7e07aed68c1f84461aaf5ffe14e5f17150b4d8dfd&token=571155477&lang=zh_CN#rd)
# * 掌握生化系统之匙：二沉池建模的关键作用[Go](https://mp.weixin.qq.com/s?__biz=MzkzMzYzMDQ3NQ==&mid=2247483823&idx=1&sn=78f9cd3e603bb76cbfe869a03d93f316&chksm=c248c6cbf53f4fddbe8a22278ade894524a831a7bfb3586472e3d70a02cd011bf21f655a50be&token=571155477&lang=zh_CN#rd)
# * 二沉池的设计-关于污泥体积指数SVI[Go](https://mp.weixin.qq.com/s?__biz=MzkzMzYzMDQ3NQ==&mid=2247483848&idx=1&sn=af59fd888a84682081b80a7f6f6e4b47&chksm=c248c6acf53f4fba13e3ad3dce6102cca1f707129c014e6b9abbfebcba285764450bc47108c9&token=571155477&lang=zh_CN#rd)
# * 二沉池的设计-DWA德国水协会设计方法[Go](https://mp.weixin.qq.com/s?__biz=MzkzMzYzMDQ3NQ==&mid=2247483924&idx=1&sn=7beda6304856f3ebf4529197b347cb88&chksm=c248c570f53f4c666a40bc4772cfd8708c6e2f3ccea8084dde6b38c0d97566ae901566b6f3a0&token=571155477&lang=zh_CN#rd)

# ##### 👨‍🏫 2024 AO工艺设计学习笔记
            
# * 关于硝化最小泥龄的确定[Go](https://mp.weixin.qq.com/s?__biz=MzkzMzYzMDQ3NQ==&mid=2247483870&idx=1&sn=cb4c46c02d5e5ddf363718d504937606&chksm=c248c6baf53f4faccf0fd4cb96144605dd827ee5730e30e5eb2316bbdc8f5f941c7bc7a794a9&token=571155477&lang=zh_CN#rd)
# * 反硝化体积比例的确定[Go](https://mp.weixin.qq.com/s?__biz=MzkzMzYzMDQ3NQ==&mid=2247483891&idx=1&sn=8e810919cc21e080f292e85100e858e6&chksm=c248c697f53f4f81726f547cd746883ef1e9def5191c65fb363e3cd1864c09fad9915c9e60d1&token=571155477&lang=zh_CN#rd)
# * 室外排水设计标准(2021GB)缺氧/好氧设计Streamlit简单应用[Go](https://mp.weixin.qq.com/s?__biz=MzkzMzYzMDQ3NQ==&mid=2247483941&idx=1&sn=4928de41f7d6cf0533428070b2c7feb1&chksm=c248c541f53f4c57c883527a7dbd41db7894e461f423d456b03e5208bc6306b95de26b978b0a&token=571155477&lang=zh_CN#rd)
# * 德国DWA-131-A AO工艺设计计算[Go](https://mp.weixin.qq.com/s?__biz=MzkzMzYzMDQ3NQ==&mid=2247483977&idx=1&sn=63d1672a0d709023b999f63009cab07d&chksm=c248c52df53f4c3b230fce8285d94ac685355bfc998a75cd06d3faf86ca984fb71c99d2ed89c&token=209354273&lang=zh_CN#rd)
                    
# ##### ⭐ 2024 其他 
# * breecho已上线ASM1模型简易计算应用[Go](https://mp.weixin.qq.com/s/tbzTzgh2va4dCkJMV4Ys8A)
# * 【文献翻译】一种澄清-浓缩过程的动态模型[Go](https://mp.weixin.qq.com/s?__biz=MzkzMzYzMDQ3NQ==&mid=2247483799&idx=1&sn=cffd150a265c0d256951d6e4b9ea966f&chksm=c248c6f3f53f4fe58243b1d784b922a6bb57906434fead711f05098aac32e4e6839a32bdf4b9&token=571155477&lang=zh_CN#rd)
# * 说好的计算功能网页已经完成预览[Go](https://mp.weixin.qq.com/s?__biz=MzkzMzYzMDQ3NQ==&mid=2247483988&idx=1&sn=b599795fed488ad845585ebbbeaf1d7e&chksm=c248c530f53f4c2671c82126d8ed5c490247d61cf19c696de59fbb94cc528ce32bad798281fa&token=209354273&lang=zh_CN#rd)
# * 带计算功能的网页应用breecho.cn已经可以正常访问[Go](https://mp.weixin.qq.com/s?__biz=MzkzMzYzMDQ3NQ==&mid=2247484030&idx=1&sn=dab49bafaaae1cef2e6870acca931c48&chksm=c248c51af53f4c0c41d16bb3716a1a9ce76d8430515f62027d62477e5b36aad25425e943313d&token=170394764&lang=zh_CN#rd)                               
# """, unsafe_allow_html=True)

cols_1 = st.columns(3)

cards_data = [
    {
        "title": "✨New!储水池余氯衰减高级模拟系统(源自EPANET)", 
        "key": "image6",
        "img_src": f"data:image/png;base64,{epanet_img}",
        "btn_text": "点击打开储水池余氯模拟",
        "url": "https://breecho.cn/New!%E5%82%A8%E6%B0%B4%E6%B1%A0%E4%BD%99%E6%B0%AF%E8%A1%B0%E5%87%8F%E9%AB%98%E7%BA%A7%E6%A8%A1%E6%8B%9F%E7%B3%BB%E7%BB%9F(%E6%BA%90%E8%87%AAEPANET)"
    },
    {
        "title": "⭐New!新ASM简易不简单-易用又准确模型",
        "key": "image5",
        "img_src": f"data:image/png;base64,{asm2_img}",
        "btn_text": "点击打开新ASM模型",
        "url": "https://breecho.cn/New!%E6%96%B0ASM%E7%AE%80%E6%98%93%E4%B8%8D%E7%AE%80%E5%8D%95-%E6%98%93%E7%94%A8%E5%8F%88%E5%87%86%E7%A1%AE%E6%A8%A1%E5%9E%8B"
    },
    {
        "title": "💭公众号文档阅读AI助手",
        "key": "image8",
        "img_src": f"data:image/png;base64,{qa_img}",
        "btn_text": "点击打开公众号文档阅读AI助手",
        "url": "https://breecho.cn/QA%E5%A4%84%E7%90%86"
    },
    {
        "title": "📢ASM1活性污泥模型计算",
        "key": "image4",
        "img_src": f"data:image/png;base64,{asm1_img}",
        "btn_text": "点击打开ASM1模型计算",
        "url": "https://breecho.cn/ASM1%E6%B4%BB%E6%80%A7%E6%B1%A1%E6%B3%A5%E6%A8%A1%E5%9E%8B%E8%AE%A1%E7%AE%97"
    },
    {
        "title": "🎯室外排水设计标准AO计算",
        "key": "image1",
        "img_src": f"data:image/png;base64,{gb_img}",
        "btn_text": "点击打开室外排水设计标准AO计算",
        "url": "https://breecho.cn/%E5%AE%A4%E5%A4%96%E6%8E%92%E6%B0%B4%E8%AE%BE%E8%AE%A1%E6%A0%87%E5%87%86AO%E8%AE%A1%E7%AE%97"
    },
    {
        "title": "💯德国DWA-131-A AO计算",
        "key": "image2", 
        "img_src": f"data:image/png;base64,{dwa_img}",
        "btn_text": "点击打开德国DWA-131-A AO计算",
        "url": "https://breecho.cn/%E5%BE%B7%E5%9B%BDDWA-131-A%20AO%E8%AE%A1%E7%AE%97"
    },
    {
        "title": "🚀二沉池一维建模计算",
        "key": "image3",
        "img_src": f"data:image/png;base64,{takacs_img}",
        "btn_text": "点击打开二沉池一维建模计算", 
        "url": "https://breecho.cn/%E4%BA%8C%E6%B2%89%E6%B1%A0%E4%B8%80%E7%BB%B4%E5%BB%BA%E6%A8%A1%E8%AE%A1%E7%AE%97"
    },
    {
        "title": "🎯24点计算器",
        "key": "image7",
        "img_src": f"data:image/png;base64,{twentyfour_img}",
        "btn_text": "点击打开24点计算器",
        "url": "https://breecho.cn/24%E7%82%B9%E8%AE%A1%E7%AE%97"
    },
    {
        "title": "🤖excel分析助手demo",
        "key": "image9",
        "img_src": f"data:image/png;base64,{app_img}",
        "btn_text": "点击打开excel分析助手demo",
        "url": "https://breecho.cn/excel%E5%88%86%E6%9E%90%E5%8A%A9%E6%89%8Bdemo"
    }   
]

for i, card in enumerate(cards_data):
    with cols_1[i%3]:
        with ui.card(card["title"], key=card["key"]):
            if card["img_src"]:
                ui.element("img", src=card["img_src"], className="w-full")
            ui.element("link_button", text=card["btn_text"], url=card["url"], className="mt-2", key=f"link_btn{i}")
        

st.markdown("---")

cols_2 = st.columns([2,1])

with cols_2[0]:
    st.markdown("""### 🔎 学习玩乐笔记 by [StudyWithFun]()           
- 最近在学着搭建可以用来计算的网页，方便学习理解与实践；
- 接下来会做一些PHREEQC的笔记课程，包括PHREEQC的理解、计算和模拟；

        """
        , unsafe_allow_html=True)
with cols_2[1]:
    with st.expander("📢欢迎关注我的公众号"):
        st.image("/opt/stapp/weixin.png")
