import streamlit as st
import numpy as np
import torch
from torchdiffeq import odeint
import matplotlib.pyplot as plt
import functools
from st_pages import add_page_title, hide_pages

# 设置页面配置
st.set_page_config(
    page_title="微风轻语BreeCho",  # 自定义页面标题
    page_icon="💭",  # 可以是一个URL链接，或者是emoji表情符号
    layout="wide",  # 页面布局: "centered" 或 "wide"
    initial_sidebar_state="auto",  # 初始侧边栏状态: "auto", "expanded", "collapsed"
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


add_page_title(layout="wide")

hide_pages(["Thank you"])

# 定义污泥通量分布函数 Takacs双指数模型
def delta_Takacs(C, q_out, V_zs, r_h, r_p, M, v0):
    C = C.unsqueeze(1).repeat(1, M)
    J = torch.zeros(M, M)
    J_q = q_out * C # 每一层上、下流带出的污泥通量 kg/(m2*h)
    # 该方法未考虑实际最大沉降速率v0
    V_max= torch.ones(M,M) * v0
    J_s = torch.minimum(V_max, V_zs*(torch.exp(-r_h*C)-torch.exp(-r_p*C))) * C # 每一层污泥沉降的污泥通量 kg/(m2*h)
    J_s[:-1,:-1] = torch.minimum(J_s[:-1,:-1], J_s[1:,1:]) # 防止污泥沉降速度过大，导致污泥通量过大
    J_s[-2,-1] = 0 # 最底层污泥沉降为0
    J = J_q + J_s # 每一层流出通量分布
    J_out = J.sum(dim=1) # 每一层流出通量求和
    J_in = J.sum(dim=0) # 每一层流入通量求和
    delta = J_in - J_out # 通量差
    return delta 


def oneDTakacs_model(t, y, q_out, V_zs, r_p, r_h, M, delta_h, v0):
    y[y < 0] = 0
    delta = delta_Takacs(y, q_out, V_zs, r_h, r_p, M, v0)
    dy = torch.zeros_like(y)
    dy[1:-1] = delta[1:-1] /delta_h
    return dy

def Takacs_modelrun(hours, N, N_in, depth, q_A, RV, sludge_concentration, sludge_settling_velocity, r_h,v0, r_p):
    M = N + 2 # 增加进水和排水层
    # q_in = inflow_rate / area # 进水流速率 m/h
    q_ov = q_A # 沉淀池水流上流速率 m/h
    q_re = RV*q_A # 回流下流速率 m/h
    delta_h = depth / N # 每层层高
    q_out = torch.zeros(M, M)
    for i in range(N+1):
        if i == 0: # 假设沉淀池的进水来自于外部虚拟的第0层，沉淀池出水也排至虚拟的第0层
            q_out[i, N_in] = q_re+q_ov
        elif i < N_in: # 进水层以上，逐层定义上流速率
            q_out[i, i-1] = q_ov
        elif i == N_in: # 进水层，同时具有上流和下流
            q_out[i, i-1] = q_ov
            q_out[i, i+1] = q_re
        else: # 进水层以下，逐层定义下流速率
            q_out[i, i+1] = q_re
    # 定义污泥下沉速率初始值 m/h
    V_zs = torch.zeros(M, M) 
    for i in range(1,M-1):
        V_zs[i ,i+1] = sludge_settling_velocity
    # 定义污泥初始浓度 kg/m3
    C = torch.zeros(M)
    # 定义进水污泥浓度 kg/m3
    C[0] = sludge_concentration
    t0 = torch.linspace(0, hours, hours*150)
    Takacs_modified = functools.partial(oneDTakacs_model, q_out=q_out, V_zs=V_zs, r_p=r_p, r_h=r_h, M=M, delta_h=delta_h, v0=v0)
    x= odeint (Takacs_modified,C,t0, method='rk4') 
    return x[-1]

def show_page1():
    # st.title('💭室外排水设计标准AO计算')
    st.sidebar.header('1️⃣二沉池一维模型计算')
    input_labels_1 = [
        '二沉池深度 m',
        '设计表面负荷 m/h',
        '污泥回流比',
        '进水污泥浓度 kg/m3',
        '理论最大沉降速率 m/h',
        '污泥沉降速率系数 r_h',
        '实际最大沉降速率 m/h',
    ]
    defaults_1 = [4.5, 1.2, 0.5, 5.62, 29.7, 0.326, 16.6]
    inputs_1 = []

    for i, (label, default_value) in enumerate(zip(input_labels_1, defaults_1)):
        key = f"input_1_{label.replace(' ', '_')}"
        if '二沉池深度' in label:
            input_val = st.sidebar.slider(label, 2.0, 6.0, float(default_value), 0.1, key=key)
        elif '设计表面负荷' in label:
            input_val = st.sidebar.slider(label, 0.6, 1.6, float(default_value), 0.1, key=key)
        elif '污泥回流比' in label:
            input_val = st.sidebar.slider(label, 0.5, 1.2, float(default_value), 0.1, key=key)
        elif '进水污泥浓度' in label:
            input_val = st.sidebar.slider(label, 1.0, 9.0, float(default_value), 0.01, key=key)
        elif '理论最大沉降速率' in label:
            input_val = st.sidebar.slider(label, 10.0, 40.0, float(default_value), 0.1, key=key)
        elif '污泥沉降速率系数' in label:
            input_val = st.sidebar.slider(label, 0.250, 0.500, float(default_value), 0.001, key=key)
        elif '实际最大沉降速率' in label:
            input_val = st.sidebar.slider(label, 10.0, 40.0, float(default_value), 0.1, key=key)
        else:
            input_val = st.sidebar.text_input(label, str(default_value), key=key)
        inputs_1.append(input_val)

    st.markdown('---')
    # st.header("💻计算结果")
    col1, col2 = st.columns(2)
    with col1:
        if st.button('请点击按钮进行计算'):
            try:
                inputs_1 = [float(val) for val in inputs_1]
                result = Takacs_modelrun(10,10,6,*inputs_1,5)*1000

                output_str = "**二沉池每层污泥浓度** \n\n"      
                output_str += f"进水浓度: {result[0]:.2f} mg/L \n\n"
                output_str += f"第1层污泥浓度 : {result[1]:.2f} mg/L \n\n"
                output_str += f"第2层污泥浓度 : {result[2]:.2f} mg/L \n\n"
                output_str += f"第3层污泥浓度 : {result[3]:.2f} mg/L \n\n"
                output_str += f"第4层污泥浓度 : {result[4]:.2f} mg/L \n\n"
                output_str += f"第5层污泥浓度 : {result[5]:.2f} mg/L \n\n"
                output_str += f"第6层污泥浓度 : {result[6]:.2f} mg/L \n\n"
                output_str += f"第7层污泥浓度 : {result[7]:.2f} mg/L \n\n"
                output_str += f"第8层污泥浓度 : {result[8]:.2f} mg/L \n\n"
                output_str += f"第9层污泥浓度 : {result[9]:.2f} mg/L \n\n"
                output_str += f"第10层污泥浓度 : {result[10]:.2f} mg/L \n\n"
                output_str += f"回流污泥浓度 : {result[10]:.2f} mg/L \n\n"
                
                
            
                with st.expander("💻计算结果",expanded=True):
                    st.markdown(f" \n{output_str}\n ")
                with st.expander("💻污泥浓度对比图",expanded=True):
                    st.markdown('**每层污泥浓度对比图**')

                    y = torch.linspace(1, 10, 10)

                    # 创建Matplotlib图形
                    fig_1, ax_1 = plt.subplots(figsize=(16, 7))  # 使用 plt.subplots() 创建 Figure 和 Axes

                    # 绘制图形
                    ax_1.plot(y, result[1:-1], color='orange', label='Sludge_concentration mg/L')
                    ax_1.set_yscale('log')
                    ax_1.set_ylabel('Sludge_concentration mg/L')
                    ax_1.set_xlabel('Height NO.')
                    ax_1.set_xticks(np.arange(0, 11, 1))
                    ax_1.set_ylim(bottom=1)
                    ax_1.set_title('Relationship between Height NO. and Sludge_concentration')
                    ax_1.grid(True, linestyle='--')
                    ax_1.legend()

                    # 使用 Streamlit 显示 Matplotlib 图形
                    st.pyplot(fig_1)

            except ValueError:
                st.error('请输入正确的数值')
    with col2:
        with st.expander("一维沉淀池示意图",expanded=True):
            st.image('/opt/stapp/takacs1.png') 
        with st.expander("一维沉淀池计算示意图",expanded=True):
            st.image('/opt/stapp/takacs2.png')   
def main():
    show_page1()

if __name__ == '__main__':
    main()
