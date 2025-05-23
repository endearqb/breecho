import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from st_pages import add_page_title, hide_pages
import streamlit_shadcn_ui as ui
import pandas as pd


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

# custom_css = """
# <style>
# @font-face {
#     font-family: 'LXGW WenKai GB';
#     src: url('/static/LXGWWenKaiGB.ttf') format('truetype');
# }
# body, p, h1, h2, h3, h4, h5, h6, div {
#     font-family: 'LXGW WenKai GB', serif !important;
# }
# </style>
# """

# st.markdown(custom_css, unsafe_allow_html=True)

add_page_title(layout="wide")

hide_pages(["Thank you"])
# streamlit run dwaweb.py
# 以下计算代码来自https://gitee.com/wang-nan-watertreatment/BioReactor
def V_D_over_V_BB_F(f_s, f_A, f_COD, f_B, S_orgN_AN, Y_COD_abb, b, C_0, C_S,
           S_NO3_ZB, miu_A_max, 
           Q_d_Knoz, C_COD_ZB, C_BOD5_ZB, C_P_ZB, C_TN_ZB, C_SS_ZB, T_C, 
           S_COD_AN, S_BOD5_AN, S_TP_AN, S_TN_AN, S_NH4_AN, S_SS_AN, TS_BB,
           alfa, beta, h_TB2A, h_tk, h_El, E_A, COD_dos_name, P_dos_name, max):
    # 原始代码中的函数代码
    # ...
    # 计算设计流量
    Q_h_Knoz = Q_d_Knoz / 24
    if Q_h_Knoz <= 13:
        Kz = 2.7
    elif Q_h_Knoz >= 2600:
        Kz = 1.5
    else:
        Kz = 3.5778 * Q_h_Knoz**(-0.112) #变化系数
    Q_d_max = Q_d_Knoz * Kz

    # 计算碳平衡

    X_TS_ZB = C_SS_ZB #进水可过滤物质
    X_COD_ZB = X_TS_ZB * 1.6 * (1-f_B) #颗粒性COD(可过滤物质COD),有机干物质颗粒按1.6gCOD/oTS计
    S_COD_ZB = C_COD_ZB - X_COD_ZB #可溶解性COD
    S_COD_inert_ZB = f_s * C_COD_ZB #溶解性惰性组分
    X_COD_inert_ZB = f_A * X_COD_ZB #颗粒性惰性组分
    C_COD_abb_ZB = C_COD_ZB - S_COD_inert_ZB - X_COD_inert_ZB #可降解COD
    C_COD_la_ZB = f_COD * C_COD_abb_ZB #易降解COD
    X_anorg_TS_ZB = f_B * X_TS_ZB #进水可过滤无机物质(仅算数,进水颗粒性COD没有直接用)

    # 计算出水氮平衡

    S_TKN_AN = S_NH4_AN + S_orgN_AN  #mg/L,出水凯氏氮
    S_anorgN_UW = S_TN_AN - S_TKN_AN #mg/L,出水硝酸盐氮

    # 计算硝化菌泥龄
    B_d_COD_Z = Q_d_Knoz * C_COD_ZB / 1000 #mg/L,COD日负荷
    if B_d_COD_Z <= 2400:
        PF = 2.1
    elif B_d_COD_Z > 12000:
        PF = 1.5
    else:
        PF = 2.1 - (B_d_COD_Z - 2400) * 0.6 /9600
    t_TS_aerob_Bem = PF * 1.6 / miu_A_max * 1.103**(15-T_C) #d,硝化菌污泥龄

    #投加碳源类型
    if COD_dos_name == '甲醇': 
        Y_COD_dos = 0.45
    elif COD_dos_name == '乙醇' or '醋酸': 
        Y_COD_dos = 0.42
    F_T = 1.072**(T_C-15) #内源呼吸的衰减系数
    C_COD_dos_f = 0     #外加碳源化学需氧量
    V_D_over_V_BB_f = 0.2
    x_f = 0 
    while x_f < 1:
        #4.1污泥产量的计算
        t_TS_Bem_f = t_TS_aerob_Bem / (1-V_D_over_V_BB_f) #设计污泥泥龄
        X_COD_BM_f = (C_COD_abb_ZB * Y_COD_abb + C_COD_dos_f * \
            Y_COD_dos)/(1 + b * t_TS_Bem_f * F_T)     #生物体中的COD
        X_COD_inert_BM_f = 0.2 * X_COD_BM_f * t_TS_Bem_f * b * F_T  #剩余惰性固体
        US_d_C_f = Q_d_Knoz * (X_COD_inert_ZB / 1.33 +(X_COD_BM_f + \
            X_COD_inert_ZB) / (0.93 * 1.42) + f_B * X_TS_ZB) / 1000 #污泥产量
        #4.2反硝化硝态氮浓度计算
        S_NO3_AN_f = 0.7 * S_anorgN_UW #出水硝态氮
        X_orngN_BM_f = 0.07 * X_COD_BM_f #形成活性污泥的氮
        X_orgN_inert_f = 0.03 * (X_COD_inert_BM_f + X_COD_inert_ZB) #与惰性颗粒结合的氮
        S_NO3_D_f = C_TN_ZB - S_NO3_AN_f - S_orgN_AN - S_NH4_AN - \
            X_orngN_BM_f - X_orgN_inert_f #每日平均反硝化的硝态氮浓度
        #4.3碳降解的需氧量
        OV_C_f = C_COD_abb_ZB + C_COD_dos_f - X_COD_BM_f - \
            X_COD_inert_BM_f #碳降解的总需氧量
        OV_C_la_vorg_f = f_COD * C_COD_abb_ZB * (1-Y_COD_abb) +\
            C_COD_dos_f*(1-Y_COD_dos) #反硝化区易降解及外加碳源需氧量
        OV_C_D_f = 0.75 * (OV_C_la_vorg_f + (OV_C_f - OV_C_la_vorg_f) *\
            V_D_over_V_BB_f**0.68) #反硝化区总需氧量
        #4.4耗氧量和供氧量平衡
        x_f = OV_C_D_f / 2.86 / S_NO3_D_f
        if V_D_over_V_BB_f < max and x_f < 1:
            V_D_over_V_BB_f += 0.01
        elif V_D_over_V_BB_f >= max and x_f < 1:
            V_D_over_V_BB_f = max
            C_COD_dos_f += 0.01

    T_TS_D_Bem = t_TS_Bem_f - t_TS_aerob_Bem #d,反硝化菌泥龄

    #5.1 生物处理与化学除磷量
    C_P_AN = 0.7 * S_TP_AN  #mg/L,出水浓度
    X_P_BM = 0.005 * C_COD_ZB   #形mg/L,成活性污泥的氮
    X_P_BioP = 0.006 * C_COD_ZB    #mg/L,生物法除磷量
    X_P_Fall = C_P_ZB - C_P_AN - X_P_BM - X_P_BioP  #mg/L,需要沉析的磷酸盐
    Me_3plus = 1.5 * X_P_Fall / 31   #mol/L,化学除磷药剂投加量

    #5.2 除磷污泥产量
    if P_dos_name == '铝盐':
        X_P_Fall_Fe = 0  #折合铁盐投加量
        X_P_Fall_Al = 27 * Me_3plus     #折合铝盐投加量
    elif P_dos_name == '铁盐':
        X_P_Fall_Fe = 55.8 * Me_3plus   #mg/L,折合铁盐投加量
        X_P_Fall_Al = 0   #mg/L,折合铝盐投加量   
    US_d_P = Q_d_Knoz * (3 * X_P_BioP + 6.8 * X_P_Fall_Fe + 5.3 * X_P_Fall_Al) / 1000   #化学除磷产泥量
    #5.3 污泥产量
    US_d_r = US_d_C_f + US_d_P    #kg/d,剩余污泥量
    M_TS_BB = t_TS_Bem_f * US_d_r   #kg,生物段保持的污泥质量
    M_TS_D = V_D_over_V_BB_f * M_TS_BB    #kg,缺氧池污泥量
    M_TS_aero = M_TS_BB - M_TS_D    #kg,好氧池污泥量
    K_de = Q_d_Knoz * S_NO3_D_f / M_TS_D / 1000   #kgN/kgSS·d,反硝化速率
    L_C = (C_COD_ZB - S_COD_AN) * Q_d_Knoz / 1000 / M_TS_aero #kgCOD/kgSS·d,好氧池COD负荷
    L_B = (C_BOD5_ZB - S_BOD5_AN) * Q_d_Knoz / 1000 / M_TS_aero #kgBOD/kgSS·d,好氧池BOD负荷

    #7.1生物池容积
    V_BB = M_TS_BB / TS_BB  #m3,曝气池的容积
    V_an = 1 * Q_h_Knoz     #m3,厌氧池容积
    V_D = V_BB * V_D_over_V_BB_f  #m3,缺氧池容积
    V_aero = V_BB - V_D    #m3,好氧池容积
    V_bioT = V_BB + V_an    #m3,总容积
    HRT_an = V_an / Q_h_Knoz     #h,厌氧池水力停留时间
    HRT_D = V_D / Q_h_Knoz  #h,缺氧池水力停留时间
    HRT_aero = V_aero / Q_h_Knoz   #h,好氧池水力停留时间
    HRT_bioT = V_bioT / Q_h_Knoz    #h,总水力停留时间
    #7.2 回流比
    RF = (S_NO3_D_f-S_NO3_ZB) / S_NO3_AN_f     #反硝化所需的回流比
    RZ = RF - 1  #反硝化所需的内回流比
    eta_0 = 1 - 1 / (1 + RF)        #反硝化最大效率 

    #8.1 耗氧量物料平衡
    OV_d_C = Q_d_Knoz * OV_C_f / 1000     #kgO2/d, 碳去除的耗氧量
    OV_d_N = Q_d_Knoz * 4.3 * (S_NO3_D_f - S_NO3_ZB + S_NO3_AN_f) / 1000   #kgO2/d, 反硝化回收供氧量
    OV_d_D = Q_d_Knoz * 2.86 * S_NO3_D_f / 1000   #kgO2/d, 反硝化回收供氧量
    OV_h_aM = ((OV_d_C - OV_d_D) + OV_d_N) / 24     #kgO2/h，平均耗氧量
    OV_h_max = Kz * OV_h_aM     #kgO2/h，最高耗氧量  

    # 8.2 标准传氧速率
    O_t = 21 * (1 - E_A) / (79 + 21 * (1 - E_A))
    P_a = (101325 - h_El /12 / 133) / 1000000 #Mpa,当地大气压力
    P_b = P_a + (h_tk - h_TB2A) * 9.81  / 1000 #Mpa曝气装置处绝对压力
    C_SW = 8.24 * P_a / 0.101325    #mg/L,清水表面饱和溶解氧
    C_SM = C_SW * (O_t / 42 + P_b / (2 * P_a)) #mg/L,水下深度到池面清水平均溶氧值
    FCF = alfa * (beta * C_SM - C_0) / C_S #AOR与SOR转换系数
    SOR = OV_h_aM  / FCF #kgO2/h,标准传氧速率SOR
    G_S = SOR / (0.28 * E_A)
    V_GS_over_V_knoz = G_S / Q_h_Knoz   

    # 计算二沉池的各项参数
      
    return(Q_d_max,X_TS_ZB,X_COD_ZB,S_COD_ZB,S_COD_inert_ZB,X_COD_inert_ZB,C_COD_abb_ZB,C_COD_la_ZB,
           X_anorg_TS_ZB,S_TKN_AN,S_anorgN_UW,B_d_COD_Z,t_TS_aerob_Bem,F_T,T_TS_D_Bem,C_COD_dos_f,
           V_D_over_V_BB_f,x_f ,t_TS_Bem_f,X_COD_BM_f,X_COD_inert_BM_f,US_d_C_f,S_NO3_AN_f,X_orngN_BM_f,
           X_orgN_inert_f ,S_NO3_D_f,OV_C_f,OV_C_la_vorg_f,OV_C_D_f,
           C_P_AN,X_P_BM,X_P_BioP,X_P_Fall,Me_3plus,US_d_P,US_d_r,M_TS_BB,M_TS_D ,M_TS_aero,K_de,L_C,L_B,
           V_BB,V_an,V_D ,V_aero ,V_bioT ,HRT_an,HRT_D,HRT_aero,HRT_bioT,RF,RZ,eta_0,
           OV_d_C,OV_d_N,OV_d_D,OV_h_aM,OV_h_max,P_a,P_b,C_SW,C_SM,FCF,SOR,G_S,V_GS_over_V_knoz)

def sst(Q_M, DSVI, t_E, RV, f, q_A):
    Q_h_Knoz = Q_M / 24
    if Q_h_Knoz <= 13:
        Kz = 2.7
    elif Q_h_Knoz >= 2600:
        Kz = 1.5
    else:
        Kz = 3.5778 * Q_h_Knoz**(-0.112) #变化系数
    Q_d_max = Q_M * Kz
    TS_BS = 1000/DSVI * (t_E**(1/3))
    TS_RS = f*TS_BS    
    TS_BB = RV*TS_RS/(1+RV)
    DSV = TS_BB * DSVI
    Ast = Q_d_max/24/q_A
    h4 = TS_BB * q_A * (1+RV)*t_E/TS_BS
    h2 = 0.5*q_A*(1+RV)/(1-DSV/1000)
    h3 = 1.5*0.3*q_A*DSV*(1+RV)/500
    q_sv = q_A*DSV
    h1 = 0.5
    h = h1+h2+h3+h4
    return(Q_d_max, Ast, TS_BB, TS_RS, h1, h2, h3, h4, h, q_sv)


def show_page2():
    # st.title("DWA A/O工艺设计计算程序")
    # 第一列输入
    st.sidebar.header("0️⃣二沉池设计参数")
    input_labels_0 = [
        "日平均进水水量(m3/d)",
        "污泥体积指数DSVI(50-200L/kg)",
        "设计浓缩时间(1.0-2.5)",
        "污泥回流比(0.5-1.2)",
        "污泥回流短流系数(0.5-1)",
        "设计表面负荷(0.8-2)",        
    ]
    defaults_0 = ["10000","120","2.0","0.7","0.8","1.2"]
    inputs_0 = []

 
    for i, (label, default_value) in enumerate(zip(input_labels_0, defaults_0)):
        key = f"input_3_{label.replace(' ', '_')}"
        if '污泥体积指数' in label:
            input_val0 = st.sidebar.slider(label, 50, 200, int(default_value), 10, key=key)
        elif '设计浓缩时间' in label:
            input_val0 = st.sidebar.slider(label, 1.0, 2.5, float(default_value), 0.1, key=key)
        elif '污泥回流比' in label:
            input_val0 = st.sidebar.slider(label, 0.5, 1.2, float(default_value), 0.1, key=key)
        elif '设计表面负荷' in label:
            input_val0 = st.sidebar.slider(label, 0.8, 2.0, float(default_value), 0.1, key=key)
        elif '污泥回流短流系数' in label:
            input_val0 = st.sidebar.slider(label, 0.5, 1.0, float(default_value), 0.1, key=key)
        
        else:
            input_val0 = st.sidebar.text_input(label, str(default_value), key=key)
        inputs_0.append(input_val0)   



    st.sidebar.header("1️⃣常数或建议数值")
    input_labels_1 = [
        "溶解性的惰性COD占总COD比例(0.05-0.1)", 
        "颗粒性惰性组分比例(0.2-0.35)", 
        "易降解COD比例(0.15-0.25)",
        "进水可过滤无机物质,进厂污水取0.3,初沉池污水取0.2",
        "出水有机氮(mg/L)", 
        "可降解COD产泥系数(0.67)",
        "15℃衰减系数(0.17)", 
        "混合液剩余DO值 mg/L", 
        "标准条件下清水中饱和溶解氧 mg/L",
        # "计算混合液温度(标况)℃",
        # "夏季温度℃",
        # "管道阻力m", 
        # "曝气器水头损失m",
        # "每升高1℃需补偿压力值m", 
        "设定进水硝酸盐氮", 
        "15℃硝化菌最大比生长速率(0.47)"
    ]

    # 为每个输入框添加默认值
    defaults_1 = ["0.1", "0.3", "0.2", "0.3", "2.0", "0.67", "0.17",  "2.0", "9.17", "0", "0.47"]
    # inputs_1 = [st.sidebar.text_input(label, default_value, key=f"input_{i}") for i, (label, default_value) in enumerate(zip(input_labels_1, defaults_1))]

    inputs_1 = []

    for i, (label, default_value) in enumerate(zip(input_labels_1, defaults_1)):
        key = f"input_1_{label.replace(' ', '_')}"
        if '溶解性的惰性COD占总COD比例' in label:
            input_val = st.sidebar.slider(label, 0.050, 0.100, float(default_value), 0.005, key=key)
        elif '颗粒性惰性组分比例' in label:
            input_val = st.sidebar.slider(label, 0.20, 0.35, float(default_value), 0.01, key=key)
        elif '易降解COD比例' in label:
            input_val = st.sidebar.slider(label, 0.15, 0.25, float(default_value), 0.01, key=key)
        elif '进水可过滤无机物质' in label:
            input_val = st.sidebar.slider(label, 0.15, 0.35, float(default_value), 0.01, key=key)
        elif '出水有机氮' in label:
            input_val = st.sidebar.slider(label, 0.0, 5.0, float(default_value), 0.1, key=key)
        elif '可降解COD产泥系数' in label:
            input_val = st.sidebar.slider(label, 0.30, 0.80, float(default_value), 0.01, key=key)
        elif '15℃衰减系数' in label:
            input_val = st.sidebar.slider(label, 0.10, 0.20, float(default_value), 0.01, key=key)
        elif '污泥体积指数' in label:
            input_val = st.sidebar.slider(label, 50, 200, int(default_value), 5, key=key)
        elif '设计浓缩时间' in label:
            input_val = st.sidebar.slider(label, 1.0, 3.0, float(default_value), 0.1, key=key)
        elif '混合液剩余DO值' in label:
            input_val = st.sidebar.slider(label, 0.5, 5.0, float(default_value), 0.1, key=key)
        elif '硝化菌最大比生长速率' in label:
            input_val = st.sidebar.slider(label, 0.20, 0.47, float(default_value), 0.01, key=key)
        else:
            input_val = st.sidebar.text_input(label, str(default_value), key=key)
        inputs_1.append(input_val)
    # 第二列输入
    st.sidebar.header("2️⃣进出水流量与水质")
    input_labels_2 = [
        "日平均进水水量(m3/d)", "进水化学需氧量(mg/L)", "进水生物需氧量(mg/L)","进水总磷(mg/L)",
        "进水总氮(mg/L)", "进水悬浮固体(mg/L)", "设计温度(℃)", "出水化学需氧量(mg/L)", 
        "出水生物需氧量(mg/L)", "出水总磷(mg/L)", "出水总氮(mg/L)", "出水氨氮(mg/L)",
        "出水悬浮固体(mg/L)","生物池污泥浓度(g/L)"
    ]
    
    # 为每个输入框添加默认值
    defaults_2 = ["10000", "400", "200", "8", "50", "300", "15", "50", "10", "1", "15", "5", "10", "5"]
    inputs_2 = [st.sidebar.text_input(label, default_value, key=f"input_{i+len(defaults_1)}") for i, (label, default_value) in enumerate(zip(input_labels_2, defaults_2))]
    
    # 第3列输入
    st.sidebar.header("3️⃣传氧速率参数")
    input_labels_3 = [
        "α(0.8-0.85)混合液KLa/清水KLa", "β(0.9-0.97)混合液饱和溶解氧/清水饱和溶解氧", 
        "曝气装置与池底距离(m)", "设计水深(m)", "当地海拔高度(m)", "氧利用率"
    ]

    # 为每个输入框添加默认值
    defaults_3 = ["0.85", "0.95", "0.2", "5.0", "100", "0.3"]
    # inputs_3 = [st.sidebar.text_input(label, default_value, key=f"input_{i+len(defaults_1)+len(defaults_2)}") for i, (label, default_value) in enumerate(zip(input_labels_3, defaults_3))]

    inputs_3 = []

    for i, (label, default_value) in enumerate(zip(input_labels_3, defaults_3)):
        key = f"input_3_{label.replace(' ', '_')}"
        if '(0.8-0.85)混合液KLa/清水KLa' in label:
            input_val3 = st.sidebar.slider(label, 0.80, 0.85, float(default_value), 0.01, key=key)
        elif '(0.9-0.97)混合液饱和溶解氧/清水饱和溶解氧' in label:
            input_val3 = st.sidebar.slider(label, 0.90, 0.97, float(default_value), 0.01, key=key)
        
        else:
            input_val3 = st.sidebar.text_input(label, str(default_value), key=key)
        inputs_3.append(input_val3)

    carbon_source_options = ["甲醇", "乙醇", "醋酸"]
    carbon_source_name = st.sidebar.selectbox("请输入甲醇,乙醇或醋酸之一", carbon_source_options, index=2)

    p_source_options = ["铁盐", "铝盐"]
    p_source_name = st.sidebar.selectbox("请输入除磷剂名称,铁盐或铝盐", p_source_options, index=1)
    st.markdown('---')

    col1, col2 = st.columns(2)
    with col1:
        if st.button("二沉池计算"):
            try:
                inputs_0 = [float(val) for val in inputs_0]
                result_0 = sst(*inputs_0)

                output_str0 = "#### 0️⃣二沉池设计计算\n\n"
                output_str0 += f"设计流量: {result_0[0]:.2f} m3/d \n\n"
                output_str0 += f"二沉池表面积: {result_0[1]:.2f} m2 \n\n"
                output_str0 += f"二沉池进水污泥浓度: {result_0[2]:.2f} kg/m3 \n\n"
                output_str0 += f"回流污泥浓度: {result_0[3]:.2f} kg/m3 \n\n"
                output_str0 += f"h1清水区高度: {result_0[4]:.2f} m \n\n"
                output_str0 += f"h2过渡区高度: {result_0[5]:.2f} m \n\n"
                output_str0 += f"h3缓冲区高度: {result_0[6]:.2f} m \n\n"
                output_str0 += f"h4浓缩区高度: {result_0[7]:.2f} m \n\n"
                output_str0 += f"二沉池深度: {result_0[8]:.2f} m \n\n"
                output_str0 += f"进水污泥体积负荷: {result_0[9]:.2f} L/(m3/h) \n\n"
                output_str0 += "进水污泥体积负荷应小于500L/(m3/h)，对于竖向流的二沉池，污泥絮凝较好，该值应小于650L/(m3/h),生物池污泥浓度(g/L)可按二沉池进水污泥浓度取值。\n\n"
                with st.expander("二沉池设计计算结果", expanded = True):
                    st.markdown(output_str0, unsafe_allow_html=True)


            except ValueError:
                st.error("请输入正确的数值")
        # 计算按钮
        if ui.button(text="AO计算", key="styled_btn_tailwind", className="bg-orange-500 text-white"):
            # with st.expander("二沉池设计计算结果", expanded = False):
            #     st.markdown(output_str0, unsafe_allow_html=True)
            try:
                inputs_0 = [float(val) for val in inputs_0]
                result_0 = sst(*inputs_0)
                output_str0 = "#### 0️⃣二沉池设计计算\n\n"
                output_str0 += f"设计流量: {result_0[0]:.2f} m3/d \n\n"
                output_str0 += f"二沉池表面积: {result_0[1]:.2f} m2 \n\n"
                output_str0 += f"二沉池进水污泥浓度: {result_0[2]:.2f} kg/m3 \n\n"
                output_str0 += f"回流污泥浓度: {result_0[3]:.2f} kg/m3 \n\n"
                output_str0 += f"h1清水区高度: {result_0[4]:.2f} m \n\n"
                output_str0 += f"h2过渡区高度: {result_0[5]:.2f} m \n\n"
                output_str0 += f"h3缓冲区高度: {result_0[6]:.2f} m \n\n"
                output_str0 += f"h4浓缩区高度: {result_0[7]:.2f} m \n\n"
                output_str0 += f"二沉池深度: {result_0[8]:.2f} m \n\n"
                output_str0 += f"进水污泥体积负荷: {result_0[9]:.2f} L/(m3/h) \n\n"
                output_str0 += "进水污泥体积负荷应小于500L/(m3/h)，对于竖向流的二沉池，污泥絮凝较好，该值应小于650L/(m3/h),生物池污泥浓度(g/L)可按二沉池进水污泥浓度取值。\n\n"
                with st.expander("二沉池设计计算结果"):
                    st.markdown(output_str0, unsafe_allow_html=True)
                inputs_1 = [float(val) for val in inputs_1]
                inputs_2 = [float(val) for val in inputs_2]
                inputs_3 = [float(val) for val in inputs_3]

                result = V_D_over_V_BB_F(*inputs_1, *inputs_2, *inputs_3, carbon_source_name, p_source_name, 0.6)
                max = [0.2,0.25,0.3,0.35,0.4,0.45,0.5,0.55,0.6]
                # 创建一个空列表来存储所有数据
                data = []
                for i in range(9):
                    Result_max = V_D_over_V_BB_F(*inputs_1, *inputs_2, *inputs_3, carbon_source_name, p_source_name, max[i])
                    # V_bb = round(Result_max[42],2)
                    V_a = round(Result_max[44],2)
                    V_o = round(Result_max[45],2)
                    COD_dos = round(Result_max[15],2)
                
                # 将每组数据作为一个列表添加到 data 列表中
                    data.append([max[i], V_a, V_o, COD_dos])
                columns = ['最大缺氧区比例', '缺氧池体积m3', '好氧池体积m3', '外加碳源浓度mg/L']

                invoice_df = pd.DataFrame(data, columns=columns)

                output_str1 = "#### 1️⃣缺氧池/曝气池体积比例计算\n\n"
                output_str1 += f"设计流量: {result[0]:.2f} m3/d \n\n"
                output_str1 += f"进水可过滤物质: {result[1]:.2f} mg/L \n\n"
                output_str1 += f"颗粒性COD(可过滤物质COD,有机干物质颗粒按1.6gCOD/oTS计): {result[2]:.2f} mg/L \n\n"
                output_str1 += f"可溶解性COD: {result[3]:.2f} mg/L \n\n"
                output_str1 += f"溶解性惰性组分: {result[4]:.2f} mg/L \n\n"
                output_str1 += f"颗粒性惰性组分: {result[5]:.2f} mg/L \n\n"
                output_str1 += f"可降解COD: {result[6]:.2f} kg/d \n\n"
                output_str1 += f"易降解COD: {result[7]:.2f} mg/L \n\n"
                output_str1 += f"进水可过滤无机物质: {result[8]:.2f} mg/L \n\n"
                output_str1 += f"出水凯氏氮: {result[9]:.2f} mg/L\n\n"
                output_str1 += f"出水硝酸盐氮: {result[10]:.2f} mg/L \n\n"
                
                output_str1 += f"COD日负荷: {result[11]:.2f} kg/d \n\n"
                output_str1 += f"硝化最小泥龄: {result[12]:.2f} d \n\n"
                output_str1 += f"内源呼吸的衰减系数: {result[13]:.2f} \n\n"
                output_str1 += f"反硝化菌泥龄: {result[14]:.2f} d \n\n"
                output_str1 += f"外加碳源化学需氧量: {result[15]:.2f} mg/L \n\n"

                output_str1 += f"缺氧池/曝气池体积比例: {result[16]:.2f} \n\n"
                output_str1 += f"耗氧量和供氧量平衡: {result[17]:.2f}  \n\n"
                output_str1 += f"设计污泥泥龄: {result[18]:.2f} d \n\n"
                output_str1 += f"生物体中的COD: {result[19]:.2f} kg/d \n\n"
                output_str1 += f"剩余惰性固体: {result[20]:.2f} kg/d \n\n"
                output_str1 += f"污泥产量: {result[21]:.2f} kg/d \n\n"
                output_str1 += f"出水硝态氮: {result[22]:.2f} mg/L \n\n"
                output_str1 += f"形成活性污泥的氮: {result[23]:.2f} mg/L \n\n"
                output_str1 += f"与惰性颗粒结合的氮: {result[24]:.2f} mg/L \n\n"
                output_str1 += f"每日平均反硝化的硝态氮浓度: {result[25]:.2f} mg/L \n\n"

                output_str1 += f"碳降解的总需氧量: {result[26]:.2f} \n\n"
                output_str1 += f"反硝化区易降解及外加碳源需氧量: {result[27]:.2f} \n\n"
                output_str1 += f"反硝化总需氧量: {result[28]:.2f} \n\n"

                output_str3 = "#### 2️⃣除磷污泥产量 \n\n"
                output_str3 += f"出水磷浓度: {result[29]:.2f} mg/L \n\n"
                output_str3 += f"形成活性污泥的磷: {result[30]:.2f} mg/L \n\n"
                output_str3 += f"生物法除磷量: {result[31]:.2f} mg/L \n\n"
                output_str3 += f"需要沉析的磷酸盐: {result[32]:.2f} mg/L \n\n"
                output_str3 += f"化学除磷药剂投加量: {result[33]:.2f} mol/L \n\n"
                output_str3 += f"化学除磷产泥量: {result[34]:.2f} kg/d \n\n"
                output_str3 += "#### 3️⃣污泥产量 \n\n"
                output_str3 += f"剩余污泥量: {result[35]:.2f} kg/d \n\n"
                output_str3 += f"生物段保持的污泥质量: {result[36]:.2f} kg \n\n"
                output_str3 += f"缺氧池污泥量: {result[37]:.2f} kg \n\n"
                output_str3 += f"好氧池污泥量: {result[38]:.2f} kg \n\n"
                output_str3 += f"反硝化速率: {result[39]:.2f} kgN/kgSS·d \n\n"
                output_str3 += f"好氧池COD负荷: {result[40]:.2f} kgCOD/kgSS·d \n\n"       
                output_str3 += f"好氧池BOD负荷: {result[41]:.2f} kgBOD/kgSS·d \n\n"
                
                output_str2 = "#### 4️⃣生物池容积 \n\n"
                output_str2 += f"曝气池的容积: {result[42]:.2f} m3\n\n"
                output_str2 += f"厌氧池容积: {result[43]:.2f} m3 \n\n"
                output_str2 += f"缺氧池容积: {result[44]:.2f} m3 \n\n"
                output_str2 += f"好氧池容积: {result[45]:.2f} m3 \n\n"
                output_str2 += f"总容积: {result[46]:.2f} m3 \n\n"
                output_str2 += f"厌氧池水力停留时间: {result[47]:.2f} h \n\n"
                output_str2 += f"缺氧池水力停留时间: {result[48]:.2f} h \n\n"
                output_str2 += f"好氧池水力停留时间: {result[49]:.2f} h \n\n"
                output_str2 += f"总水力停留时间: {result[50]:.2f} h \n\n" 
                output_str2 += "### 5️⃣回流比 \n\n"
                output_str2 += f"反硝化所需的回流比(包括污泥回流): {result[51]:.2f} \n\n"
                output_str2 += f"反硝化所需的内回流比(已减去回流到厌氧的100%): {result[52]:.2f} \n\n"
                output_str2 += f"反硝化最大效率: {result[53]:.2f} \n\n"
                
                output_str4 = "### 6️⃣耗氧量物料平衡 \n\n"
                output_str4 += f"碳去除的耗氧量: {result[54]:.2f} kgO2/d \n\n"
                output_str4 += f"硝化耗氧量: {result[55]:.2f} kgO2/d\n\n"
                output_str4 += f"反硝化回收供氧量: {result[56]:.2f} kgO2/d \n\n"
                output_str4 += f"平均耗氧量: {result[57]:.2f} kgO2/h \n\n"
                output_str4 += f"最高耗氧量: {result[58]:.2f} kgO2/h \n\n"
                output_str4 += "### 7️⃣标准传氧速率 \n\n"
                output_str4 += f"当地大气压力: {result[59]:.2f} Mpa \n\n"
                output_str4 += f"每曝气装置处绝对压力: {result[60]:.2f} Mpa \n\n"     
                output_str4 += f"清水表面饱和溶解氧: {result[61]:.2f} mg/L \n\n"
                output_str4 += f"水下深度到池面清水平均溶氧值: {result[62]:.2f} mg/L \n\n"
                output_str4 += f"AOR与SOR转换系数: {result[63]:.2f} \n\n"
                output_str4 += f"标准传氧速率SOR: {result[64]:.2f} kgO2/h \n\n"
                output_str4 += f"标准状况供空气体积m3/h: {result[65]:.2f} m3/h \n\n"
                output_str4 += f"气水比: {result[66]:.2f} \n\n"   



                with st.expander("**生物池容积与回流比**", expanded=True):                  
                    st.markdown(f" \n{output_str2}\n ")
                with st.expander("**不同最大缺氧区比例下的生物池容积与碳源投加**", expanded=True):
                    ui.table(data=invoice_df, maxHeight=300)
                with st.expander("**污泥产量**", expanded=True):
                    st.markdown(f" \n{output_str3}\n ")
                with st.expander("**计算过程参数**"):
                    st.markdown(f" \n{output_str1}\n ")
                with st.expander("**耗氧量物料平衡**"):
                    st.markdown(f" \n{output_str4}\n ")

            # Constants for the equation
            # f_Sb = 0.6    # Given value
            # f_cv = 1.48   # Given value
            # Y_Hv = 0.4527 # Given value
            # K_T = 0.241   # Given value
            # b_HT = 0.14    # Given value

            # # Define SRT range
            # SRT_values = np.linspace(0.1, 20, 400)  # SRT from 0.1 to 20, 400 points

            # # Calculate fx1min
            # def fx1min(f_Sb, f_cv, Y_Hv, K_T, b_HT, SRT):
            #     values = f_Sb * (1 - f_cv * Y_Hv) * (1 + b_HT * SRT) / (2.86 * K_T * Y_Hv * SRT)
            #     values = np.where(values > 1, 1, values)  # If values are greater than 1, set them to 1
            #     return values

            # # Plotting

            # fig, ax = plt.subplots(figsize=(10, 6))
            # ax.plot(SRT_values, fx1min(f_Sb, f_cv, Y_Hv, K_T, b_HT, SRT_values), color='orange', label='$f_{x1_{min}}$ vs SRT')
            # ax.set_xlabel('SRT (Sludge Retention Time)')
            # ax.set_ylabel('$f_{x1_{min}}$')
            # ax.set_title('Relationship between SRT and $f_{x1_{min}}$')
            # ax.grid(True, linestyle='--')  # Set grid lines as dashed
            # ax.legend()

            # # Display the plot in Streamlit
            # st.pyplot(fig)
            except ValueError:
                st.error("请输入正确的数值")
    with col2:
        with st.expander("📝计算说明",expanded=True):
            st.markdown("""
        #### 📝计算说明
            
        **1.好氧污泥泥龄计算**
                        
        $$t_{硝化最小泥龄}=PF \\bullet 1.6 \\frac{1}{\\mu_{硝化菌}}=PF\\bullet 1.6 \\frac{1.103^{(15-T_{设计温度})}}{0.47}\\quad [天]$$

        **2-1.估算$V_{A池}/V_{AO}$ 0.2-0.6**

        $$t_{设计泥龄}=t_{硝化最小系统泥龄} = t_{硝化最小泥龄} \\bullet \\frac{1}{1-V_{A池}/V_{AO}} \\quad [天]$$

        **2-2.污泥产率系数**
                        
        $$Y_{进水} = 0.67[kgCOD_{可降解}/kgCOD_{微生物}]$$
                        
        $$Y_{碳源}= 0.42 - 0.45[kgCOD_{碳源}/kgCOD_{微生物}]$$
                        
        **2-3.污泥产量的计算**
                        
        $$X_{COD,微生物} = (C_{进水,可降解}Y_{进水}+C_{碳源}Y_{碳源})\\bullet \\frac{1}{1+b_{衰减}t_{设计泥龄}F_{T,温度系数}}[mg/L]$$
                        
        $$X_{COD,衰减，惰性} = 0.2\\bullet X_{COD,微生物}\\bullet b_{衰减}\\bullet t_{设计泥龄}\\bullet F_{T,温度系数}[mg/L]$$
                        
        **2-4.计算需要反硝化的硝酸盐浓度**
                        
        $$S_{NO3,反硝化} = C_{总氮，进水} - S_{有机氮，出水}-S_{氨氮，出水}-S_{硝态氮，出水}-X_{有机氮，微生物}-X_{有机氮，惰性}[mg/L]$$
                        
        - $S_{有机氮，出水} = 2 mg/L$
        - $S_{氨氮} ≈ 0mg/L$
        - $S_{硝态氮，出水} = 0.8 或 0.6\\bullet S_{总氮，出水限值}[mg/L]$
        - $X_{有机氮，微生物}= 0.07\\bullet X_{COD,微生物}[mg/L]$
        - $X_{有机氮，惰性}=0.03\\bullet(X_{COD,衰减,惰性}+X_{COD,进水,惰性})[mg/L]$

        **2-5.降解有机物所消耗的 $O_{2}$**
                        
        合计耗氧量：
                        
        $$OU_{合计} = C_{进水,可降解} + C_{碳源} -X_{COD,微生物}-X_{COD,衰减，惰性} \\quad[mg/L]$$
                        
        易降解的COD部分所需耗氧量
                        
        $$OU_{易降解}=f_{易降解}*C_{进水,可降解}(1-Y_{进水})+C_{碳源}(1-Y_{碳源})\\quad[mg/L]$$
                        
        $$OU_{碳源}= C_{碳源}(1-Y_{碳源})\\quad[mg/L]$$
                        
        $$OU_{缓慢降解} = OU_{合计}-OU_{易降解}\\quad[mg/L]$$
                        
        $$OU_{进水}= OU_{合计}-OU_{碳源}\\quad[mg/L]$$
                        
        反硝化区的“耗氧量”-硝酸盐供氧
                        
        $$OU_{反硝化} = 0.75\\bullet [OU_{易降解}+OU_{缓慢降解} \\bullet (V_{A池}/V_{AO})^{0.68}] \\quad [mg/L](1)$$
                        
        $$OU_{反硝化} = 0.75\\bullet [OU_{碳源}+OU_{进水} \\bullet (V_{A池}/V_{AO})] \\quad [mg/L](2)$$
                        
        $$OU_{反硝化} = 0.75\\bullet [OU_{合计}\\bullet (V_{A池}/V_{AO})] \\quad [mg/L](3)$$
                        
        - 硝酸盐呼吸系数=0.75
        - 前置反硝化(常规AO)区呼吸量增加 = $(V_{A池}/V_{AO})^{0.68}$
        - $f_{易降解} = C_{易降解,进水}/C_{可降解,进水}$ 城市污水中易降解COD的比例为0.15-0.25，超过该范围应予以验证

        **2-6.“有机物的氧消耗”和“硝酸盐的氧供给”的比较**
                        
        $$x = \\frac{OU_{反硝化}}{2.86\\bullet S_{NO3,反硝化}}$$

        - 如果x > 1，降低缺氧区比例$(V_{A池}/V_{AO})$或减少碳源投加C_{碳源},**重新回到2-1**
        - 如果x < 1，提高缺氧区比例$(V_{A池}/V_{AO})$或增加碳源投加C_{碳源},**重新回到2-1**
        - 如果x = 1，进行下一步计算

        **3.计算设计泥龄下平均每天产生污泥浓度**
                        
        $$TSS = VSS_{有机}+ISS_{无机}+SS_{除磷}\\quad [kg/m^3]$$

        **4.来自二沉池设计的MLSS[g/L]**

        **5.计算容积**
                        
        $$V_{AO} = TSS \\bullet t_{设计泥龄} /MLSS $$
                        
        $$V_{A池} = V_{AO} * (V_{A池}/V_{AO})$$
                        
        $$V_{O池} = V_{AO}- V_{A池}$$

                """, unsafe_allow_html=True)
        with st.expander("污泥龄与活性微生物量及衰减惰性颗粒的关系图", expanded = True):
            st.image("/opt/stapp/SRT.png")
            st.markdown("""**污泥龄与活性微生物量及衰减惰性颗粒的关系图** \n\n
- $SRT$：污泥龄
- Per$COD*Y_{Hv}$：每日COD负荷下产生活性微生物当量，
- $X_{COD,BM}$：当前泥龄下活性微生物量当量
- $X_{COD,innet}$:：当前泥龄下衰减惰性颗粒量当量
- $X_{COD,BM}/X_{COD,innet}$：当前泥龄下活性微生物量与衰减惰性颗粒量的比值
                """, unsafe_allow_html=True)

def main():
    show_page2()

if __name__ == "__main__":
    main()

    