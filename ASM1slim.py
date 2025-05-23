import streamlit as st
from st_pages import add_page_title, hide_pages
import streamlit_shadcn_ui as ui
import pandas as pd
import functools
import torch
import numpy as np
from torchdiffeq import odeint
import matplotlib.pyplot as plt # 用于绘图
from mpl_toolkits.mplot3d import Axes3D # 用于3D绘图
import random
from streamlit_mermaid import st_mermaid

# 设置页面配置
st.set_page_config(
    page_title="微风轻语BreeCho",  # 自定义页面标题
    page_icon="💭",  # 可以是一个URL链接，或者是emoji表情符号
    layout="wide",  # 页面布局: "centered" 或 "wide"
    initial_sidebar_state="auto",  # 初始侧边栏状态: "auto", "expanded", "collapsed"
)

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

def plot_3d_bars(tensor_data):
    z_data = tensor_data[:, :, 2]
    
    # 创建网格点坐标
    x = np.arange(z_data.shape[0])
    y = np.arange(z_data.shape[1])
    X, Y = np.meshgrid(x, y)
    
    # 创建图形
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # 将网格数据转换为一维数组
    x_pos = X.flatten()
    y_pos = Y.flatten()
    z_pos = np.zeros_like(x_pos)  # 柱子的起始位置
    dx = dy = 0.8  # 柱子的宽度
    dz = z_data.T.flatten()  # 柱子的高度
    
    # 绘制3D柱状图
    ax.bar3d(x_pos, y_pos, z_pos, dx, dy, dz, color='b', alpha=0.6)
    
    # 设置标签
    ax.set_xlabel('First Dimension')
    ax.set_ylabel('Variables')
    
    # 定义y轴标签
    y_labels = ['input'] + [f'A{i}' for i in range(1,7)] + [f'O{i}' for i in range(1,7)] + ['sst', 'output']
    ax.set_yticks(range(len(y_labels)))
    ax.set_yticklabels(y_labels)
    
    ax.set_zlabel('Values')
    plt.title('3D Bar Plot')
    
    # 调整视角
    ax.view_init(elev=30, azim=45)
    plt.tight_layout()
    
    return fig

def plot_3d_lines(tensor_data):
    z_data = tensor_data[:, :, 2]
    
    # 创建网格点坐标
    x = np.arange(z_data.shape[0])
    y = np.arange(z_data.shape[1])
    X, Y = np.meshgrid(x, y)
    
    # 创建图形
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # 绘制3D折线图
    for i in range(len(y)):
        ax.plot(x, [i]*len(x), z_data[:, i], 
               marker='o',  # 添加数据点标记
               label=f'Line {i}')
    
    # 设置标签
    ax.set_xlabel('First Dimension')
    ax.set_ylabel('Variables')
    
    # 定义y轴标签
    y_labels = ['input'] + [f'A{i}' for i in range(1,7)] + [f'O{i}' for i in range(1,7)] + ['sst', 'output']
    ax.set_yticks(range(len(y_labels)))
    ax.set_yticklabels(y_labels)
    
    ax.set_zlabel('Values')
    plt.title('3D Line Plot')
    
    # 调整视角
    ax.view_init(elev=30, azim=45)
    plt.tight_layout()
    
    return fig

def plot_3d_tensor(tensor_data, num):
    # 获取数据维度
    z_data = tensor_data[:, :, num]  # 提取第三维的数据 
    # 创建网格点坐标
    x = np.arange(z_data.shape[0])
    y = np.arange(z_data.shape[1])
    X, Y = np.meshgrid(x, y)  
    # 创建3D图
    fig = plt.figure(figsize=(15, 12))
    ax = fig.add_subplot(111, projection='3d')

        # 定义可选的颜色方案列表
    cmaps = [
        'viridis', 'plasma', 'inferno', 'magma', 'hot', 'cool', 
        'coolwarm', 'rainbow', 'jet', 'Paired'
    ]
   
    # 随机选择一个颜色方案，并随机决定是否反转
    chosen_cmap = random.choice(cmaps)
    if random.random() > 0.5:  # 50%的概率反转颜色
        chosen_cmap = chosen_cmap + '_r'  
    # 绘制3D表面
    surf = ax.plot_surface(X, Y, z_data.T, 
                          cmap=chosen_cmap,  # 使用viridis配色
                          linewidth=0,
                          antialiased=True)
    
    # 添加颜色条
    fig.colorbar(surf, ax=ax, shrink=0.5, aspect=5) 
    # 设置标签
    ax.set_xlabel('Time Dimension (minutes)')
    # 设置y轴标签和刻度
    ax.set_ylabel('Tanks')
    # 定义新的标签
    y_labels = ['input'] + [f'A{i}' for i in range(1,7)] + [f'O{i}' for i in range(1,7)] + ['sst']
    # 设置y轴刻度标签
    ax.set_yticklabels(y_labels)
    ax.set_yticks(range(len(y_labels)))  # 明确设置刻度位置
    ax.set_zlabel('Values')   
    # 设置标题
    plt.title('3D Visualization of Data')    
    # 调整视角
    ax.view_init(elev=30, azim=45)
        # 调整布局，确保所有标签都可见
    plt.tight_layout()    
    return fig


def balanceQ(Q_out):
    # 将输入矩阵设置为Q_out的转置
    Q_in = Q_out.t()
    m = Q_out.shape[0]
    # 针对每个k循环m次
    for k in range (m):
        # 针对每个i循环m-2次
        for i in range(m-2):
            # 更新Q_out矩阵中第i+1行和第i+2列的值
            Q_out[i+1,i+2] = sum(Q_in[i+1]) - sum(Q_out[i+1]) + Q_out[i+1,i+2]
        # 在每次迭代结束后，将输入矩阵重新设置为Q_out的转置
    return (Q_out)

def map_tensor(input_tensor, size=15):
    """优化后的映射函数"""
    newtensor = torch.zeros((size, size))  
    # 定义映射关系
    row_map = {0: 0, 1: 6, 2: -3, 3: -2, 4: -1}
    col_map = {0: 0, 1: 1, 2: 7, 3: -2, 4: -1}    
    # 使用高级索引一次性完成映射
    src_rows, src_cols = torch.where(input_tensor != 0)
    for r, c in zip(src_rows, src_cols):
        newtensor[row_map[r.item()], col_map[c.item()]] = input_tensor[r, c]   
    return newtensor

def create_expanded_tensor(tensor):
    # 获取第二行和倒数第三行
    second_row = tensor[1:2]
    third_last_row = tensor[-3:-2] 
    # 创建重复行
    repeated_second = second_row.repeat(5, 1)
    repeated_third_last = third_last_row.repeat(5, 1)
    # 分割原始tensor
    part1 = tensor[:2]  # 前两行
    part2 = tensor[2:-2]  # 中间部分
    part3 = tensor[-2:]  # 最后两行  
    # 拼接所有部分
    new_tensor = torch.cat([
        part1,
        repeated_second,
        part2,
        repeated_third_last,
        part3
    ], dim=0)
    return new_tensor

def ASM_1_ode_slim(ASMparam:torch.tensor, C0_matrix:torch.tensor):
    # Extract the relevant variables from the matrix
    S_O = C0_matrix[:, 0]
    S_S = C0_matrix[:, 1]
    S_NO = C0_matrix[:, 2]
    S_NH = C0_matrix[:, 3]
    S_ALK = C0_matrix[:, 4]
    n = len(S_NO)
    ASMparam = ASMparam.unsqueeze(1).repeat(1,n)
    dSNOmax,dSNHmax,CNRatio,K_S, K_NO, n_g, K_NH = ASMparam
    KNO = S_NO / (K_NO + S_NO)
    KNH = torch.where(S_ALK < 0.4, torch.tensor(0), S_NH / (K_NH + S_NH))
    KS = S_S / (K_S + S_S)
    dS_NO0 = torch.where(S_O<0.15, -dSNOmax*torch.min(KS, KNO) ,torch.tensor(0))
    dS_S0 = dS_NO0 * CNRatio   
    dS_S1 = torch.where(S_O>1,-KS* dSNOmax * CNRatio/n_g ,torch.tensor(0))  #6-1 溶解有机物增长速率-异养菌好氧生长
    dS_S2 = dS_S0 
    dS_NO1 = dS_NO0
    dS_NO2 = torch.where(S_O>1, KNH * dSNHmax ,torch.tensor(0)) #2-2 硝化硝态氮增长速率
    dS_NH1 = -dS_NO2  #5-1 氨氮增长速率-自养菌好氧生长
    dS_ALK2 = - dS_NO1 / 14
    dS_ALK3 = dS_NH1/14 - dS_NO2 / 14
    dS_O1 = dS_S1 + dS_S1*(CNRatio-2.86)/CNRatio
    dS_O2 = -4.57 * dS_NO2 + dS_NO2 * 0.24
    dS_S = dS_S1 + dS_S2
    dS_NO = dS_NO1 + dS_NO2 
    dS_NH = dS_NH1
    dS_Alk = dS_ALK2 + dS_ALK3
    dS_O = dS_O1 + dS_O2
    return torch.stack([dS_O, dS_S, dS_NO, dS_NH, dS_Alk], dim=1)

def balanceParam(C,Q_out):
    m, n = Q_out.shape
    r = C.shape[1]
    C_init = C.unsqueeze(1).repeat(1,n,1)
    C_out = C_init
    q_out = Q_out.clone().unsqueeze(-1) 
        # 在最后一维增加一维，形状变为 (m, n, 1) 
    m_out = q_out * C_out
    sum_m_out = m_out.sum(dim=1).view(n,r)  
    sum_m_in = m_out.sum(dim=0).view(m,r)
    delta_m=sum_m_in-sum_m_out
    return (delta_m, C_out, m_out, sum_m_out)

def ode_Deni(t, y, Q_out, V_liq, ASMparam):
    y[y < 0] = 0
    delta_m = balanceParam(y, Q_out)[0]
    Deni_func = ASM_1_ode_slim(ASMparam, y[1:14])
    dy = torch.zeros_like(y)
    dy[1:14,:] = delta_m[1:14,:] / V_liq[1:14,None] + Deni_func[None,:]
    dy[1:14, 0] = torch.zeros_like(dy[1:14, 0])
    return dy

def ASMrun_hours(hours, x0,  Q_out, V_liq, ASMparam):
    x0 = x0[-1,:].float()
    t0 = torch.linspace(0, hours, hours*60+1)
    ode_ASM1_modified = functools.partial(ode_Deni, Q_out=Q_out, V_liq=V_liq, ASMparam=ASMparam)
    x= odeint (ode_ASM1_modified,x0,t0, method='rk4') 
    return x

def dataframe_to_tensor(df, exclude_columns=None):
    """
    将DataFrame转换为PyTorch tensor，排除指定的非数值列
    
    Parameters:
        df (pd.DataFrame): 输入的DataFrame
        exclude_columns (list): 需要排除的列名列表
    
    Returns:
        torch.Tensor: 转换后的tensor
        list: 保留的列名列表
    """
    if exclude_columns is None:
        exclude_columns = []
    
    # 复制DataFrame并删除非数值列
    numeric_df = df.copy()
    for col in exclude_columns:
        if col in numeric_df.columns:
            numeric_df = numeric_df.drop(columns=[col]) 
    # 保存列名，用于之后转换回DataFrame
    retained_columns = numeric_df.columns.tolist()
    # 转换为tensor
    return torch.tensor(numeric_df.values, dtype=torch.float32), retained_columns

# 动态生成 Mermaid 图的代码
def generate_mermaid_flowchart(df):
    connections = []
    rows = df.iloc[:, 0]
    cols = df.columns[1:]
    
    for i, row in enumerate(rows):
        for j, col in enumerate(cols):
            value = df.at[i, col]
            if value != 0:
                connections.append(f"{row}['{row}'] -->|{value}| {col}['{col}']")

    mermaid_code = "\n".join(connections)
    return f"graph LR\n{mermaid_code}\n"

def tensor_to_dataframe(tensor, columns, original_df=None, exclude_columns=None):
    """
    将tensor转换回DataFrame
    
    Parameters:
        tensor (torch.Tensor): 输入的tensor
        columns (list): 数值列的列名列表
        original_df (pd.DataFrame, optional): 原始DataFrame，用于恢复非数值列
        exclude_columns (list, optional): 在原始DataFrame中被排除的列名列表
    
    Returns:
        pd.DataFrame: 转换后的DataFrame
    """
    # 将tensor转换为numpy数组，然后创建DataFrame
    numeric_df = pd.DataFrame(tensor.numpy(), columns=columns)   
    # 如果提供了原始DataFrame，恢复非数值列
    if original_df is not None and exclude_columns is not None:
        for col in exclude_columns:
            if col in original_df.columns:
                numeric_df.insert(
                    loc=original_df.columns.get_loc(col),
                    column=col,
                    value=original_df[col]
                )
    return numeric_df
def show_page():

    st.markdown("""
    #### 相关文章链接:
    * [ASM1模型的简化-ASM简易不简单](https://mp.weixin.qq.com/s/fDD_H7h-dRiWB_aRhxN51w)
    """)
    
    initial_data = {
        "流出量表 m3/h": [
            "进水池",
            "A池",
            "O池",
            "沉淀池",
            "出水池"
        ],
        "进水池": [0.0] * 5,
        "A池": [0.0] * 5,
        "O池": [0.0] * 5,
        "沉淀池": [0.0] * 5,
        "出水池": [0.0] * 5,
    }

    input_data = {
        "位置": [
            "进水池",
            "A池",
            "O池",
            "沉淀池",
            "出水池"
        ],
        "容积 m3": [0.0] * 5,
        "溶解氧": [0.0] * 5,
        "COD": [0.0] * 5,
        "硝态氮": [0.0] * 5,
        "氨氮": [0.0] * 5,
        "总碱度 mmol/L": [0.0] * 5,
    }

    # 创建初始 DataFrame
    col4, col5 = st.columns(2)

    if 'df' not in st.session_state:
        st.session_state.df = pd.DataFrame(initial_data)

    if 'input_df' not in st.session_state:
        st.session_state.input_df = pd.DataFrame(input_data)

    # 创建标签页
    with col4:
        st.subheader("流量数据表")
        edited_df = st.data_editor(
            st.session_state.df,
            num_rows="fixed",
            hide_index=True,
            column_config={
                "流出量表 m3/h": st.column_config.TextColumn(
                    "流出量表 m3/h",
                    help="池体名称",
                    disabled=True
                ),
            },
            key="flow_editor"
        )
        
        if st.button("流量平衡", key="save_flow"):
            st.session_state.df = edited_df
            st.success("流量已平衡！")
            flow_exclude_columns = ["流出量表 m3/h"]
            flow_tensor, flow_columns = dataframe_to_tensor(st.session_state.df, exclude_columns=flow_exclude_columns)
            balanced_tensor = balanceQ(flow_tensor)
            balanced_tensor_df = tensor_to_dataframe(balanced_tensor, flow_columns, st.session_state.df, flow_exclude_columns)
            st.write("平衡后的流量数据：")
            st.write(balanced_tensor_df)
            # 生成 Mermaid 代码
            mermaid_code = generate_mermaid_flowchart(balanced_tensor_df)

            # 在 Streamlit 中显示
            st.write("平衡后流程图：")
            st_mermaid(mermaid_code)


    with col5:
        st.subheader("水质数据表 mg/L")
        edited_input_df = st.data_editor(
            st.session_state.input_df,
            num_rows="fixed",
            hide_index=True,
            column_config={
                "位置": st.column_config.TextColumn(
                    "位置",
                    help="采样位置",
                    disabled=True
                ),
            },
            key="quality_editor"
        )


    st.subheader("参数设置") 
    
    # 创建三列布局以放置滑块
    col1, col2, col3 = st.columns(3)

    
    with col1:
        with st.expander("经验参数", expanded=True):

            dSNOmax = st.slider(
                "经验最大反硝化速率 mg/(L*h)",
                min_value=0.0,
                max_value=100.0,
                value=20.0,
                step=1.0,
                format="%.1f"
            )
            
            dSNHmax = st.slider(
                "经验最大硝化速率 mg/(L*h)",
                min_value=0.0,
                max_value=30.0,
                value=5.0,
                step=0.5,
                format="%.1f"
            )
            
            CNRatio = st.slider(
                "经验碳氮比",
                min_value=3.0,
                max_value=8.0,
                value=5.0,
                step=0.1,
                format="%.1f"
            )
        
    with col2:
        with st.expander("模拟运行时间", expanded=True):

            # 模拟运行时间

            hour = st.slider(
                "模拟运行时间 h",
                min_value=0,
                max_value=60,
                value=5,
                step=1,
                format="%d"

            )    

    
    with col3:
        with st.expander("微调参数"):

            K_S = st.slider(
                "COD浓度对反硝化速率的影响参数",
                min_value=5.0,
                max_value=100.0,
                value=20.0,
                step=1.0,
                format="%.1f"
            )
        
            K_NO = st.slider(
                "硝态氮浓度对反硝化速率的影响参数",
                min_value=0.0,
                max_value=5.0,
                value=0.1,
                step=0.01,
                format="%.2f"
            )
            n_g = st.slider(
                "好氧COD降解速率参数",
                min_value=0.3,
                max_value=2.0,
                value=0.8,
                step=0.01,
                format="%.2f"
            )
            
            K_NH = st.slider(
                "氨氮浓度对硝化速率的影响参数",
                min_value=0.0,
                max_value=10.0,
                value=1.0,
                step=0.01,
                format="%.2f"
            )

    

    # 将参数按固定顺序收集到列表中
    param_list = [
        dSNOmax, dSNHmax, CNRatio,
        K_S, K_NO,
        n_g, K_NH
    ]
    
    # 转换为tensor
    ASMparam = torch.tensor(param_list, dtype=torch.float32)
    st.subheader("模拟计算")
    if st.button("模拟运行"):
        st.session_state.df = edited_df
        flow_exclude_columns = ["流出量表 m3/h"]
        flow_tensor, flow_columns = dataframe_to_tensor(st.session_state.df, exclude_columns=flow_exclude_columns)
        map_flow = map_tensor(flow_tensor)
        Q_out = balanceQ(map_flow)
        # st.write(Q_out)

        st.session_state.input_df = edited_input_df
        quality_exclude_columns = ["位置"]
        quality_tensor, quality_columns = dataframe_to_tensor(st.session_state.input_df, exclude_columns=quality_exclude_columns)
        V_liq = quality_tensor[:, 0]
        V_liq = V_liq.ravel()
        # st.write(V_liq)
        V_liq1 = torch.zeros(15, dtype=torch.float)  # 明确指定数据类型，创建一个浮点型张量

        V_liq1[0] = V_liq[0]  # 0.0
        for i in range(6):
            V_liq1[1+i] = float(V_liq[1])/6  # 确保是浮点数除法
        for j in range(6):
            V_liq1[7+j] = float(V_liq[2])/6  # 确保是浮点数除法
        V_liq1[13] = V_liq[3]  # 500.0
        V_liq1[14] = V_liq[4]  # 0.0
        V_liq = V_liq1
        # st.write(V_liq)
        x0 = quality_tensor[:, 1:]
        # st.write(x0)
        x0=create_expanded_tensor(x0)
        # st.write(x0)
        x0 = x0.unsqueeze(0)
        x1 = ASMrun_hours(hour, x0,  Q_out, V_liq, ASMparam)
        # st.write(x1[-1])
            # 创建行标签
        st.subheader("模拟结果")
        row_labels = ['进水'] + [f'A{i}' for i in range(1, 7)] + [f'O{i}' for i in range(1, 7)] + ['沉淀池']
        
        data=x1[-1]
        data = data[:-1]
        # 将tensor转换为numpy数组
        numpy_array = data.numpy()
        
        # 创建DataFrame，设置行索引和列名
        df = pd.DataFrame(
            numpy_array,
            index=row_labels,
            columns=['DO mg/L', 'COD mg/L', 'NO3-N mg/L', 'NH3-N mg/L', 'Alk mmol/L']
        )
        
        # 设置显示格式，保留2位小数
        pd.set_option('display.float_format', lambda x: '%.2f' % x)
        
        # 在Streamlit中显示DataFrame
        st.dataframe(df, width=800)
        
        # 添加下载按钮
        st.download_button(
            label="Download DataFrame as CSV",
            data=df.to_csv().encode('utf-8'),
            file_name="water_quality_data.csv",
            mime="text/csv"
        )
        x1 = x1[:,:-1,:]
        col6, col7 = st.columns(2)
        with col6:
            st.subheader("COD数据图mg/L")
            st.pyplot(plot_3d_tensor(x1,1))
            st.subheader("硝态氮数据图 mg/L")
            fig = plot_3d_tensor(x1,2)
            st.pyplot(fig)
        with col7:
            st.subheader("氨氮数据图 mg/L")
            st.pyplot(plot_3d_tensor(x1,3))
            st.subheader("碱度数据图 mmol/L")
            st.pyplot(plot_3d_tensor(x1,4))  

            
def main():
    show_page()

if __name__ == '__main__':
    main()
