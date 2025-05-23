import streamlit as st
import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
import pandas as pd
from io import BytesIO

# 设置页面基本配置
st.set_page_config(
    page_title="微风轻语BreeCho",  # 自定义页面标题
    page_icon="💭",  # 可以是一个URL链接，或者是emoji表情符号
    layout="wide",  # 页面布局: "centered" 或 "wide"
    initial_sidebar_state="auto",  # 初始侧边栏状态: "auto", "expanded", "collapsed"
)

plt.rcParams["font.sans-serif"]=["SimHei"] # 设置字体为黑体
plt.rcParams["axes.unicode_minus"]=False 


class RectangularTank:
    def __init__(self, length, width, height):
        """
        初始化长方体水池
        length: 池子长度(米)
        width: 池子宽度(米)
        height: 池子高度(米)
        """
        if any(dim <= 0 for dim in (length, width, height)):
            raise ValueError("所有尺寸必须大于0")
        
        self.length = length
        self.width = width
        self.height = height
        self.base_area = length * width
    
    def calculate_volume(self, level):
        """
        根据液位计算体积
        level: 液位高度(米)
        返回: 体积(立方米)
        """
        if level < 0:
            raise ValueError("液位不能为负数")
        if level > self.height:
            raise ValueError(f"液位不能超过水池高度 {self.height}米")
        
        return self.base_area * level
    
    def calculate_level(self, volume):
        """
        根据体积计算液位
        volume: 体积(立方米)
        返回: 液位高度(米)
        """
        if volume < 0:
            raise ValueError("体积不能为负数")
        
        max_volume = self.base_area * self.height
        if volume > max_volume:
            raise ValueError(f"体积不能超过水池最大容量 {max_volume}立方米")
        
        return volume / self.base_area
    
    def calculate_wetted_area(self, volume):
        """
        计算给定体积时的液体接触面积（包括底面和侧面）
        volume: 体积(立方米)
        返回: 接触面积(平方米)
        """
        if volume < 0:
            raise ValueError("体积不能为负数")
            
        max_volume = self.base_area * self.height
        if volume > max_volume:
            raise ValueError(f"体积不能超过水池最大容量 {max_volume}立方米")
            
        level = self.calculate_level(volume)
        # 接触面积 = 底面积 + 四周侧面积
        wetted_area = (
            self.base_area +  # 底面积
            2 * (self.length + self.width) * level  # 侧面积
        )
        return wetted_area
    
    def get_tank_info(self):
        """获取水池信息"""
        return {
            "长度": f"{self.length}米",
            "宽度": f"{self.width}米",
            "高度": f"{self.height}米",
            "底面积": f"{self.base_area}平方米",
            "最大容积": f"{self.base_area * self.height}立方米"
        }
    

def bulkreact(c, kb, order, c_limit=0.0):
    """
    计算体相反应速率
    
    参数:
        c: float - 当前物质浓度
        kb: float - 体相反应系数 
        order: float - 反应级数
        c_limit: float - 限制浓度
        
    返回:
        float - 体相反应速率
    """
    # 如果浓度为负,设为0
    if c < 0:
        c = 0
        
    # 零阶反应
    if order == 0:
        return kb
        
    # 米氏反应(order < 0)
    elif order < 0:
        # 计算限制项
        limit_term = c_limit + (1 if kb > 0 else -1) * c
        if abs(limit_term) < 1e-6:  # 避免除以0
            limit_term = 1e-6 if limit_term >= 0 else -1e-6
        return kb * c / limit_term
        
    # n阶反应 
    else:
        if c_limit == 0:
            # 普通n阶反应
            return kb * (c ** order)
        else:
            # 带限制浓度的n阶反应
            if kb > 0:  # 增长反应
                c1 = max(0, c_limit - c)
            else:      # 衰减反应  
                c1 = max(0, c - c_limit)
            return kb * c1 * (c ** (order-1))
        

def walleact(c, area_per_vol, kw, kf, order=1):
    """
    计算壁面反应速率
    
    参数:
        c: float - 当前物质浓度
        area_per_vol: float - 单位体积表面积
        kw: float - 壁面反应系数
        kf: float - 质量传递系数
        order: int - 反应级数(0或1)
        
    返回:
        float - 壁面反应速率
    """
    if kw == 0 or area_per_vol == 0:
        return 0.0
        
    if order == 0:  # 零阶反应
        # 反应不能快于质量传递
        rate = min(abs(kw), kf * c)
        # 保持与反应系数同号
        rate = rate if kw > 0 else -rate
        return rate * area_per_vol
        
    else:  # 一阶反应
        # 考虑壁面反应和质量传递的综合效果
        k_total = kw * kf / (kw + kf)
        return k_total * c * area_per_vol

def tank_react(c, v, kb, kw=0, area=0, kf=0, bulk_order=1, wall_order=1, c_limit=0):
    """计算储水池总反应速率"""
    # 计算体相反应
    bulk_rate = bulkreact(c, kb, bulk_order, c_limit)
    # bulk_mass_rate = bulk_rate
    
    # 计算壁面反应
    # wall_mass_rate = 0
    if kw != 0 and area != 0:
        area_per_vol = area / v
        wall_rate = walleact(c, area_per_vol, kw, kf, wall_order)
        # wall_mass_rate = wall_rate * v
        
    return bulk_rate + wall_rate

def completemix(inflow_rate, outflow_rate, Cin, C, volume, kb, order, dt):
    """
    完全混合模型计算
    
    Parameters:
        dt (float): 时间步长 (h)
        
    Returns:
        float: 更新后的浓度
    """
    # 计算时间步内的进水体积
    vin = inflow_rate * dt

    vout = outflow_rate * dt    
    # 计算净流量（进水-出水）
    vnet = vin - vout
    
    # 更新浓度（考虑混合和反应）
    if volume + vnet > 0:
        # 考虑混合效应
        C = (C * volume + Cin * vin) / (volume + vin)
        # 考虑反应动力学
        react_c =bulkreact(C, kb, order, c_limit=0)*dt
        
        # 更新质量平衡参数
        mass_in = Cin * vin
        mass_out = C * vout
        
        C = react_c + C
    
    # 更新池容
    volume = max(0, volume + vnet)
    mass_reacted = react_c * volume
    return volume, C, mass_in, mass_out, mass_reacted

def twocompmix(volume, v1max, C1, C2, inflow_rate, outflow_rate, Cin, kb, order, dt):
    """
    两室混合模型计算
    
    Parameters:
        dt (float): 时间步长 (h)
        
    Returns:
        dict: 包含混合区、停滞区和加权平均浓度的字典
    """
    if volume > v1max:
        v1 = v1max
        v2 = volume - v1max
    else:
        v1 = volume
        v2 = 0
    # 计算时间步内的进水体积
    vin = inflow_rate * dt

    vout = outflow_rate * dt    
    # 计算净流量（进水-出水）
    vnet = vin - vout

    vt = 0  # 区域间交换流量
    
    if vnet > 0:  # 水位上升情况
        # 计算从混合区溢出到停滞区的体积
        vt = max(0, v1 + vnet - v1max)
        
        # 更新混合区浓度
        if vin > 0:
            C1 = (C1 * v1 + Cin * vin) / (v1 + vin)
        
        # 更新停滞区浓度
        if vt > 0:
            C2 = (C2 * v2 + C1 * vt) / (v2 + vt)
            
    else:  # 水位下降情况
        if v2 > 0:
            vt = min(v2, -vnet)  # 从停滞区返回的水量
        
        # 更新混合区浓度
        if vin + vt > 0:
            C1 = (C1 * v1 + Cin * vin + 
                        C2 * vt) / (v1 + vin + vt)
    mass_in = Cin * vin
    mass_out = C1 * vout
    # 更新区域体积

    if vt > 0:
        
        if vnet > 0:
            v2 += vt
            v1 = v1max
        else:
            v2 = max(0, v2 - vt)
            v1 = v1max + vt + vnet
    else:
        v1 += vnet 
        v1 = min(v1max, v1)
        v1 = max(0, v1)
        if v1max > v1:
            v2 = 0

    # 考虑反应动力学
    if v1 > 0:
        react_c1 =bulkreact(C1, kb, order, c_limit=0)*dt
        C1 = C1 + react_c1
    else:
        react_c1 = 0
    if v2 > 0:
        react_c2 =bulkreact(C2, kb, order, c_limit=0)*dt
    
        C2 = C2 + react_c2
    else:
        react_c2 = 0

    
    # 更新体积
    volume = v1 + v2
    mass_reacted = react_c1 * v1 + react_c2 * v2

    
    return volume, C1, mass_in, mass_out, mass_reacted, v1, v2, C2 

import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple

def simulate_mixing_models(
    duration: float,
    dt: float,
    initial_volume: float,
    initial_conc: float,
    v1max: float,
    inflow_rate: float,
    outflow_rate: float,
    inlet_conc: float,
    kb: float,
    order: float
) -> Dict[str, Dict[str, List[float]]]:
    """
    模拟完全混合模型和两室混合模型
    
    Parameters:
        duration: float - 模拟总时长(h)
        dt: float - 时间步长(h)
        initial_volume: float - 初始水池体积
        initial_conc: float - 初始浓度
        v1max: float - 两室模型中混合区最大体积
        inflow_rate: float - 进水流量
        outflow_rate: float - 出水流量
        inlet_conc: float - 进水浓度
        kb: float - 反应系数
        order: float - 反应级数
        
    Returns:
        dict: 包含两个模型的模拟结果
    """
    # 计算时间步数
    steps = int(duration / dt+1)
    time_points = np.linspace(0, duration, steps)

    if initial_volume > v1max:
        v1 = v1max
        v2 = initial_volume - v1max
    else:
        v1 = initial_volume
        v2 = 0
    
    # 初始化结果存储
    results = {
        'complete_mix': {
            'volume': [initial_volume],
            'concentration': [initial_conc],
            'mass_in': [0],
            'mass_out': [0],
            'mass_reacted': [0]
        },
        'two_comp': {
            'volume': [initial_volume],
            'volume1': [v1],
            'volume2': [v2],
            'concentration1': [initial_conc],
            'concentration2': [initial_conc],
            'mass_in': [0],
            'mass_out': [0],
            'mass_reacted': [0]
        },
        'time': time_points
    }
    
    # 模拟循环
    for t in range(1, steps):
        # 完全混合模型
        vol_cm, conc_cm, mass_in_cm, mass_out_cm, mass_reacted_cm = completemix(
            inflow_rate, outflow_rate, inlet_conc,
            results['complete_mix']['concentration'][-1],
            results['complete_mix']['volume'][-1],
            kb, order, dt
        )
        
        # 两室混合模型
        vol_tc, conc1_tc, mass_in_tc, mass_out_tc, mass_reacted_tc, v_mix, v_stag, conc2_tc = twocompmix(
            results['two_comp']['volume'][-1],
            v1max,
            results['two_comp']['concentration1'][-1],
            results['two_comp']['concentration2'][-1],
            inflow_rate, outflow_rate, inlet_conc,
            kb, order, dt
        )
        
        # 存储结果
        results['complete_mix']['volume'].append(vol_cm)
        results['complete_mix']['concentration'].append(conc_cm)
        results['complete_mix']['mass_in'].append(mass_in_cm)
        results['complete_mix']['mass_out'].append(mass_out_cm)
        results['complete_mix']['mass_reacted'].append(mass_reacted_cm)
        
        results['two_comp']['volume'].append(vol_tc)
        results['two_comp']['volume1'].append(v_mix)
        results['two_comp']['volume2'].append(v_stag)
        results['two_comp']['concentration1'].append(conc1_tc)
        results['two_comp']['concentration2'].append(conc2_tc)
        results['two_comp']['mass_in'].append(mass_in_tc)
        results['two_comp']['mass_out'].append(mass_out_tc)
        results['two_comp']['mass_reacted'].append(mass_reacted_tc)
    
    return results

def plot_results(results: Dict[str, Dict[str, List[float]]]) -> None:
    """绘制模拟结果对比图"""
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
    
    # 绘制体积变化
    ax1.plot(results['time'], results['complete_mix']['volume'], 
             label='完全混合模型', linestyle='-')
    ax1.plot(results['time'], results['two_comp']['volume'], 
             label='两室模型', linestyle='--')
    ax1.set_xlabel('时间 (h)')
    ax1.set_ylabel('体积')
    ax1.set_title('水池体积变化')
    ax1.legend()
    
    # 绘制浓度变化
    ax2.plot(results['time'], results['complete_mix']['concentration'], 
             label='完全混合模型', linestyle='-')
    ax2.plot(results['time'], results['two_comp']['concentration1'], 
             label='两室模型-混合区', linestyle='--')
    ax2.plot(results['time'], results['two_comp']['concentration2'], 
             label='两室模型-停滞区', linestyle=':')
    ax2.set_xlabel('时间 (h)')
    ax2.set_ylabel('浓度')
    ax2.set_title('浓度变化')
    ax2.legend()
    
    # 绘制质量累积变化
    cumulative_mass_cm = np.cumsum(results['complete_mix']['mass_reacted'])
    cumulative_mass_tc = np.cumsum(results['two_comp']['mass_reacted'])
    
    ax3.plot(results['time'], cumulative_mass_cm, 
             label='完全混合模型', linestyle='-')
    ax3.plot(results['time'], cumulative_mass_tc, 
             label='两室模型', linestyle='--')
    ax3.set_xlabel('时间 (h)')
    ax3.set_ylabel('累积反应质量')
    ax3.set_title('累积反应质量变化')
    ax3.legend()
    
    # 绘制质量平衡
    mass_balance_cm = np.array(results['complete_mix']['mass_in']) - \
                      np.array(results['complete_mix']['mass_out']) - \
                      np.array(results['complete_mix']['mass_reacted'])
    mass_balance_tc = np.array(results['two_comp']['mass_in']) - \
                      np.array(results['two_comp']['mass_out']) - \
                      np.array(results['two_comp']['mass_reacted'])
                      
    ax4.plot(results['time'], mass_balance_cm, 
             label='完全混合模型', linestyle='-')
    ax4.plot(results['time'], mass_balance_tc, 
             label='两室模型', linestyle='--')
    ax4.set_xlabel('时间 (h)')
    ax4.set_ylabel('质量平衡')
    ax4.set_title('质量平衡变化')
    ax4.legend()
    
    plt.tight_layout()
    plt.show()

# 示例运行


def calculate_mass_balance(results, model_type):
    """计算质量守恒分析"""
    total_mass_in = np.sum(results[model_type]['mass_in'])
    total_mass_out = np.sum(results[model_type]['mass_out'])
    total_mass_reacted = np.sum(results[model_type]['mass_reacted'])
    
    # 计算初始和最终储存质量
    if model_type == 'complete_mix':
        initial_mass = results[model_type]['volume'][0] * results[model_type]['concentration'][0]
        final_mass = results[model_type]['volume'][-1] * results[model_type]['concentration'][-1]
    else:
        initial_mass = results[model_type]['volume'][0] * results[model_type]['concentration1'][0]  # 假设初始时两个区域浓度相同
        final_mass = results[model_type]['volume1'][-1] * results[model_type]['concentration1'][-1] + results[model_type]['volume2'][-1] * results[model_type]['concentration2'][-1]

    
    # 计算质量平衡误差
    mass_balance = initial_mass + total_mass_in - total_mass_out + total_mass_reacted - final_mass
    mass_balance_percent = abs(mass_balance) / (initial_mass + total_mass_in) * 100 if (initial_mass + total_mass_in) > 0 else 0
    
    return {
        'initial_mass': initial_mass,
        'final_mass': final_mass,
        'total_mass_in': total_mass_in,
        'total_mass_out': total_mass_out,
        'total_mass_reacted': total_mass_reacted,
        'mass_balance_error': mass_balance,
        'mass_balance_percent': mass_balance_percent
    }

def create_simulation_page():
    st.title("储水池混合模型模拟系统")
    
    # Create two columns for input parameters
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("基本参数设置")
        duration = st.number_input("模拟时长 (小时)", min_value=1.0, value=24.0, step=1.0)
        dt = st.number_input("时间步长 (小时)", min_value=0.1, value=1.0, step=0.1)
        initial_volume = st.number_input("初始水池体积 (m³)", min_value=0.0, value=1000.0, step=100.0)
        initial_conc = st.number_input("初始浓度", min_value=0.0, value=1.0, step=0.1)
        v1max = st.number_input("混合区最大体积 (m³)", min_value=0.0, value=500.0, step=100.0)
    
    with col2:
        st.subheader("流量和反应参数")
        inflow_rate = st.number_input("进水流量 (m³/h)", min_value=0.0, value=100.0, step=10.0)
        outflow_rate = st.number_input("出水流量 (m³/h)", min_value=0.0, value=100.0, step=10.0)
        inlet_conc = st.number_input("进水浓度", min_value=0.0, value=2.0, step=0.1)
        kb = st.number_input("反应系数", value=-0.1, step=0.01)
        order = st.number_input("反应级数", value=1.0, step=0.1)

    # Simulation button
    if st.button("运行模拟"):
        # Run simulation
        results = simulate_mixing_models(
            duration, dt, initial_volume, initial_conc, v1max,
            inflow_rate, outflow_rate, inlet_conc, kb, order
        )
        
        # Create plots
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 15))
        
        # Plot concentrations
        ax1.plot(results['time'], results['complete_mix']['concentration'], 
                label='complete_mixing', linestyle='-')
        ax1.plot(results['time'], results['two_comp']['concentration1'], 
                label='two_comp_mix', linestyle='--')
        ax1.plot(results['time'], results['two_comp']['concentration2'], 
                label='two_comp_stag', linestyle=':')
        ax1.set_xlabel('time(hours)')
        ax1.set_ylabel('concentration(mg/L)')
        ax1.legend()
        ax1.grid(True)
        ax1.set_title('concentration change')

        # Plot volumes
        ax2.plot(results['time'], results['complete_mix']['volume'], 
                label='complete_mixing', linestyle='-')
        ax2.plot(results['time'], results['two_comp']['volume'], 
                label='two_comp', linestyle='--')
        ax2.set_xlabel('time(hours)')
        ax2.set_ylabel('volume(m³)')
        ax2.legend()
        ax2.grid(True)
        ax2.set_title('volume change')

        # Plot cumulative mass balance
        cumulative_mass_in_cm = np.cumsum(results['complete_mix']['mass_in'])
        cumulative_mass_out_cm = np.cumsum(results['complete_mix']['mass_out'])
        cumulative_mass_reacted_cm = np.cumsum(results['complete_mix']['mass_reacted'])
        
        ax3.plot(results['time'], cumulative_mass_in_cm, 
                label='total_mass_in', linestyle='-')
        ax3.plot(results['time'], cumulative_mass_out_cm, 
                label='total_mass_out', linestyle='--')
        ax3.plot(results['time'], cumulative_mass_reacted_cm, 
                label='total_mass_reacted', linestyle=':')
        ax3.set_xlabel('time(hours)')
        ax3.set_ylabel('mass(g)')
        ax3.legend()
        ax3.grid(True)
        ax3.set_title('total mass change')

    
        
        
        # Calculate mass balance for both models
        cm_balance = calculate_mass_balance(results, 'complete_mix')
        tc_balance = calculate_mass_balance(results, 'two_comp')
        
        # Display mass balance results in two columns
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("模拟结果")
            st.pyplot(fig)


        with col2:
            # Mass balance analysis
            st.subheader("质量守恒分析")
            col3, col4 = st.columns(2)
            with col3:
                st.write("完全混合模型质量平衡")
                balance_cm_df = pd.DataFrame({
                    '项目': ['初始质量', '总进水质量', '总出水质量', '总反应质量', '最终质量', '误差', '误差百分比'],
                    '数值': [
                        f"{cm_balance['initial_mass']:.2f}",
                        f"{cm_balance['total_mass_in']:.2f}",
                        f"{cm_balance['total_mass_out']:.2f}",
                        f"{cm_balance['total_mass_reacted']:.2f}",
                        f"{cm_balance['final_mass']:.2f}",
                        f"{cm_balance['mass_balance_error']:.2f}",
                        f"{cm_balance['mass_balance_percent']:.2f}%"
                    ]
                })
                st.dataframe(balance_cm_df)
                
                # Add error assessment
                if cm_balance['mass_balance_percent'] < 1:
                    st.success("质量守恒误差小于1%，模拟结果可靠")
                elif cm_balance['mass_balance_percent'] < 5:
                    st.warning("质量守恒误差在1-5%之间，模拟结果可接受")
                else:
                    st.error("质量守恒误差大于5%，请检查模拟参数")
        
            with col4:
                st.write("两室模型质量平衡")
                balance_tc_df = pd.DataFrame({
                    '项目': ['初始质量', '总进水质量', '总出水质量', '总反应质量', '最终质量', '误差', '误差百分比'],
                    '数值': [
                        f"{tc_balance['initial_mass']:.2f}",
                        f"{tc_balance['total_mass_in']:.2f}",
                        f"{tc_balance['total_mass_out']:.2f}",
                        f"{tc_balance['total_mass_reacted']:.2f}",
                        f"{tc_balance['final_mass']:.2f}",
                        f"{tc_balance['mass_balance_error']:.2f}",
                        f"{tc_balance['mass_balance_percent']:.2f}%"
                    ]
                })
                st.dataframe(balance_tc_df)
                
                # Add error assessment
                if tc_balance['mass_balance_percent'] < 1:
                    st.success("质量守恒误差小于1%，模拟结果可靠")
                elif tc_balance['mass_balance_percent'] < 5:
                    st.warning("质量守恒误差在1-5%之间，模拟结果可接受")
                else:
                    st.error("质量守恒误差大于5%，请检查模拟参数")

            # Create summary tables
            st.subheader("模拟结果数据")
            
            # Complete mix summary
            st.write("完全混合模型汇总")
            complete_mix_df = pd.DataFrame({
                '时间(h)': results['time'],
                '浓度': results['complete_mix']['concentration'],
                '体积(m³)': results['complete_mix']['volume'],
                '进水质量': results['complete_mix']['mass_in'],
                '出水质量': results['complete_mix']['mass_out'],
                '反应质量': results['complete_mix']['mass_reacted']
            })
            st.dataframe(complete_mix_df)

            # Two compartment summary
            st.write("两室模型汇总")
            two_comp_df = pd.DataFrame({
                '时间(h)': results['time'],
                '混合区体积(m³)': results['two_comp']['volume1'],
                '停滞区体积(m³)': results['two_comp']['volume2'],
                '混合区浓度': results['two_comp']['concentration1'],
                '停滞区浓度': results['two_comp']['concentration2'],
                '总体积(m³)': results['two_comp']['volume'],
                '进水质量': results['two_comp']['mass_in'],
                '出水质量': results['two_comp']['mass_out'],
                '反应质量': results['two_comp']['mass_reacted']
            })
            st.dataframe(two_comp_df)

if __name__ == "__main__":
    create_simulation_page()

