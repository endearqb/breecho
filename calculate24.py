# -*- coding: utf-8 -*-
# @Time    : 2024/05/10 17:01
# @Author  : endearqb

import math
import streamlit as st
import itertools
from st_pages import add_page_title, hide_pages
import streamlit_shadcn_ui as ui



# 设置全局字体
# mpl.rc('font', family='Times New Roman', size=12)  # 可以选择你系统支持的字体

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

def calculate24(nums):
    permutations = set(itertools.permutations(nums))
    ops = ['+', '-', '*', '/']
    op_combinations = list(itertools.product(ops, repeat=3))
    patterns = [
        '({}{}{}){}{}{}{}',
        '{}{}({}{}{}){}{}',
        '{}{}{}{}({}{}{})',
        '({}{}{}{}{}){}{}',
        '{}{}({}{}{}{}{})',
        '(({}{}{}){}{}){}{}',
        '({}{}({}{}{})){}{}',
        '{}{}(({}{}{}){}{})',
        '{}{}({}{}({}{}{}))',
        '({}{}{}){}({}{}{})',
        '{}{}{}{}{}{}{}',  # 无括号的情况
        # 您可以继续添加更多的括号组合
    ]
    results = set()

    for num_perm in permutations:
        for op_comb in op_combinations:
            for pattern in patterns:
                expr = pattern.format(
                    num_perm[0], op_comb[0], num_perm[1],
                    op_comb[1], num_perm[2], op_comb[2], num_perm[3]
                )
                try:
                    if abs(eval(expr) - 24) < 1e-6:
                        # 将表达式转换为 LaTeX 格式
                        expr_latex = expr.replace('*', r'\times')
                        expr_latex = expr_latex.replace('/', r'\div')
                        expr_latex = expr_latex.replace('-', r'-')
                        expr_latex = expr_latex.replace('+', r'+')
                        # 添加等于 24
                        expr_latex = f"{expr_latex}=24"
                        results.add(expr_latex)
                except ZeroDivisionError:
                    continue
                except Exception:
                    continue
    return results

def show_page1():
    st.title("24 点计算器")
    st.write("请输入四个 1 到 10 之间的整数，程序将计算所有可能得到 24 点的算式。")

    nums = []
    cols = st.columns(4)
    for i in range(4):
        num = cols[i].number_input(f"数字 {i+1}", min_value=1, max_value=10, value=1, step=1)
        nums.append(num)

    if st.button("计算"):
        with st.spinner("计算中，请稍候..."):
            results = calculate24(nums)
            if results:
                st.success(f"共找到 {len(results)} 个算式：")
                for expr in sorted(results):
                    st.latex(expr)
            else:
                st.warning("无法得到 24 点。")

def main():
    show_page1()

if __name__ == '__main__':
    main()



