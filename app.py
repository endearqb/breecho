# app.py

import streamlit as st
import numpy as np
from st_pages import add_page_title, hide_pages
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from modules.data_cleaning import clean_data
from modules.analysis import auto_explore_data, generate_plots_automatically
from modules.ai_insight import generate_ai_insight
import gc  # 添加在其他 import 语句旁边

# 设置页面配置
st.set_page_config(
    page_title="微风轻语BreeCho",
    page_icon="💭",
    layout="wide",
    initial_sidebar_state="auto",
)

# 隐藏Streamlit默认元素
hide_streamlit_style = """
    <style>
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    .css-1lsmgbg.egzxvld1 {visibility: hidden;}
    </style>
"""
st.markdown(hide_streamlit_style, unsafe_allow_html=True)

add_page_title(layout="wide")
hide_pages(["Thank you"])

# 设置中文字体
plt.rcParams['font.family'] = ['SimHei']

def convert_df_for_analysis(df):
    """将 DataFrame 转换为适合 AI 分析的格式"""
    df_converted = df.copy()
    
    # 将时间戳列转换为字符串
    for col in df_converted.select_dtypes(include=['datetime64']).columns:
        df_converted[col] = df_converted[col].dt.strftime('%Y-%m-%d %H:%M:%S')
    
    return df_converted

def display_column_info(column_types: dict):
    """展示列类型信息"""
    col1, col2 = st.columns(2)
    
    with col1:
        if column_types["integer"]:
            st.write("**整数类型列:**", ", ".join(column_types["integer"]))
        if column_types["float"]:
            st.write("**浮点类型列:**", ", ".join(column_types["float"]))
        if column_types["datetime"]:
            st.write("**时间类型列:**", ", ".join(column_types["datetime"]))
    
    with col2:
        if column_types["categorical"]:
            st.write("**分类类型列:**", ", ".join(column_types["categorical"]))
        if column_types["high_cardinality"]:
            st.write("**高基数分类列** (>20个不同值):", ", ".join(column_types["high_cardinality"]))

def main():
    # 初始化 session_state 变量
    if 'analysis_complete' not in st.session_state:
        st.session_state.analysis_complete = False
    if 'analysis_data' not in st.session_state:
        st.session_state.analysis_data = {
            "distribution_stats": {},
            "correlation_stats": {},
            "groupby_stats": {},
            "timeseries_stats": {}
        }
    if 'df_cleaned' not in st.session_state:
        st.session_state.df_cleaned = None
    if 'explore_result' not in st.session_state:
        st.session_state.explore_result = None
    if 'figs' not in st.session_state:
        st.session_state.figs = []
    if 'current_data_source' not in st.session_state:
        st.session_state.current_data_source = None
    if 'current_file' not in st.session_state:
        st.session_state.current_file = None

    # st.title("Excel自动分析工具（Demo）")

    st.markdown("### 功能使用说明")
    st.markdown("1. 选择数据来源：使用示例数据或上传文件")
    st.markdown("2. 点击`开始分析`按钮，等待数据分析完成")
    st.markdown("3. 查看自动生成的分析结果，包括数据概览、自动化数据分析、可视化分析和详细统计信息")
    st.markdown("4. 基于数据的前3600行自动生成的最多30个图表")
    st.markdown("5. 展开各个部分查看详细信息")
    st.markdown("6. 生成 AI 分析报告获取智能分析洞察")
    with st.expander("功能说明",expanded=False):
        st.markdown("""
这是一个基于 Streamlit 开发的 Excel 自动分析工具,主要功能包括:

1. 数据导入功能
- 支持示例数据和用户上传的 Excel/CSV 文件
- 自动检测文件类型并进行相应的数据读取
- 展示原始数据和清洗后的数据预览

2. excel数据格式要求
- 数据表中，第一行是列名
- 列名不能有重复
- 列名不能有空值
- 数据表中，只处理前2000行数据
- 数据表中，只处理前3个数值列
- 数据表中，只处理前1个分类列

3. 数据概览分析
- 显示数据行数、列数、缺失值情况等基本信息
- 自动识别并展示不同类型的列(数值、分类、时间等)
- 提供数据结构的详细分析

4. 自动化数据分析
- 生成推荐的可视化图表
- 提供统计分析建议
- 展示描述性统计信息

5. 可视化分析
- `分布分析`:直方图、箱线图、小提琴图、柱状图等
- `相关性分析`:热力图、散点图、矩阵图、成对图等
- `分类分析`:分组箱线图、计数图、分类柱状图等
- `时间序列分析`:趋势图、季节性分解图等

6. 详细统计信息
- `分布统计`:均值、中位数、标准差等
- `相关性分析`:变量间的相关系数
- `分组统计`:各组的统计特征
- `时间序列特征`:趋势、季节性分析

7. AI 洞察
- 基于数据自动生成分析报告
- 提供数据特征的智能解读

使用方法:
1. 选择数据来源(示例数据或上传文件)
2. 点击"开始分析"按钮
3. 查看自动生成的分析结果
4. 可展开各个部分查看详细信息
5. 点击"绘制图表"按钮获取自动生成的图表
6. 点击"生成 AI 分析报告"获取智能分析洞察
            """)
    # 选择数据来源
    data_source = st.radio("请选择数据来源：", ("使用示例数据", "上传文件"))

    # 检查数据源是否改变
    if data_source != st.session_state.current_data_source:
        st.session_state.clear()
        st.session_state.current_data_source = data_source
        st.rerun()

    if data_source == "使用示例数据":
        st.info("当前使用示例数据")
        try:
            df = pd.read_csv('/opt/stapp/data/housing.csv')
        except Exception as e:
            st.error(f"示例数据读取失败: {str(e)}")
            return
    else:
        # 清除之前的上传文件
        if 'uploaded_file_name' not in st.session_state:
            st.session_state.uploaded_file_name = None

        uploaded_file = st.file_uploader(
            "选择一个Excel或CSV文件", 
            type=["xlsx", "xls", "csv"],
            key="file_uploader"
        )

        # 检查是否有新文件上传
        if uploaded_file is not None:
            if st.session_state.uploaded_file_name != uploaded_file.name:
                # 新文件上传，清除之前的分析结果
                st.session_state.clear()
                st.session_state.current_data_source = data_source
                st.session_state.uploaded_file_name = uploaded_file.name
                # st.rerun()  # 立即重新运行，确保状态被清除
                
            try:
                file_extension = uploaded_file.name.split('.')[-1].lower()
                
                with st.spinner("正在加载数据..."):
                    if file_extension == 'csv':
                        df = pd.read_csv(uploaded_file)
                    else:
                        df = pd.read_excel(uploaded_file, sheet_name=0)
                st.success("文件上传成功！")
            except Exception as e:
                st.error(f"文件读取失败: {str(e)}")
                return
        else:
            st.warning("请先上传文件再进行后续操作。")
            return

    if 'analysis_complete' not in st.session_state:
        st.session_state.analysis_complete = False
    
    if not st.session_state.analysis_complete:
        # 添加3列，每列的列宽比例是1：2：1
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            if st.button("🚀 开始分析", type="primary", use_container_width=True):  
                # # 清理会话状态中的对象
                # # 清理所有会话状态
                # st.session_state.clear()

                # # 关闭所有打开的matplotlib图表
                # plt.close('all')

                # # 强制垃圾回收
                # gc.collect()
                
                df_cleaned = clean_data(df)

                # # 自动分析
                # with st.spinner("正在分析数据..."):explore_result = auto_explore_data(df_cleaned)

                        
                # 可视化部分
                # st.subheader("📈 数据分析")
                with st.spinner("正在分析数据、生成图表..."):
                    explore_result = auto_explore_data(df_cleaned)
                    figs, analysis_data = generate_plots_automatically(
                                df_cleaned, 
                                explore_result["recommended_plots"]
                    )
                
                st.session_state.df_cleaned = df_cleaned
                st.session_state.explore_result = explore_result
                st.session_state.analysis_data = analysis_data
                st.session_state.figs = figs    
                st.session_state.analysis_complete = True
        
    else:
        st.info("👆 点击上方按钮开始分析数据")

    if st.session_state.analysis_complete == True:
        st.info("数据分析已完成")
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("原始数据预览")
            st.dataframe(df.head(), use_container_width=True)
        with col2:
            st.subheader("清洗后数据预览")
            st.dataframe(st.session_state.df_cleaned.head(), use_container_width=True)

        # 自动分析
        st.subheader("📊 数据概览")
        overview = st.session_state.explore_result["data_overview"]
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("总行数", overview["rows"])
        with col2:
            st.metric("总列数", overview["cols"])
        with col3:
            missing_cols = sum(1 for v in overview["missing_counts"].values() if v > 0)
            st.metric("含缺失值的列数", missing_cols)

        # 列类型信息
        st.subheader("📑 数据结构分析")
        display_column_info(st.session_state.explore_result["column_types"])

        # 增加dataframe展示所有数据 默认折叠 增加数据分布展示
        st.subheader("📊 数据")
        st.dataframe(st.session_state.df_cleaned, use_container_width=True, hide_index=True)
                # 分析建议
        st.subheader("🔍 分析建议 &📊 数据分布")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.write("**推荐的可视化:**")
            for plot in st.session_state.explore_result["recommended_plots"]:
                st.write(f"- {plot}")
        with col2:
            st.write("**推荐的统计分析:**")
            for stat in st.session_state.explore_result["recommended_stats"]:
                st.write(f"- {stat}")
        with col3:
            st.write("**数据描述性统计:**")
            st.dataframe(st.session_state.df_cleaned.describe())

            # 缺失值详情
            if missing_cols > 0:
                with st.expander("查看缺失值详情"):
                    missing_df = pd.DataFrame.from_dict(
                        overview["missing_counts"], 
                        orient='index',
                        columns=['缺失值数量']
                    )
                    missing_df = missing_df[missing_df['缺失值数量'] > 0]
                    st.dataframe(missing_df)
           

    if  st.session_state.analysis_complete:
            # 显示关键分析数据
        with st.expander("📊 查看详细分析数据",expanded=False):
            # 分布统计
            cols1 = st.columns(4)
            if st.session_state.analysis_data["distribution_stats"]:
                with cols1[0]:  
                    st.subheader("分布统计")
                    for col, stats in st.session_state.analysis_data["distribution_stats"].items():
                        st.write(f"**{col}** 的统计数据:")
                        st.write(f"- 均值: {stats['mean']:.2f}")
                        st.write(f"- 中位数: {stats['median']:.2f}")
                        st.write(f"- 标准差: {stats['std']:.2f}")
                        st.write(f"- 偏度: {stats['skew']:.2f}")
                        st.write(f"- 峰度: {stats['kurtosis']:.2f}")
            
            # 相关性分析
            if st.session_state.analysis_data["correlation_stats"]:
                with cols1[1]:
                    st.subheader("显著相关性")
                    for corr in st.session_state.analysis_data["correlation_stats"].get("high_correlations", []):
                        st.write(
                            f"- {corr['var1']} 与 {corr['var2']} 的相关系数: "
                            f"{corr['correlation']:.2f}"
                        )
            
            # 分组分析
            if st.session_state.analysis_data["groupby_stats"]:
                with cols1[2]:
                    st.subheader("分组统计")
                    for key, stats in st.session_state.analysis_data["groupby_stats"].items():
                        st.write(f"**{key}** 的分组分析:")
                    st.write("组间差异显著性检验:")
                    p_value = stats["anova_test"]["p_value"]
                    st.write(
                        f"- P值: {p_value:.4f} "
                        f"({'显著' if p_value < 0.05 else '不显著'})"
                    )
                    st.write("各组统计数据:")
                    st.json(stats["group_stats"])
            
            # 时间序列分析
            if st.session_state.analysis_data["timeseries_stats"]:
                with cols1[3]:
                    st.subheader("时间序列分析")
                    for col, stats in st.session_state.analysis_data["timeseries_stats"].items():
                        st.write(f"**{col}** 的时间序列特征:")
                        st.write(f"- 总体变化: {stats['trend']['change_pct']:.2f}%")
                        st.write(f"- 波动性(标准差): {stats['volatility']:.2f}")
                    if stats['seasonality']:
                        st.write("- 月度均值趋势:")
                        st.line_chart(pd.Series(stats['seasonality']['monthly_mean']))

        # 可视化部分
        if st.session_state.figs:
            st.subheader("📈 可视化分析")
            st.markdown(f"""
共有 {len(st.session_state.figs)} 个图表，绘制图表时请耐心等待
                        """)
            # if st.button("🚀点击绘制图表"):
                # plt.close('all')  # 关闭所有未使用的图表
                # gc.collect()  # 强制垃圾回收
            with st.spinner("正在生成图表..."): 
                # 在生成图表之前添加
                plt.rcParams['figure.max_open_warning'] = 100  # 增加最大图表数量限制
                plt.rcParams['agg.path.chunksize'] = 10000  # 降低路径复杂度

                # 使用tabs来组织不同类型的图表
                viz_types = {
                    "分布分析": ["histogram", "distribution", "violin", "boxplot"],
                    "相关性分析": ["correlation", "scatter", "heatmap", "matrix", "pair"],
                    "分类分析": ["count", "bar", "category","grouped"],
                    "时间序列": ["time", "series", "decompose"]
                }
                
                # 对图表进行分类
                categorized_figs = {k: [] for k in viz_types.keys()}
                
                for fig in st.session_state.figs:
                    try:
                        if hasattr(fig, 'axes') and len(fig.axes) > 0:
                            # 获取图表标题
                            if hasattr(fig, 'texts') and fig.texts:
                                title = fig.texts[0].get_text().lower()
                            else:
                                title = fig.axes[0].get_title().lower()
                            
                            # 将图表分配到对应类别
                            for category, keywords in viz_types.items():
                                if any(keyword in title for keyword in keywords):
                                    # 限制图表大小
                                    fig.set_size_inches(10, 6)  # 设置统一的图表大小
                                    fig.set_dpi(100)  # 设置适中的DPI
                                    categorized_figs[category].append(fig)
                                    break
                    except Exception as e:
                        st.warning(f"处理图表时出错: {str(e)}")
                        continue
        
                # 创建标签页并显示图表
                tabs = st.tabs(list(viz_types.keys()))
                for tab, category in zip(tabs, viz_types.keys()):
                    with tab:
                        if categorized_figs[category]:
                            # 使用列布局显示图表
                            cols = st.columns(2)  # 每行显示2个图表
                            for i, fig in enumerate(categorized_figs[category]):
                                try:
                                    with cols[i % 2]:
                                        st.pyplot(fig)
                                except Exception as e:
                                    st.warning(f"显示图表时出错: {str(e)}")
                                    continue
                        else:
                            st.info(f"没有{category}类型的图表")

        # 在分析数据展示部分添加新的统计信息展示
        if "distribution_stats" in st.session_state.analysis_data:
            with st.expander("📊 分布统计详情",expanded=False):
                for col, stats in st.session_state.analysis_data["distribution_stats"].items():
                    st.write(f"**{col}** 的统计数据:")
                    col1, col2 = st.columns(2)
                    with col1:
                        st.write(f"- 均值: {stats['mean']:.2f}")
                        st.write(f"- 中位数: {stats['median']:.2f}")
                        st.write(f"- 标准差: {stats['std']:.2f}")
                    with col2:
                        st.write(f"- 偏度: {stats['skew']:.2f}")
                        st.write(f"- 峰度: {stats['kurtosis']:.2f}")

        # 添加时间序列分解结果展示
        if "timeseries_stats" in st.session_state.analysis_data:
            with st.expander("📈 时间序列分析详情",expanded=False):
                for col, stats in st.session_state.analysis_data["timeseries_stats"].items():
                    st.write(f"**{col}** 的时间序列特征:")
                    col1, col2 = st.columns(2)
                    with col1:
                        st.write("趋势分析:")
                        st.write(f"- 起始值: {stats['trend']['start_value']:.2f}")
                        st.write(f"- 结束值: {stats['trend']['end_value']:.2f}")
                        st.write(f"- 变化率: {stats['trend']['change_pct']:.2f}%")
                    with col2:
                        st.write("波动性分析:")
                        st.write(f"- 标准差: {stats['volatility']:.2f}")
                        if stats.get('seasonality'):
                            st.write("季节性分析:")
                            monthly_data = pd.Series(stats['seasonality']['monthly_mean'])
                            st.line_chart(monthly_data)

            # 在显示分析数据的部分添加分组分析结果的展示
        if st.session_state.analysis_data.get("group_analysis"):
            st.subheader("📊 分组分析结果")
            for group_col, analysis in st.session_state.analysis_data["group_analysis"].items():
                with st.expander(f"查看 {group_col} 的分组分析",expanded=False):
                    # 基础统计
                    st.write("**基础统计指标**")
                    for metric, values in analysis["basic_stats"].items():
                        st.write(f"- {metric}:")
                        st.json(values)
                    
                    # 高级统计
                    st.write("**高级统计分析**")
                    for num_col, stats in analysis["advanced_stats"].items():
                        st.write(f"**{num_col}** 的分析:")
                        st.write("- 各组占比:")
                        st.json(stats["proportions"])
                        if "growth_rates" in stats:
                            st.write("- 增长率:")
                            st.json(stats["growth_rates"])
                    
                    # 分布分析
                    st.write("**分布分析**")
                    for num_col, dist_stats in analysis["distribution_analysis"].items():
                        st.write(f"**{num_col}** 的分布:")
                        st.write("- 四分位数:")
                        st.json(dist_stats["quartiles"])
                        st.write("- 异常值数量:")
                        st.json(dist_stats["outliers_count"])
                    

            # 在显示分析数据的部分修改相关性分析的展示
        if st.session_state.analysis_data["correlation_stats"]:
            with st.expander("📊 相关性分析详情",expanded=False):
                # 原始数据相关性
                st.write("**原始数据相关性**")
                st.dataframe(
                    pd.DataFrame(st.session_state.analysis_data["correlation_stats"]["original"]["matrix"])
                )
                
                # 变换后的相关性
                st.write("**数据变换后的相关性**")
                for transform_type, transform_results in st.session_state.analysis_data["correlation_stats"]["transformed"].items():
                    st.write(f"*{transform_type.capitalize()} 变换后的相关性*")
                    st.dataframe(pd.DataFrame(transform_results["matrix"]))
                    
                    if transform_results["high_correlations"]:
                        st.write(f"*{transform_type.capitalize()} 变换后的显著相关性 (|r| > 0.5)*")
                        for corr in transform_results["high_correlations"]:
                            st.write(
                                f"- {corr['var1']} 与 {corr['var2']}: "
                                f"{corr['correlation']:.3f}"
                            )
                
                # 显著相关性变化
                st.write("**相关性变化分析**")
                
                if st.session_state.analysis_data["correlation_stats"]["significant_correlations"]["improved_by_transform"]:
                    st.write("*相关性显著提高的变量对:*")
                    for improved in st.session_state.analysis_data["correlation_stats"]["significant_correlations"]["improved_by_transform"]:
                        st.write(
                            f"- {improved['var1']} 与 {improved['var2']}: "
                            f"原始相关性 {improved['original_corr']:.3f} -> "
                            f"{improved['transform']}变换后 {improved['transformed_corr']:.3f}"
                        )
                
                if st.session_state.analysis_data["correlation_stats"]["significant_correlations"]["consistent_high"]:
                    st.write("*始终保持高相关的变量对:*")
                    for consistent in st.session_state.analysis_data["correlation_stats"]["significant_correlations"]["consistent_high"]:
                        st.write(
                            f"- {consistent['var1']} 与 {consistent['var2']}: "
                            f"相关性保持在 {min(consistent['correlations'].values()):.3f} 以上"
                        )
                
        # AI 洞察
        st.subheader("🤖 AI 洞察")
        if st.session_state.analysis_complete == True:
            with st.spinner("AI正在分析数据..."):
                summary = generate_ai_insight(
                    st.session_state.df_cleaned, 
                    {"auto_explore": st.session_state.explore_result,
                    "auto_analysis": st.session_state.analysis_data
                    }
                )
                st.write(summary)
        
        

if __name__ == "__main__":
    main()

