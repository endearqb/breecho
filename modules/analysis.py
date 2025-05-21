# modules/analysis.py

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict, Any, Tuple
import numpy as np
from scipy import stats



def auto_explore_data(df: pd.DataFrame) -> dict:
    """自动探索DataFrame并输出分析方案"""
    
    # 基础数据概览
    row_count, col_count = df.shape
    dtypes_info = df.dtypes.to_dict()
    missing_info = df.isnull().sum().to_dict()
    
    data_overview = {
        "rows": row_count,
        "cols": col_count,
        "dtypes": {k: str(v) for k, v in dtypes_info.items()},
        "missing_counts": missing_info,
    }
    
    # 细分各类型列
    int_cols = df.select_dtypes(include=['int']).columns.tolist()
    float_cols = df.select_dtypes(include=['float']).columns.tolist()
    numeric_cols = int_cols + float_cols
    
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    
    # 尝试识别并转换时间列
    date_cols = []
    for col in categorical_cols[:]:  # 使用切片创建副本以避免在迭代时修改列表
        try:
            # 尝试将列转换为datetime
            pd.to_datetime(df[col])
            date_cols.append(col)
            categorical_cols.remove(col)  # 从分类列中移除时间列
        except (ValueError, TypeError):
            continue
            
    # 添加已经是datetime类型的列
    date_cols.extend([
        col for col in df.columns 
        if pd.api.types.is_datetime64_any_dtype(df[col])
    ])
    
    # 识别高基数分类变量 (超过20个不同值)
    high_cardinality_cols = [
        col for col in categorical_cols 
        if df[col].nunique() > 20
    ]
    
    # 推荐可视化和统计分析
    recommended_plots = []
    recommended_stats = ["descriptive"]  # 基础描述性统计总是需要的
    
    # 数值列分析建议
    if numeric_cols:
        recommended_plots.extend([
            "histogram", "boxplot", "violin" # 添加小提琴图

        ])
        
        if len(numeric_cols) > 1:
            recommended_plots.extend([
                "correlation_heatmap",
                "scatter_matrix"
            ])
    
    # 分类变量分析建议
    if categorical_cols:
        recommended_plots.extend([
            "barplot_category",
            "countplot"  # 添加计数图
        ])
        
        # 对高基数列的特殊处理建议
        if high_cardinality_cols:
            recommended_stats.append("top_n_categories")
        
        # 如果同时存在数值列，建议交叉分析
        if numeric_cols:
            recommended_plots.append("grouped_boxplot")
            recommended_stats.extend(["pivot_table", "group_statistics"])
    
    # 时间序列分析建议
    if date_cols and numeric_cols:
        recommended_plots.extend([
            "multi_timeseries",
            "time_decompose"  # 添加时间序列分解图
        ])
        recommended_stats.extend([
            "time_series_analysis",
            "temporal_aggregation",
            "trend_analysis"
        ])
    
    return {
        "data_overview": data_overview,
        "column_types": {
            "integer": int_cols,
            "float": float_cols,
            "categorical": categorical_cols,
            "high_cardinality": high_cardinality_cols,
            "datetime": date_cols
        },
        "recommended_plots": list(set(recommended_plots)),
        "recommended_stats": list(set(recommended_stats))
    }


def generate_plots_automatically(df: pd.DataFrame, recommended_plots: List[str]) -> Tuple[List[plt.Figure], Dict[str, Any]]:
    """根据推荐生成可视化图表并返回关键分析数据"""
    
    # 设置全局图表参数
    plt.rcParams['figure.figsize'] = (6, 3)  # 默认图表大小
    plt.rcParams['figure.dpi'] = 50  # 默认DPI
    plt.rcParams['figure.max_open_warning'] = 30  # 增加最大图表数量限制
    
    # 限制处理的数据量
    max_rows = 4000 # 设置最大行数限制
    if len(df) > max_rows:
        df = df.sample(n=max_rows, random_state=42)  # 随机采样
    
    figs = []
    analysis_data = {
        "distribution_stats": {},
        "correlation_stats": {},
        "groupby_stats": {},
        "timeseries_stats": {}
    }
    
    # 配色方案
    colors = ['#EF4444', '#F59E0B', '#10B981', '#3B82F6', '#6366F1', '#8B5CF6', '#EC4899']
    
    # 获取各类型列
    numeric_cols = df.select_dtypes(include=['int', 'float']).columns.tolist()
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    
    # 识别并转换时间列
    date_cols = []
    
    # 首先添加已经是datetime类型的列
    date_cols.extend([
        col for col in df.columns 
        if pd.api.types.is_datetime64_any_dtype(df[col])
    ])
    
    # 然后尝试转换categorical列中的时间格式
    for col in categorical_cols[:]:  # 使用切片创建副本以避免在迭代时修改列表
        if col not in date_cols:  # 避免重复处理
            try:
                df[col] = pd.to_datetime(df[col])
                date_cols.append(col)
                categorical_cols.remove(col)
            except (ValueError, TypeError):
                continue

    # 直方图
    if "histogram" in recommended_plots and numeric_cols:
        for i, col in enumerate(numeric_cols[:3]):
            fig, ax = plt.subplots(figsize=(6, 3))
            sns.histplot(df[col], kde=True, ax=ax, color=colors[i % len(colors)])
            ax.set_title(f"histogram of {col}")
            figs.append(fig)
            
            # 保存分布统计数据
            stats = {
                "mean": df[col].mean(),
                "median": df[col].median(),
                "std": df[col].std(),
                "skew": df[col].skew(),
                "kurtosis": df[col].kurtosis()
            }
            analysis_data["distribution_stats"][col] = stats
    
    if "boxplot" in recommended_plots:
        for i, col in enumerate(numeric_cols[:3]):
            fig, ax = plt.subplots(figsize=(6, 3))
            sns.boxplot(data=df[col], ax=ax, color=colors[i % len(colors)])
            plt.xticks(rotation=45)
            ax.set_title(f"boxplot of {col}")
            figs.append(fig)
    
    
    # 相关性热力图
    if "correlation_heatmap" in recommended_plots and len(numeric_cols) > 1:
        # 获取详细的相关性分析结果
        correlation_results = analyze_correlations(df, numeric_cols)
        analysis_data["correlation_stats"] = correlation_results
        
        # 原始数据热力图
        fig1, ax1 = plt.subplots(figsize=(10, 8))
        corr_original = df[numeric_cols].corr()
        sns.heatmap(corr_original, annot=True, cmap='coolwarm', ax=ax1)
        ax1.set_title("Original Data Correlations")
        plt.tight_layout()
        figs.append(fig1)
        
        # 变换后的数据热力图
        transformed_data = transform_numeric_data(df, numeric_cols)
        for transform_type, transformed_df in transformed_data.items():
            if transformed_df is not None:
                fig, ax = plt.subplots(figsize=(10, 8))
                corr = transformed_df.corr()
                sns.heatmap(corr, annot=True, cmap='coolwarm', ax=ax)
                ax.set_title(f"{transform_type.capitalize()} Transformed Correlations")
                plt.tight_layout()
                figs.append(fig)
    
    # 分组箱线图
    if "grouped_boxplot" in recommended_plots and numeric_cols and categorical_cols:
        for cat_col in categorical_cols[:2]:
            if df[cat_col].nunique() <= 20:
                for num_col in numeric_cols[:3]:
                    fig, ax = plt.subplots(figsize=(6, 3))
                    # fig.suptitle(f"grouped_boxplot")
                    sns.boxplot(x=cat_col, y=num_col, data=df, ax=ax, palette=colors)
                    plt.xticks(rotation=45)
                    ax.set_title(f"grouped_boxs of {num_col} by {cat_col}")
                    figs.append(fig)
                    
                    # 保存分组统计数据
                    group_stats = df.groupby(cat_col)[num_col].agg([
                        'mean', 'median', 'std', 'count'
                    ]).to_dict('index')
                    
                    # 计算组间差异显著性
                    # groups = [group for _, group in df.groupby(cat_col)[num_col]]
                    # f_stat, p_value = stats.f_oneway(*groups)
                    
                    # analysis_data["groupby_stats"][f"{num_col}_by_{cat_col}"] = {
                    #     "group_stats": group_stats,
                    #     "anova_test": {
                    #         "f_statistic": f_stat,
                    #         "p_value": p_value
                    #     }
                    # }
    
    # 时间序列图
    if "multi_timeseries" in recommended_plots and date_cols and numeric_cols:
        date_col = date_cols[0]
        fig, ax = plt.subplots(figsize=(6, 3))
        df_sorted = df.sort_values(date_col)
        # 增加但col的时序图
        
        for i, num_col in enumerate(numeric_cols[:3]):
            ax.plot(df_sorted[date_col], df_sorted[num_col], 
                   label=num_col, color=colors[i % len(colors)])
            
            # 计算时间序列统计数据
            ts_data = df_sorted[num_col]
            analysis_data["timeseries_stats"][num_col] = {
                "trend": {
                    "start_value": ts_data.iloc[0],
                    "end_value": ts_data.iloc[-1],
                    "change_pct": ((ts_data.iloc[-1] - ts_data.iloc[0]) / ts_data.iloc[0] * 100)
                    if ts_data.iloc[0] != 0 else float('inf')
                },
                "volatility": ts_data.std(),
                "seasonality": {
                    "monthly_mean": df_sorted.set_index(date_col)[num_col]
                    .resample('M').mean().to_dict()
                } if pd.api.types.is_datetime64_any_dtype(df_sorted[date_col]) else None
            }
        ax.legend()
        ax.set_title(f"Time Series Analysis")
        plt.xticks(rotation=45)
        figs.append(fig)
        

    # 散点矩阵
    if "scatter_matrix" in recommended_plots and len(numeric_cols) > 1:
        plot_cols = numeric_cols[:6]
        fig = plt.figure(figsize=(12, 12))
        fig.suptitle("Scatter Matrix", fontsize=16, fontweight='bold')
        
        for i, col1 in enumerate(plot_cols):
            for j, col2 in enumerate(plot_cols):
                if i != j:
                    ax = fig.add_subplot(len(plot_cols), len(plot_cols), 
                                       i * len(plot_cols) + j + 1)
                    ax.scatter(df[col1], df[col2], alpha=0.5, 
                             color=colors[i % len(colors)])
                    if i == len(plot_cols)-1:
                        ax.set_xlabel(col1)
                    if j == 0:
                        ax.set_ylabel(col2)
                    ax.set_title(f"Scatter matrix of {col1} vs {col2}")
        plt.tight_layout()
        figs.append(fig)
    
    # 添加分组分析
    group_cols = identify_group_columns(df)
    if group_cols:
        analysis_data["group_analysis"] = perform_group_analysis(df, group_cols)
        
        # 为每个分组创建可视化
        for group_col in group_cols:
            numeric_cols = df.select_dtypes(include=['int', 'float']).columns.tolist()
            for num_col in numeric_cols:
                # 分组柱状图
                fig, ax = plt.subplots(figsize=(6, 3))
                df.groupby(group_col)[num_col].mean().plot(kind='bar', ax=ax, color=colors)
                plt.xticks(rotation=45)
                ax.set_title(f"grouped_bar of {num_col} by {group_col}")
                figs.append(fig)
                
                # 分组箱线图
                fig, ax = plt.subplots(figsize=(6, 3))
                sns.boxplot(x=group_col, y=num_col, data=df, ax=ax, palette=colors)
                plt.xticks(rotation=45)
                ax.set_title(f"grouped_box of {num_col} by {group_col}")
                figs.append(fig)
    
    # 添加小提琴图
    if "violin" in recommended_plots and numeric_cols and categorical_cols:
        for cat_col in categorical_cols[:1]:  # 限制处理前1个分类列
            if df[cat_col].nunique() <= 3:  # 限制类别数量
                for num_col in numeric_cols[:3]:
                    fig, ax = plt.subplots(figsize=(6, 3))
                    sns.violinplot(x=cat_col, y=num_col, data=df, ax=ax, palette=colors)
                    plt.xticks(rotation=45)
                    ax.set_title(f"violin of {num_col} by {cat_col}")
                    figs.append(fig)
    
    # # 添加成对关系图
    # if "pairplot" in recommended_plots and len(numeric_cols) > 1:
    #     try:
    #         # 限制用于pairplot的变量数量
    #         max_vars = 6
    #         plot_numeric_cols = numeric_cols[:max_vars]
            
    #         if len(categorical_cols) > 0:
    #             hue_col = categorical_cols[0]
    #             g = sns.pairplot(df[plot_numeric_cols + [hue_col]], 
    #                            hue=hue_col,
    #                            diag_kind="kde",
    #                            plot_kws={'alpha': 0.6},
    #                            height=2)  # 减小每个子图的大小
    #         else:
    #             g = sns.pairplot(df[plot_numeric_cols],
    #                            diag_kind="kde",
    #                            plot_kws={'alpha': 0.6},
    #                            height=2)
    #         g.fig.suptitle("Pair Plot of Numeric Variables", y=1.02)
    #         figs.append(g.fig)
    #     except Exception as e:
    #         print(f"生成pairplot时出错: {str(e)}")
    
    # 添加计数图
    if "countplot" in recommended_plots and categorical_cols:
        for cat_col in categorical_cols:
            if df[cat_col].nunique() <= 20:
                fig, ax = plt.subplots(figsize=(12, 6))
                sns.countplot(data=df, x=cat_col, ax=ax, palette=colors)
                plt.xticks(rotation=45)
                ax.set_title(f"Count Distribution of {cat_col}")
                
                # 添加数值标签
                for p in ax.patches:
                    ax.annotate(f'{int(p.get_height())}', 
                              (p.get_x() + p.get_width()/2., p.get_height()),
                              ha='center', va='bottom')
                
                figs.append(fig)
    
    # # 添加热力图
    # if "heatmap" in recommended_plots and numeric_cols:
    #     # 计算数值列的相关性矩阵
    #     corr_matrix = df[numeric_cols].corr()
        
    #     # 创建热力图
    #     fig, ax = plt.subplots(figsize=(10, 8))
    #     sns.heatmap(corr_matrix, 
    #                annot=True,  # 显示数值
    #                cmap='coolwarm',  # 使用红蓝配色
    #                center=0,  # 将0设为中心点
    #                fmt='.2f',  # 保留两位小数
    #                ax=ax)
    #     ax.set_title("Correlation Heatmap")
    #     figs.append(fig)
    
    #添加时间序列分解图
    if "time_decompose" in recommended_plots and date_cols and numeric_cols:
        from statsmodels.tsa.seasonal import seasonal_decompose
        
        date_col = date_cols[0]
        for num_col in numeric_cols[:3]:  # 限制处理前三个数值列
            try:
                # 设置时间索引
                ts_data = df.set_index(date_col)[num_col]
                
                # 执行时间序列分解
                decomposition = seasonal_decompose(ts_data, period=12)
                
                # 创建分解图
                fig = plt.figure(figsize=(12, 10))
                fig.suptitle(f'Time Series Decomposition of {num_col}')
                
                # 原始数据
                plt.subplot(411)
                plt.plot(ts_data, color=colors[0])
                plt.title('Original')
                
                # 趋势
                plt.subplot(412)
                plt.plot(decomposition.trend, color=colors[1])
                plt.title('Trend')
                
                # 季节性
                plt.subplot(413)
                plt.plot(decomposition.seasonal, color=colors[2])
                plt.title('Seasonal')
                
                # 残差
                plt.subplot(414)
                plt.plot(decomposition.resid, color=colors[3])
                plt.title('Residual')
                
                plt.tight_layout()
                figs.append(fig)
                
            except Exception as e:
                print(f"无法为 {num_col} 创建时间序列分解图: {str(e)}")
    
    return figs, analysis_data


def identify_group_columns(df: pd.DataFrame) -> List[str]:
    """
    自动识别可能的分组列
    规则：
    1. 分类列且唯一值数量适中(2-50)
    2. 列名包含特定关键词
    """
    potential_group_cols = []
    group_keywords = ['group', 'category', 'type', 'class', 'department', 'region', 
                     'status', 'level', 'grade', '组', '类', '部门', '地区', '状态', '等级']
    
    for col in df.columns:
        unique_count = df[col].nunique()
        # 检查是否为分类列且基数适中
        if unique_count >= 2 and unique_count <= 50:
            # 检查列名是否包含关键词
            if any(keyword in col.lower() for keyword in group_keywords):
                potential_group_cols.append(col)
            # 检查是否为分类数据类型
            elif df[col].dtype in ['object', 'category']:
                potential_group_cols.append(col)
                
    return potential_group_cols


def perform_group_analysis(df: pd.DataFrame, group_cols: List[str] = None) -> Dict[str, Any]:
    """
    执行通用分组分析
    
    Parameters:
    - df: 数据框
    - group_cols: 指定的分组列列表，如果为None则自动识别
    
    Returns:
    - 包含分组分析结果的字典
    """
    if group_cols is None:
        group_cols = identify_group_columns(df)
    
    if not group_cols:
        return {"error": "No suitable grouping columns found"}
    
    numeric_cols = df.select_dtypes(include=['int', 'float']).columns.tolist()
    if not numeric_cols:
        return {"error": "No numeric columns found for analysis"}
    
    analysis_results = {}
    
    for group_col in group_cols:
        group_analysis = {
            "basic_stats": {},
            "advanced_stats": {},
            "distribution_analysis": {},
            "statistical_tests": {}
        }
        
        # 1. 基础统计
        basic_stats = df.groupby(group_col)[numeric_cols].agg([
            'count', 'mean', 'median', 'std', 'min', 'max'
        ]).round(2)
        group_analysis["basic_stats"] = basic_stats.to_dict()
        
        # 2. 高级统计
        for num_col in numeric_cols:
            # 计算组内占比
            total = df[num_col].sum()
            group_sums = df.groupby(group_col)[num_col].sum()
            proportions = (group_sums / total * 100).round(2)
            
            # 计算增长率（如果是时间序列数据）
            if any(pd.api.types.is_datetime64_any_dtype(df[col]) for col in df.columns):
                date_col = [col for col in df.columns 
                           if pd.api.types.is_datetime64_any_dtype(df[col])][0]
                growth_rates = df.groupby(group_col).apply(
                    lambda x: ((x[num_col].iloc[-1] - x[num_col].iloc[0]) / x[num_col].iloc[0] * 100)
                    if len(x) > 1 and x[num_col].iloc[0] != 0 else 0
                ).round(2)
                
                group_analysis["advanced_stats"][num_col] = {
                    "proportions": proportions.to_dict(),
                    "growth_rates": growth_rates.to_dict()
                }
            else:
                group_analysis["advanced_stats"][num_col] = {
                    "proportions": proportions.to_dict()
                }
        
        # 3. 分布分析
        for num_col in numeric_cols:
            # 计算四分位数
            quartiles = df.groupby(group_col)[num_col].quantile([0.25, 0.5, 0.75]).unstack()
            # 计算异常值
            group_analysis["distribution_analysis"][num_col] = {
                "quartiles": quartiles.to_dict(),
                "outliers_count": df.groupby(group_col).apply(
                    lambda x: len(x[
                        (x[num_col] < x[num_col].quantile(0.25) - 1.5 * (x[num_col].quantile(0.75) - x[num_col].quantile(0.25))) |
                        (x[num_col] > x[num_col].quantile(0.75) + 1.5 * (x[num_col].quantile(0.75) - x[num_col].quantile(0.25)))
                    ])
                ).to_dict()
            }
        
        # # 4. 统计检验
        # for num_col in numeric_cols:
        #     # ANOVA检验
        #     groups = [group[num_col].values for name, group in df.groupby(group_col)]
        #     try:
        #         f_stat, p_value = stats.f_oneway(*groups)
        #         group_analysis["statistical_tests"][num_col] = {
        #             "anova": {
        #                 "f_statistic": float(f_stat),
        #                 "p_value": float(p_value)
        #             }
        #         }
        #     except:
        #         group_analysis["statistical_tests"][num_col] = {
        #             "anova": "Could not perform ANOVA test"
        #         }
        
        analysis_results[group_col] = group_analysis
    
    return analysis_results


def transform_numeric_data(df: pd.DataFrame, numeric_cols: List[str]) -> Dict[str, pd.DataFrame]:
    """
    对数值列进行多种变换，包括标准化、归一化和对数变换
    
    Parameters:
    - df: 原始数据框
    - numeric_cols: 数值列名列表
    
    Returns:
    - 包含各种变换后数据的字典
    """
    from sklearn.preprocessing import StandardScaler, MinMaxScaler
    import numpy as np
    
    transformed_data = {}
    
    # 1. 标准化 (Z-Score)
    try:
        scaler = StandardScaler()
        standardized = pd.DataFrame(
            scaler.fit_transform(df[numeric_cols]),
            columns=numeric_cols,
            index=df.index
        )
        transformed_data['standardized'] = standardized
    except:
        transformed_data['standardized'] = None
    
    # 2. 归一化 (Min-Max)
    try:
        scaler = MinMaxScaler()
        normalized = pd.DataFrame(
            scaler.fit_transform(df[numeric_cols]),
            columns=numeric_cols,
            index=df.index
        )
        transformed_data['normalized'] = normalized
    except:
        transformed_data['normalized'] = None
    
    # 3. 对数变换
    try:
        # 对数变换前将非正值处理为很小的正数
        log_data = df[numeric_cols].copy()
        min_values = log_data[log_data > 0].min()
        for col in numeric_cols:
            if (log_data[col] <= 0).any():
                min_positive = min_values[col] if col in min_values else 1e-6
                log_data[col] = log_data[col] + abs(log_data[col].min()) + min_positive
        
        log_transformed = np.log(log_data)
        transformed_data['log'] = log_transformed
    except:
        transformed_data['log'] = None
    
    return transformed_data


def analyze_correlations(df: pd.DataFrame, numeric_cols: List[str]) -> Dict[str, Any]:
    """
    对原始数据和变换后的数据进行相关性分析
    
    Parameters:
    - df: 原始数据框
    - numeric_cols: 数值列名列表
    
    Returns:
    - 包含各种相关性分析结果的字典
    """
    correlation_analysis = {
        'original': {},
        'transformed': {},
        'significant_correlations': {}
    }
    
    # 1. 原始数据相关性
    original_corr = df[numeric_cols].corr()
    correlation_analysis['original']['matrix'] = original_corr.round(3).to_dict()
    
    # 2. 变换后的数据相关性
    transformed_data = transform_numeric_data(df, numeric_cols)
    
    for transform_type, transformed_df in transformed_data.items():
        if transformed_df is not None:
            corr = transformed_df.corr()
            correlation_analysis['transformed'][transform_type] = {
                'matrix': corr.round(3).to_dict(),
                'high_correlations': []
            }
            
            # 记录显著相关性
            for i in range(len(numeric_cols)):
                for j in range(i+1, len(numeric_cols)):
                    corr_value = corr.iloc[i, j]
                    if abs(corr_value) > 0.5:
                        correlation_analysis['transformed'][transform_type]['high_correlations'].append({
                            'var1': numeric_cols[i],
                            'var2': numeric_cols[j],
                            'correlation': float(corr_value)
                        })
    
    # 3. 显著相关性对比
    correlation_analysis['significant_correlations'] = {
        'improved_by_transform': [],
        'consistent_high': [],
        'transform_specific': []
    }
    
    # 比较原始数据和变换后数据的相关性
    original_high_corr = set()
    for i in range(len(numeric_cols)):
        for j in range(i+1, len(numeric_cols)):
            orig_corr = abs(original_corr.iloc[i, j])
            if orig_corr > 0.5:
                original_high_corr.add((numeric_cols[i], numeric_cols[j]))
    
    # 检查每种变换后的相关性变化
    for transform_type, transformed_df in transformed_data.items():
        if transformed_df is not None:
            corr = transformed_df.corr()
            for i in range(len(numeric_cols)):
                for j in range(i+1, len(numeric_cols)):
                    trans_corr = abs(corr.iloc[i, j])
                    orig_corr = abs(original_corr.iloc[i, j])
                    var_pair = (numeric_cols[i], numeric_cols[j])
                    
                    # 相关性显著提高
                    if trans_corr > orig_corr + 0.1:
                        correlation_analysis['significant_correlations']['improved_by_transform'].append({
                            'transform': transform_type,
                            'var1': numeric_cols[i],
                            'var2': numeric_cols[j],
                            'original_corr': float(orig_corr),
                            'transformed_corr': float(trans_corr)
                        })
                    
                    # 始终保持高相关
                    if orig_corr > 0.5 and trans_corr > 0.5:
                        correlation_analysis['significant_correlations']['consistent_high'].append({
                            'var1': numeric_cols[i],
                            'var2': numeric_cols[j],
                            'correlations': {
                                'original': float(orig_corr),
                                transform_type: float(trans_corr)
                            }
                        })
    
    return correlation_analysis

    """

    auto_explore_data 函数说明:
    
    该函数对数据进行自动探索分析,包含以下几个主要部分:

    1. 基础数据概览:
       - 统计行数和列数
       - 获取每列的数据类型
       - 统计缺失值情况
    
    2. 列类型识别:
       - 识别数值型列(整数和浮点数)
       - 识别分类型列
       - 识别高基数分类列(不同值超过20个)
       - 识别日期时间列
       
    3. 推荐可视化:
       - 根据数据特征推荐合适的图表类型
       - 包括直方图、箱线图、相关性热图等
       
    4. 推荐统计分析:
       - 根据数据特征推荐统计分析方法
       - 包括描述性统计、相关性分析、分组分析等
       
    参数说明:
    df: pandas DataFrame - 输入数据
    
    返回值:
    dict - 包含数据概览、列类型、推荐图表和统计分析的字典

    generate_plots_automatically 函数说明:
    
    该函数根据推荐自动生成可视化图表,包含以下几个主要部分:

    1. 分布图:
       - 为数值列生成直方图
       - 添加核密度估计曲线
       - 计算并保存分布统计数据
    
    2. 相关性分析:
       - 生成相关性热力图
       - 识别并记录高相关性变量对
       
    3. 分组分析可视化:
       - 生成分组箱线图
       - 执行方差分析检验
       - 保存分组统计数据
       
    4. 时间序列分析:
       - 生成多变量时间序列图
       - 计算趋势和季节性指标
       - 保存时间序列统计数据
       
    5. 散点矩阵:
       - 生成数值变量间的散点图矩阵
       - 可视化变量间的关系
    
    参数说明:
    df: pandas DataFrame - 输入数据
    recommended_plots: List[str] - 推荐的图表类型列表
    
    返回值:
    Tuple[List[plt.Figure], Dict] - 图表对象列表和分析数据字典

    identify_group_columns 函数说明:
    
    该函数用于自动识别适合分组的列,包含以下判断标准:

    1. 基数判断:
       - 列的唯一值数量在2-50之间
       - 避免基数过高或过低的列
    
    2. 列名关键词匹配:
       - 检查列名是否包含分组相关关键词
       - 支持中英文关键词匹配
       
    3. 数据类型判断:
       - 识别object和category类型的列
       - 这些列通常适合作为分组依据
    
    参数说明:
    df: pandas DataFrame - 输入数据
    
    返回值:
    List[str] - 识别出的适合分组的列名列表
    """
