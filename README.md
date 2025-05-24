# 微风轻语耳边风 (Breecho) - 中文说明

## 介绍

本仓库包含 "微风轻语耳边风" (breecho.cn) 项目中使用 Streamlit 构建的多个应用程序和模块的源代码。这些应用主要围绕环境工程计算、AI 智能体和数据分析等功能。
![image](https://github.com/user-attachments/assets/620d3e68-ab2e-4ac8-8013-f8e996017562)
## 文件说明

以下是仓库中主要 Python 文件的功能简介：

* **`Breecho.py`**:
    * **功能**: 项目的主应用程序文件。它使用 Streamlit 构建了一个Web界面，作为访问和展示仓库中其他计算模型和 AI 智能体的入口。用户可以通过这个界面导航到不同的功能模块。
    * **技术栈**: Streamlit, Pandas, Numpy。

* **`AITutor.py`**:
    * **功能**: 一个AI家庭教师聊天机器人。它采用苏格拉底式的对话风格，引导用户解决问题，而不是直接给出答案。它可以根据用户的知识水平调整问题难度，并鼓励独立思考。
    * **技术栈**: Streamlit, OpenAI API。

* **`AITutormath.py`**:
    * **功能**: 一个AI小学数学应用题学习伙伴。它首先会将用户输入的数学题进行“精读”，将条件分段并转换为更清晰的数字信息，然后引导用户一步步解决问题。
    * **技术栈**: Streamlit, OpenAI API (特别是 `deepseek-coder` 模型)。

* **`ASM1slim.py`**:
    * **功能**: 活性污泥模型1 (ASM1) 的简化版模拟器。它用于模拟污水处理过程中好氧和缺氧池的生化反应，用户可以输入流量、水质、经验参数等，程序会计算并展示COD、硝态氮、氨氮、碱度等指标随时间的变化。
    * **技术栈**: Streamlit, PyTorch, TorchDiffEq, Matplotlib。
    * **核心功能**: 实现了基于ASM1模型的常微分方程 (ODE) 求解，用于模拟不同工况下的污染物去除效果。包含流量平衡、水质平衡计算，以及3D可视化结果。

* **`DocQA.py`**:
    * **功能**: 一个文档问答和分析系统。用户可以输入URL或直接粘贴文本，系统能够：
        * 提取文章内容。
        * 生成文章摘要。
        * 从文本中提取知识三元组并生成知识图谱 (使用 Mermaid 或 agraph 可视化)。
        * 基于文本内容生成问答对。
        * 导出分析报告 (TXT格式)。
    * **技术栈**: Streamlit, OpenAI API, BeautifulSoup, Requests, Pandas, FPDF。

* **`app.py` (Excel分析助手Demo)**:
    * **功能**: 一个Excel数据自动分析工具的演示版本。用户可以上传Excel或CSV文件（或使用内置示例数据），程序会自动进行数据清洗、数据概览、自动化数据分析（推荐可视化和统计分析），并生成多种图表（如分布图、相关性图、分类图、时间序列图）和AI洞察报告。
    * **技术栈**: Streamlit, Pandas, Matplotlib, Seaborn, OpenAI API。
    * **模块**: 依赖 `modules/data_cleaning.py`, `modules/analysis.py`, `modules/ai_insight.py`。

* **`calculate24.py`**:
    * **功能**: 一个计算24点游戏的程序。用户输入四个1到10之间的整数，程序会找出所有可能通过加、减、乘、除运算得到24的算式，并以LaTeX格式展示。
    * **技术栈**: Streamlit, itertools。

* **`chatwithstudywithfun.py`**:
    * **功能**: 一个基于知识库的聊天机器人。它使用 LlamaIndex 从预设的知识库（例如“微风轻语公众号”文章或“ASM与二沉池建模设计”相关文档）中检索信息，并回答用户的问题。聊天记录会被保存。
    * **技术栈**: Streamlit, LlamaIndex (OpenAI LLM and Embedding)。

* **`dwa.py`**:
    * **功能**: 根据德国水协 DWA-A-131 标准进行A/O（缺氧/好氧）工艺设计的计算器。用户输入进出水水质、流量、设计参数等，程序计算包括污泥龄、污泥产量、各池容积、回流比、耗氧量、标准传氧速率以及二沉池相关参数。
    * **技术栈**: Streamlit, Numpy, Matplotlib, Pandas。
    * **核心计算**: 包含硝化、反硝化、碳平衡、磷平衡、污泥产量、池容积和供氧计算。

* **`gb.py`**:
    * **功能**: 根据中国《室外排水设计标准》(GB 50014-2021) 进行A/O（缺氧/好氧）工艺设计的计算器。用户输入设计流量、进出水总氮、BOD5浓度、污泥产率系数、设计温度等参数，程序计算缺氧池容积、好氧池容积以及混合液回流量。
    * **技术栈**: Streamlit, Numpy, Matplotlib。

* **`llm.py`**:
    * **功能**: 一个基础的聊天机器人界面，直接调用 OpenAI API (如 `yi-medium` 模型) 进行对话。
    * **技术栈**: Streamlit, OpenAI API。

* **`mix1.py` (储水池余氯衰减高级模拟系统)**:
    * **功能**: 模拟储水池中余氯等物质浓度随时间变化的系统，基于EPANET中的混合模型原理。它实现了完全混合模型和两室混合模型（一个混合区，一个停滞区）的计算，考虑了体相反应和壁面反应。用户可以设置模拟时长、初始条件、流量、反应系数等参数，程序会输出浓度、体积和物料平衡随时间的变化图表。
    * **技术栈**: Streamlit, Numpy, Scipy, Matplotlib, Pandas。

* **`takacs.py`**:
    * **功能**: 二沉池一维 Takacs 双指数模型的计算器。用于模拟二沉池内不同深度的污泥浓度分布。用户输入二沉池深度、表面负荷、回流比、进水污泥浓度、沉降参数等，程序计算并展示各层污泥浓度。
    * **技术栈**: Streamlit, PyTorch, TorchDiffEq, Matplotlib。

* **`translator.py`**:
    * **功能**: 一个实现三步翻译（直译 -> 分析问题 -> 意译）的英译中工具。它旨在将专业学术内容翻译成易于理解的中文科普风格，并能处理Markdown格式和特定术语。
    * **技术栈**: Streamlit, OpenAI API (特别是 `yi-large-turbo` 模型)。

### `modules/` 文件夹:

* **`ai_insight.py`**:
    * **功能**: 提供将数据分析结果传递给大语言模型（如 `deepseek-chat`）以生成文字性洞察和解释的功能。
    * **技术栈**: OpenAI API, Pandas, JSON。

* **`analysis.py`**:
    * **功能**: 包含自动数据探索和可视化图表生成的函数。
        * `auto_explore_data`: 分析DataFrame，输出数据概览（行数、列数、数据类型、缺失值）、列类型细分（数值、分类、高基数分类、时间），并推荐合适的可视化图表和统计分析方法。
        * `generate_plots_automatically`: 根据推荐自动生成多种图表（如直方图、箱线图、相关性热力图、分组箱线图、时间序列图、散点矩阵、计数图、时间序列分解图等），并返回关键分析数据。
        * 还包含识别分组列、执行分组分析、数值数据变换（标准化、归一化、对数变换）和详细相关性分析的辅助函数。
    * **技术栈**: Pandas, Matplotlib, Seaborn, Numpy, Scipy.stats, Statsmodels。

* **`data_cleaning.py`**:
    * **功能**: 提供基础的数据清洗功能，包括去除完全空的行和列，使用均值（数值列）或特定字符串（对象列）填充缺失值，以及去除重复行。
    * **技术栈**: Pandas。

## 安装与运行

1.  **安装 Python**: 确保您的系统已安装 Python (建议 3.8 或更高版本)。
2.  **克隆仓库**:
    ```bash
    git clone <your-repository-url>
    cd <repository-name>
    ```
3.  **创建虚拟环境** (推荐):
    ```bash
    python -m venv venv
    source venv/bin/activate  # Linux/macOS
    # venv\Scripts\activate  # Windows
    ```
4.  **安装依赖**:
    ```bash
    pip install -r requirements.txt
    ```requirements.txt` 文件内容如下:
    ```txt
    streamlit
    pandas
    numpy
    openai
    torch
    torchdiffeq
    matplotlib
    scipy
    llama-index
    llama-index-llms-openai
    llama-index-embeddings-openai
    beautifulsoup4
    requests
    seaborn
    statsmodels
    fpdf
    streamlit-shadcn-ui
    st-pages
    streamlit-mermaid
    streamlit-agraph
    # itertools 是Python标准库的一部分，不需要在requirements.txt中列出
    ```
5.  **配置 API 密钥**:
    * 在涉及 OpenAI API 调用的文件中 (如 `AITutor.py`, `DocQA.py`, `llm.py`, `translator.py`, `modules/ai_insight.py`, `chatwithstudywithfun.py`)，找到 `API_KEY` 和 `API_BASE` (如果需要) 的占位符，并替换为您的有效密钥和API基础URL。
    ```python
    API_KEY = "您的OpenAI API密钥"
    API_BASE = "您的OpenAI API基础URL (如果不是官方API)"
    ```
    * 对于 `chatwithstudywithfun.py`，还需要设置环境变量 `OPENAI_API_KEY` 和 `OPENAI_API_BASE`。

6.  **准备数据和模型文件** (如果适用):
    * 对于 `chatwithstudywithfun.py`，确保 LlamaIndex 的存储路径 (如 `/opt/stapp/storagefun`, `/opt/stapp/asmstorage`) 包含预先构建好的索引文件。
    * 对于 `Breecho.py` 和其他可能引用本地图片或数据文件的脚本，请确保路径正确 (如 `/opt/stapp/gb.png`, `/opt/stapp/data/housing.csv`)。如果这些文件不在仓库中，您需要提供它们或修改代码中的路径。

7.  **运行主应用**:
    ```bash
    streamlit run Breecho.py
    ```
    或者运行单个应用，例如:
    ```bash
    streamlit run AITutor.py
    ```

## 使用说明

* 启动应用后，通过浏览器访问 Streamlit 提供的本地URL (通常是 `http://localhost:8501`)。
* 根据 `Breecho.py` 中的导航或直接运行单个 `.py` 文件来使用不同的计算器或AI工具。
* 大多数应用在侧边栏或主界面提供了参数输入区域，调整参数后点击相应的计算或分析按钮即可。

## 参与贡献

1.  Fork 本仓库。
2.  创建新的特性分支 (例如 `git checkout -b feature/AmazingFeature`)。
3.  提交您的更改 (例如 `git commit -m 'Add some AmazingFeature'`)。
4.  将您的更改推送到分支 (例如 `git push origin feature/AmazingFeature`)。
5.  打开一个 Pull Request。
