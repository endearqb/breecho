import streamlit as st
from openai import OpenAI
from st_pages import add_page_title, hide_pages
import time
import requests
from bs4 import BeautifulSoup
import re
import json
from urllib.parse import urlparse
import time
from datetime import datetime  
from typing import List, Dict, Optional, Tuple, Union
from pathlib import Path
from openai import OpenAI
from tenacity import retry, stop_after_attempt, wait_exponential
import argparse
import logging
from tqdm import tqdm
from streamlit_mermaid import st_mermaid
import unicodedata
from streamlit_agraph import agraph, Node, Edge, Config
from concurrent.futures import ThreadPoolExecutor, as_completed
from urllib.parse import urlparse
import pandas as pd
import os
from datetime import datetime
from fpdf import FPDF

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

add_page_title(layout="wide")


API_BASE_1 = "这里是base_url"
API_KEY_1 = "这里是key"



def parse_triples(triples_text):
    """
    Parse triples from text that may contain different formats of triple representations.
    
    Args:
        triples_text: List[str] or str - Input text containing triples
        
    Returns:
        List[Tuple[str, str, str]] - List of (entity1, relation, entity2) tuples
    """
    triples = []
    
    # Handle list input
    if isinstance(triples_text, list):
        # Join all text elements if it's a list
        triples_text = ' '.join(str(item) for item in triples_text)
    
    # Find all triple patterns in the text
    # Pattern 1: (entity1, relation, entity2)
    pattern1 = r'\(([^,]+),\s*([^,]+),\s*([^)]+)\)'
    
    # Pattern 2: (entity1, relation, entity2) with possible line numbers and brackets
    pattern2 = r'(?:\d+\.\s*)?\(([^,]+),\s*([^,]+),\s*([^)]+)\)'
    
    # Pattern 3: Triple patterns within code blocks
    pattern3 = r'\(([^,]+),\s*([^,]+),\s*([^)]+)\)'
    
    # Remove markdown code block syntax if present
    triples_text = re.sub(r'```(?:plaintext)?\n?(.*?)```', r'\1', triples_text, flags=re.DOTALL)
    
    # Find all matches using different patterns
    for pattern in [pattern1, pattern2, pattern3]:
        matches = re.finditer(pattern, triples_text)
        for match in matches:
            entity1 = match.group(1).strip()
            relation = match.group(2).strip()
            entity2 = match.group(3).strip()
            
            # Clean up entities and relations
            entity1 = re.sub(r'^[\'"(]|[\'")]$', '', entity1)
            relation = re.sub(r'^[\'"(]|[\'")]$', '', relation)
            entity2 = re.sub(r'^[\'"(]|[\'")]$', '', entity2)
            
            # Only add if all components are non-empty
            if entity1 and relation and entity2:
                triples.append((entity1, relation, entity2))
    
    # Remove duplicates while preserving order
    seen = set()
    unique_triples = []
    for triple in triples:
        if triple not in seen:
            seen.add(triple)
            unique_triples.append(triple)
    
    return unique_triples

def create_knowledge_graph_triples(file_path: str, sheet_name: str = 'Sheet1') -> List[Tuple[str, str, str]]:
    # Load the Excel file
    df = pd.read_excel(file_path, sheet_name=sheet_name)

    # Create a list to store the triples
    triples = []

    # Iterate over the rows to create triples for the knowledge graph
    for _, row in df.iterrows():
        # Extracting values from the dataframe
        process_stage = row['工艺段分类']
        process_name = row['工艺名']
        function_name = row['功能名称']
        chemical_name = row['化学名称']
        chemical_formula = row['分子式/俗名/主要成分']
        effect = row['作用']

        # Create triples for each relevant relationship, skipping if any value is missing
        if pd.notna(process_stage) and pd.notna(process_name):
            triples.append((process_stage, "包含工艺名", process_name))
        
        if pd.notna(process_name) and pd.notna(function_name):
            triples.append((process_name, "包含功能", function_name))
        
        if pd.notna(function_name) and pd.notna(chemical_name):
            triples.append((function_name, "使用化学品", chemical_name))
        
        if pd.notna(chemical_name) and pd.notna(chemical_formula):
            triples.append((chemical_name, "具有分子式", chemical_formula))
        
        if pd.notna(chemical_name) and pd.notna(effect):
            triples.append((chemical_name, "具有作用", effect))

    return triples

def create_knowledge_graph(triples):
    # 构建知识图谱中的节点和边
    nodes = {}
    edges = []

    # 创建节点和边
    for entity1, relation, entity2 in triples:
        if entity1 not in nodes:
            nodes[entity1] = Node(id=entity1, label=entity1)
        if entity2 not in nodes:
            nodes[entity2] = Node(id=entity2, label=entity2)
        edges.append(Edge(source=entity1, target=entity2, label=relation))

    return nodes.values(), edges

# class ArticleScraper:
#     def __init__(self):
#         self.headers = {
#             'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
#         }

class ArticleScraper:
    def __init__(self, max_workers: int = 5, timeout: int = 10):
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        self.max_workers = max_workers
        self.timeout = timeout
        
    def get_articles(self, urls: List[str], output_format: str = 'dict') -> List[Dict]:
        """
        批量获取多个文章的内容
        
        Args:
            urls: URL列表
            output_format: 输出格式，支持 'dict' 或 'dataframe'
            
        Returns:
            根据output_format返回字典列表或DataFrame
        """
        results = []
        failed_urls = []
        
        # 使用线程池并行处理URLs
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # 提交所有任务
            future_to_url = {executor.submit(self.get_article, url): url for url in urls}
            
            # 收集结果
            for future in as_completed(future_to_url):
                url = future_to_url[future]
                try:
                    result = future.result()
                    result['url'] = url  # 添加URL到结果中
                    results.append(result)
                    
                    if result['status'] == 'error':
                        failed_urls.append(url)
                        
                except Exception as e:
                    failed_urls.append(url)
                    results.append({
                        'status': 'error',
                        'message': f'处理失败: {str(e)}',
                        'url': url,
                        'content': None
                    })
        
        # 如果有失败的URL，添加到结果中
        if failed_urls:
            print(f"\n获取失败的URL数量: {len(failed_urls)}")
            print("失败的URL列表:")
            for url in failed_urls:
                print(f"- {url}")
        
        # 根据指定格式返回结果
        if output_format.lower() == 'dataframe':
            return pd.DataFrame(results)
        return results

    def get_article(self, url):
        """获取文章内容的主函数"""
        try:
            domain = urlparse(url).netloc
            
            # 判断是否为微信公众号文章
            if 'mp.weixin.qq.com' in domain:
                return self._get_wechat_article(url)
            else:
                return self._get_general_webpage(url)
                
        except Exception as e:
            return {
                'status': 'error',
                'message': f'抓取失败: {str(e)}',
                'content': None
            }
    
    def _get_wechat_article(self, url):
        """处理微信公众号文章"""
        try:
            response = requests.get(url, headers=self.headers, timeout=10)
            response.encoding = 'utf-8'
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # 获取文章标题
            title = soup.find('h1', class_='rich_media_title').get_text(strip=True)
            
            # 获取作者信息
            author = soup.find('a', class_='rich_media_meta rich_media_meta_link')
            author = author.get_text(strip=True) if author else '未知'
            
            # 获取发布时间
            publish_time = soup.find('em', class_='rich_media_meta rich_media_meta_text')
            publish_time = publish_time.get_text(strip=True) if publish_time else ''
            
            # 获取文章主体内容
            content = soup.find('div', class_='rich_media_content')
            if content:
                # 清理文本内容
                text_content = self._clean_content(content.get_text())
                
                return {
                    'status': 'success',
                    'title': title,
                    'author': author,
                    'publish_time': publish_time,
                    'content': text_content
                }
            else:
                raise Exception('无法找到文章内容')
                
        except Exception as e:
            raise Exception(f'处理微信文章时出错: {str(e)}')
    
    def _get_general_webpage(self, url):
        """处理一般网页文章"""
        try:
            response = requests.get(url, headers=self.headers, timeout=10)
            # 尝试检测编码
            response.encoding = response.apparent_encoding
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # 移除脚本和样式元素
            for script in soup(['script', 'style']):
                script.decompose()
            
            # 尝试找到文章标题
            title = soup.find('title').get_text(strip=True) if soup.find('title') else ''
            
            # 尝试找到文章主体
            # 常见的文章容器class名称
            content_classes = ['article', 'post', 'content', 'entry', 'main-content']
            content = None
            
            for class_name in content_classes:
                content = soup.find(['article', 'div', 'section'], class_=re.compile(class_name, re.I))
                if content:
                    break
            
            if not content:
                # 如果没找到特定类名，取body下的所有文本
                content = soup.find('body')
            
            if content:
                text_content = self._clean_content(content.get_text())
                
                return {
                    'status': 'success',
                    'title': title,
                    'content': text_content
                }
            else:
                raise Exception('无法找到文章内容')
                
        except Exception as e:
            raise Exception(f'处理网页时出错: {str(e)}')
    
    def _clean_content(self, content):
        """清理文本内容"""
        # 删除多余的空白字符
        content = re.sub(r'\s+', ' ', content)
        # 删除空行
        content = re.sub(r'\n\s*\n', '\n', content)
        # 整理段落
        content = content.strip()
        return content


# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('doc_qa.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class DocQASystem:
    def __init__(self, api_key: str, api_base:str, model: str = "deepseek-coder"):
        """
        初始化文档问答系统
        
        Args:
            api_key (str): API密钥
            model (str): 使用的模型名称
            chunk_size (int): 文本分块大小
        """
        self.client = OpenAI(api_key=api_key, base_url=api_base)
        self.model = model
        
    def split_text(self, text: str, min_size: int = 600, max_size: int = 900) -> List[str]:
        """
        将文本按照自然段落切分，并清理标点符号
        
        Args:
            text (str): 输入文本
            min_size (int): 最小段落长度，默认100字
                
        Returns:
            List[str]: 清理后的文本块列表
        """
        def clean_text(text: str) -> str:
            """清理文本中的多余标点和空白"""
            # 替换多个换行为单个换行
            text = re.sub(r'\n+', '\n', text.strip())
            # 替换多个空格为单个空格
            text = re.sub(r'\s+', ' ', text)
            # 清理重复的标点符号
            text = re.sub(r'([。！？，、：；])\1+', r'\1', text)
            # 清理括号和引号前后的空格
            text = re.sub(r'\s*([（）【】『』「」""''])\s*', r'\1', text)
            # 清理标点符号前的空格
            text = re.sub(r'\s+([。，！？：；、])', r'\1', text)
            return text.strip()
        
        # 首先按段落分割
        paragraphs = text.split('\n\n')
        chunks = []
        current_chunk = []
        current_length = 0
        
        for para in paragraphs:
            para = clean_text(para)
            if not para:
                continue
                
            para_length = len(para)
            
            # 如果当前段落本身就超过chunk_size，需要按句子切分
            if para_length > max_size:
                sentences = re.split('([。！？])', para)
                temp_para = ''
                for i in range(0, len(sentences), 2):
                    if i + 1 < len(sentences):
                        sentence = sentences[i] + sentences[i+1]
                    else:
                        sentence = sentences[i]
                    
                    if len(temp_para) + len(sentence) <= max_size:
                        temp_para += sentence
                    else:
                        # 确保临时段落达到最小长度
                        if len(temp_para) >= min_size:
                            chunks.append(clean_text(temp_para))
                        elif len(chunks) > 0:
                            # 如果不够最小长度，尝试合并到前一个chunk
                            if len(chunks[-1]) + len(temp_para) <= max_size:
                                chunks[-1] = clean_text(chunks[-1] + temp_para)
                            else:
                                chunks.append(clean_text(temp_para))
                        else:
                            chunks.append(clean_text(temp_para))
                        temp_para = sentence
                
                # 处理最后剩余的文本
                if temp_para:
                    if len(temp_para) >= min_size:
                        chunks.append(clean_text(temp_para))
                    elif len(chunks) > 0:
                        if len(chunks[-1]) + len(temp_para) <= max_size:
                            chunks[-1] = clean_text(chunks[-1] + temp_para)
                        else:
                            chunks.append(clean_text(temp_para))
                    else:
                        chunks.append(clean_text(temp_para))
                continue
            
            # 处理正常长度的段落
            if current_length + para_length <= max_size:
                current_chunk.append(para)
                current_length += para_length
            else:
                # 确保当前chunk达到最小长度
                current_text = ' '.join(current_chunk)  # 使用空格连接而不是换行
                if len(current_text) >= min_size:
                    chunks.append(clean_text(current_text))
                elif len(chunks) > 0:
                    # 如果不够最小长度，尝试合并到前一个chunk
                    if len(chunks[-1]) + len(current_text) <= max_size:
                        chunks[-1] = clean_text(chunks[-1] + ' ' + current_text)
                    else:
                        chunks.append(clean_text(current_text))
                else:
                    chunks.append(clean_text(current_text))
                current_chunk = [para]
                current_length = para_length
        
        # 处理最后的chunk
        if current_chunk:
            current_text = ' '.join(current_chunk)
            if len(current_text) >= min_size:
                chunks.append(clean_text(current_text))
            elif len(chunks) > 0:
                if len(chunks[-1]) + len(current_text) <= max_size:
                    chunks[-1] = clean_text(chunks[-1] + ' ' + current_text)
                else:
                    chunks.append(clean_text(current_text))
            else:
                chunks.append(clean_text(current_text))
        
        return chunks
    
    def generate_qa_prompt(self, text_chunk: str) -> str:
        """
        生成用于问答生成的prompt
        
        Args:
            text_chunk (str): 文本块
            
        Returns:
            str: 生成的prompt
        """
        prompt = f"""请分析以下文本，生成3-5个深入的问题并提供答案。要求：
1. 问题应该关注文本的关键信息和深层含义
2. 答案必须来自文本内容，并标注相关原文
3. 同时注意问题的多样性，可以包括：
   - 概念解释类问题
   - 因果关系类问题
   - 比较分析类问题
   - 应用实践类问题

请按照以下格式输出：
Q1: [问题1]
A1: [答案1]
原文: [相关文本摘录]

Q2: [问题2]
A2: [答案2]
原文: [相关文本摘录]

文本内容如下：
{text_chunk}
"""
        return prompt
    
    def generate_mindmap(self, text_chunk: str) -> str:
        """
        生成思维导图

        Args:
            text (str): 文本内容

        Returns:
            str: 生成的mermaid思维导图
        """
        prompt = f"""请分析以下内容，并将其内容转化为知识图谱的三元组结构，形式为 (实体1, 关系, 实体2)。对于输入文本，提取出其中具有明确关系的实体对，并定义相应的关系，注意要保持信息完整和简洁。
要求：
1. 尽量从文本中提取出所有重要的实体和关系。
2. 三元组结构应具有逻辑一致性，确保关系明确且表达清晰。
3. 输出结果应为 (实体1, 关系, 实体2) 的列表。
例如： 
输入文本："乔布斯是苹果公司的创始人之一，他领导开发了iPhone。" 
输出三元组：
(乔布斯, 创立, 苹果公司) 
(乔布斯, 领导开发, iPhone)

请分析文本内容并生成知识图谱的三元组结构，文本内容如下：
{text_chunk}
""" 
        return prompt
    
    def generate_summary(self, text_chunk: str) -> str:
        """
        生成总结

        Args:
            text (str): 文本内容

        Returns:
            str: 生成的mermaid思维导图
        """
        prompt = f"""请对以下内容进行总结，要求：
    1. 提取文章的主要观点和关键信息
    2. 总结长度控制在300字以内
    3. 保持文章的核心含义
    4. 使用简洁清晰的语言
    
    文章内容：
    {text_chunk}

        """
        return prompt


    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=10),
        retry_error_callback=lambda retry_state: None
    )
    def query_llm(self, chunk: str) -> Optional[str]:
        """
        查询LLM API
        
        Args:
            chunk (str): 文本块
            
        Returns:
            Optional[str]: API响应内容
        """
        try:
            prompt = self.generate_qa_prompt(chunk)
            messages = [{"role": "user", "content": prompt}]
            
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=0.3,
                max_tokens=1000,
                timeout=30
            )
            
            if response.choices and len(response.choices) > 0:
                return response.choices[0].message.content
            return None
            
        except Exception as e:
            logger.error(f"Error querying LLM: {str(e)}")
            raise
            
    def parse_qa_response(self, response: str) -> List[Dict]:
        """
        解析LLM的响应
        
        Args:
            response (str): LLM响应内容
            
        Returns:
            List[Dict]: 解析后的问答对列表
        """
        if not response:
            return []
            
        qa_pairs = []
        current_qa = {}
        
        lines = response.strip().split('\n')
        for line in lines:
            line = line.strip()
            if not line:
                continue
                
            if line.startswith('Q'):
                if current_qa.get('question'):
                    qa_pairs.append(current_qa.copy())
                current_qa = {'question': line[line.find(':')+1:].strip()}
            elif line.startswith('A'):
                current_qa['answer'] = line[line.find(':')+1:].strip()
            elif line.startswith('原文'):
                current_qa['reference'] = line[line.find(':')+1:].strip()
                
        if current_qa.get('question'):
            qa_pairs.append(current_qa)
            
        return qa_pairs
        
    def process_document(self, input_file: str) -> List[Dict]:
        """
        处理整个文档
        
        Args:
            input_file (str): 输入文件路径
            
        Returns:
            List[Dict]: 所有问答对列表
        """
        try:
            text = Path(input_file).read_text(encoding='utf-8')
        except Exception as e:
            logger.error(f"Error reading file {input_file}: {str(e)}")
            return []
            
        chunks = self.split_text(text)
        results = []
        
        for i, chunk in enumerate(tqdm(chunks, desc="Processing chunks")):
            try:
                response = self.query_llm(chunk)
                if response:
                    parsed_qa = self.parse_qa_response(response)
                    results.extend(parsed_qa)
                    time.sleep(1)  # 速率限制
                    
            except Exception as e:
                logger.error(f"Error processing chunk {i+1}: {str(e)}")
                continue
                
        return results
    
    def QAprocess_text(self, text: str, min_size: int = 1500, max_size: int = 2000) -> List[Dict]:
        """
        处理整个文档
        
        Args:
            input_file (str): 输入文件路径
            
        Returns:
            List[Dict]: 所有问答对列表
        """
            
        chunks = self.split_text(text, min_size, max_size)
        results = []
        
        for i, chunk in enumerate(tqdm(chunks, desc="Processing chunks")):
            try:
                response = self.query_llm(chunk)
                if response:
                    parsed_qa = self.parse_qa_response(response)
                    results.extend(parsed_qa)
                    time.sleep(1)  # 速率限制
                    
            except Exception as e:
                logger.error(f"Error processing chunk {i+1}: {str(e)}")
                continue
                
        return results
    
    def Summaryprocess_text(self, text: str, min_size: int = 2000, max_size: int = 3000) -> List[Dict]:
        """
        处理整个文档

        Args:
            input_file (str): 输入文件路径

        Returns:
            List[Dict]: 所有问答对列表
        """

        chunks = self.split_text(text, min_size, max_size)
        results = []

        for i, chunk in enumerate(tqdm(chunks, desc="Processing chunks")):
            try:
                prompt = self.generate_summary(chunk)
                messages = [{"role": "user", "content": prompt}]

                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    temperature=0.3,
                    max_tokens=1000,
                    timeout=30
                )

                if response.choices and len(response.choices) > 0:
                    summary = response.choices[0].message.content
                    results.append(summary)
                    time.sleep(1)  # 速率限制

            except Exception as e:
                logger.error(f"Error processing chunk {i + 1}: {str(e)}")
                continue

        return results
    
    def Mindmapprocess_text(self, text: str, min_size: int = 2500, max_size: int = 3000) -> List[Dict]:
        """
        处理整个文档

        Args:
            input_file (str): 输入文件路径

        Returns:
            List[Dict]: 所有问答对列表
        """

        chunks = self.split_text(text, min_size, max_size)
        results = []

        for i, chunk in enumerate(tqdm(chunks, desc="Processing chunks")):
            try:
                prompt = self.generate_mindmap(chunk)
                messages = [{"role": "user", "content": prompt}]

                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    temperature=0.3,
                    max_tokens=5000,
                    timeout=30
                )

                if response.choices and len(response.choices) > 0:
                    mindmap = response.choices[0].message.content
                    results.append(mindmap)
                    time.sleep(1)  # 速率限制

            except Exception as e:
                logger.error(f"Error processing chunk {i + 1}: {str(e)}")
                continue

        return results


    def save_results(self, results: List[Dict], output_file: str):
        """
        保存处理结果
        
        Args:
            results (List[Dict]): 问答对列表
            output_file (str): 输出文件路径
        """
        try:
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(results, f, ensure_ascii=False, indent=2)
            logger.info(f"Results saved to {output_file}")
        except Exception as e:
            logger.error(f"Error saving results to {output_file}: {str(e)}")

def convert_qa_to_markdown(qa_results: List[Dict], include_reference: bool = True) -> str:
    """
    将问答结果转换为Markdown表格格式
    
    Args:
        qa_results (List[Dict]): 问答结果列表
        include_reference (bool): 是否包含原文引用
    
    Returns:
        str: Markdown格式的表格
    """
    if not qa_results:
        return "No results found."
    
    # 构建表头
    headers = ["序号", "问题", "答案"]
    if include_reference:
        headers.append("原文引用")
    
    # 构建表格分隔符
    separator = "|" + "|".join(["---"] * len(headers)) + "|"
    
    # 构建表头行
    header_row = "|" + "|".join(headers) + "|"
    
    # 构建表格内容
    rows = []
    for i, qa in enumerate(qa_results, 1):
        row = [
            str(i),
            qa.get("question", "").replace("\n", "<br>"),
            qa.get("answer", "").replace("\n", "<br>")
        ]
        if include_reference:
            row.append(qa.get("reference", "").replace("\n", "<br>"))
        
        # 处理表格中的竖线字符，避免破坏表格结构
        row = [cell.replace("|", "\\|") for cell in row]
        rows.append("|" + "|".join(row) + "|")
    
    # 组合所有行
    markdown_table = "\n".join([header_row, separator] + rows)
    
    # 添加标题和基本统计信息
    title = f"# 文档问答分析结果\n\n"
    stats = f"- 总问题数：{len(qa_results)}\n- 生成时间：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
    
    return title + stats + markdown_table

from typing import List, Tuple, Union


def clean_node_id(text: str) -> str:
    """
    清理节点ID，移除特殊字符并替换空格
    处理各种特殊情况包括Unicode控制字符、不可见字符、表情符号等
    
    Args:
        text: 输入文本
        
    Returns:
        str: 清理后的节点ID
    """
    if not isinstance(text, str):
        return str(text)
    
    # 1. Unicode正规化，将组合字符转换为单个字符
    text = unicodedata.normalize('NFKC', text)
    
    # 2. 预处理各种特殊字符
    # 商标、版权相关
    text = re.sub(r'[™®©℠℗]', '', text)
    
    # 3. 处理各种空白字符
    text = re.sub(r'\s+', '_', text)
    
    # 4. 定义替换规则字典
    replacements = {
        # 英文标点和特殊字符
        ' ': '_',
        '®': '',
        '/': '_',
        '.': '_',
        '-': '_',
        '(': '',
        ')': '',
        '[': '',
        ']': '',
        '"': '',
        "'": '',
        ',': '_',
        ':': '_',
        ';': '_',
        '!': '',
        '?': '',
        '~': '',
        '@': '_at_',
        '#': '_hash_',
        '$': '_dollar_',
        '%': '_percent_',
        '&': '_and_',
        '*': '_star_',
        '+': '_plus_',
        '=': '_equals_',
        '<': '_lt_',
        '>': '_gt_',
        '|': '_pipe_',
        '\\': '_',
        
        # 中文标点符号
        '，': '_',  # 中文逗号
        '。': '_',  # 中文句号
        '、': '_',  # 顿号
        '；': '_',  # 中文分号
        '：': '_',  # 中文冒号
        '？': '',   # 中文问号
        '！': '',   # 中文感叹号
        '”': '',    # 中文引号
        '“': '',    # 中文引号
        '‘': '',    # 中文引号
        '’': '',    # 中文引号
        '（': '',   # 中文括号
        '）': '',   # 中文括号
        '【': '',   # 中文方括号
        '】': '',   # 中文方括号
        '《': '',   # 中文书名号
        '》': '',   # 中文书名号
        '～': '_',  # 中文波浪线
        '·': '_',  # 中文间隔号
        '…': '',    # 中文省略号
        '￥': '_yuan_', # 人民币符号
        '％': '_percent_', # 中文百分号
        '〈': '',   # 中文尖括号
        '〉': '',   # 中文尖括号
        '「': '',   # 中文引号
        '」': '',   # 中文引号
        '『': '',   # 中文引号
        '』': '',   # 中文引号
        '〔': '',   # 中文括号
        '〕': '',   # 中文括号
        '—': '_',   # 破折号
        '－': '_',  # 连接号
        '±': '_plus_minus_',

        # 新增：分数字符
        '½': '_half_',
        '¼': '_quarter_',
        '¾': '_three_quarters_',
        '⅓': '_third_',
        '⅔': '_two_thirds_',
        
        # 新增：音标符号
        'á': 'a',
        'à': 'a',
        'ã': 'a',
        'â': 'a',
        'ä': 'a',
        'å': 'a',
        'ā': 'a',
        'é': 'e',
        'è': 'e',
        'ê': 'e',
        'ë': 'e',
        'ē': 'e',
        'í': 'i',
        'ì': 'i',
        'î': 'i',
        'ï': 'i',
        'ī': 'i',
        'ó': 'o',
        'ò': 'o',
        'ô': 'o',
        'õ': 'o',
        'ö': 'o',
        'ō': 'o',
        'ú': 'u',
        'ù': 'u',
        'û': 'u',
        'ü': 'u',
        'ū': 'u',
        'ý': 'y',
        'ÿ': 'y',
        'ñ': 'n',
        
        # 新增：特殊字母变体
        'æ': 'ae',
        'œ': 'oe',
        'ß': 'ss',
        'ð': 'd',
        'þ': 'th',
        'ø': 'o',
        
        # 新增：单位符号
        '°': '_deg_',
        '′': '_prime_',
        '″': '_double_prime_',
        '℃': '_celsius_',
        '℉': '_fahrenheit_',
        'µ': '_micro_',
        'Ω': '_ohm_',
        '℧': '_mho_',
        
        # 新增：括号变体
        '❨': '',
        '❩': '',
        '❪': '',
        '❫': '',
        '❬': '',
        '❭': '',
        '❮': '',
        '❯': '',
        '❰': '',
        '❱': '',
        
        # 新增：方向性字符
        '←': '_left_',
        '→': '_right_',
        '↑': '_up_',
        '↓': '_down_',
        '↔': '_leftrightarrow_',
        '↕': '_updownarrow_',
        
        # 新增：技术符号
        '⌘': '_cmd_',
        '⌥': '_opt_',
        '⇧': '_shift_',
        '⌃': '_ctrl_',
        '⎋': '_esc_',
        '⏎': '_return_',
        '⌫': '_backspace_',
        
        # 新增：音乐符号
        '♩': '_quarter_note_',
        '♪': '_eighth_note_',
        '♫': '_beamed_eighth_notes_',
        '♬': '_beamed_sixteenth_notes_',
        
        # 新增：其他特殊符号
        '§': '_section_',
        '¶': '_paragraph_',
        '†': '_dagger_',
        '‡': '_double_dagger_',
        '•': '_bullet_',
        '⁂': '_asterism_',
        '⁎': '_low_asterisk_',
        '⁑': '_double_asterisk_',
    }
    
    # 5. 应用所有替换规则
    cleaned_text = text.strip()
    for old, new in replacements.items():
        cleaned_text = cleaned_text.replace(old, new)
    
    # 6. 移除表情符号和其他特殊Unicode字符
    cleaned_text = ''.join(c for c in cleaned_text if not unicodedata.category(c).startswith(('So', 'Cn', 'Co', 'Cs')))
    
    # 7. 移除控制字符
    cleaned_text = ''.join(c for c in cleaned_text if not unicodedata.category(c).startswith('C'))
    
    # 8. 只保留字母、数字、下划线
    cleaned_text = re.sub(r'[^\w\s]', '', cleaned_text)
    
    # 9. 处理连续的下划线
    cleaned_text = re.sub(r'_+', '_', cleaned_text)
    
    # 10. 移除开头和结尾的下划线
    cleaned_text = cleaned_text.strip('_')
    
    # 11. 长度限制（可选，根据需求设置）
    max_length = 255  # 或其他合适的长度
    if len(cleaned_text) > max_length:
        cleaned_text = cleaned_text[:max_length].rstrip('_')
    
    # 12. 确保结果不为空
    if not cleaned_text:
        cleaned_text = 'node'
        
    # 13. 如果节点ID以数字开头，添加前缀
    if cleaned_text and cleaned_text[0].isdigit():
        cleaned_text = 'n_' + cleaned_text
    
    return cleaned_text

def create_mermaid_flowchart(triples: List[Tuple[str, str, str]]) -> str:
    """
    创建Mermaid流程图
    
    Args:
        triples: 三元组列表
        
    Returns:
        str: Mermaid流程图字符串
    """
    if not triples:
        return "flowchart TD\n    A[No valid triples found]"
    
    # 创建Mermaid图表头部
    mermaid = ["flowchart LR"]
    
    # 用于存储已处理的边，避免重复
    processed_edges = set()
    
    # 为每个三元组创建边
    for subject, predicate, object in triples:
        try:
            # 清理节点ID
            subject_id = clean_node_id(subject)
            object_id = clean_node_id(object)
            
            if not subject_id or not object_id:
                continue
                
            # 创建节点标签
            edge = f"    {subject_id}[\"{subject}\"] -->|{predicate}| {object_id}[\"{object}\"]"
            
            # 只添加未处理过的边
            if edge not in processed_edges:
                mermaid.append(edge)
                processed_edges.add(edge)
        except Exception as e:
            print(f"Error processing triple ({subject}, {predicate}, {object}): {e}")
            continue
    
    return "\n".join(mermaid)

# 使用示例
def process_text_to_mermaid(input_text):
    """
    处理输入文本并生成Mermaid流程图
    
    Args:
        input_text: 输入文本
        
    Returns:
        str: Mermaid流程图字符串
    """
    try:
        # 解析三元组
        triples = parse_triples(input_text)
        
        # 创建Mermaid图表
        mermaid_chart = create_mermaid_flowchart(triples)
        
        return mermaid_chart
    except Exception as e:
        print(f"Error in process_text_to_mermaid: {e}")
        return "flowchart TD\n    A[Error processing input]"

def show_page():
    # 创建保存记录的目录
    SAVE_DIR = "analysis_records"
    if not os.path.exists(SAVE_DIR):
        os.makedirs(SAVE_DIR)

    # 创建session state来存储结果
    if 'results_summary' not in st.session_state:
        st.session_state.results_summary = None
    if 'results_mindmap' not in st.session_state:
        st.session_state.results_mindmap = None
    if 'results_qa' not in st.session_state:
        st.session_state.results_qa = None
    if 'knowledge_graph' not in st.session_state:
        st.session_state.knowledge_graph = None
    if 'content' not in st.session_state:
        st.session_state.content = None

    # 添加输入方式选择
    input_method = st.radio(
        "选择输入方式",
        ["URL输入", "文本输入"]
    )
    
    content_changed = False
    
    if input_method == "URL输入":
        url = st.text_input("输入网页URL")
        
        if url:
            st.info("开始处理URL内容...")
            logger.info(f"开始处理URL: {url}")
            scraper = ArticleScraper()
            result = scraper.get_article(url)
            
            # 获取内容
            with st.spinner("正在获取URL内容..."):
                content = result['content']
                if content != st.session_state.content:
                    content_changed = True
                    st.session_state.content = content
    
    else:  # 文本输入
        input_text = st.text_area(
            "直接输入文本内容",
            height=300,
            placeholder="请在此输入要分析的文本内容..."
        )
        
        if input_text:
            if input_text != st.session_state.content:
                content_changed = True
                st.session_state.content = input_text
            
    if st.session_state.content:
        # 显示提取的内容
        with st.expander("查看输入内容", expanded=False):
            st.text_area("内容预览", st.session_state.content, height=200)

        # 如果内容发生变化，清除之前的结果
        if content_changed:
            st.session_state.results_summary = None
            st.session_state.results_mindmap = None
            st.session_state.results_qa = None
            st.session_state.knowledge_graph = None
            
        qa_system = DocQASystem(
            api_key = API_KEY_1,
            api_base = API_BASE_1,
            model = "yi-lightning"
        )
            
        tab1, tab2, tab3, tab4 = st.tabs(["文章摘要", "知识图谱", "问答分析", "导出结果"])
        
        with tab1:
            if st.button("生成摘要") or st.session_state.results_summary is None:
                with st.spinner("正在生成摘要..."):
                    results_summary = qa_system.Summaryprocess_text(st.session_state.content)
                    if len(results_summary) > 1:
                        if isinstance(results_summary, list):
                            results_summary = '\n\n'.join(results_summary)
                        results_summary = results_summary.replace('"', '')
                        results_summary = qa_system.Summaryprocess_text(results_summary)
                        if isinstance(results_summary, list):
                            results_summary = '\n\n'.join(results_summary)
                        results_summary = results_summary.replace('"', '') 
                    else:
                        if isinstance(results_summary, list):
                            results_summary = '\n\n'.join(results_summary)
                        results_summary = results_summary.replace('"', '')               

                    st.session_state.results_summary = results_summary
            
            if st.session_state.results_summary:
                st.markdown(st.session_state.results_summary, unsafe_allow_html=True)

        with tab2:
            if st.session_state.results_mindmap is None:
                if st.button("生成知识图谱"):
                    with st.spinner("正在生成知识图谱..."):
                        results_mindmap = qa_system.Mindmapprocess_text(st.session_state.content)
                        triples = parse_triples(results_mindmap)
                        st.session_state.results_mindmap = triples
                        
                        nodes, edges = create_knowledge_graph(triples)
                        st.session_state.knowledge_graph = {"nodes": nodes, "edges": edges}
                        
                        config = Config(
                            width=1920,
                            height=1280,
                            directed=True,
                            nodeHighlightBehavior=True,
                            highlightColor="#F7A7A6",
                            collapsible=False,
                            staticGraph=True,
                        )
                        
                        agraph(nodes, edges, config)
            else:
                # 已经生成过知识图谱，只显示结果
                config = Config(
                    width=1920,
                    height=1280,
                    directed=True,
                    nodeHighlightBehavior=True,
                    highlightColor="#F7A7A6",
                    collapsible=False,
                    staticGraph=True,
                )
                agraph(
                    st.session_state.knowledge_graph["nodes"],
                    st.session_state.knowledge_graph["edges"],
                    config
                )
                
                # 可以选择性添加重新生成按钮
                if st.button("重新生成知识图谱"):
                    st.session_state.results_mindmap = None
                    st.rerun()

        with tab3:
            if st.session_state.results_qa is None:
                if st.button("生成问答分析"):
                    with st.spinner("正在生成问答分析..."):
                        results = qa_system.QAprocess_text(st.session_state.content)
                        results_qa = convert_qa_to_markdown(results, include_reference=False)
                        st.session_state.results_qa = results_qa
                        st.markdown(results_qa, unsafe_allow_html=True)
            else:
                # 已经生成过问答分析，显示保存的结果
                st.markdown(st.session_state.results_qa, unsafe_allow_html=True)
                
                # 可以选择性添加重新生成按钮
                if st.button("重新生成问答分析"):
                    st.session_state.results_qa = None
                    st.rerun()

        with tab4:
            st.header("导出分析结果")
            
            if st.button("生成分析报告"):
                # 检查是否有任何内容可以导出
                if not any([st.session_state.results_summary, 
                           st.session_state.results_mindmap, 
                           st.session_state.results_qa]):
                    st.warning("还没有生成任何分析内容")
                else:
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    filename_base = f"analysis_{timestamp}"
                    txt_path = os.path.join(SAVE_DIR, f"{filename_base}.txt")
                    
                    # 保存文本文件
                    with open(txt_path, 'w', encoding='utf-8') as f:
                        f.write("文档分析报告\n")
                        f.write(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
                        f.write("原始内容:\n")
                        f.write(st.session_state.content)
                        
                        # 如果有摘要，添加摘要
                        if st.session_state.results_summary:
                            f.write("\n\n文章摘要:\n")
                            f.write(st.session_state.results_summary)
                        
                        # 如果有知识图谱，添加知识图谱
                        if st.session_state.results_mindmap:
                            f.write("\n\n知识图谱:\n")
                            f.write('\n'.join([str(triple) for triple in st.session_state.results_mindmap]))
                        
                        # 如果有问答分析，添加问答分析
                        if st.session_state.results_qa:
                            f.write("\n\n问答分析:\n")
                            f.write(st.session_state.results_qa)
                    
                    # 创建下载按钮
                    with open(txt_path, 'r', encoding='utf-8') as f:
                        record_content = f.read()
                        # TXT下载
                        st.download_button(
                            label="下载TXT文件",
                            data=record_content,
                            file_name=f"{filename_base}.txt",
                            mime="text/plain"
                        )
                        
                        # 显示内容预览
                        with st.expander("查看内容"):
                            st.text_area("内容", record_content, height=400)
                    
                    # 显示已包含的内容提示
                    included_content = []
                    if st.session_state.results_summary:
                        included_content.append("文章摘要")
                    if st.session_state.results_mindmap:
                        included_content.append("知识图谱")
                    if st.session_state.results_qa:
                        included_content.append("问答分析")
                    
                    st.success(f"已导出内容包括：{', '.join(included_content)}")

def main():
    show_page()

if __name__ == '__main__':
    main()