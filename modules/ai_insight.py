from openai import OpenAI
import pandas as pd
import datetime
import json

API_BASE = "这里是base_url"
API_KEY = "这里是key"

client = OpenAI(
    api_key=API_KEY,
    base_url=API_BASE
)

def convert_keys_to_string(obj):
    """递归地将字典中的所有键转换为字符串"""
    if isinstance(obj, dict):
        # 对于字典，递归处理键和值
        # 键总是转换为字符串，值递归调用自身
        return {str(k): convert_keys_to_string(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        # 对于列表或元组，递归处理每个元素
        return [convert_keys_to_string(elem) for elem in obj]
    else:
        # 其他类型（字符串、数字、布尔、None、Timestamp等）保持不变
        # 后续的 json.dumps 会用 default 参数处理 Timestamp 等
        return obj
        
def default_serializer(obj):
    """自定义 JSON 序列化函数"""
    if isinstance(obj, (datetime.datetime, datetime.date, pd.Timestamp)):
        # 将日期时间对象转换为 ISO 8601 格式的字符串
        return obj.isoformat()
    # 如果有其他自定义类型需要处理，可以在这里添加 isinstance 判断
    # ...
    # 如果是其他无法序列化的类型，则抛出 TypeError，让错误更明显
    raise TypeError(f"Type {type(obj)} not serializable")

def generate_ai_insight(df: pd.DataFrame, analysis_result: dict) -> str:
    """
    将数据或分析结果传递给 GPT，生成文字洞察
    """
    # 注意：这里是一个简单的示例。
    # 你可能需要对 df 进行降维、抽象或只选取关键信息，以减少 token 消耗。

    # 假设我们把 analysis_result 转化为字符串
    analysis_result = {str(k): v for k, v in analysis_result.items()}
    analysis_result = convert_keys_to_string(analysis_result)
    analysis_text = json.dumps(analysis_result, ensure_ascii=False, indent=2, default=default_serializer)

    prompt = f"""
    下面是一些数据的统计结果，请你以简洁易懂的方式给出对数据的洞察和解释:
    {analysis_text}
    """
    messages = [{"role": "user", "content": prompt}]
    try:
        response = client.chat.completions.create(
            model="deepseek-chat",
            messages=messages,
            temperature=0.5, 
        )
        insight = response.choices[0].message.content
        return insight
    except Exception as e:
        return f"AI 分析失败: {e}"
