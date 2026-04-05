import os
import re
import json
import logging
logger = logging.getLogger(__file__)


class CustomToolParser:
    def __init__(self) -> None:
        self.tool_call_start_token: str = "<google_search>"
        self.tool_call_end_token: str = "</google_search>"
    
    def parse_tool_call(self, text: str):
        """
            解析工具调用字符串。
            
            输入示例: "<google_search>current weather in Tokyo</google_search>"
            输出示例: {
                "name": "google_search",
                "arguments": {"query_list": ["current weather in Tokyo"]}
            }
            
            如果格式不匹配，返回 None。
        """
        if not isinstance(text, str):
            return None

        text = text.strip()
        if not text:
            return None
        # 使用正则表达式精确匹配
        pattern = r'<google_search>\s*(.*?)\s*</google_search>'
        match = re.fullmatch(pattern, text, flags=re.DOTALL)
        
        if not match:
            return None
        
        query = match.group(1)  # 提取中间的 query 内容
        return {
            "name": "google_search",
            "arguments": {
                "query_list": [query]
            }
        }
    
    def extract_tool_calls(self, text: str):
        if self.tool_call_start_token not in text or self.tool_call_end_token not in text:
            return text, []
        
        # 查找第一个 <google_search> 的起始位置
        first_match = re.search(r'<google_search>', text)
        
        if first_match is None:
            # 没有找到任何 <google_search>，整个 text 都是 content，tool_calls 为空
            return text, []
        
        split_index = first_match.start()
        content = text[:split_index]
        
        # 从第一个 <google_search> 开始的位置截取剩余部分
        remaining = text[split_index:]
        
        # 在 remaining 中找出所有 <google_search>...</google_search> 匹配项
        # 使用非贪婪匹配，支持跨行（re.DOTALL）
        # matches = re.findall(r'<google_search>.*?</google_search>', remaining, flags=re.DOTALL)
        pattern = r'<google_search>\s*.*?\s*</google_search>'
        matches = re.findall(pattern, remaining, flags=re.DOTALL)
        
        tool_calls = []
        for match_str in matches:
            # print(match_str)
            parsed = self.parse_tool_call(match_str)
            if parsed is not None:
                tool_calls.append(parsed)
        
        return content.strip(), tool_calls


if __name__ == "__main__":
    text = """I need to look up two things.
<google_search>Peder Severin Krøyer artistic style\n</google_search>\n
<google_search>capital of Canada\n</google_search>"""
    parser = CustomToolParser()
    content, tool_calls = parser.extract_tool_calls(text)
    print(content)
    print(tool_calls)