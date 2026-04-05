import os
import re
import json
import logging
logger = logging.getLogger(__file__)


class CustomToolParser:
    def __init__(self) -> None:
        self.tool_call_start_token: str = "<tool_call>"
        self.tool_call_end_token: str = "</tool_call>"
    
    def parse_tool_call(self, text: str):
        """
            解析工具调用字符串。
            
            输入示例: "<tool_call>{\"name\":\"google_search\",\"arguments\":{\"query_list\":[\"current weather in Tokyo\"]}}</tool_call>"
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
        # 优先解析标准 JSON tool call：<tool_call>{...}</tool_call>
        pattern = r'<tool_call>\s*(\{.*?\})\s*</tool_call>'
        match = re.fullmatch(pattern, text, flags=re.DOTALL)

        if match:
            try:
                payload = json.loads(match.group(1))
                if not isinstance(payload, dict):
                    return None
                if payload.get("name") != "google_search":
                    return None
                arguments = payload.get("arguments", {})
                if not isinstance(arguments, dict):
                    return None
                query_list = arguments.get("query_list", [])
                if not isinstance(query_list, list):
                    return None
                cleaned_queries = [q.strip() for q in query_list if isinstance(q, str) and q.strip()]
                if not cleaned_queries:
                    return None
                return {
                    "name": "google_search",
                    "arguments": {
                        "query_list": cleaned_queries
                    }
                }
            except json.JSONDecodeError:
                return None

        # 兼容旧格式，避免历史模型输出导致崩溃
        legacy_pattern = r'<google_search>\s*(.*?)\s*</google_search>'
        legacy_match = re.fullmatch(legacy_pattern, text, flags=re.DOTALL)
        if not legacy_match:
            return None
        query = legacy_match.group(1).strip()
        if not query:
            return None
        return {
            "name": "google_search",
            "arguments": {
                "query_list": [query]
            }
        }
    
    def extract_tool_calls(self, text: str):
        if not isinstance(text, str) or not text:
            return "", []

        has_json_tool_call = self.tool_call_start_token in text and self.tool_call_end_token in text
        has_legacy_tool_call = "<google_search>" in text and "</google_search>" in text
        if not has_json_tool_call and not has_legacy_tool_call:
            return text, []

        # 查找首个工具调用起点（新格式优先）
        first_match = re.search(r'<tool_call>|<google_search>', text)
        
        if first_match is None:
            # 没有找到任何工具调用，整个 text 都是 content
            return text, []
        
        split_index = first_match.start()
        content = text[:split_index]
        
        # 从第一个工具调用开始的位置截取剩余部分
        remaining = text[split_index:]

        # 同时支持 <tool_call>...</tool_call> 与旧格式 <google_search>...</google_search>
        pattern = r'<tool_call>\s*.*?\s*</tool_call>|<google_search>\s*.*?\s*</google_search>'
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
<tool_call>{"name":"google_search","arguments":{"query_list":["Peder Severin Krøyer artistic style"]}}</tool_call>
<tool_call>{"name":"google_search","arguments":{"query_list":["capital of Canada"]}}</tool_call>"""
    parser = CustomToolParser()
    content, tool_calls = parser.extract_tool_calls(text)
    print(content)
    print(tool_calls)