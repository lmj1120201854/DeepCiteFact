# Copyright 2023-2024 SGLang Team
# Copyright 2025 ModelBest Inc. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import pickle
from typing import Any, Iterator, Optional

import numpy as np
import torch
import json
import torch.distributed as dist

from verl.utils.device import get_device_name

import logging
from typing import Dict, List, Literal, Optional, Set, Tuple, Type, Union

from sglang.srt.entrypoints.openai.protocol import Tool

from sglang.srt.function_call.core_types import (
    StreamingParseResult,
    ToolCallItem,
    StructureInfo,
    _GetInfoFunc,
)

import re

logger = logging.getLogger(__name__)

def parse_call_tool(s: str):
    if not isinstance(s, str):
        return None

    s = s.strip()
    if not s:
        return None

    # 严格匹配：<call_tool ...>...</call_tool>
    # 注意：content 部分不能包含 '<'（因为会破坏 XML-like 结构）
    pattern = r'^<call_tool\s+name=["\']([^"\']+)["\']([^>]*)>\s*(.*?)\s*</call_tool>$'
    match = re.fullmatch(pattern, s, re.DOTALL)
    if not match:
        return None

    name = match.group(1)
    attrs_str = match.group(2)
    content = match.group(3)

    # 只接受指定的两种 name
    if name not in ("google_search", "browse_webpage"):
        return None

    # 安全解析属性：只允许 key="value" 或 key='value'，且 key 必须是字母/数字/下划线
    attr_pattern = r'(\w+)=[\"\']([^\"\']*)[\"\']'
    try:
        attrs = dict(re.findall(attr_pattern, attrs_str))
    except Exception:
        return None  # 属性解析异常（理论上不会，但防御性编程）

    # 检查 content 是否合法
    if not content:
        return None

    # 如果 content 中包含 '<'，说明可能嵌套或格式错误（按你的需求应视为非法）
    if '<' in content:
        return None

    content = content.strip()

    # 分支处理
    if name == "google_search":
        result = {
            "name": name,
            "parameters": {k: v for k, v in attrs.items() if k != "name"}
        }
        result["parameters"]["query_list"] = [content]
        return result

    elif name == "browse_webpage":
        # browse_webpage 不应有任何属性（即使有也忽略，但 content 必须像 URL）
        # 这里不做强 URL 校验（避免过度约束），但可选加简单判断
        # 例如：至少包含 :// 或 . 
        # 此处保持宽松，只要 content 非空即可
        result = {
            "name": name,
            "parameters": {"url": content}
        }
        return result

    return None


class DrTuluFunctionCallParser:
    def __init__(self, tools: List[Tool]):
        self.bot_token = "<call_tool"
        self.eot_token = "</call_tool>"
        self.tools = tools   # 从配置文件中预定义的tool列表
    
    def has_tool_call(self, text: str):
        if not self.tools:
            return False
        # 含有</call_tool>就说明有工具
        return self.eot_token in text
    
    def parse_base_json(self, parsed_call: dict):
        name = parsed_call.get("name", "")
        tool_indices =  {
            tool.function.name: i for i, tool in enumerate(self.tools) if tool.function.name
        }
        if not (name and name in tool_indices):
            logger.warning(f"Model attempted to call undefined function: {name}")
            return []
        
        tool_item = ToolCallItem(
            tool_index=-1,  # Caller should update this based on the actual tools array called
            name=name,
            parameters=json.dumps(
                parsed_call.get("parameters") or parsed_call.get("arguments", {}),
                ensure_ascii=False,
            )
        )
        return [tool_item]

    def parse_non_stream(self, text: str):
        if not self.tools:
            return text, []

        # 解析text返回normal_text和tool list
        idx = text.find(self.bot_token)
        normal_text = text[:idx].strip() if idx != -1 else text
        if self.bot_token not in text:
            return normal_text, []
        
        # Find all <call_tool>\n...\n</call_tool> blocks
        pattern = r'<call_tool\s+[^>]*>.*?</call_tool>'
        match_result_list = re.findall(pattern, text, re.DOTALL)
        calls = []
        for match_result in match_result_list:
            try:
                # 形如：
                # 1. <call_tool name="google_search" num="5" gl="cn" hl="zh-CN">北京工业大学 计算机学院 竹翠 简介</call_tool>
                # 2. <call_tool name="browse_webpage">https://cs.bjut.edu.cn/info/1509/3619.htm</call_tool>
                parsed_call = parse_call_tool(match_result)
                if isinstance(parsed_call, dict):
                    calls.extend(self.parse_base_json(parsed_call))

            except json.JSONDecodeError as e:
                logger.warning(
                    f"Failed to parse JSON part: {match_result}, JSON parse error: {str(e)}"
                )
                continue
        
        if calls:
            return normal_text, calls
        else:
            return text, calls


if __name__ == "__main__":
    # test_cases = [
    #     # 正常
    #     '<call_tool name="google_search">query</call_tool>',
    #     '<call_tool name="browse_webpage">http://example.com</call_tool>',
    #     '<call_tool name="google_search" gl="cn">q</call_tool>',

    #     # 异常
    #     None,                                      # 非字符串
    #     "",                                        # 空
    #     "   ",                                     # 空白
    #     '<call_tool name="xxx">q</call_tool>',     # 未知 name
    #     '<call_tool name="google_search">',        # 无闭合标签
    #     '<call_tool name="google_search">q',       # 无结束标签
    #     '<call_tool name="google_search">q<oops>', # content 含 <
    #     '<call_tool name="google_search" gl=cn>q</call_tool>',  # 属性无引号（会被忽略，但整体合法）
    #     '<call_tool name="browse_webpage"></call_tool>',        # url 为空 → None
    # ]

    # for case in test_cases:
    #     result = parse_call_tool(case)
    #     print(f"Input: {repr(case)} → Output: {result}")
    test_str = """
    thinksxsxsxsxsxsxsxs
    <call_tool name="google_search" num="5" gl="cn" hl="zh-CN">北京工业大学 计算机学院 竹翠 简介</call_tool>
    <call_tool name="browse_webpage">http://example.com</call_tool>
    """
    # parse = DrTuluFunctionCallParser(["xxx"])
    # normal_text, tools = parse.parse_non_stream(test_str)
    # print(normal_text)
    # print(tools)

