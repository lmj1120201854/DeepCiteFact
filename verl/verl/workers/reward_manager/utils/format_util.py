import re

def calculate_format_reward(text):
    text = text.strip()
    # 1. 边界检查：必须以 <think> 开头，以 </answer> 结尾
    if not (text.startswith("<think>") and text.endswith("</answer>")):
        return 0.0

    # 2. 提取所有关键标记（包含开始标签、结束标签以及中间的 user）
    # 我们把 user 也当做一个“虚拟标签”来处理，方便检查顺序
    # 匹配 <think>, </think>, <google_search>, </google_search>, <tool_response>, </tool_response>, <answer>, </answer> 以及 user
    # token_pattern = r'<(?:/)?(?:think|google_search|tool_response|answer)>|\buser\b'
    token_pattern = r'<(?:/)?(?:think|google_search|tool_response|answer)>'
    tokens = re.findall(token_pattern, text)

    # 3. 必须包含至少一次搜索流程
    if '<google_search>' not in tokens:
        return 0.0

    # 4. 严苛的顺序流检查
    # 我们定义一个合法的搜索序列模式
    # 正确序列应该是：<google_search>, </google_search>, user, <tool_response>, </tool_response>
    
    for i in range(len(tokens)):
        t = tokens[i]
        
        # 这里宽松一点
        # 规则：google_search 结束标签后面必须紧跟 user，user 后面必须紧跟 tool_response 开始标签
        if t == '<think>':
            if i + 1 >= len(tokens):
                return 0.0
            if tokens[i+1] != '</think>':
                return 0.0
        
        if t == '<google_search>':
            if i + 1 >= len(tokens):
                return 0.0
            if tokens[i+1] != '</google_search>':
                return 0.0

        if t == '<tool_response>':
            if i + 1 >= len(tokens):
                return 0.0
            if tokens[i+1] != '</tool_response>':
                return 0.0
        
        # 规则：answer 必须是最后一个开始标签
        if t == '<answer>':
            if i != len(tokens) - 2: # 倒数第二个是 <answer>，最后一个必须是 </answer>
                return 0.0
            if tokens[i+1] != '</answer>':
                return 0.0

    # 5. 闭合性验证（确保没有未闭合或嵌套错误的标签）
    # 使用计数栈或简单的正则对检查
    for tag in ['think', 'google_search', 'tool_response', 'answer']:
        if text.count(f'<{tag}>') != text.count(f'</{tag}>'):
            return 0.0
    
    # 如果最后是不能回复，则返回0.5，这样鼓励更好回复
    if "Cannot determine an answer based on the available information" in text:
        return 0.5
            
    return 1.0