import re

def calculate_search_reward(text, max_searches=5):
    """
    计算搜索奖励：有效搜索次数 / max_searches
    有效定义：必须成对出现，且 tool_response 中包含 </snippet>
    """
    try:
        if not text or not isinstance(text, str):
            return 0.0, 0.0

        # 1. 提取所有 google_search 到 tool_response 的配对块
        # pattern = r'<google_search>.*?</google_search>.*?<tool_response>(.*?)</tool_response>'
        pattern = r'<google_search>.*?</google_search>\s*user\s*<tool_response>(.*?)</tool_response>'
        responses = re.findall(pattern, text, re.DOTALL)
        if not responses:
            return 0.0, 0.0
        
        # 2. 统计包含有效内容（</snippets>）的响应数量
        valid_count = sum(1 for resp in responses if "</snippet>" in resp)
        
        # 3. 计算奖励并限制上限
        reward = valid_count / max_searches
        return min(reward, 1.0), valid_count

    except Exception:
        # 极端异常情况下返回 0，确保训练流程不中断
        return 0.0, 0.0