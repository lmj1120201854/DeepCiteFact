import uuid
import time
from concurrent.futures import ThreadPoolExecutor
from ddgs import DDGS
# from crawl4ai import *


def generate_snippet_id() -> str:
    """
    结合 UUID 的唯一性和 Base62 的字符集。
    生成格式如: 'S_BP4aUVA'
    """
    # 1. 定义字符集 (62个字符)
    alphabet = "0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"
    
    # 2. 生成一个随机的 UUID
    # uuid4().int 会得到一个非常大的整数
    u_int = uuid.uuid4().int
    
    # 3. 将这个大整数转换为 62 进制字符串
    # 我们只需要取其中一段，这里通过取模和连续除法获得
    res = []
    # 循环 8 次生成 8 位字符（根据你的示例 S_BP4aUVA，BP4aUVA是7位，你可以改循环次数）
    for _ in range(7):
        u_int, remainder = divmod(u_int, 62)
        res.append(alphabet[remainder])
    
    return "S_" + "".join(res)

    
def generate_search_snippets(results):
    """
    results: [{"query":, "", "title": "", "body": "", "href": ""} ... ]
    Return:
    <snippet id="S_BP4aUVA">
    Title: xxx
    URL: https://cs.bjut.edu.cn/info/1509/3619.htm
    Text: xxx
    </snippet>
    """
    if not isinstance(results, list) or len(results) == 0:
        return "<tool_response>\nGoogle search encountered an error and was unable to extract valid information.\n</tool_response>"
    
    result_text = ""
    for item in results:
        if not isinstance(item, dict):
            continue
        
        snippet_id = generate_snippet_id()
        start_str = "<snippet id=" + generate_snippet_id() + ">\n"
        end_str = "\n</snippet>"
        content = "Title: " + item.get("title", "") + "\n" + "URL: " + item.get("href", "") + "\n" + "Text: " + item.get("body", "")
        result_text += (start_str + content + end_str + "\n")
    
    return "<tool_response>\n" + result_text.strip() + "\n</tool_response>"

    
def ddgs_search(queries, top_k=5, ddgs_backend='auto'):
    def ddgs_single_search(query):
        for i in range(5):
            with DDGS(timeout=180) as _ddgs:
                try:
                    single_result = _ddgs.text(
                        query,
                        region="us-en",
                        safesearch="off",
                        max_results=10,
                        timelimit=None,
                        ddgs_backend=ddgs_backend
                    )
                    single_result = [{'query': query, **x} for x in single_result]
                    return single_result
                except Exception as e:
                    time.sleep(1)
                    continue
        return []

    results = []
    
    try:
        result = ddgs_single_search(queries[0])
        if result:
            return result[:top_k]
    except:
        pass
    
    return []