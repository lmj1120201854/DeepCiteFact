import re
import json
import time
import openai
import concurrent.futures
import random
import os
from tqdm import tqdm
from prompts import citation_extraction_template, citation_judge_template


def request_model(query):
    CHECK_SERVER = "127.0.0.1:8000"
    CHECK_SERVER_PATH = "Qwen2.5-32B-Instruct"
    client = openai.Client(base_url=f"http://{CHECK_SERVER}/v1", api_key="EMPTY")
    for _ in range(5):
        try:
            response = client.chat.completions.create(
                model=CHECK_SERVER_PATH,
                messages=[
                    {"role": "user", "content": query},
                ],
                temperature=0.1,
                max_tokens=8192
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            time.sleep(1)
            continue
    
    return ""


# ---------- extract tool_output blocks ----------
def extract_tool_blocks(text):
    return re.findall(r"<tool_response>(.*?)</tool_response>", text, flags=re.S)


def extract_url_content(text):
    """从tool_response中提取snippet的url和content
    snippet格式:
    <snippet id="...">
    Title: ...
    URL: ...
    Text: ...
    </snippet>
    """
    final = {}

    # 提取所有tool_response块
    tool_blocks = extract_tool_blocks(text)

    for block in tool_blocks:
        # 提取所有snippet块
        snippets = re.findall(r'<snippet[^>]*>(.*?)</snippet>', block, flags=re.S)

        for snippet in snippets:
            # 提取URL
            url_match = re.search(r'URL:\s*(.+)', snippet)
            # 提取Text内容
            text_match = re.search(r'Text:\s*(.+)', snippet, flags=re.S)

            if url_match and text_match:
                url = url_match.group(1).strip()
                content = text_match.group(1).strip()
                final.setdefault(url, content)

    return final


def calculate_format_reward(response, url_to_content):
    # 模式1: [index](url)
    PATTERN_INDEX = r'\[([1-9]\d*)\]\((.*?)\)'
    # 模式2: [text](url)
    PATTERN_TEXT = r'\[(?!\d+\])([^\[\]]+)\]\((.*?)\)'

    # 去掉工具调用结果来提取引用
    pattern = r'<tool_response>.*?</tool_response>'
    text = re.sub(pattern, '', response, flags=re.DOTALL)

    index_links = re.findall(PATTERN_INDEX, text)
    text_links = re.findall(PATTERN_TEXT, text)  # 目标citation格式

    # print("text_links:", text_links)
    # print("index_links:", index_links)

    # 1. text 引用质量（基础分）
    if len(text_links) == 0:
        if url_to_content:
            base_reward = 0.0   # 搜了但没引用，无影响
        else:
            base_reward = 0.0   # 没搜没引用，无影响
    else:
        if not url_to_content:
            base_reward = -0.5  # 没搜却引用了，惩罚0.5
        else:
            cnt_correct = sum(1 for _, url in text_links if url in url_to_content)
            base_reward = cnt_correct / len(text_links)  # 分母只看 text
    
    # 2. index 显式惩罚
    # 每个 index 引用扣固定分，封顶 -0.5
    index_penalty = min(len(index_links) * 0.1, 0.5)
    format_reward = base_reward - index_penalty
    return max(format_reward, -0.5)


# 计算Precision
def calculate_f1(response, url_to_content):
    pattern = r'<tool_response>.*?</tool_response>'
    text = re.sub(pattern, '', response, flags=re.DOTALL)
    prompt = citation_extraction_template.format(report_text=text)
    result = request_model(prompt)
    # json解析
    try:
        data = json.loads(result)
    except:
        data = []
    
    # print("data:", data)
    # 形成prompts
    prompts = []
    for item in data:
        try:
            claim = item.get("fact", "")
            url = item.get("url", "")
            if url not in url_to_content:
                continue
            document = url_to_content.get(url, "")  # 事实
            prompt = citation_judge_template.format(claim=claim, document=document)
            prompts.append(prompt)
        except:
            continue

    # 异步并发
    if prompts:
        # 并发发起所有请求
        futures = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=32) as executor:
            futures = [executor.submit(request_model, prompt) for prompt in prompts]
            results = [future.result() for future in futures]
    else:
        results = []
    
    total_score = 0
    total_cnt = 0
    right_claim_cnt = 0

    for result in results:
        if result == "Fully supported":
            total_score += 1
            right_claim_cnt += 1
        elif result == "Partially supported":
            total_score += 0.5
        else:
            total_score += 0

        total_cnt += 1

    if total_cnt == 0:
        # 不存在引用，需考虑两种情况
        if not url_to_content:
            precision = 0.1  # 不搜且没有引用，给一个基础分0.1
        else:
            precision = 0  # 搜索且没有引用
    else:
        if not url_to_content:
            precision = -0.5  # 不搜且存在引用，扣分，不鼓励自己编造url（内化）
        else:
            precision = total_score / total_cnt
    
    # 计算recall=正确claim的数量
    recall = right_claim_cnt
    # f1
    recall_bonus = min(right_claim_cnt * 0.1, 0.5)  # 每条正确引用+0.1，封顶0.5
    f1 = max(min(precision + recall_bonus, 1.0), 0.0)

    return precision, recall, f1


def calculate_citation_reward(response):
    url_to_content = extract_url_content(response)
    # print(url_to_content)
    citation_format = calculate_format_reward(response, url_to_content)
    citation_precision, citation_recall, citation_f1 = calculate_f1(response, url_to_content)

    citation_reward = 0.6 * citation_f1 + 0.4 * citation_format

    return citation_format, citation_precision, citation_recall, citation_f1, citation_reward


def calculate_citation_rewards(responses):
    max_workers = 512
    futures = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        print(f"Start calculating citation rewards, number of tasks {len(responses)}, number of concurrent workers {max_workers}")
        futures = [executor.submit(calculate_citation_reward, response) for response in responses]
        rewards = [future.result() for future in tqdm(futures)]
    
    return rewards