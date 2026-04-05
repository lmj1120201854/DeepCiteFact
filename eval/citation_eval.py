import re
import json
import time
import openai
import concurrent.futures
import random
import asyncio

import argparse
from pyarrow import field
from tqdm import tqdm
import re
import concurrent.futures
import json
import numpy as np


citation_extraction_template = """Your task is to analyze a research document and locate all inline citations that follow a specific markdown-style pattern.

The target citation format is: [descriptive text](hyperlink)
For instance: "Recent studies show that [global temperatures have risen by 1.2°C since pre-industrial times](https://www.climate.gov/news-features/understanding-climate/climate-change-global-temperature)."

Instructions for extraction:
- Scan the entire document and identify every citation matching the [text](url) pattern
- For each match, extract the factual claim being made along with its source URL
- Include sufficient surrounding context so the extracted fact is self-contained and verifiable
- When one statement references multiple sources (e.g., [claim A](url1) and [claim B](url2)), create separate entries for each
- Ignore any other reference styles such as numbered footnotes [1], parenthetical citations (Author, Year), or plain URLs
- Return an empty list if no valid [text](url) citations exist

Output format - a JSON array of objects:
[
    {{
        "fact": "The complete factual statement extracted from the document. Ensure proper escaping of quotes for JSON compatibility.",
        "url": "https://example.org/reference-page"
    }}
]

Document to analyze:
{report_text}

Respond with only the JSON array. Do not include any explanatory text or commentary."""


citation_judge_template = """You are a meticulous fact verification specialist. Your objective is to assess whether a given reference document substantiates a specific claim. Base your evaluation exclusively on the document content provided—do not incorporate prior knowledge or external information.

Rating criteria:

- "Fully supported": The document explicitly confirms all major aspects of the claim. Key facts, figures, and assertions are directly stated or unambiguously implied.
- "Partially supported": The document validates a substantial portion (over 50%) of the claim, but certain details remain unaddressed, unclear, or show minor discrepancies.
- "No support": The document fails to provide relevant evidence for the claim—either the content is unrelated, contradicts the assertion, or offers only superficial connections.

Inputs for evaluation:

Claim: {claim}

Document: {document}

Based strictly on the document above, determine the level of support. Output exactly one of these labels: Fully supported / Partially supported / No support"""


def request_model(args, query):
    client = openai.Client(base_url=f"http://{args.check_base_url}/v1", api_key="EMPTY")
    for _ in range(5):
        try:
            response = client.chat.completions.create(
                model=args.check_model_name,
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
def calculate_f1(args, response, url_to_content):
    pattern = r'<tool_response>.*?</tool_response>'
    text = re.sub(pattern, '', response, flags=re.DOTALL)
    prompt = citation_extraction_template.format(report_text=text)
    result = request_model(args, prompt)
    # json解析
    try:
        data = json.loads(result)
    except:
        data = []
    
    # print("data:", data)
    # reward = 支持数量/总的数量
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
            futures = [executor.submit(request_model, args, prompt) for prompt in prompts]
            results = [future.result() for future in futures]
    else:
        results = []
    
    total_score = 0
    total_cnt = 0
    right_claim_cnt = 0

    for result in results:
        if result == "Fully supported" or result == "Partially supported":
            total_score += 1
            right_claim_cnt += 1
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


def calculate_citation(args, response):
    url_to_content = extract_url_content(response)
    citation_precision, citation_recall, citation_f1 = calculate_f1(args, response, url_to_content)
    return citation_precision, citation_recall, citation_f1


def calculate_citation_metrics(args, responses):
    max_workers = 512
    futures = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        print(f"Start calculating citation rewards, number of tasks {len(responses)}, number of concurrent workers {max_workers}")
        futures = [executor.submit(calculate_citation, args, response) for response in responses]
        rewards = [future.result() for future in tqdm(futures)]
    
    citation_precisions = [item[0] for item in rewards]
    citation_recall = [item[1] for item in rewards]
    
    return np.mean(citation_precisions), np.mean(citation_recall)


if __name__ == "__main__":
    # print(res)
    parser = argparse.ArgumentParser(description="agent loop response")
    # 添加参数
    parser.add_argument("--top_k", type=int, default=5)
    parser.add_argument("--base_url", type=str, default="")
    parser.add_argument("--model_name", type=str, default="")
    parser.add_argument("--check_base_url", type=str, default="")
    parser.add_argument("--check_model_name", type=str, default="")
    parser.add_argument("--dataset", type=str, default="sft_train_data.jsonl", help="输入数据集, 需要在dataset目录下")
    
    parser.add_argument("--concurrent", type=int, default=256, help="并发数")  # 可选参数

    args = parser.parse_args()

    # 测评如下指标
    # 1. url命中率：提取所有citation的url，如果url处于搜索结果中则算命中，最后是命中数/总数
    # 2. url准确率: 提取所有citation的url和claim，判断支不支持，即计算所有reward的precision指标，然后算平均值
    responses = []
    hitrates = []
    with open(f"./output/{args.model_name}.{args.dataset}", "r") as fin:
        for line in fin:
            data = json.loads(line)
            url_to_content = extract_url_content(data["full_response"])
            responses.append(data["full_response"])
            # 提取url
            PATTERN_TEXT = r'\[(?!\d+\])([^\[\]]+)\]\((.*?)\)'
            # 去掉工具调用结果来提取引用
            pattern = r'<tool_response>.*?</tool_response>'
            text = re.sub(pattern, '', data["full_response"], flags=re.DOTALL)
            text_links = re.findall(PATTERN_TEXT, text)  # 目标citation格式
            if len(text_links) == 0:
                continue

            cnt = 0
            for _, url in text_links:
                if url in url_to_content:
                    cnt += 1
            
            hitrates.append(cnt / len(text_links))

    # 准确率
    precision, recall = calculate_citation_metrics(args, responses)
    
    print(f"{args.model_name} {args.dataset}：")
    print(f"引用命中率: {sum(hitrates)/len(hitrates)}")
    print(f"引用准确率: {precision}")
    print(f"平均正确引用个数: {recall}")
            


