import concurrent.futures
import json
import logging
import os
import re
import time

import openai
from tqdm import tqdm

from verl.workers.reward_manager.utils.prompts import citation_extraction_template, citation_judge_template

logger = logging.getLogger(__name__)

CHECK_SERVER = os.environ.get("CHECK_SERVER")
CHECK_SERVER_PATH = os.environ.get("CHECK_SERVER_PATH")
CHECK_SERVER_API_KEY = os.environ.get("CHECK_SERVER_API_KEY")
CHECK_SERVER_TIMEOUT = float(os.environ.get("CHECK_SERVER_TIMEOUT", "60"))
CHECK_SERVER_DISABLE_THINKING = os.environ.get("CHECK_SERVER_DISABLE_THINKING", "1") == "1"
CITATION_REWARD_MAX_WORKERS = int(os.environ.get("CITATION_REWARD_MAX_WORKERS", "32"))
CITATION_JUDGE_MAX_WORKERS = int(os.environ.get("CITATION_JUDGE_MAX_WORKERS", "32"))

if not CHECK_SERVER or not CHECK_SERVER_PATH:
    print("error: CheckServer not found")
    exit(-1)


PATTERN_INDEX = r'\[([1-9]\d*)\]\((.*?)\)'
PATTERN_TEXT = r'\[(?!\d+\])([^\[\]]+)\]\((.*?)\)'
TOOL_RESPONSE_PATTERN = re.compile(r'<tool_response>.*?</tool_response>', re.DOTALL)
JSON_ARRAY_PATTERN = re.compile(r'\[.*\]', re.DOTALL)


def _iter_api_keys():
    seen = set()
    for key in [CHECK_SERVER_API_KEY, "EMPTY", "None"]:
        if key and key not in seen:
            seen.add(key)
            yield key



def request_model(query: str, max_tokens: int = 1024) -> str:
    last_error = None
    extra_bodies = []
    if CHECK_SERVER_DISABLE_THINKING:
        extra_bodies.append({"chat_template_kwargs": {"enable_thinking": False}})
    extra_bodies.append(None)

    for api_key in _iter_api_keys():
        client = openai.Client(
            base_url=f"http://{CHECK_SERVER}/v1",
            api_key=api_key,
            timeout=CHECK_SERVER_TIMEOUT,
        )
        for extra_body in extra_bodies:
            for _ in range(5):
                try:
                    kwargs = {
                        "model": CHECK_SERVER_PATH,
                        "messages": [{"role": "user", "content": query}],
                        "temperature": 0.0,
                        "top_p": 1.0,
                        "max_tokens": max_tokens,
                    }
                    if extra_body is not None:
                        kwargs["extra_body"] = extra_body
                    response = client.chat.completions.create(**kwargs)
                    return (response.choices[0].message.content or "").strip()
                except Exception as exc:
                    last_error = exc
                    time.sleep(1)

    logger.warning("Citation verifier request failed after retries: %s", last_error)
    return ""


# ---------- extract tool_output blocks ----------
def extract_tool_blocks(text):
    return re.findall(r"<tool_response>(.*?)</tool_response>", text, flags=re.S)



def extract_url_content(text):
    final = {}
    tool_blocks = extract_tool_blocks(text)

    for block in tool_blocks:
        snippets = re.findall(r'<snippet[^>]*>(.*?)</snippet>', block, flags=re.S)
        for snippet in snippets:
            url_match = re.search(r'URL:\s*(.+)', snippet)
            text_match = re.search(r'Text:\s*(.+)', snippet, flags=re.S)
            if url_match and text_match:
                url = url_match.group(1).strip()
                content = text_match.group(1).strip()
                final.setdefault(url, content)

    return final



def extract_json_array(text: str):
    normalized = (text or "").strip()
    if not normalized:
        return []

    for candidate in [normalized]:
        try:
            parsed = json.loads(candidate)
            return parsed if isinstance(parsed, list) else []
        except Exception:
            pass

    fenced_match = re.search(r"```(?:json)?\s*(\[.*?\])\s*```", normalized, flags=re.DOTALL | re.IGNORECASE)
    if fenced_match:
        try:
            parsed = json.loads(fenced_match.group(1))
            return parsed if isinstance(parsed, list) else []
        except Exception:
            pass

    array_match = JSON_ARRAY_PATTERN.search(normalized)
    if array_match:
        try:
            parsed = json.loads(array_match.group(0))
            return parsed if isinstance(parsed, list) else []
        except Exception:
            pass

    return []



def normalize_support_label(text: str) -> str:
    normalized = (text or "").strip().lower()
    if not normalized:
        return ""
    if "fully supported" in normalized:
        return "fully supported"
    if "partially supported" in normalized:
        return "partially supported"
    if "no support" in normalized:
        return "no support"
    return ""



def calculate_format_reward(response, url_to_content):
    text = re.sub(TOOL_RESPONSE_PATTERN, '', response)
    index_links = re.findall(PATTERN_INDEX, text)
    text_links = re.findall(PATTERN_TEXT, text)

    if len(text_links) == 0:
        base_reward = 0.0
    else:
        if not url_to_content:
            base_reward = -0.5
        else:
            cnt_correct = sum(1 for _, url in text_links if url in url_to_content)
            base_reward = cnt_correct / len(text_links)

    index_penalty = min(len(index_links) * 0.1, 0.5)
    format_reward = base_reward - index_penalty
    return max(format_reward, -0.5)



def calculate_f1(response, url_to_content):
    text = re.sub(TOOL_RESPONSE_PATTERN, '', response)
    prompt = citation_extraction_template.format(report_text=text)
    result = request_model(prompt, max_tokens=2048)
    data = extract_json_array(result)

    prompts = []
    for item in data:
        if not isinstance(item, dict):
            continue
        claim = str(item.get("fact", "")).strip()
        url = str(item.get("url", "")).strip()
        if not claim or url not in url_to_content:
            continue
        document = url_to_content.get(url, "")
        prompt = citation_judge_template.format(claim=claim, document=document)
        prompts.append(prompt)

    if prompts:
        max_workers = max(1, min(len(prompts), CITATION_JUDGE_MAX_WORKERS))
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            results = list(executor.map(request_model, prompts))
    else:
        results = []

    total_score = 0.0
    total_cnt = 0
    right_claim_cnt = 0

    for result in results:
        label = normalize_support_label(result)
        if label == "fully supported":
            total_score += 1.0
            right_claim_cnt += 1
        elif label == "partially supported":
            total_score += 0.5
        total_cnt += 1

    if total_cnt == 0:
        precision = 0.1 if not url_to_content else 0.0
    else:
        if not url_to_content:
            precision = -0.5
        else:
            precision = total_score / total_cnt

    recall = right_claim_cnt
    recall_bonus = min(right_claim_cnt * 0.1, 0.5)
    f1 = max(min(precision + recall_bonus, 1.0), 0.0)

    return precision, recall, f1



def calculate_citation_reward(response):
    url_to_content = extract_url_content(response)
    citation_format = calculate_format_reward(response, url_to_content)
    citation_precision, citation_recall, citation_f1 = calculate_f1(response, url_to_content)
    citation_reward = 0.6 * citation_f1 + 0.4 * citation_format
    return citation_format, citation_precision, citation_recall, citation_f1, citation_reward



def calculate_citation_rewards(responses):
    if not responses:
        return []

    max_workers = max(1, min(len(responses), CITATION_REWARD_MAX_WORKERS))
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        print(f"Start calculating citation rewards, number of tasks {len(responses)}, number of concurrent workers {max_workers}")
        rewards = list(tqdm(executor.map(calculate_citation_reward, responses), total=len(responses)))

    return rewards
