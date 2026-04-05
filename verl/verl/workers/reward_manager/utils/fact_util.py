import openai
import os
import math
import concurrent.futures
import random
import time
import re
import requests
import json
import numpy as np
from collections import defaultdict
from tqdm import tqdm

from verl.workers.reward_manager.utils.prompts import claim_request_prompt_template, claim_check_prompt_template

# 加载server
CHECK_SERVER = os.environ.get('CHECK_SERVER')
CHECK_SERVER_PATH = os.environ.get('CHECK_SERVER_PATH')

CLAIM_SERVER = os.environ.get('CLAIM_SERVER')
CLAIM_SERVER_PATH = os.environ.get('CLAIM_SERVER_PATH')


if not CHECK_SERVER or not CHECK_SERVER_PATH:
    print("error: CheckServer not found")
    exit(-1)


if not CLAIM_SERVER or not CLAIM_SERVER_PATH:
    print("error: ClaimServer not found")
    exit(-1)
    

def extract_solution(solution_str):
    """Extract the equation from the solution string."""
    answer_pattern = r'<answer>(.*)</answer>'
    
    match = re.finditer(answer_pattern, solution_str, re.DOTALL)
    matches = list(match)

    if len(matches) <= 0:
        return ""

    # If there are 2 or more matches, return the last one
    return matches[-1].group(1).strip()


def request_claims(prompt):
    client = openai.Client(base_url=f"http://{CLAIM_SERVER}/v1", api_key="EMPTY")
    for _ in range(5):
        try:
            response = client.chat.completions.create(
                model=CLAIM_SERVER_PATH,
                messages=[
                    {"role": "user", "content": prompt},
                ],
                temperature=0.1,
                max_tokens=8192
            )
            return response.choices[0].message.content.strip()

        except:
            time.sleep(1)
            continue
            
    return ""


def parse_claims(response):
    if not response or "no verifiable objective claims" in response:
        return []
    try:
        claims = []
        for raw_line in response.strip().split('\n'):
            line = raw_line.strip()
            if line.startswith("* ") or line.startswith("- "):
                claims.append(line[2:].strip())
                continue
            # Support numbered outputs like "1. ..."
            m = re.match(r"^\d+\.\s+(.*)$", line)
            if m:
                claims.append(m.group(1).strip())
        return list(dict.fromkeys(claims))  # drop duplicate
    except Exception as e:
        print(f"error {e}")
        return []


def request_check(claim):
    prompt = claim_check_prompt_template.format(claim=claim)
    client = openai.Client(base_url=f"http://{CHECK_SERVER}/v1", api_key="EMPTY")
    for _ in range(5):
        try:
            response = client.chat.completions.create(
                model=CHECK_SERVER_PATH,
                messages=[
                    {"role": "user", "content": prompt},
                ],
                temperature=0.1,
                top_p=0.1,
                logprobs=True,
                max_tokens=2  # True or False
            )
            logprobs = response.choices[0].logprobs.content

            # Prefer token-level probability when available, but normalize token text first.
            if logprobs and len(logprobs) > 0:
                token_text = (logprobs[0].token or "").strip().lower()
                prob_true = math.exp(logprobs[0].logprob)
                if token_text.startswith("true"):
                    return prob_true
                if token_text.startswith("false"):
                    return 1.0 - prob_true

            # Fallback: parse text content robustly.
            content = (response.choices[0].message.content or "").strip().lower()
            if content.startswith("true"):
                return 1.0
            if content.startswith("false"):
                return 0.0
        except:
            time.sleep(1)
            continue
            
    return 0.0


def compute_single_reward(response):
    # 1. 使用Qwen2.5-32B-Instruct分解
    prompt = claim_request_prompt_template.format(response=response)
    claim_response = request_claims(prompt)
    claims = parse_claims(claim_response)
    # 自身验证
    max_workers = 32
    futures = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(request_check, claim) for claim in claims]
        scores = [future.result() for future in futures]
    
    if len(scores) == 0:
        return 1.0
    else:
        return np.mean(scores)
    

def compute_fact_rewards(responses):
    responses = [extract_solution(r) for r in responses]
    max_workers = 512
    futures = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        print(f"Start calculating fact rewards, number of tasks {len(responses)}, number of concurrent workers {max_workers}")
        futures = [executor.submit(compute_single_reward, response) for response in responses]
        rewards = [future.result() for future in tqdm(futures)]
    
    return rewards
    