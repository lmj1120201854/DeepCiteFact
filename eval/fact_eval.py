import openai
import os
import math
import concurrent.futures
import random
import time
import re
import requests
import argparse
import json
import numpy as np
from collections import defaultdict
from tqdm import tqdm

claim_request_prompt_template = """\
Below you will receive a piece of text. Your task is:

1. Determine whether the text contains verifiable objective claims.
2. If verifiable objective claims exist in the text, you must extract these claims from the answer (regardless of whether these claims are true).
3. If the text does not contain any verifiable objective claims, return "no verifiable objective claims".

Response format:

* Claim 1
* Claim 2
...
(or "no verifiable objective claims")

The claims you extract must adhere to the following 3 principles:

1. Objectively verifiable: The claim must describe an objectively verifiable fact, not a subjective judgment, evaluation, or opinion.
2. Indivisible: The objective fact described by the claim cannot be further broken down.
3. Explicit meaning: Each claim must be a complete, self-contained sentence with all coreferences resolved. There should be no nouns or pronouns with unclear meaning.
 
Please strictly follow the above rules to complete the following task:
[Text]: {response}
[Verifiable objective claims]:""".strip()


claim_check_prompt_template="""You will determine if the following claim is true based on your knowledge. Answer only with "True" or "False".

Example:
Claim: The capital of France is London.
Answer: False

Example:
Claim: The capital of France is Paris.
Answer: True

Now, please evaluate the following:
Claim: {claim}
Answer:""".strip()


def request_claims(prompt):
    client = openai.Client(base_url=f"http://{args.check_base_url}/v1", api_key="EMPTY")
    for _ in range(5):
        try:
            response = client.chat.completions.create(
                model=args.check_model_name,
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
        claims = [line[2:] for line in response.strip().split('\n') if line.startswith('* ')]
        return list(dict.fromkeys(claims))  # drop duplicate
    except Exception as e:
        print(f"error {e}")
        return []


def request_check(claim):
    prompt = claim_check_prompt_template.format(claim=claim)
    client = openai.Client(base_url=f"http://{args.check_base_url}/v1", api_key="EMPTY")
    for _ in range(5):
        try:
            response = client.chat.completions.create(
                model=args.check_model_name,
                messages=[
                    {"role": "user", "content": prompt},
                ],
                temperature=0.1,
                top_p=0.1,
                logprobs=True,
                max_tokens=1  # True or False
            )
            logprobs = response.choices[0].logprobs.content
            if logprobs[0].token == "True":
                return math.exp(logprobs[0].logprob)
            if logprobs[0].token == "False":
                return 1.0 - math.exp(logprobs[0].logprob)
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
    max_workers = 512
    futures = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        print(f"Start calculating fact metrics, number of tasks {len(responses)}, number of concurrent workers {max_workers}")
        futures = [executor.submit(compute_single_reward, response) for response in responses]
        rewards = [future.result() for future in tqdm(futures)]
    
    return np.mean(rewards)


if __name__ == "__main__":
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

    responses = []
    with open(f"./output/{args.model_name}.{args.dataset}", "r") as fin:
        for line in fin:
            data = json.loads(line)
            responses.append(data["response"])
    
    fact_score = compute_fact_rewards(responses)
    
    print(f"{args.model_name} {args.dataset}：")
    print(f"fact指标: {fact_score}")
    