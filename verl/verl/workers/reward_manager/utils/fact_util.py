import concurrent.futures
import logging
import math
import os
import re
import time

import numpy as np
import openai
from tqdm import tqdm

from verl.workers.reward_manager.utils.prompts import claim_request_prompt_template, claim_check_prompt_template

logger = logging.getLogger(__name__)

CHECK_SERVER = os.environ.get("CHECK_SERVER")
CHECK_SERVER_PATH = os.environ.get("CHECK_SERVER_PATH")
CHECK_SERVER_API_KEY = os.environ.get("CHECK_SERVER_API_KEY")
CHECK_SERVER_TIMEOUT = float(os.environ.get("CHECK_SERVER_TIMEOUT", "60"))
CHECK_SERVER_DISABLE_THINKING = os.environ.get("CHECK_SERVER_DISABLE_THINKING", "1") == "1"

CLAIM_SERVER = os.environ.get("CLAIM_SERVER")
CLAIM_SERVER_PATH = os.environ.get("CLAIM_SERVER_PATH")
CLAIM_SERVER_API_KEY = os.environ.get("CLAIM_SERVER_API_KEY")
CLAIM_SERVER_TIMEOUT = float(os.environ.get("CLAIM_SERVER_TIMEOUT", "60"))
CLAIM_SERVER_DISABLE_THINKING = os.environ.get("CLAIM_SERVER_DISABLE_THINKING", "1") == "1"

FACT_REWARD_MAX_WORKERS = int(os.environ.get("FACT_REWARD_MAX_WORKERS", "32"))
FACT_CHECK_MAX_WORKERS = int(os.environ.get("FACT_CHECK_MAX_WORKERS", "8"))

if not CHECK_SERVER or not CHECK_SERVER_PATH:
    print("error: CheckServer not found")
    exit(-1)

if not CLAIM_SERVER or not CLAIM_SERVER_PATH:
    print("error: ClaimServer not found")
    exit(-1)


ANSWER_PATTERN = re.compile(r'<answer>(.*?)</answer>', re.DOTALL | re.IGNORECASE)
BULLET_PATTERN = re.compile(r'^\s*(?:[-*]|\d+[.)])\s+(.*)$')
TRUE_FALSE_PATTERN = re.compile(r'\b(true|false)\b', re.IGNORECASE)



def extract_solution(solution_str):
    matches = list(ANSWER_PATTERN.finditer(solution_str or ""))
    if not matches:
        return ""
    return matches[-1].group(1).strip()



def _iter_api_keys(primary_key):
    seen = set()
    for key in [primary_key, "EMPTY", "None"]:
        if key and key not in seen:
            seen.add(key)
            yield key



def _request_chat(base_url: str, model: str, api_key: str, timeout: float, disable_thinking: bool, prompt: str, max_tokens: int):
    last_error = None
    extra_bodies = []
    if disable_thinking:
        extra_bodies.append({"chat_template_kwargs": {"enable_thinking": False}})
    extra_bodies.append(None)

    for key in _iter_api_keys(api_key):
        client = openai.Client(base_url=base_url, api_key=key, timeout=timeout)
        for extra_body in extra_bodies:
            for _ in range(5):
                try:
                    kwargs = {
                        "model": model,
                        "messages": [{"role": "user", "content": prompt}],
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

    logger.warning("Fact verifier request failed after retries: %s", last_error)
    return ""



def request_claims(prompt):
    return _request_chat(
        base_url=f"http://{CLAIM_SERVER}/v1",
        model=CLAIM_SERVER_PATH,
        api_key=CLAIM_SERVER_API_KEY,
        timeout=CLAIM_SERVER_TIMEOUT,
        disable_thinking=CLAIM_SERVER_DISABLE_THINKING,
        prompt=prompt,
        max_tokens=1024,
    )



def parse_claims(response):
    normalized = (response or "").strip()
    if not normalized or "no verifiable objective claims" in normalized.lower():
        return []

    claims = []
    for line in normalized.splitlines():
        match = BULLET_PATTERN.match(line)
        if match:
            claim = match.group(1).strip()
            if claim:
                claims.append(claim)

    if not claims and normalized.startswith("["):
        try:
            parsed = eval(normalized, {"__builtins__": {}}, {})
            if isinstance(parsed, list):
                claims = [str(item).strip() for item in parsed if str(item).strip()]
        except Exception:
            pass

    unique_claims = []
    seen = set()
    for claim in claims:
        if claim not in seen:
            seen.add(claim)
            unique_claims.append(claim)
    return unique_claims



def normalize_true_false(response: str) -> str:
    normalized = (response or "").strip()
    if not normalized:
        return ""
    match = TRUE_FALSE_PATTERN.search(normalized)
    if not match:
        return ""
    return match.group(1).lower()



def request_check(claim):
    prompt = claim_check_prompt_template.format(claim=claim)
    result = _request_chat(
        base_url=f"http://{CHECK_SERVER}/v1",
        model=CHECK_SERVER_PATH,
        api_key=CHECK_SERVER_API_KEY,
        timeout=CHECK_SERVER_TIMEOUT,
        disable_thinking=CHECK_SERVER_DISABLE_THINKING,
        prompt=prompt,
        max_tokens=8,
    )
    normalized = normalize_true_false(result)
    if normalized == "true":
        return 1.0
    if normalized == "false":
        return 0.0
    return 0.0



def compute_single_reward(response):
    prompt = claim_request_prompt_template.format(response=response)
    claim_response = request_claims(prompt)
    claims = parse_claims(claim_response)

    if not claims:
        return 1.0

    max_workers = max(1, min(len(claims), FACT_CHECK_MAX_WORKERS))
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        scores = list(executor.map(request_check, claims))

    return float(np.mean(scores)) if scores else 1.0



def compute_fact_rewards(responses):
    responses = [extract_solution(r) for r in responses]
    if not responses:
        return []

    max_workers = max(1, min(len(responses), FACT_REWARD_MAX_WORKERS))
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        print(f"Start calculating fact rewards, number of tasks {len(responses)}, number of concurrent workers {max_workers}")
        rewards = list(tqdm(executor.map(compute_single_reward, responses), total=len(responses)))

    return rewards
