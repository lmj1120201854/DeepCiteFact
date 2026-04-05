import argparse
from pyarrow import field
from tqdm import tqdm
import re
import concurrent.futures
import json
import numpy as np
import os, sys

from tool_utils.apis import request_model
from prompts import user_prompt, system_prompt
from tool_utils.tool_agent_loop import ToolAgentLoop


if __name__ == "__main__":
    # print(res)
    parser = argparse.ArgumentParser(description="agent loop response")
    # 添加参数
    parser.add_argument("--top_k", type=int, default=5)
    parser.add_argument("--base_url", type=str, default="")
    parser.add_argument("--model_name", type=str, default="")
    parser.add_argument("--max_response_length", type=int, default=np.inf, help="") 
    parser.add_argument("--max_assistant_turns", type=int, default=10, help="")
    parser.add_argument("--unique_key", type=str, default="id", help="")
    parser.add_argument("--max_user_turns", type=int, default=10, help="")
    parser.add_argument("--concurrent", type=int, default=256, help="并发数")  # 可选参数

    args = parser.parse_args()

    agent_loop = ToolAgentLoop(args)
    with open("../data/sft_data.jsonl", "r") as fin:
        with open(f"../data/sft_trace.jsonl", "w") as fout:
            datas = []
            for line in fin:
                line = line.strip()
                data = json.loads(line)
                datas.append(data)
            
            # datas = datas[:3]
            with concurrent.futures.ThreadPoolExecutor(max_workers=args.concurrent) as executor:
                futures = []
                for data in datas:
                    futures.append(executor.submit(agent_loop.run, data))
                    
                pairs = [(data, future) for data, future in zip(datas, futures)]
                print(f"开始抓取response, 数量{len(datas)}, 并发数{args.concurrent}")
                for data, future in tqdm(pairs):
                    if args.model_name == "Qwen3-8B":
                        response = future.result()
                        data["response"] = response
                    else:
                        response, full_response = future.result()
                        data["response"] = response
                        data["full_response"] = full_response

                    fout.write(json.dumps(data, ensure_ascii=False) + "\n")