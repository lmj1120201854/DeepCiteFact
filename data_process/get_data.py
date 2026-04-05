import json
import pandas as pd
import os, re
import concurrent.futures
from citation_util import calculate_citation_reward
from tqdm import tqdm
from prompts import system_prompt, user_prompt
import glob
from collections import defaultdict
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import matplotlib.pyplot as plt
import numpy as np

# conver parquet
def convert(input_path, output_path):
    json_datas = []
    with open(input_path, 'r') as f:
        for line in f:
            data = json.loads(line)
            json_datas.append(data)
    # convert
    df = pd.DataFrame(json_datas)
    table = pa.Table.from_pandas(df)
    pq.write_table(table, output_path)


def parquet_to_jsonl(parquet_path, jsonl_path):
    """
    将 Parquet 文件转换为 JSONL 文件。
    适用于小数据集（能一次性载入内存）。
    """
    # 1. 读取 Parquet 为 DataFrame
    df = pd.read_parquet(parquet_path, engine='pyarrow')
    
    # 2. 转换为 JSONL (orient='records' 表示每行一个对象，lines=True 表示每行独立)
    df.to_json(jsonl_path, orient='records', lines=True, force_ascii=False)
    
    print(f"转换完成: {jsonl_path}")


def merge_query():
    add_query_set = set()
    all_datas = []
    for file_name in ["sft_rl_7b.jsonl", "sft_rl_14b.jsonl", "sft_rl_32b.jsonl"]:
        with open(f"../data/{file_name}", 'r') as f:
            for line in f:
                data = json.loads(line)
                if data['query'] not in add_query_set:
                    all_datas.append(data)
                    add_query_set.add(data['query'])
    
    with open("../data/all_data.jsonl", 'w') as f:
        for data in all_datas:
            new_data = {"query": data["query"]}
            f.write(json.dumps(new_data, ensure_ascii=False) + '\n')


def split_data():
    with open("../data/all_data.jsonl", 'r') as fin:
        all_data = [json.loads(line) for line in fin]
    
    print(len(all_data))  # 14358
    import random
    random.seed(42)
    random.shuffle(all_data)
    # 1000测试，3000sft（多留数据，最终会筛选一部分），6000rl
    rl_data = all_data[:5000]
    test_data = all_data[5000:6000]
    sft_data = all_data[6000:]
    with open("../data/rl_data.jsonl", 'w') as f:
        for data in rl_data:
            f.write(json.dumps(data, ensure_ascii=False) + '\n')
    with open("../data/test_data.jsonl", 'w') as f:
        for data in test_data:
            f.write(json.dumps(data, ensure_ascii=False) + '\n')
    with open("../data/sft_data.jsonl", 'w') as f:
        for data in sft_data:
            f.write(json.dumps(data, ensure_ascii=False) + '\n')


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


def get_reward(data):
    response = data["full_response"]
    format_reward = calculate_format_reward(response)
    # format
    citation_format, citation_precision, citation_recall, citation_f1, citation_reward = calculate_citation_reward(response)
    return format_reward, citation_format, citation_precision, citation_recall, citation_f1, citation_reward


def filter_sft_data():
    with open("../data/sft_trace.jsonl", 'r') as fin:
        datas = [json.loads(line) for line in fin]
    
    # 过滤准则
    # 1. format必须满足
    # 2. 提取cite，不能有幻觉
    # datas = datas[:100]
    with open("../data/sft_trace_filter.jsonl", 'w') as fout:
        futures = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=64) as executor:
            for data in datas:
                futures.append(executor.submit(get_reward, data))
        
            results = [future.result() for future in tqdm(futures)]
            for data, result in zip(datas, results):
                format_reward, citation_format, citation_precision, citation_recall, citation_f1, citation_reward = result
                data["format_reward"] = format_reward
                data["citation_format"] = citation_format
                data["citation_precision"] = citation_precision
                data["citation_recall"] = citation_recall
                data["citation_f1"] = citation_f1
                data["citation_reward"] = citation_reward
                if format_reward == 1.0 and citation_format == 1.0 and citation_precision >= 0.5:
                    fout.write(json.dumps(data, ensure_ascii=False) + '\n')
    

def convert_to_llama_factory():
    save_data = []
    with open("../data/sft_trace_filter.jsonl", 'r') as fin:
        for line in fin:
            data = json.loads(line)
            conversations = [
                {"from": "system", "value": system_prompt},
                {"from": "human", "value": user_prompt.format(query=data["query"])},
                {"from": "gpt", "value": data["full_response"]}
            ]
            save_item = {"conversations": conversations}
            save_data.append(save_item)
    
    with open("../LlamaFactory/data/sft_data.json", "w", encoding="utf-8") as f:
        json.dump(save_data, f, indent=4, ensure_ascii=False)
    
    # save json

def form_rl_data():
    with open("../data/rl_data.jsonl", 'r') as fin:
        with open("../data/rl_train_data.jsonl", 'w') as fout:
            for line in fin:
                data = json.loads(line)
                train_item = {
                    "agent_name": "tool_agent",
                    "query": data["query"],
                    "prompt": [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt.format(query=data["query"])}
                    ]
                }
                fout.write(json.dumps(train_item, ensure_ascii=False) + '\n')
    
    convert("../data/rl_train_data.jsonl", "../data/rl_train_data.parquet")


def get_std_gdpo(q_metrics):
    eps = 1e-6
    f_score = np.mean(q_metrics["fact"])
    f_std = np.std(q_metrics["fact"])
    c_score = np.mean(q_metrics["cite"])
    c_std = np.std(q_metrics["cite"])
    fmt_score = np.mean(q_metrics["format"])
    fmt_std = np.std(q_metrics["format"])
    s_score = np.mean(q_metrics["search"])
    s_std = np.std(q_metrics["search"])
    gdpo_adv = []
    for i in range(len(q_metrics["fact"])):
        fact_train_adv = (q_metrics["fact"][i] - f_score) / (f_std + eps)
        citation_train_adv = (q_metrics["cite"][i] - c_score) / (c_std + eps)
        format_train_adv = (q_metrics["format"][i] - fmt_score) / (fmt_std + eps)
        search_train_adv = (q_metrics["search"][i] - s_score) / (s_std + eps)
        adv = 2.0 * fact_train_adv + 1.0 * citation_train_adv + 0.5 * format_train_adv + 0.5 * search_train_adv
        gdpo_adv.append(adv)

    std = np.std(gdpo_adv)
    return std


def get_std_grpo(q_metrics):
    f_score = np.mean(q_metrics["fact"])
    f_std = np.std(q_metrics["fact"])
    c_score = np.mean(q_metrics["cite"])
    c_std = np.std(q_metrics["cite"])
    fmt_score = np.mean(q_metrics["format"])
    fmt_std = np.std(q_metrics["format"])
    s_score = np.mean(q_metrics["search"])
    s_std = np.std(q_metrics["search"])
    rewards = []
    for i in range(len(q_metrics["fact"])):
        reward = 0.5 * q_metrics["fact"][i] + 0.35 * q_metrics["cite"][i] + 0.05 * q_metrics["format"][i] + 0.1 * q_metrics["search"][i]
        rewards.append(reward)

    std = np.std(rewards)
    return std


def filter_simple_queries(data_dirs, original_file, output_file):
    """
    基于多维 Reward 过滤简单样本
    """
    # 存储每个 query 对应的各项 reward 列表
    stats = defaultdict(lambda: {"fact": [], "cite": [], "format": [], "search": []})
    rollout_files = []
    for data_dir in data_dirs:
        rollout_files_temp = glob.glob(os.path.join(data_dir, "rollout_data_step_*.jsonl"))
        rollout_files.extend(rollout_files_temp)
    
    print(f"Found {len(rollout_files)} rollout files. Analyzing...")
    
    for fpath in rollout_files:
        with open(fpath, 'r', encoding='utf-8') as f:
            for line in f:
                if not line.strip(): continue
                item = json.loads(line)
                q = item["query"]
                stats[q]["fact"].append(item.get("fact_reward"))
                stats[q]["cite"].append(item.get("citation_reward")[-1])
                stats[q]["format"].append(item.get("format_reward"))
                stats[q]["search"].append(item.get("search_reward"))
    
    # 画图
    plot_figure(stats)

    # 定义简单样本判定逻辑
    def is_simple(q_metrics):
        f_score = np.mean(q_metrics["fact"])
        f_std = np.std(q_metrics["fact"])
        c_score = np.mean(q_metrics["cite"])
        c_std = np.std(q_metrics["cite"])
        fmt_score = np.mean(q_metrics["format"])
        fmt_std = np.std(q_metrics["format"])
        s_score = np.mean(q_metrics["search"])
        s_std = np.std(q_metrics["search"])
        # 总的std
        std = get_std_grpo(q_metrics)
        # 1. fact如果方差太小肯定要过滤，无法学习
        if f_std < 0.1:
            return True  # std大一点有利于学习
        if c_std < 0.1:
            return True
        if f_score < 0.1 or f_score > 0.9:
            return True
        if c_score < 0.05 or c_score > 0.8:
            return True

        return False
    
    print("rollout query:", len(stats))
    simple_queries = {q for q, metrics in stats.items() if is_simple(metrics)}
    print(f"Identified {len(simple_queries)} truly simple queries.")

    # 过滤原始数据集
    count = 0
    with open(original_file, 'r', encoding='utf-8') as f_in, \
         open(output_file, 'w', encoding='utf-8') as f_out:
        for line in f_in:
            data = json.loads(line)
            # 如果 query 不在简单集合里，或者是全新的 query，则保留
            if data.get("query") not in simple_queries:
                f_out.write(json.dumps(data, ensure_ascii=False) + "\n")
            else:
                count += 1
    
    print(f"Filter complete. Removed {count} simple samples. Saved to {output_file}")
    # 转为parquet文件
    convert(output_file, output_file.replace(".jsonl", ".parquet"))



if __name__ == '__main__':
    # merge_query()
    # split_data()
    # form_rl_data()
    convert_to_llama_factory()
    # filter_simple_queries(["../verl/rollout_data_n4", "../verl/rollout_data_n8", "../verl/rollout_data_n8_1", "../verl/rollout_data_n8_2"], "../data/rl_train_data.jsonl", "../data/rl_train_data_filter_grpo.jsonl")