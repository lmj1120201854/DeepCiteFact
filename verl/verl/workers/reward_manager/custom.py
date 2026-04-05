# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple

import torch
import json

from verl import DataProto
from verl.utils.reward_score import default_compute_score
from verl.workers.reward_manager import register
from verl.workers.reward_manager.abstract import AbstractRewardManager

# fact
# compute_fact_rewards(responses)
from verl.workers.reward_manager.utils.fact_util import compute_fact_rewards

# citation
# calculate_citation_rewards(responses)
from verl.workers.reward_manager.utils.citation_util import calculate_citation_rewards

# search
# calculate_search_reward(text, max_searches=5)
from verl.workers.reward_manager.utils.search_util import calculate_search_reward

# format
# calculate_format_reward(text)
from verl.workers.reward_manager.utils.format_util import calculate_format_reward


@register("custom")
class CustomRewardManager(AbstractRewardManager):
    """The reward manager."""

    def __init__(self, tokenizer, num_examine, compute_score=None, reward_fn_key="data_source") -> None:
        """
        Initialize the DrTuluRewardManager instance.

        Args:
            tokenizer: The tokenizer used to decode token IDs into text.
            num_examine: The number of batches of decoded responses to print to the console for debugging purpose.
            compute_score: A function to compute the reward score. If None, `default_compute_score` will be used.
            reward_fn_key: The key used to access the data source in the non-tensor batch data. Defaults to
                "data_source".
        """
        self.tokenizer = tokenizer  # Store the tokenizer for decoding token IDs
        self.num_examine = num_examine  # the number of batches of decoded responses to print to the console
        self.compute_score = compute_score or default_compute_score
        self.reward_fn_key = reward_fn_key  # Store the key for accessing the data source

    def __call__(self, data: DataProto, global_steps: int = 0, return_dict: bool = False):
        """We will expand this function gradually based on the available datasets"""

        # If there is rm score, we directly return rm score. Otherwise, we compute via rm_score_fn
        # if "rm_scores" in data.batch.keys():
        #     if return_dict:
        #         reward_extra_keys = data.meta_info.get("reward_extra_keys", [])
        #         reward_extra_info = {key: data.non_tensor_batch[key] for key in reward_extra_keys}
        #         return {"reward_tensor": data.batch["rm_scores"], "reward_extra_info": reward_extra_info}
        #     else:
        #         return data.batch["rm_scores"]

        reward_tensor = torch.zeros_like(data.batch["responses"], dtype=torch.float32)
        reward_extra_info = defaultdict(list)

        already_print_data_sources = {}

        format_rewards = []

        valid_response_lengths = []
        search_rewards = []
        search_nums = []
        acc_items = []
        json_datas = []

        responses = []

        for i in range(len(data)):
            data_item = data[i]  # DataProtoItem

            prompt_ids = data_item.batch["prompts"]

            prompt_length = prompt_ids.shape[-1]

            valid_prompt_length = data_item.batch["attention_mask"][:prompt_length].sum()
            valid_prompt_ids = prompt_ids[-valid_prompt_length:] # left pad

            response_ids = data_item.batch["responses"]
            valid_response_length = data_item.batch["attention_mask"][prompt_length:].sum()
            valid_response_lengths.append(valid_response_length)
            valid_response_ids = response_ids[:valid_response_length]

            # decode
            prompt_str = self.tokenizer.decode(valid_prompt_ids, skip_special_tokens=True)
            response_str = self.tokenizer.decode(valid_response_ids, skip_special_tokens=True)
            json_datas.append({prompt_str: response_str})

            # print("prompt:")
            # print(prompt_str)

            # print("response:")
            # print(response_str)
            responses.append(response_str)

            # 1. format reward
            format_reward = calculate_format_reward(response_str)
            format_rewards.append(format_reward)

            # 2. search reward
            search_reward, search_num = calculate_search_reward(response_str)
            search_rewards.append(search_reward)
            search_nums.append(search_num)
        

        # 3. fact reward
        fact_rewards = compute_fact_rewards(responses)
        # 4. citation reward
        citation_rewards = calculate_citation_rewards(responses)

        for i in range(len(data)):
            save_item = {
                "query": data[i].non_tensor_batch["query"],
                "response": responses[i],
                "fact_reward": fact_rewards[i],
                "citation_reward": citation_rewards[i],
                "search_reward": search_rewards[i],
                "format_reward": format_rewards[i],
                "search_num": search_nums[i]
            }
            json_datas.append(save_item)
            reward_tensor[i, valid_response_lengths[i] - 1] = 0.5 * fact_rewards[i] + 0.35 * citation_rewards[i][-1] + 0.05 * format_rewards[i] + 0.1 * search_rewards[i]
        
        # save rollout_data
        with open(f"./rollout_data/rollout_data_step_{global_steps}.jsonl", "w") as fout:
            for item in json_datas:
                fout.write(json.dumps(item, ensure_ascii=False) + "\n")

        return reward_tensor, fact_rewards, citation_rewards, search_rewards, search_nums, format_rewards