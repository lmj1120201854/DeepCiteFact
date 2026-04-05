set -x


# model: sft模型 rl模型
# dataset: test_data.jsonl

# 需要评测的模型，部署完，填好
base_url=127.0.0.1:8001
model_name=Qwen3-8B
dataset=test_data.jsonl

# check模型，部署完填好，如果没资源可以先只部署评测模型，然后把除get_response之外的其他都注释，先走完模型抓取，再走后续流程
check_base_url=127.0.0.1:8000
check_model_name=Qwen2.5-32B-Instruct


python3 get_response.py \
  --dataset $dataset \
  --base_url $base_url \
  --model_name $model_name \
  --concurrent 256

report_path="./report/${dataset}.${model_name}.cite.eval.txt"

python3 citation_eval.py \
  --dataset $dataset \
  --model_name $model_name \
  --concurrent 128 \
  --check_base_url $check_base_url \
  --check_model_name $check_model_name > $report_path


report_path="./report/${dataset}.${model_name}.fact.eval.txt"

python3 fact_eval.py \
  --dataset $dataset \
  --model_name $model_name \
  --concurrent 128 \
  --check_base_url $check_base_url \
  --check_model_name $check_model_name > $report_path