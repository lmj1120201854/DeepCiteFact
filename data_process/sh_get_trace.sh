set -x

base_url=127.0.0.1:8000
model_name=Qwen2.5-72B-Instruct

python3 get_trace.py \
  --base_url $base_url \
  --model_name $model_name \
  --concurrent 128 \
  --top_k 5