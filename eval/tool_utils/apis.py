import openai

def request_model(base_url, model_name, messages):
    client = openai.Client(base_url=f"http://{base_url}/v1", api_key="None")  # sglang为None，vllm为Empty
    for _ in range(5):
        try:
            response = client.chat.completions.create(
                model=model_name,
                messages=messages,
                temperature=0.7,
                top_p=1.0,
                max_tokens=8192,
                extra_body={
                    "stop": ["</google_search>"],
                    "include_stop_str_in_output": True
                }
            )
            #print("response:" + response.choices[0].message.content)
            if model_name == "Qwen3-8B":
                return response.choices[0].message.content.split("</think>")[-1].strip()
            else:
                return response.choices[0].message.content
        except:
            continue
    
    return ""