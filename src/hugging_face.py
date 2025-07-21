from huggingface_hub import hf_hub_download
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import RAG
import os

HUGGING_FACE_API_KEY = "------------------------"
huggingface_model = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

required_files = [
    "special_tokens_map.json",
    "generation_config.json",
    "tokenizer_config.json",
    "model.safetensors",
    "eval_results.json",
    "tokenizer.model",
    "tokenizer.json",
    "config.json",
]

for filename in required_files:
    download_loc = hf_hub_download(
        repo_id=huggingface_model,
        filename=filename,
        token="------------------------------"
    )
    print(f"Downloaded {filename} to {download_loc}")

model = AutoModelForCausalLM.from_pretrained(huggingface_model)
tokenizer = AutoTokenizer.from_pretrained(huggingface_model)

text_generation_pipeline  = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    max_length=1000
)

max_history_length = 20

for i in range(5):
    address = os.path.join('Example Data', f'comment_{i+1}.txt')
    if not os.path.isfile(address):
        print(f"File not found: {address}")
        continue
    paragraphs = RAG.parse(address)
    comment = "\n".join(paragraphs)
    instruction = "ONLY reply with a single word good or bad. Do not add any words or explanation. Rate the following comment:\n"
    input_text = instruction + comment
    inputs = tokenizer(input_text, return_tensors="pt", truncation=True, max_length=1024)
    
    try:
        outputs = model.generate(**inputs, max_new_tokens=10)
    except Exception as e:
        print(f"Error generating response: {e}")
        continue
    
    response = tokenizer.decode(outputs[0], skip_special_tokens=True).strip()
    print(f"Bot: {response}")