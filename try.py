from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-1.5B")
print("eos_token:", tokenizer.eos_token, "id:", tokenizer.eos_token_id)
im_end_id = tokenizer.convert_tokens_to_ids("<|im_end|>")
print("<|im_end|> id:", im_end_id)
print("Same token?", tokenizer.eos_token_id == im_end_id)