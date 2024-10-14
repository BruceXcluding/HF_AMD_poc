import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, LlamaForCausalLM

tokenizer = AutoTokenizer.from_pretrained("TheBloke/Llama-2-7B-fp16")

model = AutoModelForCausalLM.from_pretrained(
        "TheBloke/Llama-2-7B-fp16",
        torch_dtype=torch.float16,
        use_flash_attention_2=True,
        device_map={"": 0},
)

input_text = 'I liked "Breaking Bad" and "Band of Brothers". Do you have any recommendations of other shows I might like?\n'
inputs = tokenizer(input_text,return_tensors="pt").to("cuda")

outputs = model.generate(
        **inputs,
        do_sample=True,
        top_k=10,
        eos_token_id=tokenizer.eos_token_id,
        max_length=200,
)
print(tokenizer.decode(outputs[0],skip_special_tokens=True))
