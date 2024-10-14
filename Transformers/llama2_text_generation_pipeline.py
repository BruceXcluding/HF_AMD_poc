import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

model_id = "TheBloke/Llama-2-7B-fp16"

tokenizer = AutoTokenizer.from_pretrained(model_id)

model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.float16,
    use_flash_attention_2=True,
    device_map={"": 0},
)


generate_text = pipeline(
    model=model,
    tokenizer=tokenizer,
    task='text-generation',
)

res = generate_text(
        'I liked "Breaking Bad" and "Band of Brothers". Do you have any recommendations of other shows I might like?\n',
        do_sample=True,
        top_k=10,
        num_return_sequences=1,
        eos_token_id=tokenizer.eos_token_id,
        max_length=200,
)

for seq in res:
    print(f"Result: {seq['generated_text']}")
