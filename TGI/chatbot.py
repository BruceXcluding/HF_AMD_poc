import gradio as gr
from huggingface_hub import InferenceClient

client = InferenceClient(model="http://127.0.0.1:8080")

def inference(message,history,max_new_tokens,temperature,top_p,top_k,repetition_penalty):
    partial_message = ""
    for token in client.text_generation(message,max_new_tokens=max_new_tokens,temperature=temperature,top_p=top_p,top_k=top_k,repetition_penalty=repetition_penalty,stream=True):
        partial_message += token
        yield partial_message

with gr.Blocks() as demo:

    chatbot=gr.Chatbot(height=300,render=False)
    max_new_tokens=gr.Slider(1,1024,label="Max new tokens", value=32,render=False)
    temperature=gr.Slider(0.1,4,label="temperature", value=0.6, step=0.1,render=False)
    top_p=gr.Slider(0.05,1,label="Top-p", value=0.9,render=False)
    top_k=gr.Slider(1,1000,label="Top-k", value=50,render=False)
    repetition_penalty=gr.Slider(1,2,label="Repetition Penalty", value=1,render=False)

    gr.ChatInterface(
            inference,
            chatbot=chatbot,
            additional_inputs=[max_new_tokens,temperature,top_p,top_k,repetition_penalty],
            title="Gradio Chatbot 🤝 TGI",
            description="Gradio UI consuming TGI endpoint with LLaMA 7B-Chat model",
            examples=[["Hi,how are you?"]],
            retry_btn="Retry",
            undo_btn="Undo",
            clear_btn="Clear"
            )
gr.close_all()
demo.queue().launch(share=True)
