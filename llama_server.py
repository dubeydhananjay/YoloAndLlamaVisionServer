# llama_server.py
import torch
from PIL import Image
from transformers import MllamaForConditionalGeneration, AutoProcessor

# Load the LLaMA Vision-Instruct model and processor
llama_model_id = "meta-llama/Llama-3.2-11B-Vision-Instruct"
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

llama_model = MllamaForConditionalGeneration.from_pretrained(
    llama_model_id,
    torch_dtype=torch.bfloat16,
    device_map="auto",
).to(device)

llama_processor = AutoProcessor.from_pretrained(llama_model_id)

def generate_image_description(image, prompt):
    # Prepare input for LLaMA model
    messages = [
        {"role": "user", "content": [
            {"type": "image"},
            {"type": "text", "text": prompt}
        ]}
    ]
    input_text = llama_processor.apply_chat_template(messages, add_generation_prompt=True)
    inputs = llama_processor(image, input_text, add_special_tokens=False, return_tensors="pt").to(device)

    # Generate description
    with torch.no_grad():
        output = llama_model.generate(**inputs, max_new_tokens=200)
    description = llama_processor.decode(output[0], skip_special_tokens=True)
    return description
