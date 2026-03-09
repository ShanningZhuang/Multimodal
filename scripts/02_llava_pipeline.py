"""
Lab 2: LLaVA Pipeline — Simple Forward Pass

Load llava-1.5-7b and run a single forward + generate so you can
step through the internals in the debugger.

Run: uv run python Multimodal/scripts/02_llava_pipeline.py
"""

import argparse

import torch
from PIL import Image
import numpy as np
from transformers import AutoProcessor, LlavaForConditionalGeneration

parser = argparse.ArgumentParser()
parser.add_argument("--image", type=str, default=None)
args = parser.parse_args()

model_id = "llava-hf/llava-1.5-7b-hf"
print(f"Loading {model_id} ...")

processor = AutoProcessor.from_pretrained(model_id)
model = LlavaForConditionalGeneration.from_pretrained(
    model_id, torch_dtype=torch.float16, device_map="auto",
)
model.eval()

# Prepare image
if args.image:
    image = Image.open(args.image).convert("RGB")
else:
    img_array = np.zeros((300, 400, 3), dtype=np.uint8)
    img_array[50:250, 30:180, 0] = 220
    img_array[50:250, 220:370, 2] = 220
    img_array[260:290, 30:370, 1] = 200
    image = Image.fromarray(img_array)

prompt = "USER: <image>\nDescribe what you see in this image.\nASSISTANT:"
inputs = processor(text=prompt, images=image, return_tensors="pt")
device = next(model.parameters()).device
inputs = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}

print(f"pixel_values: {inputs['pixel_values'].shape}")
print(f"input_ids:    {inputs['input_ids'].shape}")

# Set a breakpoint here and step into model.generate / model.forward
with torch.no_grad():
    output_ids = model.generate(**inputs, max_new_tokens=128)  # breakpoint

generated = processor.decode(output_ids[0], skip_special_tokens=True)
if "ASSISTANT:" in generated:
    generated = generated.split("ASSISTANT:")[-1].strip()
print(f"\nGenerated: {generated}")
