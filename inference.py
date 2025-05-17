# Copyright (c) 2023-2024 DeepSeek.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy of
# this software and associated documentation files (the "Software"), to deal in
# the Software without restriction, including without limitation the rights to
# use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of
# the Software, and to permit persons to whom the Software is furnished to do so,
# subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
# FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
# COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
# IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
# CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

import torch
import argparse
import os
import sys
from transformers import AutoModelForCausalLM

from janus.models import MultiModalityCausalLM, VLChatProcessor
from janus.utils.io import load_pil_images


def run_inference(model_path, image_path):
    """Run inference with the specified model on the given image"""
    print(f"Loading model from {model_path}...")
    vl_chat_processor = VLChatProcessor.from_pretrained(model_path)
    tokenizer = vl_chat_processor.tokenizer

    vl_gpt = AutoModelForCausalLM.from_pretrained(
        model_path, trust_remote_code=True
    )
    vl_gpt = vl_gpt.to(torch.bfloat16).cuda().eval()
    
    # Check if image exists
    if not os.path.exists(image_path):
        print(f"Error: Image not found at {image_path}")
        return
    
    prompt = "Give a radiology report based on the chest x-ray image, including FINDINGS and IMPRESSION."
    
    # Set up conversation
    conversation = [
        {
            "role": "User",
            "content": f"<image_placeholder>\n{prompt}",
            "images": [image_path],
        },
        {"role": "Assistant", "content": ""},
    ]
    
    # Load images and prepare inputs
    print(f"Processing image: {image_path}")
    pil_images = load_pil_images(conversation)
    prepare_inputs = vl_chat_processor(
        conversations=conversation, images=pil_images, force_batchify=True
    ).to(vl_gpt.device)
    
    # Generate image embeddings and run the model
    inputs_embeds = vl_gpt.prepare_inputs_embeds(**prepare_inputs)
    outputs = vl_gpt.language_model.generate(
        inputs_embeds=inputs_embeds,
        attention_mask=prepare_inputs.attention_mask,
        pad_token_id=tokenizer.eos_token_id,
        bos_token_id=tokenizer.bos_token_id,
        eos_token_id=tokenizer.eos_token_id,
        max_new_tokens=512,
        do_sample=False,
        use_cache=True,
        output_scores=True,
        return_dict_in_generate=True,
    )
    
    # Get the generated sequence
    generated_sequence = outputs.sequences[0].cpu().tolist()
    answer = tokenizer.decode(generated_sequence, skip_special_tokens=True)
    
    
    print("\n" + "=" * 50)
    print("INFERENCE RESULTS")
    print("=" * 50)
    print(f"Input prompt: {prompt}")
    print("-" * 50)
    print(answer)
    
    return answer


def main():
    parser = argparse.ArgumentParser(description="Run inference with Janus-Pro-CXR model")
    parser.add_argument("model_path", type=str, help="Path to the model directory")
    parser.add_argument("image_path", type=str, help="Path to the chest X-ray image")
    
    args = parser.parse_args()
    run_inference(args.model_path, args.image_path)


if __name__ == "__main__":
    main()

