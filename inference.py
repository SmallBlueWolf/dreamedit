import os
import torch

from diffusers import StableDiffusionPipeline
from peft import PeftModel, LoraConfig

MODEL_NAME = "CompVis/stable-diffusion-v1-4"

def get_lora_sd_pipeline(
    ckpt_dir, base_model_name_or_path=MODEL_NAME, dtype=torch.float16, device="cuda", adapter_name="default"
):
    unet_sub_dir = os.path.join(ckpt_dir, "unet")
    text_encoder_sub_dir = os.path.join(ckpt_dir, "text_encoder")
    if os.path.exists(text_encoder_sub_dir) and base_model_name_or_path is None:
        config = LoraConfig.from_pretrained(text_encoder_sub_dir)
        base_model_name_or_path = config.base_model_name_or_path

    if base_model_name_or_path is None:
        raise ValueError("Please specify the base model name or path")

    pipe = StableDiffusionPipeline.from_pretrained(base_model_name_or_path, torch_dtype=dtype).to(device)
    
    # 检查unet_sub_dir是否存在
    if not os.path.exists(unet_sub_dir):
        raise ValueError(f"unet子目录不存在: {unet_sub_dir}")

    pipe.unet = PeftModel.from_pretrained(pipe.unet, unet_sub_dir, adapter_name=adapter_name)

    if os.path.exists(text_encoder_sub_dir):
        pipe.text_encoder = PeftModel.from_pretrained(
            pipe.text_encoder, text_encoder_sub_dir, adapter_name=adapter_name
        )

    if dtype in (torch.float16, torch.bfloat16):
        pipe.unet.half()
        pipe.text_encoder.half()

    pipe.to(device)
    
    # 禁用StableDiffusionPipeline的进度条
    pipe.set_progress_bar_config(disable=True)
    
    return pipe

def main():
    pipe = get_lora_sd_pipeline(
        "results",
        adapter_name="dog"
    )
    
    prompt = "big dog playing fetch in the beach"
    negative_prompt = "low quality, blurry, unfinished"
    image = pipe(prompt, num_inference_steps=50, guidance_scale=7, negative_prompt=negative_prompt).images[0]
    image.save("infer_img.jpg")

if __name__ == "__main__":
    main()