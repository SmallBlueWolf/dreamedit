import os
import torch
import argparse
import random
import numpy as np
from pathlib import Path
import torch.nn.functional as F

from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler
from diffusers.pipelines.stable_diffusion import StableDiffusionPipeline
from peft import PeftModel, LoraConfig
from huggingface_hub import snapshot_download
from transformers.utils import is_offline_mode

# 导入配置加载函数
from src.utils import load_config

# 获取默认缓存目录，与Hugging Face库使用相同的目录
DEFAULT_CACHE_DIR = os.path.expanduser("~/.cache/huggingface/hub")

def get_model_path_from_cache(model_id, cache_dir=None):
    """
    尝试从本地缓存获取模型路径
    
    Args:
        model_id: 模型ID，例如 "CompVis/stable-diffusion-v1-4"
        cache_dir: 缓存目录，默认使用Hugging Face的默认缓存目录
        
    Returns:
        缓存的模型路径或None
    """
    if cache_dir is None:
        cache_dir = DEFAULT_CACHE_DIR
    
    # 检查该模型是否已经在缓存中
    try:
        # 这个函数会在本地查找，不会下载
        model_path = snapshot_download(
            repo_id=model_id, 
            local_files_only=True,
            cache_dir=cache_dir
        )
        print(f"找到本地缓存模型: {model_path}")
        return model_path
    except Exception as e:
        print(f"在缓存中未找到模型 {model_id}: {e}")
        return None

def examine_output_dir(output_dir):
    """
    检查训练输出目录的结构，确定保存的模型类型和结构
    
    Args:
        output_dir: 训练脚本的输出目录路径
        
    Returns:
        dict: 包含目录结构信息的字典
    """
    if not os.path.exists(output_dir):
        return {"exists": False}
    
    result = {
        "exists": True,
        "is_lora": False,
        "is_diffusers": False,
        "is_checkpoint": False,
        "components": [],
        "checkpoint_files": []
    }
    
    # 检查是否为LoRA模型（包含unet子目录但不是完整diffusers结构）
    unet_dir = os.path.join(output_dir, "unet")
    text_encoder_dir = os.path.join(output_dir, "text_encoder")
    
    if os.path.exists(unet_dir):
        result["components"].append("unet")
        
        # 检查是否为LoRA格式 - 关键是看adapter_config.json是否存在
        adapter_config = os.path.join(unet_dir, "adapter_config.json")
        if os.path.exists(adapter_config):
            result["is_lora"] = True
    
    if os.path.exists(text_encoder_dir):
        result["components"].append("text_encoder")
    
    # 检查是否为diffusers格式（包含model_index.json）
    if os.path.exists(os.path.join(output_dir, "model_index.json")):
        result["is_diffusers"] = True
        
        # 检查diffusers格式下的组件
        components = ["vae", "scheduler", "tokenizer", "feature_extractor"]
        for component in components:
            if os.path.exists(os.path.join(output_dir, component)):
                result["components"].append(component)
    
    # 检查是否为checkpoint文件
    for ext in [".safetensors", ".bin", ".ckpt", ".pt"]:
        files = [f for f in os.listdir(output_dir) if f.endswith(ext)]
        if files:
            result["is_checkpoint"] = True
            result["checkpoint_files"].extend(files)
    
    return result

def get_lora_sd_pipeline(
    ckpt_dir, base_model_name_or_path, dtype=torch.float16, device="cuda", adapter_name="default", 
    local_files_only=False, cache_dir=None
):
    """
    加载使用LoRA训练的Stable Diffusion模型
    """
    # 获取LoRA权重目录路径
    unet_sub_dir = os.path.join(ckpt_dir, "unet")
    text_encoder_sub_dir = os.path.join(ckpt_dir, "text_encoder")
    
    print(f"正在检查LoRA权重目录: {ckpt_dir}")
    print(f"  - unet目录: {'存在' if os.path.exists(unet_sub_dir) else '不存在'}")
    print(f"  - text_encoder目录: {'存在' if os.path.exists(text_encoder_sub_dir) else '不存在'}")
    
    if base_model_name_or_path is None:
        raise ValueError("请指定基础模型名称或路径")

    # 检查LoRA权重目录是否存在
    if not os.path.exists(unet_sub_dir):
        raise ValueError(f"unet子目录不存在: {unet_sub_dir}")
    
    # 首先尝试从缓存中加载基础模型
    cached_model_path = get_model_path_from_cache(base_model_name_or_path, cache_dir)
    model_path_to_use = cached_model_path if cached_model_path else base_model_name_or_path
    
    # 标记是否从本地加载
    is_local_path = cached_model_path is not None or os.path.exists(base_model_name_or_path)
    actual_local_files_only = local_files_only or (is_local_path and is_offline_mode())
    
    # 加载基础模型
    try:
        print(f"加载基础模型: {model_path_to_use}")
        print(f"是否仅使用本地文件: {actual_local_files_only}")
        pipe = StableDiffusionPipeline.from_pretrained(
            model_path_to_use, 
            torch_dtype=dtype,
            local_files_only=actual_local_files_only,
            cache_dir=cache_dir
        )
        # 使用更快的调度器
        pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
        pipe = pipe.to(device)
    except Exception as e:
        print(f"加载基础模型失败: {e}")
        if local_files_only or is_offline_mode():
            print("错误：离线模式下无法加载模型。")
            raise
        
        # 尝试在线模式加载
        print("尝试在线模式加载模型...")
        try:
            pipe = StableDiffusionPipeline.from_pretrained(
                base_model_name_or_path, 
                torch_dtype=dtype,
                cache_dir=cache_dir
            )
            pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
            pipe = pipe.to(device)
        except Exception as e:
            print(f"在线加载基础模型失败: {e}")
            raise
    
    # 加载unet LoRA权重
    try:
        print(f"加载unet LoRA权重: {unet_sub_dir}")
        # 检查是否有特定的配置文件
        adapter_config_path = os.path.join(unet_sub_dir, "adapter_config.json")
        if os.path.exists(adapter_config_path):
            print(f"发现adapter_config.json: {adapter_config_path}")
        else:
            print("未找到adapter_config.json，使用默认LoRA配置")
            
        pipe.unet = PeftModel.from_pretrained(pipe.unet, unet_sub_dir, adapter_name=adapter_name)
    except Exception as e:
        print(f"加载unet LoRA权重失败: {e}")
        raise

    # 如果有text_encoder LoRA权重，加载它
    if os.path.exists(text_encoder_sub_dir):
        try:
            print(f"加载text_encoder LoRA权重: {text_encoder_sub_dir}")
            pipe.text_encoder = PeftModel.from_pretrained(
                pipe.text_encoder, text_encoder_sub_dir, adapter_name=adapter_name
            )
        except Exception as e:
            print(f"加载text_encoder LoRA权重失败: {e}")
            print("将继续使用原始text_encoder")

    # 根据数据类型调整精度
    if dtype in (torch.float16, torch.bfloat16):
        print(f"将模型转换为{dtype}精度")
        pipe.unet.half()
        if hasattr(pipe.text_encoder, 'half'):
            pipe.text_encoder.half()

    # 确保模型在正确的设备上
    pipe.to(device)
    
    # 禁用StableDiffusionPipeline的进度条
    pipe.set_progress_bar_config(disable=True)
    
    print("LoRA模型加载完成")
    return pipe

def get_sd_pipeline(model_path, dtype=torch.float16, device="cuda", local_files_only=False, cache_dir=None, use_lora_config=None):
    """
    加载完整的Stable Diffusion模型（非LoRA）
    """
    # 将model_path转换为绝对路径
    model_path_abs = os.path.abspath(model_path)
    print(f"尝试从本地路径加载模型: {model_path_abs}")
    
    # 首先分析目录结构
    dir_info = examine_output_dir(model_path_abs)
    print(f"目录分析结果: {dir_info}")
    
    if not dir_info["exists"]:
        print(f"错误：路径 {model_path_abs} 不存在")
        # 尝试从缓存加载预训练模型
        cached_model_path = get_model_path_from_cache("CompVis/stable-diffusion-v1-4", cache_dir)
        pipe = StableDiffusionPipeline.from_pretrained(
            cached_model_path if cached_model_path else "CompVis/stable-diffusion-v1-4", 
            torch_dtype=dtype,
            local_files_only=local_files_only,
            cache_dir=cache_dir
        )
    # 仅当配置中use_lora=True时才考虑将模型作为LoRA加载
    elif use_lora_config == True and dir_info["is_lora"]:
        # 如果是LoRA模型且配置中启用了LoRA，使用LoRA加载函数
        print("检测到LoRA格式且配置启用了LoRA，使用LoRA加载器...")
        cached_base_model = get_model_path_from_cache("CompVis/stable-diffusion-v1-4", cache_dir)
        return get_lora_sd_pipeline(
            model_path_abs,
            cached_base_model if cached_base_model else "CompVis/stable-diffusion-v1-4",
            dtype=dtype,
            device=device,
            local_files_only=local_files_only,
            cache_dir=cache_dir
        )
    elif dir_info["is_diffusers"]:
        # 如果是完整的diffusers格式
        print("检测到完整diffusers格式，直接加载...")
        pipe = StableDiffusionPipeline.from_pretrained(
            model_path_abs, 
            torch_dtype=dtype,
            local_files_only=True
        )
    elif dir_info["is_checkpoint"]:
        # 如果包含checkpoint文件
        print(f"检测到checkpoint文件: {dir_info['checkpoint_files'][0]}")
        ckpt_path = os.path.join(model_path_abs, dir_info['checkpoint_files'][0])
        pipe = StableDiffusionPipeline.from_single_file(
            ckpt_path,
            torch_dtype=dtype
        )
    elif "unet" in dir_info["components"] and "vae" not in dir_info["components"]:
        # 可能是带有独立unet的非标准格式（在use_lora=False情况下的train.py输出）
        print("检测到非标准格式（仅包含unet），尝试特殊加载方式...")
        
        # 加载基础模型
        cached_base_model = get_model_path_from_cache("CompVis/stable-diffusion-v1-4", cache_dir)
        base_model_path = cached_base_model if cached_base_model else "CompVis/stable-diffusion-v1-4"
        
        pipe = StableDiffusionPipeline.from_pretrained(
            base_model_path,
            torch_dtype=dtype,
            local_files_only=local_files_only,
            cache_dir=cache_dir
        )
        
        # 尝试加载独立的unet
        try:
            unet_path = os.path.join(model_path_abs, "unet")
            from diffusers import UNet2DConditionModel
            
            # 检查是否有权重文件
            if os.path.exists(os.path.join(unet_path, "diffusion_pytorch_model.bin")):
                print(f"找到unet权重文件，加载中...")
                unet = UNet2DConditionModel.from_pretrained(unet_path, torch_dtype=dtype)
                pipe.unet = unet
                print("成功加载微调后的unet")
                
            # 检查是否有text_encoder
            if "text_encoder" in dir_info["components"]:
                text_encoder_path = os.path.join(model_path_abs, "text_encoder")
                if os.path.exists(os.path.join(text_encoder_path, "pytorch_model.bin")):
                    print("找到text_encoder权重文件，加载中...")
                    from transformers import CLIPTextModel
                    text_encoder = CLIPTextModel.from_pretrained(text_encoder_path, torch_dtype=dtype)
                    pipe.text_encoder = text_encoder
                    print("成功加载微调后的text_encoder")
        except Exception as e:
            print(f"加载独立组件失败: {e}")
            print("将使用基础模型")
            fine_tuned_pipe = base_pipe
    else:
        # 如果目录结构不明确，尝试常规方法
        print("目录结构不明确，尝试直接加载...")
        try:
            pipe = StableDiffusionPipeline.from_pretrained(
                model_path_abs, 
                torch_dtype=dtype,
                local_files_only=True
            )
        except Exception as e:
            print(f"直接加载失败: {e}")
            print("尝试加载预训练模型...")
            
            # 如果本地加载失败，回退到预训练模型
            cached_model_path = get_model_path_from_cache("CompVis/stable-diffusion-v1-4", cache_dir)
            pipe = StableDiffusionPipeline.from_pretrained(
                cached_model_path if cached_model_path else "CompVis/stable-diffusion-v1-4", 
                torch_dtype=dtype,
                local_files_only=local_files_only,
                cache_dir=cache_dir
            )
    
    # 使用更快的调度器
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
    pipe = pipe.to(device)
    pipe.set_progress_bar_config(disable=True)
    return pipe

def get_text_embeddings(pipe, prompt, device):
    """
    从模型中获取文本嵌入向量
    """
    # 对提示词进行编码
    text_inputs = pipe.tokenizer(
        prompt,
        padding="max_length",
        max_length=pipe.tokenizer.model_max_length,
        truncation=True,
        return_tensors="pt"
    )
    text_inputs = {k: v.to(device) for k, v in text_inputs.items()}
    
    # 计算文本嵌入
    with torch.no_grad():
        text_embedding = pipe.text_encoder(text_inputs["input_ids"])[0]
    
    return text_embedding

def get_latent_embeddings(pipe, prompt, generator, device):
    """
    获取模型对提示词编码后的潜在空间表示
    """
    # 处理提示词和负面提示词
    text_embeddings = get_text_embeddings(pipe, prompt, device)
    
    # 获取必要的配置值，避免直接访问属性
    # 修复FutureWarning并确保数据类型一致性
    if hasattr(pipe.unet, 'config'):
        in_channels = pipe.unet.config.in_channels
        sample_size = pipe.unet.config.sample_size
    else:
        # 向后兼容旧版本
        in_channels = pipe.unet.in_channels if hasattr(pipe.unet, 'in_channels') else 4
        sample_size = pipe.unet.sample_size if hasattr(pipe.unet, 'sample_size') else 64
    
    # 确保text_embeddings的数据类型与unet期望的一致
    dtype = next(pipe.unet.parameters()).dtype
    text_embeddings = text_embeddings.to(dtype)
    
    # 生成初始潜在向量（随机噪声）
    latents = torch.randn(
        (1, in_channels, sample_size, sample_size),
        generator=generator,
        device=device,
        dtype=dtype  # 确保噪声的数据类型与模型一致
    )
    
    # 通过UNet预测噪声残差
    with torch.no_grad():
        # 确保时间步张量与模型dtype一致
        timesteps = torch.tensor([0], device=device, dtype=dtype)
        noise_pred = pipe.unet(latents, timesteps, text_embeddings).sample
    
    return noise_pred

def compute_cosine_similarity(tensor1, tensor2):
    """
    计算两个张量之间的余弦相似度
    """
    # 将张量展平为向量
    flat1 = tensor1.view(-1)
    flat2 = tensor2.view(-1)
    
    # 计算余弦相似度
    similarity = F.cosine_similarity(flat1.unsqueeze(0), flat2.unsqueeze(0))
    
    return similarity.item()

def parse_args():
    parser = argparse.ArgumentParser(description="脚本从配置文件加载模型并进行推理。")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/default.yaml",
        help="配置文件路径，将与默认配置合并",
    )
    parser.add_argument(
        "--prompts",
        nargs='+',
        default=None,
        help="用于推理的提示词列表",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="随机种子，如果不指定则随机生成",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="inference_results",
        help="生成图像的输出目录",
    )
    parser.add_argument(
        "--negative_prompt",
        type=str,
        default="low quality, blurry, unfinished, poorly drawn",
        help="负面提示词",
    )
    parser.add_argument(
        "--offline",
        action="store_true",
        help="启用离线模式，仅使用本地缓存的模型",
    )
    parser.add_argument(
        "--cache_dir",
        type=str,
        default=None,
        help="模型缓存目录，默认使用huggingface默认缓存目录"
    )
    parser.add_argument(
        "--use_original_model_only",
        action="store_true",
        help="仅使用原始模型进行推理，不加载微调模型"
    )
    
    args = parser.parse_args()
    
    # 加载默认配置
    default_config = load_config("configs/default.yaml")
    
    # 如果指定了其他配置文件，则加载并合并配置
    if args.config != "configs/default.yaml":
        user_config = load_config(args.config)
        # 合并配置，用户配置优先级高于默认配置
        for key, value in user_config.items():
            default_config[key] = value
    
    # 将配置字典转换为命名空间
    config = argparse.Namespace(**default_config)
    
    # 将命令行参数合并到配置中 - 只有在命令行参数不是None时才覆盖
    if args.prompts is not None:
        config.prompts = args.prompts
    # 对于其他参数，保持原有逻辑
    config.seed = args.seed if args.seed is not None else random.randint(0, 1000000)
    config.inference_output_dir = args.output_dir
    config.negative_prompt = args.negative_prompt
    config.offline = args.offline
    config.cache_dir = args.cache_dir
    config.use_original_model_only = args.use_original_model_only
    
    return config

def is_black_image(image):
    """
    检查图像是否为黑色（NSFW检测触发的结果）
    """
    # 将PIL图像转换为NumPy数组
    img_array = np.array(image)
    
    # 检查图像是否全黑或接近全黑
    if img_array.ndim == 3 and img_array.shape[2] == 3:  # RGB图像
        # 计算平均像素值，如果接近0则认为是黑色图像
        mean_value = np.mean(img_array)
        return mean_value < 5.0  # 阈值设为5，因为完全黑色的平均值为0
    
    return False

def main():
    # 解析配置
    config = parse_args()
    
    # 创建输出目录
    output_dir = Path(config.inference_output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 创建相似度结果文件
    similarity_file = output_dir / "similarity_results.txt"
    
    # 设置设备和数据类型
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32
    
    # 设置随机种子
    if config.seed is not None:
        random.seed(config.seed)
        torch.manual_seed(config.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(config.seed)
    
    print(f"使用随机种子: {config.seed}")
    print(f"原始模型: {config.pretrained_model_name_or_path}")
    print(f"微调模型目录: {config.output_dir}")
    print(f"是否为LoRA配置: {config.use_lora}")
    print(f"离线模式: {'已启用' if config.offline else '未启用'}")
    if config.cache_dir:
        print(f"使用缓存目录: {config.cache_dir}")
    if config.use_original_model_only:
        print("注意: 仅使用原始模型进行推理")
    
    # 加载原始模型
    print("正在加载原始模型...")
    # 首先检查缓存中是否存在模型
    cached_model_path = get_model_path_from_cache(config.pretrained_model_name_or_path, config.cache_dir)
    
    # 尝试导入必要的类，确保作用域正确
    from diffusers import StableDiffusionPipeline
    
    try:
        base_pipe = StableDiffusionPipeline.from_pretrained(
            cached_model_path if cached_model_path else config.pretrained_model_name_or_path, 
            torch_dtype=dtype,
            local_files_only=config.offline,
            cache_dir=config.cache_dir
        )
        base_pipe.scheduler = DPMSolverMultistepScheduler.from_config(base_pipe.scheduler.config)
        base_pipe = base_pipe.to(device)
        base_pipe.set_progress_bar_config(disable=True)
    except Exception as e:
        print(f"加载原始模型失败: {e}")
        if config.offline:
            print("错误：离线模式下无法找到模型。请检查缓存路径或禁用离线模式。")
            return
        else:
            print("尝试重新下载模型...")
            try:
                base_pipe = StableDiffusionPipeline.from_pretrained(
                    config.pretrained_model_name_or_path, 
                    torch_dtype=dtype,
                    cache_dir=config.cache_dir
                )
                base_pipe.scheduler = DPMSolverMultistepScheduler.from_config(base_pipe.scheduler.config)
                base_pipe = base_pipe.to(device)
                base_pipe.set_progress_bar_config(disable=True)
            except Exception as e:
                print(f"加载/下载原始模型再次失败: {e}")
                return
    
    # 如果仅使用原始模型进行推理，则跳过微调模型加载
    if config.use_original_model_only:
        fine_tuned_pipe = base_pipe
        print("跳过微调模型加载，仅使用原始模型")
    else:
        # 加载微调模型
        print("正在加载微调模型...")
        try:
            # 首先检查output_dir目录结构
            output_dir_path = os.path.abspath(config.output_dir)
            
            # 分析目录结构
            dir_info = examine_output_dir(output_dir_path)
            print(f"微调模型目录分析结果: {dir_info}")
            
            if not dir_info["exists"]:
                print(f"警告: 微调模型目录 {output_dir_path} 不存在，回退到使用原始模型")
                fine_tuned_pipe = base_pipe
            else:
                print(f"微调模型目录存在: {output_dir_path}")
                
                # 配置文件中use_lora设置为true且目录包含adapter_config.json才加载为LoRA
                if config.use_lora == True and dir_info["is_lora"]:
                    print("配置使用LoRA且检测到LoRA模型结构，使用LoRA加载器...")
                    # 对于LoRA模型，需要先加载基础模型，然后应用LoRA权重
                    fine_tuned_pipe = get_lora_sd_pipeline(
                        output_dir_path,
                        config.pretrained_model_name_or_path,
                        dtype=dtype,
                        device=device,
                        adapter_name="default",
                        local_files_only=config.offline,
                        cache_dir=config.cache_dir
                    )
                elif dir_info["is_diffusers"]:
                    # 检测到diffusers格式，尝试直接加载
                    print("检测到标准diffusers格式，直接加载完整模型...")
                    try:
                        fine_tuned_pipe = StableDiffusionPipeline.from_pretrained(
                            output_dir_path,
                            torch_dtype=dtype,
                            local_files_only=True
                        )
                        fine_tuned_pipe.scheduler = DPMSolverMultistepScheduler.from_config(fine_tuned_pipe.scheduler.config)
                        fine_tuned_pipe = fine_tuned_pipe.to(device)
                        fine_tuned_pipe.set_progress_bar_config(disable=True)
                    except Exception as e:
                        print(f"加载完整diffusers模型失败: {e}")
                        # 使用通用加载方法
                        fine_tuned_pipe = get_sd_pipeline(
                            output_dir_path,
                            dtype=dtype,
                            device=device,
                            local_files_only=config.offline,
                            cache_dir=config.cache_dir,
                            use_lora_config=config.use_lora
                        )
                else:
                    # 在train.py中use_lora=False时，有可能只保存了单个组件
                    # 检查是否只包含unet和可能的text_encoder
                    if ("unet" in dir_info["components"] and 
                        len([c for c in dir_info["components"] if c not in ["unet", "text_encoder"]]) == 0):
                        print("检测到非LoRA单组件模型，加载独立组件...")
                        # 先加载基础模型
                        fine_tuned_pipe = base_pipe.copy()
                        
                        # 加载unet
                        try:
                            unet_path = os.path.join(output_dir_path, "unet")
                            from diffusers import UNet2DConditionModel
                            print(f"加载微调的unet: {unet_path}")
                            
                            # 检查unet目录中的模型文件
                            unet_model_files = [f for f in os.listdir(unet_path) 
                                             if f in ["diffusion_pytorch_model.bin", "model.safetensors"]]
                            
                            if unet_model_files:
                                unet = UNet2DConditionModel.from_pretrained(
                                    unet_path, 
                                    torch_dtype=dtype
                                ).to(device)
                                fine_tuned_pipe.unet = unet
                                print("成功加载微调后的unet")
                            else:
                                print("未找到unet权重文件")
                                
                            # 加载text_encoder（如果存在）
                            if "text_encoder" in dir_info["components"]:
                                text_encoder_path = os.path.join(output_dir_path, "text_encoder")
                                text_encoder_files = [f for f in os.listdir(text_encoder_path) 
                                                   if f in ["pytorch_model.bin", "model.safetensors"]]
                                
                                if text_encoder_files:
                                    print(f"加载微调的text_encoder: {text_encoder_path}")
                                    from transformers import CLIPTextModel
                                    text_encoder = CLIPTextModel.from_pretrained(
                                        text_encoder_path, 
                                        torch_dtype=dtype
                                    ).to(device)
                                    fine_tuned_pipe.text_encoder = text_encoder
                                    print("成功加载微调后的text_encoder")
                        except Exception as e:
                            print(f"加载单组件模型失败: {e}")
                            print("回退到使用原始模型")
                            fine_tuned_pipe = base_pipe
                    else:
                        # 使用通用加载方法尝试加载
                        print("使用通用加载器尝试加载模型...")
                        fine_tuned_pipe = get_sd_pipeline(
                            output_dir_path,
                            dtype=dtype,
                            device=device,
                            local_files_only=config.offline,
                            cache_dir=config.cache_dir,
                            use_lora_config=config.use_lora
                        )
        except Exception as e:
            print(f"加载微调模型失败: {e}")
            print("回退到使用原始模型")
            fine_tuned_pipe = base_pipe
    
    # 定义推理用的prompt列表
    if config.prompts is None:
        # 如果命令行未提供prompts，则使用配置文件中的默认值
        # 如果配置中也没有提供，则使用默认提示词列表
        prompt_inference = getattr(config, 'prompts', ["a photo of a dog", "a painting of a dog"])
        print(f"使用配置文件中的提示词列表: {prompt_inference}")
    else:
        prompt_inference = config.prompts
        print(f"使用命令行指定的提示词列表: {prompt_inference}")
    
    # 确保prompt_inference不为None且是可迭代对象
    if prompt_inference is None or not isinstance(prompt_inference, (list, tuple)):
        prompt_inference = ["a photo of a dog", "a painting of a dog"] 
        print(f"未找到有效的提示词列表，使用默认值: {prompt_inference}")
    
    # 写入相似度结果文件头部
    with open(similarity_file, "w") as f:
        f.write("Prompt,Text Embedding Similarity,Latent Embedding Similarity\n")
    
    # 对每个prompt进行推理
    for i, prompt in enumerate(prompt_inference):
        print(f"\n正在生成第 {i+1}/{len(prompt_inference)} 个提示词的图像: '{prompt}'")
        
        # 设置随机种子生成器，保存原始种子用于相似度计算
        original_seed = config.seed
        generator = torch.Generator(device=device).manual_seed(original_seed)
        
        print("计算文本和潜在空间编码的相似度...")
        
        # 计算文本嵌入的相似度
        base_text_embedding = get_text_embeddings(base_pipe, prompt, device)
        fine_tuned_text_embedding = get_text_embeddings(fine_tuned_pipe, prompt, device)
        text_similarity = compute_cosine_similarity(base_text_embedding, fine_tuned_text_embedding)
        
        print(f"文本嵌入余弦相似度: {text_similarity:.6f}")
        
        # 计算潜在空间编码的相似度
        generator_latent = torch.Generator(device=device).manual_seed(original_seed) # 重置种子
        base_latent = get_latent_embeddings(base_pipe, prompt, generator_latent, device)
        
        generator_latent = torch.Generator(device=device).manual_seed(original_seed) # 重置种子
        fine_tuned_latent = get_latent_embeddings(fine_tuned_pipe, prompt, generator_latent, device)
        
        latent_similarity = compute_cosine_similarity(base_latent, fine_tuned_latent)
        print(f"潜在空间编码余弦相似度: {latent_similarity:.6f}")
        
        # 将相似度结果写入文件
        with open(similarity_file, "a") as f:
            f.write(f'"{prompt}",{text_similarity:.6f},{latent_similarity:.6f}\n')
        
        # 使用原始模型生成图像 - 添加重试逻辑
        print("使用原始模型生成图像...")
        orig_image = None
        orig_retry_count = 0
        max_retries = 5000  # 最大重试次数
        current_seed = original_seed
        
        while orig_retry_count < max_retries:
            generator = torch.Generator(device=device).manual_seed(current_seed)
            
            orig_image = base_pipe(
                prompt, 
                num_inference_steps=50, 
                guidance_scale=7.5, 
                negative_prompt=config.negative_prompt,
                generator=generator
            ).images[0]
            
            # 检查是否为黑色图像(NSFW触发)
            if is_black_image(orig_image):
                orig_retry_count += 1
                # 生成新的随机种子
                current_seed = random.randint(0, 1000000)
                print(f"检测到NSFW内容，原始模型重试 ({orig_retry_count}/{max_retries})，新种子: {current_seed}")
            else:
                # 非黑色图像，保存并退出循环
                break
                
        # 如果达到最大重试次数仍然是黑色图像，给出警告
        if orig_retry_count >= max_retries and is_black_image(orig_image):
            print(f"警告: 原始模型在{max_retries}次尝试后仍然触发NSFW检测")
        
        # 保存原始模型生成的图像
        orig_image_path = output_dir / f"original_{i}_{prompt.replace(' ', '_')[:50]}.png"
        orig_image.save(orig_image_path)
        print(f"原始模型图像已保存至: {orig_image_path}")
        
        # 使用微调模型生成图像 - 添加重试逻辑
        print("使用微调模型生成图像...")
        fine_tuned_image = None
        fine_tuned_retry_count = 0
        current_seed = original_seed  # 重置种子为原始种子
        
        while fine_tuned_retry_count < max_retries:
            generator = torch.Generator(device=device).manual_seed(current_seed)
            
            fine_tuned_image = fine_tuned_pipe(
                prompt, 
                num_inference_steps=50, 
                guidance_scale=7.5, 
                negative_prompt=config.negative_prompt,
                generator=generator
            ).images[0]
            
            # 检查是否为黑色图像(NSFW触发)
            if is_black_image(fine_tuned_image):
                fine_tuned_retry_count += 1
                # 生成新的随机种子
                current_seed = random.randint(0, 1000000)
                print(f"检测到NSFW内容，微调模型重试 ({fine_tuned_retry_count}/{max_retries})，新种子: {current_seed}")
            else:
                # 非黑色图像，保存并退出循环
                break
                
        # 如果达到最大重试次数仍然是黑色图像，给出警告
        if fine_tuned_retry_count >= max_retries and is_black_image(fine_tuned_image):
            print(f"警告: 微调模型在{max_retries}次尝试后仍然触发NSFW检测")
        
        # 保存微调模型生成的图像
        fine_tuned_image_path = output_dir / f"finetuned_{i}_{prompt.replace(' ', '_')[:50]}.png"
        fine_tuned_image.save(fine_tuned_image_path)
        print(f"微调模型图像已保存至: {fine_tuned_image_path}")
    
    print(f"\n所有图像生成完成! 相似度结果已保存至 {similarity_file}")

if __name__ == "__main__":
    main()