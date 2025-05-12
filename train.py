import hashlib
import itertools
import logging
import math
import os
from contextlib import nullcontext
from pathlib import Path

import datasets
import diffusers
import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
import transformers
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import set_seed
from diffusers import (
    AutoencoderKL,
    DDPMScheduler,
    DiffusionPipeline,
    DPMSolverMultistepScheduler,
    UNet2DConditionModel,
)
from diffusers.optimization import get_scheduler
from diffusers.utils import check_min_version
from diffusers.utils.import_utils import is_xformers_available
from huggingface_hub import HfApi
from tqdm.auto import tqdm
from transformers import AutoTokenizer

from peft import LoraConfig, get_peft_model

from src.dataset import DreamBoothDataset, PromptDataset, collate_fn
from src.utils import TorchTracemalloc, import_model_class_from_model_name_or_path, b2mb
from src.args import parse_args


# Will error if the minimal version of diffusers is not installed. Remove at your own risks.
check_min_version("0.10.0.dev0")

logger = get_logger(__name__)

UNET_TARGET_MODULES = ["to_q", "to_v", "query", "value"]  # , "ff.net.0.proj"]
TEXT_ENCODER_TARGET_MODULES = ["q_proj", "v_proj"]

def compute_projection_matrix(embeddings, threshold=1e-5):
    """
    计算投影矩阵，基于AlphaEdit方法。
    
    注意：保留此函数是为了向后兼容，现在它只是简单地调用compute_projection_matrices。
    
    Args:
        embeddings: 知识表示的嵌入向量 [batch_size, embed_dim]
        threshold: 特征值筛选的阈值
        
    Returns:
        projection_matrix: 投影矩阵，用于将扰动投影到零空间
    """
    # 确保threshold是浮点数
    threshold = float(threshold)
    
    # 将embeddings包装在列表中，模拟单层情况
    layer_embeddings = [embeddings]
    
    # 调用新函数计算投影矩阵
    projection_matrices = compute_projection_matrices(layer_embeddings, threshold)
    
    # 返回唯一的投影矩阵
    if projection_matrices and projection_matrices[0] is not None:
        return projection_matrices[0]
    
    # 如果出错，返回单位矩阵
    logger.warning("Failed to compute projection matrix, returning identity matrix")
    return torch.eye(embeddings.shape[1], device=embeddings.device)

def find_cross_attention_layers(unet):
    """
    查找UNet中所有交叉注意力层
    
    Args:
        unet: UNet模型
        
    Returns:
        ca_layers: 交叉注意力层列表
    """
    # 首先添加一些调试信息，了解模型结构
    logger.info(f"Searching for cross-attention layers in model: {type(unet).__name__}")
    
    # 用于存储找到的交叉注意力层
    ca_layers = []
    
    # 检查是否是使用LoRA包装的模型
    is_peft_model = "peft" in str(type(unet)).lower()
    if is_peft_model:
        logger.info("Detected PEFT/LoRA model, adjusting search strategy")
        # 如果是PEFT模型，我们需要查找base_model
        if hasattr(unet, "base_model"):
            base_model = unet.base_model
            logger.info(f"Found base_model: {type(base_model).__name__}")
            # 递归查找
            if hasattr(base_model, "model"):
                logger.info("Searching in base_model.model")
                actual_model = base_model.model
            else:
                logger.info("Using base_model directly")
                actual_model = base_model
        else:
            logger.info("No base_model found, using unet directly")
            actual_model = unet
    else:
        logger.info("Using standard UNet model")
        actual_model = unet
    
    # 特定于StableDiffusion UNet的交叉注意力层定位
    # 注意：我们需要找到正确的父模块，而不是子组件
    target_modules = []
    
    # 收集所有疑似CrossAttention模块
    for name, module in actual_model.named_modules():
        # 只考虑可能是交叉注意力的模块，跳过子组件
        if 'attn2' in name and not any(x in name for x in ['.to_', '.processor']):
            target_modules.append((name, module))
            logger.info(f"Found potential cross-attention module: {name}, type: {type(module).__name__}")
    
    # 现在检查每个疑似模块是否具有必要的属性
    for name, module in target_modules:
        # 检查基本属性
        has_to_v = hasattr(module, 'to_v')
        has_to_q = hasattr(module, 'to_q')
        has_to_k = hasattr(module, 'to_k')
        
        # 详细的调试信息
        logger.info(f"Checking module {name}: has_to_v={has_to_v}, has_to_q={has_to_q}, has_to_k={has_to_k}")
        
        # 确认这是一个完整的交叉注意力模块
        if has_to_v and has_to_q and has_to_k:
            # 再次检查to_v是否有weight属性
            if hasattr(module.to_v, 'weight'):
                ca_layers.append((name, module))
                logger.info(f"✅ Confirmed cross-attention layer: {name}")
            else:
                logger.warning(f"❌ Module {name} has to_v but no weight attribute")
        else:
            logger.warning(f"❌ Module {name} is not a complete cross-attention module")
    
    logger.info(f"Found {len(ca_layers)} valid cross-attention layers")
    return ca_layers

def collect_layer_specific_embeddings(unet, text_encoder, tokenizer, prompts, device, batch_size=16):
    """
    为UNet中的每个交叉注意力层收集特定的知识嵌入。
    
    Args:
        unet: UNet模型
        text_encoder: 文本编码器模型
        tokenizer: 分词器
        prompts: 表示知识的文本提示列表
        device: 计算设备
        batch_size: 批处理大小
        
    Returns:
        layer_embeddings: 每一层的知识嵌入字典，键为维度大小
        dimensions: 每一层投影矩阵应有的维度
    """
    # 获取UNet中所有的交叉注意力层
    ca_layers = find_cross_attention_layers(unet)
    
    if not ca_layers:
        raise ValueError("No valid cross-attention layers found in UNet. Cannot proceed with AlphaEdit.")
    
    logger.info(f"Found {len(ca_layers)} valid cross-attention layers in UNet")
    
    # 提取每层权重的维度
    layer_dimensions = []
    unique_out_dimensions = set()  # 改为收集输出维度
    
    for name, layer in ca_layers:
        # 不再需要检查to_v属性，因为find_cross_attention_layers已经确保了这一点
        weight = layer.to_v.weight
        out_dim, in_dim = weight.shape
        logger.info(f"Layer {name}: weight shape torch.Size([{out_dim}, {in_dim}])")
        layer_dimensions.append((out_dim, in_dim))
        unique_out_dimensions.add(out_dim)  # 存储输出维度
    
    # 验证已找到所有需要的维度
    if not unique_out_dimensions:
        raise ValueError("Failed to detect any valid dimensions from UNet layers")
    
    logger.info(f"Detected unique output dimensions: {unique_out_dimensions}")
    
    # 获取text_encoder的输出维度（通常是768用于CLIP）
    text_encoder_dim = None
    with torch.no_grad():
        # 对单个提示进行编码
        sample_text = tokenizer(
            prompts[0],
            padding="max_length",
            max_length=tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt"
        )
        sample_text = {k: v.to(device) for k, v in sample_text.items()}
        text_encoder_output = text_encoder(sample_text['input_ids'])[0]
        text_encoder_dim = text_encoder_output.shape[-1]  # 通常是768
    
    logger.info(f"Text encoder output dimension: {text_encoder_dim}")
    
    # 处理每个提示词，收集text_encoder的输出
    all_text_embeddings = []
    
    for i in tqdm(range(0, len(prompts), batch_size), desc="Collecting text embeddings"):
        batch_prompts = prompts[i:i + batch_size]
        
        # 对提示进行编码
        text_inputs = tokenizer(
            batch_prompts,
            padding="max_length",
            max_length=tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt"
        )
        
        # 确保输入在正确的设备上
        text_inputs = {k: v.to(device) for k, v in text_inputs.items()}
        
        # 获取文本编码
        with torch.no_grad():
            # 确认text_encoder是否在正确设备上
            text_encoder_device = next(text_encoder.parameters()).device
            if text_encoder_device != device:
                logger.warning(f"Text encoder on {text_encoder_device} but inputs on {device}. Moving text encoder.")
                text_encoder = text_encoder.to(device)
                
            # 获取文本嵌入
            text_embeddings = text_encoder(text_inputs['input_ids'])[0].detach()
            
            # 提取每个序列的嵌入
            for j, input_ids in enumerate(text_inputs['input_ids']):
                # 找到序列的实际长度（排除padding标记）
                seq_len = (input_ids != tokenizer.pad_token_id).sum().item()
                
                # 获取最后一个非padding标记的嵌入
                token_embedding = text_embeddings[j, seq_len-1, :].unsqueeze(0)
                all_text_embeddings.append(token_embedding)
    
    # 合并所有文本嵌入
    if all_text_embeddings:
        base_text_embeddings = torch.cat(all_text_embeddings, dim=0)
        logger.info(f"Collected {base_text_embeddings.shape[0]} text embeddings of dimension {base_text_embeddings.shape[1]}")
    else:
        raise ValueError("No text embeddings were collected!")
    
    # 创建投影矩阵生成器，用于为不同维度创建正交基
    def create_orthogonal_basis(dim, num_vectors, device):
        # 创建随机矩阵
        rand_matrix = torch.randn(dim, num_vectors, device=device)
        # 使用QR分解获取正交基
        q, _ = torch.linalg.qr(rand_matrix)
        return q
    
    # 为每个独特的输出维度创建特定的嵌入和投影矩阵
    dimension_embeddings = {}
    
    logger.info("Creating dimension-specific embeddings for gradient dimensions...")
    
    for dim in unique_out_dimensions:
        # 创建适当大小的随机嵌入
        # 我们需要为每个输出维度创建对应的嵌入和投影矩阵
        # 这个嵌入矩阵的目的是为了捕捉知识在这个特定维度上的表示
        num_samples = base_text_embeddings.shape[0]
        
        # 从文本嵌入投影到目标维度的嵌入空间
        if text_encoder_dim != dim:
            logger.info(f"Creating projection from text embeddings ({text_encoder_dim}) to dimension {dim}")
            projection = torch.nn.Linear(text_encoder_dim, dim, bias=False).to(device)
            torch.nn.init.orthogonal_(projection.weight)
            
            with torch.no_grad():
                dim_embeddings = projection(base_text_embeddings)
        else:
            # 如果维度匹配，直接使用
            logger.info(f"Using text encoder embeddings directly for dimension {dim}")
            dim_embeddings = base_text_embeddings
            
        # 存储这个维度的嵌入
        dimension_embeddings[dim] = dim_embeddings
    
    return dimension_embeddings, layer_dimensions

def compute_projection_matrices(embeddings_dict, dimensions, threshold=1e-5):
    """
    为每一层计算专属的投影矩阵。
    
    Args:
        embeddings_dict: 每个维度的知识嵌入字典
        dimensions: 每一层的维度元组列表 (out_dim, in_dim)
        threshold: 特征值筛选的阈值
        
    Returns:
        projection_matrices: 每层的投影矩阵列表
    """
    projection_matrices = []
    
    # 确保阈值是浮点数
    threshold_value = float(threshold)
    
    # 为每个输出维度计算基础投影矩阵
    dimension_projections = {}
    
    # 计算每个维度的基础投影矩阵
    for dim, embeddings in embeddings_dict.items():
        device = embeddings.device
        
        # 计算非中心协方差矩阵
        product = embeddings.T @ embeddings
        
        # 对协方差矩阵进行SVD分解
        try:
            U, S, _ = torch.linalg.svd(product, full_matrices=False)
            
            # 获取小于阈值的特征值对应的特征向量
            mask = S < threshold_value
            smallest_indices = mask.nonzero().squeeze()
            
            if len(smallest_indices) == 0 or smallest_indices.numel() == 0:
                # 计算保留10%特征向量的数量
                k = max(1, int(0.1 * len(S)))
                smallest_indices = torch.argsort(S)[:k]
                logger.info(f"No eigenvalues below threshold for dim {dim}. Using smallest {k} eigenvalues.")
            else:
                logger.info(f"Found {len(smallest_indices)} eigenvalues below threshold for dim {dim}")
            
            # 构建投影矩阵
            U_smallest = U[:, smallest_indices]
            projection = U_smallest @ U_smallest.T
            
            dimension_projections[dim] = projection
            logger.info(f"Created base projection for dimension {dim} with shape {projection.shape}")
            
        except Exception as e:
            # 不再使用备用方案，直接抛出错误
            raise ValueError(f"Error computing projection matrix for dimension {dim}: {e}")
    
    # 为每一层分配对应维度的投影矩阵
    for layer_idx, (out_dim, in_dim) in enumerate(dimensions):
        if out_dim in dimension_projections:  # 使用输出维度匹配
            projection_matrices.append(dimension_projections[out_dim])
            logger.info(f"Layer {layer_idx}: Using projection matrix for output dimension {out_dim}")
        else:
            # 找不到匹配的维度，抛出错误
            raise ValueError(f"Layer {layer_idx}: Cannot find projection matrix for dimension {out_dim}")
    
    # 验证所有层都有对应的投影矩阵
    if len(projection_matrices) != len(dimensions):
        raise ValueError(f"Not all layers have projection matrices: {len(projection_matrices)}/{len(dimensions)}")
    
    return projection_matrices

def train_step(batch, vae, text_encoder, unet, noise_scheduler, optimizer, lr_scheduler, accelerator, args, weight_dtype):
    # Convert images to latent space
    latents = vae.encode(batch["pixel_values"].to(dtype=weight_dtype)).latent_dist.sample()
    latents = latents * 0.18215

    # Sample noise that we'll add to the latents
    noise = torch.randn_like(latents)
    bsz = latents.shape[0]
    # Sample a random timestep for each image
    timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (bsz,), device=latents.device)
    timesteps = timesteps.long()

    # Add noise to the latents according to the noise magnitude at each timestep
    # (this is the forward diffusion process)
    noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

    # Get the text embedding for conditioning
    encoder_hidden_states = text_encoder(batch["input_ids"])[0]
    
    # 如果使用AlphaEdit，并且模型已初始化投影矩阵，应用投影矩阵修改嵌入向量
    if hasattr(args, 'use_alphaedit') and args.use_alphaedit and hasattr(unet, 'projection_matrices'):
        # 应用投影矩阵到交叉注意力层权重的梯度
        # 这是实现空间约束的关键步骤
        with torch.no_grad():
            # 获取所有交叉注意力层
            ca_layers = find_cross_attention_layers(unet)
            
            # 建立模块到投影矩阵索引的映射
            module_to_idx = {module: idx for idx, (_, module) in enumerate(ca_layers)}
            
            # 模块投影矩阵缓存，避免重复创建
            module_projections = {}
            
            # 使用钩子在反向传播时应用投影
            def hook_fn(module, grad_input, grad_output):
                try:
                    # 如果没有梯度输入，直接返回
                    if grad_input is None or len(grad_input) == 0 or grad_input[0] is None:
                        return grad_input
                    
                    # 使用预先计算的映射直接获取对应的投影矩阵
                    if hasattr(unet, 'layer_to_matrix_map') and module in unet.layer_to_matrix_map:
                        P = unet.layer_to_matrix_map[module]
                        
                        # 确保设备和类型匹配
                        P = P.to(grad_output[0].device, grad_output[0].dtype)
                        
                        # 获取梯度输入的形状和维度
                        grad_dim = grad_input[0].shape[-1]
                        
                        # 验证投影矩阵维度是否匹配梯度维度
                        if P.shape[0] != grad_dim:
                            raise ValueError(f"Projection matrix dimension {P.shape[0]} doesn't match "
                                            f"gradient input dimension {grad_dim}")
                        
                        # 应用投影矩阵
                        if len(grad_input) > 1:
                            return (grad_input[0] @ P, *grad_input[1:])
                        else:
                            return (grad_input[0] @ P,)
                    else:
                        # 如果映射中没有该模块，抛出错误
                        raise ValueError(f"Module {module.__class__.__name__} not found in layer_to_matrix_map")
                except Exception as e:
                    logger.error(f"Error in AlphaEdit hook: {e}")
                    # 出错时抛出异常，而不是静默使用原始梯度
                    raise
            
            # 为UNet中的交叉注意力模块注册钩子
            hooks = []
            ca_layers = find_cross_attention_layers(unet)
            for _, module in ca_layers:
                hooks.append(module.register_full_backward_hook(hook_fn))
    
    # Predict the noise residual
    model_pred = unet(noisy_latents, timesteps, encoder_hidden_states).sample

    # Get the target for loss depending on the prediction type
    if noise_scheduler.config.prediction_type == "epsilon":
        target = noise
    elif noise_scheduler.config.prediction_type == "v_prediction":
        target = noise_scheduler.get_velocity(latents, noise, timesteps)
    else:
        raise ValueError(f"Unknown prediction type {noise_scheduler.config.prediction_type}")

    if args.with_prior_preservation:
        # Chunk the noise and model_pred into two parts and compute the loss on each part separately.
        model_pred, model_pred_prior = torch.chunk(model_pred, 2, dim=0)
        target, target_prior = torch.chunk(target, 2, dim=0)

        # Compute instance loss
        loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")

        # Compute prior loss
        prior_loss = F.mse_loss(model_pred_prior.float(), target_prior.float(), reduction="mean")

        # Add the prior loss to the instance loss.
        loss = loss + args.prior_loss_weight * prior_loss
    else:
        loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")

    accelerator.backward(loss)
    if accelerator.sync_gradients:
        params_to_clip = (
            itertools.chain(unet.parameters(), text_encoder.parameters())
            if args.train_text_encoder
            else unet.parameters()
        )
        accelerator.clip_grad_norm_(params_to_clip, args.max_grad_norm)
    optimizer.step()
    lr_scheduler.step()
    optimizer.zero_grad()
    
    # 如果使用AlphaEdit，清除注册的钩子
    if hasattr(args, 'use_alphaedit') and args.use_alphaedit and hasattr(unet, 'projection_matrices') and 'hooks' in locals():
        for hook in hooks:
            hook.remove()
    
    return loss

def run_validation(args, accelerator, epoch, step, unet, text_encoder, num_update_steps_per_epoch):
    logger.info(
        f"Running validation... \n Generating {args.num_validation_images} images with prompt:"
        f" {args.validation_prompt}."
    )
    # create pipeline
    pipeline = DiffusionPipeline.from_pretrained(
        args.pretrained_model_name_or_path,
        safety_checker=None,
        revision=args.revision,
    )
    # set `keep_fp32_wrapper` to True because we do not want to remove
    # mixed precision hooks while we are still training
    pipeline.unet = accelerator.unwrap_model(unet, keep_fp32_wrapper=True)
    pipeline.text_encoder = accelerator.unwrap_model(text_encoder, keep_fp32_wrapper=True)
    pipeline.scheduler = DPMSolverMultistepScheduler.from_config(pipeline.scheduler.config)
    pipeline = pipeline.to(accelerator.device)
    pipeline.set_progress_bar_config(disable=True)

    # run inference
    if args.seed is not None:
        generator = torch.Generator(device=accelerator.device).manual_seed(args.seed)
    else:
        generator = None
    images = []
    for _ in range(args.num_validation_images):
        image = pipeline(args.validation_prompt, num_inference_steps=25, generator=generator).images[0]
        images.append(image)

    for tracker in accelerator.trackers:
        if tracker.name == "tensorboard":
            np_images = np.stack([np.asarray(img) for img in images])
            tracker.writer.add_images("validation", np_images, epoch, dataformats="NHWC")
        if tracker.name == "wandb":
            import wandb

            tracker.log(
                {
                    "validation": [
                        wandb.Image(image, caption=f"{i}: {args.validation_prompt}")
                        for i, image in enumerate(images)
                    ]
                }
            )

    del pipeline
    torch.cuda.empty_cache()

def train_epoch(
    epoch, 
    train_dataloader, 
    unet, 
    text_encoder, 
    vae, 
    noise_scheduler, 
    optimizer, 
    lr_scheduler, 
    accelerator, 
    args, 
    weight_dtype,
    global_step, 
    progress_bar, 
    num_update_steps_per_epoch, 
    first_epoch, 
    resume_step
):
    unet.train()
    if args.train_text_encoder:
        text_encoder.train()
    
    with TorchTracemalloc() if not args.no_tracemalloc else nullcontext() as tracemalloc:
        for step, batch in enumerate(train_dataloader):
            # Skip steps until we reach the resumed step
            if args.resume_from_checkpoint and epoch == first_epoch and step < resume_step:
                if step % args.gradient_accumulation_steps == 0:
                    progress_bar.update(1)
                    if args.report_to == "wandb":
                        accelerator.print(progress_bar)
                continue

            with accelerator.accumulate(unet):
                loss = train_step(
                    batch, 
                    vae, 
                    text_encoder, 
                    unet, 
                    noise_scheduler, 
                    optimizer, 
                    lr_scheduler, 
                    accelerator, 
                    args, 
                    weight_dtype
                )

            # Checks if the accelerator has performed an optimization step behind the scenes
            if accelerator.sync_gradients:
                progress_bar.update(1)
                if args.report_to == "wandb":
                    accelerator.print(progress_bar)
                global_step += 1

                # if global_step % args.checkpointing_steps == 0:
                #     if accelerator.is_main_process:
                #         save_path = os.path.join(args.output_dir, f"checkpoint-{global_step}")
                #         accelerator.save_state(save_path)
                #         logger.info(f"Saved state to {save_path}")

            logs = {"loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0]}
            progress_bar.set_postfix(**logs)
            accelerator.log(logs, step=global_step)

            if (
                args.validation_prompt is not None
                and (step + num_update_steps_per_epoch * epoch) % args.validation_steps == 0
            ):
                run_validation(args, accelerator, epoch, step, unet, text_encoder, num_update_steps_per_epoch)

            if global_step >= args.max_train_steps:
                break
    
    # Printing the GPU memory usage details
    if not args.no_tracemalloc:
        accelerator.print(f"GPU Memory before entering the train : {b2mb(tracemalloc.begin)}")
        accelerator.print(f"GPU Memory consumed at the end of the train (end-begin): {tracemalloc.used}")
        accelerator.print(f"GPU Memory peaked during the train (max-begin): {tracemalloc.peaked}")
        accelerator.print(
            f"GPU Total Peak Memory consumed during the train (max): {tracemalloc.peaked + b2mb(tracemalloc.begin)}"
        )

        accelerator.print(f"CPU Memory before entering the train : {b2mb(tracemalloc.cpu_begin)}")
        accelerator.print(f"CPU Memory consumed at the end of the train (end-begin): {tracemalloc.cpu_used}")
        accelerator.print(f"CPU Peak Memory consumed during the train (max-begin): {tracemalloc.cpu_peaked}")
        accelerator.print(
            f"CPU Total Peak Memory consumed during the train (max): {tracemalloc.cpu_peaked + b2mb(tracemalloc.cpu_begin)}"
        )
    
    return global_step

def collect_knowledge_embeddings(text_encoder, tokenizer, prompts, device, batch_size=16):
    """
    收集知识表示的嵌入向量。
    
    注意：此函数保留为向后兼容，实际会调用collect_layer_specific_embeddings。
    
    Args:
        text_encoder: 文本编码器模型
        tokenizer: 分词器
        prompts: 表示知识的文本提示列表
        device: 计算设备
        batch_size: 批处理大小
        
    Returns:
        total_embeddings: 收集到的嵌入向量 [n_prompts, embed_dim]
    """
    logger.warning("collect_knowledge_embeddings被调用，但现在推荐使用collect_layer_specific_embeddings")
    
    # 初始化一个简单的UNet，仅用于收集embeddings
    from diffusers import UNet2DConditionModel
    
    # 创建一个临时UNet
    unet = UNet2DConditionModel(
        sample_size=64,
        in_channels=4,
        out_channels=4,
        layers_per_block=2,
        block_out_channels=(128, 128, 256, 256, 512, 512),
        down_block_types=(
            "DownBlock2D",
            "DownBlock2D",
            "DownBlock2D",
            "DownBlock2D",
            "AttnDownBlock2D",
            "DownBlock2D",
        ),
        up_block_types=(
            "UpBlock2D",
            "AttnUpBlock2D",
            "UpBlock2D",
            "UpBlock2D",
            "UpBlock2D",
            "UpBlock2D",
        ),
    )
    unet.to(device)
    
    # 调用新方法
    layer_embeddings, dimensions = collect_layer_specific_embeddings(unet, text_encoder, tokenizer, prompts, device, batch_size)
    
    # 返回第一个有效的嵌入
    for embeddings in layer_embeddings.values():
        if embeddings is not None and embeddings.numel() > 0:
            return embeddings
    
    # 如果没有有效的嵌入，抛出错误
    raise ValueError("未能收集有效的知识嵌入")

def main(args):
    logging_dir = Path(args.output_dir, args.logging_dir)

    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with=args.report_to,
        project_dir=logging_dir,
    )
    if args.report_to == "wandb":
        import wandb

        wandb.login(key=args.wandb_key)
        wandb.init(project=args.wandb_project_name)
    # Currently, it's not possible to do gradient accumulation when training two models with accelerate.accumulate
    # This will be enabled soon in accelerate. For now, we don't allow gradient accumulation when training two models.
    # TODO (patil-suraj): Remove this check when gradient accumulation with two models is enabled in accelerate.
    if args.train_text_encoder and args.gradient_accumulation_steps > 1 and accelerator.num_processes > 1:
        raise ValueError(
            "Gradient accumulation is not supported when training the text encoder in distributed training. "
            "Please set gradient_accumulation_steps to 1. This feature will be supported in the future."
        )

    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:
        datasets.utils.logging.set_verbosity_warning()
        transformers.utils.logging.set_verbosity_warning()
        diffusers.utils.logging.set_verbosity_info()
    else:
        datasets.utils.logging.set_verbosity_error()
        transformers.utils.logging.set_verbosity_error()
        diffusers.utils.logging.set_verbosity_error()

    # If passed along, set the training seed now.
    if args.seed is not None:
        set_seed(args.seed)

    # Generate class images if prior preservation is enabled.
    if args.with_prior_preservation:
        class_images_dir = Path(args.class_data_dir)
        if not class_images_dir.exists():
            class_images_dir.mkdir(parents=True)
        cur_class_images = len(list(class_images_dir.iterdir()))

        if cur_class_images < args.num_class_images:
            torch_dtype = torch.float16 if accelerator.device.type == "cuda" else torch.float32
            if args.prior_generation_precision == "fp32":
                torch_dtype = torch.float32
            elif args.prior_generation_precision == "fp16":
                torch_dtype = torch.float16
            elif args.prior_generation_precision == "bf16":
                torch_dtype = torch.bfloat16
            pipeline = DiffusionPipeline.from_pretrained(
                args.pretrained_model_name_or_path,
                torch_dtype=torch_dtype,
                safety_checker=None,
                revision=args.revision,
            )
            pipeline.set_progress_bar_config(disable=True)

            num_new_images = args.num_class_images - cur_class_images
            logger.info(f"Number of class images to sample: {num_new_images}.")

            sample_dataset = PromptDataset(args.class_prompt, num_new_images)
            sample_dataloader = torch.utils.data.DataLoader(sample_dataset, batch_size=args.sample_batch_size)

            sample_dataloader = accelerator.prepare(sample_dataloader)
            pipeline.to(accelerator.device)

            for example in tqdm(
                sample_dataloader, desc="Generating class images", disable=not accelerator.is_local_main_process
            ):
                images = pipeline(example["prompt"]).images

                for i, image in enumerate(images):
                    hash_image = hashlib.sha1(image.tobytes()).hexdigest()
                    image_filename = class_images_dir / f"{example['index'][i] + cur_class_images}-{hash_image}.jpg"
                    image.save(image_filename)

            del pipeline
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    # Handle the repository creation
    if accelerator.is_main_process:
        if args.push_to_hub:
            api = HfApi(token=args.hub_token)

            # Create repo (repo_name from args or inferred)
            repo_name = args.hub_model_id
            if repo_name is None:
                repo_name = Path(args.output_dir).absolute().name
            repo_id = api.create_repo(repo_name, exist_ok=True).repo_id

            with open(os.path.join(args.output_dir, ".gitignore"), "w+") as gitignore:
                if "step_*" not in gitignore:
                    gitignore.write("step_*\n")
                if "epoch_*" not in gitignore:
                    gitignore.write("epoch_*\n")
        elif args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)

    # Load the tokenizer
    if args.tokenizer_name:
        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name, revision=args.revision, use_fast=False)
    elif args.pretrained_model_name_or_path:
        tokenizer = AutoTokenizer.from_pretrained(
            args.pretrained_model_name_or_path,
            subfolder="tokenizer",
            revision=args.revision,
            use_fast=False,
        )

    # import correct text encoder class
    text_encoder_cls = import_model_class_from_model_name_or_path(args.pretrained_model_name_or_path, args.revision)

    # Load scheduler and models
    noise_scheduler = DDPMScheduler(
        beta_start=0.00085,
        beta_end=0.012,
        beta_schedule="scaled_linear",
        num_train_timesteps=1000,
    )  # DDPMScheduler.from_pretrained(args.pretrained_model_name_or_path, subfolder="scheduler")
    text_encoder = text_encoder_cls.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="text_encoder", revision=args.revision
    )
    vae = AutoencoderKL.from_pretrained(args.pretrained_model_name_or_path, subfolder="vae", revision=args.revision)
    unet = UNet2DConditionModel.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="unet", revision=args.revision
    )

    if args.use_lora:
        config = LoraConfig(
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            target_modules=UNET_TARGET_MODULES,
            lora_dropout=args.lora_dropout,
            bias=args.lora_bias,
        )
        unet = get_peft_model(unet, config)
        unet.print_trainable_parameters()
        # print(unet)

    vae.requires_grad_(False)
    if not args.train_text_encoder:
        text_encoder.requires_grad_(False)
    elif args.train_text_encoder and args.use_lora:
        config = LoraConfig(
            r=args.lora_text_encoder_r,
            lora_alpha=args.lora_text_encoder_alpha,
            target_modules=TEXT_ENCODER_TARGET_MODULES,
            lora_dropout=args.lora_text_encoder_dropout,
            bias=args.lora_text_encoder_bias,
        )
        text_encoder = get_peft_model(text_encoder, config)
        text_encoder.print_trainable_parameters()
        # print(text_encoder)

    # 实现AlphaEdit功能
    if hasattr(args, 'use_alphaedit') and args.use_alphaedit:
        logger.info("AlphaEdit mode enabled. Initializing projection matrices...")
        
        # 检查是否提供了知识目录
        if args.alphaedit_knowledge_dir:
            # 从知识目录收集提示词
            prompts = []
            logger.info(f"Loading knowledge prompts from {args.alphaedit_knowledge_dir}...")
            # 这里简单实现为读取文本文件，可以根据需求修改为从图像或其他资源生成提示词
            knowledge_files = list(Path(args.alphaedit_knowledge_dir).glob("*.txt"))
            for file in knowledge_files:
                with open(file, "r", encoding="utf-8") as f:
                    prompts.extend(f.read().splitlines())
        else:
            # 如果未提供知识目录，使用一组简单的通用提示词
            logger.info("No knowledge directory specified. Using default prompts.")
            prompts = [
                "A photograph of a dog",
                "A painting of a landscape",
                "A portrait of a person",
                "A sketch of a building",
                "A digital art of a futuristic city",
                # 可以根据需要添加更多提示词
            ]
        
        # 限制样本数量
        if len(prompts) > args.alphaedit_samples:
            logger.info(f"Limiting prompts to {args.alphaedit_samples} samples...")
            prompts = prompts[:args.alphaedit_samples]
        else:
            logger.info(f"Using {len(prompts)} prompts for AlphaEdit knowledge embedding...")
        
        # 确保阈值是浮点数
        try:
            threshold_value = float(args.alphaedit_threshold)
            logger.info(f"Using threshold value: {threshold_value}")
        except ValueError:
            logger.warning(f"Invalid threshold value: {args.alphaedit_threshold}, using default 1e-5")
            threshold_value = 1e-5
            
        # 收集知识嵌入
        with torch.no_grad():
            text_encoder.eval()
            # 记录当前设备信息
            if hasattr(text_encoder, 'parameters'):
                text_encoder_device = next(text_encoder.parameters()).device
                logger.info(f"Text encoder is on device: {text_encoder_device}, accelerator device: {accelerator.device}")
                
                # 确保text_encoder在正确的设备上
                if text_encoder_device != accelerator.device:
                    logger.info(f"Moving text_encoder to {accelerator.device}")
                    text_encoder = text_encoder.to(accelerator.device)
                
            try:
                # 使用改进的维度特定嵌入收集方法
                dimension_embeddings, layer_dimensions = collect_layer_specific_embeddings(
                    unet, 
                    text_encoder, 
                    tokenizer, 
                    prompts, 
                    accelerator.device,
                    batch_size=16
                )
                
                # 计算层特定的投影矩阵
                logger.info("Computing layer-specific projection matrices...")
                projection_matrices = compute_projection_matrices(
                    dimension_embeddings, 
                    layer_dimensions, 
                    threshold_value
                )
                
                # 验证得到了所有需要的投影矩阵
                valid_matrices = [p is not None for p in projection_matrices]
                if not all(valid_matrices):
                    raise ValueError(f"Some projection matrices are None: {sum(valid_matrices)}/{len(projection_matrices)}")
                
                logger.info(f"Successfully created {len(projection_matrices)} projection matrices")
                
                # 确保所有投影矩阵在正确的设备上
                projection_matrices = [matrix.to(accelerator.device) for matrix in projection_matrices]
                
                # 将投影矩阵保存到unet模型中
                unet.projection_matrices = projection_matrices
                
                # 创建层到投影矩阵的映射，提高钩子函数效率
                unet.layer_to_matrix_map = {}
                
                # 找到所有交叉注意力层
                for idx, (name, module) in enumerate(find_cross_attention_layers(unet)):
                    if idx < len(projection_matrices):
                        # 将模块映射到其投影矩阵
                        unet.layer_to_matrix_map[module] = projection_matrices[idx]
                
                logger.info(f"Created mapping for {len(unet.layer_to_matrix_map)} layers")
                
                logger.info("AlphaEdit initialization completed successfully!")
                
                print("[INFO ]All projection matrices initialized!")
                
            except Exception as e:
                logger.error(f"Error during AlphaEdit initialization: {e}")
                raise  # 重新抛出异常，中断训练过程

    if args.enable_xformers_memory_efficient_attention:
        if is_xformers_available():
            unet.enable_xformers_memory_efficient_attention()
        else:
            raise ValueError("xformers is not available. Make sure it is installed correctly")

    if args.gradient_checkpointing:
        unet.enable_gradient_checkpointing()
        # below fails when using lora so commenting it out
        if args.train_text_encoder and not args.use_lora:
            text_encoder.gradient_checkpointing_enable()

    # Enable TF32 for faster training on Ampere GPUs,
    # cf https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices
    if args.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True

    if args.scale_lr:
        args.learning_rate = (
            args.learning_rate * args.gradient_accumulation_steps * args.train_batch_size * accelerator.num_processes
        )

    # Use 8-bit Adam for lower memory usage or to fine-tune the model in 16GB GPUs
    if args.use_8bit_adam:
        try:
            import bitsandbytes as bnb
        except ImportError:
            raise ImportError(
                "To use 8-bit Adam, please install the bitsandbytes library: `pip install bitsandbytes`."
            )

        optimizer_class = bnb.optim.AdamW8bit
    else:
        optimizer_class = torch.optim.AdamW

    # Optimizer creation
    params_to_optimize = (
        itertools.chain(unet.parameters(), text_encoder.parameters()) if args.train_text_encoder else unet.parameters()
    )
    optimizer = optimizer_class(
        params_to_optimize,
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
    )

    # Dataset and DataLoaders creation:
    train_dataset = DreamBoothDataset(
        instance_data_root=args.instance_data_dir,
        instance_prompt=args.instance_prompt,
        class_data_root=args.class_data_dir if args.with_prior_preservation else None,
        class_prompt=args.class_prompt,
        tokenizer=tokenizer,
        size=args.resolution,
        center_crop=args.center_crop,
    )

    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.train_batch_size,
        shuffle=True,
        collate_fn=lambda examples: collate_fn(examples, args.with_prior_preservation),
        num_workers=args.num_dataloader_workers,
    )

    # Scheduler and math around the number of training steps.
    overrode_max_train_steps = False
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
        overrode_max_train_steps = True

    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps * args.gradient_accumulation_steps,
        num_training_steps=args.max_train_steps * args.gradient_accumulation_steps,
        num_cycles=args.lr_num_cycles,
        power=args.lr_power,
    )

    # Prepare everything with our `accelerator`.
    if args.train_text_encoder:
        unet, text_encoder, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
            unet, text_encoder, optimizer, train_dataloader, lr_scheduler
        )
    else:
        unet, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
            unet, optimizer, train_dataloader, lr_scheduler
        )

    # For mixed precision training we cast the text_encoder and vae weights to half-precision
    # as these models are only used for inference, keeping weights in full precision is not required.
    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    # Move vae and text_encoder to device and cast to weight_dtype
    vae.to(accelerator.device, dtype=weight_dtype)
    if not args.train_text_encoder:
        text_encoder.to(accelerator.device, dtype=weight_dtype)

    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if overrode_max_train_steps:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    # Afterwards we recalculate our number of training epochs
    args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.
    if accelerator.is_main_process:
        accelerator.init_trackers("dreambooth", config=vars(args))

    # Train!
    total_batch_size = args.train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num batches each epoch = {len(train_dataloader)}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")
    global_step = 0
    first_epoch = 0

    # Potentially load in the weights and states from a previous save
    if args.resume_from_checkpoint:
        if args.resume_from_checkpoint != "latest":
            path = os.path.basename(args.resume_from_checkpoint)
        else:
            # Get the mos recent checkpoint
            dirs = os.listdir(args.output_dir)
            dirs = [d for d in dirs if d.startswith("checkpoint")]
            dirs = sorted(dirs, key=lambda x: int(x.split("-")[1]))
            path = dirs[-1]
        accelerator.print(f"Resuming from checkpoint {path}")
        accelerator.load_state(os.path.join(args.output_dir, path))
        global_step = int(path.split("-")[1])

        resume_global_step = global_step * args.gradient_accumulation_steps
        first_epoch = resume_global_step // num_update_steps_per_epoch
        resume_step = resume_global_step % num_update_steps_per_epoch

    # Only show the progress bar once on each machine.
    progress_bar = tqdm(range(global_step, args.max_train_steps), disable=not accelerator.is_local_main_process)
    progress_bar.set_description("Steps")

    # 初始化resume_step变量，避免未定义错误
    resume_step = 0

    for epoch in range(first_epoch, args.num_train_epochs):
        global_step = train_epoch(
            epoch=epoch,
            train_dataloader=train_dataloader,
            unet=unet,
            text_encoder=text_encoder,
            vae=vae,
            noise_scheduler=noise_scheduler,
            optimizer=optimizer,
            lr_scheduler=lr_scheduler,
            accelerator=accelerator,
            args=args,
            weight_dtype=weight_dtype,
            global_step=global_step,
            progress_bar=progress_bar,
            num_update_steps_per_epoch=num_update_steps_per_epoch,
            first_epoch=first_epoch,
            resume_step=resume_step
        )
        
        if global_step >= args.max_train_steps:
            break

    # Create the pipeline using using the trained modules and save it.
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        if args.use_lora:
            unwarpped_unet = accelerator.unwrap_model(unet)
            unwarpped_unet.save_pretrained(
                os.path.join(args.output_dir, "unet"), state_dict=accelerator.get_state_dict(unet)
            )
            if args.train_text_encoder:
                unwarpped_text_encoder = accelerator.unwrap_model(text_encoder)
                unwarpped_text_encoder.save_pretrained(
                    os.path.join(args.output_dir, "text_encoder"),
                    state_dict=accelerator.get_state_dict(text_encoder),
                )
        else:
            pipeline = DiffusionPipeline.from_pretrained(
                args.pretrained_model_name_or_path,
                unet=accelerator.unwrap_model(unet),
                text_encoder=accelerator.unwrap_model(text_encoder),
                revision=args.revision,
            )
            pipeline.save_pretrained(args.output_dir)

        if args.push_to_hub:
            api.upload_folder(
                repo_id=repo_id,
                folder_path=args.output_dir,
                commit_message="End of training",
                run_as_future=True,
            )

    accelerator.end_training()


if __name__ == "__main__":
    args = parse_args()
    main(args)
