import hashlib
import itertools
import logging
import math
import os
import random
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
    
    注意：这个函数已弃用，由collect_layer_specific_embeddings替代。
    保留此函数只是为了防止老配置文件调用导致的兼容性问题。
    
    Args:
        embeddings: 知识表示的嵌入向量 [batch_size, embed_dim]
        threshold: 特征值筛选的阈值
        
    Returns:
        projection_matrix: 投影矩阵，用于将扰动投影到零空间
    """
    logger.warning("compute_projection_matrix函数已弃用，请使用collect_layer_specific_embeddings")
    
    # 创建一个简单的投影矩阵
    dim = embeddings.shape[1]
    return torch.eye(dim, device=embeddings.device)

def find_cross_attention_layers(unet):
    """
    查找UNet中所有交叉注意力层
    
    Args:
        unet: UNet模型
        
    Returns:
        ca_layers: 交叉注意力层列表 [(name, module), ...]
    """
    logger.info(f"Searching for cross-attention layers in model: {type(unet).__name__}")
    
    # 用于存储找到的交叉注意力层
    ca_layers = []
    
    # 提取所有子网络
    sub_nets = unet.named_children()
    
    # 遍历所有子网络，收集交叉注意力层
    for net in sub_nets:
        if 'up' in net[0] or 'down' in net[0]:
            for block_idx, block in enumerate(net[1]):
                if hasattr(block, 'attentions'):
                    for attn_idx, attn in enumerate(block.attentions):
                        for transformer_idx, transformer in enumerate(attn.transformer_blocks):
                            if hasattr(transformer, 'attn2'):
                                layer_name = f"{net[0]}.{block_idx}.attentions.{attn_idx}.transformer_blocks.{transformer_idx}.attn2"
                                ca_layers.append((layer_name, transformer.attn2))
                                logger.info(f"✅ Found cross-attention layer: {layer_name}")
        
        if 'mid' in net[0]:
            if hasattr(net[1], 'attentions'):
                for attn_idx, attn in enumerate(net[1].attentions):
                    for transformer_idx, transformer in enumerate(attn.transformer_blocks):
                        if hasattr(transformer, 'attn2'):
                            layer_name = f"{net[0]}.attentions.{attn_idx}.transformer_blocks.{transformer_idx}.attn2"
                            ca_layers.append((layer_name, transformer.attn2))
                            logger.info(f"✅ Found cross-attention layer: {layer_name}")
    
    logger.info(f"Found {len(ca_layers)} cross-attention layers in total")
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
        layer_embeddings: 每一层的嵌入字典，键为层名称
        layer_dimensions: 每一层的维度字典，键为层名称
    """
    # 获取UNet中所有的交叉注意力层
    ca_layers = find_cross_attention_layers(unet)
    
    if not ca_layers:
        raise ValueError("No cross-attention layers found in UNet. Cannot proceed with AlphaEdit.")
    
    logger.info(f"Collecting embeddings for {len(ca_layers)} cross-attention layers")
    
    # 收集每一层的权重维度
    layer_dimensions = {}
    for name, layer in ca_layers:
        # 检查to_v和to_k权重
        if hasattr(layer, 'to_v') and hasattr(layer.to_v, 'weight'):
            v_weight = layer.to_v.weight
            out_dim, in_dim = v_weight.shape
            layer_dimensions[f"{name}.to_v"] = (out_dim, in_dim)
            logger.info(f"Layer {name}.to_v: weight shape {v_weight.shape}")
        
        if hasattr(layer, 'to_k') and hasattr(layer.to_k, 'weight'):
            k_weight = layer.to_k.weight
            out_dim, in_dim = k_weight.shape
            layer_dimensions[f"{name}.to_k"] = (out_dim, in_dim)
            logger.info(f"Layer {name}.to_k: weight shape {k_weight.shape}")
    
    # 验证已找到所有需要的维度
    if not layer_dimensions:
        raise ValueError("Failed to detect any valid dimensions from UNet layers")
    
    logger.info(f"Detected dimensions for {len(layer_dimensions)} weight matrices")
    
    # 获取text_encoder的输出维度（通常是768用于CLIP）
    with torch.no_grad():
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
        text_input = tokenizer(
            batch_prompts,
            padding="max_length",
            max_length=tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt"
        )
        
        # 确保输入在正确的设备上
        text_input = {k: v.to(device) for k, v in text_input.items()}
        
        # 获取文本嵌入
        with torch.no_grad():
            # 确保text_encoder在正确设备上
            if next(text_encoder.parameters()).device != device:
                text_encoder = text_encoder.to(device)
                
            text_embeddings = text_encoder(text_input['input_ids'])[0].detach()
            
            # 获取每个序列的实际嵌入
            for j in range(text_embeddings.shape[0]):
                # 找到序列的实际长度（排除padding标记）
                attention_mask = text_input['attention_mask'][j]
                valid_length = attention_mask.sum().item()
                
                # 获取有效token的嵌入
                valid_embeddings = text_embeddings[j, :valid_length, :]
                all_text_embeddings.append(valid_embeddings)
    
    # 合并所有文本嵌入
    if all_text_embeddings:
        # 展平所有嵌入为一个大矩阵，每行是一个token的嵌入
        all_embs = torch.cat([emb for emb in all_text_embeddings], dim=0)
        logger.info(f"Collected {all_embs.shape[0]} token embeddings of dimension {all_embs.shape[1]}")
    else:
        raise ValueError("No text embeddings were collected!")
    
    # 为每一层计算知识空间投影矩阵
    layer_projections = {}
    
    for layer_name, (out_dim, in_dim) in layer_dimensions.items():
        logger.info(f"Computing projection matrix for layer {layer_name} with shape ({out_dim}, {in_dim})")
        
        # 对文本嵌入进行投影，或者直接使用它们
        if text_encoder_dim != in_dim:
            # 需要投影到目标维度
            projection = torch.nn.Linear(text_encoder_dim, in_dim, bias=False).to(device)
            with torch.no_grad():
                projected_embs = projection(all_embs)
                
            logger.info(f"Projected embeddings from {text_encoder_dim} to {in_dim} dimensions")
        else:
            # 维度匹配，直接使用
            projected_embs = all_embs
            logger.info(f"Using direct embeddings with {in_dim} dimensions")
        
        # 计算协方差矩阵
        with torch.no_grad():
            # 使用batch matrix multiplication代替常规矩阵乘法
            # 这样可以确保投影是在正确的特征空间中
            emb_dim = projected_embs.shape[-1]
            emb_mean = projected_embs.mean(dim=0, keepdim=True)
            centered_embs = projected_embs - emb_mean
            
            # 计算协方差矩阵 (特征维度 x 特征维度)
            product = centered_embs.T @ centered_embs
            
            # 添加一些噪声以确保数值稳定性
            product = product + torch.eye(emb_dim, device=product.device) * 1e-5
            
            # SVD分解
            try:
                U, S, V = torch.linalg.svd(product, full_matrices=False)
                
                # 选择对应最小特征值的特征向量
                total_variance = S.sum()
                var_ratio = S / total_variance
                
                # 计算要保留的维度 - 保留95%的变异，或至少保留1个维度
                cumulative_var = torch.cumsum(var_ratio, dim=0)
                k = torch.sum(cumulative_var < 0.95).item()
                k = max(1, min(k, int(0.2 * len(S))))  # 至少保留1个维度，最多保留20%
                
                logger.info(f"For layer {layer_name}, keeping {k} out of {len(S)} dimensions in null space")
                
                # 检查是否需要使用严格保护模式
                use_strict_protection = getattr(args, 'alphaedit_strict_protection', False)
                
                if use_strict_protection:
                    # 严格保护模式: 将知识空间和非知识空间完全正交分离
                    
                    # 1. 获取知识空间表示 (大特征值对应的特征向量)
                    knowledge_vectors = U[:, :len(S)-k]  # 取最大的特征值对应的特征向量
                    
                    # 2. 构建知识空间投影矩阵 (这个投影到知识空间)
                    K = knowledge_vectors @ knowledge_vectors.T
                    
                    # 3. 构建非知识空间投影矩阵 (使用I-K)
                    P = torch.eye(emb_dim, device=product.device) - K
                    
                    # 4. 确保严格正交性
                    # 做一次SVD重整
                    orthogonalized_P, _, _ = torch.linalg.svd(P, full_matrices=False)
                    P = orthogonalized_P @ orthogonalized_P.T
                    
                    logger.info(f"Created STRICT protection matrix for {layer_name} with shape {P.shape}")
                    
                    # 5. 额外保存知识空间投影矩阵，用于严格保护模式
                    layer_projections[f"{layer_name}.knowledge"] = K
                else:
                    # 标准模式：使用小特征值对应的特征向量构建投影矩阵
                    null_vectors = U[:, -k:]  # 取最小的k个特征值对应的特征向量
                    P = null_vectors @ null_vectors.T  # 投影到小特征值的子空间
                
                # 保存到字典
                layer_projections[layer_name] = P
                
                logger.info(f"Created projection matrix for {layer_name} with shape {P.shape}, " 
                           f"using {k} smallest singular values out of {len(S)}")
                
            except Exception as e:
                logger.error(f"Error computing projection matrix for {layer_name}: {e}")
                # 在出错时使用单位矩阵
                P = torch.eye(in_dim, device=device)
                layer_projections[layer_name] = P
                logger.warning(f"Using identity matrix for {layer_name}")
            
            # 额外记录一些调试信息
            if P is not None:
                logger.info(f"Projection matrix for {layer_name}: mean={P.mean().item():.6f}, std={P.std().item():.6f}")
                if torch.isnan(P).any():
                    logger.warning(f"NaN values detected in projection matrix for {layer_name}")
                    # 替换为单位矩阵
                    P = torch.eye(in_dim, device=device)
                    layer_projections[layer_name] = P
    
    return layer_projections, layer_dimensions

def compute_projection_matrices(layer_projections, layer_dimensions, threshold=1e-5):
    """
    处理每层特定的投影矩阵。
    
    Args:
        layer_projections: 每层的投影矩阵字典，键为层名称
        layer_dimensions: 每层的维度信息字典，键为层名称
        threshold: 特征值筛选的阈值（此处不再使用，保留只为了兼容性）
        
    Returns:
        projection_dict: 每层的投影矩阵字典，键为层名称
    """
    logger.info(f"Processing projection matrices for {len(layer_projections)} layers")
    
    # 直接返回已经计算好的投影矩阵字典
    return layer_projections

def train_step(batch, vae, text_encoder, unet, noise_scheduler, optimizer, lr_scheduler, accelerator, args, weight_dtype, global_step, tokenizer):
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
    
    # 如果使用AlphaEdit，并且采用直接权重修改方法
    if hasattr(args, 'use_alphaedit') and args.use_alphaedit and hasattr(unet, 'projection_dict') and hasattr(args, 'use_direct_edit') and args.use_direct_edit:
        # 构造实例提示词列表
        instance_prompts = []
        if hasattr(args, 'instance_prompt') and args.instance_prompt:
            instance_prompts.append(args.instance_prompt)
        
        # 如果我们有class_prompt，也包含进来以增强泛化性
        if hasattr(args, 'class_prompt') and args.class_prompt:
            instance_prompts.append(args.class_prompt)
        
        # 每隔N步应用一次AlphaEdit直接权重修改
        if accelerator.sync_gradients and global_step % getattr(args, 'alphaedit_direct_every', 50) == 0:
            # 设置强度因子，随着训练进度减少编辑强度
            lambda_factor = getattr(args, 'alphaedit_lambda', 0.1) * (1.0 - min(1.0, global_step / args.max_train_steps))
            regularization_factor = getattr(args, 'alphaedit_regularization', 10.0)
            
            logger.info(f"第 {global_step} 步: 应用AlphaEdit直接权重修改，lambda={lambda_factor}")
            
            # 直接修改权重
            apply_alphaedit_direct(
                unet, 
                text_encoder, 
                tokenizer, 
                instance_prompts, 
                accelerator.device, 
                unet.projection_dict, 
                lambda_factor=lambda_factor,
                regularization_factor=regularization_factor
            )
    
    # Predict the noise residual
    model_pred = unet(noisy_latents, timesteps, encoder_hidden_states).sample

    # Get the target for loss depending on the prediction type
    if noise_scheduler.config.prediction_type == "epsilon":
        target = noise
    elif noise_scheduler.config.prediction_type == "v_prediction":
        target = noise_scheduler.get_velocity(latents, noise, timesteps)
    else:
        raise ValueError(f"Unknown prediction type {noise_scheduler.config.prediction_type}")

    # 使用AlphaEdit替代PPL - 如果启用了AlphaEdit，则跳过Prior Preservation Loss
    if args.with_prior_preservation and not (hasattr(args, 'use_alphaedit') and args.use_alphaedit):
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
    resume_step,
    tokenizer
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
                    weight_dtype,
                    global_step,
                    tokenizer
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

def expand_prompt_templates(base_prompts, templates=None, num_variants=30):
    """
    为每个基础prompt生成变体，扩充prompt池
    
    Args:
        base_prompts: 基础prompt列表，例如["dog", "cat"]
        templates: 模板列表，如果为None则使用默认模板
        num_variants: 每个基础prompt生成的变体数量
        
    Returns:
        expanded_prompts: 扩充后的prompt列表
    """
    if templates is None:
        templates = [
            "a photo of {prompt}",
            "a picture of {prompt}",
            "an image of {prompt}",
            "a {prompt} in the scene",
            "the {prompt} is in the picture",
            "{prompt} in the image",
            "a {prompt} on display",
            "this is a {prompt}",
            "this is my {prompt}",
            "a close-up of {prompt}",
            "a rendering of {prompt}",
            "a painting of {prompt}",
            "a drawing of {prompt}",
            "a sketch of {prompt}",
            "a photograph of {prompt}",
            "{prompt} in the wild",
            "{prompt} in nature",
            "{prompt} in a studio setting",
            "a {prompt} outdoors",
            "a {prompt} indoors",
            "a {prompt} with natural lighting",
            "a {prompt} in artificial lighting",
            "a high quality photo of {prompt}",
            "a detailed image of {prompt}",
            "a clear picture of {prompt}",
            "a professional photo of {prompt}",
            "a colored photo of {prompt}",
            "a black and white photo of {prompt}",
            "a bright photo of {prompt}",
            "a dark photo of {prompt}",
            "a vintage photo of {prompt}",
            "a modern photo of {prompt}",
            "a studio photo of {prompt}",
            "a candid photo of {prompt}",
            "a photo of one {prompt}",
            "a photo of multiple {prompt}",
            "a macro photo of {prompt}",
            "a wide angle photo of {prompt}",
            "a photo of {prompt} from above",
            "a photo of {prompt} from below",
            "a photo of {prompt} from the side",
            "a photo of {prompt} from the front",
            "a photo of {prompt} from the back",
            "a photo of {prompt} in daylight",
            "a photo of {prompt} at night",
            "a photo of {prompt} in the morning",
            "a photo of {prompt} in the evening",
            "a photo of {prompt} in the city",
            "a photo of {prompt} in a rural area",
        ]
    
    expanded_prompts = []
    
    # 对每个基础prompt
    for prompt in base_prompts:
        # 随机选择模板生成变体
        selected_templates = random.sample(templates, min(num_variants, len(templates)))
        
        # 应用模板
        for template in selected_templates:
            expanded_prompt = template.format(prompt=prompt)
            expanded_prompts.append(expanded_prompt)
    
    # 确保返回的prompt数量合理
    logger.info(f"Generated {len(expanded_prompts)} prompt variants from {len(base_prompts)} base prompts")
    return expanded_prompts

def calculate_modifiable_params_ratio(unet, projection_matrices, layer_dimensions):
    """
    计算可修改参数占总参数的比例
    
    Args:
        unet: UNet模型
        projection_matrices: 投影矩阵列表
        layer_dimensions: 每一层的维度元组列表 (out_dim, in_dim)
        
    Returns:
        ratio: 可修改参数占总参数的比例
        modifiable_params: 可修改参数数量
        total_params: 总参数数量
    """
    # 计算总参数量
    total_params = sum(p.numel() for p in unet.parameters())
    
    # 计算可修改参数量
    modifiable_params = 0
    
    # 遍历所有投影矩阵和对应的层维度
    for idx, (P, (out_dim, in_dim)) in enumerate(zip(projection_matrices, layer_dimensions)):
        # 计算投影矩阵P的维度，获取P允许修改的维度数
        # P是out_dim x out_dim维度的矩阵
        
        # 计算P对应特征值小于阈值的特征向量数量(P的秩)
        rank = torch.linalg.matrix_rank(P).item()
        
        # 每层交叉注意力的可修改参数量 = 输出维度 * 输入维度 * (P中特征值小于阈值的比例)
        layer_modifiable = out_dim * in_dim * (rank / out_dim)
        
        modifiable_params += layer_modifiable
        
        logger.info(f"Layer {idx}: out_dim={out_dim}, in_dim={in_dim}, rank={rank}, "
                   f"modifiable parameters: {layer_modifiable}")
    
    # 计算可修改参数占总参数的比例
    ratio = modifiable_params / total_params if total_params > 0 else 0
    
    return ratio, modifiable_params, total_params

def apply_alphaedit_direct(unet, text_encoder, tokenizer, instance_prompts, device, projection_matrices, lambda_factor=0.1, regularization_factor=10.0):
    """
    直接对模型权重应用AlphaEdit修改，类似ACE-zero的方法。
    
    Args:
        unet: UNet模型
        text_encoder: 文本编码器
        tokenizer: 分词器
        instance_prompts: 实例提示词列表
        device: 计算设备
        projection_matrices: 零空间投影矩阵字典，键为层名称
        lambda_factor: 编辑强度因子
        regularization_factor: 正则化因子
        
    Returns:
        modified: 是否成功修改了模型
    """
    logger.info(f"使用直接权重修改策略应用AlphaEdit，lambda={lambda_factor}, regularization={regularization_factor}")
    
    # 检查是否启用严格保护模式
    use_strict_protection = getattr(args, 'alphaedit_strict_protection', False)
    if use_strict_protection:
        logger.info("启用严格知识空间保护模式 - 将完全保护知识空间参数不被修改")
    
    # 找到所有交叉注意力层
    ca_layers = find_cross_attention_layers(unet)
    
    # 记录原始权重，以便在失败时恢复
    original_weights = {}
    for layer_name, layer in ca_layers:
        if hasattr(layer, 'to_v') and hasattr(layer.to_v, 'weight'):
            original_weights[f"{layer_name}.to_v"] = layer.to_v.weight.detach().clone()
        if hasattr(layer, 'to_k') and hasattr(layer.to_k, 'weight'):
            original_weights[f"{layer_name}.to_k"] = layer.to_k.weight.detach().clone()
    
    try:
        # 对每个交叉注意力层应用AlphaEdit
        for layer_name, layer in ca_layers:
            # 处理to_v权重
            if hasattr(layer, 'to_v') and f"{layer_name}.to_v" in projection_matrices:
                P = projection_matrices[f"{layer_name}.to_v"]
                W_old = layer.to_v.weight.detach()
                
                # 获取输入提示词的嵌入
                embeddings_dict = {}
                for prompt in instance_prompts:
                    # 对提示进行编码
                    text_input = tokenizer(
                        prompt,
                        padding="max_length",
                        max_length=tokenizer.model_max_length,
                        truncation=True,
                        return_tensors="pt"
                    ).to(device)
                    
                    # 获取文本嵌入
                    with torch.no_grad():
                        text_embedding = text_encoder(text_input.input_ids)[0].detach()
                    
                    # 提取最后一个非填充token的嵌入作为概念嵌入
                    final_token_idx = text_input.attention_mask.sum(dim=1).item() - 2
                    concept_embedding = text_embedding[0, final_token_idx].unsqueeze(0)
                    
                    if prompt not in embeddings_dict:
                        embeddings_dict[prompt] = concept_embedding
                
                # 合并所有嵌入
                if embeddings_dict:
                    all_embeddings = torch.cat(list(embeddings_dict.values()), dim=0)
                    
                    if use_strict_protection:
                        # 严格保护模式下的权重更新
                        # 获取知识空间投影矩阵 (如果不存在则创建)
                        knowledge_key = f"{layer_name}.to_v.knowledge"
                        if f"{layer_name}.to_v.knowledge" in projection_matrices:
                            K = projection_matrices[f"{layer_name}.to_v.knowledge"]
                        else:
                            # 创建互补投影矩阵 (I - P)
                            K = torch.eye(P.shape[0], device=P.device) - P
                        
                        # 1. 计算标准梯度更新
                        context = all_embeddings
                        for_mat2 = context.T @ context
                        current_values = context @ W_old.T
                        
                        # 2. 计算期望的更新值，但只在非知识空间方向上
                        for_mat3 = current_values.T @ context - W_old @ for_mat2
                        # 确保更新仅在非知识空间
                        update_vector = lambda_factor * for_mat3 @ P  # P是投影到非知识空间
                        
                        # 3. 应用更新，但保持知识空间完全不变
                        # 首先，将当前权重分解到知识空间和非知识空间
                        knowledge_part = W_old @ K  # 知识空间部分
                        non_knowledge_part = W_old @ P  # 非知识空间部分
                        
                        # 只更新非知识空间部分
                        updated_non_knowledge = non_knowledge_part + update_vector
                        
                        # 重新组合为完整权重
                        new_weight = knowledge_part + updated_non_knowledge
                        
                        # 最后安全检查：确保知识空间部分确实没有变化
                        knowledge_diff = torch.norm(new_weight @ K - knowledge_part)
                        if knowledge_diff > 1e-5:
                            logger.warning(f"知识空间部分检测到微小变化: {knowledge_diff:.8f}，强制纠正")
                            # 强制纠正
                            new_weight = knowledge_part + (new_weight @ P)
                        
                        layer.to_v.weight = torch.nn.Parameter(new_weight)
                        logger.info(f"严格保护模式: 成功修改层 {layer_name}.to_v 的权重，同时完全保留知识空间")
                    else:
                        # 原有的标准修改方法
                        # 计算权重更新
                        context = all_embeddings  # 形状: (n_prompts, emb_dim)
                        
                        # 计算关键矩阵
                        for_mat1 = torch.eye(W_old.shape[1], dtype=torch.float, device=device)
                        for_mat2 = context.T @ context  # X^T X
                        
                        # 应用零空间投影，类似ACE-zero的方法
                        result1 = lambda_factor * for_mat2 @ P + regularization_factor * for_mat1
                        
                        # 计算当前权重下的输出
                        current_values = context @ W_old.T  # X @ W^T
                        
                        # 计算期望的更新值
                        for_mat3 = current_values.T @ context - W_old @ for_mat2  # A @ X
                        result2 = lambda_factor * for_mat3 @ P
                        
                        # 使用线性方程求解器计算更新值
                        try:
                            upd_matrix = torch.linalg.solve(
                                result1.transpose(0, 1),
                                result2.transpose(0, 1)
                            )
                            
                            # 应用更新
                            new_weight = W_old + upd_matrix.T
                            layer.to_v.weight = torch.nn.Parameter(new_weight)
                            
                            logger.info(f"成功修改层 {layer_name}.to_v 的权重")
                        except Exception as e:
                            logger.error(f"计算层 {layer_name}.to_v 的更新时出错: {e}")
                
            # 处理to_k权重 (使用类似to_v的逻辑)
            if hasattr(layer, 'to_k') and f"{layer_name}.to_k" in projection_matrices:
                P = projection_matrices[f"{layer_name}.to_k"]
                W_old = layer.to_k.weight.detach()
                
                # 获取输入提示词的嵌入 (与to_v相同)
                embeddings_dict = {}
                for prompt in instance_prompts:
                    text_input = tokenizer(
                        prompt,
                        padding="max_length",
                        max_length=tokenizer.model_max_length,
                        truncation=True,
                        return_tensors="pt"
                    ).to(device)
                    
                    with torch.no_grad():
                        text_embedding = text_encoder(text_input.input_ids)[0].detach()
                    
                    final_token_idx = text_input.attention_mask.sum(dim=1).item() - 2
                    concept_embedding = text_embedding[0, final_token_idx].unsqueeze(0)
                    
                    if prompt not in embeddings_dict:
                        embeddings_dict[prompt] = concept_embedding
                
                # 合并所有嵌入
                if embeddings_dict:
                    all_embeddings = torch.cat(list(embeddings_dict.values()), dim=0)
                    
                    if use_strict_protection:
                        # 严格保护模式 - 与to_v相同的逻辑
                        # 获取知识空间投影矩阵
                        knowledge_key = f"{layer_name}.to_k.knowledge"
                        if f"{layer_name}.to_k.knowledge" in projection_matrices:
                            K = projection_matrices[f"{layer_name}.to_k.knowledge"]
                        else:
                            # 创建互补投影矩阵
                            K = torch.eye(P.shape[0], device=P.device) - P
                        
                        # 计算标准梯度更新
                        context = all_embeddings
                        for_mat2 = context.T @ context
                        current_values = context @ W_old.T
                        
                        # 计算期望的更新值，但只在非知识空间方向上
                        for_mat3 = current_values.T @ context - W_old @ for_mat2
                        update_vector = lambda_factor * for_mat3 @ P
                        
                        # 应用更新，但保持知识空间完全不变
                        knowledge_part = W_old @ K
                        non_knowledge_part = W_old @ P
                        
                        # 只更新非知识空间部分
                        updated_non_knowledge = non_knowledge_part + update_vector
                        
                        # 重新组合
                        new_weight = knowledge_part + updated_non_knowledge
                        
                        # 安全检查
                        knowledge_diff = torch.norm(new_weight @ K - knowledge_part)
                        if knowledge_diff > 1e-5:
                            logger.warning(f"知识空间部分检测到微小变化: {knowledge_diff:.8f}，强制纠正")
                            new_weight = knowledge_part + (new_weight @ P)
                        
                        layer.to_k.weight = torch.nn.Parameter(new_weight)
                        logger.info(f"严格保护模式: 成功修改层 {layer_name}.to_k 的权重，同时完全保留知识空间")
                    else:
                        # 原有的标准修改方法
                        # 计算权重更新
                        context = all_embeddings
                        
                        # 计算关键矩阵
                        for_mat1 = torch.eye(W_old.shape[1], dtype=torch.float, device=device)
                        for_mat2 = context.T @ context
                        
                        # 应用零空间投影
                        result1 = lambda_factor * for_mat2 @ P + regularization_factor * for_mat1
                        
                        # 计算当前权重下的输出
                        current_values = context @ W_old.T
                        
                        # 计算期望的更新值
                        for_mat3 = current_values.T @ context - W_old @ for_mat2
                        result2 = lambda_factor * for_mat3 @ P
                        
                        # 使用线性方程求解器计算更新值
                        try:
                            upd_matrix = torch.linalg.solve(
                                result1.transpose(0, 1),
                                result2.transpose(0, 1)
                            )
                            
                            # 应用更新
                            new_weight = W_old + upd_matrix.T
                            layer.to_k.weight = torch.nn.Parameter(new_weight)
                            
                            logger.info(f"成功修改层 {layer_name}.to_k 的权重")
                        except Exception as e:
                            logger.error(f"计算层 {layer_name}.to_k 的更新时出错: {e}")
        
        return True
        
    except Exception as e:
        # 出错时恢复原始权重
        logger.error(f"应用AlphaEdit直接权重修改时出错: {e}")
        import traceback
        logger.error(traceback.format_exc())
        
        # 恢复原始权重
        for key, weight in original_weights.items():
            layer_name, matrix_type = key.rsplit(".", 1)
            for ca_name, layer in ca_layers:
                if ca_name == layer_name:
                    if matrix_type == "to_v" and hasattr(layer, 'to_v'):
                        layer.to_v.weight = torch.nn.Parameter(weight)
                    elif matrix_type == "to_k" and hasattr(layer, 'to_k'):
                        layer.to_k.weight = torch.nn.Parameter(weight)
        
        logger.info("已恢复原始权重")
        return False

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
        
        # 添加日志记录Stable Diffusion版本和配置信息
        logger.info(f"Working with model: {args.pretrained_model_name_or_path}")
        logger.info(f"UNet architecture type: {type(unet).__name__}")
        
        # 设置AlphaEdit直接权重修改的默认参数
        if not hasattr(args, 'use_direct_edit'):
            args.use_direct_edit = True
            logger.info("启用AlphaEdit直接权重修改策略")
        
        if not hasattr(args, 'alphaedit_direct_every'):
            args.alphaedit_direct_every = 50  # 每50步应用一次
            logger.info(f"设置AlphaEdit直接权重修改频率: 每{args.alphaedit_direct_every}步")
        
        if not hasattr(args, 'alphaedit_lambda'):
            args.alphaedit_lambda = 0.1  # 默认编辑强度
            logger.info(f"设置AlphaEdit编辑强度系数: {args.alphaedit_lambda}")
        
        if not hasattr(args, 'alphaedit_regularization'):
            args.alphaedit_regularization = 10.0  # 默认正则化强度
            logger.info(f"设置AlphaEdit正则化系数: {args.alphaedit_regularization}")
        
        # 检查是否提供了知识目录或基础prompt列表
        if hasattr(args, 'alphaedit_knowledge') and args.alphaedit_knowledge:
            # 使用提供的基础prompt列表
            base_prompts = args.alphaedit_knowledge
            logger.info(f"Using provided base knowledge prompts: {base_prompts}")
            
            # 扩充prompt变体，增加数量以覆盖更多知识空间
            prompts = expand_prompt_templates(base_prompts, num_variants=100)
            
            # 限制样本数量
            if hasattr(args, 'alphaedit_samples') and args.alphaedit_samples > 0:
                if len(prompts) > args.alphaedit_samples:
                    logger.info(f"Limiting prompts to {args.alphaedit_samples} samples...")
                    prompts = prompts[:args.alphaedit_samples]
            
            logger.info(f"Generated {len(prompts)} prompts for AlphaEdit knowledge embedding")
            
        elif args.alphaedit_knowledge_dir:
            # 从知识目录收集提示词
            prompts = []
            logger.info(f"Loading knowledge prompts from {args.alphaedit_knowledge_dir}...")
            # 这里简单实现为读取文本文件，可以根据需求修改为从图像或其他资源生成提示词
            knowledge_files = list(Path(args.alphaedit_knowledge_dir).glob("*.txt"))
            for file in knowledge_files:
                with open(file, "r", encoding="utf-8") as f:
                    prompts.extend(f.read().splitlines())
            
            # 限制样本数量
            if hasattr(args, 'alphaedit_samples') and args.alphaedit_samples > 0:
                if len(prompts) > args.alphaedit_samples:
                    logger.info(f"Limiting prompts to {args.alphaedit_samples} samples...")
                    prompts = prompts[:args.alphaedit_samples]
        else:
            # 如果未提供知识目录和知识列表，使用一组通用知识提示词
            logger.info("No knowledge directory or knowledge list specified. Using default prompts.")
            prompts = [
                "A photograph of a dog",
                "A painting of a landscape",
                "A portrait of a person",
                "A sketch of a building",
                "A digital art of a futuristic city",
                "A cat sitting on a windowsill",
                "Mountains under a clear sky",
                "A forest with tall trees",
                "A beach with waves",
                "A cityscape at night",
                "People walking in a park",
                "Birds flying in the sky",
                "Fish swimming in the ocean",
                "A red apple on a table",
                "A blue car parked on the street",
                "A cup of coffee on a desk",
                "A book on a bookshelf",
                "A person riding a bicycle",
                "Children playing in a playground",
                "A laptop computer on a desk",
                "A flower in a garden",
                "Stars in the night sky",
                "A river flowing through a valley",
                "A butterfly on a flower",
            ]
            
            # 扩充prompt变体
            prompts = expand_prompt_templates(prompts, num_variants=50)
            
            # 确保数量合适
            if hasattr(args, 'alphaedit_samples') and args.alphaedit_samples > 0:
                target_samples = min(args.alphaedit_samples, 2000)  # 确保至少有足够的样本
                while len(prompts) < target_samples:
                    # 复制已有prompt直到达到目标数量
                    prompts.extend(prompts[:min(len(prompts), target_samples - len(prompts))])
        
        logger.info(f"Using {len(prompts)} prompts for AlphaEdit knowledge embedding...")
        
        # 打印一些示例提示词，帮助调试
        if len(prompts) > 0:
            n_samples = min(5, len(prompts))
            logger.info(f"Sample prompts for debugging: {prompts[:n_samples]}")
        
        # 确保阈值是浮点数
        try:
            threshold_value = float(args.alphaedit_threshold) if hasattr(args, 'alphaedit_threshold') else 1e-5
            logger.info(f"Using threshold value: {threshold_value}")
        except ValueError:
            logger.warning(f"Invalid threshold value, using default 1e-5")
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
                logger.info("Beginning collection of layer-specific embeddings...")
                layer_projections, layer_dimensions = collect_layer_specific_embeddings(
                    unet, 
                    text_encoder, 
                    tokenizer, 
                    prompts, 
                    accelerator.device,
                    batch_size=16
                )
                
                # 投影矩阵已经在collect_layer_specific_embeddings中计算完成
                logger.info("Computing projection matrices...")
                projection_matrices = compute_projection_matrices(
                    layer_projections, 
                    layer_dimensions, 
                    threshold_value
                )
                
                # 验证得到了所有需要的投影矩阵
                if not projection_matrices:
                    raise ValueError("Failed to compute any valid projection matrices")
                
                # 记录每一层的投影矩阵信息
                for layer_name, proj_matrix in projection_matrices.items():
                    if proj_matrix is not None:
                        logger.info(f"Layer: {layer_name}, Projection matrix shape: {proj_matrix.shape}, "
                                   f"mean: {proj_matrix.mean().item():.6f}, std: {proj_matrix.std().item():.6f}")
                
                logger.info(f"Successfully created projection matrices for {len(projection_matrices)} layer weights")
                
                # 保存投影矩阵到模型中
                unet.projection_dict = projection_matrices
                
                # 告诉用户AlphaEdit将替代PPL
                if args.with_prior_preservation:
                    logger.info("AlphaEdit will be used instead of Prior Preservation Loss")
                
                logger.info("AlphaEdit initialization completed successfully!")
                
            except Exception as e:
                logger.error(f"Error during AlphaEdit initialization: {e}")
                import traceback
                logger.error(traceback.format_exc())
                
                # 如果出错，尝试禁用AlphaEdit
                logger.warning("Disabling AlphaEdit due to initialization error")
                args.use_alphaedit = False
                
                # 如果启用了prior preservation，则给出提示
                if args.with_prior_preservation:
                    logger.info("Will fall back to Prior Preservation Loss")
                else:
                    logger.warning("Training without Prior Preservation Loss or AlphaEdit - model may overfit")
                
                # 继续训练而不是终止程序
                # raise  # 取消这行注释以在错误时中断训练

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
    # 确保优化器的数值参数都是浮点数类型
    args.learning_rate = float(args.learning_rate)
    args.adam_beta1 = float(args.adam_beta1)
    args.adam_beta2 = float(args.adam_beta2)
    args.adam_weight_decay = float(args.adam_weight_decay)
    args.adam_epsilon = float(args.adam_epsilon)

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
            resume_step=resume_step,
            tokenizer=tokenizer
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
