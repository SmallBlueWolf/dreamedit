import argparse
import os
import warnings
from pathlib import Path

from src.utils import load_config

def parse_args(input_args=None):
    parser = argparse.ArgumentParser(description="Script to train a model using DreamBooth.")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/default.yaml",
        help="Path to config file, will be merged with default config",
    )
    
    if input_args is not None:
        args = parser.parse_args(input_args)
    else:
        args = parser.parse_args()
    
    # 加载默认配置
    default_config = load_config("configs/default.yaml")
    
    # 如果指定了其他配置文件，则加载并合并配置
    if args.config != "configs/default.yaml":
        user_config = load_config(args.config)
        # 合并配置，用户配置优先级高于默认配置
        for key, value in user_config.items():
            default_config[key] = value
    
    config = default_config
    
    # 处理环境变量
    env_local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if env_local_rank != -1 and env_local_rank != config["local_rank"]:
        config["local_rank"] = env_local_rank
    
    # 验证必要的参数
    if config["pretrained_model_name_or_path"] is None:
        raise ValueError("pretrained_model_name_or_path cannot be empty")
    if config["instance_data_dir"] is None:
        raise ValueError("instance_data_dir cannot be empty")
    if config["instance_prompt"] is None:
        raise ValueError("instance_prompt cannot be empty")
    
    if config["with_prior_preservation"]:
        if config["class_data_dir"] is None:
            raise ValueError("class_data_dir must be specified when using with_prior_preservation")
        if config["class_prompt"] is None:
            raise ValueError("class_prompt must be specified when using with_prior_preservation")
    else:
        if config["class_data_dir"] is not None:
            warnings.warn("class_data_dir is not needed without with_prior_preservation")
        if config["class_prompt"] is not None:
            warnings.warn("class_prompt is not needed without with_prior_preservation")
    
    # 处理AlphaEdit相关配置
    if 'use_alphaedit' in config and config['use_alphaedit']:
        # 将alphaedit_knowledge列表转换为字符串列表
        if 'alphaedit_knowledge' in config and config['alphaedit_knowledge'] is not None:
            if isinstance(config['alphaedit_knowledge'], str):
                # 如果是单个字符串，将其转换为列表
                config['alphaedit_knowledge'] = [config['alphaedit_knowledge']]
            elif not isinstance(config['alphaedit_knowledge'], list):
                # 如果既不是字符串也不是列表，则发出警告
                warnings.warn(f"alphaedit_knowledge should be a list of strings, got {type(config['alphaedit_knowledge'])}. Converting to string list.")
                config['alphaedit_knowledge'] = [str(config['alphaedit_knowledge'])]
        
        # 验证阈值参数
        if 'alphaedit_threshold' in config:
            try:
                config['alphaedit_threshold'] = float(config['alphaedit_threshold'])
            except (ValueError, TypeError):
                warnings.warn(f"Invalid alphaedit_threshold value: {config['alphaedit_threshold']}, using default 1e-5")
                config['alphaedit_threshold'] = 1e-5
        
        # 验证样本数量参数
        if 'alphaedit_samples' in config:
            try:
                config['alphaedit_samples'] = int(config['alphaedit_samples'])
                if config['alphaedit_samples'] <= 0:
                    warnings.warn(f"alphaedit_samples must be positive, got {config['alphaedit_samples']}. Using default 1000.")
                    config['alphaedit_samples'] = 1000
            except (ValueError, TypeError):
                warnings.warn(f"Invalid alphaedit_samples value: {config['alphaedit_samples']}, using default 1000")
                config['alphaedit_samples'] = 1000
                
        # 处理严格保护模式参数
        if 'alphaedit_strict_protection' in config:
            try:
                if isinstance(config['alphaedit_strict_protection'], bool):
                    pass  # 已经是布尔值，无需转换
                elif isinstance(config['alphaedit_strict_protection'], str):
                    config['alphaedit_strict_protection'] = config['alphaedit_strict_protection'].lower() in ('true', 'yes', '1', 'y')
                else:
                    config['alphaedit_strict_protection'] = bool(config['alphaedit_strict_protection'])
            except (ValueError, TypeError):
                warnings.warn(f"Invalid alphaedit_strict_protection value: {config['alphaedit_strict_protection']}, using default False")
                config['alphaedit_strict_protection'] = False
            
            # 如果启用了严格保护模式，输出日志消息
            if config['alphaedit_strict_protection']:
                print("严格知识空间保护模式已启用，将完全保留知识空间参数不被修改")
    
    return argparse.Namespace(**config) 