import argparse
import os
import warnings
from pathlib import Path

from src.utils import load_config

def parse_args(input_args=None):
    parser = argparse.ArgumentParser(description="Simple example of a training script.")
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
    
    return argparse.Namespace(**config) 