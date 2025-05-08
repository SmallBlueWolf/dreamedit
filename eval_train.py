#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
批量训练脚本：遍历dataset目录下所有类别的数据，并为每个类别调用train.py进行训练
"""

import os
import subprocess
import argparse
import yaml
from pathlib import Path
from tqdm.auto import tqdm
import shutil
import json

def parse_args():
    parser = argparse.ArgumentParser(description='批量训练DreamBooth模型')
    parser.add_argument('--dataset_dir', type=str, default='dataset',
                        help='数据集根目录，包含所有主体的文件夹')
    parser.add_argument('--output_dir', type=str, default='eval_weights',
                        help='保存所有训练权重的根目录')
    parser.add_argument('--base_model', type=str, default='CompVis/stable-diffusion-v1-4',
                        help='基础模型路径')
    parser.add_argument('--subjects', type=str, nargs='+', default=None,
                        help='要训练的特定主体名称列表，不指定则训练所有主体')
    parser.add_argument('--category_filter', type=str, default=None,
                        help='仅训练特定类别的主体，例如"dog"。不指定则训练所有类别')
    parser.add_argument('--identifier_token', type=str, default='sks',
                        help='唯一标识符token，例如：sks')
    parser.add_argument('--max_train_steps', type=int, default=800,
                        help='训练步数')
    parser.add_argument('--train_batch_size', type=int, default=1,
                        help='训练批次大小')
    parser.add_argument('--learning_rate', type=float, default=1e-4,
                        help='学习率')
    parser.add_argument('--class_img_dir', type=str, default='class_img',
                        help='类别图像目录')
    parser.add_argument('--skip_existing', action='store_true',
                        help='跳过已有训练结果的主体')
    parser.add_argument('--config_template', type=str, default='configs/dreambooth.yaml',
                        help='训练配置模板文件')
    return parser.parse_args()

def load_class_map(file_path):
    """加载主体-类别映射"""
    class_map = {}
    
    with open(file_path, 'r') as f:
        content = f.read()
        
    # 提取类别部分
    classes_section = content.split('Classes')[1].split('Prompts')[0].strip()
    
    # 解析每行
    for line in classes_section.split('\n'):
        if ',' in line and not line.startswith('subject_name'):
            parts = line.strip().split(',')
            if len(parts) == 2:
                subject_name, class_name = parts
                class_map[subject_name] = class_name
    
    return class_map

def is_live_subject(subject_name, class_map):
    """判断是否为活体主体"""
    live_classes = ['cat', 'dog', 'animal', 'pet']
    class_name = class_map.get(subject_name, subject_name)
    return class_name.lower() in live_classes

def ensure_class_dir(class_img_base_dir, class_name):
    """确保类别图像目录存在"""
    class_dir = os.path.join(class_img_base_dir, class_name)
    if not os.path.exists(class_dir):
        # 如果不存在特定类别的目录，使用dog作为备用
        fallback_dir = os.path.join(class_img_base_dir, 'dog')
        if os.path.exists(fallback_dir):
            # 创建目标目录
            os.makedirs(class_dir, exist_ok=True)
            # 复制dog目录中的所有图像到目标目录
            for img_file in os.listdir(fallback_dir):
                if img_file.endswith(('.jpg', '.jpeg', '.png')):
                    shutil.copy(
                        os.path.join(fallback_dir, img_file),
                        os.path.join(class_dir, img_file)
                    )
            print(f"从dog类复制规则化图像创建了{class_name}类的规则化图像目录")
        else:
            print(f"警告：无法找到规则化图像目录 {fallback_dir}")
            return False
    return True

def create_config_file(args, subject_name, class_name, output_config_path):
    """创建训练配置文件"""
    # 加载模板配置
    with open(args.config_template, 'r') as f:
        config = yaml.safe_load(f)
    
    # 更新配置
    config['pretrained_model_name_or_path'] = args.base_model
    config['instance_data_dir'] = os.path.join(args.dataset_dir, subject_name)
    config['class_data_dir'] = os.path.join(args.class_img_dir, class_name)
    config['instance_prompt'] = f"a photo of {args.identifier_token} {class_name}"
    config['class_prompt'] = f"a photo of {class_name}"
    config['output_dir'] = os.path.join(args.output_dir, subject_name)
    config['max_train_steps'] = args.max_train_steps
    config['train_batch_size'] = args.train_batch_size
    config['learning_rate'] = args.learning_rate
    
    # 保存配置
    os.makedirs(os.path.dirname(output_config_path), exist_ok=True)
    with open(output_config_path, 'w') as f:
        yaml.dump(config, f)
    
    return output_config_path

def train_subject(args, subject_name, class_name):
    """为单个主体训练模型"""
    print(f"\n{'='*50}")
    print(f"开始训练主体: {subject_name} (类别: {class_name})")
    print(f"{'='*50}")
    
    # 确保类别图像目录存在
    if not ensure_class_dir(args.class_img_dir, class_name):
        print(f"跳过 {subject_name} - 无法创建类别图像目录")
        return False
    
    # 创建训练配置文件
    config_dir = "configs/eval_train"
    os.makedirs(config_dir, exist_ok=True)
    config_file = os.path.join(config_dir, f"{subject_name}.yaml")
    config_file = create_config_file(args, subject_name, class_name, config_file)
    
    # 构建训练命令
    train_cmd = ["python", "train.py", "--config", config_file]
    
    # 执行训练
    try:
        print(f"执行命令: {' '.join(train_cmd)}")
        result = subprocess.run(train_cmd, check=True)
        print(f"训练完成，返回码: {result.returncode}")
        return result.returncode == 0
    except subprocess.CalledProcessError as e:
        print(f"训练失败: {e}")
        return False

def main():
    args = parse_args()
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 加载类别映射
    class_map = load_class_map(os.path.join(args.dataset_dir, "prompts_and_classes.txt"))
    
    # 获取所有主体列表
    if args.subjects:
        subject_list = args.subjects
    else:
        subject_list = [d for d in os.listdir(args.dataset_dir) 
                      if os.path.isdir(os.path.join(args.dataset_dir, d)) and 
                         not d.startswith('.') and 
                         not d.startswith('__') and
                         d != 'class_data']  # 排除非主体目录
    
    # 如果指定了类别过滤器，只保留该类别的主体
    if args.category_filter:
        filtered_subjects = []
        for subject in subject_list:
            subject_class = class_map.get(subject, subject)
            if subject_class.lower() == args.category_filter.lower():
                filtered_subjects.append(subject)
        
        if not filtered_subjects:
            print(f"警告: 没有找到类别为 '{args.category_filter}' 的主体！")
            return
        
        subject_list = filtered_subjects
        print(f"已筛选出 {len(subject_list)} 个类别为 '{args.category_filter}' 的主体进行训练")
    else:
        print(f"找到 {len(subject_list)} 个主体需要训练")
    
    # 记录训练结果
    training_results = {}
    
    # 遍历主体进行训练
    for subject_name in tqdm(subject_list, desc="训练进度"):
        # 检查是否跳过已有结果
        subject_output_dir = os.path.join(args.output_dir, subject_name)
        if args.skip_existing and os.path.exists(subject_output_dir):
            print(f"跳过已有训练结果的主体: {subject_name}")
            training_results[subject_name] = {
                "subject": subject_name,
                "class": class_map.get(subject_name, subject_name),
                "status": "skipped",
                "output_dir": subject_output_dir
            }
            continue
        
        # 获取主体的类别
        class_name = class_map.get(subject_name, subject_name)
        
        # 训练主体
        success = train_subject(args, subject_name, class_name)
        
        # 记录结果
        training_results[subject_name] = {
            "subject": subject_name,
            "class": class_name,
            "status": "success" if success else "failed",
            "output_dir": os.path.join(args.output_dir, subject_name)
        }
    
    # 保存训练结果摘要
    summary_file = os.path.join(args.output_dir, "training_results.json")
    with open(summary_file, 'w') as f:
        json.dump(training_results, f, indent=2)
    
    # 打印训练结果摘要
    success_count = sum(1 for result in training_results.values() if result["status"] == "success")
    skip_count = sum(1 for result in training_results.values() if result["status"] == "skipped")
    fail_count = sum(1 for result in training_results.values() if result["status"] == "failed")
    
    print("\n" + "="*60)
    print(f"训练完成! 结果摘要:")
    print(f"总主体数: {len(training_results)}")
    print(f"成功: {success_count}")
    print(f"跳过: {skip_count}")
    print(f"失败: {fail_count}")
    print(f"详细结果已保存到: {summary_file}")
    print("="*60)

if __name__ == "__main__":
    main() 