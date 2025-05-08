import os
import torch
import numpy as np
import argparse
from PIL import Image
from tqdm.auto import tqdm
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics.pairwise import cosine_similarity
import json
import random  # 导入random用于生成随机种子

from diffusers import StableDiffusionPipeline
from peft import PeftModel
import clip
# 对于DINO特征提取，需要导入正确的模型
from torchvision.models import vision_transformer, vit_b_16, ViT_B_16_Weights
from torchvision.models._utils import IntermediateLayerGetter

# 导入torch hub以便加载DINO模型
import torch.hub

# 从inference.py导入已有的函数
from inference import get_lora_sd_pipeline

def parse_args():
    parser = argparse.ArgumentParser(description='评估微调后的DreamBooth模型')
    parser.add_argument('--lora_path', type=str, default='results', 
                        help='微调LoRA模型路径（已弃用，建议使用--weight_root）')
    parser.add_argument('--weight_root', type=str, default='eval_weights',
                        help='所有微调权重的根目录，每个主体权重应在weight_root/主体名/')
    parser.add_argument('--dataset_dir', type=str, default='dataset', 
                        help='数据集根目录，包含所有主体的文件夹')
    parser.add_argument('--subjects', type=str, nargs='+', default=None,
                        help='要评估的特定主体名称列表，不指定则评估所有主体')
    parser.add_argument('--category_filter', type=str, default=None,
                        help='仅评估特定类别的主体，例如"dog"。不指定则评估所有类别')
    parser.add_argument('--adapter_name', type=str, default='default', 
                        help='LoRA适配器名称')
    parser.add_argument('--identifier_token', type=str, default='sks', 
                        help='唯一标识符token，例如：sks')
    parser.add_argument('--images_per_prompt', type=int, default=4, 
                        help='每个提示词生成的图像数量')
    parser.add_argument('--inference_steps', type=int, default=50, 
                        help='推理步数')
    parser.add_argument('--guidance_scale', type=float, default=7.0, 
                        help='引导比例')
    parser.add_argument('--result_dir', type=str, default='eval_results',
                        help='评估结果保存目录')
    parser.add_argument('--prompts_file', type=str, default='dataset/prompts_and_classes.txt',
                        help='提示词和类别列表文件')
    parser.add_argument('--is_live_subject', action='store_true',
                        help='是否为活体对象（使用活体专用提示词）')
    parser.add_argument('--skip_existing', action='store_true',
                        help='跳过已有评估结果的主体')
    return parser.parse_args()

def load_prompts_and_classes(file_path, is_live_subject=False):
    """加载提示词和类别"""
    with open(file_path, 'r') as f:
        content = f.read()
    
    # 加载提示词列表
    if is_live_subject:
        # 寻找活体专用提示词列表
        prompt_section = content.split('Live Subject Prompts')[1].strip()
    else:
        # 寻找物体提示词列表
        prompt_section = content.split('Object Prompts')[1].split('Live Subject Prompts')[0].strip()
    
    # 提取提示词列表
    prompts_list = []
    for line in prompt_section.split('\n'):
        if "format(unique_token, class_token)" in line:
            # 提取提示词模板
            template = line.strip().split("'")[1]
            prompts_list.append(template)
    
    # 加载类别映射
    classes_section = content.split('Classes')[1].split('Prompts')[0].strip()
    classes_map = {}
    for line in classes_section.split('\n'):
        if ',' in line and not line.startswith('subject_name'):
            parts = line.strip().split(',')
            if len(parts) == 2:
                subject_name, class_name = parts
                classes_map[subject_name] = class_name
    
    return prompts_list, classes_map

def is_live_subject(subject_name, classes_map):
    """根据类别判断是否为活体主体"""
    live_classes = ['cat', 'dog', 'animal', 'pet']
    if subject_name in classes_map:
        class_name = classes_map[subject_name]
        return class_name.lower() in live_classes
    return False

class SubjectDataset(Dataset):
    def __init__(self, subject_dir, transform=None):
        self.image_paths = [os.path.join(subject_dir, f) for f in os.listdir(subject_dir) 
                          if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = Image.open(self.image_paths[idx]).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image

def load_models():
    """加载评估所需的模型"""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # 加载CLIP模型
    clip_model, clip_preprocess = clip.load("ViT-B/32", device=device)
    
    # 加载真正的DINO模型 - 使用ViT-S/16，与DreamBooth论文一致
    print("加载官方DINO ViT-S/16模型用于主体保真度评估...")
    try:
        # 使用Facebook Research的官方DINO小型ViT模型
        dino_model = torch.hub.load('facebookresearch/dino:main', 'dino_vits16')
        print("成功加载DINO ViT-S/16模型")
    except Exception as e:
        print(f"加载DINO ViT-S/16模型出错: {e}")
        print("尝试加载大型DINO模型...")
        try:
            dino_model = torch.hub.load('facebookresearch/dino:main', 'dino_vits8')
            print("成功加载DINO ViT-S/8模型")
        except Exception as e2:
            print(f"加载备用DINO模型也失败: {e2}")
            print("回退到使用标准ViT-B/16模型...")
            dino_model = vit_b_16(weights=ViT_B_16_Weights.IMAGENET1K_V1)
    
    # 确保模型处于评估模式
    dino_model.eval()
    dino_model.to(device)
    
    return {
        "clip": clip_model,
        "dino": dino_model,
        "device": device,
        "clip_preprocess": clip_preprocess
    }

def get_dino_features(model, images, device):
    """
    使用DINO ViT-S/16提取图像特征
    
    根据DreamBooth论文:
    "DINO metric: 利用ViT-S/16的自监督embedding，衡量生成图与真实主体图片间的相似度(cosine similarity)"
    
    论文表1显示RealImages DINO分数为0.774，DreamBooth(Imagen)为0.696
    """
    features = []
    # 根据DINO官方代码使用的预处理
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])
    
    with torch.no_grad():
        for image in images:
            if isinstance(image, Image.Image):
                # 转换PIL图像为张量
                img_tensor = transform(image).unsqueeze(0).to(device)
            else:
                # 如果已经是张量，确保形状正确
                img_tensor = image.unsqueeze(0) if image.dim() == 3 else image
                img_tensor = transforms.functional.resize(img_tensor, 256)
                img_tensor = transforms.functional.center_crop(img_tensor, 224)
                img_tensor = transforms.functional.normalize(
                    img_tensor, 
                    mean=[0.485, 0.456, 0.406], 
                    std=[0.229, 0.224, 0.225]
                ).to(device)
            
            # 从DINO模型获取特征 - DINO论文中使用CLS token
            feature = model(img_tensor)
            
            # 归一化特征（确保计算余弦相似度时值域正确）
            feature = feature / feature.norm(dim=-1, keepdim=True)
            
            features.append(feature.cpu().numpy())
    
    return np.vstack(features)

def get_clip_image_features(model, images, preprocess, device):
    """使用CLIP提取图像特征"""
    features = []
    with torch.no_grad():
        for image in images:
            if isinstance(image, Image.Image):
                image = preprocess(image).unsqueeze(0).to(device)
            else:
                # 如果已经是张量，确保形状正确
                image = image.unsqueeze(0) if image.dim() == 3 else image
                image = image.to(device)
            
            feature = model.encode_image(image)
            features.append(feature.cpu().numpy())
    
    return np.vstack(features)

def get_clip_text_features(model, texts, device):
    """使用CLIP提取文本特征"""
    with torch.no_grad():
        text_tokens = clip.tokenize(texts).to(device)
        text_features = model.encode_text(text_tokens)
    return text_features.cpu().numpy()

def compute_similarity(features1, features2):
    """
    计算两组特征之间的余弦相似度，使用DreamBooth论文中的方法
    
    参考DreamBooth论文中的描述:
    "计算相似度矩阵，对于每个生成的图像，找到与参考图像的最大相似度，返回这些最大相似度的平均值"
    """
    # 确保特征已经L2归一化
    if not np.allclose(np.linalg.norm(features1[0]), 1.0, atol=1e-5):
        features1 = features1 / np.linalg.norm(features1, axis=1, keepdims=True)
    if not np.allclose(np.linalg.norm(features2[0]), 1.0, atol=1e-5):
        features2 = features2 / np.linalg.norm(features2, axis=1, keepdims=True)
    
    # 计算相似度矩阵（每个生成图像与每个参考图像之间的相似度）
    similarity_matrix = cosine_similarity(features1, features2)
    
    # 对于每个生成的图像，找到与参考图像的最大相似度
    row_max = np.max(similarity_matrix, axis=1)
    
    # 返回平均最大相似度作为最终分数
    return np.mean(row_max)

def is_black_image(image, threshold=5.0):
    """
    检测图像是否为黑色图像（NSFW内容被过滤后返回的黑图）
    
    参数:
        image: PIL.Image - 需要检测的图像
        threshold: float - 亮度阈值，低于此值被视为黑图
        
    返回:
        bool - 是否为黑图
    """
    # 转换图像为numpy数组
    img_array = np.array(image)
    # 计算平均像素值（亮度）
    mean_value = np.mean(img_array)
    # 如果平均值低于阈值，认为是黑图
    return mean_value < threshold

def evaluate_subject(args, subject_name, class_word, is_live_subject_flag, models, pipe):
    """对单个主体进行评估"""
    
    # 加载提示词和类别
    prompts_templates, _ = load_prompts_and_classes(args.prompts_file, is_live_subject_flag)
    
    subject_dir = os.path.join(args.dataset_dir, subject_name)
    device = models["device"]
    
    print(f"\n评估主体: {subject_name}")
    print(f"- 主体类别: {class_word}")
    print(f"- 主体类型: {'活体' if is_live_subject_flag else '物体'}")
    print(f"- 唯一标识符: {args.identifier_token}")
    print(f"- 评估提示词数量: {len(prompts_templates)}")
    print(f"- 每个提示词生成图像数: {args.images_per_prompt}")
    
    # 加载主体参考图像
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])
    subject_dataset = SubjectDataset(subject_dir, transform)
    subject_loader = DataLoader(subject_dataset, batch_size=1, shuffle=False)
    
    print(f"找到 {len(subject_dataset)} 张参考图像")
    
    reference_images = []
    for image in subject_loader:
        reference_images.append(image.squeeze(0))
    
    # 提取参考图像的特征
    ref_dino_features = get_dino_features(models["dino"], reference_images, device)
    ref_clip_features = get_clip_image_features(models["clip"], reference_images, models["clip_preprocess"], device)
    
    results = {
        "dino_scores": [],
        "clip_i_scores": [],
        "clip_t_scores": [],
        "prompts": []
    }
    
    # 对每个提示词生成图像并评估
    # 设置进度条嵌套格式，使其在完成后不留在控制台
    prompt_progress = tqdm(prompts_templates, desc=f"处理提示词 - {subject_name}", leave=False, position=1)
    
    for prompt_template in prompt_progress:
        # 替换提示词模板中的占位符
        prompt = prompt_template.format(args.identifier_token, class_word)
        negative_prompt = "low quality, blurry, unfinished"
        
        results["prompts"].append(prompt)
        
        generated_images = []
        # 为图像生成添加内部进度条，设置leave=False确保完成后不留在控制台
        img_progress = tqdm(range(args.images_per_prompt), desc=f"生成图像", leave=False, position=2)
        
        for _ in img_progress:
            # 设置最大重试次数
            max_retries = 10
            retry_count = 0
            success = False
            
            while not success and retry_count < max_retries:
                try:
                    # 生成随机种子
                    seed = random.randint(1, 2147483647)
                    generator = torch.Generator(device=device).manual_seed(seed)
                    
                    # 使用随机种子生成图像
                    output = pipe(
                        prompt, 
                        num_inference_steps=args.inference_steps, 
                        guidance_scale=args.guidance_scale, 
                        negative_prompt=negative_prompt,
                        generator=generator
                    )
                    
                    # 获取生成的图像
                    generated_image = output.images[0]
                    
                    # 检查是否为黑色图像 (可能是NSFW内容被过滤)
                    if is_black_image(generated_image):
                        retry_count += 1
                        if retry_count < max_retries:
                            print(f"检测到可能的NSFW内容，已被过滤为黑图。更换随机种子重试 ({retry_count}/{max_retries})...")
                        continue
                    
                    # 如果不是黑图，则成功生成
                    generated_images.append(generated_image)
                    success = True
                    
                except Exception as e:
                    retry_count += 1
                    print(f"生成图像时出错: {e}")
                    if retry_count < max_retries:
                        print(f"尝试重试 ({retry_count}/{max_retries})...")
                    else:
                        print("达到最大重试次数，跳过此图像生成")
            
            # 如果所有重试都失败，记录日志
            if not success:
                print(f"警告: 经过 {max_retries} 次重试后仍无法为提示词 '{prompt}' 生成有效图像")
        
        if not generated_images:
            print(f"警告: 提示词 '{prompt}' 未能生成任何图像，跳过")
            continue
        
        # 保存生成的图像
        result_dir = os.path.join(args.result_dir, subject_name)
        os.makedirs(result_dir, exist_ok=True)
        for i, img in enumerate(generated_images):
            # 创建一个合法且可识别的文件名
            prompt_name = prompt.replace(" ", "_")[:30]
            prompt_name = ''.join(c for c in prompt_name if c.isalnum() or c in ['_', '-'])
            img.save(f"{result_dir}/{prompt_name}_{i}.jpg")
        
        # 提取生成图像的特征
        gen_dino_features = get_dino_features(models["dino"], generated_images, device)
        gen_clip_features = get_clip_image_features(models["clip"], generated_images, models["clip_preprocess"], device)
        
        # 计算DINO得分 (主体保真度) - 根据论文的描述
        # "DINO metric：利用ViT-S/16的自监督embedding，衡量生成图与真实主体图片间的相似度（cosine similarity）"
        dino_sim = compute_similarity(gen_dino_features, ref_dino_features)
        dino_score = np.mean(dino_sim)
        results["dino_scores"].append(float(dino_score))  # 确保可JSON序列化
        
        # 计算CLIP-I得分 (主体保真度)
        clip_i_sim = compute_similarity(gen_clip_features, ref_clip_features)
        clip_i_score = np.mean(clip_i_sim)
        results["clip_i_scores"].append(float(clip_i_score))
        
        # 计算CLIP-T得分 (提示词匹配度)
        text_features = get_clip_text_features(models["clip"], [prompt], device)
        clip_t_sim = compute_similarity(gen_clip_features, text_features)
        clip_t_score = np.mean(clip_t_sim)
        results["clip_t_scores"].append(float(clip_t_score))
    
    # 计算平均得分
    avg_results = {
        "avg_dino_score": float(np.mean(results["dino_scores"])) if results["dino_scores"] else 0.0,
        "avg_clip_i_score": float(np.mean(results["clip_i_scores"])) if results["clip_i_scores"] else 0.0, 
        "avg_clip_t_score": float(np.mean(results["clip_t_scores"])) if results["clip_t_scores"] else 0.0
    }
    
    # 打印结果
    print(f"\n{subject_name} 评估结果:")
    print(f"平均 DINO 分数 (主体保真度): {avg_results['avg_dino_score']:.4f}")
    print(f"平均 CLIP-I 分数 (主体保真度): {avg_results['avg_clip_i_score']:.4f}")
    print(f"平均 CLIP-T 分数 (提示词匹配度): {avg_results['avg_clip_t_score']:.4f}")
    
    # 保存结果
    import json
    result_dir = os.path.join(args.result_dir, subject_name)
    os.makedirs(result_dir, exist_ok=True)
    with open(f"{result_dir}/metrics.json", "w") as f:
        json.dump({
            "detailed": results,
            "average": avg_results,
            "config": {
                "subject_name": subject_name,
                "class_word": class_word,
                "is_live_subject": is_live_subject_flag,
                "identifier_token": args.identifier_token,
                "inference_steps": args.inference_steps,
                "guidance_scale": args.guidance_scale,
                "images_per_prompt": args.images_per_prompt
            }
        }, f, indent=2)
    
    return subject_name, avg_results

def main():
    # 解析命令行参数
    args = parse_args()
    
    # 加载提示词和类别映射
    _, classes_map = load_prompts_and_classes(args.prompts_file)
    
    # 获取所有主体列表
    if args.subjects:
        # 如果指定了特定主体，则只评估这些主体
        subject_list = args.subjects
    else:
        # 否则评估dataset目录下的所有主体
        subject_list = [d for d in os.listdir(args.dataset_dir) 
                      if os.path.isdir(os.path.join(args.dataset_dir, d)) and 
                         not d.startswith('.') and
                         not d.startswith('__')]
    
    # 如果指定了类别过滤器，只保留该类别的主体
    if args.category_filter:
        filtered_subjects = []
        for subject in subject_list:
            # 获取主体的类别
            subject_class = classes_map.get(subject, subject)
            # 如果主体类别与过滤器匹配，则保留
            if subject_class.lower() == args.category_filter.lower():
                filtered_subjects.append(subject)
        
        if not filtered_subjects:
            print(f"警告: 没有找到类别为 '{args.category_filter}' 的主体！")
            print("请检查类别名称或确保数据集中包含该类别的主体。")
            return
        
        subject_list = filtered_subjects
        print(f"已筛选出 {len(subject_list)} 个类别为 '{args.category_filter}' 的主体进行评估")
    else:
        print(f"找到 {len(subject_list)} 个主体需要评估")
    
    # 加载评估所需的模型（全局只需加载一次）
    print("加载模型...")
    models = load_models()
    
    # 创建结果目录
    os.makedirs(args.result_dir, exist_ok=True)
    
    # 存储每个主体的评估结果
    all_results = {}
    
    # 添加最外层的总进度条，显示整体评估进度
    total_progress = tqdm(total=len(subject_list), desc="总体评估进度", position=0, leave=True)
    
    # 遍历每个主体进行评估
    subject_progress = tqdm(subject_list, desc="当前评估主体", leave=False, position=1)
    for subject_name in subject_progress:
        # 检查是否跳过已有结果
        result_path = os.path.join(args.result_dir, subject_name, "metrics.json")
        if args.skip_existing and os.path.exists(result_path):
            print(f"跳过已有评估结果的主体: {subject_name}")
            # 加载已有结果
            with open(result_path, 'r') as f:
                result_data = json.load(f)
                all_results[subject_name] = result_data["average"]
            total_progress.update(1)  # 更新总进度
            continue
        
        # 确定主体的类别
        class_word = classes_map.get(subject_name, subject_name)
        
        # 判断是否为活体主体
        subject_is_live = args.is_live_subject or is_live_subject(subject_name, classes_map)
        
        # 每个主体动态加载其专属权重
        lora_path = os.path.join(args.weight_root, subject_name)
        if not os.path.exists(lora_path):
            print(f"警告: 未找到主体 {subject_name} 的权重目录: {lora_path}，跳过该主体")
            total_progress.update(1)
            continue
        print(f"加载微调后的模型 from {lora_path}...")
        try:
            pipe = get_lora_sd_pipeline(
                lora_path,
                adapter_name=args.adapter_name
            )
        except Exception as e:
            print(f"加载权重失败: {e}，跳过主体 {subject_name}")
            total_progress.update(1)
            continue
        
        # 评估主体
        try:
            _, avg_results = evaluate_subject(
                args=args,
                subject_name=subject_name,
                class_word=class_word,
                is_live_subject_flag=subject_is_live,
                models=models,
                pipe=pipe
            )
            all_results[subject_name] = avg_results
        except Exception as e:
            print(f"评估主体 {subject_name} 时出错: {e}")
            continue
        finally:
            total_progress.update(1)  # 确保无论成功与否都更新总进度
    
    # 计算所有主体的平均分数
    if all_results:
        avg_dino = np.mean([res["avg_dino_score"] for res in all_results.values()])
        avg_clip_i = np.mean([res["avg_clip_i_score"] for res in all_results.values()])
        avg_clip_t = np.mean([res["avg_clip_t_score"] for res in all_results.values()])
        
        print("\n======== 所有主体的评估结果汇总 ========")
        print(f"主体数量: {len(all_results)}")
        print(f"平均 DINO 分数 (主体保真度): {avg_dino:.4f}")
        print(f"平均 CLIP-I 分数 (主体保真度): {avg_clip_i:.4f}")
        print(f"平均 CLIP-T 分数 (提示词匹配度): {avg_clip_t:.4f}")
        
        # 保存总体结果
        with open(f"{args.result_dir}/overall_metrics.json", "w") as f:
            json.dump({
                "subject_results": all_results,
                "average": {
                    "avg_dino_score": float(avg_dino),
                    "avg_clip_i_score": float(avg_clip_i),
                    "avg_clip_t_score": float(avg_clip_t)
                },
                "config": vars(args)
            }, f, indent=2)

if __name__ == "__main__":
    main() 