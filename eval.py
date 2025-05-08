import os
import torch
import numpy as np
from PIL import Image
from tqdm import tqdm
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics.pairwise import cosine_similarity

from diffusers import StableDiffusionPipeline
from peft import PeftModel
import clip
from torchvision.models import vit_s_16

# 从inference.py导入已有的函数
from inference import get_lora_sd_pipeline

# 标准化提示词列表 - 这里需要根据论文中的25个标准化提示词进行补充
STANDARD_PROMPTS = [
    "a [V] in the snow",
    "a [V] on the beach",
    "a [V] on a marble table",
    "a [V] in a forest",
    "a [V] in the desert",
    # 补充更多提示词...
]

class SubjectDataset(Dataset):
    def __init__(self, subject_dir, transform=None):
        self.image_paths = [os.path.join(subject_dir, f) for f in os.listdir(subject_dir) 
                          if f.endswith(('.jpg', '.jpeg', '.png'))]
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
    
    # 加载DINO模型 (ViT-S/16)
    dino_model = vit_s_16(pretrained=True)
    dino_model.eval()
    dino_model.to(device)
    
    return {
        "clip": clip_model,
        "dino": dino_model,
        "device": device,
        "clip_preprocess": clip_preprocess
    }

def get_dino_features(model, images, device):
    """使用DINO (ViT-S/16)提取图像特征"""
    # 这里应该根据DINO的实际实现进行调整
    features = []
    with torch.no_grad():
        for image in images:
            if isinstance(image, Image.Image):
                transform = transforms.Compose([
                    transforms.Resize((224, 224)),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ])
                image = transform(image).unsqueeze(0).to(device)
            
            feature = model(image)
            features.append(feature.cpu().numpy())
    
    return np.vstack(features)

def get_clip_image_features(model, images, preprocess, device):
    """使用CLIP提取图像特征"""
    features = []
    with torch.no_grad():
        for image in images:
            if isinstance(image, Image.Image):
                image = preprocess(image).unsqueeze(0).to(device)
            
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
    """计算两组特征之间的余弦相似度"""
    return cosine_similarity(features1, features2)

def evaluate_subject(
    lora_path, 
    subject_dir, 
    identifier_token="[V]", 
    class_word="dog",
    num_images_per_prompt=4,
    num_inference_steps=50,
    guidance_scale=7,
    adapter_name="default"
):
    """对单个主体进行评估"""
    
    # 加载模型
    models = load_models()
    device = models["device"]
    
    # 加载微调后的Stable Diffusion模型
    pipe = get_lora_sd_pipeline(
        lora_path,
        adapter_name=adapter_name
    )
    
    # 加载主体参考图像
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])
    subject_dataset = SubjectDataset(subject_dir, transform)
    subject_loader = DataLoader(subject_dataset, batch_size=1, shuffle=False)
    
    reference_images = []
    for image in subject_loader:
        reference_images.append(image.squeeze(0))
    
    # 提取参考图像的特征
    ref_dino_features = get_dino_features(models["dino"], reference_images, device)
    ref_clip_features = get_clip_image_features(models["clip"], reference_images, models["clip_preprocess"], device)
    
    results = {
        "dino_scores": [],
        "clip_i_scores": [],
        "clip_t_scores": []
    }
    
    # 对每个提示词生成图像并评估
    for prompt_template in tqdm(STANDARD_PROMPTS):
        prompt = prompt_template.replace(identifier_token, class_word)
        negative_prompt = "low quality, blurry, unfinished"
        
        generated_images = []
        for _ in range(num_images_per_prompt):
            output = pipe(prompt, num_inference_steps=num_inference_steps, 
                         guidance_scale=guidance_scale, 
                         negative_prompt=negative_prompt)
            generated_images.append(output.images[0])
        
        # 保存生成的图像
        os.makedirs(f"eval_results/{class_word}", exist_ok=True)
        for i, img in enumerate(generated_images):
            prompt_name = prompt_template.replace(identifier_token, class_word).replace(" ", "_")[:20]
            img.save(f"eval_results/{class_word}/{prompt_name}_{i}.jpg")
        
        # 提取生成图像的特征
        gen_dino_features = get_dino_features(models["dino"], generated_images, device)
        gen_clip_features = get_clip_image_features(models["clip"], generated_images, models["clip_preprocess"], device)
        
        # 计算DINO得分 (主体保真度)
        dino_sim = compute_similarity(gen_dino_features, ref_dino_features)
        dino_score = np.mean(dino_sim)
        results["dino_scores"].append(dino_score)
        
        # 计算CLIP-I得分 (主体保真度)
        clip_i_sim = compute_similarity(gen_clip_features, ref_clip_features)
        clip_i_score = np.mean(clip_i_sim)
        results["clip_i_scores"].append(clip_i_score)
        
        # 计算CLIP-T得分 (提示词匹配度)
        text_features = get_clip_text_features(models["clip"], [prompt], device)
        clip_t_sim = compute_similarity(gen_clip_features, text_features)
        clip_t_score = np.mean(clip_t_sim)
        results["clip_t_scores"].append(clip_t_score)
    
    # 计算平均得分
    avg_results = {
        "avg_dino_score": np.mean(results["dino_scores"]),
        "avg_clip_i_score": np.mean(results["clip_i_scores"]),
        "avg_clip_t_score": np.mean(results["clip_t_scores"])
    }
    
    return results, avg_results

def main():
    # 评估参数
    lora_path = "results"  # 微调模型路径
    subject_dir = "data/subject_images"  # 主体参考图像路径
    class_word = "dog"  # 主体类别词
    
    print(f"开始评估 {class_word} 的微调模型...")
    results, avg_results = evaluate_subject(
        lora_path=lora_path,
        subject_dir=subject_dir,
        class_word=class_word,
        adapter_name="dog"
    )
    
    print("\n评估结果:")
    print(f"平均 DINO 分数 (主体保真度): {avg_results['avg_dino_score']:.4f}")
    print(f"平均 CLIP-I 分数 (主体保真度): {avg_results['avg_clip_i_score']:.4f}")
    print(f"平均 CLIP-T 分数 (提示词匹配度): {avg_results['avg_clip_t_score']:.4f}")
    
    # 保存详细结果
    import json
    with open(f"eval_results/{class_word}/metrics.json", "w") as f:
        json.dump({
            "detailed": results,
            "average": avg_results
        }, f, indent=2)

if __name__ == "__main__":
    main() 