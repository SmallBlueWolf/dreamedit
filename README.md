# DreamEdit

基于Stable Diffusion的DreamBooth微调和训练工具。

## 安装

```bash
pip install -r requirements.txt
```

## 配置系统

DreamEdit使用YAML配置文件来管理训练参数。配置系统支持配置继承和合并：

1. 默认配置定义在`configs/default.yaml`
2. 用户可以通过`--config`参数指定自定义配置文件
3. 自定义配置会与默认配置合并，自定义配置的值会覆盖默认配置

### 配置文件示例

提供了两个示例配置文件：
- `configs/default.yaml`: 包含所有默认参数
- `configs/dreambooth.yaml`: 一个最小化配置示例，仅包含必要参数

您可以基于这些示例创建自己的配置文件：

```yaml
# 自定义配置 - my_config.yaml
pretrained_model_name_or_path: "CompVis/stable-diffusion-v1-4"   
instance_data_dir: "path/to/your/instance/images"  
class_data_dir: "path/to/your/class/images"

instance_prompt: "a photo of sks dog"  
class_prompt: "a photo of dog"  

output_dir: "results"
```

## 训练

使用以下命令开始训练：

```bash
python train.py --config configs/dreambooth.yaml
```

所有训练参数都应该在配置文件中指定，不再需要在命令行中传递其他参数。

## 主要参数说明

以下是一些重要的配置参数：

- `pretrained_model_name_or_path`: 预训练模型路径或标识符
- `instance_data_dir`: 包含实例图像的目录
- `class_data_dir`: 包含类别图像的目录
- `instance_prompt`: 包含标识符的提示文本
- `class_prompt`: 类别提示文本
- `output_dir`: 输出目录
- `use_lora`: 是否使用LoRA训练
- `train_text_encoder`: 是否训练文本编码器
- `max_train_steps`: 最大训练步数
- `learning_rate`: 学习率

完整参数列表请参考`configs/default.yaml`。 