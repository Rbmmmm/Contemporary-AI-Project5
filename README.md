# EmotionMemeCLIP：多模态情感分类实验

本仓库为 **实验五：多模态情感分类** 的完整代码实现。  
项目基于 CLIP 多模态预训练模型，结合**显式语义对齐、原型推理（Prototype-based Classification）** 与 **监督式对比学习**，用于处理多模态场景下的隐含、反讽与跨模态情感表达。

## 参考项目以及仓库

本项目主要参考 **MemeCLIP: Leveraging CLIP Representations for Multimodal Meme Classification**

论文地址为：

```
https://arxiv.org/abs/2409.14703
```

代码仓库为：

```
https://github.com/SiddhantBikram/MemeCLIP
```

- 主要参考内容为 CLIP ENCODER 使用方式，以及 LINEAR PROJECTION 和残差连接部分，其余新机制均为自己实现。

## 一、环境配置

本项目基于 **Python 3.10.18**。

安装依赖：
```bash
pip install -r requirements.txt
```

主要依赖包括：

* PyTorch
* PyTorch Lightning
* CLIP
* SentenceTransformers
* yacs
* numpy / scikit-learn / matplotlib

---

## 二、代码结构说明

```
EmotionMemeCLIP/
│
├── data/                    # 数据集（不会上传）
│   ├── *.jpg                # meme 图像
│   └── *.txt                # 对应文本（以 guid 命名）
│
├── train.txt                # 训练集标签文件
├── test_without_label.txt   # 测试集（无标签）
│
├── configs.py               # 实验配置文件（所有参数与消融开关）
│
├── datasets.py              # Dataset 与 DataLoader 实现
├── collator.py              # 多模态 batch 组织逻辑
│
├── models.py                # 模型辅助函数
│   
├── EmotionCLIP.py           # 本文提出的主要模型
│  
├── main.py                  # 主入口（训练 / 验证 / 测试）
│
├── results/                 # 保存训练日志与模型权重文件夹
├── requirements.txt
└── README.md
```

---

## 三、运行流程

### 训练模型

在工作目录下直接运行：

```bash
python main.py
```

如果是在邮箱下载的代码，会放在 `code/` 文件夹下，此时请把 `code/` 当作工作目录，并把 `data/` 放到 `code/` 文件夹下，或者在 `configs.py` 里修改数据的路径。

默认行为训练并测试配置最优的模型。

## 四、更多实验配置说明

本项目通过 **配置文件切换** 的方式完成消融实验，无需修改代码。

### 多模态融合方式

```python
cfg.fusion_strategy = "early"      # early / late
cfg.fusion_mode = "multimodal"     # image_only / text_only / multimodal
cfg.fusion_alpha = 0.3
```

### 分类头与原型设置

```python
cfg.classifier_type = "prototype"  # prototype / linear / cosine
cfg.prototype_type = "multimodal"  # text / image / multimodal
```

### 对比学习开关

```python
cfg.use_contrastive = True
cfg.contrastive_weight = 0.2
```

通过上述参数可系统比较：

* 基线模型 vs 原型推理
* Early Fusion vs Late Fusion
* 单模态 vs 多模态建模
* 是否引入监督式对比学习

### vlm 增强实验

由于该实验涉及 VLM（Vision-Language Model）增强的数据处理流程，相关数据体积较大，未直接纳入 Git 仓库进行版本管理。

如需完整复现实验结果，请前往以下 Release 页面下载对应的 data/ 数据目录：

```
https://github.com/Rbmmmm/Contemporary-AI-Project5/releases/tag/v1.0-data
```

下载并解压后，请将 `data/` 文件夹放置于项目根目录下，再修改 `configs.py` 里的 `cfg.text_encoder_type` 为 `"sentence_transformer"` 运行即可。

## 五、说明

* 本仓库代码与实验报告中的模型结构与实验设置保持一致
* 训练、验证与测试流程可完整复现
* 所有实验均基于统一配置，确保对比公平性
