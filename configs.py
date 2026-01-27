import os
from yacs.config import CfgNode

cfg = CfgNode()

# =============================
# Paths (Experiment 5)
# =============================
cfg.root_dir = "."
cfg.data_dir = os.path.join(cfg.root_dir, "data")   # *.jpg / *.txt
cfg.train_file = os.path.join(cfg.root_dir, "train.txt")
cfg.test_file = os.path.join(cfg.root_dir, "test_without_label.txt")

cfg.checkpoint_dir = os.path.join(cfg.root_dir, "checkpoints")
cfg.checkpoint_file = os.path.join(cfg.checkpoint_dir, "model.ckpt")

# "emotionclip", "baseline"

cfg.model_type = "emotionclip"

# =============================
# Basic Settings
# =============================
cfg.name = "EmotionMemeCLIP"
cfg.dataset_name = "Emotion"
cfg.task = "emotion_classification"

cfg.seed = 42
cfg.device = "cuda"
cfg.gpus = [0]
cfg.test_only = False

# =============================
# Backbone
# =============================
cfg.clip_variant = "ViT-L/14"
cfg.image_size = 224

# =============================
# Text Encoder Selection
# =============================

# options: ["clip", "sentence_transformer"]
cfg.text_encoder_type = "clip"

# SentenceTransformer model (only used if type == "sentence_transformer")
cfg.sentence_model = "paraphrase-mpnet-base-v2"

# =============================
# Classes (Experiment 5)
# =============================

# ⚠️ yacs 不允许 dict，这里用 list + index 约定
# label id = index in class_names
cfg.class_names = ["negative", "neutral", "positive"]
cfg.num_classes = len(cfg.class_names)

# =============================
# Dataloader
# =============================
cfg.batch_size = 16
cfg.num_workers = 4
cfg.pin_memory = True

# validation split ratio
cfg.val_ratio = 0.2   # 20% validation

# =============================
# Feature Dimensions
# =============================

# CLIP ViT-L/14 image feature dim
cfg.image_dim = 768

# CLIP text / SentenceTransformer dim
cfg.text_dim = 768

# Shared embedding space (MemeCLIP core design)
cfg.map_dim = 1024

cfg.num_mapping_layers = 1
cfg.num_pre_output_layers = 1

cfg.drop_probs = [0.1, 0.3, 0.2]

# Adapter / fusion strength
cfg.ratio = 0.2

# =============================
# Contrastive Learning (KEEP)
# =============================

cfg.use_contrastive = True

# NT-Xent / InfoNCE temperature
cfg.temperature = 0.07

# Contrastive loss weight (λ)
cfg.contrastive_weight = 0.2

# =============================
# Classification Loss
# =============================

cfg.loss_type = "cross_entropy"

# =============================
# Optimizer
# =============================
cfg.optimizer = "adamw"
cfg.lr = 1e-4
cfg.weight_decay = 1e-4
cfg.max_epochs = 1

# =============================
# Logging & Debug
# =============================
cfg.print_model = True
cfg.save_best = True
cfg.log_interval = 50

# Prototype ablation (🔥 核心对照变量)
cfg.prototype_type = "text"   # ["text", "image", "multimodal"]

cfg.test_only = False
cfg.model_path = "results/Jan26_22-41-18_lr-0.0001_bs-16/checkpoints/model-best-epoch=07-val/acc=0.8414.ckpt" 
cfg.test_dir = "results/Jan26_22-41-18_lr-0.0001_bs-16"