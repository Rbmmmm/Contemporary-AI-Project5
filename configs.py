import os
from yacs.config import CfgNode as CN

cfg = CN()

# ============================================================
# 1. Paths & Experiment I/O (⚠️ rarely changed)
# ============================================================

cfg.root_dir = "."

# Dataset paths
cfg.data_dir   = os.path.join(cfg.root_dir, "data")                 # *.jpg / *.txt
cfg.train_file = os.path.join(cfg.root_dir, "train.txt")
cfg.test_file  = os.path.join(cfg.root_dir, "test_without_label.txt")

# Output & checkpoint
cfg.output_dir      = "results"
cfg.checkpoint_dir  = os.path.join(cfg.output_dir, "checkpoints")
cfg.checkpoint_file = os.path.join(cfg.checkpoint_dir, "model.ckpt")

# For test-only mode
cfg.test_only  = False
cfg.model_path = ""          # path to checkpoint (used only if test_only=True)
cfg.test_dir   = ""          # directory to save test predictions

# ============================================================
# 2. Basic Experiment Settings (⚠️ rarely changed)
# ============================================================

cfg.model_type = "emotionclip"
cfg.name         = "EmotionMemeCLIP"
cfg.dataset_name = "Emotion"
cfg.task         = "emotion_classification"

cfg.seed   = 42
cfg.device = "cuda"
cfg.gpus   = [0]

# ============================================================
# 3. Backbone & Encoder Configuration (⚠️ rarely changed)
# ============================================================

# CLIP backbone
cfg.clip_variant = "ViT-L/14"
cfg.image_size   = 224

# Text encoder selection
# options: ["clip", "sentence_transformer"]
cfg.text_encoder_type = "clip"

# Only used when text_encoder_type == "sentence_transformer"
cfg.sentence_model = "paraphrase-mpnet-base-v2"

# ============================================================
# 4. Dataset & Label Space (⚠️ fixed by task)
# ============================================================

# NOTE: yacs does not support dict, use index as label id
cfg.class_names = ["negative", "neutral", "positive"]
cfg.num_classes = len(cfg.class_names)

# ============================================================
# 5. Dataloader Settings (🟡 occasionally tuned)
# ============================================================

cfg.batch_size   = 16
cfg.num_workers  = 4
cfg.pin_memory   = True

# Train / validation split
cfg.val_ratio = 0.2     # 20% validation split from train.txt

# ============================================================
# 6. Feature Dimensions & Embedding Space (⚠️ rarely changed)
# ============================================================

# CLIP ViT-L/14 feature dims
cfg.image_dim = 768
cfg.text_dim  = 768

# Shared embedding dimension (core MemeCLIP design)
cfg.map_dim = 1024

# Projection / adapter depth
cfg.num_mapping_layers    = 1
cfg.num_pre_output_layers = 1

# Dropout probabilities (if used)
cfg.drop_probs = [0.1, 0.3, 0.2]

# ============================================================
# 7. Multimodal Fusion & Adapter Settings (⭐ main ablation area)
# ============================================================

# Fusion strategy: ["early", "late"]
cfg.fusion_strategy = "early"

# Fusion mode: ["image_only", "text_only", "multimodal"]
cfg.fusion_mode = "multimodal"

# Fusion weight (used only in early fusion)
# z = alpha * image + (1 - alpha) * text
cfg.fusion_alpha = 0.3

# Adapter strength / residual scaling (if used)
cfg.image_ratio = 0.2
cfg.text_ratio  = 0.2

# ============================================================
# 8. Prototype & Classification Head (⭐ main ablation area)
# ============================================================

# Classifier type: ["prototype", "linear", "cosine"]
cfg.classifier_type = "prototype"

# Prototype modality: ["text", "image", "multimodal"]
cfg.prototype_type = "multimodal"

# ============================================================
# 9. Contrastive Learning (⭐ main ablation area)
# ============================================================

cfg.use_contrastive = True

# NT-Xent / InfoNCE temperature
cfg.temperature = 0.07

# Contrastive loss weight (λ)
cfg.contrastive_weight = 0.2

# ============================================================
# 10. Data Augmentation (🟡 optional exploration)
# ============================================================

cfg.data_augmentation = False

# ============================================================
# 11. Optimization & Training (🟡 occasionally tuned)
# ============================================================

cfg.optimizer     = "adamw"
cfg.lr            = 1e-4
cfg.weight_decay  = 1e-4
cfg.max_epochs    = 5

# ============================================================
# 12. Logging & Debug (⚠️ rarely changed)
# ============================================================

cfg.print_model = True
cfg.save_best   = True
cfg.log_interval = 50
