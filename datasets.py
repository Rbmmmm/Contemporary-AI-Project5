import os
import re
import torch
from torch.utils.data import Dataset
from PIL import Image
import clip
from sentence_transformers import SentenceTransformer
from torchvision import transforms

torch.set_default_dtype(torch.float32)


# ============================================================
# Dataset: Emotion Classification (Experiment 5)
# ============================================================
class Custom_Dataset(Dataset):
    def __init__(self, cfg, split="train"):
        super().__init__()
        self.cfg = cfg
        self.split = split
        self.data_dir = cfg.data_dir
        self.use_aug = getattr(cfg, "data_augmentation", False)

        # samples: List[(guid, label_str or None)]
        self.samples = []

        # -------------------------------
        # Load annotations
        # -------------------------------
        if split in ["train", "val"]:
            split_file = cfg.train_file if split == "train" else cfg.val_file
            with open(split_file, "r") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    if line.lower().startswith("guid"):
                        continue
                    guid, label = line.split(",")
                    self.samples.append((guid, label))

        elif split == "test":
            with open(cfg.test_file, "r") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    if line.lower().startswith("guid"):
                        continue
                    guid = line.split(",")[0]
                    self.samples.append((guid, None))
        else:
            raise ValueError(f"Unknown split: {split}")

        # -------------------------------
        # Image augmentation (train only)
        # -------------------------------
        if self.use_aug and split == "train":
            self.image_transform = transforms.Compose([
                transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
                transforms.RandomHorizontalFlip(),
                transforms.ColorJitter(
                    brightness=0.2, contrast=0.2, saturation=0.2
                ),
            ])
        else:
            self.image_transform = None

    def __len__(self):
        return len(self.samples)

    # ============================================================
    # helpers
    # ============================================================
    def _load_image(self, guid):
        img_path = os.path.join(self.data_dir, f"{guid}.jpg")
        if not os.path.exists(img_path):
            raise FileNotFoundError(f"Image not found: {img_path}")
        image = Image.open(img_path).convert("RGB")

        # 🔑 图像增强（仅 train + 启用时）
        if self.image_transform is not None:
            image = self.image_transform(image)

        return image

    def _load_text_txt(self, guid):
        """原始 meme 文本：guid.txt"""
        txt_path = os.path.join(self.data_dir, f"{guid}.txt")
        if not os.path.exists(txt_path):
            return ""
        with open(txt_path, "r", encoding="utf-8", errors="ignore") as f:
            return f.read().strip()

    def _load_text_vlm(self, guid):
        """VLM 增强文本：guid.vlm.txt"""
        vlm_path = os.path.join(self.data_dir, f"{guid}.vlm.txt")
        if not os.path.exists(vlm_path):
            return ""
        with open(vlm_path, "r", encoding="utf-8", errors="ignore") as f:
            return f.read().strip()

    def _clean_text(self, text: str):
        """
        Light text cleaning for meme text.
        Applied only when data_augmentation=True.
        """
        text = text.lower()
        text = re.sub(r"http\S+", "", text)        # remove URLs
        text = re.sub(r"@\w+", "", text)            # remove mentions
        text = re.sub(r"#\w+", "", text)            # remove hashtags
        text = re.sub(r"[!?]{2,}", "!", text)        # !!! -> !
        text = re.sub(r"\s+", " ", text)             # normalize spaces
        return text.strip()

    # ============================================================
    # main getter
    # ============================================================
    def __getitem__(self, idx):
        guid, label_str = self.samples[idx]

        # -------------------------------
        # Image
        # -------------------------------
        image = self._load_image(guid)

        # -------------------------------
        # Text (encoder-dependent)
        # -------------------------------
        if self.cfg.text_encoder_type == "sentence_transformer":
            text = self._load_text_vlm(guid)
        else:  # "clip"
            text = self._load_text_txt(guid)

        # 🔑 文本清洗（仅启用增强时）
        if self.use_aug:
            text = self._clean_text(text)

        # -------------------------------
        # Label
        # -------------------------------
        if label_str is not None:
            label_id = self.cfg.class_names.index(label_str)
        else:
            label_id = -1  # test split

        return {
            "image": image,
            "text": text,
            "label": label_id,
            "guid": guid,
        }


# ============================================================
# Collator: CLIP Image + (CLIP Text | SentenceTransformer)
# ============================================================
class Custom_Collator(object):
    def __init__(self, cfg):
        self.cfg = cfg
        self.device = cfg.device

        # -------------------------------
        # CLIP (visual always enabled)
        # -------------------------------
        self.clip_model, self.clip_preprocess = clip.load(
            cfg.clip_variant, device=self.device, jit=False
        )
        self.clip_model.eval()

        # -------------------------------
        # Text encoder switch
        # -------------------------------
        if cfg.text_encoder_type == "sentence_transformer":
            self.text_encoder = SentenceTransformer(
                cfg.sentence_model, device=self.device
            )
            self.text_encoder.eval()
        elif cfg.text_encoder_type == "clip":
            self.text_encoder = None
        else:
            raise ValueError(
                f"Unknown text_encoder_type: {cfg.text_encoder_type}"
            )

    def __call__(self, batch):
        # -------------------------------
        # GUIDs & Labels
        # -------------------------------
        guids = [item["guid"] for item in batch]
        labels = torch.tensor(
            [item["label"] for item in batch], dtype=torch.long
        )

        # -------------------------------
        # Image features (CLIP visual)
        # -------------------------------
        images = [item["image"] for item in batch]
        image_inputs = torch.stack(
            [self.clip_preprocess(img) for img in images]
        ).to(self.device).type(self.clip_model.dtype)

        with torch.no_grad():
            image_features = self.clip_model.visual(image_inputs)
        image_features = image_features.float()

        # -------------------------------
        # Text features
        # -------------------------------
        texts = [item["text"] for item in batch]

        with torch.no_grad():
            if self.cfg.text_encoder_type == "sentence_transformer":
                text_features = self.text_encoder.encode(
                    texts,
                    convert_to_tensor=True,
                    normalize_embeddings=True,
                )
            else:
                text_tokens = clip.tokenize(
                    texts, truncate=True
                ).to(self.device)
                text_features = self.clip_model.encode_text(text_tokens)
                text_features = text_features.float()

        return {
            "image_features": image_features,
            "text_features": text_features,
            "labels": labels,
            "guids": guids,
        }


# ============================================================
# Loader interface
# ============================================================
def load_dataset(cfg, split="train"):
    return Custom_Dataset(cfg=cfg, split=split)
