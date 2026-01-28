import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchmetrics
import os

from models import Adapter, CosineClassifier, LinearClassifier


# ============================================================
# SimCLR Projection Head
# ============================================================
class ProjectionHead(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.ReLU(),
            nn.Linear(out_dim, out_dim),
        )

    def forward(self, x):
        return self.net(x)


# ============================================================
# NT-Xent Loss
# ============================================================
def nt_xent_loss(image_z, text_z, temperature=0.07):
    image_z = F.normalize(image_z, dim=-1)
    text_z = F.normalize(text_z, dim=-1)

    sim = image_z @ text_z.T / temperature
    labels = torch.arange(sim.size(0), device=sim.device)

    loss_i2t = F.cross_entropy(sim, labels)
    loss_t2i = F.cross_entropy(sim.T, labels)
    return 0.5 * (loss_i2t + loss_t2i)


# ============================================================
# EmotionCLIP (Checkpoint-safe + Ablations)
# ============================================================
class EmotionCLIP(pl.LightningModule):
    """
    Ablations supported:
    - classifier_type: prototype / cosine / linear
    - use_contrastive: True / False
    """

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg

        # ----------------------------
        # Metrics (stateless, used manually)
        # ----------------------------
        self.acc = torchmetrics.Accuracy(
            task="multiclass", num_classes=cfg.num_classes
        )
        self.f1 = torchmetrics.F1Score(
            task="multiclass",
            num_classes=cfg.num_classes,
            average="macro",
        )

        # ----------------------------
        # Feature mapping
        # ----------------------------
        self.image_map = nn.Linear(cfg.image_dim, cfg.map_dim)
        self.text_map = nn.Linear(cfg.text_dim, cfg.map_dim)

        self.img_adapter = Adapter(cfg.map_dim)
        self.txt_adapter = Adapter(cfg.map_dim)

        self.image_proj = ProjectionHead(cfg.map_dim, cfg.map_dim)
        self.text_proj = ProjectionHead(cfg.map_dim, cfg.map_dim)

        self.ce_loss = nn.CrossEntropyLoss()

        # ----------------------------
        # Classifier heads
        # ----------------------------
        if cfg.classifier_type == "cosine":
            self.classifier = CosineClassifier(
                feat_dim=cfg.map_dim,
                num_classes=cfg.num_classes,
                scale=30,
            )
        elif cfg.classifier_type == "linear":
            self.classifier = LinearClassifier(
                feat_dim=cfg.map_dim,
                num_classes=cfg.num_classes,
            )
        else:
            self.classifier = None  # prototype-based

        # ----------------------------
        # Prototype buffers (checkpoint-safe)
        # ----------------------------
        self.register_buffer(
            "class_prototypes",
            torch.zeros(cfg.num_classes, cfg.map_dim),
        )
        self.register_buffer(
            "_prototype_initialized",
            torch.tensor(False),
        )

        # ----------------------------
        # History (used by utils.py)
        # ----------------------------
        self.train_loss_history = []
        self.val_acc_history = []
        self.val_f1_history = []

        self._epoch_train_losses = []

        # 🔑 validation epoch buffers
        self._epoch_val_preds = []
        self._epoch_val_labels = []

    # ========================================================
    # Encode helpers
    # ========================================================
    def _encode(self, image_features, text_features):
        
        image_features = image_features.clone()
        text_features = text_features.clone()
        
        v = self.image_map(image_features)
        t = self.text_map(text_features)

        # v = self.cfg.ratio * self.img_adapter(v) + (1 - self.cfg.ratio) * v
        # t = self.cfg.ratio * self.txt_adapter(t) + (1 - self.cfg.ratio) * t

        v = self.cfg.image_ratio * self.img_adapter(v) + (1 - self.cfg.image_ratio) * v
        t = self.cfg.text_ratio  * self.txt_adapter(t) + (1 - self.cfg.text_ratio)  * t

        v_z = self.image_proj(v)
        t_z = self.text_proj(t)
        return v_z, t_z

    def _fuse(self, v_z, t_z):
        """
        Return feature used for classification according to fusion_mode
        """
        if self.cfg.fusion_mode == "image_only":
            z = v_z
        elif self.cfg.fusion_mode == "text_only":
            z = t_z
        elif self.cfg.fusion_mode == "multimodal":
            alpha = self.cfg.fusion_alpha  # e.g. 0.5
            z = alpha * v_z + (1 - alpha) * t_z
        else:
            raise ValueError(f"Unknown fusion_mode: {self.cfg.fusion_mode}")

        return F.normalize(z, dim=-1)

    # ========================================================
    # Prototype builder
    # ========================================================
    def _build_prototypes(self, v_feats, t_feats, labels):
        prototypes = []
        for k in range(self.cfg.num_classes):
            mask = labels == k
            if not mask.any():
                proto = torch.zeros(v_feats.size(1), device=v_feats.device)
            else:
                if self.cfg.prototype_type == "text":
                    proto = t_feats[mask].mean(dim=0)
                elif self.cfg.prototype_type == "image":
                    proto = v_feats[mask].mean(dim=0)
                elif self.cfg.prototype_type == "multimodal":
                    proto = F.normalize(
                        v_feats[mask] + t_feats[mask], dim=-1
                    ).mean(dim=0)
                else:
                    raise ValueError(
                        f"Unknown prototype_type: {self.cfg.prototype_type}"
                    )
            prototypes.append(proto)

        return F.normalize(torch.stack(prototypes, dim=0), dim=-1)

    # ========================================================
    # Forward
    # ========================================================
    # def forward(self, image_features, text_features, labels=None):
    #     v_z, t_z = self._encode(image_features, text_features)
    #     v_cls = F.normalize(v_z, dim=-1)

    #     # contrastive loss
    #     if self.cfg.use_contrastive:
    #         loss_con = nt_xent_loss(v_z, t_z, self.cfg.temperature)
    #     else:
    #         loss_con = torch.tensor(0.0, device=self.device)

    #     # ----------------------------
    #     # Train / Val
    #     # ----------------------------
    #     if labels is not None:
    #         if self.cfg.classifier_type == "prototype":
    #             batch_proto = self._build_prototypes(v_z, t_z, labels)
    #             logits = v_cls @ batch_proto.T
    #         else:
    #             logits = self.classifier(v_cls)

    #         loss_cls = self.ce_loss(logits, labels)
    #         loss = loss_cls + self.cfg.contrastive_weight * loss_con
    #         preds = logits.argmax(dim=1)

    #         return {"logits": logits, "preds": preds, "loss": loss}

    #     # ----------------------------
    #     # Test
    #     # ----------------------------
    #     if self.cfg.classifier_type == "prototype":
    #         if not bool(self._prototype_initialized.item()):
    #             raise RuntimeError(
    #                 "Prototype classifier selected but prototypes "
    #                 "are not initialized in checkpoint."
    #             )
    #         logits = v_cls @ F.normalize(self.class_prototypes, dim=-1).T
    #     else:
    #         logits = self.classifier(v_cls)

    #     preds = logits.argmax(dim=1)
    #     return {"logits": logits, "preds": preds}

    def forward(self, image_features, text_features, labels=None):
        v_z, t_z = self._encode(image_features, text_features)
        z_cls = self._fuse(v_z, t_z)

        # contrastive loss
        if self.cfg.use_contrastive:
            loss_con = nt_xent_loss(v_z, t_z, self.cfg.temperature)
        else:
            loss_con = torch.tensor(0.0, device=self.device)

        # ----------------------------
        # Train / Val
        # ----------------------------
        if labels is not None:
            if self.cfg.classifier_type == "prototype":
                batch_proto = self._build_prototypes(v_z, t_z, labels)
                logits = z_cls @ batch_proto.T
            else:
                logits = self.classifier(z_cls)

            loss_cls = self.ce_loss(logits, labels)
            loss = loss_cls + self.cfg.contrastive_weight * loss_con
            preds = logits.argmax(dim=1)

            return {"logits": logits, "preds": preds, "loss": loss}

        # ----------------------------
        # Test
        # ----------------------------
        if self.cfg.classifier_type == "prototype":
            if not bool(self._prototype_initialized.item()):
                raise RuntimeError(
                    "Prototype classifier selected but prototypes "
                    "are not initialized in checkpoint."
                )
            logits = z_cls @ F.normalize(self.class_prototypes, dim=-1).T
        else:
            logits = self.classifier(z_cls)

        preds = logits.argmax(dim=1)
        return {"logits": logits, "preds": preds}

    # ========================================================
    # Training
    # ========================================================
    def training_step(self, batch, batch_idx):
        out = self(batch["image_features"], batch["text_features"], batch["labels"])
        self.log("train/loss", out["loss"], on_epoch=True)
        self.log(
            "train/acc",
            self.acc(out["preds"], batch["labels"]),
            on_epoch=True,
        )

        self._epoch_train_losses.append(out["loss"].detach())
        return out["loss"]

    def on_train_epoch_end(self):
        if self._epoch_train_losses:
            self.train_loss_history.append(
                torch.stack(self._epoch_train_losses).mean().item()
            )
            self._epoch_train_losses.clear()

        if self.cfg.classifier_type == "prototype":
            self._recompute_and_store_train_prototypes()

    # ========================================================
    # Recompute global prototypes
    # ========================================================
    @torch.no_grad()
    def _recompute_and_store_train_prototypes(self):
        trainer = self.trainer
        train_loader = trainer.train_dataloader

        self.eval()
        all_v, all_t, all_y = [], [], []

        for batch in train_loader:
            v_z, t_z = self._encode(
                batch["image_features"], batch["text_features"]
            )
            all_v.append(v_z.detach())
            all_t.append(t_z.detach())
            all_y.append(batch["labels"].detach())

        v_feats = torch.cat(all_v, dim=0)
        t_feats = torch.cat(all_t, dim=0)
        labels = torch.cat(all_y, dim=0)

        prototypes = self._build_prototypes(v_feats, t_feats, labels)
        self.class_prototypes.copy_(prototypes)
        self._prototype_initialized.fill_(True)

        self.train()

    # ========================================================
    # Validation (🔑 修复点)
    # ========================================================
    def validation_step(self, batch, batch_idx):
        out = self(batch["image_features"], batch["text_features"], batch["labels"])

        preds = out["preds"]
        labels = batch["labels"]

        # 手动缓存
        self._epoch_val_preds.append(preds.detach())
        self._epoch_val_labels.append(labels.detach())

        self.log("val/loss", out["loss"], on_epoch=True, prog_bar=True)

    def on_validation_epoch_end(self):
        if not self._epoch_val_preds:
            return

        preds = torch.cat(self._epoch_val_preds, dim=0)
        labels = torch.cat(self._epoch_val_labels, dim=0)

        acc = self.acc(preds, labels)
        f1 = self.f1(preds, labels)

        self.log("val/acc", acc, prog_bar=True)
        self.log("val/f1", f1)
        
        self.val_acc_history.append(acc.item())
        self.val_f1_history.append(f1.item())

        self._epoch_val_preds.clear()
        self._epoch_val_labels.clear()

    # ========================================================
    # Optimizer
    # ========================================================
    def configure_optimizers(self):
        return torch.optim.AdamW(
            self.parameters(),
            lr=self.cfg.lr,
            weight_decay=self.cfg.weight_decay,
        )

    # ========================================================
    # Test
    # ========================================================
    def on_test_start(self):
        self.test_results = []

    def test_step(self, batch, batch_idx):
        out = self(batch["image_features"], batch["text_features"], labels=None)
        for guid, pred in zip(batch["guids"], out["preds"].tolist()):
            self.test_results.append((guid, pred))

    def on_test_end(self):
        os.makedirs(self.cfg.test_dir, exist_ok=True)
        path = os.path.join(self.cfg.test_dir, "test_with_label.txt")
        with open(path, "w") as f:
            for guid, pred in self.test_results:
                f.write(f"{guid},{self.cfg.class_names[pred]}\n")
        print(f">>> test_with_label.txt saved to: {path}")


# ============================================================
def create_model(cfg):
    return EmotionCLIP(cfg)
