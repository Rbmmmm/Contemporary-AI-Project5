import pytorch_lightning as pl
import torch
import torch.nn as nn
import torchmetrics
import os


class SimpleFusionBaseline(pl.LightningModule):
    """
    Simple Multimodal Fusion Baseline

    - Image encoder : CLIP visual (features provided by collator)
    - Text encoder  : CLIP text or SentenceTransformer (features provided by collator)
    - Fusion        : feature concatenation
    - Classifier    : single linear layer
    """

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg

        # ----------------------------
        # Classifier
        # ----------------------------
        self.classifier = nn.Linear(
            cfg.image_dim + cfg.text_dim,
            cfg.num_classes
        )

        self.loss_fn = nn.CrossEntropyLoss()

        # ----------------------------
        # Metrics
        # ----------------------------
        self.acc = torchmetrics.Accuracy(
            task="multiclass",
            num_classes=cfg.num_classes
        )
        self.f1 = torchmetrics.F1Score(
            task="multiclass",
            num_classes=cfg.num_classes,
            average="macro"
        )

        # ----------------------------
        # 🔑 History for experiment logging
        # ----------------------------
        self.train_loss_history = []
        self.val_acc_history = []
        self.val_f1_history = []

        # 临时缓存（用于 epoch 统计）
        self._train_losses_epoch = []

    # ========================================================
    # Forward
    # ========================================================
    def forward(self, image_features, text_features):
        fused = torch.cat([image_features, text_features], dim=1)
        return self.classifier(fused)

    # ========================================================
    # Training
    # ========================================================
    def training_step(self, batch, batch_idx):
        logits = self(
            batch["image_features"],
            batch["text_features"]
        )
        loss = self.loss_fn(logits, batch["labels"])
        preds = logits.argmax(dim=1)

        self.log("train/loss", loss, prog_bar=False)
        self.log(
            "train/acc",
            self.acc(preds, batch["labels"]),
            prog_bar=False
        )

        # 🔑 手动记录
        self._train_losses_epoch.append(loss.detach().cpu())

        return loss

    def on_train_epoch_end(self):
        """
        Lightning 2.x 推荐做法：在 on_train_epoch_end 里汇总
        """
        if len(self._train_losses_epoch) > 0:
            avg_loss = torch.stack(self._train_losses_epoch).mean().item()
            self.train_loss_history.append(avg_loss)
            self._train_losses_epoch.clear()

    # ========================================================
    # Validation
    # ========================================================
    def validation_step(self, batch, batch_idx):
        logits = self(
            batch["image_features"],
            batch["text_features"]
        )
        preds = logits.argmax(dim=1)

        acc = self.acc(preds, batch["labels"])
        f1 = self.f1(preds, batch["labels"])

        self.log("val/acc", acc, prog_bar=True)
        self.log("val/f1", f1, prog_bar=False)

    def on_validation_epoch_end(self):
        """
        从 Lightning logger 中读取 epoch 级指标
        """
        metrics = self.trainer.callback_metrics

        if "val/acc" in metrics:
            self.val_acc_history.append(metrics["val/acc"].item())
        if "val/f1" in metrics:
            self.val_f1_history.append(metrics["val/f1"].item())

    # ========================================================
    # Optimizer
    # ========================================================
    def configure_optimizers(self):
        return torch.optim.AdamW(
            self.parameters(),
            lr=self.cfg.lr,
            weight_decay=self.cfg.weight_decay
        )

    # ========================================================
    # Test
    # ========================================================
    def on_test_start(self):
        self.test_results = []

    def test_step(self, batch, batch_idx):
        logits = self(
            batch["image_features"],
            batch["text_features"]
        )
        preds = logits.argmax(dim=1)

        for guid, pred in zip(batch["guids"], preds.tolist()):
            self.test_results.append((guid, pred))

    def on_test_end(self):
        assert hasattr(self.cfg, "test_dir"), "cfg.test_dir must be set"

        os.makedirs(self.cfg.test_dir, exist_ok=True)
        output_path = os.path.join(self.cfg.test_dir, "test_with_label.txt")

        with open(output_path, "w") as f:
            for guid, pred in self.test_results:
                label_str = self.cfg.class_names[pred]
                f.write(f"{guid},{label_str}\n")

        print(f">>> [Baseline] test_with_label.txt saved to: {output_path}")


# ============================================================
def create_baseline_model(cfg):
    return SimpleFusionBaseline(cfg)
