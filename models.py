import torch
import torch.nn as nn
import torch.nn.functional as F

torch.set_default_dtype(torch.float32)


# ============================================================
# Adapter
# ============================================================
class Adapter(nn.Module):
    def __init__(self, c_in, reduction=4):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(c_in, c_in // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(c_in // reduction, c_in, bias=False),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.fc(x)


# ============================================================
# Classifiers
# ============================================================
class Classifier(nn.Module):
    def __init__(self, feat_dim, num_classes, dtype=None):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(num_classes, feat_dim, dtype=dtype))
        self.reset_parameters()

    def reset_parameters(self):
        self.weight.data.uniform_(-1, 1)
        self.weight.data = F.normalize(self.weight.data, dim=-1)

    def forward(self, x):
        raise NotImplementedError


class CosineClassifier(Classifier):
    def __init__(self, feat_dim, num_classes, scale=30, dtype=None):
        super().__init__(feat_dim, num_classes, dtype)
        self.scale = scale

    def forward(self, x):
        x = F.normalize(x, dim=-1)
        w = F.normalize(self.weight, dim=-1)
        return self.scale * F.linear(x, w)


class LinearClassifier(Classifier):
    def __init__(self, feat_dim, num_classes, dtype=None):
        super().__init__(feat_dim, num_classes, dtype)
        self.bias = nn.Parameter(torch.zeros(num_classes, dtype=dtype))

    def forward(self, x):
        return F.linear(x, self.weight, self.bias)


# ============================================================
# Linear Projection
# ============================================================
class LinearProjection(nn.Module):
    def __init__(self, input_dim, output_dim, num_layers, drop_probs):
        super().__init__()
        layers = [
            nn.Linear(input_dim, output_dim),
            nn.Dropout(p=drop_probs[0]),
        ]
        for _ in range(1, num_layers):
            layers += [
                nn.ReLU(),
                nn.Linear(output_dim, output_dim),
                nn.Dropout(p=drop_probs[0]),
            ]
        self.proj = nn.Sequential(*layers)

    def forward(self, x):
        return self.proj(x)


# ============================================================
# CLIP Text Encoder (保留原实现，供 CLIP text 用)
# ============================================================
class CLIP_Text(nn.Module):
    def __init__(self, clip_model):
        super().__init__()
        self.token_embedding = clip_model.token_embedding
        self.positional_embedding = clip_model.positional_embedding
        self.transformer = clip_model.transformer
        self.ln_final = clip_model.ln_final
        self.text_projection = clip_model.text_projection
        self.dtype = clip_model.dtype

    def forward(self, text):
        x = self.token_embedding(text).to(self.dtype)
        x = x + self.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2).type(torch.float32)
        x = self.transformer(x)
        x = x.permute(1, 0, 2)
        x = self.ln_final(x).to(self.dtype)
        x = x[torch.arange(x.shape[0]), text.argmax(dim=-1)] @ self.text_projection.to(self.dtype)
        return x
