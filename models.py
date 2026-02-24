"""EfficientNet-based multi-head model with optional view transformer for mammography analysis."""

import torch
import torch.nn as nn
from torchvision.models import (
    efficientnet_b0, EfficientNet_B0_Weights,
    efficientnet_v2_s, EfficientNet_V2_S_Weights,
)


class EfficientNetBackbone(nn.Module):
    def __init__(self, name="b0", pretrained=True, in_chans=1):
        super().__init__()
        if name == "b0":
            weights = EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None
            m = efficientnet_b0(weights=weights)
            feat_dim = 1280
        elif name == "v2s":
            weights = EfficientNet_V2_S_Weights.IMAGENET1K_V1 if pretrained else None
            m = efficientnet_v2_s(weights=weights)
            feat_dim = 1280
        else:
            raise ValueError("name must be 'b0' or 'v2s'")
        # adapt stem conv from 3->1
        conv = m.features[0][0]
        w = conv.weight
        m.features[0][0] = nn.Conv2d(in_chans, w.shape[0], kernel_size=3, stride=2, padding=1, bias=False)
        with torch.no_grad():
            if w.shape[1] == 3 and in_chans == 1:
                m.features[0][0].weight.copy_(w.mean(dim=1, keepdim=True))
        self.features = m.features
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.feat_dim = feat_dim

    def forward(self, x):
        x = self.features(x)
        x = self.pool(x).flatten(1)  # [B, feat_dim]
        return x


class ViewTransformer(nn.Module):
    """Transformer encoder over per-view features for exam-level aggregation.

    Replaces masked mean pooling with learned cross-view attention.
    Uses a CLS token whose output becomes the exam-level representation.

    View ID convention (default 4 views):
        0=LCC, 1=LMLO, 2=RCC, 3=RMLO
    Adjust num_view_types if your dataset has additional views.
    """

    def __init__(self, feat_dim=1280, num_heads=8, num_layers=2,
                 dropout=0.1, num_view_types=4):
        super().__init__()
        self.view_embed = nn.Embedding(num_view_types, feat_dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=feat_dim,
            nhead=num_heads,
            dim_feedforward=feat_dim * 2,
            dropout=dropout,
            batch_first=True,
            norm_first=True,  # pre-norm for training stability
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.cls_token = nn.Parameter(torch.randn(1, 1, feat_dim) * 0.02)
        self.norm = nn.LayerNorm(feat_dim)

    def forward(self, feats, view_ids, view_mask):
        """
        Args:
            feats:     (B, V, D) backbone features per view
            view_ids:  (B, V) int tensor — view type index per slot
            view_mask: (B, V) bool — True for valid views

        Returns:
            (B, D) exam-level representation from CLS token
        """
        B = feats.shape[0]

        # Add view-type positional embeddings
        feats = feats + self.view_embed(view_ids)

        # Prepend CLS token
        cls = self.cls_token.expand(B, -1, -1)
        tokens = torch.cat([cls, feats], dim=1)  # (B, 1+V, D)

        # Build padding mask: True = IGNORE for nn.TransformerEncoder
        cls_valid = torch.ones(B, 1, dtype=torch.bool, device=view_mask.device)
        pad_mask = ~torch.cat([cls_valid, view_mask], dim=1)  # (B, 1+V)

        out = self.transformer(tokens, src_key_padding_mask=pad_mask)
        return self.norm(out[:, 0])  # CLS → (B, D)


class MultiHeadNet(nn.Module):
    """Multi-head mammography model with optional view transformer.

    Args:
        backbone:    'effb0' or 'effv2s'
        pretrained:  use ImageNet weights
        in_chans:    input channels (1 for grayscale mammo)
        use_transformer: if True, use ViewTransformer for exam-level pooling
                         instead of masked mean pooling
        transformer_kwargs: dict passed to ViewTransformer
            (num_heads, num_layers, dropout, num_view_types)
    """

    def __init__(self, backbone="effv2s", pretrained=True, in_chans=1,
                 use_transformer=False, transformer_kwargs=None):
        super().__init__()
        if backbone == "effb0":
            self.backbone = EfficientNetBackbone("b0", pretrained, in_chans)
        elif backbone == "effv2s":
            self.backbone = EfficientNetBackbone("v2s", pretrained, in_chans)
        else:
            raise ValueError("unknown backbone (use 'effb0' or 'effv2s')")

        self._feat_dim = self.backbone.feat_dim
        self.heads = nn.ModuleDict()

        # Exam-level aggregation
        self.use_transformer = use_transformer
        if use_transformer:
            tkwargs = transformer_kwargs or {}
            self.view_transformer = ViewTransformer(
                feat_dim=self._feat_dim, **tkwargs
            )
        else:
            self.view_transformer = None

    @property
    def feat_dim(self):
        return self._feat_dim

    def add_class_head(self, name: str, num_classes: int):
        """Add a classification head that outputs logits [B, num_classes]."""
        head = nn.Linear(self._feat_dim, num_classes)
        nn.init.kaiming_normal_(head.weight)
        if head.bias is not None:
            nn.init.zeros_(head.bias)
        self.heads[name] = head

    def add_reg_head(self, name: str, out_dim: int = 1):
        """Add a regression head that outputs raw values [B, out_dim]."""
        head = nn.Linear(self._feat_dim, out_dim)
        nn.init.kaiming_normal_(head.weight)
        if head.bias is not None:
            nn.init.zeros_(head.bias)
        self.heads[name] = head

    def _extract_view_features(self, views, view_mask):
        """Run backbone on each valid view, return stacked features.

        Args:
            views:     (B, V, C, H, W)
            view_mask: (B, V) bool

        Returns:
            feats: (B, V, D)
        """
        B, V, C, H, W = views.shape
        feats = []
        for v in range(V):
            feats.append(self.backbone(views[:, v]))  # (B, D)
        return torch.stack(feats, dim=1)  # (B, V, D)

    def forward_exam(self, views, view_mask, view_ids=None):
        """Exam-level forward pass aggregating multiple views.

        Args:
            views:     (B, V, 1, H, W)
            view_mask: (B, V) bool — True for valid views
            view_ids:  (B, V) int  — view type indices (required if use_transformer=True)
                       Convention: 0=LCC, 1=LMLO, 2=RCC, 3=RMLO

        Returns:
            dict of head outputs, each (B, num_outputs)
        """
        feats = self._extract_view_features(views, view_mask)  # (B, V, D)

        if self.use_transformer:
            if view_ids is None:
                raise ValueError(
                    "view_ids required when use_transformer=True. "
                    "Pass (B, V) int tensor with view type indices."
                )
            pooled = self.view_transformer(feats, view_ids, view_mask)  # (B, D)
        else:
            # Fallback: masked mean pooling
            mask_exp = view_mask.unsqueeze(-1).float()  # (B, V, 1)
            pooled = (feats * mask_exp).sum(dim=1) / mask_exp.sum(dim=1).clamp(min=1)

        return {name: head(pooled) for name, head in self.heads.items()}

    def forward(self, x, view_mask=None, view_ids=None):
        """Forward pass — dispatches on input dimensionality.

        4D (B, C, H, W):        single-image / aux mode.
        5D (B, V, C, H, W):     exam mode — requires view_mask (B, V).
        """
        if x.dim() == 5:
            return self.forward_exam(x, view_mask, view_ids=view_ids)
        z = self.backbone(x)  # [B, D]
        return {name: head(z) for name, head in self.heads.items()}


def build_model(backbone="effv2s", in_chans=1, pretrained=True, gpus=[0],
                cat_specs=None, reg_specs=None, risk_specs=None,
                use_transformer=False, transformer_kwargs=None):
    """Build model with optional transformer aggregation.

    Args:
        use_transformer: Enable ViewTransformer for exam-level predictions
        transformer_kwargs: dict with keys:
            num_heads (default 8), num_layers (default 2),
            dropout (default 0.1), num_view_types (default 4)
    """
    model = MultiHeadNet(
        backbone=backbone, in_chans=in_chans, pretrained=pretrained,
        use_transformer=use_transformer,
        transformer_kwargs=transformer_kwargs,
    )

    if cat_specs:
        for sp in cat_specs:
            model.add_class_head(sp.name, sp.num_classes)
    if reg_specs:
        for sp in reg_specs:
            model.add_reg_head(sp.name, out_dim=1)
    if risk_specs:
        for sp in risk_specs:
            model.add_reg_head(sp.name, out_dim=sp.horizons)

    device = torch.device("cuda" if torch.cuda.is_available() and len(gpus) > 0 else "cpu")
    model.to(device)
    if device.type == "cuda" and len(gpus) > 1:
        model = nn.DataParallel(model, device_ids=gpus)
    return model, device
