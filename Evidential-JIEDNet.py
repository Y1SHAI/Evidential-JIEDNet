from typing import List
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class EDLLoss(nn.Module):
    def __init__(
        self,
        num_classes: int,
        annealing_step: int = 10,
        evidence_activation: str = "softplus",  # Paper-3 FIX: "relu" or "softplus"
    ):
        super().__init__()
        self.num_classes = num_classes
        self.annealing_step = annealing_step
        assert evidence_activation in ["relu", "softplus"]
        self.evidence_activation = evidence_activation

    @staticmethod
    def _evidence(x: torch.Tensor, mode: str) -> torch.Tensor:
        if mode == "relu":
            return F.relu(x)
        return F.softplus(x)

    @staticmethod
    def kl_divergence(alpha: torch.Tensor, num_classes: int) -> torch.Tensor:
        """KL( Dir(alpha) || Dir(ones) )"""
        ones = torch.ones((1, num_classes), dtype=torch.float32, device=alpha.device)
        sum_alpha = torch.sum(alpha, dim=1, keepdim=True)

        first_term = (
            torch.lgamma(sum_alpha)
            - torch.lgamma(alpha).sum(dim=1, keepdim=True)
            + torch.lgamma(ones).sum(dim=1, keepdim=True)
            - torch.lgamma(ones.sum(dim=1, keepdim=True))
        )
        second_term = (
            (alpha - ones)
            .mul(torch.digamma(alpha) - torch.digamma(sum_alpha))
            .sum(dim=1, keepdim=True)
        )
        return first_term + second_term

    def forward(self, logits: torch.Tensor, target: torch.Tensor, epoch: int) -> torch.Tensor:
        logits = logits.float()

        evidence = self._evidence(logits, self.evidence_activation)
        alpha = evidence + 1.0

        if target.dim() == 1:
            y = F.one_hot(target, num_classes=self.num_classes).float()
        else:
            y = target.float()

        S = torch.sum(alpha, dim=1, keepdim=True).clamp_min(1e-8) 
        p = alpha / S

        err = torch.sum((y - p) ** 2, dim=1, keepdim=True)
        var = torch.sum(alpha * (S - alpha) / (S * S * (S + 1.0) + 1e-8), dim=1, keepdim=True) 
        loss_main = err + var

        kl_alpha = (alpha - 1.0) * (1.0 - y) + 1.0
        kl_div = self.kl_divergence(kl_alpha, self.num_classes)

        annealing_coef = min(1.0, max(0.0, float(epoch) / float(self.annealing_step)))

        return torch.mean(loss_main + annealing_coef * kl_div)

def trunc_normal_(tensor, mean=0., std=1., a=-2., b=2.):
    try:
        return nn.init.trunc_normal_(tensor, mean=mean, std=std, a=a, b=b)
    except Exception:
        with torch.no_grad():
            tensor.normal_(mean, std)
            tensor.clamp_(mean + a * std, mean + b * std)
        return tensor


class DropPath(nn.Module):
    def __init__(self, drop_prob: float = 0.0):
        super().__init__()
        self.drop_prob = float(drop_prob)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.drop_prob == 0.0 or not self.training:
            return x
        keep_prob = 1.0 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor.floor_()
        return x.div(keep_prob) * random_tensor


class LayerNorm2d(nn.Module):
    def __init__(self, num_channels: int, eps: float = 1e-6, affine: bool = True):
        super().__init__()
        self.eps = eps
        self.affine = affine
        if affine:
            self.weight = nn.Parameter(torch.ones(num_channels))
            self.bias = nn.Parameter(torch.zeros(num_channels))
        else:
            self.register_parameter("weight", None)
            self.register_parameter("bias", None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        if self.affine:
            w = self.weight.view(1, -1, 1, 1)
            b = self.bias.view(1, -1, 1, 1)
            x = x * w + b
        return x


class ConvNeXtBlock(nn.Module):
    def __init__(self, dim: int, drop_path: float = 0.0):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim)
        self.norm = LayerNorm2d(dim, eps=1e-6, affine=True)
        self.pwconv1 = nn.Conv2d(dim, 4 * dim, kernel_size=1)
        self.act = nn.GELU()
        self.pwconv2 = nn.Conv2d(4 * dim, dim, kernel_size=1)
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self._reset_parameters()

    def _reset_parameters(self):
        trunc_normal_(self.dwconv.weight, std=0.02)
        nn.init.zeros_(self.dwconv.bias)
        for m in [self.pwconv1, self.pwconv2]:
            trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shortcut = x
        x = self.dwconv(x)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        x = shortcut + self.drop_path(x)
        return x

class FilmModulator(nn.Module):
    """特征仿射调制 (FiLM)"""
    def __init__(self, tab_dim: int, channels: int, hidden: int = 256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(tab_dim, hidden), nn.GELU(),
            nn.Linear(hidden, 2 * channels)
        )
        nn.init.zeros_(self.mlp[-1].weight)
        nn.init.zeros_(self.mlp[-1].bias)

    def forward(self, fmap: torch.Tensor, tab_feat: torch.Tensor) -> torch.Tensor:
        B, C, H, W = fmap.shape
        gb = self.mlp(tab_feat)
        gamma, beta = gb[:, :C], gb[:, C:]
        gamma = gamma.view(B, C, 1, 1)
        beta = beta.view(B, C, 1, 1)
        return (1.0 + gamma) * fmap + beta


class FourierFeatures(nn.Module):
    def __init__(self, in_dim: int = 1, n_feats: int = 8):
        super().__init__()
        self.freq = nn.Parameter(torch.randn(n_feats), requires_grad=True)

    def forward(self, x: torch.Tensor):
        wx = x * self.freq.view(1, -1)
        return torch.cat([torch.sin(wx), torch.cos(wx)], dim=-1)


class RBFFeatures(nn.Module):
    def __init__(self, n_centers: int = 8):
        super().__init__()
        centers = torch.linspace(-2.0, 2.0, steps=n_centers)
        self.centers = nn.Parameter(centers.view(1, -1), requires_grad=True)
        self.log_sigma = nn.Parameter(torch.zeros(1, n_centers), requires_grad=True)

    def forward(self, x: torch.Tensor):
        sigma2 = torch.exp(2.0 * self.log_sigma) + 1e-6
        dist2 = (x - self.centers) ** 2
        return torch.exp(-dist2 / (2.0 * sigma2))


class TabularEncoder(nn.Module):
    def __init__(self, tab_embed_dim: int):
        super().__init__()
        in_dim = 1
        n_fourier = 8
        n_rbf = 8

        self.poly = nn.Sequential(nn.Linear(in_dim, 4), nn.GELU())
        self.fourier = FourierFeatures(in_dim=1, n_feats=n_fourier)
        self.rbf = RBFFeatures(n_centers=n_rbf)

        feat_dim = 4 + 2 * n_fourier + n_rbf

        self.mlp = nn.Sequential(
            nn.Linear(feat_dim, 128),
            nn.GELU(),
            nn.Dropout(0.05),
            nn.Linear(128, tab_embed_dim)
        )
        nn.init.zeros_(self.mlp[-1].weight)
        nn.init.zeros_(self.mlp[-1].bias)

    def forward(self, e_hat: torch.Tensor) -> torch.Tensor:
        f = torch.cat([self.poly(e_hat), self.fourier(e_hat), self.rbf(e_hat)], dim=-1)
        out = self.mlp(f)
        return out

class EvidentialJIEDNet(nn.Module):
    def __init__(
        self,
        num_classes: int = 6,
        depths: List[int] = [3, 3, 9, 3],
        dims: List[int] = [96, 192, 384, 768],
        drop_path_rate: float = 0.0,
        tab_embed_dim: int = 128,
        evidence_activation: str = "softplus", 
    ):
        super().__init__()
        self.dims = dims
        self.num_classes = num_classes
        assert evidence_activation in ["relu", "softplus"]
        self.evidence_activation = evidence_activation

        self.downsample_layers = nn.ModuleList()
        stem = nn.Sequential(
            nn.Conv2d(3, dims[0], kernel_size=4, stride=4),
            LayerNorm2d(dims[0], eps=1e-6, affine=True)
        )
        self.downsample_layers.append(stem)
        for i in range(3):
            self.downsample_layers.append(
                nn.Sequential(
                    LayerNorm2d(dims[i], eps=1e-6, affine=True),
                    nn.Conv2d(dims[i], dims[i + 1], kernel_size=2, stride=2)
                )
            )

        self.stages = nn.ModuleList()
        dp_rates = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        cur = 0
        for stage_idx in range(4):
            blocks = []
            for b in range(depths[stage_idx]):
                blocks.append(ConvNeXtBlock(dim=dims[stage_idx], drop_path=dp_rates[cur + b]))
            cur += depths[stage_idx]
            self.stages.append(nn.Sequential(*blocks))

        self.weight_encoder = TabularEncoder(tab_embed_dim=tab_embed_dim)

        self.film = FilmModulator(tab_dim=tab_embed_dim, channels=dims[-1])

        self.norm = nn.LayerNorm(dims[-1], eps=1e-6)
        self.head = nn.Linear(dims[-1] + tab_embed_dim, num_classes)

        self.apply(self._init_weights)

        nn.init.zeros_(self.film.mlp[-1].weight)
        nn.init.zeros_(self.film.mlp[-1].bias)
        nn.init.zeros_(self.weight_encoder.mlp[-1].weight)
        nn.init.zeros_(self.weight_encoder.mlp[-1].bias)

    def _init_weights(self, m):
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    @staticmethod
    def _evidence(x: torch.Tensor, mode: str) -> torch.Tensor:
        if mode == "relu":
            return F.relu(x)
        return F.softplus(x)

    def forward_features(self, x_img: torch.Tensor) -> torch.Tensor:
        for i in range(4):
            x_img = self.downsample_layers[i](x_img)
            x_img = self.stages[i](x_img)
        return x_img  # [B, C, H, W]

    def forward(self, x_img: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:

        if weight.dim() == 1:
            weight = weight.unsqueeze(1)

        x = self.forward_features(x_img)  # [B, C, H, W]
        w_feat = self.weight_encoder(weight)  # [B, tab_embed_dim]
        x = self.film(x, w_feat)
        x = x.mean(dim=(-2, -1))  # [B, C]
        x = self.norm(x)

        out = self.head(torch.cat([x, w_feat], dim=1))  # [B, K]
        return out

    @torch.no_grad()
    def predict_risk(self, x_img: torch.Tensor, weight: torch.Tensor) -> dict:

        if weight.dim() == 1:
            weight = weight.unsqueeze(1)

        evidence_logits = self.forward(x_img, weight)

        evidence = self._evidence(evidence_logits, self.evidence_activation)
        alpha = evidence + 1.0

        S = torch.sum(alpha, dim=1, keepdim=True).clamp_min(1e-8)
        K = evidence_logits.shape[1]
        uncertainty = (float(K) / S).squeeze(1) 

        probs = alpha / S
        conf, pred_class = torch.max(probs, dim=1) 

        return {
            "class": pred_class,      
            "confidence": conf,         
            "uncertainty": uncertainty,  
            "probs": probs,              
            "alpha": alpha, 
        }


def evidential_jied_net(num_classes: int = 6, **kwargs) -> EvidentialJIEDNet:
    return EvidentialJIEDNet(num_classes=num_classes, **kwargs)
