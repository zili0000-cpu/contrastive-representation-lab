
import copy

import torch
import torch.nn as nn
import torch.nn.functional as F

from torchvision.models import resnet18


# ResNet-18 Encoder
class ContrastiveEncoder(nn.Module):
    def __init__(self, feature_dim=128):
        super().__init__()

        self.backbone = resnet18(weights=None)

        # Adapt ResNet-18 for CIFAR-10 (32x32)
        self.backbone.conv1 = nn.Conv2d(
            3, 64, kernel_size=3,
            stride=1, padding=1, bias=False
        )
        self.backbone.maxpool = nn.Identity()

        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Identity()

        # Two-layer projection head
        self.projector = nn.Sequential(
            nn.Linear(in_features, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, feature_dim)
        )

    def forward(self, x):
        h = self.backbone(x)
        z = self.projector(h)

        # Unit-length embeddings for cosine similarity
        return F.normalize(z, dim=1)


# MoCo class
class MoCo(nn.Module):
    def __init__(
        self,
        feature_dim=128,
        queue_size=1024,
        momentum=0.999,
        temperature=0.2
    ):
        super().__init__()

        assert queue_size > 0
        assert 0 < momentum < 1
        assert temperature > 0

        self.momentum = momentum
        self.temperature = temperature
        self.queue_size = queue_size

        # 1. Two encoders
        self.encoder_q = ContrastiveEncoder(feature_dim)
        self.encoder_k = copy.deepcopy(self.encoder_q)

        # Key encoder is not updated by gradients
        for param in self.encoder_k.parameters():
            param.requires_grad = False

        # 2. Initialize normalized negative queue
        queue = torch.randn(feature_dim, queue_size)
        queue = F.normalize(queue, dim=0)

        self.register_buffer("queue", queue)
        self.register_buffer(
            "queue_ptr", torch.zeros(1, dtype=torch.long)
        )

    
    @torch.no_grad()
    def momentum_update(self):
        """Update key encoder parameters using EMA."""
        for p_q, p_k in zip(
            self.encoder_q.parameters(),
            self.encoder_k.parameters()
        ):
            p_k.mul_(self.momentum).add_(
                p_q, alpha=1.0 - self.momentum
            )
 

    @torch.no_grad()
    def enqueue_dequeue(self, keys):
        keys = keys.detach()
        batch_size = keys.shape[0]

        # Keep the most recent K keys
        if batch_size > self.queue_size:
            keys = keys[-self.queue_size:]
            batch_size = keys.shape[0]

        ptr = int(self.queue_ptr.item())
        end = ptr + batch_size

        if end <= self.queue_size:
            self.queue[:, ptr:end] = keys.T
        else:
            first = self.queue_size - ptr
            self.queue[:, ptr:] = keys[:first].T
            self.queue[:, :end - self.queue_size] = keys[first:].T

        self.queue_ptr[0] = end % self.queue_size

    
    def forward(self, im_q, im_k, update_queue=None):
    # Training: update EMA and queue by default
    # Evaluation: keep model state unchanged
        if update_queue is None:
           update_queue = self.training

        q = self.encoder_q(im_q)


        # 4. Key encoder: momentum update, no gradients
        with torch.no_grad():
            if update_queue:
                self.momentum_update()
            k = self.encoder_k(im_k)

        # 5. Positive logits: [B, 1]
        l_pos = torch.sum(q * k, dim=1, keepdim=True)

        # 6. Negative logits: [B, K]
        # Clone prevents autograd version conflicts
        l_neg = q @ self.queue.clone().detach()

        # 7. Classification logits
        logits = torch.cat([l_pos, l_neg], dim=1)
        logits = logits / self.temperature

        # Positive key is always column 0
        labels = torch.zeros(
            logits.size(0),
            dtype=torch.long,
            device=logits.device
        )

        loss = F.cross_entropy(logits, labels)

        # 8. Store current keys for future batches
        if update_queue:
            self.enqueue_dequeue(k)

        return loss, logits
