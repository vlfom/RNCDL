import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from discovery.modeling.discovery.sinkhorn_knopp import SinkhornKnopp, SinkhornKnoppLognormalPrior


class Prototypes(nn.Module):
    def __init__(self, output_dim, num_prototypes):
        super().__init__()

        self.prototypes = nn.Linear(output_dim, num_prototypes, bias=False)

    @torch.no_grad()
    def normalize_prototypes(self):
        w = self.prototypes.weight.data.clone()
        w = F.normalize(w, dim=1, p=2)
        self.prototypes.weight.copy_(w)

    def forward(self, x):
        x = F.normalize(x)
        return self.prototypes(x)


class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_hidden_layers=1):
        super().__init__()

        if num_hidden_layers > 0:
            layers = [
                nn.Linear(input_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(inplace=True),
            ]
            for _ in range(num_hidden_layers - 1):
                layers += [
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.BatchNorm1d(hidden_dim),
                    nn.ReLU(inplace=True),
                ]
            layers.append(nn.Linear(hidden_dim, output_dim))
            self.mlp = nn.Sequential(*layers)
        else:
            self.mlp = nn.Identity()

    def forward(self, x):
        return self.mlp(x)


class MultiHead(nn.Module):
    def __init__(
            self, input_dim, hidden_dim, output_dim, num_prototypes, num_heads, num_hidden_layers=1
    ):
        super().__init__()
        self.num_heads = num_heads

        # projectors
        self.projectors = torch.nn.ModuleList(
            [MLP(input_dim, hidden_dim, output_dim, num_hidden_layers) for _ in range(num_heads)]
        )

        # prototypes
        if num_hidden_layers > 0:
            self.prototypes = torch.nn.ModuleList(
                [Prototypes(output_dim, num_prototypes) for _ in range(num_heads)]
            )
        else:  # no reprojection
            self.prototypes = torch.nn.ModuleList(
                [Prototypes(input_dim, num_prototypes) for _ in range(num_heads)]
            )

        self.normalize_prototypes()

    @torch.no_grad()
    def normalize_prototypes(self):
        for p in self.prototypes:
            p.normalize_prototypes()

    def forward_head(self, head_idx, feats):
        z = self.projectors[head_idx](feats)
        return self.prototypes[head_idx](z), z

    def forward(self, feats):
        out = [self.forward_head(h, feats) for h in range(self.num_heads)]
        return [torch.stack(o) for o in map(list, zip(*out))]


class DiscoveryClassifier(nn.Module):
    def __init__(
            self,
            num_labeled,
            num_unlabeled,
            feat_dim,
            hidden_dim,
            proj_dim,
            num_views,
            memory_batches,
            items_per_batch,
            memory_patience,
            num_iters_sk,
            epsilon_sk,
            temperature,
            batch_size,

            sk_mode="classical",
            gauss_sd_sk_new=0.1,
            lamb_sk_new=20,
            num_hidden_layers=1,
    ):
        super().__init__()

        self.head_lab = nn.Linear(feat_dim, num_labeled)

        self.head_unlab = MultiHead(
            input_dim=feat_dim,
            hidden_dim=hidden_dim,
            output_dim=proj_dim,
            num_prototypes=num_unlabeled,
            num_heads=1,
            num_hidden_layers=num_hidden_layers,
        )

        self.num_views = num_views
        self.feat_dim = feat_dim
        self.num_labeled = num_labeled
        self.num_unlabeled = num_unlabeled
        self.num_heads = 1
        self.num_hidden_layers = 1
        self.memory_batches = memory_batches
        self.items_per_batch = items_per_batch
        self.memory_batches = memory_batches
        self.memory_patience = memory_patience
        self.num_iters_sk = num_iters_sk
        self.epsilon_sk = epsilon_sk
        self.temperature = temperature
        self.batch_size = batch_size

        # Initialize Sinkhorn-Knopp
        if sk_mode == "classical":
            self.sk = SinkhornKnopp(
                num_iters=self.num_iters_sk, epsilon=self.epsilon_sk
            )
        else:
            self.sk = SinkhornKnoppLognormalPrior(
                temp=temperature, gauss_sd=gauss_sd_sk_new, lamb=lamb_sk_new,
            )

        # Initialize feature memory bank
        self.memory_size = self.memory_batches * self.items_per_batch * self.batch_size
        self.memory_last_idx = torch.zeros(self.num_views).long()  # indices of the last free memory cell
        self.register_buffer("memory_feat", torch.empty((
            self.num_views,        # one per each crop
            self.memory_size,      # size of memory
            self.feat_dim          # feature dim.
        )))

    def update_memory(self, view_num, features):
        _n = features.shape[0]  # number of new features to be appended
        features = features.detach()

        last_idx = self.memory_last_idx[view_num]  # last used index

        if last_idx + _n <= self.memory_size:
            self.memory_feat[view_num][last_idx:last_idx + _n] = features
        else:
            _n1 = self.memory_size - last_idx
            _n2 = _n - _n1
            self.memory_feat[view_num][last_idx:] = features[:_n1]
            self.memory_feat[view_num][:_n2] = features[_n1:]

        self.memory_last_idx[view_num] = (self.memory_last_idx[view_num] + _n) % self.memory_size

    def cross_entropy_loss(self, preds, targets):
        preds = F.log_softmax(preds, dim=-1)
        return -torch.mean(torch.sum(targets * preds, dim=-1))

    def get_swapped_prediction_loss(self, logits, targets):
        loss = 0
        for view in range(self.num_views):
            for other_view in np.delete(range(self.num_views), view):
                loss += self.cross_entropy_loss(logits[other_view], targets[view])
        return loss / (self.num_views * (self.num_views - 1))

    @torch.no_grad()
    def normalize_prototypes(self):
        self.head_unlab.normalize_prototypes()

    def forward_knowns_bg_head_single_view(self, x):
        self.normalize_prototypes()
        logits = self.head_lab(x)
        return logits

    def forward_heads_single_view(self, x):
        """Note: does not support multi-head scenario (uses only the first head)."""

        self.normalize_prototypes()

        # Knowns head
        logits_knowns = self.head_lab(x)
        logits_knowns = logits_knowns[None, :, :]  # Make shape-compatible with multi-head novels heads

        # Novels heads
        logits_novels = self.head_unlab(x)[0] / self.temperature

        # Concatenate
        logits_full = torch.cat([logits_knowns, logits_novels], dim=-1)

        # Use only the first head output (does not support multi-head)
        logits_full = logits_full[0]

        return logits_full

    def forward_heads(self, feats):
        logits_lab = self.head_lab(feats)
        logits_unlab, _ = self.head_unlab(feats)
        logits_unlab /= self.temperature

        out = {
            "logits_lab": logits_lab,
            "logits_unlab": logits_unlab,
        }
        return out

    def forward_classifier(self, feats):
        out = [self.forward_heads(f) for f in feats]
        out_dict = {"feats": torch.stack(feats)}
        for key in out[0].keys():
            out_dict[key] = torch.stack([o[key] for o in out])
        return out_dict

    def forward(self, views):
        # Forward through classifier heads
        outputs = self.forward_classifier(views)

        # Process outputs
        outputs["logits_lab"] = (
            outputs["logits_lab"].unsqueeze(1).expand(-1, self.num_heads, -1, -1)
        )
        logits = torch.cat([outputs["logits_lab"], outputs["logits_unlab"]], dim=-1)

        # Create targets placeholders
        targets = torch.zeros_like(logits)

        # If memory has not been filled yet, just fill it without calculating the losses
        if self.memory_patience > 0:
            self.memory_patience -= 1

            for v in range(self.num_views):
                self.update_memory(v, outputs["feats"][v])

            # Compute arbitrary losses
            loss_cluster = torch.zeros(1).to(logits.device)[0]

        else:
            # Generate pseudo-labels with sinkhorn-knopp and fill targets
            _batch_size = logits.shape[2]

            for v in range(self.num_views):
                for h in range(self.num_heads):
                    logits_sk = logits[v, h]

                    # Use extra features from memory
                    mem_feat = self.memory_feat[v]  # get memory features for the current view
                    mem_logits_lab = self.head_lab(mem_feat)                                           # get logits for knowns

                    mem_logits_unlab, _ = self.head_unlab.forward_head(h, mem_feat)                    # get logits for novels
                    mem_logits_unlab /= self.temperature

                    mem_logits_full = torch.cat([mem_logits_lab, mem_logits_unlab], dim=1)             # logits for knowns + novels
                    logits_sk = torch.cat([logits_sk, mem_logits_full], dim=0)                         # concat batch logits with memory logits

                    # Pseudo-labels from SK for knowns + novels
                    logits_sk *= self.temperature  # downweight the logits
                    targets_sk = self.sk(logits_sk).type_as(targets)  # SK

                    targets_sk = targets_sk[:_batch_size]             # keep only the batch codes
                    targets[v, h] = targets_sk

                # Update memory for the current view
                self.update_memory(v, outputs["feats"][v])

            # Compute swapped prediction loss
            loss_cluster = self.get_swapped_prediction_loss(logits, targets)

        # Finalize losses
        losses = {
            "loss": loss_cluster,
            "loss_cluster": loss_cluster,
        }

        return losses
