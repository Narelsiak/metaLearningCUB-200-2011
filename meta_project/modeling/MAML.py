import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from copy import deepcopy
import pytorch_lightning as pl
import torchvision
import math

from meta_project.utils.split_batch import split_batch

class MAML(pl.LightningModule):

    def __init__(self, proto_dim, lr, lr_inner, lr_output, num_inner_steps):
        super().__init__()
        self.save_hyperparameters()
        self.model = torchvision.models.DenseNet(
            growth_rate=32,
            block_config=(6, 6, 6, 6),
            bn_size=2,
            num_init_features=64,
            num_classes=self.hparams.proto_dim  # Used for feature size
        )

    def configure_optimizers(self):
        optimizer = optim.AdamW(self.parameters(), lr=self.hparams.lr)
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[140, 180], gamma=0.1)
        return [optimizer], [scheduler]

    def run_model(self, local_model, output_weight, output_bias, imgs, labels):
        feats = local_model(imgs)
        preds = F.linear(feats, output_weight, output_bias)
        loss = F.cross_entropy(preds, labels)
        acc = (preds.argmax(dim=1) == labels).float()
        return loss, preds, acc

    def adapt_few_shot(self, support_imgs, support_targets):
        # Remap targets to 0...(n_classes-1)
        classes, support_labels = torch.unique(support_targets, return_inverse=True)
        n_classes = len(classes)
        feat_dim = self.hparams.proto_dim

        # Create local model and optimizer
        local_model = deepcopy(self.model)
        local_model.train()
        local_optim = optim.SGD(local_model.parameters(), lr=self.hparams.lr_inner)
        local_optim.zero_grad()

        # Initialize output layer randomly
        output_weight = torch.empty((n_classes, feat_dim), device=self.device, requires_grad=True)
        output_bias = torch.empty(n_classes, device=self.device, requires_grad=True)
        nn.init.kaiming_uniform_(output_weight, a=math.sqrt(5))
        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(output_weight)
        bound = 1 / math.sqrt(fan_in)
        nn.init.uniform_(output_bias, -bound, bound)

        # Inner loop adaptation
        for _ in range(self.hparams.num_inner_steps):
            loss, _, _ = self.run_model(local_model, output_weight, output_bias, support_imgs, support_labels)
            grads = torch.autograd.grad(loss, list(local_model.parameters()) + [output_weight, output_bias], create_graph=False)
            param_grads = grads[:len(list(local_model.parameters()))]
            w_grad, b_grad = grads[-2], grads[-1]

            for p, g in zip(local_model.parameters(), param_grads):
                p.data -= self.hparams.lr_inner * g
            output_weight = output_weight - self.hparams.lr_output * w_grad
            output_bias = output_bias - self.hparams.lr_output * b_grad

        return local_model, output_weight, output_bias, classes

    def outer_loop(self, batch, mode="train"):
        accuracies = []
        losses = []
        self.model.zero_grad()

        for task_batch in batch:
            imgs, targets = task_batch
            support_imgs, query_imgs, support_targets, query_targets = split_batch(imgs, targets)

            local_model, output_weight, output_bias, classes = self.adapt_few_shot(support_imgs, support_targets)

            # Remap query targets
            _, query_labels = torch.unique(query_targets, return_inverse=True)

            loss, preds, acc = self.run_model(local_model, output_weight, output_bias, query_imgs, query_labels)

            if mode == "train":
                loss.backward()

                # Adding gradients to global model
                for p_global, p_local in zip(self.model.parameters(), local_model.parameters()):
                    if p_local.grad is not None:  # Check if gradients exist
                        if p_global.grad is None:
                            p_global.grad = torch.zeros_like(p_global)
                        p_global.grad += p_local.grad  # First-order MAML gradient approximation

                # Update the output layer's weights and biases (no double-gradient accumulation)
                if output_weight.grad is not None:
                    if output_weight.grad is None:
                        output_weight.grad = torch.zeros_like(output_weight)
                    output_weight.grad += output_weight.grad
                if output_bias.grad is not None:
                    if output_bias.grad is None:
                        output_bias.grad = torch.zeros_like(output_bias)
                    output_bias.grad += output_bias.grad

            accuracies.append(acc.mean().detach())
            losses.append(loss.detach())

        if mode == "train":
            opt = self.optimizers()
            opt.step()
            opt.zero_grad()

        avg_loss = torch.stack(losses).mean()
        avg_acc = torch.stack(accuracies).mean()

        self.log(f"{mode}_loss", avg_loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log(f"{mode}_acc", avg_acc, on_step=True, on_epoch=True, prog_bar=True)

    def training_step(self, batch, batch_idx):
        self.outer_loop(batch, mode="train")
        return None

    def validation_step(self, batch, batch_idx):
        torch.set_grad_enabled(True)
        self.outer_loop(batch, mode="val")
        torch.set_grad_enabled(False)