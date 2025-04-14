import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from copy import deepcopy
import pytorch_lightning as pl
import torchvision

from meta_project.utils.split_batch import split_batch
# class MAML(pl.LightningModule):
#     def __init__(self, input_dim, lr=1e-3, lr_inner=0.01, num_inner_steps=1):
#         """
#         Args:
#             input_dim: Dimension of input features
#             num_classes: Number of classes in classification
#             lr: Outer loop learning rate
#             lr_inner: Inner loop learning rate
#             num_inner_steps: Number of gradient updates in inner loop
#         """
#         super().__init__()
#         self.save_hyperparameters()
        
#         self.model = torchvision.models.DenseNet(growth_rate=32,
#                                           block_config=(6, 6, 6, 6),
#                                           bn_size=2,
#                                           num_init_features=64,
#                                           num_classes=self.hparams.input_dim  # Output dimensionality
#                                          )

#     def configure_optimizers(self):
#         return optim.Adam(self.parameters(), lr=self.hparams.lr)

#     def forward(self, x):
#         return self.model(x)

#     def adapt(self, support_imgs, support_targets):
#         """Inner loop adaptation"""
#         fast_weights = {n: p.clone() for n, p in self.model.named_parameters()}
        
#         for _ in range(self.hparams.num_inner_steps):
#             # Forward pass with current fast weights
#             logits = self._forward_with_weights(support_imgs, fast_weights)
#             loss = F.cross_entropy(logits, support_targets)
            
#             # Compute gradients and update fast weights
#             grads = torch.autograd.grad(loss, fast_weights.values(), create_graph=True)
#             fast_weights = {
#                 name: weight - self.hparams.lr_inner * grad
#                 for (name, weight), grad in zip(fast_weights.items(), grads)
#             }
            
#         return fast_weights

#     def _forward_with_weights(self, x, weights):
#         """Helper function to do forward pass with custom weights"""
#         x = F.linear(x, weights['model.0.weight'], weights['model.0.bias'])
#         x = F.relu(x)
#         x = F.linear(x, weights['model.2.weight'], weights['model.2.bias'])
#         return x

#     def meta_step(self, batch, mode="train"):
#         """Process a batch of tasks"""
#         task_losses = []
#         task_accs = []
        
#         # Assuming batch is a tuple of (support, query) pairs
#         support_batch, query_batch = batch
#         support_imgs, support_targets = support_batch
#         query_imgs, query_targets = query_batch
        
#         # Inner loop adaptation
#         fast_weights = self.adapt(support_imgs, support_targets)
        
#         # Evaluate on query set
#         query_logits = self._forward_with_weights(query_imgs, fast_weights)
#         loss = F.cross_entropy(query_logits, query_targets)
#         acc = (query_logits.argmax(dim=1) == query_targets).float().mean()
        
#         self.log(f"{mode}_loss", loss)
#         self.log(f"{mode}_acc", acc)
        
#         return loss

#     def training_step(self, batch, batch_idx):
#         loss = self.meta_step(batch, mode="train")
#         return loss

#     def validation_step(self, batch, batch_idx):
#         self.meta_step(batch, mode="val")

#     def test_step(self, batch, batch_idx):
#         self.meta_step(batch, mode="test")

class MAML(pl.LightningModule):

    def __init__(self, proto_dim, lr, lr_inner, lr_output, num_inner_steps):
        """
        Inputs
            proto_dim - Dimensionality of prototype feature space
            lr - Learning rate of the outer loop Adam optimizer
            lr_inner - Learning rate of the inner loop SGD optimizer
            lr_output - Learning rate for the output layer in the inner loop
            num_inner_steps - Number of inner loop updates to perform
        """
        super().__init__()
        self.save_hyperparameters()
        self.model = torchvision.models.DenseNet(growth_rate=32,
                                          block_config=(6, 6, 6, 6),
                                          bn_size=2,
                                          num_init_features=64,
                                          num_classes=self.hparams.proto_dim  # Output dimensionality
                                         )

    def configure_optimizers(self):
        optimizer = optim.AdamW(self.parameters(), lr=self.hparams.lr)
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[140,180], gamma=0.1)
        return [optimizer], [scheduler]

    def run_model(self, local_model, output_weight, output_bias, imgs, labels):
        # Execute a model with given output layer weights and inputs
        feats = local_model(imgs)
        preds = F.linear(feats, output_weight, output_bias)
        loss = F.cross_entropy(preds, labels)
        acc = (preds.argmax(dim=1) == labels).float()
        return loss, preds, acc

    def adapt_few_shot(self, support_imgs, support_targets):
        # Determine prototype initialization
        support_feats = self.model(support_imgs)
        prototypes, classes = ProtoNet.calculate_prototypes(support_feats, support_targets)
        support_labels = (classes[None,:] == support_targets[:,None]).long().argmax(dim=-1)
        # Create inner-loop model and optimizer
        local_model = deepcopy(self.model)
        local_model.train()
        local_optim = optim.SGD(local_model.parameters(), lr=self.hparams.lr_inner)
        local_optim.zero_grad()
        # Create output layer weights with prototype-based initialization
        init_weight = 2 * prototypes
        init_bias = -torch.norm(prototypes, dim=1)**2
        output_weight = init_weight.detach().requires_grad_()
        output_bias = init_bias.detach().requires_grad_()

        # Optimize inner loop model on support set
        for _ in range(self.hparams.num_inner_steps):
            # Determine loss on the support set
            loss, _, _ = self.run_model(local_model, output_weight, output_bias, support_imgs, support_labels)
            # Calculate gradients and perform inner loop update
            loss.backward()
            local_optim.step()
            # Update output layer via SGD
            # (https://discuss.pytorch.org/t/the-difference-between-torch-tensor-data-and-torch-tensor/25995/4):
            with torch.no_grad():
                output_weight.copy_(output_weight - self.hparams.lr_output * output_weight.grad)
                output_bias.copy_(output_bias - self.hparams.lr_output * output_bias.grad)

            # Reset gradients
            local_optim.zero_grad()
            output_weight.grad.fill_(0)
            output_bias.grad.fill_(0)

        # Re-attach computation graph of prototypes
        output_weight = (output_weight - init_weight).detach() + init_weight
        output_bias = (output_bias - init_bias).detach() + init_bias

        return local_model, output_weight, output_bias, classes

    def outer_loop(self, batch, mode="train"):
        accuracies = []
        losses = []
        self.model.zero_grad()

        # Determine gradients for batch of tasks
        for task_batch in batch:
            imgs, targets = task_batch
            support_imgs, query_imgs, support_targets, query_targets = split_batch(imgs, targets)
            # Perform inner loop adaptation
            local_model, output_weight, output_bias, classes = self.adapt_few_shot(support_imgs, support_targets)
            # Determine loss of query set
            query_labels = (classes[None,:] == query_targets[:,None]).long().argmax(dim=-1)
            loss, preds, acc = self.run_model(local_model, output_weight, output_bias, query_imgs, query_labels)
            # Calculate gradients for query set loss
            if mode == "train":
                loss.backward()

                for p_global, p_local in zip(self.model.parameters(), local_model.parameters()):
                    p_global.grad += p_local.grad  # First-order approx. -> add gradients of finetuned and base model

            accuracies.append(acc.mean().detach())
            losses.append(loss.detach())

        # Perform update of base model
        if mode == "train":
            opt = self.optimizers()
            opt.step()
            opt.zero_grad()

        avg_loss = torch.stack(losses).mean()
        avg_acc = torch.stack(accuracies).mean()

        # Logowanie z opcją `on_step` i `on_epoch` (ważne dla Lightning)
        self.log(f"{mode}_loss", avg_loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log(f"{mode}_acc", avg_acc, on_step=True, on_epoch=True, prog_bar=True)
        # print(f"{mode}_loss", sum(losses) / len(losses))
        # print(f"{mode}_acc", sum(accuracies) / len(accuracies))

    def training_step(self, batch, batch_idx):
        self.outer_loop(batch, mode="train")
        return None  # Returning None means we skip the default training optimizer steps by PyTorch Lightning

    def validation_step(self, batch, batch_idx):
        # Validation requires to finetune a model, hence we need to enable gradients
        torch.set_grad_enabled(True)
        self.outer_loop(batch, mode="val")
        torch.set_grad_enabled(False)


class ProtoNet(pl.LightningModule):
    @staticmethod
    def calculate_prototypes(features, targets):
        # Given a stack of features vectors and labels, return class prototypes
        # features - shape [N, proto_dim], targets - shape [N]
        classes, _ = torch.unique(targets).sort()  # Determine which classes we have
        prototypes = []
        for c in classes:
            p = features[torch.where(targets == c)[0]].mean(dim=0)  # Average class feature vectors
            prototypes.append(p)
        prototypes = torch.stack(prototypes, dim=0)
        # Return the 'classes' tensor to know which prototype belongs to which class
        return prototypes, classes