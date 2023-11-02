import torch
import pytorch_lightning as pl
import torch.nn.functional as F
import torch.nn as nn
import timm
from torchmetrics import MaxMetric, MeanMetric

from torchmetrics.classification import MulticlassAccuracy

from typing import Any

# import matplotlib
# import matplotlib.pyplot as plt
# matplotlib.use('Agg')
# from PIL import Image


class Model(nn.Module):  # pylint: disable=too-many-ancestors
    """
    model wrapper for cifar10 classification
    """

    def __init__(self, **kwargs):
        """
        Initializes the network, optimizer and scheduler
        """
        super().__init__()
        self.model = timm.create_model(
            "vit_tiny_patch16_224",
            pretrained=True,
            num_classes=10,
        )

    def forward(self, x):  # pylint: disable=arguments-differ
        out = self.model(x)
        return out


class LitModel(pl.LightningModule):
    def __init__(self, num_classes=10, **kwargs):
        super().__init__()

        self.save_hyperparameters()
        self.num_classes = num_classes
        self.model = Model()

        self.criterion = nn.CrossEntropyLoss()

        # for averaging loss across batches
        self.train_loss = MeanMetric()
        self.val_loss = MeanMetric()
        self.test_loss = MeanMetric()

        # for tracking best so far validation accuracy
        self.val_acc_best = MaxMetric()

        self.train_acc = MulticlassAccuracy(num_classes=num_classes)
        self.val_acc = MulticlassAccuracy(num_classes=num_classes)
        self.test_acc = MulticlassAccuracy(num_classes=num_classes)

        self.scheduler = None
        self.optimizer = None
        self.args = kwargs

        self.preds = []
        self.target = []
        self.example_input_array = torch.rand((1, 3, 224, 224))

    def forward(self, x):
        out = self.model(x)
        return out

    def on_train_start(self):
        # by default lightning executes validation step sanity checks before training starts,
        # so it's worth to make sure validation metrics don't store results from these checks
        self.val_loss.reset()
        self.val_acc.reset()
        self.val_acc_best.reset()

    def model_step(self, batch: Any):
        x, y = batch
        logits = self.forward(x)
        loss = self.criterion(logits, y)
        preds = torch.argmax(logits, dim=1)
        return loss, preds, y

    def training_step(self, batch, batch_idx):
        if batch_idx == 0:
            self.reference_image = (batch[0][0]).unsqueeze(
                0
            )  # pylint: disable=attribute-defined-outside-init
            # self.reference_image.resize((1,1,28,28))
            print("\n\nREFERENCE IMAGE!!!")
            print(self.reference_image.shape)

        train_loss, preds, targets = self.model_step(batch)
        self.train_loss(train_loss)
        self.train_acc(preds, targets)
        self.log(
            "train/loss", self.train_loss, on_step=False, on_epoch=True, prog_bar=True
        )
        self.log(
            "train/acc",
            self.train_acc,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )

        # self.log('train/acc_step', self.train_acc)
        # self.log('train/loss_step', train_loss)
        return {"loss": train_loss}

    def validation_step(self, batch, batch_idx):
        val_loss, preds, targets = self.model_step(batch)
        self.val_loss(val_loss)
        self.val_acc(preds, targets)
        self.log("val/loss", self.val_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log(
            "val/acc",
            self.val_acc,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )

        # self.log('val/acc/step', self.val_acc)
        # self.log('val/loss/step', val_loss)
        return {"loss": val_loss}

    def test_step(self, batch, batch_idx):
        test_loss, preds, targets = self.model_step(batch)
        self.preds += preds.tolist()
        self.target += targets.tolist()
        self.test_loss(test_loss)
        self.test_acc(preds, targets)
        self.log(
            "test/loss", self.test_loss, on_step=False, on_epoch=True, prog_bar=True
        )
        self.log(
            "test/acc",
            self.test_acc,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )

        # self.log('test/acc/step', self.test_acc)
        # self.log('test/loss/step', test_loss)
        return {"loss": test_loss, "test_preds": preds, "test_targ": targets}

    def on_validation_epoch_end(self):
        acc = self.val_acc.compute()  # get current val acc
        self.val_acc_best(acc)  # update best so far val acc
        val_acc_best = self.val_acc_best.compute()
        self.log("val/acc_best", val_acc_best, prog_bar=True)

    def on_test_epoch_end(self):
        pass

    def on_train_epoch_end(self):
        pass

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            lr=1e-3, weight_decay=0.0, params=self.parameters()
        )

        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            mode="min", factor=0.1, patience=10, optimizer=optimizer
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val/loss",
                "interval": "epoch",
                "frequency": 1,
            },
        }
