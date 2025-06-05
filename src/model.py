import lightning as L
import torch
from torch import nn
import torch.nn.init as init
from torchmetrics.classification import (
    MulticlassF1Score,
    MulticlassAccuracy,
    MulticlassPrecision,
)
from line_profiler import profile

NUM_CLASSES = 1000


class ExponentialMovingAverage:
    def __init__(self, alpha):
        self.alpha = alpha
        self.smoothed_value = None

    def update(self, new_value):
        if self.smoothed_value is None:
            self.smoothed_value = new_value
        else:
            self.smoothed_value = (
                self.alpha * new_value + (1 - self.alpha) * self.smoothed_value
            )
        return self.smoothed_value

    def get_smoothed_value(self):
        return self.smoothed_value

    def reset(self):
        self.smoothed_value = None


class AlexNet(L.LightningModule):

    def __init__(self, params):
        super(AlexNet, self).__init__()
        self.save_hyperparameters()
        self.lr = params["lr"]
        self.model = nn.Sequential(
            nn.Conv2d(3, 96, kernel_size=11, stride=4, padding=2),  # Index 0
            nn.ReLU(inplace=True),  # Index 1
            nn.LocalResponseNorm(size=5, alpha=0.0001, beta=0.75, k=2.0),  # Index 2
            nn.MaxPool2d(kernel_size=3, stride=2),  # Index 3
            nn.Conv2d(96, 256, kernel_size=5, padding=2),  # Index 4
            nn.ReLU(inplace=True),  # Index 5
            nn.LocalResponseNorm(size=5, alpha=0.0001, beta=0.75, k=2.0),  # Index 6
            nn.MaxPool2d(kernel_size=3, stride=2),  # Index 7
            nn.Conv2d(256, 384, kernel_size=3, padding=1),  # Index 8
            nn.ReLU(inplace=True),  # Index 9
            nn.Conv2d(384, 384, kernel_size=3, padding=1),  # Index 10
            nn.ReLU(inplace=True),  # Index 11
            nn.Conv2d(384, 256, kernel_size=3, padding=1),  # Index 12
            nn.ReLU(inplace=True),  # Index 13
            nn.MaxPool2d(kernel_size=3, stride=2),  # Index 14
            # Classification
            nn.AdaptiveAvgPool2d((6, 6)),  # Index 15
            nn.Flatten(),  # Index 16
            nn.Dropout(p=0.5),  # Index 17
            nn.Linear(256 * 6 * 6, 4096),  # Index 18
            nn.ReLU(inplace=True),  # Index 19
            nn.Dropout(p=0.5),  # Index 20
            nn.Linear(4096, 4096),  # Index 21
            nn.ReLU(inplace=True),  # Index 22
            nn.Linear(4096, NUM_CLASSES),  # Index 23
        )

        self.criterion = nn.CrossEntropyLoss()

        smoothing_factor = 0.18  # Around 10 datapoints
        self.loss_smoother = ExponentialMovingAverage(smoothing_factor)
        self.f1_metric = MulticlassF1Score(num_classes=NUM_CLASSES, average="macro")
        self.top1_accuracy_metric = MulticlassAccuracy(
            num_classes=NUM_CLASSES,
            average="macro",
            top_k=1,
        )
        self.top5_accuracy_metric = MulticlassAccuracy(
            num_classes=NUM_CLASSES,
            average="macro",
            top_k=5,
        )
        self.precision_metric = MulticlassPrecision(
            num_classes=NUM_CLASSES, average="macro"
        )

        self.val_loss_smoother = ExponentialMovingAverage(smoothing_factor)
        self.val_f1_metric = MulticlassF1Score(num_classes=NUM_CLASSES, average="macro")
        self.val_top1_accuracy_metric = MulticlassAccuracy(
            num_classes=NUM_CLASSES,
            average="macro",
            top_k=1,
        )
        self.val_top5_accuracy_metric = MulticlassAccuracy(
            num_classes=NUM_CLASSES,
            average="macro",
            top_k=5,
        )
        self.val_precision_metric = MulticlassPrecision(
            num_classes=NUM_CLASSES, average="macro"
        )

        self.test_loss_smoother = ExponentialMovingAverage(smoothing_factor)
        self.test_f1_metric = MulticlassF1Score(
            num_classes=NUM_CLASSES, average="macro"
        )
        self.test_top1_accuracy_metric = MulticlassAccuracy(
            num_classes=NUM_CLASSES,
            average="macro",
            top_k=1,
        )
        self.test_top5_accuracy_metric = MulticlassAccuracy(
            num_classes=NUM_CLASSES,
            average="macro",
            top_k=5,
        )
        self.test_precision_metric = MulticlassPrecision(
            num_classes=NUM_CLASSES, average="macro"
        )

    # def forward(self, input, *args, **kwargs):
    #     return self.model(input)

    def configure_optimizers(self):
        optimizer = torch.optim.SGD(
            self.parameters(), lr=self.lr, momentum=0.9, weight_decay=0.0005
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": torch.optim.lr_scheduler.ReduceLROnPlateau(
                    optimizer,
                    mode="max",
                    factor=0.1,
                    patience=5,
                    min_lr=1e-5,
                    cooldown=20,
                ),
                "interval": "epoch",
                "monitor": "val_acc_top1",
                "frequency": 1,
            },
        }

    # @profile
    def training_step(self, train_batch, batch_idx, *args, **kwargs):
        x, y = train_batch

        logits = self.model(x)
        loss = self.criterion(logits, y)

        predictions = torch.argmax(logits, dim=-1)

        loss_item = loss.item()
        self.loss_smoother.update(loss_item)
        self.f1_metric.update(predictions, y)
        self.top1_accuracy_metric.update(logits, y)
        self.top5_accuracy_metric.update(logits, y)
        self.precision_metric.update(predictions, y)

        self.log("train_loss", loss_item)
        self.log("train_loss_smoothed", self.loss_smoother.get_smoothed_value())
        self.log("train_f1", self.f1_metric.compute().cpu().item())
        self.log("train_acc_top1", self.top1_accuracy_metric.compute().cpu().item())
        self.log("train_acc_top5", self.top5_accuracy_metric.compute().cpu().item())
        self.log("train_prec", self.precision_metric.compute().cpu().item())

        return loss

    def on_validation_start(self):
        self.val_loss_smoother.reset()
        self.val_f1_metric.reset()
        self.val_top1_accuracy_metric.reset()
        self.val_top5_accuracy_metric.reset()
        self.val_precision_metric.reset()

    def validation_step(self, val_batch, *args, **kwargs):
        x, y = val_batch

        logits = self.model(x)
        loss = self.criterion(logits, y)

        predictions = torch.argmax(logits, dim=-1)

        loss_item = loss.item()
        self.val_loss_smoother.update(loss_item)
        self.val_f1_metric.update(predictions, y)
        self.val_top1_accuracy_metric.update(logits, y)
        self.val_top5_accuracy_metric.update(logits, y)
        self.val_precision_metric.update(predictions, y)

        self.log("val_loss", loss_item)
        self.log("val_loss_smoothed", self.val_loss_smoother.get_smoothed_value())
        self.log("val_f1", self.val_f1_metric.compute().cpu().item())
        self.log("val_acc_top1", self.val_top1_accuracy_metric.compute().cpu().item())
        self.log("val_acc_top5", self.val_top5_accuracy_metric.compute().cpu().item())
        self.log("val_prec", self.val_precision_metric.compute().cpu().item())

        return loss

    def test_step(self, test_batch, *args, **kwargs):
        x, y = test_batch

        logits = self.model(x)
        loss = self.criterion(logits, y)

        predictions = torch.argmax(logits, dim=-1)

        loss_item = loss.item()
        self.test_loss_smoother.update(loss_item)
        self.test_f1_metric.update(predictions, y)
        self.test_top1_accuracy_metric.update(logits, y)
        self.test_top5_accuracy_metric.update(logits, y)
        self.test_precision_metric.update(predictions, y)

        self.log("test_loss", loss_item)
        self.log("test_loss_smoothed", self.test_loss_smoother.get_smoothed_value())
        self.log("test_f1", self.test_f1_metric.compute().cpu().item())
        self.log("test_acc_top1", self.test_top1_accuracy_metric.compute().cpu().item())
        self.log("test_acc_top5", self.test_top5_accuracy_metric.compute().cpu().item())
        self.log("test_prec", self.test_precision_metric.compute().cpu().item())
        return loss

    # @profile
    def backward(self, loss, *args, **kwargs):
        loss.backward()


# --- Custom Initialization Function for AlexNet ---
def alexnet_initialize_weights(alexnet: AlexNet):
    """
    Initializes weights and biases of an AlexNet model according to the original paper.
    """
    for m in alexnet.modules():
        if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
            # All weights from a zero-mean Gaussian distribution with standard deviation 0.01
            init.normal_(m.weight, mean=0.0, std=0.01)
            if m.bias is not None:
                # Initialize all biases to 0 by default
                init.constant_(m.bias, 0)

    # Second convolutional layer
    if alexnet.model[4].bias is not None:
        init.constant_(alexnet.model[4].bias, 1)

    # Fourth convolutional layer
    if alexnet.model[10].bias is not None:
        init.constant_(alexnet.model[10].bias, 1)

    # Fifth convolutional layer
    if alexnet.model[12].bias is not None:
        init.constant_(alexnet.model[12].bias, 1)

    # First fully-connected hidden layer
    if alexnet.model[18].bias is not None:
        init.constant_(alexnet.model[18].bias, 1)

    # Second fully-connected hidden layer
    if alexnet.model[21].bias is not None:
        init.constant_(alexnet.model[21].bias, 1)
