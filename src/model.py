import lightning as L
import torch
from torch import nn
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
            nn.Conv2d(3, 96, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.LocalResponseNorm(size=5, alpha=0.0001, beta=0.75, k=2.0),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(96, 256, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.LocalResponseNorm(size=5, alpha=0.0001, beta=0.75, k=2.0),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(256, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            # Classification
            nn.AdaptiveAvgPool2d((6, 6)),
            nn.Flatten(),
            nn.Dropout(p=0.5),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, NUM_CLASSES),
        )

        self.criterion = nn.CrossEntropyLoss()

        smoothing_factor = 0.18  # Around 10 datapoints
        self.loss_smoother = ExponentialMovingAverage(smoothing_factor)
        self.f1_metric = MulticlassF1Score(num_classes=NUM_CLASSES, average="macro")
        self.accuracy_metric = MulticlassAccuracy(
            num_classes=NUM_CLASSES, average="macro"
        )
        self.precision_metric = MulticlassPrecision(
            num_classes=NUM_CLASSES, average="macro"
        )

        self.val_loss_smoother = ExponentialMovingAverage(smoothing_factor)
        self.val_f1_metric = MulticlassF1Score(num_classes=NUM_CLASSES, average="macro")
        self.val_accuracy_metric = MulticlassAccuracy(
            num_classes=NUM_CLASSES, average="macro"
        )
        self.val_precision_metric = MulticlassPrecision(
            num_classes=NUM_CLASSES, average="macro"
        )

        self.test_loss_smoother = ExponentialMovingAverage(smoothing_factor)
        self.test_f1_metric = MulticlassF1Score(
            num_classes=NUM_CLASSES, average="macro"
        )
        self.test_accuracy_metric = MulticlassAccuracy(
            num_classes=NUM_CLASSES, average="macro"
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
        return optimizer

    # @profile
    def training_step(self, train_batch, batch_idx, *args, **kwargs):
        x, y = train_batch

        logits = self.model(x)

        loss = self.criterion(logits, y)

        if batch_idx % 50 == 0:

            probabilities = torch.nn.functional.softmax(logits, dim=-1)
            predictions = torch.argmax(probabilities, dim=-1)
            predictions = torch.nn.functional.one_hot(
                predictions, num_classes=NUM_CLASSES
            )

            item = loss.item()
            self.loss_smoother.update(item)
            self.f1_metric.update(predictions, y)
            self.accuracy_metric.update(predictions, y)
            self.precision_metric.update(predictions, y)

            self.log("train_loss", loss)
            self.log("train_loss_smoothed", self.loss_smoother.get_smoothed_value())
            self.log("train_f1", self.f1_metric.compute().cpu().item())
            self.log("train_acc", self.accuracy_metric.compute().cpu().item())
            self.log("train_prec", self.precision_metric.compute().cpu().item())

        return loss

    def on_validation_start(self):
        self.val_loss_smoother.reset()
        self.val_f1_metric.reset()
        self.val_accuracy_metric.reset()
        self.val_precision_metric.reset()

    def validation_step(self, val_batch, *args, **kwargs):
        x, y = val_batch

        logits = self.model(x)
        loss = self.criterion(logits, y)

        probabilities = torch.nn.functional.softmax(logits, dim=-1)
        predictions = torch.argmax(probabilities, dim=-1)
        predictions = torch.nn.functional.one_hot(predictions, num_classes=NUM_CLASSES)

        self.val_loss_smoother.update(loss.item())
        self.val_f1_metric.update(predictions, y)
        self.val_accuracy_metric.update(predictions, y)
        self.val_precision_metric.update(predictions, y)

        self.log("val_loss", loss)
        self.log("val_loss_smoothed", self.val_loss_smoother.get_smoothed_value())
        self.log("val_f1", self.val_f1_metric.compute().cpu().item())
        self.log("val_acc", self.val_accuracy_metric.compute().cpu().item())
        self.log("val_prec", self.val_precision_metric.compute().cpu().item())
        return loss

    def test_step(self, test_batch, *args, **kwargs):
        x, y = test_batch

        logits = self.model(x)
        loss = self.criterion(logits, y)

        probabilities = torch.nn.functional.softmax(logits, dim=-1)
        predictions = torch.argmax(probabilities, dim=-1)
        predictions = torch.nn.functional.one_hot(predictions, num_classes=NUM_CLASSES)

        self.test_loss_smoother.update(loss.item())
        self.test_f1_metric.update(predictions, y)
        self.test_accuracy_metric.update(predictions, y)
        self.test_precision_metric.update(predictions, y)

        self.log("test_loss", loss)
        self.log("test_loss_smoothed", self.test_loss_smoother.get_smoothed_value())
        self.log("test_f1", self.test_f1_metric.compute().cpu().item())
        self.log("test_acc", self.test_accuracy_metric.compute().cpu().item())
        self.log("test_prec", self.test_precision_metric.compute().cpu().item())
        return loss

    # @profile
    def backward(self, loss, *args, **kwargs):
        loss.backward()
