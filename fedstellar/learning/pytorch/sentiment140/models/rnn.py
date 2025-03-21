#
# This file is part of the Fedstellar platform (see https://github.com/enriquetomasmb/fedstellar).
# Copyright (c) 2023 Enrique Tomás Martínez Beltrán.
#

# To Avoid Crashes with a lot of nodes
import torch.multiprocessing

torch.multiprocessing.set_sharing_strategy("file_system")

import lightning as pl
from torchmetrics.classification import MulticlassAccuracy, MulticlassRecall, MulticlassPrecision, MulticlassF1Score, MulticlassConfusionMatrix
from torchmetrics import MetricCollection
from torch.nn import functional as F


class SentimentModelRNN(pl.LightningModule):
    """
    LightningModule for MNIST.
    """

    def process_metrics(self, phase, y_pred, y, loss=None):
        """
        Calculate and log metrics for the given phase.
        Args:
            phase (str): One of 'Train', 'Validation', or 'Test'
            y_pred (torch.Tensor): Model predictions
            y (torch.Tensor): Ground truth labels
            loss (torch.Tensor, optional): Loss value
        """
        if loss is not None:
            self.log(f"{phase}/Loss", loss, prog_bar=True, logger=True)

        y_pred_classes = torch.argmax(y_pred, dim=1)
        if phase == "Train":
            output = self.train_metrics(y_pred_classes, y)
        elif phase == "Validation":
            output = self.val_metrics(y_pred_classes, y)
        elif phase == "Test":
            output = self.test_metrics(y_pred_classes, y)
        else:
            raise NotImplementedError
        # print(f"y_pred shape: {y_pred.shape}, y_pred_classes shape: {y_pred_classes.shape}, y shape: {y.shape}")  # Debug print
        output = {f"{phase}/{key.replace('Multiclass', '').split('/')[-1]}": value for key, value in output.items()}
        self.log_dict(output, prog_bar=True, logger=True)

        if self.cm is not None:
            self.cm.update(y_pred_classes, y)

    def log_metrics_by_epoch(self, phase, print_cm=False, plot_cm=False):
        """
        Log all metrics at the end of an epoch for the given phase.
        Args:
            phase (str): One of 'Train', 'Validation', or 'Test'
            :param phase:
            :param plot_cm:
        """
        print(f"Epoch end: {phase}, epoch number: {self.epoch_global_number[phase]}")
        if phase == "Train":
            output = self.train_metrics.compute()
            self.train_metrics.reset()
        elif phase == "Validation":
            output = self.val_metrics.compute()
            self.val_metrics.reset()
        elif phase == "Test":
            output = self.test_metrics.compute()
            self.test_metrics.reset()
        else:
            raise NotImplementedError

        output = {f"{phase}Epoch/{key.replace('Multiclass', '').split('/')[-1]}": value for key, value in output.items()}

        self.log_dict(output, prog_bar=True, logger=True)

        if self.cm is not None:
            cm = self.cm.compute().cpu()
            print(f"{phase}Epoch/CM\n", cm) if print_cm else None
            if plot_cm:
                import seaborn as sns
                import matplotlib.pyplot as plt
                plt.figure(figsize=(10, 7))
                ax = sns.heatmap(cm.numpy(), annot=True, fmt="d", cmap="Blues")
                ax.set_xlabel("Predicted labels")
                ax.set_ylabel("True labels")
                ax.set_title("Confusion Matrix")
                ax.set_xticks(range(10))
                ax.set_yticks(range(10))
                ax.xaxis.set_ticklabels([i for i in range(10)])
                ax.yaxis.set_ticklabels([i for i in range(10)])
                # self.logger.experiment.add_figure(f"{phase}Epoch/CM", ax.get_figure(), global_step=self.epoch_global_number[phase])
                plt.close()

        # Reset metrics

        self.epoch_global_number[phase] += 1

    def __init__(
            self,
            out_channels=2,
            learning_rate=0.005,
            metrics=None,
            confusion_matrix=None,
            seed=None
    ):
        super().__init__()
        if metrics is None:
            metrics = MetricCollection([
                MulticlassAccuracy(num_classes=out_channels),
                MulticlassPrecision(num_classes=out_channels),
                MulticlassRecall(num_classes=out_channels),
                MulticlassF1Score(num_classes=out_channels)
            ])

        # Define metrics
        self.train_metrics = metrics.clone(prefix="Train/")
        self.val_metrics = metrics.clone(prefix="Validation/")
        self.test_metrics = metrics.clone(prefix="Test/")

        if confusion_matrix is None:
            self.cm = MulticlassConfusionMatrix(num_classes=out_channels)

        # Set seed for reproducibility initialization
        if seed is not None:
            torch.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)

        self.example_input_array = torch.zeros(1, 1, 28, 28)
        self.learning_rate = learning_rate

        self.embedding_dim = 300
        self.hidden_dim = 256
        self.n_layers = 1
        self.bidirectional = True
        self.output_dim = out_channels

        self.encoder = torch.nn.LSTM(self.embedding_dim,
                                     self.hidden_dim,
                                     num_layers=self.n_layers,
                                     bidirectional=self.bidirectional,
                                     dropout=0.5,
                                     batch_first=True)
        self.fc = torch.nn.Linear(self.hidden_dim * 2, self.output_dim)
        self.dropout = torch.nn.Dropout(0.5)

        self.criterion = torch.nn.CrossEntropyLoss()

        self.l1 = torch.nn.Linear(28 * 28, 256)
        self.l2 = torch.nn.Linear(256, 128)
        self.l3 = torch.nn.Linear(128, out_channels)

        self.epoch_global_number = {"Train": 0, "Validation": 0, "Test": 0}

    def forward(self, x):
        """ """
        packed_output, (hidden, cell) = self.encoder(x)
        hidden = self.dropout(torch.cat((hidden[-2, :, :], hidden[-1, :, :]), dim=1))
        out = self.fc(hidden)
        out = F.softmax(out, dim=1)
        return out

    def configure_optimizers(self):
        """ """
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer

    def step(self, batch, phase):
        images, labels = batch
        images = images.to(self.device)
        labels = labels.to(self.device)
        y_pred = self.forward(images)
        loss = self.criterion(y_pred, labels)

        # Get metrics for each batch and log them
        self.log(f"{phase}/Loss", loss, prog_bar=True)
        self.process_metrics(phase, y_pred, labels, loss)

        return loss

    def training_step(self, batch, batch_id):
        """
        Training step for the model.
        Args:
            batch:
            batch_id:

        Returns:
        """
        return self.step(batch, "Train")

    def on_train_epoch_end(self):
        self.log_metrics_by_epoch("Train", print_cm=True, plot_cm=True)

    def validation_step(self, batch, batch_idx):
        """
        Validation step for the model.
        Args:
            batch:
            batch_idx:

        Returns:
        """
        return self.step(batch, "Validation")

    def on_validation_epoch_end(self):
        self.log_metrics_by_epoch("Validation", print_cm=True, plot_cm=True)

    def test_step(self, batch, batch_idx):
        """
        Test step for the model.
        Args:
            batch:
            batch_idx:

        Returns:
        """
        return self.step(batch, "Test")

    def on_test_epoch_end(self):
        self.log_metrics_by_epoch("Test", print_cm=True, plot_cm=True)
