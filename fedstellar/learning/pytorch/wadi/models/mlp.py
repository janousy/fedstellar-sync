#
# This file is part of the Fedstellar platform (see https://github.com/enriquetomasmb/fedstellar).
# Copyright (c) 2023 Enrique Tomás Martínez Beltrán.
#

# To Avoid Crashes with a lot of nodes
import torch.multiprocessing

torch.multiprocessing.set_sharing_strategy("file_system")

import lightning as pl
from torchmetrics.classification import BinaryAccuracy, BinaryRecall, BinaryPrecision, BinaryF1Score, BinaryConfusionMatrix
from torchmetrics import MetricCollection


class WADIModelMLP(pl.LightningModule):
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
        output = {f"{phase}/{key.replace('Binary', '').split('/')[-1]}": value for key, value in output.items()}
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

        output = {f"{phase}Epoch/{key.replace('Binary', '').split('/')[-1]}": value for key, value in output.items()}

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
            in_channels=1,
            out_channels=2,
            learning_rate=1e-3,
            metrics=None,
            confusion_matrix=None,
            seed=None
    ):
        super().__init__()
        if metrics is None:
            metrics = MetricCollection([
                BinaryAccuracy(num_classes=out_channels),
                BinaryPrecision(num_classes=out_channels),
                BinaryRecall(num_classes=out_channels),
                BinaryF1Score(num_classes=out_channels)
            ])

        # Define metrics
        self.train_metrics = metrics.clone(prefix="Train/")
        self.val_metrics = metrics.clone(prefix="Validation/")
        self.test_metrics = metrics.clone(prefix="Test/")

        if confusion_matrix is None:
            self.cm = BinaryConfusionMatrix(num_classes=out_channels)

        # Set seed for reproducibility initialization
        if seed is not None:
            torch.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)

        self.example_input_array = torch.zeros(1, 123)
        self.learning_rate = learning_rate

        self.criterion = torch.nn.BCELoss()

        self.l1 = torch.nn.Linear(123, 1024)
        self.l2 = torch.nn.Linear(1024, 512)
        self.l3 = torch.nn.Linear(512, 256)
        self.l4 = torch.nn.Linear(256, 128)
        self.l5 = torch.nn.Linear(128, 64)
        self.l6 = torch.nn.Linear(64, 32)
        self.l7 = torch.nn.Linear(32, 16)
        self.l8 = torch.nn.Linear(16, 8)
        self.l9 = torch.nn.Linear(8, out_channels)

        self.epoch_global_number = {"Train": 0, "Validation": 0, "Test": 0}

    def forward(self, x):
        """ """
        batch_size, features = x.size()

        # (b, 1, 28, 28) -> (b, 1*28*28)
        # x = x.view(batch_size, -1)
        x = self.l1(x)
        x = torch.relu(x)
        x = self.l2(x)
        x = torch.relu(x)
        x = self.l3(x)
        x = torch.relu(x)
        x = self.l4(x)
        x = torch.relu(x)
        x = self.l5(x)
        x = torch.relu(x)
        x = self.l6(x)
        x = torch.relu(x)
        x = self.l7(x)
        x = torch.relu(x)
        x = self.l8(x)
        x = torch.relu(x)
        x = self.l9(x)
        x = torch.sigmoid(x)
        return x

    def configure_optimizers(self):
        """ """
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer

    def step(self, batch, phase):
        x, labels = batch
        x = x.to(self.device)
        labels = labels.to(self.device)
        logits = self.forward(x)
        loss = self.criterion(logits, labels.unsqueeze(1).float())

        # Get metrics for each batch and log them
        self.log(f"{phase}/Loss", loss, prog_bar=True)
        self.process_metrics(phase, logits, labels, loss)

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
