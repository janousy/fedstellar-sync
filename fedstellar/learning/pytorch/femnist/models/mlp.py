import lightning as pl
import torch
from torch.nn import functional as F
from torchmetrics import Accuracy
from torchmetrics.classification import MulticlassAccuracy, MulticlassRecall, MulticlassPrecision, MulticlassF1Score, MulticlassConfusionMatrix


###############################
#    Multilayer Perceptron    #
###############################


class MLP(pl.LightningModule):
    """
    Multilayer Perceptron (MLP) to solve MNIST with PyTorch Lightning.
    """

    def __init__(
            self, metric=Accuracy, out_channels=62, lr_rate=0.001, seed=None
    ):  # low lr to avoid overfitting

        if metrics is None:
            metrics = [MulticlassAccuracy, MulticlassPrecision, MulticlassRecall, MulticlassF1Score, MulticlassConfusionMatrix]

        self.metrics = []
        if type(metrics) is list:
            try:
                for m in metrics:
                    self.metrics.append(m(num_classes=62))
            except TypeError:
                raise TypeError("metrics must be a list of torchmetrics.Metric")

        # Set seed for reproducibility initialization
        if seed is not None:
            torch.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)

        super().__init__()
        self.lr_rate = lr_rate
        self.metric = metric(num_classes=62, task="multiclass")

        self.l1 = torch.nn.Linear(28 * 28, 256)
        self.l2 = torch.nn.Linear(256, 128)
        self.l3 = torch.nn.Linear(128, out_channels)

        self.epoch_global_number = {"Train": 0, "Validation": 0, "Test": 0}

        self.epoch_num_steps = {"Train": 0, "Validation": 0, "Test": 0}
        self.epoch_loss_sum = {"Train": 0.0, "Validation": 0.0, "Test": 0.0}

        self.epoch_output = {"Train": [], "Validation": [], "Test": []}
        self.epoch_real = {"Train": [], "Validation": [], "Test": []}


    def forward(self, x):
        """ """
        batch_size, channels, width, height = x.size()

        # (b, 1, 28, 28) -> (b, 1*28*28)
        x = x.view(batch_size, -1)
        x = self.l1(x)
        x = torch.relu(x)
        x = self.l2(x)
        x = torch.relu(x)
        x = self.l3(x)
        x = torch.log_softmax(x, dim=1)
        return x

    def configure_optimizers(self):
        """ """
        return torch.optim.Adam(self.parameters(), lr=self.lr_rate)

    def log_epoch_metrics_and_loss(self, phase, print_cm=True, plot_cm=True):
        # Log loss
        epoch_loss = self.epoch_loss_sum[phase] / self.epoch_num_steps[phase]
        self.log(f"{phase}Epoch/Loss", epoch_loss, prog_bar=True, logger=True)
        self.epoch_loss_sum[phase] = 0.0

        # Log metrics
        for metric in self.metrics:
            if isinstance(metric, MulticlassConfusionMatrix):
                cm = metric(torch.cat(self.epoch_output[phase]), torch.cat(self.epoch_real[phase]))
                print(f"{phase}Epoch/CM\n", cm) if print_cm else None
                if plot_cm:
                    import seaborn as sns
                    import matplotlib.pyplot as plt
                    plt.figure(figsize=(20, 14))
                    ax = sns.heatmap(cm.numpy(), annot=True, fmt="d", cmap="Blues", annot_kws={"fontsize": 8})
                    ax.set_xlabel("Predicted labels")
                    ax.set_ylabel("True labels")
                    ax.set_title("Confusion Matrix")
                    ax.set_xticks(range(62))
                    ax.set_yticks(range(62))
                    ax.xaxis.set_ticklabels([i for i in range(62)])
                    ax.yaxis.set_ticklabels([i for i in range(62)])
                    # self.logger.experiment.add_figure(f"{phase}Epoch/CM", ax.get_figure(), global_step=self.epoch_global_number[phase])
                    plt.close()
            else:
                metric_name = metric.__class__.__name__.replace("Multiclass", "")
                metric_value = metric(torch.cat(self.epoch_output[phase]), torch.cat(self.epoch_real[phase])).detach()
                self.log(f"{phase}Epoch/{metric_name}", metric_value, prog_bar=True, logger=True)

            metric.reset()

        self.epoch_output[phase].clear()
        self.epoch_real[phase].clear()

        # Reset step count
        self.epoch_num_steps[phase] = 0

        # Increment epoch number
        self.epoch_global_number[phase] += 1

    def log_metrics(self, phase, y_pred, y, print_cm=False):
        self.epoch_output[phase].append(y_pred.detach())
        self.epoch_real[phase].append(y.detach())

        for metric in self.metrics:
            if isinstance(metric, MulticlassConfusionMatrix):
                print(f"{phase}/CM\n", metric(y_pred, y)) if print_cm else None
            else:
                metric_name = metric.__class__.__name__.replace("Multiclass", "")
                metric_value = metric(y_pred, y)
                self.log(f"{phase}/{metric_name}", metric_value, prog_bar=True, logger=True)

    def step(self, batch, phase):
        x, y = batch
        logits = self(x)
        loss = F.cross_entropy(logits, y)
        y_pred = torch.argmax(logits, dim=1)

        # Get metrics for each batch and log them
        self.log(f"{phase}/Loss", loss, prog_bar=True)
        self.log_metrics(phase, y_pred, y, print_cm=False)

        # Avoid memory leak when logging loss values
        self.epoch_loss_sum[phase] += loss
        self.epoch_num_steps[phase] += 1

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
        self.log_epoch_metrics_and_loss("Train")

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
        self.log_epoch_metrics_and_loss("Validation")

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
        self.log_epoch_metrics_and_loss("Test")

