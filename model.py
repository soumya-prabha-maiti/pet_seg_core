import lightning as pl
import torch
import torchvision.transforms.functional as TF
from torch import nn
from torchmetrics.functional.segmentation import mean_iou
from torchmetrics.classification import MulticlassConfusionMatrix
from pet_seg_core.config import PetSegTrainConfig
from functools import partial


class DoubleConvOriginal(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DoubleConvOriginal, self).__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                out_channels,
                out_channels,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.double_conv(x)

class DoubleConvDepthwiseSep(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DoubleConvDepthwiseSep, self).__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(
                in_channels,
                in_channels,
                kernel_size=3,
                stride=1,
                padding=1,
                groups=in_channels,
                bias=False,
            ),
            nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                out_channels,
                out_channels,
                kernel_size=3,
                stride=1,
                padding=1,
                groups=out_channels,
                bias=False,
            ),
            nn.Conv2d(out_channels, out_channels, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.double_conv(x)


class UNet(pl.LightningModule):
    def __init__(
        self,
        in_channels,
        out_channels,
        channels_list=[64, 128, 256, 512],
        depthwise_sep=False,
    ):
        super(UNet, self).__init__()
        self.save_hyperparameters() 
        
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.encoder = nn.ModuleList()
        self.decoder = nn.ModuleList()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        if depthwise_sep:
            DoubleConv = DoubleConvDepthwiseSep
        else:
            DoubleConv = DoubleConvOriginal

        # Encoder
        for channels in channels_list:
            self.encoder.append(DoubleConv(in_channels, channels))
            in_channels = channels

        # Decoder
        for channels in channels_list[::-1]:
            self.decoder.append(
                nn.ConvTranspose2d(channels * 2, channels, kernel_size=2, stride=2)
            )
            self.decoder.append(DoubleConv(channels * 2, channels))

        self.bottleneck = DoubleConv(channels_list[-1], channels_list[-1] * 2)
        self.out = nn.Conv2d(channels_list[0], out_channels, kernel_size=1)

        self.loss_fn = nn.CrossEntropyLoss()

        self.iou = partial(mean_iou, num_classes=out_channels)
        self.conf_mat = MulticlassConfusionMatrix(num_classes=out_channels)

    def forward(self, x):
        skip_connections = []
        for i, enc_block in enumerate(self.encoder):
            x = enc_block(x)
            skip_connections.append(x)
            x = self.pool(x)

        x = self.bottleneck(x)

        skip_connections = skip_connections[::-1]

        for i in range(0, len(self.decoder), 2):
            x = self.decoder[i](x)
            skip_connection = skip_connections[i // 2]

            if x.shape != skip_connection.shape:
                x = TF.resize(x, size=skip_connection.shape[2:])

            concat_skip = torch.cat(
                (skip_connection, x), dim=1
            )  # Concatenate along the channel dimension
            x = self.decoder[i + 1](concat_skip)

        x = self.out(x)

        return x

    def _common_step(self, batch, batch_idx, prefix):
        x, y = batch
        y = (y * 255 - 1).long().squeeze(1)  # move to dataloader
        y_hat = self(x)

        loss = self.loss_fn(y_hat, y)
        self.log(f"{prefix}_loss", loss.item(), prog_bar=True)

        y_hat_argmax = torch.argmax(y_hat, dim=1)
        
        y_hat_argmax_onehot = torch.nn.functional.one_hot(y_hat_argmax, num_classes=self.out_channels).permute(0, 3, 1, 2)
        y_onehot = torch.nn.functional.one_hot(y, num_classes=self.out_channels).permute(0, 3, 1, 2)
        iou = self.iou(y_hat_argmax_onehot, y_onehot)
        # self.log(f"{prefix}_iou", iou.mean().item(), prog_bar=True)

        self.conf_mat.update(y_hat_argmax, y)

        return y_hat, loss

    def training_step(self, batch, batch_idx):
        y_hat, loss = self._common_step(batch, batch_idx, "train")
        return loss

    def validation_step(self, batch, batch_idx):
        y_hat, loss = self._common_step(batch, batch_idx, "val")

    def test_step(self, batch, batch_idx):
        y_hat, loss = self._common_step(batch, batch_idx, "test")

    def _common_on_epoch_end(self, prefix):
        confmat = self.conf_mat.compute()
        
        for i in range(self.out_channels):
            for j in range(self.out_channels):
                self.log(f'{prefix}_confmat_true={i}_pred={j}', confmat[i][j].item(), prog_bar=True)
        
        iou = torch.zeros(self.out_channels)
        for i in range(self.out_channels):
            true_positive = confmat[i, i]
            false_positive = confmat.sum(dim=0)[i] - true_positive
            false_negative = confmat.sum(dim=1)[i] - true_positive
            union = true_positive + false_positive + false_negative
            if union > 0:
                iou[i] = true_positive / union
            else:
                iou[i] = float('nan')
            self.log(f'{prefix}_iou_class={i}', iou[i].item(), prog_bar=True)
            
        self.conf_mat.reset()

    def on_train_epoch_end(self):
        self._common_on_epoch_end("train")
        
    def on_validation_epoch_end(self):
        self._common_on_epoch_end("val")

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=PetSegTrainConfig.LEARNING_RATE)
