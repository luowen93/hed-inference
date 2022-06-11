import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.transforms as transforms
import numpy as np
import pytorch_lightning as pl

from collections import OrderedDict
from sklearn.metrics import precision_score, recall_score
from PIL import Image

# Model defintion
class HED(pl.LightningModule):
    """HED network."""
    def __init__(
        self,
        finetune=False,
        beta=0.75,
        precision=32,
    ):
        super(HED, self).__init__()
        # Hyperparameters
        self.finetune = finetune
        self.beta = beta
        self.precision = precision

        # Optimizer and scheduler
        self.optimizer = None
        self.scheduler = None

        # Layers.
        self.conv1_1 = nn.Conv2d(3, 64, 3, padding=35)
        self.conv1_2 = nn.Conv2d(64, 64, 3, padding=1)

        self.conv2_1 = nn.Conv2d(64, 128, 3, padding=1)
        self.conv2_2 = nn.Conv2d(128, 128, 3, padding=1)

        self.conv3_1 = nn.Conv2d(128, 256, 3, padding=1)
        self.conv3_2 = nn.Conv2d(256, 256, 3, padding=1)
        self.conv3_3 = nn.Conv2d(256, 256, 3, padding=1)

        self.conv4_1 = nn.Conv2d(256, 512, 3, padding=1)
        self.conv4_2 = nn.Conv2d(512, 512, 3, padding=1)
        self.conv4_3 = nn.Conv2d(512, 512, 3, padding=1)

        self.conv5_1 = nn.Conv2d(512, 512, 3, padding=1)
        self.conv5_2 = nn.Conv2d(512, 512, 3, padding=1)
        self.conv5_3 = nn.Conv2d(512, 512, 3, padding=1)

        self.relu = nn.ReLU()
        # Note: ceil_mode – when True, will use ceil instead of floor to compute the output shape.
        #       The reason to use ceil mode here is that later we need to upsample the feature maps and crop the results
        #       in order to have the same shape as the original image. If ceil mode is not used, the up-sampled feature
        #       maps will possibly be smaller than the original images.
        self.maxpool = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.score_dsn1 = nn.Conv2d(64, 1, 1)  # Out channels: 1.
        self.score_dsn2 = nn.Conv2d(128, 1, 1)
        self.score_dsn3 = nn.Conv2d(256, 1, 1)
        self.score_dsn4 = nn.Conv2d(512, 1, 1)
        self.score_dsn5 = nn.Conv2d(512, 1, 1)
        self.score_final = nn.Conv2d(5, 1, 1)

        # Fixed bilinear weights
        self.up2 = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.up3 = nn.Upsample(scale_factor=4, mode="bilinear", align_corners=True)
        self.up4 = nn.Upsample(scale_factor=8, mode="bilinear", align_corners=True)
        self.up5 = nn.Upsample(scale_factor=16, mode="bilinear", align_corners=True)

        # Prepare for aligned crop.
        (
            self.crop1_margin,
            self.crop2_margin,
            self.crop3_margin,
            self.crop4_margin,
            self.crop5_margin,
        ) = self.prepare_aligned_crop()

        # Set finetuning
        if self.finetune:
            self.activate_finetune()

    # noinspection PyMethodMayBeStatic
    def prepare_aligned_crop(self):
        """Prepare for aligned crop."""
        # Re-implement the logic in deploy.prototxt and
        #   /hed/src/caffe/layers/crop_layer.cpp of official repo.
        # Other reference materials:
        #   hed/include/caffe/layer.hpp
        #   hed/include/caffe/vision_layers.hpp
        #   hed/include/caffe/util/coords.hpp
        #   https://groups.google.com/forum/#!topic/caffe-users/YSRYy7Nd9J8

        def map_inv(m):
            """Mapping inverse."""
            a, b = m
            return 1 / a, -b / a

        def map_compose(m1, m2):
            """Mapping compose."""
            a1, b1 = m1
            a2, b2 = m2
            return a1 * a2, a1 * b2 + b1

        def deconv_map(kernel_h, stride_h, pad_h):
            """Deconvolution coordinates mapping."""
            return stride_h, (kernel_h - 1) / 2 - pad_h

        def conv_map(kernel_h, stride_h, pad_h):
            """Convolution coordinates mapping."""
            return map_inv(deconv_map(kernel_h, stride_h, pad_h))

        def pool_map(kernel_h, stride_h, pad_h):
            """Pooling coordinates mapping."""
            return conv_map(kernel_h, stride_h, pad_h)

        x_map = (1, 0)
        conv1_1_map = map_compose(conv_map(3, 1, 35), x_map)
        conv1_2_map = map_compose(conv_map(3, 1, 1), conv1_1_map)
        pool1_map = map_compose(pool_map(2, 2, 0), conv1_2_map)

        conv2_1_map = map_compose(conv_map(3, 1, 1), pool1_map)
        conv2_2_map = map_compose(conv_map(3, 1, 1), conv2_1_map)
        pool2_map = map_compose(pool_map(2, 2, 0), conv2_2_map)

        conv3_1_map = map_compose(conv_map(3, 1, 1), pool2_map)
        conv3_2_map = map_compose(conv_map(3, 1, 1), conv3_1_map)
        conv3_3_map = map_compose(conv_map(3, 1, 1), conv3_2_map)
        pool3_map = map_compose(pool_map(2, 2, 0), conv3_3_map)

        conv4_1_map = map_compose(conv_map(3, 1, 1), pool3_map)
        conv4_2_map = map_compose(conv_map(3, 1, 1), conv4_1_map)
        conv4_3_map = map_compose(conv_map(3, 1, 1), conv4_2_map)
        pool4_map = map_compose(pool_map(2, 2, 0), conv4_3_map)

        conv5_1_map = map_compose(conv_map(3, 1, 1), pool4_map)
        conv5_2_map = map_compose(conv_map(3, 1, 1), conv5_1_map)
        conv5_3_map = map_compose(conv_map(3, 1, 1), conv5_2_map)

        score_dsn1_map = conv1_2_map
        score_dsn2_map = conv2_2_map
        score_dsn3_map = conv3_3_map
        score_dsn4_map = conv4_3_map
        score_dsn5_map = conv5_3_map

        upsample2_map = map_compose(deconv_map(4, 2, 0), score_dsn2_map)
        upsample3_map = map_compose(deconv_map(8, 4, 0), score_dsn3_map)
        upsample4_map = map_compose(deconv_map(16, 8, 0), score_dsn4_map)
        upsample5_map = map_compose(deconv_map(32, 16, 0), score_dsn5_map)

        crop1_margin = int(score_dsn1_map[1])
        crop2_margin = int(upsample2_map[1])
        crop3_margin = int(upsample3_map[1])
        crop4_margin = int(upsample4_map[1])
        crop5_margin = int(upsample5_map[1])

        return crop1_margin, crop2_margin, crop3_margin, crop4_margin, crop5_margin

    def forward(self, x):
        # VGG-16 network.

        image_h, image_w = x.shape[2], x.shape[3]
        conv1_1 = self.relu(self.conv1_1(x))
        conv1_2 = self.relu(self.conv1_2(conv1_1))  # Side output 1.
        pool1 = self.maxpool(conv1_2)

        conv2_1 = self.relu(self.conv2_1(pool1))
        conv2_2 = self.relu(self.conv2_2(conv2_1))  # Side output 2.
        pool2 = self.maxpool(conv2_2)

        conv3_1 = self.relu(self.conv3_1(pool2))
        conv3_2 = self.relu(self.conv3_2(conv3_1))
        conv3_3 = self.relu(self.conv3_3(conv3_2))  # Side output 3.
        pool3 = self.maxpool(conv3_3)

        conv4_1 = self.relu(self.conv4_1(pool3))
        conv4_2 = self.relu(self.conv4_2(conv4_1))
        conv4_3 = self.relu(self.conv4_3(conv4_2))  # Side output 4.
        pool4 = self.maxpool(conv4_3)

        conv5_1 = self.relu(self.conv5_1(pool4))
        conv5_2 = self.relu(self.conv5_2(conv5_1))
        conv5_3 = self.relu(self.conv5_3(conv5_2))  # Side output 5.

        score_dsn1 = self.score_dsn1(conv1_2)
        score_dsn2 = self.score_dsn2(conv2_2)
        score_dsn3 = self.score_dsn3(conv3_3)
        score_dsn4 = self.score_dsn4(conv4_3)
        score_dsn5 = self.score_dsn5(conv5_3)

        upsample2 = self.up2(score_dsn2)
        upsample3 = self.up3(score_dsn3)
        upsample4 = self.up4(score_dsn4)
        upsample5 = self.up5(score_dsn5)

        # Aligned cropping.
        crop1 = score_dsn1[
            :,
            :,
            self.crop1_margin : self.crop1_margin + image_h,
            self.crop1_margin : self.crop1_margin + image_w,
        ]
        crop2 = upsample2[
            :,
            :,
            self.crop2_margin : self.crop2_margin + image_h,
            self.crop2_margin : self.crop2_margin + image_w,
        ]
        crop3 = upsample3[
            :,
            :,
            self.crop3_margin : self.crop3_margin + image_h,
            self.crop3_margin : self.crop3_margin + image_w,
        ]
        crop4 = upsample4[
            :,
            :,
            self.crop4_margin : self.crop4_margin + image_h,
            self.crop4_margin : self.crop4_margin + image_w,
        ]
        crop5 = upsample5[
            :,
            :,
            self.crop5_margin : self.crop5_margin + image_h,
            self.crop5_margin : self.crop5_margin + image_w,
        ]

        # Concatenate according to channels.
        fuse_cat = torch.cat((crop1, crop2, crop3, crop4, crop5), dim=1)
        # Shape: [batch_size, 1, image_h, image_w].
        fuse = self.score_final(fuse_cat)
        results = [crop1, crop2, crop3, crop4, crop5, fuse]
        if self.precision == 32:
            results = [torch.sigmoid(r) for r in results]
        results_tuple = tuple(results)
        return results_tuple

    def configure_optimizers(self):
        if self.optimizer and self.scheduler:
            return [self.optimizer], [self.scheduler]
        elif self.optimizer:
            return self.optimizer
        else:
            print("No optimizer defined")
            return None
        

    def training_step(self, batch, batch_idx):
        x, y, _, fnames = batch
        preds_tuple = self.forward(x)  # List of outputs

        # Sum over all separate predictions
        loss = sum(
            [self.weighted_cross_entropy_loss(preds, y) for preds in preds_tuple]
        )

        # self.scheduler.step()

        self.log("train_loss", loss)  # wandb logging
        self.log("epoch", self.current_epoch)

        return loss

    def validation_step(self, batch, batch_idx):
        x, y, _, fnames = batch

        preds_tuple = self.forward(x)  # List of outputs
        # Sum over all separate predictions
        loss = sum(
            [self.weighted_cross_entropy_loss(preds, y) for preds in preds_tuple]
        )
        self.log("val_loss", loss)  # wandb logging
        self.log("epoch", self.current_epoch)

        if self.precision == 16:
            preds_tuple = self.post_process_outpout(preds_tuple)

        # log f1, precision and recall metrics
        fuse_images = preds_tuple[-1]
        total_f1 = 0
        total_prec = 0
        total_rec = 0
        num_total = fuse_images.shape[0]  # num of images in the batch
        for i in range(num_total):
            truth = y[i].detach().cpu().numpy()[0]
            prediction = fuse_images[i].detach().cpu().numpy()[0]
            prediction[prediction < 0.5] = 0
            prediction[prediction >= 0.5] = 1
            truth[truth < 0.5] = 0
            truth[truth >= 0.5] = 1
            prediction = prediction.flatten()
            truth = truth.flatten()
            precision = precision_score(
                truth, prediction, pos_label=1, average="binary", zero_division=0
            )
            recall = recall_score(
                truth, prediction, pos_label=1, average="binary", zero_division=0
            )
            if (precision + recall) == 0:
                f1_score = 0
            else:
                f1_score = 2 * (precision * recall) / (precision + recall)
            total_f1 += f1_score
            total_prec += precision
            total_rec += recall
        self.log("f1", total_f1 / num_total)
        self.log("precision", total_prec / num_total)
        self.log("recall", total_rec / num_total)

        return loss

    def test_step(self, batch, batch_idx):
        x, y, _, fnames = batch

        preds_tuple = self.forward(x)  # List of outputs
        if self.precision == 16:
            preds_tuple = self.post_process_outpout(preds_tuple)
        fuse_images = preds_tuple[-1]
        for i in range(len(fuse_images)):
            orig_pil = transforms.ToPILImage()(x[i])
            output_pil = transforms.ToPILImage()(fuse_images[i])
            total_w = orig_pil.size[0] * 2
            total_h = orig_pil.size[1]
            concat_pil = Image.new("RGB", (total_w, total_h))
            concat_pil.paste(orig_pil, (0, 0))
            concat_pil.paste(output_pil, (orig_pil.size[0], 0))
            concat_pil.save("results/img{}_{}.png".format(batch_idx, i), "PNG")

    # Loss function
    def weighted_cross_entropy_loss(self, preds, edges):
        """Calculate sum of weighted cross entropy loss."""
        # Reference:
        #   hed/src/caffe/layers/sigmoid_cross_entropy_loss_layer.cpp
        #   https://github.com/s9xie/hed/issues/7
        mask = (edges > 0.5).float()
        b, c, h, w = mask.shape
        num_pos = torch.sum(mask, dim=[1, 2, 3]).float()  # Shape: [b,].
        num_neg = c * h * w - num_pos  # Shape: [b,].
        weight = torch.zeros_like(mask)  # B x C x H x W

        wp = ((1 - self.beta) * num_neg / (num_pos + num_neg)).view(b, 1, 1, 1)
        wn = (self.beta * num_pos / (num_pos + num_neg)).view(b, 1, 1, 1)

        # Express the mask as a weighted sum
        weight = torch.mul(mask, wp) + torch.mul(1 - mask, wn)

        # Calculate loss using logits function (autocasting into 16 bit precision available)
        if self.precision == 16:
            losses = torch.nn.functional.binary_cross_entropy_with_logits(
                preds, edges, weight=weight, reduction="none"
            )
        else:
            losses = torch.nn.functional.binary_cross_entropy(
                preds.float(), edges.float(), weight=weight, reduction="none"
            )
        loss = torch.sum(losses) / b  # Average
        return loss

    # Enables finetuning
    def activate_finetune(self):
        for name, param in self.named_parameters():
            if name in [
                "conv1_1.weight",
                "conv2_1.weight",
                "conv3_1.weight",
                "conv3_2.weight",
                "conv4_1.weight",
                "conv4_2.weight",
            ]:
                param.requires_grad = False
            elif name in [
                "conv1_1.bias",
                "conv2_1.bias",
                "conv3_1.bias",
                "conv3_2.bias",
                "conv4_1.bias",
                "conv4_2.bias",
            ]:
                param.requires_grad = False
            elif name in ["conv5_1.weight", "conv5_2.weight"]:
                param.requires_grad = False
            elif name in ["conv5_1.bias", "conv5_2.bias"]:
                param.requires_grad = False

    # Applies sigmoid to the output from the forward method and return as a list
    def post_process_outpout(self, output):
        results = [torch.sigmoid(r) for r in output]
        return tuple(results)

# Loads the pretrained model
# pretrained_path: path to pretrained model or None
def load_model(pretrained_path, finetune=False, beta=0.5, precision=32):
    if pretrained_path is None:
        model = HED(finetune=finetune, beta=beta, precision=precision)
    elif pretrained_path[-4:] == "ckpt":
        model = HED.load_from_checkpoint(pretrained_path, finetune=finetune, beta=beta, precision=precision)
    else:
        model = HED(finetune=finetune, beta=beta, precision=precision)
        state_dict = torch.load(pretrained_path)['net']
        weights_dict = OrderedDict()
        for k,v in state_dict.items():
            if('module.' in k):
                newkey = k.replace('module.','')
                weights_dict[newkey] = v
            else:
                weights_dict[k] = v
        model.load_state_dict(weights_dict)
    return model