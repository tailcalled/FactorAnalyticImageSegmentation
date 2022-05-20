import torchvision.models.resnet as resnet

import torch
import torch.nn.functional as F


def get_encoder_channel_counts(encoder_name):
    basic_block_models = ["resnet18", "resnet34"]
    is_basic_block = encoder_name in basic_block_models
    encoder_output_channels = 512 if is_basic_block else 2048
    encoder_output_channels_4x = 64 if is_basic_block else 256
    return encoder_output_channels, encoder_output_channels_4x


class DeepLabV3pColor(torch.nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.encoder = ResNetEncoder(
            cfg.encoder_model_name,
            pretrained=cfg.pretrained,
            zero_init_residual=True,
        )

        encoder_out_ch, encoder_out_ch_4x = get_encoder_channel_counts(
            cfg.encoder_model_name
        )

        self.layer1 = ASPPpart(
            in_channels=encoder_out_ch,
            out_channels=256,
            kernel_size=3,
        )

        self.layer2 = ASPPpart(in_channels=256, out_channels=256, kernel_size=3)
        self.decoder = Decoder(256, encoder_out_ch_4x, 3 + 3 * cfg.model_latent_dim + 1)

    def forward(self, x):
        features = self.encoder(x)

        last_layer_features = max(features.keys())
        features_last_layer = features[last_layer_features]
        out = self.layer1(features_last_layer)
        out = self.layer2(out)
        preds_4x, _ = self.decoder(out, features[4])

        color_preds = preds_4x[:, :3, :, :]
        factor_preds = preds_4x[:, 3:-1, :, :]
        error_preds = torch.exp(preds_4x[:, -1:, :, :])

        return color_preds, factor_preds, error_preds


class Decoder(torch.nn.Module):
    def __init__(self, in_channels, skip_4x_ch, num_out_ch):
        super(Decoder, self).__init__()
        self.conv_1x1 = ASPPpart(
            skip_4x_ch, 48, kernel_size=1, stride=1, padding=0, dilation=1
        )

        self.conv_3x3 = torch.nn.Sequential(
            ASPPpart(
                in_channels + 48,
                256,
                kernel_size=3,
                stride=1,
                padding=1,
                dilation=1,
            ),
            ASPPpart(256, 256, kernel_size=3, stride=1, padding=1, dilation=1),
        )

        self.final_conv_1x1 = torch.nn.Conv2d(
            256, num_out_ch, kernel_size=1, stride=1, dilation=1
        )

    def forward(self, features_bottleneck, features_skip_4x):
        """
        DeepLabV3+ style decoder
        :param features_bottleneck: bottleneck features of scale > 4
        :param features_skip_4x: features of encoder of scale == 4
        :return: features with 256 channels and the final tensor of predictions
        """
        features_4x = F.interpolate(
            features_bottleneck,
            size=features_skip_4x.shape[2:],
            mode="bilinear",
            align_corners=False,
        )

        out_1x1 = self.conv_1x1(features_skip_4x)
        out_concat = torch.cat((features_4x, out_1x1), dim=1)
        out_3x3 = self.conv_3x3(out_concat)
        out_final = self.final_conv_1x1(out_3x3)

        return out_final, out_3x3  # predictions_4x, features_4x


class ResNetEncoder(torch.nn.Module):
    def __init__(self, name, **kwargs):
        super().__init__()
        encoder = self._create(name, **kwargs)
        del encoder.avgpool
        del encoder.fc
        self.encoder = encoder

    def _create(self, name, **kwargs):
        pretrained = kwargs.pop("pretrained", False)
        progress = kwargs.pop("progress", True)
        if name == "resnet18":
            model = resnet.resnet18(pretrained=pretrained, progress=progress, **kwargs)
        elif name == "resnet34":
            model = resnet.resnet34(pretrained=pretrained, progress=progress, **kwargs)
        elif name == "resnet50":
            model = resnet.resnet50(pretrained=pretrained, progress=progress, **kwargs)
        else:
            raise ValueError(f"Unsupported model: {name}")
        return model

    def update_skip_dict(self, skips, x, inp_size):
        rem, scale = inp_size % x.shape[3], inp_size // x.shape[3]
        assert rem == 0
        skips[scale] = x

    def forward(self, x):
        out = {1: x}
        inp_size = x.shape[3]
        x = self.encoder.conv1(x)
        x = self.encoder.bn1(x)
        x = self.encoder.relu(x)
        self.update_skip_dict(out, x, inp_size)

        x = self.encoder.maxpool(x)
        self.update_skip_dict(out, x, inp_size)

        x = self.encoder.layer1(x)
        self.update_skip_dict(out, x, inp_size)

        x = self.encoder.layer2(x)
        self.update_skip_dict(out, x, inp_size)

        x = self.encoder.layer3(x)
        self.update_skip_dict(out, x, inp_size)

        x = self.encoder.layer4(x)
        self.update_skip_dict(out, x, inp_size)
        return out


class ASPPpart(torch.nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 0,
        dilation: int = 1,
    ):
        super().__init__()
        self.layers = torch.nn.Sequential(
            torch.nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size,
                stride,
                padding,
                dilation,
                bias=False,
            ),
            torch.nn.BatchNorm2d(out_channels),
            torch.nn.ReLU(),
        )

    def forward(self, x):
        out = self.layers(x)
        return out
