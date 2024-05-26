from torch import nn, Tensor, cat


def contracting_block(in_channels: int, out_channels: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True)
    )


def expansive_block(in_channels: int, out_channels: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True)
    )


def final_block(in_channels: int, out_channels: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1),
        nn.BatchNorm2d(in_channels),
        nn.ReLU(inplace=True),
        nn.Conv2d(in_channels, out_channels, kernel_size=1)
    )


def crop_and_concat(enc: Tensor, dec: Tensor) -> Tensor:
    enc_size = enc.size()[2:]
    dec_size = dec.size()[2:]

    crop_size = [(enc_size[i] - dec_size[i]) // 2 for i in range(len(enc_size))]

    enc = enc[:, :, crop_size[0]:crop_size[0] + dec_size[0], crop_size[1]:crop_size[1] + dec_size[1]]
    return enc


class UNet(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.max_pool = nn.MaxPool2d(2)
        self.encoder1 = contracting_block(in_channels, 64)
        self.encoder2 = contracting_block(64, 128)
        self.encoder3 = contracting_block(128, 256)
        self.encoder4 = contracting_block(256, 512)

        self.middle = contracting_block(512, 1024)

        self.upconv4 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.decoder4 = expansive_block(1024, 512)
        self.upconv3 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.decoder3 = expansive_block(512, 256)
        self.upconv2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.decoder2 = expansive_block(256, 128)
        self.upconv1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.decoder1 = expansive_block(128, 64)
        self.final_layer = final_block(64, out_channels)

    def forward(self, x: Tensor) -> Tensor:
        enc1 = self.encoder1(x)
        enc2 = self.encoder2(self.max_pool(enc1))
        enc3 = self.encoder3(self.max_pool(enc2))
        enc4 = self.encoder4(self.max_pool(enc3))

        middle = self.middle(self.max_pool(enc4))

        up4 = self.upconv4(middle)
        dec4 = self.decoder4(cat([up4, crop_and_concat(enc4, up4)], dim=1))

        up3 = self.upconv3(dec4)
        dec3 = self.decoder3(cat([up3, crop_and_concat(enc3, up3)], dim=1))

        up2 = self.upconv2(dec3)
        dec2 = self.decoder2(cat([up2, crop_and_concat(enc2, up2)], dim=1))

        up1 = self.upconv1(dec2)
        dec1 = self.decoder1(cat([up1, crop_and_concat(enc1, up1)], dim=1))

        out = self.final_layer(dec1)
        return out
