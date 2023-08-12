import torch
import torch.nn as nn


class VGGNet(nn.Module):
    def __init__(self, architecture, input_channels=3, num_classes=3):
        super().__init__()
        self.input_channels = input_channels
        self.conv_layers = self._create_layer(architecture)
        self.fcs = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(4096, num_classes),
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = x.view(x.shape[0], -1)
        x = self.fcs(x)
        return x

    def _create_layer(self, architecture):
        layers = []
        in_channels = self.input_channels

        for x in architecture:
            if type(x) == int:
                out_channels = x

                layers += [
                    nn.Conv2d(
                        in_channels=in_channels,
                        out_channels=out_channels,
                        kernel_size=(3, 3),
                        stride=(1, 1),
                        padding=(1, 1),
                    ),
                    nn.BatchNorm2d(x),
                    nn.ReLU(),
                ]
                in_channels = x
            elif x == "M":
                layers += [nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))]

        return nn.Sequential(*layers)


if __name__ == "__main__":
    VGG_types = {
        "VGG11": [64, "M", 128, "M", 256, 256, "M", 512, 512, "M", 512, 512, "M"],
        "VGG13": [
            64,
            64,
            "M",
            128,
            128,
            "M",
            256,
            256,
            "M",
            512,
            512,
            "M",
            512,
            512,
            "M",
        ],
        "VGG16": [
            64,
            64,
            "M",
            128,
            128,
            "M",
            256,
            256,
            256,
            "M",
            512,
            512,
            512,
            "M",
            512,
            512,
            512,
            "M",
        ],
        "VGG19": [
            64,
            64,
            "M",
            128,
            128,
            "M",
            256,
            256,
            256,
            256,
            "M",
            512,
            512,
            512,
            512,
            "M",
            512,
            512,
            512,
            512,
            "M",
        ],
    }

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = VGGNet(VGG_types["VGG16"], input_channels=3, num_classes=1000).to(device)
    BATCH_SIZE = 3
    x = torch.randn(3, 3, 224, 224).to(device)
    assert model(x).shape == torch.Size([BATCH_SIZE, 1000])
    print(model(x).shape)
