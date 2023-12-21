import torch.nn as nn

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        self.conv_scratch = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.2)
        )

        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 128 * 128, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.conv_scratch(x)
        x = self.maxpool(x)
        x = self.fc(x)
        return x