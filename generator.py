import torch.nn as nn
import torch


class Generator(nn.Module):
    def __init__(self, text_embedding_dim=256):
        super(Generator, self).__init__()
        #encoder
        # 3 to 64
        self.conv_vgg1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )

        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)

        # 64 to 128
        self.conv_vgg2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU()
        )

        # feature funsion
        self.text_feature_conv = nn.Conv2d(128, 128, kernel_size=2)
        self.fusion_conv = nn.Conv2d(256, 128, kernel_size=3, padding=1)

        #decoder
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        self.conv_scratch = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )

        self.sigmoid = nn.Sequential(
            nn.Conv2d(64, 1, kernel_size=3, padding=1),
            nn.Sigmoid()
        )

    def forward(self, x, text_features):
        # encoder
        x = self.conv_vgg1(x)
        #print(f'vgg1: {x.shape}')
        x = self.maxpool(x)
        #print(f'mp1: {x.shape}')
        x = self.conv_vgg2(x)
        #print(f'vgg2: {x.shape}')
        #shape:[batch,128,128,128]

        # text feature fusion
        text_feat = text_features.reshape(text_features.size(0), 128, 2, 2).float()
        text_feat = self.text_feature_conv(text_feat)
        #print(f'text_feat: {text_feat.shape}')
        text_feat = text_feat.expand(-1, -1, 128, 128)
        #print(f'ex_text_feat: {text_feat.shape}')
        cat_feat = torch.cat([x, text_feat], dim=1)
        #print(f'cat_feat: {cat_feat.shape}')
        fusion_feat = self.fusion_conv(cat_feat)
        #print(f'ex_cat_feat: {fusion_feat.shape}')

        #shape:[batch,128,128,128]
        # decoder
        x = self.upsample(fusion_feat)
        #print(f'us: {x.shape}')
        x = self.conv_scratch(x)
        #print(f'sc: {x.shape}')

        x = self.sigmoid(x)
        return x
    


if __name__ == '__main__':
    g = Generator()
    x = torch.rand([17, 3, 256, 256])
    text = torch.rand([17,512])
    print(text.shape)
    print('Input :', x.size())
    out = g(x,text)
    print('Output: ', out.size())