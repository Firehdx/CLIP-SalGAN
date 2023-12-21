import torch
from torchvision import transforms
import torch.nn as nn
from dataset import *
import train, generator, discriminator


def main(task = 'total'):# 'sal', 'non_sal', 'general', 'total'
    batch_size = 16
    num_epochs = 80
    image_path = 'saliency/image'
    target_path = 'saliency/map'
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    criterion = nn.BCELoss()

    image_paths, target_paths, text = get_data(image_path, target_path, task)

    train_data, val_data, test_data = split_data(image_paths, target_paths, text)

    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor()
    ])

    train_loader = dataloader(train_data, transform, batch_size=batch_size)
    val_loader = dataloader(val_data, transform, batch_size=batch_size)
    test_loader = dataloader(test_data, transform, batch_size=batch_size, shuffle=False)

    gen = generator.Generator().to(device)
    dis = discriminator.Discriminator().to(device)

    g_opt = torch.optim.Adam(gen.parameters(), lr=0.0001, betas=(0.5, 0.999))
    d_opt = torch.optim.Adam(dis.parameters(), lr=0.0001, betas=(0.5, 0.999))

    train.train(train_loader, val_loader, gen, dis, criterion, g_opt, d_opt, device, task, num_epochs=num_epochs)
    torch.save(gen.state_dict(), 'generator_{}.pt'.format(task))
    torch.save(dis.state_dict(), 'discriminator_{}.pt'.format(task))

if __name__ == '__main__':
    for task in ['total', 'general', 'sal', 'non_sal']:
        main(task=task)