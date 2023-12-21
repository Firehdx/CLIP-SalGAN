import torch.nn as nn
import torch
from torchvision import transforms
from PIL import Image
from generator import *
from discriminator import *
from metrics import *
from dataset import *
import clip
import numpy as np



batch_size = 4
task = 'non_sal'
test_size = 300

image_path = 'saliency/image'
target_path = 'saliency/map'

device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)

# Define the transformations
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor()
])

criterion = nn.BCELoss()
generator = Generator()
discriminator = Discriminator()
generator.load_state_dict(torch.load(f'model/generator_{task}.pt', map_location = torch.device('cpu')))
discriminator.load_state_dict(torch.load(f'model/discriminator_{task}.pt', map_location = torch.device('cpu')))

image_paths, target_paths, texts = get_data(image_path, target_path, task)
test_idx = np.random.randint(0, len(image_paths), size=test_size)
image_paths = [image_paths[i] for i in test_idx]
target_paths = [target_paths[i] for i in test_idx]
texts = [texts[i] for i in test_idx]
test_data = list(zip(image_paths, target_paths, texts))
test_loader = dataloader(test_data, transform, batch_size, shuffle=False)


generator.eval()
discriminator.eval()
auc = 0.0
sauc = 0.0
cc = 0.0
nss = 0.0
with torch.no_grad():
    for m, (images, targets, text_features) in enumerate(test_loader):
        img_size = Image.open(image_paths[m]).size
       
        fake_targets = generator(images, text_features)
        output = discriminator(fake_targets)
        picture = fake_targets.squeeze(0)

        AUC_score = AUC(fake_targets, targets)
        sAUC_score = sAUC(fake_targets, targets)
        CC_score = CC(fake_targets, targets)
        NSS_score = NSS(fake_targets, targets)

        auc += AUC_score
        sauc += sAUC_score
        cc += CC_score
        nss += NSS_score
        print("AUC Score: {}".format(AUC_score))
        print("sAUC Score: {}".format(sAUC_score))
        print("CC Score: {}".format(CC_score))
        print("NSS Score: {}".format(NSS_score))

auc /= len(test_loader)
sauc /= len(test_loader)
cc /= len(test_loader)
nss /= len(test_loader)

print(f'Task: {task} Mean scores: AUC: {auc}, sAUC: {sauc}, CC: {cc}, NSS: {nss}')