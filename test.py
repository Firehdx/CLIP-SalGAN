import torch
from torchvision import transforms
from PIL import Image
from generator import *
from discriminator import *
import clip
import torchvision.transforms.functional as TF
from dataset import *



if __name__ == '__main__':
    task = 'total' # 'sal', 'non_sal", 'general', 'total'
    text_type = 'non_sal' # 'general', 'sal', 'non_sal'

    image_paths = ['saliency/image/000000021571_0.png']
    target_paths = ['saliency/map/000000021571_0.png']
    save_paths = ['000000021571_sal.png']
    # you can find them in prepocessed image_text.json
    text_options = {
        "sal": "A dog stares out the window",
        "non_sal": "Gray cars were parked in the courtyard",
        "general": "A dog is sitting in the window looking at a car"
    }
    texts = [text_options[text_type]]

    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor()
    ])



    test_data = list(zip(image_paths, target_paths, texts))
    test_loader = dataloader(test_data, transform, batch_size=1, shuffle=False)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load("ViT-B/32", device=device)
    generator = Generator()
    discriminator = Discriminator()
    generator.load_state_dict(torch.load('model/generator_{}.pt'.format(task), map_location=torch.device('cpu')))
    discriminator.load_state_dict(torch.load('model/discriminator_{}.pt'.format(task), map_location=torch.device('cpu')))

    generator.eval()
    discriminator.eval()
    with torch.no_grad():
        for i, (image, target, text_feature) in enumerate(test_loader):
            img_size = Image.open(image_paths[i]).size
            salmap = generator(image, text_feature).squeeze(0)
            saliency_map = TF.to_pil_image(salmap).resize(img_size, Image.BILINEAR)
            saliency_map.save(save_paths[i])