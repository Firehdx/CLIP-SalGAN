import torch
import clip
import os
import json
import random
from torch.utils.data import Dataset, DataLoader
from PIL import Image

device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)



class CLIPSalDataset(Dataset):
    def __init__(self, image_paths, target_paths, text_list, transform=None):
        self.image_paths = image_paths
        self.target_paths = target_paths
        self.transform = transform
        self.text_list = []
        with torch.no_grad():
            for i in range(len(image_paths)):
                self.text_list.append(model.encode_text(
                    clip.tokenize([text_list[i]]).to(device)))

    def __getitem__(self, idx):
        image = Image.open(self.image_paths[idx]).convert('RGB')
        target = Image.open(self.target_paths[idx]).convert('L')
        text = self.text_list[idx]
        text_feature = torch.tensor(text, dtype=torch.long)

        if self.transform:
            image = self.transform(image)
            target = self.transform(target)

        return image, target, text_feature
    
    def __len__(self):
        return len(self.image_paths)
    
#type = 'total', 'general', 'sal', 'non_sal'
def path_processer(paths, type='total'):
    if type == 'total':
        return paths
    elif type == 'general':
        return [i for i in paths if "_1.png" in i or "_0.png" in i or ("_2" not in i  and "_3" not in i)]
    elif type == 'non_sal':
        return [i for i in paths if '_2.png' in i or '_0.png' in i]
    elif type == 'sal':
        return [i for i in paths if '_3.png' in i or '_0.png' in i]
    else:
        return None

# the data for total model training
def get_data(image_path, target_path, type='total'):
    with open('image_text.json', 'r') as f:
        image_text = json.load(f)
    text = []

    image_paths = [image_path + "/" + f for f in os.listdir(image_path) if os.path.isfile(os.path.join(image_path, f))]
    target_paths = [target_path + "/" + f for f in os.listdir(target_path) if os.path.isfile(os.path.join(target_path, f))]
    image_paths = path_processer(image_paths, type)
    target_paths = path_processer(target_paths, type)

    for path in target_paths:
        path = path[path.rfind('/') + 1:]
        first_nonzero_index = None

        for i in range(len(path)):
            if path[i] != '0':
                first_nonzero_index = i
                break
        if first_nonzero_index != None:
            path = path[first_nonzero_index:]
        
        path = path[:path.find('.') ]  
        text.append(image_text[path])

    return image_paths, target_paths, text
 
def split_data(image_paths, target_paths, texts):
    data = list(zip(image_paths, target_paths, texts))
    random.shuffle(data)

    total_size = len(data)
    train_size = int(total_size * 0.7)
    val_size = int(total_size * 0.15)

    train_data = data[:train_size]
    val_data = data[train_size:train_size + val_size]
    test_data = data[train_size + val_size:]

    return train_data, val_data, test_data

def dataloader(data, transform, batch_size = 32, shuffle=True):
    image_paths, target_paths, text_list = zip(*data)
    dataset = CLIPSalDataset(list(image_paths), list(target_paths), list(text_list), transform=transform)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)