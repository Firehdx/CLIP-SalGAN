from pandas import read_excel
import json



data = dict()
images = read_excel('saliency/text最终版本.xlsx', sheet_name='部分-实验设置', usecols='A').values
categories = read_excel('saliency/text最终版本.xlsx', sheet_name='部分-实验设置', usecols='B').values
texts = read_excel('saliency/text最终版本.xlsx', sheet_name='部分-实验设置', usecols='C').values

for i in range(len(images)):
    image = str(images[i].item())
    category = categories[i].item()

    if category == '整体':
        data[image+"_0"] = ""
        image += "_1" 
    elif category == '非显著':
        image += "_2"
    else:
        image += "_3"

    data[image] = texts[i].item()

images = read_excel('saliency/text最终版本.xlsx', sheet_name='整体', usecols='A').values
texts = read_excel('saliency/text最终版本.xlsx', sheet_name='整体', usecols='B').values

for i in range(len(images)):
    image = str(images[i].item())
    data[image] = texts[i].item()
    image += "_0"
    data[image] = ""

with open('image_text.json', 'w') as f:
    json.dump(data, f, ensure_ascii=True, indent=4)
