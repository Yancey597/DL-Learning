from torch.utils.data import Dataset
from PIL import Image
import os

class MyData(Dataset):
    def __init__(self, root_dir, label_dir):
        self.root_dir = root_dir
        self.label_dir = label_dir
        self.path = os.path.join(self.root_dir, self.label_dir)
        self.img_path = os.listdir(self.path)  # 这里是图片文件名列表

    def __getitem__(self, index):
        img_name = self.img_path[index]  # 取出图片文件名（字符串）
        img_item_path = os.path.join(self.path, img_name)  # 拼接成完整路径
        img = Image.open(img_item_path)
        label = self.label_dir
        return img, label

    def __len__(self):
        return len(self.img_path)

root_dir = "dataset/train"
label_dir = "ants"
ants_dataset = MyData(root_dir, label_dir)
print(ants_dataset[0])
