import torch
from PIL import Image

class Resize:
    def __init__(self, target_w, target_h):
        self.target_w = target_w
        self.target_h = target_h
    
    def __call__(self, img):
        if img.mode != 'RGB':
            img = img.convert('RGB')
        w,h = img.size
        ratio = min(self.target_h/h, self.target_w/w)
        new_width = int(ratio*w)
        new_height = int(ratio*h)

        img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)

        bg = Image.new("RGB",(self.target_w, self.target_h), (0,0,0))
        x = (self.target_w - new_width)//2
        y = (self.target_h - new_height)//2
        bg.paste(img, (x,y))
    
        return bg

def mean_std(data_loader):
    mean = torch.zeros(3)
    ss = torch.zeros(3)

    total_pixel = 0

    for images, _ in data_loader:
        batch, channels, height, width = images.size()
        total_pixel += (batch * height * width)
        mean += images.sum((0,2,3))
        ss += (images ** 2).sum((0,2,3))
    
    mean /= total_pixel
    std = (ss/total_pixel - mean**2).sqrt()

    return mean, std
