import torch
import torchvision

class VGG(torch.nn.Module):
    def __init__(self, resize=False, pooling=False, levels=2):
        super(VGG, self).__init__()
        layer_id = [0, 4, 9, 16, 23]
        if pooling:
            layer_id = [0, 5, 10, 17, 24]
        blocks = []
        for i in range(levels):
            blocks.append(torchvision.models.vgg16(pretrained=True).features[layer_id[i]:layer_id[i+1]].eval())
        for bl in blocks:
            for p in bl.parameters():
                p.requires_grad = False
        self.blocks = torch.nn.ModuleList(blocks)
        self.transform = torch.nn.functional.interpolate
        self.mean = torch.nn.Parameter(torch.tensor([0.485, 0.456, 0.406]).view(1,3,1,1))
        self.std = torch.nn.Parameter(torch.tensor([0.229, 0.224, 0.225]).view(1,3,1,1))
        self.resize = resize


    def forward(self, x):
        if x.shape[1] != 3:
            x = x.repeat(1, 3, 1, 1)
        x = (x - self.mean) / self.std
        if self.resize:
            x = self.transform(x, mode='bilinear', size=(224, 224), align_corners=False)
        feats = []
        for block in self.blocks:
            x = block(x)
            feats.append(x)
        return feats
