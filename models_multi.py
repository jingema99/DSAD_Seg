import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
import torchvision
from transformers import SegformerForSemanticSegmentation
import copy

class MultiHeadSegFormer(nn.Module):
    def __init__(self, num_classes, use_softmax=False):
        super(MultiHeadSegFormer, self).__init__()

        model = SegformerForSemanticSegmentation.from_pretrained("nvidia/segformer-b3-finetuned-cityscapes-1024-1024",num_labels=2,ignore_mismatched_sizes=True)

        self.segformer = model.segformer
        self.decode_heads = nn.ModuleList()
        
        for i in range(num_classes):
            self.decode_heads.append(copy.deepcopy(model.decode_head))

        self.use_softmax = use_softmax
        if use_softmax:
            self.softmax = nn.Softmax(dim=1)

    
    def forward(self, x):
        outputs = self.segformer(
            x,
            output_attentions=False,
            output_hidden_states=True,
            return_dict=False,
        )

        encoder_hidden_states = outputs[1]

        result = []
        for i in range(len(self.decode_heads)):
            logits = self.decode_heads[i](encoder_hidden_states)

            x = F.interpolate(logits, size=x.shape[-2:], mode="bilinear", align_corners=False)

            if self.use_softmax:
                x = self.softmax(x)

            result.append(x)

        return result

class MultiHeadDeepLab(nn.Module):
    def __init__(self, num_classes, use_softmax=False):
        super(MultiHeadDeepLab, self).__init__()

        self.model = models.segmentation.deeplabv3_resnet50(pretrained=True, progress=True)
        channels = 2048

        self.scale_factor = 8

        self.classifiers = nn.ModuleList()
        
        for i in range(num_classes):
            classifier = torchvision.models.segmentation.deeplabv3.DeepLabHead(channels, 2)
            self.classifiers.append(classifier) 

        self.use_softmax = use_softmax
        if use_softmax:
            self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        input_shape = x.shape[-2:]

        result = []
        features = self.model.backbone(x)
        features = features["out"]
        
        for i in range(len(self.classifiers)):
            classifier = self.classifiers[i]
            x = classifier(features)
            x = F.interpolate(x, size=input_shape, mode='bilinear', align_corners=False)

            if self.use_softmax:
                x = self.softmax(x)

            result.append(x)

        return result
