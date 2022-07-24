import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision import models

nclasses = 20 

pretrained_alex = models.alexnet(pretrained=True)
pretrained_vgg = models.vgg19(pretrained=True)
pretrained_wideresnet = models.wide_resnet101_2(pretrained=True)
pretrained_efficient_b7 = models.efficientnet_b7(pretrained=True)
pretrained_resnet = models.resnet152(pretrained=True)

class ResNet(nn.Module):
    def __init__(self):
        super(ResNet, self).__init__()
        self.pretrained_resnet = pretrained_resnet
        self.Dense1 = nn.Sequential(nn.Linear(1000, 512), nn.ReLU(), torch.nn.Dropout(p=0.5))
        self.Dense2 = nn.Sequential(nn.Linear(512, 64), nn.ReLU(), torch.nn.Dropout(p=0.5))
        self.Classifier = nn.Linear(64, nclasses)

    def forward(self, x):
        # print('x_shape 0:', x.shape)
        output = self.pretrained_resnet(x)
        # print('x_shape 1:', output.shape)
        output = self.Dense1(output)
        # print('x_shape 2:', output.shape)
        output = self.Dense2(output)
        # print('x_shape 3:', output.shape)
        return self.Classifier(output)



class EfficientNetB7(nn.Module):
    def __init__(self):
        super(EfficientNetB7, self).__init__()
        self.pretrained_efficient_b7 = pretrained_efficient_b7
        self.efficientnet_b7 = nn.Sequential(*list(self.pretrained_efficient_b7.features.children()))
        self.conv1 = nn.Conv2d(32, 16, kernel_size=5)
        self.conv2 = nn.Conv2d(16, 8, kernel_size=5)
        self.Dense1 = nn.Sequential(nn.Linear(384, 128), nn.ReLU(), torch.nn.Dropout(p=0.5))
        self.Classifier = nn.Linear(128, nclasses)

    def forward(self, x):
        # print('x_shape 0:',x.shape)
        output = self.efficientnet_b7(x)
        output = output.view(-1, 32, 32, 10)
        # print('x_shape 1:',output.shape)
        output = self.conv1(output)
        # print('x_shape 2:',output.shape)
        output = self.conv2(output)
        # print('x_shape 3:',output.shape)
        output = output.view(-1, 384)
        output = self.Dense1(output)
        return self.Classifier(output)

        

class WideResNet(nn.Module):
    def __init__(self):
        super(WideResNet, self).__init__()

        self.pretrained_wideresnet = pretrained_wideresnet
        

        self.Dense1 = nn.Sequential(nn.Linear(12288, 1024), nn.ReLU(), torch.nn.Dropout(p=0.5))
        self.Dense2 = nn.Sequential(nn.Linear(1024, 128), nn.ReLU(), torch.nn.Dropout(p=0.5))
        self.Classifier = nn.Linear(128, nclasses)

    def forward(self, x):
        # print('x_shape 0:',x.shape)
        output = self.pretrained_wideresnet(x)
        output = output.view(-1, 12288)
        # print('x_shape 1:',output.shape)
        output = self.Dense1(output)
        # print('x_shape 2:',output.shape)
        output = self.Dense2(output)
        # print('x_shape 3:',output.shape)
        return self.Classifier(output)

class VggNet(nn.Module):
    def __init__(self, output_dim=nclasses):
        super(VggNet, self).__init__()

        vgg_model = torchvision.models.vgg19(pretrained=True)

        self.Conv1 = nn.Sequential(*list(vgg_model.features.children())[0:25])

        for param in self.Conv1.parameters():
            param.requires_grad = False

        self.Conv2 = nn.Sequential(*list(vgg_model.features.children())[25:37])

        self.Dense1 = nn.Sequential(nn.Linear(25088, 1024), nn.ReLU(), torch.nn.Dropout(p=0.5))
        self.Dense2 = nn.Sequential(nn.Linear(1024, 64), nn.ReLU(), torch.nn.Dropout(p=0.5))
        self.Classifier = nn.Linear(64, nclasses)

    def forward(self, x):
        output = self.Conv1(x)
        output = self.Conv2(output)
        # print('x_shape 1:',output.shape)

        output = output.view(-1, 25088)

        output = self.Dense1(output)
        output = self.Dense2(output)
        # output = self.Dense3(output)

        output = self.Classifier(output)

        return output


class AlexNet(models.AlexNet):
    def __init__(self):
        super(AlexNet, self).__init__(num_classes=nclasses)
        self.pretrained_alex = pretrained_alex
        self.classifier = nn.Sequential(nn.Linear(1000, 64),
                                        nn.ReLU(),
                                        # nn.Linear(512, 64),
                                        # nn.ReLU(),
                                        nn.Linear(64, nclasses),
                                        )

    def forward(self, x):
        print('x_shape 0:',x.shape)
        x = self.pretrained_alex(x)
        print('x_shape 1:',x.shape)
        return self.classifier(x)


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 5, kernel_size=3)
        self.conv2 = nn.Conv2d(5, 5, kernel_size=3)
        self.conv3 = nn.Conv2d(5, 5, kernel_size=3)
        self.conv4 = nn.Conv2d(5, 5, kernel_size=3)
        self.classifier = nn.Sequential(nn.Linear(180, 64), 
                                        nn.ReLU(),
                                        nn.Linear(64, nclasses),
                                        )

    def forward(self, x):
        # print('x_shape 0:',x.shape)
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        # print('x_shape 1:',x.shape)
        x = F.relu(F.max_pool2d(self.conv2(x), 2))
        # print('x_shape 2:',x.shape)
        x = F.relu(F.max_pool2d(self.conv3(x), 2))
        # print('x_shape 3:',x.shape)
        # x = F.relu(F.max_pool2d(self.conv4(x), 2))
        # print('x_shape 4:',x.shape)
        # x = F.relu(F.max_pool2d(self.conv5(x), 2))
        x = x.view(-1, 180)
        return self.classifier(x)
