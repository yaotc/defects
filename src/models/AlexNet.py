
from torch import nn
from .BasicModule import BasicModule


class AlexNet(BasicModule):

    def __init__(self , num_classes = 12):

        super(AlexNet , self).__init__()

        self.model_name = 'alexnet'

        self.features = nn.Sequential(
            nn.Conv2d(3 , 64 , kernel_size=3 , stride=4 , padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3 , stride=1),

            nn.Conv2d(64 , 192 , kernel_size=3 , padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3 , padding=1),

            nn.Conv2d(192 , 384 , kernel_size=3 , padding=1),
            nn.ReLU(inplace=True),

            nn.Conv2d(384 , 256 , kernel_size=3 , padding=1),
            nn.ReLU(inplace=True),

            nn.Conv2d(256 , 256 , kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3 , stride=1),
        )

        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256*17*17, 2048),
            nn.ReLU(inplace=True),

            nn.Dropout(),
            nn.Linear(2048 , 1024),
            nn.ReLU(inplace=True),

            nn.Linear(1024 , num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0) , 256*17*17)
        x = self.classifier(x)

        return x
