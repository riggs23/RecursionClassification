import torchvision.models as models
from torch import nn

class ConvNetwork(nn.Module):
    def __init__(self, data, numClasses):
        super(ConvNetwork, self).__init__()
        x, y = data[0]
        channels, height, width = x.size()
        output = numClasses

        self.net = models.vgg19(pretrained=True)
#        self.net = nn.Sequential(
#            nn.Conv2d(channels, 10, (3,3), padding=(1,1)),
#            nn.ReLU(),
#            nn.Conv2d(10, 100, (5,5), padding=(1,1)),
#            nn.ReLU(),
#            nn.Conv2d(100, 300, (5,5), padding=(1,1)),
#            nn.ReLU(),
#            nn.Linear()
#        )

    def forward(self,x):
        return self.net(x)

