class ConvNetwork(nn.Module):
    def __init__(self, data, numClasses):
        super(ConvNetwork, self).__init__()
        x, y = data[0]
        channels, height, width = x.size()
        output = numClasses

        self.net = nn.Sequential(
            Conv2d(channels, 10, (3,3), padding=(1,1)),
            nn.ReLU(),
            Conv2d(10, 100, (5,5), padding=(1,1)),
            nn.ReLU(),
            Conv2d(100, 300, (5,5), padding=(1,1)),
            nn.ReLU(),
            Conv2d(300, output, (24,24), padding=(0,0))
        )

    def forward(self,x):
        return self.net(x)

