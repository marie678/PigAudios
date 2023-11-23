from torch import nn

# Build the model

class CNN_spectro(nn.Module):

    def __init__(self):
        super().__init__()
        # 4 conv blocks / flatten / linear / softmax
        self.conv1 = nn.Sequential(
            nn.Conv2d(1,
                out_channels=16,
                kernel_size=3,
                stride=1,
                padding=1
            ),
            nn.BatchNorm2d(num_features=16),
            nn.ReLU(),
            nn.MaxPool2d(
                kernel_size=2,
                stride=2
            ),
            nn.Dropout(p=0.5)
        )
        
        self.conv2 = nn.Sequential(
            nn.Conv2d(
                in_channels=16,
                out_channels=32,
                kernel_size=3,
                stride=2,
                padding=1
            ),
            nn.BatchNorm2d(num_features=32),
            nn.ReLU(),
            nn.MaxPool2d(
                kernel_size=2,
                stride=2
            ),
            nn.Dropout(p=0.5)
        )
        
        self.conv3 = nn.Sequential(
            nn.Conv2d(
                in_channels=32,
                out_channels=64,
                kernel_size=3,
                stride=2,
                padding=1
            ),
            nn.BatchNorm2d(num_features=64),
            nn.ReLU(),
            nn.MaxPool2d(
                kernel_size=2,
                stride=2
            ),
            nn.Dropout(p=0.5)
        )
        
        self.conv4 = nn.Sequential(
            nn.Conv2d(
                in_channels=64,
                out_channels=128,
                kernel_size=3,
                stride=2,
                padding=1
            ),
            nn.BatchNorm2d(num_features=128),
            nn.ReLU(),
            nn.MaxPool2d(
                kernel_size=2,
                stride=2
            ),
            nn.Dropout(p=0.5)
        )
        self.flatten = nn.Flatten()
        
        self.fc1 = nn.Linear(512, 128)
        self.fc2 = nn.Linear(128,64)
        self.fc3 = nn.Linear(64,4)
        self.softmax = nn.Softmax(dim=1)


    def forward(self, x):
#         print(x.shape)
        x = self.conv1(x)
#         print(x.shape)
        x = self.conv2(x)
#         print(x.shape)
        x = self.conv3(x)
#         print(x.shape)
        x = self.conv4(x)
#         print(x.shape)
        x = self.flatten(x)
#         print(x.shape)
        x = nn.ReLU()(self.fc1(x))
        x = nn.ReLU()(self.fc2(x))
        x = self.fc3(x)
        
#         predictions = self.softmax(x)

        return x



class CNN_mel(nn.Module):

    def __init__(self):
        super().__init__()
        # 4 conv blocks / flatten / linear / softmax
        self.conv1 = nn.Sequential(
            nn.Conv2d(1,
                out_channels=16,
                kernel_size=3,
                stride=1,
                padding=1
            ),
            nn.BatchNorm2d(num_features=16),
            nn.ReLU(),
            nn.MaxPool2d(
                kernel_size=2,
                stride=2
            ),
            nn.Dropout(p=0.5)
        )
        
        self.conv2 = nn.Sequential(
            nn.Conv2d(
                in_channels=16,
                out_channels=32,
                kernel_size=3,
                stride=2,
                padding=1
            ),
            nn.BatchNorm2d(num_features=32),
            nn.ReLU(),
            nn.MaxPool2d(
                kernel_size=2,
                stride=2
            ),
            nn.Dropout(p=0.5)
        )
        
        self.conv3 = nn.Sequential(
            nn.Conv2d(
                in_channels=32,
                out_channels=64,
                kernel_size=3,
                stride=1,
                padding=1
            ),
            nn.BatchNorm2d(num_features=64),
            nn.ReLU(),
            nn.MaxPool2d(
                kernel_size=2,
                stride=2
            ),
            nn.Dropout(p=0.5)
        )
        
        self.conv4 = nn.Sequential(
            nn.Conv2d(
                in_channels=64,
                out_channels=128,
                kernel_size=2,
                stride=1,
                padding=1
            ),
            nn.BatchNorm2d(num_features=128),
            nn.ReLU(),
            nn.MaxPool2d(
                kernel_size=4,
                stride=2
            ),
            nn.Dropout(p=0.5)
        )
        self.flatten = nn.Flatten()
        
        self.fc1 = nn.Linear(1920, 128)
        self.fc2 = nn.Linear(128,64)
        self.fc3 = nn.Linear(64,4)
        self.softmax = nn.Softmax(dim=1)


    def forward(self, x):
#         print(x.shape)
        x = self.conv1(x)
#         print(x.shape)
        x = self.conv2(x)
#         print(x.shape)
        x = self.conv3(x)
#         print(x.shape)
        x = self.conv4(x)
#         print(x.shape)
        x = self.flatten(x)
#         print(x.shape)
        x = nn.ReLU()(self.fc1(x))
#         x = self.fc3(x)
        x = nn.ReLU()(self.fc2(x))
        x = self.fc3(x)
        
#         predictions = self.softmax(x)

        return x



class CNN_mfcc(nn.Module):

    def __init__(self):
        super().__init__()
        # 4 conv blocks / flatten / linear / softmax
        self.conv1 = nn.Sequential(
            nn.Conv2d(1,
                out_channels=16,
                kernel_size=3,
                stride=1,
                padding=1
            ),
            nn.BatchNorm2d(num_features=16),
            nn.ReLU(),
            nn.MaxPool2d(
                kernel_size=2,
                stride=2
            ),
            nn.Dropout(p=0.5)
        )
        
        self.conv2 = nn.Sequential(
            nn.Conv2d(
                in_channels=16,
                out_channels=32,
                kernel_size=3,
                stride=1,
                padding=1
            ),
            nn.BatchNorm2d(num_features=32),
            nn.ReLU(),
            nn.MaxPool2d(
                kernel_size=2,
                stride=2
            ),
            nn.Dropout(p=0.5)
        )
        
        self.conv3 = nn.Sequential(
            nn.Conv2d(
                in_channels=32,
                out_channels=64,
                kernel_size=3,
                stride=1,
                padding=1
            ),
            nn.BatchNorm2d(num_features=64),
            nn.ReLU(),
            nn.MaxPool2d(
                kernel_size=2,
                stride=2
            ),
            nn.Dropout(p=0.5)
        )
        
        self.conv4 = nn.Sequential(
            nn.Conv2d(
                in_channels=64,
                out_channels=128,
                kernel_size=3,
                stride=2,
                padding=1
            ),
            nn.BatchNorm2d(num_features=128),
            nn.ReLU(),
            nn.MaxPool2d(
                kernel_size=4,
                stride=2
            ),
            nn.Dropout(p=0.5)
        )
        self.flatten = nn.Flatten()
        
        self.fc1 = nn.Linear(1792, 64)
#         self.fc2 = nn.Linear(128,64)
        self.fc3 = nn.Linear(64,4)
        self.softmax = nn.Softmax(dim=1)


    def forward(self, x):
#         print(x.shape)
        x = self.conv1(x)
#         print(x.shape)
        x = self.conv2(x)
#         print(x.shape)
        x = self.conv3(x)
#         print(x.shape)
        x = self.conv4(x)
#         print(x.shape)
        x = self.flatten(x)
#         print(x.shape)
        x = nn.ReLU()(self.fc1(x))
#         x = nn.ReLU()(self.fc2(x))
        x = self.fc3(x)
        
#         predictions = self.softmax(x)

        return x
