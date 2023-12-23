#import dependencies
import torch 
from PIL import Image
from torch import nn, save, load
from torch.optim import Adam
from torch.utils.data import Dataloader
from torchvision import datasets
from torchvision.transformers import ToTensor


# Get Data
train= datasets.Mnist(root='data', train=True, transform=ToTensor(), download=True)
dataset = Dataloader(train,32)
#1,28,28 - classes 0-9


#Image Classifier Neural Network
class ImageClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Convo2D(1,32,(3,3)),
            nn.ReLU(),
            nn.Convo2D(32,64,(3,3)),
            nn.ReLU(),
            nn.Convo2D(64,64,(3,3)),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(64*(28-6)*(28-6),10)
                                   )
        
        def forward(self,x):
            return self.model(x)
        
        #Instance of the neural network, loss, optimizer
clf = ImageClassifier().to('cpu') # If you have cpu
# clf = ImageClassifier().to('cuda') # if you have gpu
opt = Adam(clf.parameters(),lr=1e-3)
loss_fn = nn.CrossEntropyLoss()


# Training flow
if __name__ == "__main__":
    for epoch in range(10): #train for 10 epochs
        for batch in dataset:
            X,y = batch
            X,y = X.to('cpu'), y.to('cpu') # if cpu is available
            # X,y = X.to('cuda'), y.to('cuda') # if gpu is available
            yhat = clf(X)
            loss = loss_fn(yhat,y)

            # Apply backprop
            opt.zero_grad()
            loss.backward()
            opt.step()

            print(f'Epoch:{epoch} loss is {loss.item()}')

        with open('model_state.pt','wb') as f:
            save(clf.state_dict(),f)

        with open('model_state.pt','rb') as f:
            clf.load_state_dict(load(f))

        img = Image.open('img_3.jpg')
        img_tensor= ToTensor()(img).unsqueeze(0).to('cuda')

        print(torch.argmax(clf(img_tensor)))

test = datasets.Mnist(root='data', train=False, transform=ToTensor(), download=True)