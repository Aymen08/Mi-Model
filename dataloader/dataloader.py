import torch
import torchvision
import torchvision.transforms as transforms

class MnistData:
    '''This is the data loader class'''

    def __init__(self, path, batch_size=100):
        self.path = path
        self.batch_size = batch_size

    def train_loader(self):
        transform = transforms.Compose([
        	transforms.ToTensor(), 
        	transforms.Normalize((0.1307,), (0.3081,)),
        	transforms.RandomAffine(degrees = (-20, 20), translate = (0.2, 0.2))
        	])

        trainset = torchvision.datasets.MNIST(self.path, train=True, download=True, transform=transform)
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=self.batch_size, shuffle=True)
        return(trainloader)

    def test_loader(self):
        transform = transforms.ToTensor()
        testset = torchvision.datasets.MNIST(self.path, train=False, download=True, transform=transform)
        testloader = torch.utils.data.DataLoader(testset, batch_size=self.batch_size, shuffle=False)
        return(testloader)

def main():
    path = 'data/'
    mnistdata = MnistData(path)
    trainloader = mnistdata.train_loader()
    testloader = mnistdata.test_loader()

if __name__ == "__main__":
    main()
