import torch.optim as optim
import json
import torch
import torch.nn as nn
from torchsummary import summary
from dataloader.dataloader import MnistData
from models.cnnM3 import CnnM3
from configs.config import Configuration
from utils.ema import Ema

conf = Configuration()

class Trainer:
    '''This is a training class'''
    def __init__(self, conf, device):
        self.device = device
        
        self.name_model = conf.name_model
        self.batch_size = conf.batch_size
        self.num_ephocs = conf.num_ephocs
        self.learning_rate = conf.learning_rate
        self.gamma = conf.gamma
        self.path_data = conf.path_data
        self.output_path = conf.output_path

        self.mnistdata = MnistData(self.path_data, self.batch_size)

        if(self.name_model == 'M3'):
            self.model = CnnM3()
        elif(self.name_model == 'M5'):
            self.model = CnnM5()
        elif(self.name_model == 'M7'):
            self.model = CnnM7()
        else:
            self.model = CnnM3()

        self.model = self.model.to(self.device)

    def run_train(self):
        loss_fn = nn.CrossEntropyLoss()
        #ema = Ema(self.model, decay=0.99)
        optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=self.gamma)
        
        trainloader = self.mnistdata.train_loader()
        testloader = self.mnistdata.test_loader()
        print('Starting Training')
        for epoch in range(self.num_ephocs):  # loop over the dataset multiple times
            print('> epoch number : ',epoch)
            running_loss = 0.0
            for i, (image, label) in enumerate(trainloader, 0):
                
                # get the inputs; data is a list of [inputs, label]
                image, label = image.to(device), label.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward + backward + optimize
                # predict classes using images from the training set
                outputs = self.model(image)
                # compute the loss based on model output and real label
                loss = loss_fn(outputs, label)
                # backpropagate the loss
                loss.backward()
                # adjust parameters based on the calculated gradients
                optimizer.step()
                #ema.update()

                # statistics for every 1,000 images
                running_loss += loss.item()  # extract the loss value
                # print every 1000 
                if i % 1000 == 999:       
                    print('[%d, %5d] loss: %.3f' % (epoch + 1, i + 1, running_loss / 1000))
                    # zero the loss
                    running_loss = 0.0

            lr_scheduler.step()  
            acc = self.run_test(self.model, testloader) 

        print('Finished Training')

        print("> The accuracy of the model is: ", acc)

        # Save Model
        torch.save(self.model.state_dict(), self.output_path)

    @staticmethod
    def run_test(model, testloader):
        correct = 0 
        with torch.no_grad():
            for i, (image, label) in enumerate(testloader):
                image, label = image.to(device), label.to(device)
                output = model(image)
                pred = torch.argmax(output,dim=1)
                correct += pred.eq(label.view_as(pred)).sum().item()
        
        accuracy = 100. * correct / len(testloader.dataset)
        print(' > accuracy: ', accuracy)
        return(accuracy)
                
if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(' > device: ', device)
    trainer = Trainer(conf, device)
    trainer.run_train()
