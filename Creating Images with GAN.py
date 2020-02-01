from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
from torch.autograd import Variable
#Importing the libraries needed 

batchSize = 64 
imageSize = 64 
#Setting the batch_size and image_size to 64

transform = transforms.Compose([transforms.Scale(imageSize), transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),]) # We create a list of transformations (scaling, tensor conversion, normalization) to apply to the input images.
#With this Compose function we will transform the image dataset the way we want. First we scale the images, then transform them into tensors
#and finally we normalize them

dataset = dset.CIFAR10(root = './home/h4rzor', download = True, transform = transform) 
#Setting the path to the download folder of the cifar10 dataset. We are using the dset function from torch to do that.
#Transform is the object we created earlier. This way we can resize,normalize and so on the images from the dataset.


dataloader = torch.utils.data.DataLoader(dataset, batch_size = batchSize, shuffle = True, num_workers = 2) 
#This function of torch makes the dataset iterable, we could pass the batch_size parameter, shuffle is true, because 
#the data can be biased. num_workers is just a how many subprocesses to use for data loading


def weights_init(m):
	#Initializing the weights_init class
    classname = m.__class__.__name__
    #This way the classname is equal to whatever the class name is.
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
        #If the classname finds an Conv in the multiple classes, we set the weights of that class to the normal distribution
        #with mean of 0.0 and standard deviation of 0.02 
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)
        #This is the same like the upper function except it is with BatchNorm, we set the weights of the network to be equal to 
        #mean = 1 and standard deviation equal to 0.02
        #This layer however has bias and we fill it with 0 or 1 in some cases



class G(nn.Module):
	#We inherit the nn.Module from torch and we pass it like this to the Generator class
    def __init__(self):
    	#This is the first function that has to be there, in order the class can be called correctly. Self is an instance of the object itself.
        super(G, self).__init__()
        #The super function is used to give access to methods and properties of a parent or sibling class.
        #We should type this in order our inheritence to work properly
               self.main = nn.Sequential(
               	#The model is sequential in order to list our needed layers.
            nn.ConvTranspose2d(100, 512, 4, 1, 0, bias = False),
            #The first digit is 100. It corresponds to the input tensor length.
            #The second digit is 512. it corresponds to the feature maps that the layer applies to get to a better image. As we progress further down the image size decreases.
            #The third digit is the kernel size. It is the square size of the feature maps that is applied to the input tensor.
            #Zero is the padding of the image. It adds a layer of zeros to the borders of the image, increasing chance to get the things right
            #One is the stride with which the kernel is moving through the image
            nn.BatchNorm2d(512),
            #We normalize the feature maps with this function BatchNorm.The only parameter is the layer's feature maps.
            nn.ReLU(True),
            #The ReLU function is to break the linearity.Whatever pixel is 0>pixel, it becomes 0
            nn.ConvTranspose2d(512, 256, 4, 2, 1, bias = False),
            #As you can might see the image is getting smaller and smaller, the stride became 2 and the padding is now 1
            nn.BatchNorm2d(256),
            #Again applying BatchNorm 
            nn.ReLU(True),
            nn.ConvTranspose2d(256, 128, 4, 2, 1, bias = False),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.ConvTranspose2d(128, 64, 4, 2, 1, bias = False),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.ConvTranspose2d(64, 3, 4, 2, 1, bias = False),
            #This is the final transpose layer. 64 is the final image size,3 is the image channels..red,green or blue
            nn.Tanh()
            #It is a common function to be used. However this will work better from the sigmoid because it is centered around 0
        )

    def forward(self, input): 
        output = self.main(input)
        return output
        #This function triggers the main function in the G class. Input will be later declared and it return the output of the layers ending with Tanh


netG = G()
#Making an instance of the generator
netG.apply(weights_init)
#Applying the weights function to the generator class.



class D(nn.Module):
	#Again inheriting from the nnModule,but this time Discriminator is a convolution but the generator was a Transposed convolution.
	#The reason is - generator must generate new images and the discriminator has to decides whether the image is real or not.
	#The first input of the discriminator is an image of the generator. So the dimensions must match
    def __init__(self):
        super(D, self).__init__()
        self.main = nn.Sequential(
            nn.Conv2d(3, 64, 4, 2, 1, bias = False),
            #3 stands for the channels aka red,green,blue. 64 is the number of the feature maps applied to the generated or not image.
            #4 stands for the size of the kernels.	
            #Stride is two, and the padding is 1
            nn.LeakyReLU(0.2, inplace = True),
            #This is a fancy ReLU.It goes below zero in this case 0.2
            nn.Conv2d(64, 128, 4, 2, 1, bias = False),
            nn.BatchNorm2d(128),
            #Normalization of the feature maps
            nn.LeakyReLU(0.2, inplace = True),
            nn.Conv2d(128, 256, 4, 2, 1, bias = False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace = True),
            nn.Conv2d(256, 512, 4, 2, 1, bias = False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace = True),
            nn.Conv2d(512, 1, 4, 1, 0, bias = False),
            #The only difference here is the output of the last convolution..it is one
            nn.Sigmoid()
            #Breaking the linearity
        )

    def forward(self, input):
        output = self.main(input)
        return output.view(-1)
        #This function again returns just the output of the layers


netD = D()
#Making an instance of the discriminator
netD.apply(weights_init)
#Applying the weights



criterion = nn.BCELoss()
#This criterion measures the Binary Cross Entropy netween the output and the input
optimizerD = optim.Adam(netD.parameters(), lr = 0.0002, betas = (0.5, 0.999))
optimizerG = optim.Adam(netG.parameters(), lr = 0.0002, betas = (0.5, 0.999))
#Setting the optimizers to Adam,and to what to apply it. Learning rate is equal to 0.0002
#betas are coefficients used for computing running averages of gradient and its square 

for epoch in range(25):

    for i, data in enumerate(dataloader, 0):
        

        netD.zero_grad()
        #We should initialize the gradients to zero. It is made with this function. We have to do this because after gradients are computed optimizer has to do a step down the imaginary hill

        
        
        real, _ = data
        #Real data from the dataset
        input = Variable(real)
        #We have to make a variable which is accepted by torch.
        target = Variable(torch.ones(input.size()[0]))
        #We initialize the target variables with the ones function. this is the desired output of the discriminator when the data is real.
        output = netD(input)
        #Output of the discriminator with the real input
        errD_real = criterion(output, target)
        #Computing the BCELoss between output and target 
        
        
        noise = Variable(torch.randn(input.size()[0], 100, 1, 1))
        #These are the random pixels the generator network accepts. It is 100 long which is exactly the input of the generator
        fake = netG(noise)
        #The fake data
        target = Variable(torch.zeros(input.size()[0]))
        #This is the target variables ..zero is this is the fake data, and 1 is the real data.
        output = netD(fake.detach())
        #output of the discriminator when the data given to it is fake.
        errD_fake = criterion(output, target)
        #BCELoss calculate
        
        
        errD = errD_real + errD_fake
        #Calculating the real and the fake loss
        errD.backward()
        #This is the backpropagation of the neural network.
        optimizerD.step()
        #Optimize the gradient descent with adam 

        

        netG.zero_grad()
        #Making the zero gradient for the generator 
        target = Variable(torch.ones(input.size()[0]))
        #The target is one, because if the output is one, the discriminator can not distinguish from real and fake images, and it will guess 50:50 ,
        #thus we are making the target variables to one
        output = netD(fake)
        #Getting the output of the discriminator when the data is fake.If the image is fake, we should get a value close to zero and the opposite for the ground true data
        #The generator is trying to get the discriminator to confuse the real images and the fake ones. 
        errG = criterion(output, target)
        #Calculating the loss of the output of the discriminator and the target
        errG.backward()
        #Backpropagating the error through the network
        optimizerG.step()
        #Making sure the adam is working
        
        

        print('[%d/%d][%d/%d]' % (epoch, 25, i, len(dataloader)))
        if i % 100 == 0:
            vutils.save_image(real, '%s/real_samples.png' % "./some_folder", normalize = True)
            fake = netG(noise)
            vutils.save_image(fake.data, '%s/fake_samples_epoch_%03d.png' % ("./some_folder", epoch), normalize = True)
            #If one epoch has passed we save the images in some folder