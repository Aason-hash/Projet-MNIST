import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plot

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))])

data_train = datasets.MNIST(root='./data', train = True, transform = transform, download = True)
data_test = datasets.MNIST(root = '/data', train = False, transform = transform, download = True)

chTrain = DataLoader(data_train, batch_size = 64, shuffle = True)
chTest = DataLoader(data_test, batch_size = 64, shuffle = True)
images, labels = next(iter(chTest))
plot.imshow(images[0].squeeze(), cmap = 'gray')
plot.title('Label: {}'.format(labels[0]))
plot.show()
