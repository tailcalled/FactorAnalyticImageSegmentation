import torch
import torch.fft as fft
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')

print(torch.__version__)

BATCH_SIZE = 16
SZ = 128

## transformations
transform = transforms.Compose(
    [transforms.RandomCrop((SZ, SZ)), transforms.ToTensor()])

## download and load training dataset
trainset = torchvision.datasets.CocoDetection(root='./data/train2014/', annFile="./data/annotations/instances_train2014.json", transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=BATCH_SIZE,
                                          shuffle=True, num_workers=2, collate_fn=lambda xs: torch.stack([x[0] for x in xs]))

import matplotlib.pyplot as plt
import numpy as np

## functions to show an image
def imshow(img):
    #img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()
def showable(img):
	print(img.numpy().shape)
	return np.transpose(img.numpy(), (1, 2, 0))

## get some random training images
dataiter = iter(trainloader)
image = dataiter.next()
#image, labels = dataiter.next()
print(image)
print(image.shape)
print(image[0, 0, 0, 0])

blurred_img = transforms.GaussianBlur(15, 7)(image)
res_img = image - blurred_img
print(torch.cat([res_img, image], axis=1).shape)

deblur_img = transforms.GaussianBlur(99, 10)(res_img)
deblur_img_norm = transforms.GaussianBlur(99, 10)(torch.abs(res_img))
deblur_img = deblur_img / deblur_img_norm

res_img = (res_img + 1)/2

deblur_img = deblur_img + res_img
deblur_img = (deblur_img - torch.min(deblur_img)) / (torch.max(deblur_img) - torch.min(deblur_img))

print(res_img[0].shape, blurred_img[0].shape, image[0].shape)

## show images
#imshow(torchvision.utils.make_grid(torch.cat([deblur_img, res_img, image])))
imshow(torchvision.utils.make_grid(torch.cat([torch.stack([res_img[i], blurred_img[i], image[i], image[i]]) for i in range(BATCH_SIZE)])))

class Block(nn.Module):
    def __init__(self, outer_dim, inner_dim, repr_dim=None):
        super(Block, self).__init__()

        if repr_dim is None:
            repr_dim = inner_dim
        self.conv_inner = nn.Conv2d(in_channels=outer_dim, out_channels=inner_dim, kernel_size=1)
        self.conv_loc = nn.Conv2d(in_channels=inner_dim, out_channels=inner_dim, kernel_size=3, padding='same')
        self.conv_dil = nn.Conv2d(in_channels=inner_dim, out_channels=inner_dim, kernel_size=3, dilation=3, padding='same')
        self.conv_delta = nn.Conv2d(in_channels=inner_dim*2, out_channels=outer_dim, kernel_size=1)
    
    def forward(self, x):
        v = F.relu(self.conv_inner(x))
        a = F.relu(self.conv_loc(v))
        b = F.relu(self.conv_loc(v))
        v = torch.cat([a, b], axis=1) # TODO is this the right axis to cat on?
        x = x + self.conv_delta(v)
        return x

class MeanColorModelV2(nn.Module):
    def __init__(self):
        super(MeanColorModelV2, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=7, padding='same')
        self.blocks = []
        for i in range(5):
          self.blocks.append(Block(64, 48))
        self.blocks.append(nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding='same'))
        for i in range(6):
          self.blocks.append(Block(128, 96))
        self.blocks.append(nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding='same'))
        for i in range(3):
          self.blocks.append(Block(256, 192))
        self.convcolor = nn.Conv2d(in_channels=256, out_channels=3, kernel_size=1)
        #self.convfactor = nn.Conv2D(in_channels=256, out_channels=256, kernel_size=1)
        #self.converr = nn.Conv2D(in_channels=256, out_channels=3, kernel_size=1)

    def forward(self, x):
        x = self.conv1(x)
        for block in self.blocks:
          x = block(x)
        out = self.convcolor(x)

        #factor = self.convfactor(x)
        #err = self.converr(x)
        return out#, factor, err*err + 0.01

blur = transforms.GaussianBlur(7, 7)
def whiten(image):
	return (image - blur(image) + 1)/2

learning_rate = 0.001
num_epochs = 5

if not torch.cuda.is_available():
  print('WARNING! Training on CPU')
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = MeanColorModelV2()
#model.load_state_dict(torch.load('my_model.dat'))
#model.eval()
model = model.to(device)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

SMALL_N = 10
LARGE_N = 100

train_loss_history = []
for epoch in range(num_epochs):
    train_running_loss = 0.0
    train_acc = 0.0

    model = model.train()

    ## training step
    for i, image in enumerate(trainloader):
        if i % SMALL_N == 0:
          print(i, train_running_loss / (i+1))
        if i % LARGE_N == 0:
          dataiter = iter(trainloader)
          image = dataiter.next()
          image = image.to(device)
          if image.shape[1] == 1:
            image = torch.cat([image, image, image], axis=1)
          res_img = whiten(image)
          #out, factor, err = model(res_img)
          out = model(res_img)
          diff = image - out
          #out_norm = (out - torch.min(out)) / (torch.max(out) - torch.min(out))
          #diff = (diff - torch.min(diff)) / (torch.max(diff) - torch.min(diff))
          #factor = torch.reshape(factor, (image.shape[0], 3, model.VEC, image.shape[2], image.shape[3]))
          #loadings = torch.permute(factor, [0, 3, 4, 1, 2])
          #loadings = torch.reshape(loadings, (image.shape[0], image.shape[2] * image.shape[3], 3 * model.VEC))
          #U, S, V = torch.pca_lowrank(loadings, 9)
          #interp = torch.reshape(U, (image.shape[0], 9, image.shape[2], image.shape[3]))
          #interp = (interp - torch.min(interp)) / (torch.max(interp) - torch.min(interp))
          #print('012, 345, 678')
          print((diff + 0.5).shape, image.shape, res_img.shape, out.shape)
          plt.imshow(showable(torchvision.utils.make_grid(torch.cat([torch.stack([(diff + 0.5)[i], image[i], res_img[i], out[i]]).detach() for i in range(BATCH_SIZE)]))))
          #plt.imshow(torchvision.utils.make_grid(showable(torch.cat([diff + 0.5, image, res_img, out]).detach()).cpu()))
          plt.savefig(f"output/imgbatches/batch{i}.png")
          plt.close()
          #imshow(torchvision.utils.make_grid(torch.cat([interp[:, 0:3, :, :], interp[:, 3:6, :, :], interp[:, 6:9, :, :]]).cpu()))
          #print('out\', res, img')
          #imshow(torchvision.utils.make_grid(torch.cat([out_norm, res_img, image]).cpu()))
          #print('stder, delta, out')
          #imshow(torchvision.utils.make_grid(torch.cat([err, diff, out]).cpu()))
        
        image = image.to(device)

        if image.shape[1] == 1:
          image = torch.cat([image, image, image], axis=1)

        res_img = (image - transforms.GaussianBlur(7, 7)(image) + 1)/2

        ## forward + backprop + loss
        #preds, factor, err = model(res_img)
        preds = model(res_img)
        loss = criterion(preds, image)
        optimizer.zero_grad()
        loss.backward()
        if False:
	        delta = image - preds
	        loadings = torch.reshape(factor, (image.shape[0], 3, model.VEC, image.shape[2], image.shape[3]))
	        loadings = torch.permute(loadings, [0, 2, 1, 3, 4])
	        loadings = torch.reshape(loadings, (image.shape[0], model.VEC, 3 * image.shape[2] * image.shape[3]))
	        err = torch.reshape(err, (image.shape[0], 3 * image.shape[2] * image.shape[3]))
	        delta = torch.reshape(delta, (image.shape[0], 3 * image.shape[2] * image.shape[3]))
	        K = 20
	        poss = torch.randint(0, 3 * image.shape[2] * image.shape[3], (image.shape[0], K)).to(device)
	        sub_loadings = torch.gather(loadings, 2, poss.reshape(image.shape[0], 1, K).expand(image.shape[0], model.VEC, K))
	        sub_err = torch.gather(err, 1, poss)
	        sub_delta = torch.gather(delta, 1, poss)
	        covs = sub_loadings.transpose(1, 2) @ sub_loadings + torch.diag_embed(sub_err)
	        covs_inv = torch.inverse(covs)
	        logprob = -0.5 * torch.logdet(2 * 3.1415 * covs_inv) - 0.5 * sub_delta @ covs_inv @ sub_delta.T
	        loss -= torch.sum(logprob/K)
	        optimizer.zero_grad()
        	loss.backward()

        ## update model params
        optimizer.step()

        train_running_loss += loss.detach().item()
        train_loss_history.append(loss.detach().item())

        if i % LARGE_N == 0:
        	plt.plot(np.arange(len(train_loss_history)) * LARGE_N, train_loss_history)
        	plt.xlabel("# of batches")
        	plt.ylabel("loss")
        	plt.savefig(f"output/progress.png")

    print(i)
    
    print('Epoch: %d | Loss: %.4f | Train Accuracy: %.2f' \
          %(epoch, train_running_loss / i, train_acc/i))        