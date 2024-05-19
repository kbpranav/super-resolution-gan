import torch.nn as nn
import torch.nn.functional as F
import torch

import torchvision.transforms as transforms
from torchvision.utils import save_image, make_grid

from torch.utils.data import DataLoader
from torch.autograd import Variable

from models import Generator, ExtractFeatures, Discriminator
from dataset import SRDataset

import os
import numpy as np
import os
import itertools

print("Starting....")

cuda = torch.cuda.is_available()
hr_shape = (256, 256)

# Initialize generator and discriminator
generator = Generator()
feature_extractor = ExtractFeatures()
discriminator = Discriminator(input_shape=(3, *hr_shape))

# Set feature extractor to inference mode
feature_extractor.eval()

# Losses
criterion_GAN = nn.MSELoss()
criterion_content = nn.L1Loss()

if cuda:
    generator = generator.cuda()
    feature_extractor = feature_extractor.cuda()
    discriminator = discriminator.cuda()
    criterion_GAN = criterion_GAN.cuda()
    criterion_content = criterion_content.cuda()

# Optimizers

optimizer_G = torch.optim.Adam(generator.parameters(), lr=0.0002, betas=(0.9, 0.999))
optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=0.0002, betas=(0.9, 0.999))

Tensor = torch.cuda.FloatTensor if cuda else torch.Tensor

print("Loading data....")

dataloader = DataLoader(
    SRDataset("../data/DIV2K_train_HR/", hr_shape=hr_shape),
    batch_size=4,
    shuffle=True,
)

# Training the model
n_epochs = 3

if not os.path.exists("saved_models"):
    os.makedirs("saved_models")

if not os.path.exists("images"):
    os.makedirs("images")        

print("Training....")

for epoch in range(n_epochs):
    for i, imgs in enumerate(dataloader):

        # Configure model input
        imgs_lr = Variable(imgs["lr"].type(Tensor)).cuda()
        imgs_hr = Variable(imgs["hr"].type(Tensor)).cuda()

        # Adversarial ground truths
        valid = Variable(Tensor(np.ones((imgs_lr.size(0), *discriminator.output_shape))), requires_grad=False)
        fake = Variable(Tensor(np.zeros((imgs_lr.size(0), *discriminator.output_shape))), requires_grad=False)

        # ------------------
        #  Train Generators
        # ------------------

        optimizer_G.zero_grad()

        # Generate a high resolution image from low resolution input
        gen_hr = generator(imgs_lr)

        # Adversarial loss
        loss_GAN = criterion_GAN(discriminator(gen_hr), valid)

        # Content loss
        gen_features = feature_extractor(gen_hr)
        real_features = feature_extractor(imgs_hr)
        loss_content = criterion_content(gen_features, real_features.detach())

        # Total loss
        loss_G = loss_content + 1e-3 * loss_GAN

        loss_G.backward()
        optimizer_G.step()

        # ---------------------
        #  Train Discriminator
        # ---------------------

        optimizer_D.zero_grad()

        # Loss of real and fake images
        loss_real = criterion_GAN(discriminator(imgs_hr), valid)
        loss_fake = criterion_GAN(discriminator(gen_hr.detach()), fake)

        # Total loss
        loss_D = (loss_real + loss_fake) / 2

        loss_D.backward()
        optimizer_D.step()

        # --------------
        #  Log Progress
        # --------------

        print(
            "[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f]"
            % (epoch, n_epochs, i, len(dataloader), loss_D.item(), loss_G.item())
        )

        batches_done = epoch * len(dataloader) + i
        if batches_done % 50 == 0:
            # Save image grid with upsampled inputs and outputs
            imgs_lr = nn.functional.interpolate(imgs_lr, scale_factor=4)
            gen_hr = make_grid(gen_hr, nrow=1, normalize=True)
            imgs_lr = make_grid(imgs_lr, nrow=1, normalize=True)
            img_grid = torch.cat((imgs_lr, gen_hr), -1)
            save_image(img_grid, "images/%d.png" % batches_done, normalize=False)
        
    if epoch % 2 == 0:
        torch.save(generator.state_dict(), "saved_models/generator_%d.pth" % epoch)
        torch.save(discriminator.state_dict(), "saved_models/discriminator_%d.pth" % epoch)
    
