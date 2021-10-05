import argparse

import torch
from torch import nn, optim
#from torch.utils.data import DataLoader

from torchvision import datasets, transforms, utils

from tqdm import tqdm

from scheduler_vae import CycleScheduler

# TO-DO: join this within train_ae func - no problem with tqdm and loader? 

def train_vae(train_loader, net=None, args=None, logger=None):
    # NOTE: removed for now bc part of vq-vae image dataloader
    #transform = transforms.Compose(
    #    [
    #        transforms.Resize(args.size),
    #        transforms.CenterCrop(args.size),
    #        transforms.ToTensor(),
    #        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
    #    ]
    #)
    
    device = 'cuda'
    #dataset = datasets.ImageFolder(args.path, transform=transform)
    #loader = DataLoader(dataset, batch_size=128, shuffle=True, num_workers=4)
    #loader = tqdm(loader)
    #model = VQVAE().to(device)
    #train_loader = tqdm(train_loader)

    optimizer = optim.Adam(net.parameters(), lr=args.lr)
    scheduler = None
    if args.sched == 'cycle':
        scheduler = CycleScheduler(
            optimizer, args.lr, n_iter=len(train_loader) * args.epochs, momentum=None
        )
    criterion = nn.MSELoss()
    # TO-DO: understand these variables and modify accordingly - see line 54-57 (..sample = img[:sample_size])
    latent_loss_weight = 0.25
    #sample_size = 25
    net.train()
    for epoch in range(args.epochs):
        mse_sum = 0
        mse_n = 0
        # TO-DO: see if tuple exists in audio data - remove if not
        #for i, (img, label) in enumerate(loader):
        #for i, (img, label) in enumerate(loader):
        with tqdm(train_loader, unit="batch") as tepoch:
            for img in tepoch:
                optimizer.zero_grad()
                img = img.type(torch.cuda.FloatTensor)
                out, latent_loss = net(img)
                recon_loss = criterion(out, img)
                loss = recon_loss + latent_loss_weight * latent_loss.mean()
                loss.backward()
                mse_sum += recon_loss.item() * img.shape[0]
                mse_n += img.shape[0]

                lr = optimizer.param_groups[0]['lr']

                tepoch.set_description(
                    (
                        f'epoch: {epoch + 1}; mse: {recon_loss.item():.5f}; '
                        f'latent: {latent_loss.item():.3f}; avg mse: {mse_sum / mse_n:.5f}; '
                        f'lr: {lr:.5f}'
                    )
                )
                if scheduler is not None:
                    scheduler.step()
                optimizer.step()
                # TO-DO: may need to change below entirely for spec - see AE and Interactive Spectrogram code (Jukebox helpers)
                #if i % 100 == 0:
                #    model.eval()

                #    sample = img[:sample_size]

                #    with torch.no_grad():
                #        out, _ = model(sample)

                    #utils.save_image(
                    #    torch.cat([sample, out], 0),
                    #    f'sample/{str(epoch + 1).zfill(5)}_{str(i).zfill(5)}.png',
                    #    nrow=sample_size,
                    #    normalize=True,
                    #    range=(-1, 1),
                    #)

                    #model.train()
        torch.save(net.state_dict(), f'{str(args.logdir)}.pth')
    return net
