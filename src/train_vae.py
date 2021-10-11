import argparse
import torch
from torch import nn, optim
#from torch.utils.data import DataLoader
from torchvision import datasets, transforms, utils
from torch.utils.tensorboard import SummaryWriter
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
    # write hyper params to tensorboard
    comment = f' batch_size = {args.batch_size} lr = {args.lr} schedule = {args.sched} latent_loss_weight = {latent_loss_weight}'
    writer = SummaryWriter(comment=comment)
    net.train()
    for epoch in range(args.epochs):
        mse_sum = 0
        mse_n = 0
        running_loss = 0
        running_recon_loss = 0
        running_latent_loss = 0
        with tqdm(train_loader, unit="batch") as tepoch:
            for i, img in enumerate(tepoch):
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
                
                # tensorboard
                running_loss += loss
                running_recon_loss += recon_loss
                running_latent_loss += latent_loss

                if i % 10 == 9:    # every 1000 mini-batches...
                    # ...log the running loss
                    img_grid = utils.make_grid(img)
                    # write to tensorboard
                    writer.add_image('data samples', img_grid)
                    writer.add_graph(net, img)
                    writer.add_scalar('total training loss',
                                    running_loss / 10,
                                    epoch * len(tepoch) + i)
                    writer.add_scalar('total recon loss',
                                    running_recon_loss / 10,
                                    epoch * len(tepoch) + i)
                    writer.add_scalar('total latent loss',
                                    running_latent_loss / 10,
                                    epoch * len(tepoch) + i)
                    for name, weight in net.named_parameters():
                        writer.add_histogram(name,weight, epoch)
                        writer.add_histogram(f'{name}.grad',weight.grad, epoch)

                    # ...log a Matplotlib Figure showing the model's predictions on a
                    # random mini-batch
                    running_loss = 0.0
                    running_recon_loss = 0.0
                    running_latent_loss = 0.0
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
