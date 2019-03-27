import ssgan_resnet as model
from myargs import args
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
from tensorboardX import SummaryWriter
import torchvision.utils as vutils


logger = SummaryWriter(args.log+'/tensorboard/')

if args.multi_gpu:
    G = nn.DataParallel(model.Generator(args.z_dim)).cuda()
    D = nn.DataParallel(model.Discriminator()).cuda()
else:
    G = model.Generator(args.z_dim).cuda()
    D = model.Discriminator().cuda()


print(G)
print(D)
print(args)

def get_dataloader(batch_size):
    transform = transforms.Compose(
                [transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                            download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                            shuffle=True, num_workers=2,
                                            drop_last=True, pin_memory=True)

    #dataiter = iter(trainloader)
    while True:
        for images, labels in trainloader:
            yield images


def resume(**args):
    if args['resume_optim']:
        st = torch.load(args.path+'/G.pth')
        args['G'].load_state_dict(st['G'])
        args['optimG'].load_state_dict(st['optimG'])
        
        st = torch.load(args.path+'/D.pth')
        args['D'].load_state_dict(st['D'])
        args['optimD'].load_state_dict(st['optimD'])
    else:
        st = torch.load(args.path+'/G.pth')
        args['G'].load_state_dict(st['G'])
        
        st = torch.load(args.path+'/D.pth')
        args['D'].load_state_dict(st['D'])

def save(**args):
    # save G
    gd = {'G': args['G'].state_dict(),
        'optimG': args.get('optimG', "")}
    torch.save(gd, args['path']+'/G_{}.pth'.format(args['step']))
    dd = {'D': args['D'].state_dict(),
        'optimD': args.get('optimD', "")}
    torch.save(gd, args['path']+'/D_{}.pth'.format(args['step']))

def rotate90(batch):
    rows = batch.shape[2]
    ret = batch[:, :, -(np.arange(rows)+1), :]
    ret = torch.transpose(ret, 2, 3)
    return ret

def rotate_batch(batch):
    ret = [batch]
    ret.append(rotate90(ret[-1]))
    ret.append(rotate90(ret[-1]))
    ret.append(rotate90(ret[-1]))
    #ret.append(TF.to_tensor(TF.rotate(pili, 90)))
    #ret.append(TF.to_tensor(TF.rotate(pili, 180)))
    #ret.append(TF.to_tensor(TF.rotate(pili, 270)))
    # rotate 180

    ret = torch.cat(ret, dim=0)
    return ret

def train():
    
    dt_train = get_dataloader(args.batch_size)
    optimD = torch.optim.Adam(filter(lambda p: p.requires_grad, D.parameters()), lr=args.lr,  betas=(0, 0.9))
    optimG = torch.optim.Adam(G.parameters(), lr=args.lr, betas=(0, 0.9))

    if args.resume:
        resume(G=G, D=D, optimG=optimG, optimD=optimD, path=args.log)
    ep = 0

    one_fourth = args.batch_size//4
    label = np.zeros(one_fourth*4)
    label[1*one_fourth:2*one_fourth] = 1
    label[2*one_fourth:3*one_fourth] = 2
    label[3*one_fourth:4*one_fourth] = 3
    label = torch.tensor(label).long().cuda()
    loss = nn.CrossEntropyLoss()
    while ep < args.epoch:
        # TRAIN DISCRIMINATOR
        for p in D.parameters():
            p.requires_grad_(True)
        for _ in range(args.disc_iter):
            optimD.zero_grad()
            z = torch.FloatTensor(args.batch_size, args.z_dim).uniform_(-1, 1).cuda()
            with torch.no_grad():
                fake = G(z).detach()
            dlfake = D(fake)[0].mean()

            real = next(dt_train).cuda()
            dlreal = D(real)[0].mean()

            # calculate rotation loss
            rot_real = rotate_batch(real[:one_fourth])
            rot_score = D(rot_real)[1]
            drot_loss = loss(rot_score, label)

            (dlfake-dlreal+args.beta*drot_loss).backward()
            optimD.step()
        # =====================
        # TRAIN GENERATOR
        for p in D.parameters():
            p.requires_grad_(False)
    
        optimG.zero_grad()
        z = torch.randn(args.batch_size, args.z_dim).cuda()
        fake = G(z)
        glfake = D(fake)[0].mean()

        rot_fake = rotate_batch(fake[:one_fourth])
        rot_score = D(rot_fake)[1]
        grot_loss = loss(rot_score, label)
        (-glfake + args.alpha*grot_loss).backward()
        optimG.step()

        ep += 1
        logger.add_scalar('D', float(dlfake-dlreal+args.beta*drot_loss), ep)
        logger.add_scalar('G', float(-glfake+args.alpha*grot_loss), ep)
        if ep%1000==0 or ep==1:
            save(G=G, D=D, optimG=optimG, optimD=optimD, step=ep, path=args.log)
        if ep%500==0 or ep==1:
            print('Ep {}\n\tDloss {}\n\tGloss {}'.format(ep,
                float(dlfake-dlreal+args.beta*drot_loss),
                float(-glfake+args.alpha*grot_loss)))
        if ep%100==0 or ep==1:
            x = vutils.make_grid(fake[:64].detach(), normalize=True, scale_each=True)
            logger.add_image('Gen', x, ep)
            



if __name__=='__main__':
    train()
