from encoder_decoder import EncoderDecoder
import torch.nn.functional as F
import torch.optim as optim
from DataSet import *
import os
from torch import nn


device = 0 
ed_epoch = 100
batch_size = 8
crop_size = (512, 512) 
lrD = 1e-4
lrE = 1e-5
load_from_pretrain = True
pretrain_decoder = True

ED= EncoderDecoder().double().cuda(device)

opt = optim.SGD(ED.parameters(), lr=lrE, )



a_path = '/home/zhuyuanjin/data/Human_Matting/alpha'
img_path = '/home/zhuyuanjin/data/Human_Matting/image_matting'

param_pretrained = "/home/zhuyuanjin/data/Human_Matting/models/ed_pretrained_saved"
param = '/home/zhuyuanjin/data/Human_Matting/models/ed'


dataset = MattingDataSet(a_path=a_path, img_path=img_path, crop_size = crop_size)
dataloader = DataLoader(dataset, num_workers=10 , batch_size=batch_size, shuffle=True)

if __name__ == '__main__':
    print("Train model with size of dataset %d" % len(dataset))
    print("Picture size of %d with batch size %d" % (crop_size[0], batch_size))


    if os.path.exists(param_pretrained) & load_from_pretrain:
        print("loading param from %s" % param_pretrained)
        state_dict = torch.load(param_pretrained)
        ED.load_state_dict(state_dict['net'])
       #opt_ED.load_state_dict(state_dict['optim'])



    for epoch in range(ed_epoch):
        for batch in dataloader:
            img, alpha, trimap, unknown = batch['img'].cuda(device), \
                                              batch['alpha'].cuda(device), batch['trimap'].cuda(device), \
                                              batch['unknown'].cuda(device)

            input = torch.cat((img, trimap), 1)
            alpha_predict = ED(input)
            loss_fn = nn.SmoothL1Loss()
            #img_predict = (fg * alpha_predict + bg * (1-alpha_predict)) * unknown
            #loss_comp = F.mse_loss(img_predict * unknown, img * unknown)
            loss = loss_fn(alpha_predict * unknown, alpha * unknown)
            print(loss.item(), flush=True)
            torch.save({'net':ED.state_dict(), 'optim':opt.state_dict()}, param)
            opt.zero_grad()
            loss.backward()
            opt.step()
