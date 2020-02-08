from network2_without_trimap import Encoder, Decoder
import torch.nn.functional as F
import torch.optim as optim
from DataSet import *
import os
from torch import nn


device = 0 
ed_epoch = 100
batch_size = 8
crop_size = (512, 512) 
lrD = 1e-2
lrE = 1e-2
load_from_pretrain = True 
pretrain_decoder = False

Encoder= Encoder().double().cuda(device)
Decoder = Decoder().double().cuda(device)

optE = optim.SGD(Encoder.parameters(), lr=lrE, )
optD = optim.SGD(Decoder.parameters(), lr=lrD, )



a_path = '/home/zhuyuanjin/data/Human_Matting/alpha'
img_path = '/home/zhuyuanjin/data/Human_Matting/image_matting'

pretrainedE = "/home/zhuyuanjin/data/Human_Matting/models/paramE_seg_without_trimap"
pretrainedD = "/home/zhuyuanjin/data/Human_Matting/models/paramD_seg_without_trimap"

paramE = '/home/zhuyuanjin/data/Human_Matting/models/paramE_mat_without_trimap'
paramD = '/home/zhuyuanjin/data/Human_Matting/models/paramD_mat_without_trimap'


dataset = MattingDataSet(a_path=a_path, img_path=img_path, crop_size = crop_size)
dataloader = DataLoader(dataset, num_workers=10 , batch_size=batch_size, shuffle=True)

if __name__ == '__main__':
    print("Train model with size of dataset %d" % len(dataset))
    print("Picture size of %d with batch size %d" % (crop_size[0], batch_size))


    if os.path.exists(pretrainedE) & load_from_pretrain:
        print("loading param from %s" % pretrainedE)
        state_dict = torch.load(pretrainedE)
        Encoder.load_state_dict(state_dict['net'])
    if os.path.exists(pretrainedD) & load_from_pretrain:
        print("loading param from %s" % pretrainedD)
        state_dict = torch.load(pretrainedD)
        Decoder.load_state_dict(state_dict['net'])
       #opt_ED.load_state_dict(state_dict['optim'])



    for epoch in range(ed_epoch):
        for batch in dataloader:
            img, alpha, trimap, unknown = batch['img'].cuda(device), \
                                              batch['alpha'].cuda(device), batch['trimap'].cuda(device), \
                                              batch['unknown'].cuda(device)

            input = img
            if pretrain_decoder:
                features = Encoder(input).data
            else:
                features = Encoder(input)
            loss_bce = nn.BCELoss()
            alpha_predict = torch.sigmoid(Decoder(features))
            #img_predict = (fg * alpha_predict + bg * (1-alpha_predict)) * unknown
            #loss_comp = F.mse_loss(img_predict * unknown, img * unknown)
            loss = loss_bce(alpha_predict, alpha)
            print(loss.item(), flush=True)
            torch.save({'net':Encoder.state_dict(), 'optim':optE.state_dict()}, paramE)
            torch.save({'net':Decoder.state_dict(), 'optim':optD.state_dict()}, paramD)
            optD.zero_grad()
            optE.zero_grad()
            loss.backward()
            optD.step()
            optE.step()
