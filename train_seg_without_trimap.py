from network2_without_trimap import Encoder, Decoder
import torch.nn.functional as F
import torch.optim as optim
from DataSet import *
import os
from torch import nn

device = 1 
ed_epoch = 100
refine_epoch = 100
final_epoch = 100
batch_size = 8 
lrE = 1e-3
lrD = 1e-3
crop_size = (512, 512)
load_from_pretrain = True 

Encoder = Encoder().double().cuda(device)
Decoder = Decoder().double().cuda(device)

optE = optim.SGD(Encoder.parameters(), lr=lrD, momentum=0.9)
optD = optim.SGD(Decoder.parameters(), lr=lrE, momentum=0.9)

a_path = '/home/zhuyuanjin/data/Human_Matting/alpha'
img_path = '/home/zhuyuanjin/data/Human_Matting/image_seg'

paramE = "/home/zhuyuanjin/data/Human_Matting/models/paramE_seg_without_trimap"
paramD = '/home/zhuyuanjin/data/Human_Matting/models/paramD_seg_without_trimap'


dataset = MattingDataSet(a_path=a_path, img_path=img_path, crop_size=crop_size)
dataloader = DataLoader(dataset, num_workers=10 , batch_size=batch_size, shuffle=True)

if __name__ == '__main__':
    print("The length of the DataSet is %d" % len(dataset))

    print('Beginning to PreTrain the Encoder Decoder')
#    if os.path.exists(ed_pretrained):
#        print("loading param from %s" % ed_pretrained)
#        state_dict = torch.load(ed_pretrained)
#        ED.load_state_dict(state_dict['net'])
#        #opt_ED.load_state_dict(state_dict['optim'])
    if os.path.exists(paramE) and load_from_pretrain:
        print("Loading param from %s" % paramE)
        state_dict = torch.load(paramE)
        Encoder.load_state_dict(state_dict["net"])
    

    if os.path.exists(paramD) and load_from_pretrain:
        print("Loading param from %s" % paramD)
        state_dict = torch.load(paramD)
        Decoder.load_state_dict(state_dict["net"])

    
    for epoch in range(ed_epoch):
        for batch in dataloader:
            img, trimap, seg = batch['img'].cuda(device), batch['trimap'].cuda(device), batch["seg"].cuda(device)

           # input = torch.cat((img, trimap), 1)
            features = Encoder(img)
            seg_pred = Decoder(features) 
            #img_predict = (fg * alpha_predict + bg * (1-alpha_predict)) * unknown
            #loss_comp = F.mse_loss(img_predict * unknown, img * unknown)
            loss = F.binary_cross_entropy_with_logits(seg_pred, seg)
            print(loss.item(), flush=True)
            torch.save({'net':Encoder.state_dict(), 'optim':optE.state_dict()}, paramE)
            torch.save({'net':Decoder.state_dict(), 'optim':optD.state_dict()}, paramD)
#            total_loss += loss.item()
            optE.zero_grad()
            optD.zero_grad()
            loss.backward()
            optE.step()
            optD.step()
#            if cnt % 100 == 0:
#                torch.save(ED.state_dict(), ed_pretrained)
#                print("epoch", epoch,cnt * batch_size ,total_loss/100)
#                total_loss = 0












