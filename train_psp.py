from pspnet import PSPNet
import torch.nn.functional as F
import torch.optim as optim
from DataSet import *
import os
from torch import nn

device = 0 
ed_epoch = 100
refine_epoch = 100
final_epoch = 100
batch_size = 32 
lrE = 5e-4
lrD = 5e-4
crop_size = (512, 512)
load_from_pretrain = False

PSPNet = PSPNet().double().cuda(device)

opt = optim.SGD(PSPNet.parameters(), lr=lrE, momentum=0.9)

a_path = '/home/zhuyuanjin/data/Human_Matting/alpha'
img_path = '/home/zhuyuanjin/data/Human_Matting/image_seg'

param_PSP = "/home/zhuyuanjin/data/Human_Matting/models/param_PSP"


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
    

    if os.path.exists(param_PSP) and load_from_pretrain:
        print("Loading param from %s" % param_PSP)
        state_dict = torch.load(param_PSP)
        PSPNet.load_state_dict(state_dict["net"])

    
    for epoch in range(ed_epoch):
        for batch in dataloader:
            img, trimap, seg = batch['img'].cuda(device), batch['trimap'].cuda(device), batch["seg"].cuda(device)

            seg_pred = PSPNet(img) 
            #img_predict = (fg * alpha_predict + bg * (1-alpha_predict)) * unknown
            #loss_comp = F.mse_loss(img_predict * unknown, img * unknown)
            loss = F.binary_cross_entropy_with_logits(seg_pred, seg)
            print(loss.item(), flush=True)
            torch.save({'net':PSPNet.state_dict(), 'optim':opt.state_dict()}, param_PSP)
#            total_loss += loss.item()
            opt.zero_grad()
            loss.backward()
            opt.step()
#            if cnt % 100 == 0:
#                torch.save(ED.state_dict(), ed_pretrained)
#                print("epoch", epoch,cnt * batch_size ,total_loss/100)
#                total_loss = 0












