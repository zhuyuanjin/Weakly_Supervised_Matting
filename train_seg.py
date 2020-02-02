
from network import EncoderDecoder19
import torch.nn.functional as F
import torch.optim as optim
from DataSet import *
import os
from refinement import RefineNet
from torch import nn

device = 1 
ed_epoch = 100
refine_epoch = 100
final_epoch = 100
batch_size = 10 

ED = EncoderDecoder19().double().cuda(device)

opt_ED = optim.SGD(ED.parameters(), lr=1e-2, momentum=0.9)

a_path = '/home/zhuyuanjin/data/Human_Matting/alpha'
img_path = '/home/zhuyuanjin/data/Human_Matting/image_seg'

ed_pretrained = '/home/zhuyuanjin/data/Human_Matting/models/param_seg19'



dataset = MattingDataSet(a_path=a_path, img_path=img_path, )
dataloader = DataLoader(dataset, num_workers=10 , batch_size=batch_size, shuffle=True)

if __name__ == '__main__':
    print("The length of the DataSet is %d" % len(dataset))

    print('Beginning to PreTrain the Encoder Decoder')
#    if os.path.exists(ed_pretrained):
#        print("loading param from %s" % ed_pretrained)
#        state_dict = torch.load(ed_pretrained)
#        ED.load_state_dict(state_dict['net'])
#        #opt_ED.load_state_dict(state_dict['optim'])
    
    for epoch in range(ed_epoch):
        cnt = 0
        total_loss = 0
        for batch in dataloader:
            cnt += 1
            img, alpha, trimap, unknown, seg = batch['img'].cuda(device), \
                                              batch['alpha'].cuda(device), batch['trimap'].cuda(device), \
                                              batch['unknown'].cuda(device), batch["seg"].cuda(device)

            input = torch.cat((img, trimap), 1)
            seg_pred = ED(input)
            #img_predict = (fg * alpha_predict + bg * (1-alpha_predict)) * unknown
            #loss_comp = F.mse_loss(img_predict * unknown, img * unknown)
            loss_seg = F.binary_cross_entropy_with_logits(seg_pred, seg)
            loss = loss_seg
            print(loss.item(), flush=True)
            torch.save({'net':ED.state_dict(), 'optim':opt_ED.state_dict()}, ed_pretrained)
#            total_loss += loss.item()
            opt_ED.zero_grad()
            loss.backward()
            opt_ED.step()
#            if cnt % 100 == 0:
#                torch.save(ED.state_dict(), ed_pretrained)
#                print("epoch", epoch,cnt * batch_size ,total_loss/100)
#                total_loss = 0











