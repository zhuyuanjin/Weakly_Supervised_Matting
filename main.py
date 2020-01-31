
from encoder_decoder import EncoderDecoder
import torch.nn.functional as F
import torch.optim as optim
from DataSet import *
import os
from refinement import RefineNet


device = 1 
ed_epoch = 100
refine_epoch = 100
final_epoch = 100
batch_size = 32 

RF = RefineNet().double().cuda(device)
ED = EncoderDecoder().double().cuda(device)

opt_ED = optim.SGD(ED.parameters(), lr=5e-4, momentum=0.9)
opt_RF = optim.SGD(RF.parameters(), lr=5e-2, momentum=0.9)

a_path = '/home/zhuyuanjin/data/Human_Matting/alpha'
img_path = '/home/zhuyuanjin/data/Human_Matting/image'
name_file='/home/zhuyuanjin/data/Human_Matting/MattingSet.txt'

ed_pretrained = '/home/zhuyuanjin/data/Human_Matting/models/ed_pretrained'
rf_pretrained = '/home/zhuyuanjin/data/Human_Matting/models/rf_pretrained'

final_param = '/home/zhuyuanjin/data/Human_Matting/models/final_param'


dataset = MattingDataSet(a_path=a_path, img_path=img_path, name_file=name_file)
dataloader = DataLoader(dataset, num_workers=10 , batch_size=batch_size, shuffle=True)

if __name__ == '__main__':

    print('Beginning to PreTrain the Encoder Decoder')
    if os.path.exists(ed_pretrained):
        print("loading param from %s" % ed_pretrained)
        state_dict = torch.load(ed_pretrained)
        ED.load_state_dict(state_dict['net'])
        #opt_ED.load_state_dict(state_dict['optim'])
    
    for epoch in range(ed_epoch):
        cnt = 0
        total_loss = 0
        for batch in dataloader:
            cnt += 1
            img, alpha, trimap, unknown = batch['img'].cuda(device), \
                                              batch['alpha'].cuda(device), batch['trimap'].cuda(device), \
                                              batch['unknown'].cuda(device)

            input = torch.cat((img, trimap), 1)
            alpha_predict = ED(input)
            #img_predict = (fg * alpha_predict + bg * (1-alpha_predict)) * unknown
            #loss_comp = F.mse_loss(img_predict * unknown, img * unknown)
            loss_alpha = F.smooth_l1_loss(alpha_predict * unknown, alpha * unknown)
            loss = loss_alpha
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

    print('Beginning to PreTrain the RefineNet')
    ED.load_state_dict(torch.load(ed_pretrained)['net'])
    if os.path.exists(rf_pretrained):
        print("loading param from %s" % rf_pretrained)
        state_dict = torch.load(rf_pretrained)
        RF.load_state_dict(state_dict['net'])
        #opt_RF.load_state_dict(state_dict['optim'])

    for epoch in range(refine_epoch):
        for batch in dataloader:
            img, alpha, trimap, unknown = batch['img'].cuda(device),  \
                                              batch['alpha'].cuda(device), batch['trimap'].cuda(device), \
                                              batch['unknown'].cuda(device)
            input = torch.cat((img, trimap), 1)
            alpha_raw = ED(input) * unknown + alpha * (1-unknown)
            alpha_refined = RF(torch.cat((img, alpha_raw), 1))
            loss_refine = F.smooth_l1_loss(alpha_refined, alpha)
            print(loss_refine.item(), flush=True)
            torch.save({'net': RF.state_dict(), 'optim': opt_RF.state_dict()}, rf_pretrained)
            opt_RF.zero_grad()
            loss_refine.backward()
            opt_RF.step()


    print('Begining to Train the whole Model')

    for epoch in range(final_epoch):
        for batch in dataloader:
            img, alpha, trimap, unknown = batch['img'].cuda(device), \
                                              batch['alpha'].cuda(device), batch['trimap'].cuda(device), \
                                              batch['unknown'].cuda(device)
            input = torch.cat((img, trimap), 1)
            alpha_raw = ED(input) * unknown + alpha * (1 - unknown)
            alpha_refined = RF(torch.cat((img, alpha_raw), 1))
            loss_refine = F.smooth_l1_loss(alpha_refined, alpha)
            print(loss_refine.item(), flush=True)
            torch.save({'net_RF': RF.state_dict(), 'optim_RF': opt_RF.state_dict(), 'net_ED':ED.state_dict(), 'optim_ED':opt_ED.state_dict()},final_param) 
            opt_RF.zero_grad()
            opt_ED.zero_grad()
            loss_refine.backward()
            opt_RF.step()
            opt_ED.step()













