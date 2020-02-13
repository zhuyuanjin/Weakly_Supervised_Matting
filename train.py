from network import Encoder, Decoder
import torch.nn.functional as F
import torch.optim as optim
from DataSet import *
import os


device = 0 
ed_epoch = 100
batch_size = 5 
crop_size = (512, 512)
 
E = Encoder().double().cuda(device)
D = Decoder().double().cuda(device)

opt_E = optim.SGD(E.parameters(), lr=1e-3, momentum=0.9)
opt_D = optim.SGD(D.parameters(), lr=1e-3, momentum=0.9)

a_path = '/home/zyj/data/Human_Matting/alpha'
img_path = '/home/zyj/data/Human_Matting/image_mat'

paramE = "/home/zyj/data/Human_Matting/models/paramE_without_trimap"
paramD = "/home/zyj/data/Human_Matting/models/paramD_without_trimap"


dataset = MattingDataSet(a_path=a_path, img_path=img_path, crop_size=crop_size)
dataloader = DataLoader(dataset, num_workers=10 , batch_size=batch_size, shuffle=True)

if __name__ == '__main__':
    print("Train model with size of dataset %d, picture size %d" % (len(dataset), crop_size[0]))
    print('Beginning to PreTrain the Encoder Decoder')

    if os.path.exists(paramE):
        print("loading param from %s" % paramE)
        state_dict = torch.load(paramE)
        E.load_state_dict(state_dict['net'])
    if os.path.exists(paramD):
        print("loading param from %s" % paramD)
        state_dict = torch.load(paramD)
        D.load_state_dict(state_dict['net'])
   
 
    for epoch in range(ed_epoch):
        cnt = 0
        total_loss = 0
        for batch in dataloader:
            cnt += 1
            img, alpha, trimap, unknown = batch['img'].cuda(device), \
                                          batch['alpha'].cuda(device), batch['trimap'].cuda(device), \
                                          batch['unknown'].cuda(device)

            alpha_predict = D(E(image))
            loss = F.smooth_l1_loss(alpha_predict * unknown, alpha * unknown)
            print(loss.item(), flush=True)
            opt_E.zero_grad()
            opt_D.zero_grad()
            loss.backward()
            opt_E.step()
            opt_D.step()
            if cnt > 0 and cnt % 20 == 0:
                print("Saving param for Encoder %s" % paramE)
                torch.save(E.state_dict(), paramE)
                print("Saving param for Decoder %s" % paramD)
                torch.save(D.state_dict(), paramD)
