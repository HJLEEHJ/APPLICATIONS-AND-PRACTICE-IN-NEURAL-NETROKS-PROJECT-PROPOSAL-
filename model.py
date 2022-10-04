import torch.nn as nn


class MODEL(nn.Module):
    def __init__(self):
        super(MODEL,self).__init__()

        self.enc_0 = nn.Sequential(nn.Conv2d(1,32,kernel_size=(2, 2),padding=1),nn.BatchNorm2d(32),nn.ReLU(inplace=True))
        self.enc_1 = nn.Sequential(nn.Conv2d(32,64,kernel_size=(2, 2)),nn.BatchNorm2d(64),nn.ReLU(inplace=True))
        self.enc_2 = nn.Sequential(nn.Conv2d(64,128,kernel_size=(2, 2),padding=1),nn.BatchNorm2d(128),nn.ReLU(inplace=True))
        self.enc_3 = nn.Sequential(nn.Conv2d(128,512,kernel_size=(2, 2)),nn.BatchNorm2d(512),nn.ReLU(inplace=True))

           
        self.emb_gru = nn.GRU( 161 * 512 , 161 * 512 )

        self.dec_3 = nn.Sequential(nn.ConvTranspose2d(512,128,kernel_size=(2, 2)),nn.BatchNorm2d(128),nn.ReLU(inplace=True))
        self.dec_2 = nn.Sequential(nn.ConvTranspose2d(128,64,kernel_size=(2, 2),padding=1),nn.BatchNorm2d(64),nn.ReLU(inplace=True))
        self.dec_1 = nn.Sequential(nn.ConvTranspose2d(64,32,kernel_size=(2, 2)),nn.BatchNorm2d(32),nn.ReLU(inplace=True))
        self.dec_0 = nn.Sequential(nn.ConvTranspose2d(32,1,kernel_size=(2, 2),padding=1),nn.BatchNorm2d(1),nn.ReLU(inplace=True))


    def forward(self, input):
        
        b, _, t, _ = input.shape  

        e0 = self.enc_0(input) # [B, 32, 102, 162]
        e1 = self.enc_1(e0) # [B, 64, 101, 161]
        e2 = self.enc_2(e1) # [B, 128, 102, 162]
        e3 = self.enc_3(e2) # [B, 512, 101, 161]

        emb = e3.permute(2, 0, 1, 3).reshape(t, b, -1)  # [T, B , 512*F]
        emb, _ = self.emb_gru(emb)
        emb = emb.reshape(t, b, 512, -1).permute(1,2,0,3)

        # print("emb.shape",emb.shape)

        d3 = self.dec_3(emb)   # ([B, 128, 102, 162])
        # d3 = self.dec_3(e3)
        d2 = self.dec_2(d3)  # [B, 64, 101, 161]
        d1 = self.dec_1(d2)  # [B, 32, 102, 162]
        d0 = self.dec_0(d1)  # [B, 1, 101, 161]

        out = d0 * input

        return out
        


# model = STO()
# stft = torch.rand(6, 1, 161, 101)
# out = model(stft)
# print("out:",out.shape)
