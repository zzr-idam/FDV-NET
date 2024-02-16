import torch
from torch import nn
import torch.nn.functional as F
from mamba_ssm import Mamba

class CSSMB(nn.Module):
    def __init__(self,c=3,w=64,h=64):
        super(CSSMB, self).__init__()

        self.convb = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=8, kernel_size=3, padding=1, stride=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=8, out_channels=3, kernel_size=3, padding=1, stride=1),
        )
        self.ln = nn.LayerNorm(normalized_shape=c)
        self.sm = nn.Softmax(dim=0)
        
        self.model1 = Mamba(
    # This module uses roughly 3 * expand * d_model^2 parameters
            d_model=c, # Model dimension d_model
            d_state=16,  # SSM state expansion factor
            d_conv=4,    # Local convolution width
            expand=2,    # Block expansion factor
        )
        
        self.model2 = Mamba(
    # This module uses roughly 3 * expand * d_model^2 parameters
            d_model=c, # Model dimension d_model
            d_state=16,  # SSM state expansion factor
            d_conv=4,    # Local convolution width
            expand=2,    # Block expansion factor
        )
        self.con1_1 = nn.Conv1d(in_channels=w*h, out_channels=w*h, kernel_size=3, padding=1, stride=1)
        self.con1_2 = nn.Conv1d(in_channels=w*h, out_channels=w*h, kernel_size=3, padding=1, stride=1)
        
        self.cross_1 = nn.Conv1d(in_channels=w*h, out_channels=w*h, kernel_size=3, padding=1, stride=1)
        self.cross_2 = nn.Conv1d(in_channels=w*h, out_channels=w*h, kernel_size=3, padding=1, stride=1)
        
    def forward(self, x):

        ap = torch.fft.fft2(x)
        amp = ap.real
        pha = ap.imag
        
        raw_amp = amp
        raw_pha = pha
        
        
        b,c,w,h = amp.shape
        
        amp = amp + self.convb(amp)
        pha = pha + self.convb(pha)
        amp = self.ln(amp.reshape(b, -1, c))
        pha = self.ln(pha.reshape(b, -1, c))
        #
        amp = self.model1(amp)
        amp1 = self.con1_1(amp)
        amp2 = self.sm(amp)
        amp = amp1 * amp2
        amp_out = amp.reshape(b, c, w, h)
        #
        pha = self.model2(pha)
        pha1 = self.con1_2(pha)
        pha2 = self.sm(pha)
        pha = pha1 * pha2
        pha_out = pha.reshape(b, c, w, h)
        
        amp3 = self.cross_1(amp2)
        amp4 = self.sm(amp3)
        
        pha3 = self.cross_1(amp2)
        pha4 = self.sm(pha3)
        
        attention_amp = amp3 * pha4 
        amp_out = amp_out + attention_amp.reshape(b, c, w, h) + raw_amp 
        
        attention_pha = amp4 * pha3 
        pha_out = pha_out + attention_pha.reshape(b, c, w, h) + raw_pha 

        output = torch.real(torch.fft.ifft2(torch.complex(amp_out, pha_out)))
        
        return output
    

#data = torch.randn(1, 3, 64, 64).cuda()

#model = CSSMB().cuda()

#ap = torch.fft.fft2(data)
#ap_real = ap.real
#ap_imag = ap.imag

#amp, pha = model(ap_real, ap_imag)

#output = torch.real(torch.fft.ifft2(torch.complex(amp, pha)))

#print(output.shape)

