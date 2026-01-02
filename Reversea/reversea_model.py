import torch
import torch.nn as nn
import torch
from losses import histogram_equalization_loss,color_constancy_loss
try:
    from tqdm import trange
except:
    trange = range
    
    
class ReverseaNet(nn.Module):
    def __init__(self):
        super().__init__()
        # Define separate layers for each channel
        self.R_layer = nn.Conv2d(1, 1, kernel_size=3, stride=1, padding=1, bias=False)
        self.G_layer = nn.Conv2d(1, 1, kernel_size=3, stride=1, padding=1, bias=False)
        self.B_layer = nn.Conv2d(1, 1, kernel_size=3, stride=1, padding=1, bias=False)
        
        self.map = nn.Conv2d(3, 1, kernel_size=3, stride=1, padding=1, bias=False)

        # Initialize weights
        nn.init.uniform_(self.R_layer.weight, 0, 5)
        nn.init.uniform_(self.G_layer.weight, 0, 5)
        nn.init.uniform_(self.B_layer.weight, 0, 5)
        
        nn.init.uniform_(self.map.weight, 0, 5)

        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()
        self.sigmoid=nn.Sigmoid()

    def forward(self, image, depth):
        # Separate the R, G, B channels
        R = image[:, 0:1, :, :]
        G = image[:, 1:2, :, :]
        B = image[:, 2:3, :, :]

        # Enhance each channel separately
        R_enhanced = self.relu(self.R_layer(R))
        G_enhanced = self.relu(self.G_layer(G))
        B_enhanced = self.relu(self.B_layer(B))
        

        # Combine the enhanced channels back into a single image
        A1 = torch.cat([R_enhanced, G_enhanced, B_enhanced], dim=1)
        A = self.tanh(A1)
        direct = image - (A)

        # Dividing by depth
        div = direct / (depth + 1e-6)
        
        div = self.relu(self.map(div))
       
        # Final enhanced image
        J = (A+div)
        return A1, A, J, direct, div
    
    
class ReverseaLoss(nn.Module):
    def __init__(self, cost_ratio=1000.):
        super().__init__()
        self.mse = nn.MSELoss()
        self.relu = nn.ReLU()
        self.target_intensity = 0.5

    def forward(self,A,A1,J):
        hist_loss = histogram_equalization_loss(J)
        channel_intensities = torch.mean(J, dim=[2, 3], keepdim=True)
        cc_loss = color_constancy_loss(J)
        channel_intensities = torch.mean(J, dim=[2, 3], keepdim=True)
        lum_loss = (channel_intensities - self.target_intensity).square().mean()
        return ((0.25 *hist_loss)+(0.5*cc_loss)+(lum_loss)) 
