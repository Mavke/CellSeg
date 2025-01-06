import torch
import torch.nn as nn
import torch.nn.functional as F

class Sobel(nn.Module):
    def __init__(self):
        super(Sobel, self).__init__()
        self.sobel_x = nn.Conv2d(1, 1, kernel_size=3, stride=1, padding=1, bias=False,padding_mode='replicate')
        self.sobel_y = nn.Conv2d(1, 1, kernel_size=3, stride=1, padding=1, bias=False, padding_mode='replicate')
        
        sobel_x_weights = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32)
        sobel_y_weights = torch.tensor([[-1.0, -2.0, -1.0], [0.0, 0.0, 0.0], [1.0, 2.0, 1.0]], dtype=torch.float32)
        
        self.sobel_x.weight = nn.Parameter(sobel_x_weights.unsqueeze(0).unsqueeze(0), requires_grad=False)
        self.sobel_y.weight = nn.Parameter(sobel_y_weights.unsqueeze(0).unsqueeze(0), requires_grad=False)

    def forward(self, image_h, image_v):
        image_h = image_h.unsqueeze(1)  # Add channel dimension
        image_v = image_v.unsqueeze(1)
        x_sobel = self.sobel_x(image_h)
        y_sobel = self.sobel_y(image_v)

        return x_sobel, y_sobel

def one_hot_encoder(input_tensor, n_classes):
    tensor_list = []
    for i in range(n_classes):
        temp_prob = input_tensor == i  # * torch.ones_like(input_tensor)
        tensor_list.append(temp_prob.unsqueeze(1))
    output_tensor = torch.cat(tensor_list, dim=1)
    
    return output_tensor.float()