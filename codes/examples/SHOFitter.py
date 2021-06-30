"""
Example of SHO Fitter
"""

import torch
from torch import nn

class SHO_fitting_function:
  """
  Class that fits to SHO
  """

  def __init__(self, x_vector, frequency, device='cuda'):
    self.x_vector = x_vector
    self.frequency = frequency
    self.device = device

  def compute(self, params, device='cuda'):
    Amp = params[:, 0].type(torch.complex128)
    w_0 = params[:, 1].type(torch.complex128)
    Q = params[:, 2].type(torch.complex128)
    phi = params[:, 3].type(torch.complex128)
    wvec_freq = torch.tensor(self.frequency)

    Amp = torch.unsqueeze(Amp, 1)
    w_0 = torch.unsqueeze(w_0, 1)
    phi = torch.unsqueeze(phi, 1)
    Q = torch.unsqueeze(Q, 1)

    wvec_freq = wvec_freq.to(device)

    numer = Amp * torch.exp((1.j) * phi) * torch.square(w_0)
    den_1 = torch.square(wvec_freq)
    den_2 = (1.j) * wvec_freq.to(device) * w_0 / Q
    den_3 = torch.square(w_0)

    den = den_1 - den_2 - den_3

    func = numer / den

    return func

class SHO_Model(nn.Module):
    def __init__(self, x_vector, model, dense_params=4, 
                 device='cuda', num_channels=2, **kwargs):
        super().__init__()
        self.x_vector = x_vector
        self.dense_params = dense_params
        self.device = device
        self.num_channels = num_channels
        self.model_vector = kwargs.get('model_vector')
        self.model = model(self.x_vector, self.model_vector)

        if torch.cuda.is_available():
            self.cuda()

        # Input block of 1d convolution
        self.hidden_x1 = nn.Sequential(
            nn.Conv1d(in_channels=self.num_channels, out_channels=8, kernel_size=7),
            nn.SELU(),
            nn.Conv1d(in_channels=8, out_channels=6, kernel_size=7),
            nn.SELU(),
            nn.Conv1d(in_channels=6, out_channels=4, kernel_size=5),
            nn.SELU(),
        )

        # fully connected block
        self.hidden_xfc = nn.Sequential(
            nn.Linear(256, 20),
            nn.SELU(),
            nn.Linear(20, 20),
            nn.SELU(),
        )

        # 2nd block of 1d-conv layers
        self.hidden_x2 = nn.Sequential(
            nn.MaxPool1d(kernel_size=2),
            nn.Conv1d(in_channels=2, out_channels=4, kernel_size=5),
            nn.SELU(),
            nn.Conv1d(in_channels=4, out_channels=4, kernel_size=5),
            nn.SELU(),
            nn.Conv1d(in_channels=4, out_channels=4, kernel_size=5),
            nn.SELU(),
            nn.Conv1d(in_channels=4, out_channels=4, kernel_size=5),
            nn.SELU(),
            nn.Conv1d(in_channels=4, out_channels=4, kernel_size=5),
            nn.SELU(),
            nn.Conv1d(in_channels=4, out_channels=4, kernel_size=5),
            nn.SELU(),
            nn.AvgPool1d(kernel_size=2),
            nn.Conv1d(in_channels=4, out_channels=2, kernel_size=3),
            nn.SELU(),
            nn.AvgPool1d(kernel_size=2),
            nn.Conv1d(in_channels=2, out_channels=2, kernel_size=3),
            nn.SELU(),
            nn.AvgPool1d(kernel_size=2),
        )

        # Flatten layer
        self.flatten_layer = nn.Flatten()
        
        # Final embedding block - Output 4 values - linear
        self.hidden_embedding = nn.Sequential(
            nn.Linear(26, 16),
            nn.SELU(),
            nn.Linear(16, 8),
            nn.SELU(),
            nn.Linear(8, self.dense_params),
        )

    def forward(self, x, n=-1):
      x = torch.swapaxes(x, 1, 2) # output shape - samples, (real, imag), frequency
      x = self.hidden_x1(x)
      xfc = torch.reshape(x, (n, 256)) # batch size, features
      xfc = self.hidden_xfc(xfc)
      x = torch.reshape(x, (n, 2, 128)) # batch size, (real, imag), timesteps
      x = self.hidden_x2(x)
      cnn_flat = self.flatten_layer(x)
      encoded = torch.cat((cnn_flat, xfc), 1) # merge dense and 1d conv.
      embedding = self.hidden_embedding(encoded) # output is 4 parameters

      # corrects the scaling of the parameters
      unscaled_param = embedding*torch.tensor(params_scaler.var_[0:4]**0.5).cuda() \
                              + torch.tensor(params_scaler.mean_[0:4]).cuda()

      # passes to the pytorch fitting function 
      # fits = SHO_fit_func_torch(unscaled_param, wvec_freq, device='cuda')

      fits = self.model.compute(unscaled_param)

      # extract and return real and imaginary      
      real = torch.real(fits)
      real_scaled = (real - torch.tensor(scaler_real.mean).cuda())\
                                        /torch.tensor(scaler_real.std).cuda()
      imag = torch.imag(fits)
      imag_scaled = (imag - torch.tensor(scaler_imag.mean).cuda())\
                                        /torch.tensor(scaler_imag.std).cuda()
      out = torch.stack((real_scaled, imag_scaled), 2)
      return out

# model = SHO_Model().cuda()
model = SHO_Model(data_, SHO_fitting_function, model_vector=wvec_freq).cuda()