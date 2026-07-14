import torch
import torch.nn as nn
import torch.nn.functional as F

class Derivative(nn.Module):
    def __init__(self, shape=(512,512), L=(1,1)):
        super().__init__()
        
        self.Ny, self.Nx = shape[-2], shape[-1]
        self.Lx, self.Ly = 2 * torch.pi * L[-2], 2 * torch.pi * L[-1]
        dx = self.Lx / self.Nx
        dy = self.Ly / self.Ny
        self.register_buffer('x', torch.arange(-self.Lx/2, self.Lx/2, dx))
        self.register_buffer('y',torch.arange(-self.Ly/2, self.Ly/2, dy))
        
        # number of wavenumber components (half of real grid in x-direction)
        self.dk = int(self.Nx / 2 + 1)

        # pure wavenumbers
        self.register_buffer('ky', torch.reshape((torch.fft.fftfreq(self.Ny, self.Ly / (self.Ny * 2 * torch.pi))), 
            (self.Ny, 1)
        )[None,None,:,:]) 
        
        self.register_buffer('kx', torch.reshape((torch.fft.rfftfreq(self.Nx, self.Lx / (self.Nx * 2 * torch.pi))), 
            (1, self.dk)
        )[None,None,:,:])
        
        ksq = self.kx**2 + self.ky**2
        irsq = 1/ksq
        irsq[...,0,0] = 0.0
        self.register_buffer('irsq', irsq)

        self.register_buffer('dx', 1j * self.kx)
        self.register_buffer('dy', 1j * self.ky)
        
        
    def _dx(self, w):
        return torch.fft.irfft2(torch.fft.rfft2(w) * self.dx)
    def _dy(self, w):
        return torch.fft.irfft2(torch.fft.rfft2(w) * self.dy)
    def phi(self, w):
        return torch.fft.irfft2(torch.fft.rfft2(w) * self.irsq)
    def uv(self, w):
        phi = self.phi(w)
        return (-self._dy(phi), self._dx(phi))
    def adv(self,w,w_nz):
        u, v = self.uv(w_nz)
        wx = self._dx(w)
        wy = self._dy(w)
        return u * wx + v * wy