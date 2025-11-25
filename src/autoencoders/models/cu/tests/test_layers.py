# generate random data for testing cu layers
import torch

sizes=[
    # (16, 1, 28, 28),
    (32, 3, 64, 64),
    (4, 3, 128, 128),
    (4, 3, 512, 512),
]

def random_reduce(err, weights=None):
    if weights is None:
        weights = torch.rand_like(err)
    loss = weights * (err) ** 2
    return loss.mean(), weights

def get_random_data(batch_size: int, channels: int, height: int, width: int) -> torch.Tensor:
    torch.manual_seed(42)
    return torch.randn(batch_size, channels, height, width, device="cuda")

def _check(true_func, cu_func):
    for size in sizes:
        x = get_random_data(*size)
        x.requires_grad_()
        
        x_2 = x.clone().detach()
        x_2.requires_grad_()
        
        try:
            y = true_func(x)
            y_hat = cu_func(x_2)
            assert torch.allclose(y_hat, y, atol=0.01, rtol=0.1), f"Forward check failed for size {size}"
            
            lz, wt = random_reduce(y)
            lz.backward()
            x_g = x.grad
            
            random_reduce(y_hat, weights=wt)[0].backward()
            x_g_hat = x_2.grad
            assert torch.allclose(x_g, x_g_hat, atol=0.01, rtol=0.1), f"Gradient check failed for size {size}"
        except torch.AcceleratorError as e:
            print(f"Size {size} failed due to CUDA error: {e}")
            raise e
        

