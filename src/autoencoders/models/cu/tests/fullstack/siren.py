import os
import torch
from torch import nn
from time import time
# Convention: model class is named 'Autoencoder' or endswith 'Autoencoder', config is 'Config' or endswith 'Config'

from cu.compile import compile
nn_siren = compile(
    kernel=os.path.join(os.path.dirname(__file__), "siren.cu"),
    template_kwargs={
        "LEARNING_RATE": 1e-4,
    }
)
##
# compiled?
print("Network compiled:", nn_siren is not None)




def _test_nn_basic_siren():
    cx = torch.linspace(-1.0, 1.0, steps=32)[None, :].repeat(32, 1)
    cy = torch.linspace(-1.0, 1.0, steps=32)[:, None].repeat(1, 32)
    x = torch.stack([cx, cy], dim=0).unsqueeze(0).cuda()  # Shape: (1, 2, 32, 32)
    g = lambda c: torch.stack(
        [c[:,0], c[:,1], c[:,0]], dim=1)
    f = lambda c: torch.stack(
        [
            torch.sin(4*c[:,0]) + 0.5 * (c[:,1]),
            torch.sin(2*c[:,0]),# - 0.5*torch.cos(c[:,0] + 4*c[:,1]),
            torch.sin(-3*c[:,1]),# + 0.5
            # *[0 * c[:,0] for _ in range(13)]
        ], dim=1
    )
    
    import matplotlib.pyplot as plt
    import numpy as np
    
    y = f(x)
    yhat = torch.zeros_like(y)
    
    data_T = []
    train_T = []
    eval_T = []
    MSE = []
    Bs = [1, 10, 20, 50, 100, 200, 500]

    N = 100 * 1000
    for b in Bs:
        s = N // b
        
        dataT = 0
        trainT = 0
        evalT = 0
        
        MSE_T = []
        
        mem_pointer = 0
        mem_pointer = nn_siren.train(g(x), y, mem_pointer, 1) # warmup quirk
        for i in range(b):
            # load new data too every iteration, technically
            
            t = time()
            q = torch.rand_like(x).cuda() * 2.0 - 1.0 # New random input
            xs = g(q)
            ys = f(q)      
            t0 = time()
            dataT += (t0 - t)
            
            mem_pointer = nn_siren.train(xs, ys, mem_pointer, s)
            t1 = time()
            trainT += (t1 - t0)
            
            nn_siren.eval(g(x), yhat, mem_pointer, 0)
            t2 = time() - t1
            evalT += t2
            
            error = torch.mean((y - yhat) ** 2)
            MSE_T.append(error.item())
    
            # print("Mem pointer after training:", mem_pointer)
            
        # empty cuda cache to avoid OOM in later tests
        torch.cuda.empty_cache()

        # nn_siren.eval(x, yhat, mem_pointer, 0)
        # # print("Output after eval:", yhat)
        # error = torch.mean((y - yhat) ** 2)
        # signal = torch.mean((yhat - x) ** 2)
        # print("MSE:", error.item())
        # print("Signal after eval:", signal)
        # print(f"SNR: {10 * torch.log10(signal / error).item():.2f} dB")
        
        ms = 1000
        MSE.append(np.array(MSE_T))
        data_T.append(ms * dataT / s)
        train_T.append(ms * trainT / s)
        eval_T.append(ms * evalT / s)
        
        fig,ax = plt.subplots(1,3, figsize=(12,4))
        ax[0].imshow(g(x)[0,:3].permute(1,2,0).cpu().numpy())
        ax[0].set_title("Input")
        ax[1].imshow(y[0,:3].permute(1,2,0).cpu().numpy())
        ax[1].set_title("Target")
        ax[2].imshow(yhat[0,:3].permute(1,2,0).cpu().numpy())
        ax[2].set_title(f"Output with {b} batches")
        plt.savefig(f"siren_output_b{b}.png")
        plt.close(fig)
    
    # from cu.tests import test_layers as tl
    # tl._plot_diff(x, yhat, title="NN Siren Check Difference")
    
    # assert error < 1e-4, f"NN Siren Check failed with MSE {error}"

    data_T = np.array(data_T)
    train_T = np.array(train_T)
    eval_T = np.array(eval_T)
    
    total_T = data_T + train_T + eval_T   
    train_perc = 100.0 * train_T / total_T 

    fig, ax = plt.subplots(1,3, figsize=(18,6))
    for i in range(len(Bs)):
        b = Bs[i]
        x = (N//b) * np.arange(b)
        ax[0].plot(x, MSE[i], label=f"B={Bs[i]}")
    ax[0].set_title("MSE")
    ax[0].set_xlabel("iteration")
    ax[0].set_ylabel("MSE")
    ax[0].set_yscale("log")
    ax[0].legend()
    
    ax[1].set_title("Time per sample [ms]")
    ax[1].plot(Bs, data_T, label="Data Load Time")
    ax[1].plot(Bs, train_T, label="Train Time")
    ax[1].plot(Bs, eval_T, label="Eval Time")
    ax[1].set_xlabel("Number of Batches")
    ax[1].set_ylabel("Time [ms]")
    ax[1].legend()
    
    ax[2].plot(Bs, train_perc, label="Train %")
    ax[2].set_title("Train Time Percentage")
    ax[2].set_xlabel("Number of Batches")
    ax[2].set_ylabel("Percentage [%]")
    
    plt.savefig("siren_performance.png")
    plt.close(fig)
        
# def _test_nn_basic_siren():
#     cx = torch.linspace(-1.0, 1.0, steps=32)[None, :].repeat(32, 1)
#     cy = torch.linspace(-1.0, 1.0, steps=32)[:, None].repeat(1, 32)
#     x = torch.stack([*[cx, cy], 
#                      cx
#                     #  *([cx,] * 14) 
#                      ], dim=0).unsqueeze(0).cuda()  # Shape: (1, 2, 32, 32)
    
#     f = lambda c: torch.stack(
#         [
#             torch.sin(4*c[:,0]) + 0.5 * (c[:,1]),
#             torch.sin(2*c[:,0]),# - 0.5*torch.cos(c[:,0] + 4*c[:,1]),
#             torch.sin(-3*c[:,1]),# + 0.5
#             # *[0 * c[:,0] for _ in range(13)]
#         ], dim=1
#     )
    
#     y = f(x)
#     yhat = torch.zeros_like(y)
    
#     train_T = []
#     eval_T = []
#     MSE_T = []
#     SNR_T = []
        
#     mem_pointer = 0
#     mem_pointer = nn_siren.train(x, y, mem_pointer, 1) # warmup quirk
#     for i in range(100):
#         # load new data too every iteration, technically
        
#         t = time()
#         mem_pointer = nn_siren.train(x, y, mem_pointer, 10000)
#         t1 = time()
#         train_T.append(t1 - t)
        
#         nn_siren.eval(x, yhat, mem_pointer, 0)
#         t2 = time() - t1
#         eval_T.append(t2)
        
#         error = torch.mean((y - yhat) ** 2)
#         signal = torch.mean((yhat - x) ** 2)
#         SNR = 10 * torch.log10(signal / error).item()
#         MSE_T.append(error.item())
#         SNR_T.append(SNR)
            
#         # print("Mem pointer after training:", mem_pointer)

#     # nn_siren.eval(x, yhat, mem_pointer, 0)
#     # # print("Output after eval:", yhat)
#     # error = torch.mean((y - yhat) ** 2)
#     # signal = torch.mean((yhat - x) ** 2)
#     # print("MSE:", error.item())
#     # print("Signal after eval:", signal)
#     # print(f"SNR: {10 * torch.log10(signal / error).item():.2f} dB")
    
#     ms = 1000
#     train_T = ms * torch.tensor(train_T) / 10000.0
#     eval_T = ms * torch.tensor(eval_T) / 10000.0
    
#     import matplotlib.pyplot as plt
#     fig, ax = plt.subplots(1,4, figsize=(16,4))
#     ax[0].hist(train_T, bins=50)
#     ax[0].set_title("Train [ms] per it")
#     ax[0].set_xlabel("Time [ms]")
#     ax[0].set_ylabel("Count")
#     ax[1].set_xlabel("Time [ms]")
#     ax[1].hist(eval_T, bins=50)
#     ax[1].set_title("Eval [ms] per it")
#     ax[1].set_ylabel("Count")
#     ax[1].set_xlabel("Time [ms]")
#     ax[2].plot(MSE_T)
#     ax[2].set_ylabel("MSE")
#     ax[2].set_xlabel("Iteration")
#     ax[2].set_yscale("log")
#     ax[2].set_title("MSE")
#     ax[3].plot(SNR_T)
#     ax[3].set_ylabel(r"SNR = $log_{10}(\frac{signal}{MSE})$ [dB]")
#     ax[3].set_xlabel("Iteration")
#     ax[3].set_title("SNR (dB)")
#     plt.savefig("siren_performance.png")
#     plt.close(fig)
    
    
#     fig,ax = plt.subplots(1,3, figsize=(12,4))
#     ax[0].imshow(x[0,:3].permute(1,2,0).cpu().numpy())
#     ax[0].set_title("Input")
#     ax[1].imshow(y[0,:3].permute(1,2,0).cpu().numpy())
#     ax[1].set_title("Target")
#     ax[2].imshow(yhat[0,:3].permute(1,2,0).cpu().numpy())
#     ax[2].set_title("Output")
#     plt.savefig("siren_output.png")
    
#     from cu.tests import test_layers as tl
#     tl._plot_diff(x, yhat, title="NN Siren Check Difference")
    
#     assert error < 1e-4, f"NN Siren Check failed with MSE {error}"
