# autoencoders


#### install
- On remote, make sure to login in WandB and HuggingFace after `make install` or `make sinstall` if on a SLURM cluster.
- Don't forget to source the `venv` with `source .venv/bin/activate`

    ```bash
    wandb login
    huggingface-cli login
    ```

Also set your data cache directory in bashrc or equivalent:

```bash
export AUTOENCODERS_DATA_ROOT=/path/to/cache
```

Note HuggingFace is only needed for special models and datasets.
Occasionally, `make install` may not install all the requirements. To manually install, source the venv and run:

    ```bash
    uv pip install <package>
    ```

Note if you are running on Engaging, there is a CUDA 13 mismatch that may require you to run the following:

    ```bash
    source src/install/cuda-13-compat.sh
    ```