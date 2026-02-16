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

Finally, if you get CUDA errors, check that you are in the correct venv (or reload just in case).
And make sure you have all modules in module.sh loaded.

Note HuggingFace is only needed for special models and datasets.
Occasionally, `make install` may not install all the requirements. To manually install, source the venv and run:

    ```bash
    uv pip install <package>
    ```

Note if you are running on Engaging, there is a CUDA 13 mismatch that may require you to run the following:

    ```bash
    source src/install/cuda-13-compat.sh
    ```

And their FFMPEG doesn't have libx264 so we build our own:

    ```bash
    source src/install/ffmpeg-libx264-compat.sh
    ```