## RBT-ViT: Reconstructing Before Training for Post-Training Quantization of Vision Transformers

<!-- ![adalog](./assets/framework.png) -->

## Getting Started

- Clone this repo.

```bash
git clone git@github.com:GoatWu/RBT-ViT.git
cd AdaLog
```

- Install pytorch and [timm](https://github.com/huggingface/pytorch-image-models/tree/main).

```bash
pip install torch==1.10.0 torchvision --index-url https://download.pytorch.org/whl/cu113
pip install timm==0.9.2
```

For more details on setting up and running the quantization of detection models, please refer to [Object-Detection/README.md](https://github.com/GoatWu/RBT-ViT/blob/master/Object-Detection/README.md)

## Evaluation

You can quantize and evaluate a single model using the following command:

```bash
python test_quant.py --model <MODEL> --config <CONFIG_FILE> --dataset <DATA_DIR> [--reconstruct-mlp] [--load-reconstruct-checkpoint <RECON_CKPT>] [--calibrate] [--load-calibrate-checkpoint <CALIB_CKPT>] [--optimize]
python test_quant.py --model <MODEL> --config <CONFIG_FILE> --dataset <DATA_DIR> [--calibrate] [--test-checkpoint <CKPT_FILE>] [--save-checkpoint <SAVE_FILE>]
```

- `--model <MODEL>`: Model architecture, which can be `deit_tiny`, `deit_small`, `deit_base`, `vit_tiny`, `vit_small`, `vit_base`, `swin_tiny`, `swin_small` and `swin_base`.

- `--config <CONFIG_FILE>`: Path to the model quantization configuration file.

- `--dataset <DATA_DIR>`: Path to ImageNet dataset.

- `--reconstruct-mlp`: Wether to use MLP reconstruction.

- `--load-reconstruct-checkpoint <CALIB_CKPT>`: When using `--reconstruct-mlp`, we can directly load a reconstructed checkpoint.

- `--calibrate` and `--load-calibrate-checkpoint <CALIB_CKPT>`: A `mutually_exclusive_group` to choose between quantizing an existing model or directly load a calibrated model. The default selection is `--calibrate`.

- `--optimize`: Wether to perform Adaround optimization after calibration.

Example: Optimize the model after reconstruction and calibration.

```bash
python test_quant.py --model vit_small --config ./configs/vit_config.py --dataset ~/data/ILSVRC/Data/CLS-LOC --val-batchsize 500 --reconstruct-mlp --calibrate --optimize
```

Example: Load a reconstructed checkpoint, then run calibration and optimization.

```bash
python test_quant.py --model vit_small --config ./configs/vit_config.py --dataset ~/data/ILSVRC/Data/CLS-LOC --val-batchsize 500 --reconstruct-mlp --load-reconstruct-checkpoint ./checkpoints/quant_result/deit_tiny_reconstructed.pth --calibrate --optimize
```

Example: Load a calibrated checkpoint, and then run optimization.

```bash
python test_quant.py --model vit_small --config ./configs/vit_config.py --dataset ~/data/ILSVRC/Data/CLS-LOC --val-batchsize 500 --reconstruct-mlp --load-reconstruct-checkpoint ./checkpoints/quant_result/deit_tiny_reconstructed.pth --load-calibrate-checkpoint ./checkpoints/quant_result/deit_tiny_w3_a3_s3_mse.pth --optimize
```

