# 3D-PI0

This repository contains code for training and evaluating 3D-PI0, a vision-language model for 3D understanding and generation.

## Setup

1. Create and activate conda environment:
```bash
conda create -n cogvideo python=3.10
conda activate cogvideo
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Training

To train the model:

```bash
bash slurm/train_multi_gpu.sh
```

## Directory Structure

- `slurm/`: Training scripts and configurations
- `scripts/`: Main training code
- `src/`: Source code
  - `cog_video/`: Core model implementation

## Notes

- The model requires GPU for training
- Logs are stored in `logs/` directory
- Model checkpoints will be saved in the specified checkpoint directory

## License

MIT
