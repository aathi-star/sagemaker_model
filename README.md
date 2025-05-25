# Halton-MaskGIT 3D Training for SageMaker

This directory contains all the necessary code files for training the MaskGIT3D model on SageMaker.

## Files Included

- train_3d.py: Main training script (modify for SageMaker paths)
- Network/: Neural network architecture
- Sampler/: Sampling strategies including Halton sequence
- Losses/: Loss functions
- datasets/: Data loading utilities
- Utils/: Utility functions

## Important SageMaker Modifications Needed

1. Modify train_3d.py to read paths from SageMaker environment variables:
   - Dataset path: os.environ['SM_CHANNEL_TRAIN']
   - Output directory: os.environ['SM_MODEL_DIR']
   - Log directory: os.environ['SM_OUTPUT_DATA_DIR']

2. Make sure Halton3DSampler is properly imported

3. Upload data to S3 and use SageMaker PyTorch Estimator to train
