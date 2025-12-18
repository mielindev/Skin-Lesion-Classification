# AI in Bioinformatics: Skin Lesion Classification

## Project Overview

This project focuses on the classification of skin lesions using deep learning techniques. The dataset includes images of various skin conditions, and the models are trained to classify these conditions into predefined categories. The project employs convolutional neural networks (CNNs) and transfer learning with ResNet50 to achieve high accuracy.

## Directory Structure

```
.
├── aug.ipynb                # Data augmentation notebook
├── CNN.ipynb                # CNN model training and evaluation
├── demo.ipynb               # Model inference and demo
├── process_dataset.ipynb    # Dataset preprocessing
├── Resnet.ipynb             # ResNet50 training and evaluation
├── Data_non-process/        # Raw dataset
├── dataprocessed/           # Processed dataset
├── dataset/                 # Final dataset split into train/val/test
├── model/                   # Saved model weights
├── Test/                    # Test scripts and datasets
└── README.md                # Project documentation (this file)
```

## Notebooks

1. **aug.ipynb**: Demonstrates data augmentation techniques.
2. **CNN.ipynb**: Implements a custom CNN architecture for classification.
3. **demo.ipynb**: Provides a demo for model inference on sample images.
4. **process_dataset.ipynb**: Preprocesses the raw dataset into a structured format.
5. **Resnet.ipynb**: Trains and evaluates a ResNet50 model using transfer learning.

## Dataset

The dataset is organized into the following structure:

```
Data_non-process/
├── Acne type classification.v3i.folder/  # Raw dataset version 1
├── Acne.v18-acne-new.multiclass/         # Raw dataset version 2

Processed dataset:
├── dataprocessed/                        # Single-label processed dataset
├── dataset/                              # Train/val/test split
```

### Classes

- **dauden**: Blackheads
- **dautrang**: Whiteheads
- **mun**: Nodules, papules, pustules
- **seo**: Dark spots

## Models

1. **Custom CNN**:

   - Architecture: 5 convolutional blocks with batch normalization and dropout.
   - File: `cnn_best.pt`

2. **ResNet50**:
   - Pretrained on ImageNet, fine-tuned for 4 classes.
   - File: `resnet50_final.pt`

## Results

- **ResNet50**:
  - Achieved high accuracy on the validation and test sets.
- **Custom CNN**:
  - Lightweight model with competitive performance.

## How to Run

1. Install dependencies:
   ```bash
   pip install torch torchvision matplotlib pandas
   ```
2. Preprocess the dataset:
   - Run `process_dataset.ipynb` to organize the raw dataset.
3. Train the models:
   - Use `CNN.ipynb` for the custom CNN.
   - Use `Resnet.ipynb` for ResNet50.
4. Evaluate the models:
   - Use the test scripts in `Test/`.
5. Run the demo:
   - Open `demo.ipynb` and follow the instructions.

## Dependencies

- Python 3.8+
- PyTorch
- torchvision
- matplotlib
- pandas
- scikit-learn

## Acknowledgments

- Dataset: Acne dataset provided in `Data_non-process/`.
- Pretrained ResNet50: ImageNet weights from PyTorch.

## License

This project is licensed under the MIT License. See the LICENSE file for details.
