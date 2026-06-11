# Vietnamese CNN Text Classification in PyTorch

This repository contains an implementation of Kim's [Convolutional Neural Networks for Sentence Classification](https://arxiv.org/abs/1408.5882) in PyTorch, customized and optimized for Vietnamese Text Classification.

## 🚀 Updates & Improvements
- **Vietnamese Support**: Integrated `underthesea` for accurate Vietnamese word tokenization.
- **Modernized PyTorch**: Refactored the legacy PyTorch code (removing deprecated `.data[0]`, `Variable`, `size_average` etc.) to run natively on modern PyTorch versions.
- **Stable Training Pipeline**: Fixed dimension mismatch bugs (batch sizes) and improved the `torchtext` dataset loading mechanism (pinned to `torchtext==0.6.0` for stability).
- **Custom Dataset Loading**: Supports loading custom `.csv` datasets seamlessly.

## 📦 Requirements
To install the required dependencies, run:
```bash
pip install -r requirements.txt
```

## 📊 Dataset & Results
The model was trained on a Vietnamese classification dataset containing 5 distinct classes: `emotional`, `recruitment`, `sales`, `stu_edu`, `news`, `label_name`.

**Training Results:**
- **Best Validation Accuracy**: **~93.25%** (achieved around Epoch 9)
- **Parameters**: ~6.8M trainable parameters.
- **Training Time**: ~33 seconds per epoch.

## 💻 Usage

### 1. Training
To train the model from scratch, simply run:
```bash
python main.py -epochs 10 -test-interval 60 -save-interval 60
```
This will train the model for 10 epochs, test validation accuracy every 60 steps, and save the best checkpoints in the `snapshot/` directory.

### 2. Prediction
To run inference on a custom Vietnamese sentence:
```bash
python main.py -predict "Điện thoại này dùng rất mượt và chụp ảnh đẹp" -snapshot "./snapshot/2026-06-11_10-54-44/best_steps_540.pt"
```

## 📂 Project Structure
- `clean_data.py`: Script to preprocess and remove duplicates from the raw CSV data.
- `mydatasets.py`: Custom `torchtext` TabularDataset loader.
- `model.py`: The 1D CNN model architecture.
- `train.py`: Training, evaluation, and prediction loops.
- `main.py`: Entry point for training and CLI arguments.
- `check.py`: Sanity check utility for Tensor dimensions.

## 📚 Reference
* [Convolutional Neural Networks for Sentence Classification](https://arxiv.org/abs/1408.5882)
