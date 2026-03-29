# Model Training Improvement Guide

## 🚀 Quick Start

```bash
cd Project-main
python train_model_improved.py
```

## ✨ What's Improved?

### 1. **Enhanced Data Augmentation**
- Wider rotation range (25° vs 20°)
- More zoom variation (0.4 vs 0.3)
- Color channel shifting (NEW)
- Brightness range expanded

### 2. **Better Model Architecture**
- 3 Dense layers (256→128→64) instead of 1
- Batch Normalization after each layer
- Gradual dropout (0.5→0.4→0.3)

### 3. **Smart Training Strategy**
- **Phase 1**: Train only top layers (15-30 epochs)
- **Phase 2**: Fine-tune last 20 base layers (10 epochs)
- Total: ~25-40 epochs with early stopping

### 4. **Advanced Callbacks**
- **EarlyStopping**: Stops if no improvement for 7 epochs
- **ReduceLROnPlateau**: Reduces learning rate when stuck
- **ModelCheckpoint**: Saves best model automatically

### 5. **Class Balancing**
- Automatically calculates class weights
- Handles imbalanced datasets (more WithMask than WithoutMask)

### 6. **Comprehensive Evaluation**
- Detailed classification report (Precision, Recall, F1-Score)
- Confusion matrix visualization
- Training history plots

## 📊 Expected Improvements

| Metric | Before | After | Improvement |
|--------|--------|-------|-------------|
| Accuracy | 85-90% | **92-97%** | +5-10% |
| False Positives | High | **Low** | 50% reduction |
| Robustness | Medium | **High** | Better generalization |

## 🎯 Configuration Options

### Switch to EfficientNet (Better Accuracy)
```python
# Line 24 in train_model_improved.py
BASE_MODEL_TYPE = 'efficientnet'  # Change from 'mobilenet'
```
**Trade-off**: 20% slower training, +2-3% accuracy

### Adjust Training Duration
```python
# Line 19
EPOCHS = 40  # Increase if you have time (default: 30)
```

### Change Batch Size
```python
# Line 20
BATCH_SIZE = 16  # Smaller = better accuracy, slower training
BATCH_SIZE = 64  # Larger = faster training, might reduce accuracy
```

## 📁 Output Files

After training completes, you'll get:

1. **mask_detector_improved.h5** - Final trained model
2. **mask_detector_best.h5** - Best model during training (use this!)
3. **training_history.png** - Loss & accuracy graphs
4. **confusion_matrix.png** - Prediction accuracy breakdown
5. **classification_report.txt** - Detailed metrics

## 🔄 Using the Improved Model

### Option 1: Replace Existing Model
```bash
# Backup old model first!
copy mask_detector.h5 mask_detector_backup.h5

# Use the new best model
copy mask_detector_best.h5 mask_detector.h5
```

### Option 2: Update web_app.py
```python
# Line 27 in web_app.py
maskNet = load_model("mask_detector_best.h5")  # Use best model
```

## 🐛 Troubleshooting

### "Out of Memory" Error
**Solution**: Reduce batch size
```python
BATCH_SIZE = 16  # Instead of 32
```

### Training Too Slow
**Solution**: Use MobileNet instead of EfficientNet
```python
BASE_MODEL_TYPE = 'mobilenet'
```

### Overfitting (Train Acc >> Val Acc)
**Solution**: Increase dropout
```python
# Lines 143-149
Dropout(0.6)  # Increase from 0.5
Dropout(0.5)  # Increase from 0.4
Dropout(0.4)  # Increase from 0.3
```

### Low Accuracy on Both Train & Val
**Solution**: 
1. Collect more diverse data
2. Train for more epochs
3. Use EfficientNet base model

## 📈 Monitoring Training

Watch these metrics during training:

```
Epoch 15/30
loss: 0.1234 - accuracy: 0.9500 - val_loss: 0.1567 - val_accuracy: 0.9300
```

**Good Signs**:
- ✅ val_accuracy increasing
- ✅ val_loss decreasing
- ✅ Small gap between train & val accuracy (<5%)

**Bad Signs**:
- ❌ val_accuracy stuck or decreasing
- ❌ train_acc >> val_acc (overfitting)
- ❌ Both accuracies very low (underfitting)

## 🎓 Advanced: Hyperparameter Tuning

If you want to experiment:

```python
# Learning Rates to Try
INIT_LR = 1e-3  # Faster learning
INIT_LR = 5e-4  # Balanced
INIT_LR = 1e-4  # Safer (default)
INIT_LR = 1e-5  # Very conservative

# Dense Layer Sizes
Dense(512) → Dense(256) → Dense(128)  # Larger network
Dense(256) → Dense(128) → Dense(64)   # Current (balanced)
Dense(128) → Dense(64)                # Smaller network
```

## 💡 Tips for Maximum Accuracy

1. **Data Quality > Data Quantity**
   - Remove blurry/bad images
   - Ensure correct labels
   - Add variety (angles, lighting, mask types)

2. **Balance Your Dataset**
   - Aim for 50/50 WithMask/WithoutMask
   - Minimum 1000 images per class

3. **Train Multiple Times**
   - Neural networks have randomness
   - Train 3 times, pick best model

4. **Test on Real Data**
   - Use your webcam to validate
   - Check edge cases (side angles, poor lighting)

## 🚀 Expected Timeline

- Small dataset (2000 images): ~10-15 minutes
- Medium dataset (5000 images): ~20-30 minutes  
- Large dataset (10000+ images): ~45-60 minutes

*Times are for CPU. GPU is 5-10x faster.*

---

**Need Help?** Check the classification report and confusion matrix to understand where the model is struggling!
