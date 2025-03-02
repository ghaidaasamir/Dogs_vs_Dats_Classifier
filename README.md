# Dog vs Cat Classifier with VGG16 Transfer Learning

Image classification model that distinguishes between dogs and cats using transfer learning with the pre-trained VGG16 network.

## Approach

   - Use a pre-trained VGG16 model as a feature extractor.
   - Add custom fully connected layers with dropout and batch normalization on top.
   - Train the model first by freezing the VGG16 base, then fine-tune by unfreezing some of its layers.

## Dataset path
- [here](https://drive.google.com/drive/folders/1LoFcJ8UMjR8Zdi-jy5mf0PKO59Qjolvi?usp=drive_link)

## weights path
- [dog_cat_classifier_tuned_vgg16.keras](https://drive.google.com/file/d/1AHPmjsa7KHfPj36GoI-mCxzzjJ8sdeMO/view?usp=drive_link)
- [dog_cat_classifier_vgg16.keras](https://drive.google.com/file/d/1y3UDX9a4sinINh1ZOesY691wYMx0DXXi/view?usp=drive_link)

## Requirements

- **Python 3.6+**
- **TensorFlow 2**  
- **Pandas**
- **NumPy**
- **Matplotlib**
- **scikit-learn**

## Notebook 

1. **Dataset:**  
   - The data is split into training and validation sets.
   - Image augmentation and preprocessing.

2. **Visualization:**  
   - Sample images are deprocessed and visualized.

3. **Model Creation:**  
   - The VGG16 model is loaded with pre-trained weights from ImageNet.
   - A custom classifier is built on top with global average pooling, dense layers, batch normalization, dropout, and a final sigmoid activation for binary classification.

4. **Training and Fine-tuning:**  
   - **Initial Training:**  
     The custom top layers are trained while the VGG16 base remains frozen.
   - **Fine-tuning:**  
     The first 15 layers frozen and the model is retrained with a lower learning rate.

5. **Evaluation and Prediction:**  
   - The model is evaluated on the validation set.
   - Predictions are made on the test dataset.
