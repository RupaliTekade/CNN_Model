CNN Image Classification on CIFAR-10

Dataset Used 
I used the CIFAR-10 dataset, which contains 60,000 color images (32×32×3) across 10 classes including airplane, automobile, bird, cat, deer, dog, frog, horse, ship, and truck.
* 50,000 training images
* 10,000 test images
* All images were normalized to the range 0–1

Model Summary (Layer Details)
I built a custom Convolutional Neural Network (CNN) from scratch using TensorFlow/Keras.

CNN Architecture:
1. Conv2D (32 filters, 3×3, ReLU)
2. MaxPooling2D (2×2)
3. Dropout (0.2)

4. Conv2D (64 filters, 3×3, ReLU)
5. MaxPooling2D (2×2)
6. Dropout (0.3)

7. Conv2D (128 filters, 3×3, ReLU)
8. Dropout (0.4)

9. Flatten
10. Dense (128 units, ReLU)
11. Dropout (0.5)
12. Dense (10 units, Softmax)

Training Settings 
* Optimizer: Adam
* Loss: Sparse Categorical Crossentropy
* Metrics: Accuracy
* Epochs: 20
* Batch Size: 64
* Validation Split: 20%


Observations & Final Accuracy

-> Initial Model (Without Dropout)
* Training accuracy increased too fast (~92%)
* Validation accuracy stuck at 70–72%
* Validation loss increased → overfitting confirmed

-> Final Model (With Dropout)
* Training Accuracy: ~71%
* Validation Accuracy: ~73–75%
* Validation loss stable around 0.75
* Gap between training and validation curves reduced
* Overfitting significantly decreased


-> Test Accuracy
73–75% on unseen test images.

These results are consistent for a from-scratch CNN on CIFAR-10.




Improvements / Tuning Tried

Dropout Regularization
Dropout layers were added after each convolution block and before the dense layer to prevent overfitting.




Epoch Tuning
Trained for 20 epochs once validation loss stabilized.




Prediction Visualization
Plotted examples of:

* Correctly classified images
* Incorrectly classified images
  This helped analyze the model’s strengths and weaknesses.





Visualization (Graphs & Images)

accuracy.png – Training vs Validation Accuracy

loss.png – Training vs Validation Loss

correct.png – Correct predictions

incorrect.png – Incorrect predictions






Future Enhancements

* Add data augmentation to improve generalization.

* Use Batch Normalization for more stable and faster training.

* Try deeper CNN layers for richer feature extraction.

* Apply learning rate scheduling or early stopping to optimize training.

* Perform hyperparameter tuning (filters, dropout rate, batch size).






Conclusion

This project demonstrates how to build a CNN from scratch, diagnose overfitting, apply dropout to improve generalization, and evaluate model performance using accuracy, loss curves, classification report, confusion matrix, and prediction visualization.
The final model performs well and generalizes effectively on the CIFAR-10 dataset.

  
