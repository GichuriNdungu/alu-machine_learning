# Applied Transfer Learning

## Problem Statement

As the world continues to heat up from the effects of Carbon emmissions, it is becoming more imperative to understand regions/areas that have higher emmissions. As a first step towards this objective, this project leverages Transfer learning to differentiate between rural and Urban locations. While this is barely a touch on the surface of this grave problem, this project serves as a practical introduction to transfer learning and computer vision, with the hopes of eventually building more complex models with even larger datasets
## Dataset

- *First Dataset* used in Training this model is  is sourced from Kaggle and can be accessed via this [link](https://www.kaggle.com/datasets/dansbecker/urban-and-rural-photos?resource=download).

        *Number of samples: 36
        *Number of features: 36
- *Second Dataset* used in Training and observing Fine tuning the model is sourced from kaggle and can be accessed via this [link](https://www.kaggle.com/datasets/tongpython/cat-and-dog).

        *Number of Samples: 8000
        *Number of features: 8000

## Evaluation Metrics
The performance of the models was assessed using the following metrics:

- **Accuracy**: The proportion of the total number of predictions that were correct.
- **Loss**: A measure of how well the model's prediction matches the observed data. Lower values are better.
- **Precision**: The proportion of positive predictions that were actually correct.
- **Recall**: The proportion of actual positives that were identified correctly.
- **F1 Score**: The harmonic mean of Precision and Recall. It provides a balance between the two metrics.

## Model Performance Before Fine-tuning

The table below shows the performance of the evaluated models based on the evaluation metrics:
*First Dataset*
| Model  | Accuracy | Loss | Precision | Recall | F1 Score |
|--------|----------|------|-----------|--------|----------|
| VGG    |    95    | 0.01 |    40%    | 40%    |    40%   |
| ResNet |    95    | 0.01 |    45%    | 45%    |    45%   |
| MobNet |    85    | 0.1  |    55%    | 55%    |    55%   |

*Second Dataset*

| Model  | Accuracy | Loss | Precision | Recall | F1 Score |
|--------|----------|------|-----------|--------|----------|
| VGG    |    95    | 0.01 |    40%    | 40%    |    40%   |
| ResNet |    95    | 0.01 |    45%    | 45%    |    45%   |
| MobNet |    85    | 0.1  |    55%    | 55%    |    55%   |

## Findings

- From the results herein, it is clear that our models were overfitting quite severely. 
- Despite recording high accuracies, the precision scores were as low as 40% for the resnet model, with the mobnet model demonstrating a slightly higher precision of 55% which is still significantly lower.
- It is therefore imperative that the models are fine-tuned for better generalisations.
- Upon further investigations, it is clear that the model's sample data is quite limited which likely affects the perfomance of the models.
- Therefore, to better see/understand the effects of finetuning a model during transfer learning, the second dataset was applied at this stage and trained over 2 iterations. 
- fine tuning was also performed using this dataset

## Fine Tuning Process 
To perform fine tuning, the following steps were followed:

1. **Unfreeze the base model layers**: The first step in fine-tuning is to unfreeze the layers of the pre-trained model that you want to fine-tune. This means that the weights of these layers will be updated during training.

2. **Reduce the steps per epoch**: Due to the large size of the dataset, the number of steps per epoch is halved to reduce training time. This means that each epoch will only process half of the batches in the training data.

3. **Compile the model**: After unfreezing the layers, the model needs to be compiled again. The optimizer used is RMSprop with a very low learning rate. The low learning rate is used because we are fine-tuning pre-trained weights, and we don't want to make large updates to these weights. The loss function used is Binary Crossentropy, which is suitable for binary classification tasks. The metric used to evaluate the model is Binary Accuracy.

4. **Train the model**: Finally, the model is trained for a number of epochs (2) on the training data, and the performance is evaluated on the validation data.

This process is repeated for each of the three models. The reasoning behind this process is to adapt the pre-trained models to the specific task by fine-tuning the weights of the base model.

## Fine-Tuning Results

The fine-tuning process was performed on three pre-trained models: `new_model`, `v16_model` (VGG16), and `mobile_net` (MobileNet). Each model was trained for two epochs. The results are summarized in the table below:

| Model      | Epoch | Training Loss | Training Accuracy | Validation Loss | Validation Accuracy |
|------------|-------|---------------|-------------------|-----------------|---------------------|
| Resnet     | 1     | 0.1820        | 93.67%            | 0.2384          | 94.65%              |
| Resnet     | 2     | 0.0858        | 96.85%            | 0.1351          | 96.45%              |
| v16_model  | 1     | 1.1071        | 53.59%            | 0.6901          | 53.20%              |
| v16_model  | 2     | 0.7631        | 53.43%            | 0.7773          | 50.10%              |
| mobile_net | 1     | 0.3280        | 87.25%            | 1.3508          | 66.45%              |
| mobile_net | 2     | 0.1858        | 92.73%            | 0.7944          | 78.40%              |

It is thus clear that the `Resnet model` performed the best, with the highest accuracy and lowest loss. `v16_model` performed the worst, with low accuracy and high loss. `mobile_net` performed reasonably well but not as well as `Resnet`.