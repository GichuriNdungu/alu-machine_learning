# Project Title

## Problem Statement

As the world continues to heat up from the effects of Carbon emmissions, it is becoming more imperative to understand regions/areas that have higher emmissions. As a first step towards this objective, this project leverages Transfer learning to differentiate between rural and Urban locations. While this is barely a touch on the surface of this grave problem, this project serves as a practical introduction to transfer learning and computer vision, with the hopes of eventually building more complex models with even larger datasets
## Dataset

- *First Dataset* used in Training this model is  is sourced from Kaggle and can be accessed via this [link](https://www.kaggle.com/datasets/dansbecker/urban-and-rural-photos?resource=download).

- Number of samples: 36
- Number of features: 36
- *Second Dataset* used in Fine tuning the model is sourced from kaggle and can be accessed via this [link](https://www.kaggle.com/datasets/tongpython/cat-and-dog)

## Evaluation Metrics
The performance of the models was assessed using the following metrics:

- **Accuracy**: The proportion of the total number of predictions that were correct.
- **Loss**: A measure of how well the model's prediction matches the observed data. Lower values are better.
- **Precision**: The proportion of positive predictions that were actually correct.
- **Recall**: The proportion of actual positives that were identified correctly.
- **F1 Score**: The harmonic mean of Precision and Recall. It provides a balance between the two metrics.

## Model Performance Before Fine-tuning

The table below shows the performance of the evaluated models based on the evaluation metrics:

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
- Therefore, to better see/understand the effects of finetuning a model during transfer learning, the second dataset was applied at this stage and trained over five iterations. 
- fine tuning was also performed using this dataset

## Model Perfomances After Fine-tuning 

