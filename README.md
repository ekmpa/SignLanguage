# SLT (Sign Language Translator)
A sign language translator which uses a multi-layer perceptron and a convolutional neural network to translate a sign image into the corresponding letter. 

## Summary
This study looks into Multilayer Perceptrons (MLP) for sorting images using the Sign Language MNIST dataset from Kaggle. 

Designing the model without the use of ML libraries, the aim was to experiment how its design, activations functions, regularization, and number of layers and units affect its accuracy. 

Ultimately, the following was found: 
- The model performed best with at least 128 units per hidden layer; 
- Using ReLU or Leaky ReLU activation functions performed better than Sigmoid activation;
- Adding L2 regularization made the MLP models less accurate as it slowed down backpropagation;
- Importantly, using a Convolutional Neural Network (CNN) model greatly improved the generalizability to new unseen images by reaching almost 90% validation accuracy; therefore concluding that CNNs are better than MLPs for classifying images. This finding aligns with the literature.

## Dataset
The dataset used contains a total of 34,627 images of hand signs. The training dataset has  27,455 images, each labeled that map one-to-one to the alphabet. The separate testing dataset has 7,172 images. 

<div style="width: 100%;">
  <img width="300" alt="Screenshot 2024-12-25 at 10 19 29 PM" src="https://github.com/user-attachments/assets/e74e9fe6-e4ca-4d00-b3ec-2cac3d9c2dc0" />
  <img width="300" alt="Screenshot 2024-12-25 at 10 23 20 PM" src="https://github.com/user-attachments/assets/a1ee8f42-d1de-4338-8ad5-62a7b5da4886" />

</div>


