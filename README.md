# SLT (Sign Language Translator)
A sign language translator which uses a multi-layer perceptron and a convolutional neural network to translate a sign image into the corresponding letter. 

## Summary
This study looks into Multilayer Perceptrons (MLP), and, in a second step, Convolutional Neural Networks (CNN) for sorting images using the Sign Language MNIST dataset from Kaggle. 

The model will be used to classify each of these images into a letter category. 

Designing the model without the use of ML libraries, the aim was to experiment how its design, activations functions, regularization, and number of layers and units affect its accuracy. 

In our models, we varied the numbers of hidden layers from 0, 1, and 2 and the numbers of hidden units from 32, 64, 128 and 256. Ultimately, the following was found regarding our model: 
- It performed better with more hidden units;
- 2 hidden layers seemed to work best; 
- Using ReLU or Leaky ReLU activation functions performed better than Sigmoid activation;
- Adding L2 regularization made the MLP models less accurate as it slowed down backpropagation (slower convergence and negative impacts on accuracy);
- Importantly, using a Convolutional Neural Network (CNN) model, designed with three convolutional layers and two fully connected layers, greatly improved the generalizability to new unseen images by reaching almost 90% validation accuracy; therefore concluding that CNNs are better than MLPs for classifying images. This finding aligns with the literature (see Xu, C., 2022).
- The analysis of accuracy as a function of training epochs revealed that MLP models tend to overfit with an increasing number of epochs with its validation accuracy capped at around 70%, while CNN had a better performance.

## Dataset
The dataset used is the Sign Language MNIST dataset from Kaggle, a collection of images of hand signs representing the 24 letters of the ASL alphabet excluding J and Z which involve hand motion. It contains a total of 34,627 images of hand signs. The training dataset has  27,455 images, each labeled that map one-to-one to the alphabet. The separate testing dataset has 7,172 images. Each image is a single 28x28 pixel image with grayscale values between 0-255.

<div style="width: 100%;">
  <img width="300" alt="Screenshot 2024-12-25 at 10 19 29 PM" src="https://github.com/user-attachments/assets/e74e9fe6-e4ca-4d00-b3ec-2cac3d9c2dc0" />
  <img width="300" alt="Screenshot 2024-12-25 at 10 23 20 PM" src="https://github.com/user-attachments/assets/a1ee8f42-d1de-4338-8ad5-62a7b5da4886" />

</div>

## Preprocessing
The images were vectorized by turning them into a flattened array of pixels with the labels dropped from the train dataset.
Both the training and test datasets were normalized. 
The integer labels were converted to binary form in a label dataframe of size 24 consisting of single values 0 or 1 from labels 0 to 24 for each individual picture, as 9 = J and 25 = Z are ignored. 
The general class distribution, as seen below, was balanced between sign letters, ensuring no class will be overrepresented. 

<img width="338" alt="Screenshot 2024-12-29 at 9 06 55 AM" src="https://github.com/user-attachments/assets/8c70bbf9-144e-48fd-945a-572a8d6c8892" />


## Model design 
To implement the Multi-Layer Perceptron for our task, we defined it as a Python class with back-propagation and mini-batch Gradient Descent (and SGD).

Note that this model is a multi-label classifier whose output layer will be 24 nodes. 

#### Simple MLP (no hidden layers)

To check the basic functionalities and do some first hyperparameter tuning, we look at the accuracy of our model, with no hidden layers, under different circumstances.

We explored various weight initialization methods, namely: Xavier/Gorot (’XG’), random (’RAND’), He et al. (’HE’), and a hybrid of ’XG’ and ’HE’ (see below). 

<img width="304" alt="Screenshot 2024-12-29 at 9 06 38 AM" src="https://github.com/user-attachments/assets/c3b1db2e-2d69-4240-bde9-dfe182580e9d" />


Random initialization had highest accuracy, as expected for one of the most common methods (Narkhede et al., 2022). ’HE’, more appropriate for ReLU layers (Arpit et al., 2019), starts lower but matches ’RAND’ after ∼ 40 iterations.

Examining learning rates (see below, on the left) and lambdas (regularization parameter, on the right), we found $λ = 0.001$ as the optimal default. For the learning rate, the Robbins-Monro schedule, which dynamically updates the learning rate, enhanced accuracy; as expected, as this method usually improves convergence (Bach, 2016). With these optimal parameters, the model’s test set accuracy reached 70%, which is within the expected range for the simplest version (no hidden layers) of our model.

<img width="607" alt="Screenshot 2024-12-29 at 9 06 26 AM" src="https://github.com/user-attachments/assets/2d0ab57f-6fb2-41d2-a24a-ac4427f6d500" />

#### Incorporating hidden layers
Starting with a baseline model without hidden layers, we followed it with models with 1 and 2 hidden layers with ReLU activations. Experimenting with different numbers of hidden units, we found that a higher count of hidden units, as expected, generally enhances performance (see Figures 6 and 7). Comparing the accuracy of the three models (for the ones with hidden layers, we used 256 units for this comparison), validated this hypothesis: as shown in Figure 5, models with fewer hidden layers have lower accuracy in predicting new data (up to 15% when comparing no hidden layers with 2 hidden ReLU ones).

<img width="265" alt="Screenshot 2024-12-29 at 9 06 11 AM" src="https://github.com/user-attachments/assets/082f27fb-6dae-4c42-b448-cd2e1899bd34" />

<img width="611" alt="Screenshot 2024-12-29 at 9 06 19 AM" src="https://github.com/user-attachments/assets/12602246-b923-443a-94e0-98ef8ed291a3" />

#### Experimenting with activation function
Comparing Sigmoid and Leaky ReLU, we found the model with Leaky ReLU activation fonction has slightly better accuracies than the one with ReLU. 

The model with Sigmoid activation fonction has worst accuracies accross the board compared to the one with ReLU; see below. 

<img width="296" alt="Screenshot 2024-12-29 at 9 17 19 AM" src="https://github.com/user-attachments/assets/1196e599-b437-4726-b5df-d73e06dbf36f" />

The MLP with Sigmoid activation function is worst than the ReLU and Leaky ReLU MLPs, which is expected as ReLU is generally accepted as being better for neural networks. This is because the gradients of the Sigmoid function can become very small for larger values far from 0 that are either positive or negative, as the slope of the Sigmoid function gets smaller. The gradients may thus become too small for effective learning. The problem is worse in networks with multiple layers such as this one with two hidden layers, where gradients from the output layer have to propagate back to previous layers. ReLU and Leaky ReLU solve this issue by having the gradients large and stable even far from 0. 

The model with Leaky ReLU activation fonction may have slightly better accuracies than the one with ReLU because when inputs to a ReLU neuron are negative it leads to an output of 0. If a neuron always outputs 0, it stops contributing to the learning since the gradient through it will be 0. This potentially limits the model’s capacity to learn some complicated patterns. Leaky ReLU solves this by having a small gradient when the neuron is negative, allowing the network to learn more patterns during training. This could explain the better accuracy, as the network is less likely to ignore subtle parts of the data.

#### Experimenting with L2 regularization
Adding L2 regularization with a tuned regularization parameter λ to MLP slowed down the convergence rate of backpropagation, and 50 training epochs were insufficient to achieve satisfactory train and test accuracy. With L2 regularization, the train accuracy of MLP with Leaky ReLU activation decreases by 7.9%, and the test accuracy decreases by 28.6%. Similarly, the train accuracy of MLP with Sigmoid activation decreases by 54.3%, and the test accuracy decreases by 36.0%. These results are summarized in the table below. The regularization parameter λ for MLP is determined iteratively to approach the maximal test accuracy with a precision of 0.00005.
<img width="610" alt="Screenshot 2024-12-29 at 9 18 29 AM" src="https://github.com/user-attachments/assets/d0a8d6e0-9ab9-4609-89de-a9775f0b8fd5" />

Changing the output activation from Leaky ReLU to Softmax or Sigmoid to Softmax yields a training accuracy of approximately 97% and a test accuracy of around 68%, comparable to the MLP models without L2 regularization after 50 epochs. This highlights the impact of the output activation choice on the performance of backpropagation. Sigmoid seems like a bad choice of activation in the hidden layers.

#### Experimenting with a CNN
A convolutional neural network (CNN) with 3 convolutional layers and 2 fully connected layers was constructed using PyTorch’s nn.Module with ReLU as the default activation. Preliminary experiments indicated that the CNN’s performance plateaued after 4 training epochs, leading to the decision to fix the number of epochs at 4 for computational efficiency. To maintain efficiency, the stride length and padding were set to 1 pixel. The gradient of the cross entropy loss was minimized using PyTorch’s Adam optimizer with learning rate lr=0.001. The batch size was fixed to 64. The kernel size and number of kernels per convolutional layer were first tuned for fixed numbers of units per hidden layer. For all num hidden units ∈ 32, 64, 128, 256, using 32 kernels of size 5×5 produced the highest test accuracy. Then, the number of units per hidden layer was varied while keeping num kernels = 32 and kernel size = 5 fixed. Table 2 below compares the accuracy of CNN and MLP under these variations.

<img width="604" alt="Screenshot 2024-12-29 at 9 19 39 AM" src="https://github.com/user-attachments/assets/c7201080-58ee-41ab-a063-735fd85bdc60" />





## References
1. Xu, C. (2022). Applying MLP and CNN on Handwriting Images for Image Classification Task. 2022 5th International Conference on Advanced Electronic Materials, Computers and Software Engineering (AEMCSE), 830-835. [https://doi.org/10.1109/AEMCSE55572.2022.00167].
2. Sign Language MNIST dataset from Kaggle: [https://www.kaggle.com/datasets/datamunge/sign-language- mnist/data].
3. Narkhede, M.V., Bartakke, P.P. Sutaone, M.S. A review on weight initialization strategies for neural networks. Artif Intell Rev 55, 291–322 (2022). [https://doi.org/10.1007/s10462-021-10033-z]
4. Arpit, Devansh and Yoshua Bengio. “The Benefits of Over-parameterization at Initialization in Deep ReLU Networks.” ArXiv abs/1901.03611 (2019): n. pag.
5. Francis Bach’s lecture on the Motivation behind the Robbins-Monro schedule, available at: [https://www.di.ens.fr/ fbach/orsay2016/lecture3.pdf]
