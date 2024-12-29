# SLT (Sign Language Translator)
A sign language translator which uses a multi-layer perceptron (compared with a convolutional neural network) to translate a sign image into the corresponding letter. 

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

#### Adding hidden layers
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

Overall, the best CNN has these parameters: {num hidden units = 128, kernel size = 5, num kernels = 32}. After training and tuning, it reaches a final validation accuracy of 89.4%, on par with its test accuracy in table 2. This confirms that our tuning didn’t overfit the model to the test data. The CNNs generally performed 10% better on the test set than MLPs, needing only 4 epochs compared to 50. Also, as seen in table 2, when we increase the number of hidden units from 128 to 256, the CNN’s accuracy drops, as it starts to overfit on the training set. Both MLP and CNN performed best with 128 hidden units per layer.

#### Optimal MLP Architecture
In order to come up with an optimal MLP architecture and to confirm whether L2 regularization improves our MLP implementation, we increased the number of training epochs to 200 to account for the slower convergence of backpropagation when regularization is applied. Table 3 summarizes the accuracy for different activation functions and for the optimal number of units per hidden layer (after tuning).

<img width="632" alt="Screenshot 2024-12-29 at 10 35 25 AM" src="https://github.com/user-attachments/assets/00c7eaca-083c-4676-bd1d-7fd93b84e73e" />

When using Leaky ReLU with no regularization, the optimal model is a degenerate MLP, namely a Multinomial Logistic regression. Tanh activation was tested only with L2 regularization to compare it with ReLU and confirm that the reason Leaky ReLU performed 14% worse than ReLU is related to Leaky ReLU itself, rather than randomness. In fact, Tanh and ReLU exhibit similar validation accuracy, indicating similar generalizability after tuning their hyperparameters. All models in Table 3 slightly overfit the training data, with the effect being more pronounced with Leaky ReLU. Moreover, adding L2 regularization showed no clear trend of improvement in MLP’s performance, even after 200 epochs. In comparison, the optimal CNN model (num hidden units = 128, kernel size = 5, num kernels = 32, activ=ReLU) achieves 100.0% train accuracy, 89.8% test accuracy, and 89.4% validation accuracy after only 4 epochs. Even with improved architecture, our MLP model is more prone to overfitting than CNN. Indeed, the high similarity between test and validation accuracy for the CNN model confirms that it does not overfit; in fact, it fits perfectly.

#### Accuracy as a function of epochs

Finally, we studied how the number of training epochs impacted accuracy of both models.
- Figure 10 shows that within the first 25 epochs, the MLP model starts to overfit. Indeed, the training accuracy becomes approximately 30% greater than validation and test accuracy when the number of epochs is ≥ 25. At around 50 epochs, there is a marginal increase in the test accuracy,
but test and validation accuracy remains small compared to the training accuracy.
- Figure 9 shows that around 35 epochs, test and validation accuracy of CNN decreases by nearly 10%, meaning it starts to overfit.

<img width="604" alt="Screenshot 2024-12-29 at 10 36 24 AM" src="https://github.com/user-attachments/assets/ab5b1b51-56f1-4d76-9820-798a968318bc" />

As further evidence, the figure below shows another non-optimal CNN model. At around 50-100 epochs, the training and validation accuracy curves start to exhibit a notable gap, as can be seen between the green and red, purple and brown, and pink and gray curves.
<img width="591" alt="Screenshot 2024-12-29 at 10 36 38 AM" src="https://github.com/user-attachments/assets/9916f44a-ac8b-4fc4-a209-ceac718ffd2b" />

## Discussion

#### L2 Regularization in MLP
When adding L2 regularization to the MLP models in Experiment 3.3, backpropagation converges more slowly, as demonstrated by the continued improvement in accuracy beyond 50 epochs. The main reason is that the L2 weight penalty term introduced into the cost function slows down the loss gradient convergence during backward propagation. This penalty term opposes weight updates, resulting in slowed convergence. Consequently, the train and test accuracy remain low after 50 training epochs.

#### Convolutional Neural Network
The main takeaway from the CNN experiment is its consistent outperformance of MLPs, achieved with significantly fewer training epochs. This superiority can be attributed to CNNs’ interpretability in image classification tasks compared to MLPs. Notably, PyTorch’s CNN was likely more optimized than our MLP implementation. Furthermore, each filter in CNNs specializes in detecting specific patterns, resulting in feature maps that directly encode the geometric features of the image post convolutional layers. In contrast, the neurons in MLPs encode nonlinear behaviors that are not interpretable.

#### Optimal MLP Architecture
Without regularization, experiments revealed that Multinomial Logistic regression outperformed any MLP with at least one layer. This suggests that Leaky ReLU is not a preferred activation choice. When regularization was included, both Tanh activation and ReLU performed roughly the same and 10% better than Leaky ReLU. Our best guess for why ReLU outperforms Leaky ReLU is that ReLU is sparser than Leaky ReLU, which effectively segregates features more strongly based on their importance.

Moreover, despite saturating at extreme input values like Sigmoid, Tanh surprisingly performed on par with the commonly preferred ReLU. We attribute this to the steeper gradient around 0 in Tanh compared to Sigmoid, allowing backpropagation to converge faster. Thus, 200 epochs were largely sufficient for Tanh to catch up with ReLU. Therefore, it seems that the consistent under- performance of Sigmoid is due to a combination of saturating gradients and the slow convergence of its gradient descent.

In comparison to CNN, MLP was more prone to overfitting, required 50 times more epochs, and its validation accuracy capped around 70%, whereas CNN could attain roughly 90% validation accuracy, making it more generalizable. We believe that MLP’s performance could have been greatly increased by using more than 2 hidden layers, as CNN additionally benefited from 3 convolutional layers during which it could extract important pattern features. This would be an interesting avenue to explore in the future.

#### Accuracy as a function of epochs
Plotting the train and test accuracy of the tuned MLP and CNN models confirmed earlier obser- vations that CNN requires fewer training epochs than MLP, in addition to maintaining good test and validation scores. MLP, on the other hand, tends to overfit quickly, and test accuracy caps at roughly 70%.

## Statement of Contributions
Emma Kondrup implemented the MLP model, analysed the data, and performed hyperparameter tuning. Fadi Younes contributed to obtaining and pre-processing the data, as well as adding hidden layers. Sarah Ameur contributed to the tuning, experimenting with different activation functions, and comparing results with the CNN. Sarah, Emma and Fadi wrote every section of this report and reviewed the work of all team members.



## References
1. Xu, C. (2022). Applying MLP and CNN on Handwriting Images for Image Classification Task. 2022 5th International Conference on Advanced Electronic Materials, Computers and Software Engineering (AEMCSE), 830-835. [https://doi.org/10.1109/AEMCSE55572.2022.00167].
2. Sign Language MNIST dataset from Kaggle: [https://www.kaggle.com/datasets/datamunge/sign-language- mnist/data].
3. Narkhede, M.V., Bartakke, P.P. Sutaone, M.S. A review on weight initialization strategies for neural networks. Artif Intell Rev 55, 291–322 (2022). [https://doi.org/10.1007/s10462-021-10033-z]
4. Arpit, Devansh and Yoshua Bengio. “The Benefits of Over-parameterization at Initialization in Deep ReLU Networks.” ArXiv abs/1901.03611 (2019): n. pag.
5. Francis Bach’s lecture on the Motivation behind the Robbins-Monro schedule, available at: [https://www.di.ens.fr/ fbach/orsay2016/lecture3.pdf]
