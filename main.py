#Emma Kondrup, Sarah Ameur and Fadi Younes

import pandas as pd
import numpy as np
%matplotlib inline
import matplotlib.pyplot as plt
import keras
from sklearn.utils import shuffle
from tqdm import tqdm
from sklearn.preprocessing import LabelBinarizer

#**Task 1: Acquire the data**

## Importing train and test datasets

!wget --no-check-certificate 'https://drive.google.com/uc?export=download&id=1fsV5NEUmOOR6RrYVHI4m7_LwOBsjB_Sy' -O sign_mnist_train.csv
!wget --no-check-certificate 'https://drive.google.com/uc?export=download&id=1qgDI_D9sabYT07K35ORoYU3qVGsKpNvs' -O sign_mnist_test.csv


train_df = pd.read_csv('sign_mnist_train.csv')
test_df = pd.read_csv('sign_mnist_test.csv')

## Train dataset exploration

train_df.info()

train_df.describe()

test_df.head(6)

## Train dataset vectorization and normalization

train_label = train_df['label']
trainset = train_df.drop(['label'],axis=1)
trainset.head()

X_train = trainset.values

lb = LabelBinarizer()
y_train = lb.fit_transform(train_label)

label_count_train = np.sum(y_train == 1, axis=0)
label_count_train = np.insert(label_count_train, 9, 0)
label_count_train = np.insert(label_count_train, 25, 0)
plt.figure(figsize=(10, 6))
plt.bar(range(0, len(label_count_train)), label_count_train)
plt.xlabel('Label')
plt.ylabel('Count of labels')
plt.xticks(range(0, len(label_count_train)), range(0, len(label_count_train)))
plt.title("Frequency of each label in train dataset")
plt.show()


fig, axe = plt.subplots(2, 2)
fig.suptitle('Preview of Train dataset images before normalization')
axe[0, 0].imshow(X_train[0].reshape(28, 28), cmap='gray')
axe[0, 0].set_title('label: 3  letter: D')
axe[0, 1].imshow(X_train[1].reshape(28, 28), cmap='gray')
axe[0, 1].set_title('label: 6  letter: G')
axe[1, 0].imshow(X_train[2].reshape(28, 28), cmap='gray')
axe[1, 0].set_title('label: 2  letter: C')
axe[1, 1].imshow(X_train[4].reshape(28, 28), cmap='gray')
axe[1, 1].set_title('label: 13  letter: N')
fig.tight_layout()
plt.subplots_adjust(wspace=0, hspace=0.4)
plt.show()

X_train = X_train.astype(float)
X_train_mean = np.mean(X_train, axis = 0)
X_train -= X_train_mean
X_train_std = np.std(X_train, axis = 0)
X_train /= X_train_std

fig, axe = plt.subplots(2, 2)
fig.suptitle('Preview of Train dataset images after normalization')
axe[0, 0].imshow(X_train[0].reshape(28, 28), cmap='gray')
axe[0, 0].set_title('label: 3  letter: D')
axe[0, 1].imshow(X_train[1].reshape(28, 28), cmap='gray')
axe[0, 1].set_title('label: 6  letter: G')
axe[1, 0].imshow(X_train[2].reshape(28, 28), cmap='gray')
axe[1, 0].set_title('label: 2  letter: C')
axe[1, 1].imshow(X_train[4].reshape(28, 28), cmap='gray')
axe[1, 1].set_title('label: 13  letter: N')
fig.tight_layout()
plt.subplots_adjust(wspace=0, hspace=0.4)
plt.show()

## Test dataset exploration

test_df.info()

test_df.describe()

test_df.head(6)

## Test dataset vectorization and normalization

test_label = test_df['label']
X_test = test_df.drop(['label'],axis=1)
print(X_test.shape)
X_test.head()


X_test = X_test.values

y_test = lb.fit_transform(test_label)

label_count_test = np.sum(y_test == 1, axis=0)
label_count_test = np.insert(label_count_test, 9, 0)
label_count_test = np.insert(label_count_test, 25, 0)
plt.figure(figsize=(10, 6))
plt.bar(range(0, len(label_count_test)), label_count_test)
plt.xlabel('Label')
plt.ylabel('Count of labels')
plt.xticks(range(0, len(label_count_test)), range(0, len(label_count_test)))
plt.title("Frequency of each label in test dataset")
plt.show()

fig, axe = plt.subplots(2, 2)
fig.suptitle('Preview of Test dataset images before normalization')
axe[0, 0].imshow(X_test[0].reshape(28, 28), cmap='gray')
axe[0, 0].set_title('label: 6  letter: G')
axe[0, 1].imshow(X_test[1].reshape(28, 28), cmap='gray')
axe[0, 1].set_title('label: 5  letter: F')
axe[1, 0].imshow(X_test[2].reshape(28, 28), cmap='gray')
axe[1, 0].set_title('label: 10  letter: K')
axe[1, 1].imshow(X_test[4].reshape(28, 28), cmap='gray')
axe[1, 1].set_title('label: 3  letter: D')
fig.tight_layout()
plt.subplots_adjust(wspace=0, hspace=0.4)
plt.show()

X_test = X_test.astype(float)
X_test_mean = np.mean(X_test, axis = 0)
X_test -= X_test_mean
X_test_std = np.std(X_test, axis = 0)
X_test /= X_test_std

fig, axe = plt.subplots(2, 2)
fig.suptitle('Preview of Test dataset images after normalization')
axe[0, 0].imshow(X_test[0].reshape(28, 28), cmap='gray')
axe[0, 0].set_title('label: 6  letter: G')
axe[0, 1].imshow(X_test[1].reshape(28, 28), cmap='gray')
axe[0, 1].set_title('label: 5  letter: F')
axe[1, 0].imshow(X_test[2].reshape(28, 28), cmap='gray')
axe[1, 0].set_title('label: 10  letter: K')
axe[1, 1].imshow(X_test[4].reshape(28, 28), cmap='gray')
axe[1, 1].set_title('label: 3  letter: D')
fig.tight_layout()
plt.subplots_adjust(wspace=0, hspace=0.4)
plt.show()

# **Task 2: Implement an MLP to classify image data**

reLU_activ = lambda z: np.maximum(0,z)
reLU_deriv = lambda z: (z > 0).astype(float)


sigmoid_activ = lambda z: 1./ (1 + np.exp(-z))
sigmoid_deriv = lambda z: sigmoid_activ(z) * (1 - sigmoid_activ(z))


tanh_activ = lambda z: (np.exp(z) - np.exp(-z)) / (np.exp(z) + np.exp(-z))
tanh_deriv = lambda z: 1 - np.power(tanh_activ(z), 2)


#TMP softmax_activ = lambda z: (np.exp(z-np.max(z))) / (np.sum(np.exp(z-np.max(z)), axis=1, keepdims=True))
#TRY:
import scipy.special
softmax_activ = lambda x: scipy.special.softmax(x)

softmax_deriv = lambda z: softmax_activ(z)*softmax_activ(1-z)
#TRY:
#softmax_deriv = lambda x: np.diag(scipy.special.softmax(x)) - np.outer(scipy.special.softmax(x), scipy.special.softmax(x))


def softmax(Z_data):
    exp_Z_data = np.exp(Z_data)
    #print("exp(Z_data) = ",exp_Z_data)

    sum_of_exp_Z_data = np.sum(exp_Z_data)
    #print("sum of exponentials = ", sum_of_exp_Z_data)
    prob_dist = [exp_Zi/sum_of_exp_Z_data for exp_Zi in exp_Z_data]

    return np.array(prob_dist, dtype=float)

softmax_activ = lambda z: (np.exp(z-np.max(z))) / (np.sum(np.exp(z-np.max(z)), axis=1, keepdims=True)) # sum ?
softmax_deriv = lambda z: softmax_activ(z)*softmax_activ(1-z)

def delta_fn(y_true, y_pred, deriv):
    return (y_pred - y_true) * deriv(y_pred)

class MLP:

  def __init__(self,in_out_sizes, activ=reLU_activ, hidden_layers=0,units=[], lambd=.0001, init_mode='HE'):
    self.activ = activ

    if activ is reLU_activ:
      self.activ_deriv = reLU_deriv
    elif activ is sigmoid_activ:
      self.activ_deriv = sigmoid_deriv
    elif activ is tanh_activ:
      self.activ_deriv = tanh_deriv
    elif activ is softmax_activ:
      self.activ_deriv = softmax_deriv
    elif activ is leaky_reLu_activ:
      self.activ_deriv = leaky_reLu_deriv

    self.num_layers = hidden_layers + 2
    self.units = units

    self.lambd = lambd
    self.layers = None # init in forward pass
    self.lr = None # init in fit

    self.weights = {}
    self.biases = {}

    layer_sizes = [in_out_sizes[0]] + units + [in_out_sizes[1]]

    for i in range(len(layer_sizes) - 1):

      if init_mode == 'XG':
        self.weights[i+1] = np.random.randn(layer_sizes[i], layer_sizes[i + 1]) * np.sqrt(1 / (layer_sizes[i] + layer_sizes[i+1])) # Xavier/Glorot init
      elif init_mode == 'RAND':
        self.weights[i+1] = np.random.randn(layer_sizes[i], layer_sizes[i + 1]) * 0.01 # Small random numbers
      elif init_mode == 'HE':
        self.weights[i+1] = np.random.randn(layer_sizes[i], layer_sizes[i + 1]) * np.sqrt(2 / layer_sizes[i]) # He et al. init
      elif init_mode == 'hybrid':
        self.weights[i + 1] = np.random.randn(layer_sizes[i], layer_sizes[i + 1]) / np.sqrt(layer_sizes[i])

      self.biases[i+1] = np.zeros(layer_sizes[i + 1])


  def forward_propagation(self,X):
    z = {}
    input_layer = X
    layers = {1:input_layer}

    for i in range(1, self.num_layers) :
      z[i+1] = np.dot(layers[i], self.weights[i]) + self.biases[i]

      if (i+1 == self.num_layers):
        layers[i+1] = softmax_activ(z[i+1]) #sigmoid_activ(z[i+1]) ?
        continue
      layers[i+1] = self.activ(z[i+1])

    self.layers = layers

    return z


  def back_propagation(self, y_true) :

      delta = delta_fn(y_true, self.layers[self.num_layers], softmax_deriv)
      self.weights_gradient = {}
      self.biases_gradient = {}
      to_update = {}

      for i in range(self.num_layers, 1, -1):
        if i != self.num_layers:
          delta = np.dot(delta, self.weights[i].T) * self.activ_deriv(self.layers[i])
        dw = np.dot(self.layers[i-1].T, delta)
        self.weights_gradient[i-1] = dw
        self.biases_gradient[i-1] = delta
        to_update[i-1] = (dw, delta)

      for i in reversed(range(1, self.num_layers)):
        v = to_update[i]
        # L2 reg for weights:
        self.weights[i] -= self.lr * (v[0] + (self.lambd) * self.weights[i])
        self.biases[i] -= self.lr * np.mean(v[1],0)



  def fit(self, X, y, X_t = None, y_t = None, n_iter=200, num_batches=250, lr = None ):

    if lr is None:
      robbins = True
      self.lr_i = 50 # for Robbins - Monro schedule, modified for better performance
    else:
      robbins = False
      self.lr = lr
    batch_size = len(X) // num_batches
    train_score, test_score = [], []

    for iteration in range(n_iter):

      if robbins:
        self.lr_i += 10 # for Robbins - Monro schedule, modified for better performance
        self.lr = self.lr_i ** (-.51) # Robbins - Monro schedule

      seed = np.arange(len(X))
      np.random.shuffle(seed)
      x_shuffled, y_shuffled = X[seed], y[seed]

      for j in range(num_batches):
        self.forward_propagation( x_shuffled[ (j * batch_size) : ((j+1)*batch_size) ] )
        self.back_propagation( y_shuffled[ (j * batch_size) : ((j+1)*batch_size) ])

      train_score += [np.mean(self.predict(X) == np.argmax(y,axis=1))]

      if X_t is not None:
        test_score += [np.mean(self.predict(X_t) == np.argmax(y_t,axis=1))]

    if X_t is not None:
      print(f"After {iteration + 1} epochs, train_score is {round(train_score[-1]*100,1)}% and test_score is {round(test_score[-1]*100,1)}%")
    else:
      print(f"After {iteration + 1} epochs, train_score is {round(train_score[-1]*100,1)}%")

    return train_score, test_score

  def predict(self, x):
    self.forward_propagation(x)
    return np.argmax(self.layers[self.num_layers],axis=1)

  def evaluate_acc(self, y_true, y_pred):
    return np.mean(np.argmax(y_true,axis=1) == y_pred)

  def loss(self, y_true, y_pred):
    epsilon = 1e-8
    y_pred = np.clip(y_pred, epsilon, 1. - epsilon)
    loss = -np.mean(y_true * np.log(y_pred + epsilon))
    return loss




plt.style.use('ggplot')


def gradient_check(model, X, y_true, epsilon=1e-8):

    model.forward_propagation(X)
    model.back_propagation(y_true)

    init_weights = model.weights

    numerical_gradient_w = np.zeros_like(model.weights[1])

    for i in range(model.weights[1].shape[0]):
        for j in range(model.weights[1].shape[1]):

                # upwards
                model.weights[1][i, j] += epsilon
                model.forward_propagation(X)
                model.back_propagation(y_true)
                loss_plus_epsilon = model.loss(model.layers[model.num_layers], y_true)

                #model.weights = init_weights # Reset

                # downwards
                model.weights[1][i, j] -= 2*epsilon
                model.forward_propagation(X)
                model.back_propagation(y_true)
                loss_minus_epsilon = model.loss(model.layers[model.num_layers], y_true)

                numerical_gradient_w[i, j] = (loss_plus_epsilon - loss_minus_epsilon) / (2 * epsilon)

                model.weights = init_weights # Reset

    # compare numerical and analytical gradients
    analytical_gradient_w = model.weights_gradient[1]

    rat = numerical_gradient_w[0][0] / analytical_gradient_w[0][0]

    #analytical_gradient_w *= rat

    err = np.abs(numerical_gradient_w - analytical_gradient_w)
    relative_error = err / (np.abs(numerical_gradient_w) + np.abs(analytical_gradient_w) + epsilon)

    diff = np.linalg.norm(analytical_gradient_w - numerical_gradient_w) / (np.linalg.norm(analytical_gradient_w) + np.linalg.norm(numerical_gradient_w) + epsilon)

    print(f"Gradient check for weights {1}:")
    #print("\n Numerical Gradient:\n", numerical_gradient_w)
    #print("\n Analytical Gradient:\n", analytical_gradient_w)
            #print("Relative Error:\n", relative_error)
            #print(f"with Err mean : {np.mean(err)}")
    print("\n Diff:\n", diff)
    print(f"\n min diff: {np.min(diff)}")
    print(f"\n shape diff: {diff.shape}")



mlp_nh = MLP(in_out_sizes=[784,24]) # 784 features, 24 labels
nh_train, nh_test = mlp_nh.fit(X_train, y_train, X_test, y_test, n_iter = 20, lr = .01)


gradient_check(mlp_nh, X_train[:][:20], y_train[:][:20])

# **Task 3: Running the experiments**


## Task 3.1: basic MLP models

### 1) With no hidden layers:

mlp_nh = MLP(in_out_sizes=[784,24]) # 784 features, 24 labels

nh_train, nh_test = mlp_nh.fit(X_train, y_train, X_test, y_test, n_iter = 100, lr = .01)

# After 50 epochs, train_score is 93.4% and test_score is 68.6%

plt.plot(nh_train, label = f"Train score = {round(nh_train[-1]*100,1)}%")
plt.plot(nh_test, label = f"Test score = {round(nh_test[-1]*100,1)}%")
plt.legend(loc = 'best')
plt.ylabel('Accuracy')
plt.xlabel('Iterations')
plt.title("No hidden layers accuracy")
plt.grid(True)
plt.show()
# with X/G init: 93.4 & 68.6

pred = mlp_nh.predict(X_test)
mlp_nh.evaluate_acc(y_test, pred)
# with X/G init: 0.686

# Testing different weight initialization methods:

inits = ['XG', 'RAND', 'HE', 'hybrid']
for mode in inits:
  mlp_nh = MLP(in_out_sizes=[784,24], init_mode=mode)
  tr, te = mlp_nh.fit(X_train, y_train, X_test, y_test, n_iter = 100, lr = 0.01)
  plt.plot(te, label=f"Testing with mode : {mode}")

plt.legend(loc = 'best')
plt.ylabel('Accuracy')
plt.xlabel('Iterations')
plt.title('No hidden layer, different init methods')
plt.grid(True)
plt.show()

# best lambda is smallest
lambdas = [1., 0.1, 0.01, .001, .0001, 0]
for lambda_val in lambdas:
  mlp_nh = MLP(in_out_sizes=[784,24], lambd=lambda_val)
  tr, te = mlp_nh.fit(X_train, y_train, X_test, y_test, n_iter = 50, lr = 0.01)
  plt.plot(te, label=f"Testing with lambda : {lambda_val}")

plt.legend(loc = 'best')
plt.ylabel('Accuracy')
plt.xlabel('Iterations')
plt.title('No hidden layer, different lambdas')
plt.grid(True)
plt.show()

# With X/G init:
# After 50 epochs, train_score is 36.9% and test_score is 33.0%
# After 50 epochs, train_score is 55.7% and test_score is 47.7%
# After 50 epochs, train_score is 79.6% and test_score is 64.5%
# After 50 epochs, train_score is 92.0% and test_score is 67.2%
# After 50 epochs, train_score is 92.7% and test_score is 66.8%
# After 50 epochs, train_score is 93.1% and test_score is 68.5%

# indicate best lr is .01, after that, decreases.
# Robbin-Monro schedule has higher accuracy
lrs = [1., 0.1, 0.01, .001, 0]
for lr in lrs:
  mlp_nh = MLP(in_out_sizes=[784,24]) #, lambd=0.001)
  tr, te = mlp_nh.fit(X_train, y_train, X_test, y_test, n_iter = 50, lr = lr)
  plt.plot(te, label=f"Testing with lr : {lr}")

mlp_nh = MLP(in_out_sizes=[784,24], lambd=0.001)
tr, te = mlp_nh.fit(X_train, y_train, X_test, y_test, n_iter = 50)
plt.plot(te, label=f"Testing with Robbins-Monro schedule  : {lr}")

plt.legend(loc = 'best')
plt.ylabel('Accuracy')
plt.xlabel('Iterations')
plt.title("No hidden layer, different lr")
plt.grid(True)
plt.show()

# With X/G init:
# After 50 epochs, train_score is 93.7% and test_score is 68.7%
# After 50 epochs, train_score is 96.8% and test_score is 69.3%
# After 50 epochs, train_score is 92.0% and test_score is 68.3%
# After 50 epochs, train_score is 66.5% and test_score is 55.2%
# After 50 epochs, train_score is 3.7% and test_score is 3.5%
# After 50 epochs, train_score is 96.9% and test_score is 68.7%

# best is lower
batches = [1, 50, 250, 1000]
for num in batches:
  mlp_nh = MLP(in_out_sizes=[784,24], lambd=0.001)
  tr, te = mlp_nh.fit(X_train, y_train, X_test, y_test, n_iter = 50, num_batches=num, lr=.01)
  plt.plot(te, label=f"Testing with {num} batches")

plt.legend(loc = 'best')
plt.ylabel('Accuracy')
plt.xlabel('Iterations')
plt.title("No hidden layer, different num batches")
plt.grid(True)
plt.show()

# With X/G init
# After 50 epochs, train_score is 93.4% and test_score is 68.5%
# After 50 epochs, train_score is 93.0% and test_score is 68.3%
# After 50 epochs, train_score is 91.6% and test_score is 68.1%
# After 50 epochs, train_score is 85.6% and test_score is 66.7%

# Best is highest
niters = [50, 250, 1000]
accs = []
for niter in niters:
  mlp_nh = MLP(in_out_sizes=[784,24], lambd=0.001)
  tr, te = mlp_nh.fit(X_train, y_train, X_test, y_test, n_iter = niter, lr=.01)
  accs.append(te[-1])

plt.bar(niters, accs)
plt.legend(loc = 'best')
plt.ylabel('Accuracy')
plt.xlabel('Iterations')
plt.title("No hidden layer, different `n_iter` values ")
plt.grid(True)
plt.show()

#After 50 epochs, train_score is 92.7% and test_score is 67.6%
#After 250 epochs, train_score is 99.9% and test_score is 69.0%

### 2) Adding a reLU layer

units = [32, 64, 128, 256]
for unit in units:
  mlp_oh = MLP(in_out_sizes=[784,24], activ=reLU_activ, hidden_layers=1,units=[unit])
  tr, te = mlp_oh.fit(X_train, y_train, X_test, y_test, n_iter = 50)
  plt.plot(te, label=f"Testing with {unit} hidden units / layer")

plt.legend(loc = 'best')
plt.ylabel('Accuracy')
plt.xlabel('Iterations')
plt.title("One hidden layer, different # of hidden units")
plt.grid(True)
plt.show()

#After 50 epochs, train_score is 100.0% and test_score is 72.5%
#After 50 epochs, train_score is 100.0% and test_score is 78.2%
#After 50 epochs, train_score is 100.0% and test_score is 79.0%
#After 50 epochs, train_score is 100.0% and test_score is 79.6%

mlp_oh = MLP(in_out_sizes=[784,24], activ=reLU_activ, hidden_layers=1,units=[64])

oh_tr, oh_te = mlp_oh.fit(X_train, y_train, X_test, y_test, n_iter = 50, lr = .1) # or lr = None for R-M

plt.plot(oh_tr, label = f"Train score = {round(oh_tr[-1]*100,1)}%")
plt.plot(oh_te, label = f"Test score = {round(oh_te[-1]*100,1)}%")
plt.legend(loc = 'best')
plt.ylabel('Accuracy')
plt.xlabel('Iterations')
plt.title("One hidden ReLU layer, accuracy v. iteration")
plt.grid(True)
plt.show()
# 100 and 76.8

### 3) Two hidden ReLU layers

units = [32, 64, 128, 256]
for unit in units:
  mlp_th = MLP(in_out_sizes=[784,24], activ=reLU_activ, hidden_layers=2,units=[unit,unit])
  tr, te = mlp_th.fit(X_train, y_train, X_test, y_test, n_iter = 50)
  plt.plot(te, label=f"Testing with {unit} hidden units / layer")

plt.legend(loc = 'best')
plt.ylabel('Accuracy')
plt.xlabel('Iterations')
plt.title("Two hidden layers, different # of hidden units")
plt.grid(True)
plt.show()

# After 50 epochs, train_score is 100.0% and test_score is 73.6%
# After 50 epochs, train_score is 100.0% and test_score is 77.0%
# After 50 epochs, train_score is 100.0% and test_score is 78.5%


mlp_th = MLP(in_out_sizes=[784,24], activ=reLU_activ, hidden_layers=2,units=[256,256])

th_tr, th_te = mlp_th.fit(X_train, y_train, X_test, y_test, n_iter = 50)

# After 50 epochs, train_score is 100.0% and test_score is 79.8%

plt.plot(th_tr, label = f"Train score = {round(th_tr[-1]*100,1)}%")
plt.plot(th_te, label = f"Test score = {round(th_te[-1]*100,1)}%")
plt.legend(loc = 'best')
plt.ylabel('Accuracy')
plt.xlabel('Iterations')
plt.title("Two hidden ReLU layers, accuracy v. iteration")
plt.grid(True)
plt.show()


### Comparison

mlp_nh = MLP(in_out_sizes=[784,24])
nh_train, nh_test = mlp_nh.fit(X_train, y_train, X_test, y_test, n_iter = 50, lr = .01)


# After 50 epochs, train_score is 92.6% and test_score is 68.2%

plt.plot(nh_test, label = f"No hidden layers {round(nh_test[-1]*100,1)}%")
plt.plot(oh_te, label = f"One hidden ReLU layer = {round(oh_te[-1]*100,1)}%")
plt.plot(th_te, label = f"Two hidden ReLU layers = {round(th_te[-1]*100,1)}%")
plt.legend(loc = 'best')
plt.ylabel('Accuracy')
plt.xlabel('Iterations')
plt.title("Different models, accuracy v. iteration")
plt.grid(True)
plt.show()

## Task 3.2: sigmoid and Leaky-ReLU activation functions for MLP models

#Leaky-ReLu activation function, sigmoid was already defined
leaky_reLu_activ = lambda z, alpha=0.01: z * (z > 0) + alpha * z * (z <= 0)
leaky_reLu_deriv = lambda z, alpha=0.01: (z > 0).astype(float) + alpha * (z <= 0)

units = [32, 64, 128, 256]
for unit in units:
  mlp_th = MLP(in_out_sizes=[784,24], activ=leaky_reLu_activ, hidden_layers=2,units=[unit,unit])
  tr, te = mlp_th.fit(X_train, y_train, X_test, y_test, n_iter = 50)
  plt.plot(te, label=f"Testing with {unit} hidden units / layer")

plt.legend(loc = 'best')
plt.ylabel('Accuracy')
plt.xlabel('Iterations')
plt.title("Two hidden layers, leaky ReLu activation function, different # of hidden units")
plt.grid(True)
plt.show()


units = [32, 64, 128, 256]
for unit in units:
  mlp_th_2 = MLP(in_out_sizes=[784,24], activ=sigmoid_activ, hidden_layers=2,units=[unit,unit])
  tr_2, te_2 = mlp_th_2.fit(X_train, y_train, X_test, y_test, n_iter = 50)
  plt.plot(te_2, label=f"Testing with {unit} hidden units / layer")

plt.legend(loc = 'best')
plt.ylabel('Accuracy')
plt.xlabel('Iterations')
plt.title("Two hidden layers, sigmoid activation function, different # of hidden units")
plt.grid(True)
plt.show()

mlp_th_leak = MLP(in_out_sizes=[784,24], activ=leaky_reLu_activ, hidden_layers=2,units=[256,256])
tr_leak, te_leak = mlp_th_leak.fit(X_train, y_train, X_test, y_test, n_iter = 50)
plt.plot(te_leak, label=f"leaky ReLu")


mlp_th_re = MLP(in_out_sizes=[784,24], activ=reLU_activ, hidden_layers=2,units=[256,256])
tr_re, te_re = mlp_th_re.fit(X_train, y_train, X_test, y_test, n_iter = 50)
plt.plot(te_re, label=f"ReLu")


mlp_th_sig = MLP(in_out_sizes=[784,24], activ=sigmoid_activ, hidden_layers=2,units=[256,256])
tr_sig, te_sig = mlp_th_sig.fit(X_train, y_train, X_test, y_test, n_iter = 50)
plt.plot(te_sig, label=f"sigmoid")

plt.legend(loc = 'best')
plt.ylabel('Accuracy')
plt.xlabel('Iterations')
plt.title("Two hidden layers, 256 hidden units, different activation function")
plt.grid(True)
plt.show()


## Task 3.3: adding L2 regularization to MLP

### MLP model, L2 regularization + leaky reLU activation:

class MLP2(MLP):
  def forward_propagation(self,X):
    z = {}
    input_layer = X
    layers = {1:input_layer}

    for i in range(1, self.num_layers) :
      z[i+1] = np.dot(layers[i], self.weights[i]) + self.biases[i]

      if (i+1 == self.num_layers):
        layers[i+1] = leaky_reLu_activ(z[i+1])
        continue
      layers[i+1] = self.activ(z[i+1])

    self.layers = layers

    return z

    def loss(self, y_true, y_pred):
      epsilon = 1e-12
      y_pred = np.clip(y_pred, epsilon, 1. - epsilon)
      N = y_true.shape[0]
      cross_entropy_loss = -np.sum(y_true * np.log(y_pred + 1e-9)) / N

      # regularization term
      reg_term = 0
      for w in self.weights.values():
        reg_term += np.sum(np.square(w))
      reg_term *= self.lambd

      return cross_entropy_loss + reg_term



# initialize MLP model with L2 regularization + leaky reLU activation

mlp2 = MLP2(in_out_sizes=[784, 24], activ=leaky_reLu_activ, hidden_layers=2, units=[32,32])

# train and evaluate MLP model with L2 regularization + leaky reLU activation, 50 epochs

l2_leakyrelu_train_score, l2_leakyrelu_test_score = mlp2.fit(X_train, y_train, X_test, y_test, n_iter=50, lr=0.01)

# After 50 epochs, train_score is 92.1% and test_score is 45.0%

plt.plot(l2_leakyrelu_train_score, label = f"Train score = {round(l2_leakyrelu_train_score[-1]*100,1)}%")
plt.plot(l2_leakyrelu_test_score, label = f"Test score = {round(l2_leakyrelu_test_score[-1]*100,1)}%")
plt.legend(loc = 'best')
plt.ylabel('Accuracy')
plt.xlabel('Iterations')
plt.title("Accuracy v. Iteration (two hidden leaky reLU layers and L2 regularization)")
plt.grid(True)
plt.show()


### Tuning lambda, L2 regularization + leaky reLU activation:


lambdas = [0.00085, 0.0009, 0.00095, 0.001] #, 0.002, 0.005] #, 0.01, 1.0]
train_scores = []
test_scores = []

for lambd in lambdas:
    # instantiate
    mlp2 = MLP2(in_out_sizes=[784, 24], activ=leaky_reLu_activ, hidden_layers=2, units=[32, 32], lambd=lambd)
    # fit
    train_score, test_score = mlp2.fit(X_train, y_train, X_test, y_test, n_iter=50, lr=0.01)

    train_scores.append(train_score)
    test_scores.append(test_score)

best_lambda_index = np.argmax([score[-1] for score in test_scores])
best_lambda = lambdas[best_lambda_index]
best_test_score = test_scores[best_lambda_index][-1]

print(f"Best lambda = {best_lambda}")
print(f", yields test accuracy = {best_test_score * 100:.2f}%")


###MLP model, L2 regularization + sigmoid activation:

class MLP3(MLP):
  def forward_propagation(self,X):
    z = {}
    input_layer = X
    layers = {1:input_layer}

    for i in range(1, self.num_layers) :
      z[i+1] = np.dot(layers[i], self.weights[i]) + self.biases[i]

      if (i+1 == self.num_layers):
        layers[i+1] = sigmoid_activ(z[i+1])
        continue
      layers[i+1] = self.activ(z[i+1])

    self.layers = layers

    return z


    # TRY
    def loss(self, y_true, y_pred):
      epsilon = 1e-12
      y_pred = np.clip(y_pred, epsilon, 1. - epsilon)
      N = y_true.shape[0]
      cross_entropy_loss = -np.sum(y_true * np.log(y_pred + 1e-9)) / N

      # Regularization term
      reg_term = 0
      for w in self.weights.values():
        reg_term += np.sum(np.square(w))
      reg_term *= self.lambd

      return cross_entropy_loss + reg_term



# initialize MLP model with L2 regularization + sigmoid activation

mlp3 = MLP3(in_out_sizes=[784, 24], activ=sigmoid_activ, hidden_layers=2, units=[64,64])

# train and evaluate MLP model with L2 regularization + sigmoid activation, 50 epochs

l2_sigm_train_score, l2_sigm_test_score = mlp3.fit(X_train, y_train, X_test, y_test, n_iter=50, lr=0.01)

# After 50 epochs, train_score is 29.8% and test_score is 24.0%

plt.plot(l2_sigm_train_score, label = f"Train score = {round(l2_sigm_train_score[-1]*100,1)}%")
plt.plot(l2_sigm_test_score, label = f"Test score = {round(l2_sigm_test_score[-1]*100,1)}%")
plt.legend(loc = 'best')
plt.ylabel('Accuracy')
plt.xlabel('Iterations')
plt.title("Accuracy v. Iteration (two hidden leaky sigmoid layers and L2 regularization)")
plt.grid(True)
plt.show()


__units=[32,32]:__
train accuracy ~22.0%, test accuracy ~20.1%

__units=[64,64]:__
train accuracy ~26.7%, test accuracy ~23.5%

__units=[128,128]:__
train accuracy ~35.0%, test accuracy ~31.0%

__units=[256,256]:__
train accuracy ~45.7%, test accuracy ~40.9%

###Tuning lambda, L2 regularization + 2 hidden sigmoid layers:

lambdas = [0.0002, 0.00025, 0.0003, 0.00035] #, 0.001, 0.01, 0.1, 1.0]
train_scores = []
test_scores = []

for lambd in lambdas:
    # instantiate
    mlp3 = MLP3(in_out_sizes=[784, 24], activ=sigmoid_activ, hidden_layers=2, units=[32, 32], lambd=lambd)
    # fit
    train_score, test_score = mlp3.fit(X_train, y_train, X_test, y_test, n_iter=50, lr=0.01)

    train_scores.append(train_score)
    test_scores.append(test_score)

best_lambda_index = np.argmax([score[-1] for score in test_scores])
best_lambda = lambdas[best_lambda_index]
best_test_score = test_scores[best_lambda_index][-1]

print(f"best lambda = {best_lambda}")
print(f", yields test accuracy = {best_test_score * 100:.2f}%")

# After 50 epochs, train_score is 22.0% and test_score is 19.2%
#After 50 epochs, train_score is 19.4% and test_score is 17.6%

## Task 3.4: Convolutional Neural Network

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset


class CNN(nn.Module):
    def __init__(self, num_hidden_units, kernel_size=3, num_kernels=16, dim_after_reshape=28):
        # inherits from PyTorch's nn.Module
        super(CNN, self).__init__()

        self.num_kernels = num_kernels
        self.kernel_size = kernel_size
        self.dim_after_reshape = dim_after_reshape

        # 3 convolutional layers
        # input channels are learnable filters
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=num_kernels, kernel_size=kernel_size, stride=1, padding=1)
        self.conv2 = nn.Conv2d(in_channels=num_kernels, out_channels=num_kernels, kernel_size=kernel_size, stride=1, padding=1)
        self.conv3 = nn.Conv2d(in_channels=num_kernels, out_channels=num_kernels, kernel_size=kernel_size, stride=1, padding=1)

        # 2 fully connected layers
        # was: 28 x 28
        self.full1 = nn.Linear(num_kernels*dim_after_reshape*dim_after_reshape, num_hidden_units)
        self.full2 = nn.Linear(num_hidden_units, num_hidden_units)

        # output layer
        self.out = nn.Linear(num_hidden_units, 24)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))

        # sizes
        batch_size = x.size(0)

        num_kernels = x.size(1)
        feat_map_size = x.size(2)
        # reshape the feature map
        x = x.view(-1, self.num_kernels * self.dim_after_reshape * self.dim_after_reshape)

        x = F.relu(self.full1(x))
        x = F.relu(self.full2(x))
        x = self.out(x)

        return x

# convert X_train into PyTorch tensor of size (batch sixe, num_channels, h, w)
X_train_tensor = torch.tensor(X_train, dtype=torch.float32).reshape(-1, 1, 28, 28)
# convert y_train into PyTorch tensor
y_train_tensor = torch.tensor(y_train, dtype=torch.long)

# inherits from PyTorch's 'Dataset' class
class myDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


# instantiate
train_data = myDataset(X_train_tensor, y_train_tensor)

# batch size
batch_size = 64

# instantiate data loader
train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)

import matplotlib.pyplot as plt

num_hidden_units_list = [32, 64, 128, 256]

plt.figure(figsize=(10,6))

for nh in num_hidden_units_list:
  print(f'---------- {nh} hidden units: ---------------------')

  # instantiate a ConvNNet
  cnn = CNN(nh)

  # loss function
  criterion = nn.CrossEntropyLoss()
  # ConvNNet parameter optimization
  optimizer = optim.Adam(cnn.parameters(), lr=0.001)


  # training
  train_loss_epoch = []
  num_epochs = 10
  for epoch in range(num_epochs):
    cnn.train()
    running_loss = 0.0
    for inputs, labels in train_loader:
        optimizer.zero_grad()
        outputs = cnn(inputs)
        labels = labels.float()
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * inputs.size(0)
    epoch_train_loss = running_loss / len(train_data)
    train_loss_epoch.append(epoch_train_loss)
    print(f"epoch {epoch+1}: train loss = {epoch_train_loss:.5f}")

  plt.plot(range(1, num_epochs + 1), train_loss_epoch, label=f'{nh} hidden units')

plt.title('Train loss v. Epochs of different numbers of hidden units in the fully connected layers')
plt.xlabel('Epoch')
plt.ylabel('Train loss')
plt.legend()
plt.grid(True)
plt.show()

import itertools

kernel_sizes = [3, 5, 7]
num_kernels = [16, 32]


print('------------------ Train loss: -------------------')
for kernel_size, num_kernel in itertools.product(kernel_sizes, num_kernels):
    print(f"kernel size = {kernel_size}, number of kernels = {num_kernel}")

    # output size
    out_size = 28 if kernel_size == 3 else (22 if kernel_size == 5 else 16)
    #out_size = 28 + 2*1 - kernel_size + 1

    # best number of hidden units in the fully connected layers was 128
    cnn = CNN(128, kernel_size=kernel_size, num_kernels=num_kernel, dim_after_reshape=out_size)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(cnn.parameters(), lr=0.001)

    # training
    num_epochs = 4
    for epoch in range(num_epochs):
        cnn.train()
        running_loss = 0.0
        for inputs, labels in train_loader:
            optimizer.zero_grad()
            outputs = cnn(inputs)
            labels = labels.float()
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * inputs.size(0)
        epoch_loss = running_loss / len(train_data)
        print(f"epoch {epoch+1}: {epoch_loss:.4f}")


### Tuning the hyperparameters based on Test accuracy:

# convert X_train into PyTorch tensor of size (batch sixe, num_channels, h, w)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32).reshape(-1, 1, 28, 28)
# convert y_train into PyTorch tensor
y_test_tensor = torch.tensor(y_test, dtype=torch.long)

# inherits from PyTorch's 'Dataset' class
class myDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


# instantiate
test_data = myDataset(X_test_tensor, y_test_tensor)

# batch size
batch_size = 64

# instantiate data loader
test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=True)

hyperparams_list = [
    {"num_hidden_units": 32, "kernel_size": 5, "num_kernels": 32, "dim_after_reshape": 22},
    {"num_hidden_units": 64, "kernel_size": 5, "num_kernels": 32, "dim_after_reshape": 22},
    {"num_hidden_units": 128, "kernel_size": 5, "num_kernels": 32, "dim_after_reshape": 22},
    {"num_hidden_units": 256, "kernel_size": 5, "num_kernels": 32, "dim_after_reshape": 22}
]

for hyperparams in hyperparams_list:
    print("------------------------- Hyperparameter Tuning ---------------------------")
    print(f"Tuning: {hyperparams}")
    print("---------------------------------------------------------------------------")

    cnn = CNN(**hyperparams)

    criterion = nn.CrossEntropyLoss()

    optimizer = optim.Adam(cnn.parameters(), lr=0.001)

    # training
    print("------------------------- Train loss ---------------------------")
    num_epochs = 4
    for epoch in range(num_epochs):
        cnn.train()
        running_loss = 0.0
        correct_train = 0
        total_train = 0
        for inputs, labels in train_loader:
            optimizer.zero_grad()
            outputs = cnn(inputs)
            labels = labels.float()
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * inputs.size(0)

            # train accuracy
            _, predicted_train = torch.max(outputs, 1)
            total_train += labels.size(0)
            correct_train += (predicted_train == labels.argmax(1)).sum().item()

        epoch_loss = running_loss / len(train_data)
        epoch_accuracy_train = correct_train / total_train
        print(f"epoch {epoch+1}: Loss: {epoch_loss:.5f}, Train Accuracy: {epoch_accuracy_train:.2%}")

    # evaluate the trained CNN model on the test dataset
    cnn.eval()
    correct_test = 0
    total_test = 0
    with torch.no_grad():
        for inputs, labels in test_loader:
            outputs = cnn(inputs)
            _, predicted_test = torch.max(outputs, 1)
            total_test += labels.size(0)
            correct_test += (predicted_test == labels.argmax(1)).sum().item()
    accuracy_test = correct_test / total_test
    print(f"Accuracy on test set: {accuracy_test:.2%}")
    print("\n")


''' OUTPUT:
------------------------- Hyperparameter Tuning ---------------------------
Tuning: {'num_hidden_units': 32, 'kernel_size': 5, 'num_kernels': 32, 'dim_after_reshape': 22}
---------------------------------------------------------------------------
------------------------- Train loss ---------------------------
epoch 1: Loss: 0.55910, Train Accuracy: 83.33%
epoch 2: Loss: 0.01765, Train Accuracy: 99.53%
epoch 3: Loss: 0.00010, Train Accuracy: 100.00%
epoch 4: Loss: 0.00005, Train Accuracy: 100.00%
Accuracy on test set: 81.58%


------------------------- Hyperparameter Tuning ---------------------------
Tuning: {'num_hidden_units': 64, 'kernel_size': 5, 'num_kernels': 32, 'dim_after_reshape': 22}
---------------------------------------------------------------------------
------------------------- Train loss ---------------------------
epoch 1: Loss: 0.42779, Train Accuracy: 87.03%
epoch 2: Loss: 0.00915, Train Accuracy: 99.73%
epoch 3: Loss: 0.00007, Train Accuracy: 100.00%
epoch 4: Loss: 0.00003, Train Accuracy: 100.00%
Accuracy on test set: 87.94%


------------------------- Hyperparameter Tuning ---------------------------
Tuning: {'num_hidden_units': 128, 'kernel_size': 5, 'num_kernels': 32, 'dim_after_reshape': 22}
---------------------------------------------------------------------------
------------------------- Train loss ---------------------------
epoch 1: Loss: 0.30191, Train Accuracy: 90.87%
epoch 2: Loss: 0.01203, Train Accuracy: 99.68%
epoch 3: Loss: 0.00005, Train Accuracy: 100.00%
epoch 4: Loss: 0.00002, Train Accuracy: 100.00%
Accuracy on test set: 90.32%


------------------------- Hyperparameter Tuning ---------------------------
Tuning: {'num_hidden_units': 256, 'kernel_size': 5, 'num_kernels': 32, 'dim_after_reshape': 22}
---------------------------------------------------------------------------
------------------------- Train loss ---------------------------
epoch 1: Loss: 0.30390, Train Accuracy: 90.66%
epoch 2: Loss: 0.00938, Train Accuracy: 99.77%
epoch 3: Loss: 0.00002, Train Accuracy: 100.00%
epoch 4: Loss: 0.00001, Train Accuracy: 100.00%
Accuracy on test set: 90.09%

'''
### CNN model with optimal hyperparameters:


from sklearn.model_selection import train_test_split

X_test_va, X_test_te, y_test_va, y_test_te = train_test_split(X_test, y_test, test_size=0.5, random_state=42)

X_test_va_tensor = torch.tensor(X_test_va, dtype=torch.float32).reshape(-1, 1, 28, 28)
y_test_va_tensor = torch.tensor(y_test_va, dtype=torch.long)

validation_data = myDataset(X_test_va_tensor, y_test_va_tensor)
validation_loader = DataLoader(validation_data, batch_size=batch_size, shuffle=False)

X_test_te_tensor = torch.tensor(X_test_te, dtype=torch.float32).reshape(-1, 1, 28, 28)
y_test_te_tensor = torch.tensor(y_test_te, dtype=torch.long)

test_te_data = myDataset(X_test_te_tensor, y_test_te_tensor)
test_te_loader = DataLoader(test_te_data, batch_size=batch_size, shuffle=False)


best_hyperparams = {"num_hidden_units": 128, "kernel_size": 5, "num_kernels": 32, "dim_after_reshape": 22}
cnn_best = CNN(**best_hyperparams)

criterion = nn.CrossEntropyLoss()

optimizer = optim.Adam(cnn_best.parameters(), lr=0.001)

# training
print("------------------------- Train loss ---------------------------")
num_epochs = 4
for epoch in range(num_epochs):
    cnn_best.train()
    running_loss = 0.0
    for inputs, labels in train_loader:
        optimizer.zero_grad()
        outputs = cnn_best(inputs)
        labels = labels.float()
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * inputs.size(0)
    epoch_loss = running_loss / len(train_data)
    print(f"epoch {epoch+1}: {epoch_loss:.5f}")

# validation accuracy
cnn_best.eval()
correct = 0
total = 0
with torch.no_grad():
    for inputs, labels in validation_loader:
        outputs = cnn_best(inputs)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels.argmax(1)).sum().item()
accuracy = correct / total
print(f"validation accuracy = {accuracy:.2%}")

# validation accuracy = 89.40      #97.38%

## Task 3.5: MLP architecture optimization

X_train.shape

# (27455, 784) before splitting

y_train.shape

# (27455, 24) before splitting

### Architecture without L2 regularization:

# 16 min

import numpy as np
from sklearn.model_selection import train_test_split

hyperparams_list = [
    #{"hidden_layers": 0, "units": [], "activ": softmax_activ, "lambd": 0.0001},
    {"hidden_layers": 2, "units": [16,16], "activ": softmax_activ, "lambd": 0.0001},
    {"hidden_layers": 2, "units": [64, 64], "activ": softmax_activ, "lambd": 0.0001},
    {"hidden_layers": 2, "units": [128, 128], "activ": softmax_activ, "lambd": 0.0001},
]

best_accuracy = 0
best_model = None

input_size = X_train.shape[1]
#print("X_train.shape (before test-validation splitting) = ", X_train.shape)
output_size = len(np.unique(np.argmax(y_train, axis=1)))
#print("output_size = ", output_size)



X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)



for hyperparams in hyperparams_list:
    print("---------------------------------------------------------------------------")
    print(f"Tuning: {hyperparams}")

    mlp = MLP(in_out_sizes=[input_size, output_size], **hyperparams)

    print("current model: ", mlp)
    print("---------------------------------------------------------------------------")

    # training
    train_score, test_score = mlp.fit(X_train, y_train, X_val, y_val)

    # validation (usually "test")
    validation_accuracy = test_score[-1]

    # update best model based on validation accuracy
    if validation_accuracy > best_accuracy:
        best_accuracy = validation_accuracy
        best_model = mlp

# test (usually "validation")
test_accuracy = best_model.evaluate_acc(y_test, best_model.predict(X_test))
print("best model is: ", best_model)
print("test accuracy of the best MLP model =", test_accuracy)




## Architecture with L2 regularization:

# Observation: With reLU activation, 2 hidden layers, lr=0.01, the accuracies are highly sensitive to the regularization parameter lambda. For this reason, lambda was fixed to 0.0001 since prior experiments showed that it yields satisfactory accuracy scores.

import numpy as np
from sklearn.model_selection import train_test_split

# MLP4 inherits from MLP2 in order to include L2 regularization
class MLP4(MLP2):
    def __init__(self, in_out_sizes, activ=reLU_activ, hidden_layers=0, units=[], lambd=0.0001, lr=0.01):
        super().__init__(in_out_sizes, activ, hidden_layers, units, lambd)
        self.lr = lr


# 59 min

hyperparams_list = [
    {"hidden_layers": 2, "units": [16,16], "activ": tanh_activ, "lambd": 0.0001, "lr": 0.01},
    {"hidden_layers": 2, "units": [32, 32], "activ": tanh_activ, "lambd": 0.0001, "lr": 0.01},
    {"hidden_layers": 2, "units": [64, 64], "activ": tanh_activ, "lambd": 0.0001, "lr": 0.01},
    {"hidden_layers": 2, "units": [128, 128], "activ": tanh_activ, "lambd": 0.0001, "lr": 0.01},
    {"hidden_layers": 2, "units": [256, 256], "activ": tanh_activ, "lambd": 0.0001, "lr": 0.01},
]

best_accuracy = 0
best_model = None

input_size = X_train.shape[1]
output_size = len(np.unique(np.argmax(y_train, axis=1)))

X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=1)

for hyperparams in hyperparams_list:
    print("---------------------------------------------------------------------------")
    print(f"Tuning: {hyperparams}")

    mlp = MLP4(in_out_sizes=[input_size, output_size], **hyperparams)

    print("current model: ", mlp)
    print("---------------------------------------------------------------------------")

    # Training
    train_score, test_score = mlp.fit(X_train, y_train, X_val, y_val)

    # Validation (usually "test")
    validation_accuracy = test_score[-1]

    # Update best model based on validation accuracy
    if validation_accuracy > best_accuracy:
        best_accuracy = validation_accuracy
        best_model = mlp

# Test (usually "validation")
test_accuracy = best_model.evaluate_acc(y_test, best_model.predict(X_test))
print("best model: ", best_model)
print("Test accuracy of the best MLP model =", test_accuracy)


## Task 3.6: Plots of the testing and training accuracies of MLP and CNN as a function of epochs

### MLP model:

# 1hr

import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# for plotting
train_accuracies = []
val_accuracies = []
test_accuracies = []

epochs_list = [1, 3, 5, 10, 15, 20, 40, 50, 100, 150, 200]

# MLP model with optimal hyperparams
optimal_hyperparams = {"hidden_layers": 2, "units": [256,256], "activ": tanh_activ, "lambd": 0.0001, "lr": 0.01}
mlp = MLP4(in_out_sizes=[784, 24], **optimal_hyperparams)

# test-validation splitting (train -> validate -> test)
X_va, X_te, y_va, y_te = train_test_split(X_test, y_test, test_size=0.2, random_state=42)


for epochs in epochs_list:
    print("---------------------------------------------------------------------------")
    print(f"{epochs} epochs")

    # train and validate
    train_scores, val_scores = mlp.fit(X_train, y_train, X_va, y_va, n_iter=epochs)

    train_accuracies.append(train_scores[-1])
    val_accuracies.append(val_scores[-1])


    # test
    test_predictions = mlp.predict(X_te)
    test_accuracy = np.sum(test_predictions == np.argmax(y_te, axis=1))/len(test_predictions)
    print("validation accuracy (train -> test -> validate) = {:.2%}".format(test_accuracy))
    test_accuracies.append(test_accuracy)


# train and valid acc
plt.plot(epochs_list, train_accuracies, label="training")
plt.plot(epochs_list, val_accuracies, label="validation", linestyle='--')

# test acc
plt.plot(epochs_list, test_accuracies, label='test accuracy')


plt.title("Training and validation accuracy v. Number of epochs")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.legend()
plt.grid(True)
plt.show()


### CNN model:

# 3h...

X_test, X_val, y_test, y_val = train_test_split(X_test, y_test, test_size=0.5, random_state=1)

X_val_tensor = torch.tensor(X_val, dtype=torch.float32).reshape(-1, 1, 28, 28)
y_val_tensor = torch.tensor(y_val, dtype=torch.long)
val_data = myDataset(X_val_tensor, y_val_tensor)
val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False)

X_test_tensor = torch.tensor(X_test, dtype=torch.float32).reshape(-1, 1, 28, 28)
y_test_tensor = torch.tensor(y_test, dtype=torch.long)
test_data = myDataset(X_test_tensor, y_test_tensor)
test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)

params = {"num_hidden_units": 128, "kernel_size": 5, "num_kernels": 32, "dim_after_reshape": 22}

num_epochs_list = [0, 1, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50]
train_accuracies = []
val_accuracies = []
test_accuracies = []

# instantiate
cnn = CNN(**params)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(cnn.parameters(), lr=0.001)

for num_epochs in num_epochs_list:
    print(f'------------------{num_epochs} epochs------------------')

    # train
    for epoch in range(num_epochs):
        cnn.train()
        correct_train = 0
        total_train = 0
        for inputs, labels in train_loader:
            optimizer.zero_grad()
            outputs = cnn(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total_train += labels.size(0)

            correct_train += (predicted == labels.argmax(1)).sum().item()
            labels = labels.float()
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
        train_accuracy = 100 * correct_train / total_train

    # validation (to check overfit)
    cnn.eval()
    correct_val = 0
    total_val = 0
    with torch.no_grad():
        for inputs, labels in val_loader:
            outputs = cnn(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total_val += labels.size(0)
            correct_val += (predicted == labels.argmax(1)).sum().item()
    val_accuracy = 100 * correct_val / total_val

    # test
    correct_test = 0
    total_test = 0
    with torch.no_grad():
        for inputs, labels in test_loader:
            outputs = cnn(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total_test += labels.size(0)
            correct_test += (predicted == labels.argmax(1)).sum().item()
    test_accuracy = 100 * correct_test / total_test

    print(f"Validation accuracy after {num_epochs} epochs = {val_accuracy:.2f}%")
    print(f"Test accuracy after {num_epochs} epochs = {test_accuracy:.2f}%")

    train_accuracies.append(train_accuracy)
    val_accuracies.append(val_accuracy)
    test_accuracies.append(test_accuracy)

# Plot
plt.figure(figsize=(10, 6))
plt.plot(num_epochs_list, train_accuracies, label='training accuracy')
plt.plot(num_epochs_list, val_accuracies, label='validation accuracy')
plt.plot(num_epochs_list, test_accuracies, label='Testing accuracy', marker='s')
plt.title('Train, Validation, and Test accuracy vs. Number of epochs')
plt.xlabel('Number of epochs')
plt.ylabel('Accuracy (%)')
plt.legend()
plt.grid(True)
plt.show()



To know whether the model is overfit, look at decreasing accuracy of the test set. For CNN, we see from the plot that overfitting begins after 35-40 epochs.
