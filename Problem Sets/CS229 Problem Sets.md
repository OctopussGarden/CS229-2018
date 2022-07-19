# PS 1

## Problem 1 Linear Classifiers (logistic regression and GDA)

### b) Newton's Method

```python
import numpy as np
import util

from linear_model import LinearModel


def main(train_path, eval_path, pred_path):
    """Problem 1(b): Logistic regression with Newton's Method.

    Args:
        train_path: Path to CSV file containing dataset for training.
        eval_path: Path to CSV file containing dataset for evaluation.
        pred_path: Path to save predictions.
    """
    x_train, y_train = util.load_dataset(train_path, add_intercept=True)

    # *** START CODE HERE ***
    # Train Logistic Regression
    model = LogisticRegression(eps=1e-5)
    model.fit(x_train, y_train)

    # Plot data and decision boundary
    util.plot(x_train, y_train, model.theta, 'output/p01b.png'.format(pred_path[-5])) #TODO: Why -5?

    # Save Predictions
    x_eval, y_eval = util.load_dataset(eval_path, add_intercept=True)
    y_pred = model.predict(x_eval)
    np.savetxt(pred_path, y_pred > 0.5, fmt='%d')
    # *** END CODE HERE ***


class LogisticRegression(LinearModel):
    """Logistic regression with Newton's Method as the solver.

    Example usage:
        > clf = LogisticRegression()
        > clf.fit(x_train, y_train)
        > clf.predict(x_eval)
    """

    def fit(self, x, y):
        """Run Newton's Method to minimize J(theta) for logistic regression.

        Args:
            x: Training example inputs. Shape (m, n).
            y: Training example labels. Shape (m,).
        """
        # *** START CODE HERE ***
        # Initialization
        m, n = x.shape
        self.theta = np.zeros(n)
        # Newton's method
        while True:
            # save precious theta
            old_theta = np.copy(self.theta)
            # Sigmoid function
            h_x = 1 / (1 + np.exp(-x.dot(self.theta)))
            # Calculate Hessians
            H = (x.T * h_x * (1-h_x)).dot(x) / m
            gradient_J_theta = x.T.dot(h_x-y) / m

            # Update theta
            self.theta -= np.linalg.inv(H).dot(gradient_J_theta)
            if np.linalg.norm(self.theta - old_theta,ord=1) < self.eps:
                break
        # *** END CODE HERE ***

    def predict(self, x):
        """Make a prediction given new inputs x.

        Args:
            x: Inputs of shape (m, n).

        Returns:
            Outputs of shape (m,).
        """
        # *** START CODE HERE ***
        return 1 / (1 + np.exp(-x.dot(self.theta)))
        # *** END CODE HERE ***

```

<img src="https://gitee.com/violets/typora--images/raw/main/imgs/202203271928427.png" alt="p01b_1" style="zoom: 50%;" />

<img src="https://gitee.com/violets/typora--images/raw/main/imgs/202203271928860.png" alt="p01b_2" style="zoom:50%;" />

### e) GDA

```python
import numpy as np
import util

from linear_model import LinearModel


def main(train_path, eval_path, pred_path):
    """Problem 1(e): Gaussian discriminant analysis (GDA)

    Args:
        train_path: Path to CSV file containing dataset for training.
        eval_path: Path to CSV file containing dataset for evaluation.
        pred_path: Path to save predictions.
    """
    # Load dataset
    x_train, y_train = util.load_dataset(train_path, add_intercept=False)

    # *** START CODE HERE ***
    # Train Logistic Regression
    model = GDA()
    model.fit(x_train, y_train)

    # Plot data and decision boundary
    util.plot(x_train, y_train, model.theta, 'output/p01e_{}.png'.format(pred_path[-5]))

    # Save Predictions
    x_eval, y_eval = util.load_dataset(eval_path, add_intercept=True)
    y_pred = model.predict(x_eval)
    np.savetxt(pred_path, y_pred > 0.5, fmt='%d')
    # *** END CODE HERE ***


class GDA(LinearModel):
    """Gaussian Discriminant Analysis.

    Example usage:
        > clf = GDA()
        > clf.fit(x_train, y_train)
        > clf.predict(x_eval)
    """

    def fit(self, x, y):
        """Fit a GDA model to training set given by x and y.

        Args:
            x: Training example inputs. Shape (m, n).
            y: Training example labels. Shape (m,).

        Returns:
            theta: GDA model parameters.
        """
        # *** START CODE HERE ***
        # Initialization
        m, n = x.shape
        self.theta = np.zeros(n + 1)
        sum_y_indicator = np.sum(y)

        # GDA Parameters
        phi = sum_y_indicator/m
        mu_0 = x.T.dot(1 - y) / (m - sum_y_indicator)
        mu_1 = x.T.dot(y) / sum_y_indicator / sum_y_indicator
        sigma = ((x[y == 0] - mu_0).T.dot(x[y == 0] - mu_0) + (x[y == 1] - mu_1).T.dot(x[y == 1] - mu_1)) / m
        sigma_inv = np.linalg.inv(sigma)
        self.theta[0] = 0.5*(mu_0+mu_1).T.dot(sigma_inv).dot(mu_0-mu_1) - np.log((1 - phi) / phi)
        self.theta[1:] = sigma_inv.dot(mu_1-mu_0)

        return self.theta
        # *** END CODE HERE ***

    def predict(self, x):
        """Make a prediction given new inputs x.

        Args:
            x: Inputs of shape (m, n).

        Returns:
            Outputs of shape (m,).
        """
        # *** START CODE HERE ***
        return 1 / (1 + np.exp(-x.dot(self.theta)))
        # *** END CODE HERE

```

<img src="https://gitee.com/violets/typora--images/raw/main/imgs/202203281713662.png" alt="p01e_1" style="zoom:50%;" />

<img src="https://gitee.com/violets/typora--images/raw/main/imgs/202203281713895.png" alt="p01e_2" style="zoom:50%;" />

### g）(b) vs. (e)

**Dataset 1** is fitted worse by GDA compare with Newton. 

**Possible reason :** $x|y$ might not Gaussian distributed, so the fitted line performed worse.

### h)

**Box-Cox transformation.**

~~Because of the skewed distribution of the data. Try another distribution for dataset 1, (e.g. Poisson, parameter is $\lambda$). *<u>(img pasted from wikipedia)</u>*~~

<img src="https://upload.wikimedia.org/wikipedia/commons/thumb/1/16/Poisson_pmf.svg/1280px-Poisson_pmf.svg.png" alt="Plot of the Poisson PMF" style="zoom:25%;" />

## Problem 2 Incomplete, Positive-Only Labels

# PS 2

## Problem 1

### a)

Dataset A can converge, while Dataset B cannot.

### b)

<img src="https://gitee.com/violets/typora--images/raw/main/imgs/202204121631057.png" alt="image-20220412163139012" style="zoom: 67%;" />

<img src="https://gitee.com/violets/typora--images/raw/main/imgs/202204121631765.png" alt="image-20220412163144730" style="zoom:67%;" />

*Dataset B* is linearly separable, *Dataset A* isn't.

Because $y \in \{1,-1\}$, we can see that the gradient of the cost function is

$$
\nabla_\theta J(\theta) = - \frac{1}{m} \displaystyle \sum_{i = 1}^{m} \frac{y^{(i)} x^{(i)}}{1 + \exp (y^{(i)} \theta^T x^{(i)})}
$$
which means that the gradient descent algorithm is trying to minimize

$$
\ell (\theta) = - \frac{1}{m}  \displaystyle  \sum_{i = 1}^{m} \log \frac{1}{1 + \exp (-y^{(i)} \theta^T x^{(i)})}
$$
If a dataset is completely linearly separable, i.e. $\forall i \in \{1, \dots, m \}, \ y^{(i)} \theta^T x^{(i)} > 0$, then, by multiplying a larger positive scalar, there will always be a new $\theta$ that makes $\ell (\theta)$ even smaller, which prevents the algorithm from converging. However, if the dataset is not linearly separable, $\theta$ cannot be generated in such way while minimizing $\ell (\theta)$.

### c)



## Problem 3

### a) 

*<u>**Prob:**</u>* Show that $$\theta_{\mathrm{MAP}}=\operatorname{argmax}_{\theta} p(y \mid x, \theta) p(\theta)$$, while $$p(\theta) = p(\theta \mid x)$$
$$
\begin{aligned}
p(\theta \mid x,y) & = \frac{p(x,y,\theta)}{p(x,y)} \\
& = \frac{p(y \mid x,\theta)p(x,\theta)}{p(x,y)} \\
& = \frac{p(y \mid x,\theta)p(\theta \mid x)p(x)}{p(x,y)} \\
\end{aligned}
$$
Because $p(\theta) = p(\theta \mid x)$, thus
$$
\begin{aligned}
p(\theta \mid x,y) & = \frac{p(y \mid x,\theta)p(\theta \mid x)p(x)}{p(x,y)} \\
& = \frac{p(y \mid x,\theta)p(\theta)p(x)}{p(x,y)}
\end{aligned}
$$
then
$$
\begin{aligned}
\theta_{MAP} & = \underset{\theta}{argmax} \space p(\theta \mid x,y) \\
& = \underset{\theta}{argmax} \space p(y \mid x,\theta)p(\theta)\frac{p(x)}{p(x,y)}\\
& = \underset{\theta}{argmax} \space p(y \mid x,\theta)p(\theta)
\end{aligned}
$$

### b)

*<u>**Prob:**</u>* Show that MAP estimation with a zero-mean Gaussian prior over $\theta$ (i.e. $\theta \sim \mathcal{N}\left(0, \eta^{2} I\right)$) is equivalent to applying L2 regularization with MLE estimation.

*<u>Solution:</u>*

From (a), we know that
$$
\begin{aligned}\theta_{MAP} & = \underset{\theta}{argmax} \space p(\theta \mid x,y) \\
& = \underset{\theta}{argmax} \space p(y \mid x,\theta)p(\theta)
\end{aligned}
$$
Because $\theta \sim \mathcal{N}\left(0, \eta^{2} I\right)$, we have $p(\theta) = \frac{1}{\eta \sqrt{2 \pi}} e^{-\frac{1}{2}\left(\frac{\theta}{\eta}\right)^{2}}$
$$
\begin{aligned}
\theta_{MAP}& = \underset{\theta}{argmax} \space p(y \mid x,\theta)p(\theta) \\
& = \underset{\theta}{argmax} \space p(y \mid x,\theta) \frac{1}{\eta \sqrt{2 \pi}} e^{-\frac{1}{2}\left(\frac{\theta}{\eta}\right)^{2}} \\
& =\underset{\theta}{argmax} \space log \space {p(y \mid x,\theta)} - \frac{1}{2}\left(\frac{\theta}{\eta}\right)^{2} 
\end{aligned}
$$
While $\eta > 0$ and we do $arg$ operation, thus
$$
\begin{aligned}
\theta_{MAP}& = arg \space \underset{\theta}{max} \space p(y \mid x,\theta)p(\theta) \\
& =arg \space \underset{\theta}{min} \space - log \space {p(y \mid x,\theta)} + \frac{1}{2{\eta}^2}{\| \theta \|} ^{2}_2 
\end{aligned}
$$
so we have $\lambda =  \frac{1}{2{\eta}^2}$

### c)

*<u>**Prob:**</u>* For linear regression model. Come up with a closed form expression for $\theta_{MAP}$
$$
\begin{array}{c}
\epsilon^{(i)} \sim \mathcal{N}\left(0, \sigma^{2}\right) \\
y^{(i)}=\theta^{T} x^{(i)}+\epsilon^{(i)} \\
y^{(i)} \mid x^{(i)}, \theta \sim \mathcal{N}\left(\theta^{T} x^{(i)}, \sigma^{2}\right) \\
p\left(y^{(i)} \mid x^{(i)}, \theta\right)=\frac{1}{\sqrt{2 \pi} \sigma} \exp \left\{-\frac{1}{2 \sigma^{2}}\left(y^{(i)}-\theta^{T} x^{(i)}\right)^{2}\right\} \\
p(\vec{y} \mid X, \theta)=\prod_{i=1}^{m} p\left(y^{(i)} \mid x^{(i)}, \theta\right) \\
=\prod_{i=1}^{m} \frac{1}{\sqrt{2 \pi} \sigma} \exp \left\{-\frac{1}{2 \sigma^{2}}\left(y^{(i)}-\theta^{T} x^{(i)}\right)^{2}\right\} \\
=\frac{1}{(2 \pi)^{m / 2} \sigma^{m}} \exp \left\{-\frac{1}{2 \sigma^{2}} \sum_{i=1}^{m}\left(y^{(i)}-\theta^{T} x^{(i)}\right)^{2}\right\} \\
=\frac{1}{(2 \pi)^{m / 2} \sigma^{m}} \exp \left\{-\frac{1}{2 \sigma^{2}}\|X \theta-\vec{y}\|_{2}^{2}\right\} \\
\log p(\vec{y} \mid X, \theta)=-\frac{m}{2} \log (2 \pi)-m \log \sigma-\frac{1}{2 \sigma^{2}}\|X \theta-\vec{y}\|_{2}^{2} \\
\theta_{\mathrm{MAP}}=\arg \min _{\theta}-\log p(y \mid x, \theta)+\frac{1}{2 \eta^{2}}\|\theta\|_{2}^{2} \\
=\arg \min _{\theta} \frac{1}{2 \sigma^{2}}\|X \theta-\vec{y}\|_{2}^{2}+\frac{1}{2 \eta^{2}}\|\theta\|_{2}^{2} \\
J(\theta)=\frac{1}{2 \sigma^{2}}\|X \theta-\vec{y}\|_{2}^{2}+\frac{1}{2 \eta^{2}}\|\theta\|_{2}^{2} \\
\nabla_{\theta} J(\theta)=\frac{1}{\sigma^{2}}\left(X^{T} X \theta-X^{T} \vec{y}\right)+\frac{1}{\eta^{2}} \theta=0 \\
\theta_{\mathrm{MAP}}=\arg \min _{\theta} J(\theta)=\left(X^{T} X+\frac{\sigma^{2}}{\eta^{2}} I\right)^{-1} X^{T} \vec{y}
\end{array}
$$

### d)

*<u>**Prob:**</u>* Show that $\theta_{MAP}$ in this case is equivalent to the solution of linear regression with $L1$ regularization,
$$
\begin{aligned}
\epsilon^{(i)} &\sim \mathcal{N}\left(0, \sigma^{2}\right) \\
\theta &\sim \mathcal{L}(0, b I) \\
y^{(i)}&=\theta^{T} x^{(i)}+\epsilon^{(i)} \\
f_{\mathcal{L}}(z \mid \mu, b) &=\frac{1}{2 b} \exp \left(-\frac{|z-\mu|}{b}\right)
\end{aligned}
$$
Thus 
$$
\begin{aligned}
p(\theta) & = \frac{1}{(2b)^n} exp(-\frac{|\theta|_1}{b}) \\
\log p(\theta) &= -n\log(2b)-\frac{|\theta|_1}{b} \\
\theta_{\mathrm{MAP}} & = \arg \max_\theta p(y \ \vert \ x, \theta) \ p(\theta) \\
& = \arg \min_\theta - \sum_{i = 1}^{m} \log \frac{1}{\sqrt{2 \pi} \sigma} \exp \big( - \frac{(y^{(i)} - \theta^T x^{(i)})^2}{2 \sigma^2} \big)- \sum_{i = 1}^{n} \log \frac{1}{2 b} \exp \big( - \frac{\vert \theta_i - 0 \vert}{b} \big) \\
                      & = \arg \min_\theta \frac{1}{2 \sigma^2} \sum_{i = 1}^{m} (y^{(i)} - \theta^T x^{(i)})^2 + \sum_{i = 1}^{n} \frac{1}{b} \vert \theta_i \vert \\
                      & = \arg \min_\theta \frac{1}{2 \sigma^2} \Vert X \theta - \vec{y} \Vert_2^2 + \frac{1}{b} \Vert \theta \Vert_1 \\
                      & = \arg \min_\theta \Vert X \theta - \vec{y} \Vert_2^2 + \frac{2 \sigma^2}{b} \Vert \theta \Vert_1
\end{aligned}
$$

## Problem 4

### a)

True. Because $K_1$ and $K_2$ are kernel in same dimension, thus they are symmetric and positive semidefinite matrix in n-th dimension, so $K_1 + K_2$ is also a (Mercer) kernel in that dimension.

### b)

Not necessarily. Because $K$ isn't necessarily  a positive semidefinite matrix, thus not necessarily a kernel in n-th dimension.

### c) & d)

*(c)* True *(d)* False.

$a$ is a positive real number, thus $aK$ is a positive semidefinite matrix, while $-aK$ isn't.

### e)

$K(x, z) = K_1 (x, z) K_2 (x, z)$ is a valid kernel.

Because for any $z \in \mathbb{R}^n$,

$$
\begin{align*}
z^T K z & = \sum_i \sum_j z_i K_{ij} z_j \\
        & = \sum_i \sum_j z_i K(x^{(i)}, x^{(j)}) z_j \\
        & = \sum_i \sum_j z_i K_1 (x^{(i)}, x^{(j)}) K_2 (x^{(i)}, x^{(j)}) z_j \\
        & = \sum_i \sum_j z_i \phi_1 (x^{(i)})^T \phi_1 (x^{(j)}) \phi_2 (x^{(i)})^T \phi_2 (x^{(j)}) z_j \\
        & = \sum_i \sum_j z_i \sum_k \phi_{1k} (x^{(i)}) \phi_{1k} (x^{(j)}) \sum_l \phi_{2l} (x^{(i)}) \phi_{2l} (x^{(j)}) z_j \\
        & = \sum_k \sum_l \sum_i \sum_j z_i \phi_{1k} (x^{(i)}) \phi_{2l} (x^{(i)}) z_j \phi_{1k} (x^{(j)}) \phi_{2l} (x^{(j)}) \\
        & = \sum_k \sum_l \big( \sum_i z_i \phi_{1k} (x^{(i)}) \phi_{2l} (x^{(i)}) \big)^2 \\
        & \geq 0
\end{align*}
$$

### f)

**Yes.** $K$ is PSD 


$$
\begin{aligned}
z^T K z & = \sum_i\sum_j z_i f(x_i)f(x_j) z_j \\
& = \sum_i (z_i f(x_i))^2 \ge 0
\end{aligned}
$$


~~Not necessarily. When $f(x)f(z)<0$, then for $z^T K z$ we have a negative value multiply $z^T z$,  so $z^T K z \le 0$. $K$ isn't PSD.~~

### g)

$K$ is PSD. Because $K_3$ is a kernel no matter what the inputs are.

### h)

$K$ is a kernel. Because $p(x) = \displaystyle \sum_k c_k x^k$, thus $p(K_1 (x,z)) = \displaystyle \sum_k c_k (K_1 (x,z))^k$.

From *(e)*, we know $K(x,z)=K_1(x,z) K_2(x,z)$ is a kernel.

From *(c)*, we know $K(x,z) = a K_1(x,z), a \in \mathbb R^+$ is a kernel.

thus $K$ is a kernel. 

## Problem 5

### a)

1. **<u>How to represent $\theta^{(i)}$</u>**

   Because $\theta^{(i+1)}:=\theta^{(i)}+\alpha\left(y^{(i+1)}-h_{\theta^{(i)}}\left(\phi (x^{(i+1))}\right)\right) \phi (x^{(i+1)})$, so $\theta^{(i)}$ is a linear function of $\phi (x^{(0)}), \phi (x^{(1)})...\phi (x^{(i)})$

   *i.e.*
   $$
   \begin{aligned}
   \theta^{(i)} &= \sum_{j=1}^i \space \beta_i \phi (x^{(i)}) = \beta^T \Phi(x)\\
   \theta^{(0)} &= \sum_{j=1}^0 \space  \beta_i \phi (x^{(i)}) = 0
   \end{aligned}
   $$

2. **<u>How to represent $h_{\theta^{(i)}}\left(x^{(i+1)}\right)$</u>**
   $$
   \begin{aligned}
   h_{\theta^{(i)}}\left(x^{(i+1)}\right) &= g\left(\theta^{(i)^{T}} \phi\left(x^{(i+1)}\right)\right) \\
   & = g\left(\sum_{j=1}^i \space \beta_i \phi \left(x^{(i)}\right) \phi\left(x^{(i+1)}\right)\right) \\
   & = g\left(\sum_{j=1}^i \space \beta_i K\left(x,z\right)\right) \\
   & = g\left(\beta^T K\right) \\
   & = \mathbb I\{\beta^T K = 1\}
   \end{aligned}
   $$

3. **<u>How to modify the update rule?</u>**
   $$
   \begin{aligned}
   \theta^{(i+1)} &= \theta^{(i)} + \alpha \left(\overset {\rightarrow}{y}X - \mathbb I \left\{\beta^T K =1\right\}^T X \right)
   \end{aligned}
   $$
   $$\begin{align*}
   \theta^{(i + 1)} : & = \theta^{(i)} + \alpha \big( y^{(i + 1)} - h_{\theta^{(i)}} (\phi (x^{(i + 1)})) \big) \phi (x^{(i + 1)}) \\
                      & = \sum_{j = 1}^{i} \beta_j \phi (x^{(j)}) + \underbrace{\alpha ( y^{(i + 1)} - \mathrm{sign} \big( \sum_{j = 1}^{i} \beta_j K(x^{(j)}, x^{(i + 1)}) \big) )}_{\beta_{i + 1}} \phi (x^{(i + 1)}) \\
                      & = \sum_{j = 1}^{i + 1} \beta_j \phi (x^{(j)})
   \end{align*}$$

   Therefore, the new update rule is:

   $$\beta_{i + 1} := \alpha ( y^{(i + 1)} - \mathrm{sign} \big( \sum_{j = 1}^{i} \beta_j K(x^{(j)}, x^{(i + 1)}) \big) )$$

### b)

```python
def initial_state():
    """Return the initial state for the perceptron.

    This function computes and then returns the initial state of the perceptron.
    Feel free to use any data type (dicts, lists, tuples, or custom classes) to
    contain the state of the perceptron.

    """

    # *** START CODE HERE ***
    return []
    # *** END CODE HERE ***


def predict(state, kernel, x_i):
    """Perform a prediction on a given instance x_i given the current state
    and the kernel.

    Args:
        state: The state returned from initial_state()
        kernel: A binary function that takes two vectors as input and returns
            the result of a kernel
        x_i: A vector containing the features for a single instance
    
    Returns:
        Returns the prediction (i.e 0 or 1)
    """
    # *** START CODE HERE ***
    return sign(sum(beta * kernel(x, x_i) for x, beta in state))
    # *** END CODE HERE ***


def update_state(state, kernel, learning_rate, x_i, y_i):
    """Updates the state of the perceptron.

    Args:
        state: The state returned from initial_state()
        kernel: A binary function that takes two vectors as input and returns the result of a kernel
        learning_rate: The learning rate for the update
        x_i: A vector containing the features for a single instance
        y_i: A 0 or 1 indicating the label for a single instance
    """
    # *** START CODE HERE ***
    beta_i = learning_rate*(y_i - sign(sum(beta * kernel(x, x_i) for x, beta in state)))
    state.append((x_i, beta_i))
    # *** END CODE HERE ***


```

### c)

**<u>Dot kernel</u>**

<img src="https://gitee.com/violets/typora--images/raw/main/imgs/202204181405364.png" alt="image-20220418140540184" style="zoom: 67%;" />

**<u>Radial basis kernel</u>**

<img src="https://gitee.com/violets/typora--images/raw/main/imgs/202204181407778.png" alt="image-20220418140728685" style="zoom: 55%;" />

Dot kernel perform badly, because dataset isn't linearly separable. Because dot product kernel doesn't have feature mapping, thus the model is still linear after applying the product kernel.

## Problem 6 Spam Classification

```python
import collections

import numpy as np

import util
import svm
```



### a) processing the the spam messages into numpy arrays

```python
def get_words(message):
    """Get the normalized list of words from a message string.

    This function should split a message into words, normalize them, and return
    the resulting list. For splitting, you should split on spaces. For normalization,
    you should convert everything to lowercase.

    Args:
        message: A string containing an SMS message

    Returns:
       The list of normalized words from the message.
    """

    # *** START CODE HERE ***
    return message.lower().split()
    # *** END CODE HERE ***


def create_dictionary(messages):
    """Create a dictionary mapping words to integer indices.

    This function should create a dictionary of word to indices using the provided
    training messages. Use get_words to process each message. 

    Rare words are often not useful for modeling. Please only add words to the dictionary
    if they occur in at least five messages.

    Args:
        messages: A list of strings containing SMS messages

    Returns:
        A python dict mapping words to integers.
    """

    # *** START CODE HERE ***
    word_dict = dict()
    index = 0
    for message in messages:
        word_list = get_words(message)
        for word in word_list:
            if word not in word_dict:
                word_dict[word] = 1
            else:
                word_dict[word] += 1
    # Delete rare words
    index = 0
    for word in list(word_dict.keys()):
        if word_dict[word] >= 5:
            word_dict[word] = index
            index += 1
        else:
            del word_dict[word]
    return word_dict
    # *** END CODE HERE ***


def transform_text(messages, word_dictionary):
    """Transform a list of text messages into a numpy array for further processing.

    This function should create a numpy array that contains the number of times each word
    appears in each message. Each row in the resulting array should correspond to each 
    message and each column should correspond to a word.

    Use the provided word dictionary to map words to column indices. Ignore words that 
    are not present in the dictionary. Use get_words to get the words for a message.

    Args:
        messages: A list of strings where each string is an SMS message.
        word_dictionary: A python dict mapping words to integers.

    Returns:
        A numpy array marking the words present in each message.
    """
    # *** START CODE HERE ***
    m, n = len(messages), len(word_dictionary)
    # Initialize word matrix
    word_matrix = np.zeros([m, n], dtype=int)
    # for word in word_dictionary:
    #     word_index = word_dictionary[word]
    #     word_matrix[0, word_index] = word_index
    # Record occurrence of words in messages
    msg_index = 0
    for message in messages:
        word_list = get_words(message)
        for word in word_list:
            if word in word_dictionary:
                word_matrix[msg_index, word_dictionary[word]] += 1
        msg_index += 1
    return word_matrix
    # *** END CODE HERE ***
```

### b) Fit & Predict

**<u>Prediction accuracy:</u>** `Naive Bayes had an accuracy of 0.8870967741935484 on the testing set`

If possibility is `0.28` instead of `0.5`, the accuracy on the test set is `0.9301075268817204`

```python

def fit_naive_bayes_model(matrix, labels):
    """Fit a naive bayes model.

    This function should fit a Naive Bayes model given a training matrix and labels.

    The function should return the state of that model.

    Feel free to use whatever datatype you wish for the state of the model.

    Args:
        matrix: A numpy array containing word counts for the training data
        labels: The binary (0 or 1) labels for that training data

    Returns: The trained model
    """
    # *** START CODE HERE ***
    m, n = matrix.shape
    num_spam = sum(labels)
    phi_1 = (matrix.T.dot(labels) + 1) / (num_spam + 2)
    phi_0 = (matrix.T.dot(1-labels) + 1) / (m - num_spam + 2)
    phi_y = num_spam / m
    return phi_1, phi_0, phi_y
    # *** END CODE HERE ***


def predict_from_naive_bayes_model(model, matrix):
    """Use a Naive Bayes model to compute predictions for a target matrix.

    This function should be able to predict on the models that fit_naive_bayes_model
    outputs.

    Args:
        model: A trained model from fit_naive_bayes_model
        matrix: A numpy array containing word counts

    Returns: A numpy array contains the predictions from the model
    """
    # *** START CODE HERE ***
    phi_1, phi_0, phi_y = model
    log_y1 = np.log(matrix.dot(phi_1)) + np.log(phi_y)
    log_y0 = np.log(matrix.dot(phi_0)) + np.log(1-phi_y)
    possibility_y1 = np.exp(log_y1) / (np.exp(log_y0) + np.exp(log_y1))
    return [possibility_y1 > 0.5]
    # *** END CODE HERE ***

```

### c) Get Spam Indicators

`The top 5 indicative words for Naive Bayes are:  ['claim', 'won', 'prize', 'tone', 'urgent!']`

```python
def get_top_five_naive_bayes_words(model, dictionary):
    """Compute the top five words that are most indicative of the spam (i.e positive) class.

    Ues the metric given in 6c as a measure of how indicative a word is.
    Return the words in sorted form, with the most indicative word first.

    Args:
        model: The Naive Bayes model returned from fit_naive_bayes_model
        dictionary: A mapping of word to integer ids

    Returns: The top five most indicative words in sorted order with the most indicative first
    """
    # *** START CODE HERE ***
    phi_1, phi_0, phi_y = model
    spam_coeff = np.log(phi_1) - np.log(phi_0)
    hot5word_index = spam_coeff.argsort()[-5:]
    hot5word = list()
    for index in hot5word_index:
        for key in dictionary:
            if dictionary[key] == index:
                hot5word.append(key)
                break
    hot5word.reverse()
    return hot5word
    # *** END CODE HERE ***


```

### d) Compute Best SVM Radius

`The optimal SVM radius was 0.1
The SVM model had an accuracy of 0.9695340501792115 on the testing set`

```python
def compute_best_svm_radius(train_matrix, train_labels, val_matrix, val_labels, radius_to_consider):
    """Compute the optimal SVM radius using the provided training and evaluation datasets.

    You should only consider radius values within the radius_to_consider list.
    You should use accuracy as a metric for comparing the different radius values.

    Args:
        train_matrix: The word counts for the training data
        train_labels: The spam or not spam labels for the training data
        val_matrix: The word counts for the validation data
        val_labels: The spam or not spam labels for the validation data
        radius_to_consider: The radius values to consider
    
    Returns:
        The best radius which maximizes SVM accuracy.
    """
    # *** START CODE HERE ***
    rad_acc_dict = {}
    for radius in radius_to_consider:
        predict_label = svm.train_and_predict_svm(train_matrix, train_labels, val_matrix, radius)
        accuracy = np.mean(predict_label == val_labels)
        rad_acc_dict[radius] = accuracy
    maxaccuracy = max(rad_acc_dict.values())
    for key in rad_acc_dict:
        if rad_acc_dict[key] == maxaccuracy:
            return key
    # return rad_acc_dict.popitem()
    # *** END CODE HERE ***

```

# PS 3

## Problem 1 A Simple Neural Network

### a) Calculate gradient descent update to $w _{1,2}^{[i]}$

$$
\begin{aligned}
\frac{\part l}{\part w _{1,2}^{[i]}} & = \frac{\part l}{\part o^{[i]}} \frac{\part o^{[i]}}{\part h_2^{[i]}} \frac{\part h_2^{[i]}}{\part w _{1,2}^{[i]}}\\
& = \frac{2}{m} \sum _{i=1}^m \left(o^{[i]}-y^{[i]}\right) o^{[i]} \left(1-o^{[i]}\right) w_2^{[2]} h_2^{[i]} \left(1-h_2^{[i]}\right) x_1^{[i]} \\
& = \frac{2}{m}  w_2^{[2]}  \sum _{i=1}^m \left(o^{[i]}-y^{[i]}\right) o^{[i]} \left(1-o^{[i]}\right) h_2^{[i]} \left(1-h_2^{[i]}\right) x_1^{[i]} \\
\text{Update rule for } w _{1,2}^{[i]} \text{ is:}\\
w _{1,2}^{[i]} &= w _{1,2}^{[i]} - \alpha \frac{\part l}{\part w _{1,2}^{[i]}}\\
& = w _{1,2}^{[i]} - \alpha \frac{2}{m}  w_2^{[2]}  \sum _{i=1}^m \left(o^{[i]}-y^{[i]}\right) o^{[i]} \left(1-o^{[i]}\right) h_2^{[i]} \left(1-h_2^{[i]}\right) x_1^{[i]} 
\end{aligned}
$$

### b) Modify activation function as step function and prove its accuracy

After using step function as activation function to achieve 100% accuracy.

Because we CAN use three line(which form a triangle area) to separate different category samples for the given dataset.

```python
def optimal_step_weights():
    """Return the optimal weights for the neural network with a step activation function.
    
    This function will not be graded if there are no optimal weights.
    See the PDF for instructions on what each weight represents.
    
    The hidden layer weights are notated by [1] on the problem set and 
    the output layer weights are notated by [2].

    This function should return a dict with elements for each weight, see example_weights above.

    """
    w = example_weights()

    # *** START CODE HERE ***
    w['hidden_layer_0_1'] = -1
    w['hidden_layer_1_1'] = 2
    w['hidden_layer_2_1'] = 0
    w['hidden_layer_0_2'] = -1
    w['hidden_layer_1_2'] = 0
    w['hidden_layer_2_2'] = 2
    w['hidden_layer_0_3'] = -4
    w['hidden_layer_1_3'] = 1
    w['hidden_layer_2_3'] = 1

    w['output_layer_0'] = -1
    w['output_layer_1'] = 2
    w['output_layer_2'] = 2
    w['output_layer_3'] = 2
    # *** END CODE HERE ***

    return w

```

### c) Modify activation function as step function and prove its accuracy

It's not possible, when we use identity(linear) function as activation function and output layer activation is step function, the neural network can be viewed as a linear classifier. But given dataset is not linearly separable.

## Problem 2 KL divergence and Maximum Likelihood

### a) Non-negativity

$$
\begin{aligned}
D_{\mathrm{KL}}(P \| Q) &=\sum_{x \in \mathcal{X}} P(x) \log \frac{P(x)}{Q(x)} \\
&=-\sum_{x \in \mathcal{X}} P(x) \log \frac{Q(x)}{P(x)} \\
&=E\left[-\log \frac{Q(x)}{P(x)}\right]\\
&\ge -\log E\left[ \frac{Q(x)}{P(x)}\right]\\
&=- \log  \sum_{x \in \mathcal{X}}P(x) \frac{Q(x)}{P(x)}\\
&=- \log  \sum_{x \in \mathcal{X}} Q(x) \\
&=- \log 1 \\
&=0
\end{aligned}
$$

Further,

-  If $P = Q$, then $D_{\mathrm{KL}}(P \| Q) =\sum_{x \in \mathcal{X}} P(x) \log 1 = 0$

- Because $-\log$ is strictly convex, so if $D_{\mathrm{KL}}(P \| Q) =0$ while $\frac{Q(x)}{P(x)}$ is a constant for all $x \in \mathcal{X}$, which is $Q(x)=P(x)$

Thus, $D_{\mathrm{KL}}(P \| Q) =0$ if and only if $P=Q$

### b) Chain rule for KL divergence

$$
\begin{aligned}
D_{\mathrm{KL}}(P(X) \| Q(X)) &= \sum_{x \in \mathcal{X}} P(x) \log \frac{P(x)}{Q(x)} \\
D_{\mathrm{KL}}(P(Y \mid X) \| Q(Y \mid X)) &= \sum_{x} P(x)\left(\sum_{y} P(y \mid x) \log \frac{P(y \mid x)}{Q(y \mid x)}\right)
\end{aligned}
$$



According to Bayes, we have
$$
P(X,Y) = P(Y \mid X) P(X)
$$

$$
\begin{aligned}
D_{\mathrm{KL}}(P(X, Y) \| Q(X, Y)) &= D_{\mathrm{KL}}(P(Y \mid X) P(X) \| Q(Y \mid X) Q(X)) \\
&= \sum_{x}\sum_{y} P(y \mid x)P(x) \log {\frac {P(y \mid x)P(x)}{Q(y \mid x)Q(x)}} \\
&= \sum_{x}\sum_{y} P(y \mid x)P(x)  \left(\log {\frac{P(y \mid x)}{Q(y \mid x)}}+\log \frac {P(x)}{Q(x)} \right) \\
&= \sum_{y}P(y \mid x)\sum_{x}P(x)\log \frac {P(x)}{Q(x)} + \sum_{x} P(x)\left(\sum_{y} P(y \mid x) \log \frac{P(y \mid x)}{Q(y \mid x)}\right) \\
&= 1* \sum_{x}P(x)\log \frac {P(x)}{Q(x)} + \sum_{x} P(x)\left(\sum_{y} P(y \mid x) \log \frac{P(y \mid x)}{Q(y \mid x)}\right) \\
&=D_{\mathrm{KL}}(P(X) \| Q(X))+D_{\mathrm{KL}}(P(Y \mid X) \| Q(Y \mid X))
\end{aligned}
$$

### c) KL and maximum likelihood

$$
D_{\mathrm{KL}}\left(\hat{P} \| P_{\theta}\right) = \sum_{x} \hat{P}(x) \log \frac{\hat{P}(x)}{P_{\theta}(x)}
$$

Thus,
$$
\begin{aligned}
\arg \min _{\theta} D_{\mathrm{KL}}\left(\hat{P} \| P_{\theta}\right)&= \arg \min _{\theta} \sum_{x} \hat{P}(x) \log \left({\hat{P}(x)}-{P_{\theta}(x)}\right) \\
&= \arg \max _{\theta}\sum_{x} \hat{P}(x) \log {P_{\theta}(x)}\\
&= \arg \max _{\theta}\sum_{x} \left(\frac{1}{m} \sum_{i=1}^m 1\left\{x^{(i)} = x \right\}\right)P_{\theta}(x)\\
&= \arg \max _{\theta} \sum_{i=1}^{m} \log P_{\theta}\left(x^{(i)}\right) \\
\end{aligned}
$$

## Problem 3 KL Divergence, Fisher Information, and the Natural Gradient

### a) Score Function

*signifies the sensitivity of the likelihood function with respect to the parameters.*
$$
\begin{aligned}
\nabla_{\theta} \log p\left(y ; \theta\right) &= \frac {\nabla_{\theta} p(y ; \theta)}{p(y ; \theta)} \\
\\
\mathbb{E}_{y \sim p(y ; \theta)}\left[\left.\nabla_{\theta^{\prime}} \log p\left(y ; \theta^{\prime}\right)\right|_{\theta^{\prime}=\theta}\right] &= \int_{-\infty}^{\infty} p(y) \left.\nabla_{\theta^{\prime}} \log p\left(y ; \theta^{\prime}\right)\right|_{\theta^{\prime}=\theta} d y\\
&= \int_{-\infty}^{\infty} p(y ; \theta)\frac {\nabla_{\theta} p(y ; \theta)}{p(y ; \theta)} d y \\
&= \int_{-\infty}^{\infty} \nabla_{\theta} p(y ; \theta) d y\\
&= 0 \\
\end{aligned}
$$

### b) Fisher information

*Fisher information represents the amount of information that a random variable Y carries about a parameter θ of interest.*

From score function, we know that,
$$
\begin{aligned}
\mathbb{E}_{y \sim p(y ; \theta)}\left[\left.\nabla_{\theta^{\prime}} \log p\left(y ; \theta^{\prime}\right)\right|_{\theta^{\prime}=\theta}\right] = 0 
\end{aligned}
$$
So we can derive the following:
$$
\begin{aligned}
\mathcal{I}(\theta)&= \operatorname{Cov}_{y \sim p(y ; \theta)}\left[\left.\nabla_{\theta^{\prime}} \log p\left(y ; \theta^{\prime}\right)\right|_{\theta^{\prime}=\theta}\right]\\
&= \mathbb{E}_{y \sim p(y ; \theta)}\left[\left.\nabla_{\theta^{\prime}} \log p\left(y ; \theta^{\prime}\right) \nabla_{\theta^{\prime}} \log p\left(y ; \theta^{\prime}\right)^{T}\right|_{\theta^{\prime} = \theta}\right] - 
\mathbb{E}_{y \sim p(y ; \theta)}\left[\left.\nabla_{\theta^{\prime}} \log p\left(y;\theta^{\prime}\right) \right|_{\theta^{\prime} = \theta}\right]
\mathbb{E}_{y \sim p(y ; \theta)} \left[\left.\nabla_{\theta^{\prime}} \log p\left(y ; \theta^{\prime}\right)^{T}\right|_{\theta^{\prime}=\theta}\right]\\
&=\mathbb{E}_{y \sim p(y ; \theta)}\left[\left.\nabla_{\theta^{\prime}} \log p\left(y ; \theta^{\prime}\right) \nabla_{\theta^{\prime}} \log p\left(y ; \theta^{\prime}\right)^{T}\right|_{\theta^{\prime}=\theta}\right]  -0 \\
&=\mathbb{E}_{y \sim p(y ; \theta)}\left[\left.\nabla_{\theta^{\prime}} \log p\left(y ; \theta^{\prime}\right) \nabla_{\theta^{\prime}} \log p\left(y ; \theta^{\prime}\right)^{T}\right|_{\theta^{\prime}=\theta}\right] \\
\end{aligned}
$$

###  c)  ==Fisher Information (alternate form)==

$$
\begin{aligned}
\mathbb{E}_{y \sim p(y ; \theta)}\left[-\left.\nabla_{\theta^{\prime}}^{2} \log p\left(y ; \theta^{\prime}\right)\right|_{\theta^{\prime}=\theta}\right]=\mathcal{I}(\theta)
\end{aligned}
$$

![image-20220430105910930](https://gitee.com/violets/typora--images/raw/main/imgs/202204301059192.png)

### d) Approximating $D_{KL}$ with Fisher Information

Make $\overset \sim\theta = \theta +d$, Then Taylor expansion for $\log p\left(\overset \sim\theta \right)$ is 
$$
\log p\left(\overset \sim\theta \right) \approx \log p\left(\theta \right)+d^T \left.\nabla_{\theta ^\prime} \log p(\theta ^\prime) \right|_{\theta ^\prime = \theta}  +\frac{1}{2}d^T  \left.\nabla_{\theta ^\prime}^2 \log p(\theta ^\prime) \right|_{\theta ^\prime = \theta}d
$$


So we have
$$
\begin{align}
D_{\mathrm{KL}}\left(p_{\theta} \| p_{\theta+d}\right) &= \sum_y p_\theta \log p_\theta -\sum_y p_\theta \log p_{\theta+d}\\
&\approx \sum_y p_\theta \log p_\theta -\sum_y p_\theta \left(\log p\left(\theta \right)+d^T \left.\nabla_{\theta ^\prime} \log p(\theta ^\prime) \right|_{\theta ^\prime = \theta}  +\frac{1}{2}d^T  \left.\nabla_{\theta ^\prime}^2 \log p(\theta ^\prime) \right|_{\theta ^\prime = \theta}d\right) \\
&= -\sum_y p_\theta d^T \left.\nabla_{\theta ^\prime} \log p(\theta ^\prime) \right|_{\theta ^\prime = \theta} -\sum_y p_\theta  \frac{1}{2}d^T  \left.\nabla_{\theta ^\prime}^2 \log p(\theta ^\prime) \right|_{\theta ^\prime = \theta}d \\
&= - d^T \sum_y p_\theta \left.\nabla_{\theta ^\prime} \log p(\theta ^\prime) \right|_{\theta ^\prime = \theta} +  \frac{1}{2}d^T  \left(\sum_y p_\theta \left.\nabla_{\theta ^\prime}^2  -\log p(\theta ^\prime) \right|_{\theta ^\prime = \theta} \right)d \\
&= 0 + \frac{1}{2}d^T \mathbb{E}_{y \sim p(y ; \theta)}\left[-\left.\nabla_{\theta^{\prime}}^{2} \log p\left(y ; \theta^{\prime}\right)\right|_{\theta^{\prime}=\theta}\right] d\\
&= \frac{1}{2} d^{T} \mathcal{I}(\theta) d \\ \\
D_{\mathrm{KL}}\left(p_{\theta} \| p_{\theta+d}\right) &\approx  \frac{1}{2} d^{T} \mathcal{I}(\theta) d 
\end{align}
$$

### e) ==Natural Gradient==

$$
\begin{array}{l}
\ell(\theta+d) \approx \ell(\theta)+\left.d^{T} \nabla_{\theta^{\prime}} \ell\left(\theta^{\prime}\right)\right|_{\theta^{\prime}=\theta}\\
=\log p(y ; \theta)+\left.d^{T} \nabla_{\theta^{\prime}} \log p\left(y ; \theta^{\prime}\right)\right|_{\theta^{\prime}=\theta}\\
=\log p(y ; \theta)+d^{T} \frac{\left.\nabla_{\theta^{\prime}} p\left(y ; \theta^{\prime}\right)\right|_{\theta^{\prime}=\theta}}{p(y ; \theta)}\\
D_{\mathrm{KL}}\left(p_{\theta} \| p_{\theta+d}\right) \approx \frac{1}{2} d^{T} \mathcal{I}(\theta) d\\
\mathcal{L}(d, \lambda)=\ell(\theta+d)-\lambda\left[D_{\mathrm{KL}}\left(p_{\theta} \| p_{\theta+d}\right)-c\right]\\
\approx \log p(y ; \theta)+d^{T} \frac{\left.\nabla_{\theta^{\prime}} p\left(y ; \theta^{\prime}\right)\right|_{\theta^{\prime}=\theta}}{p(y ; \theta)}-\lambda\left[\frac{1}{2} d^{T} \mathcal{I}(\theta) d-c\right]\\
\nabla_{d} \mathcal{L}(d, \lambda) \approx \frac{\left.\nabla_{\theta^{\prime}} p\left(y ; \theta^{\prime}\right)\right|_{\theta^{\prime}=\theta}}{p(y ; \theta)}-\lambda \mathcal{I}(\theta) d=0\\
\tilde{d}=\frac{1}{\lambda} \mathcal{I}(\theta)^{-1} \frac{\left.\nabla_{\theta^{\prime}} p\left(y ; \theta^{\prime}\right)\right|_{\theta^{\prime}=\theta}}{p(y ; \theta)}\\
\nabla_{\lambda} \mathcal{L}(d, \lambda) \approx c-\frac{1}{2} d^{T} \mathcal{I}(\theta) d\\
=c-\frac{1}{2} \cdot \frac{1}{\lambda} \frac{\left.\nabla_{\theta^{\prime}} p\left(y ; \theta^{\prime}\right)\right|_{\theta^{\prime}=\theta}{ }^{T}}{p(y ; \theta)} \mathcal{I}(\theta)^{-1} \cdot \mathcal{I}(\theta) \cdot \frac{1}{\lambda} \mathcal{I}(\theta)^{-1} \frac{\left.\nabla_{\theta^{\prime}} p\left(y ; \theta^{\prime}\right)\right|_{\theta^{\prime}=\theta}}{p(y ; \theta)}\\
=c-\left.\left.\frac{1}{2 \lambda^{2}(p(y ; \theta))^{2}} \nabla_{\theta^{\prime}} p\left(y ; \theta^{\prime}\right)\right|_{\theta^{\prime}=\theta}{ }^{T} \mathcal{I}(\theta)^{-1} \nabla_{\theta^{\prime}} p\left(y ; \theta^{\prime}\right)\right|_{\theta^{\prime}=\theta}\\
=0\\
\lambda=\sqrt{\left.\left.\frac{1}{2 c(p(y ; \theta))^{2}} \nabla_{\theta^{\prime}} p\left(y ; \theta^{\prime}\right)\right|_{\theta^{\prime}=\theta} ^{T} \mathcal{I}(\theta)^{-1} \nabla_{\theta^{\prime}} p\left(y ; \theta^{\prime}\right)\right|_{\theta^{\prime}=\theta}}\\
d^{*}=\sqrt{\frac{2 c(p(y ; \theta))^{2}}{\left.\left.\nabla_{\theta^{\prime}} p\left(y ; \theta^{\prime}\right)\right|_{\theta^{\prime}=\theta} ^{T} \mathcal{I}(\theta)^{-1} \nabla_{\theta^{\prime}} p\left(y ; \theta^{\prime}\right)\right|_{\theta^{\prime}=\theta}}} \mathcal{I}(\theta)^{-1} \frac{\left.\nabla_{\theta^{\prime}} p\left(y ; \theta^{\prime}\right)\right|_{\theta^{\prime}=\theta}}{p(y ; \theta)}\\
=\left.\sqrt{\frac{2 c}{\left.\left.\nabla_{\theta^{\prime}} p\left(y ; \theta^{\prime}\right)\right|_{\theta^{\prime}=\theta} ^{T} \mathcal{I}(\theta)^{-1} \nabla_{\theta^{\prime}} p\left(y ; \theta^{\prime}\right)\right|_{\theta^{\prime}=\theta}}} \mathcal{I}(\theta)^{-1} \nabla_{\theta^{\prime}} p\left(y ; \theta^{\prime}\right)\right|_{\theta^{\prime}=\theta}
\end{array}
$$

### f) ==Relation to Newton’s Method==

For natural gradient
$$
\tilde{d}=\frac{1}{\lambda} \mathcal{I}(\theta)^{-1} \frac{\left.\nabla_{\theta^{\prime}} p\left(y ; \theta^{\prime}\right)\right|_{\theta^{\prime}=\theta}}{p(y ; \theta)}
$$
For Newton’s Method
$$
\begin{align}
\theta: & = \theta-H^{-1} \nabla_{\theta} \ell(\theta)
\end{align}
$$
Then,
$$
\begin{aligned}
\mathcal{I}(\theta) &=\mathbb{E}_{y \sim p(y ; \theta)}\left[-\left.\nabla_{\theta^{\prime}}^{2} \log p\left(y ; \theta^{\prime}\right)\right|_{\theta^{\prime}=\theta}\right] \\
&=\mathbb{E}_{y \sim p(y ; \theta)}\left[-\nabla_{\theta}^{2} \ell(\theta)\right] \\
&=-\mathbb{E}_{y \sim p(y ; \theta)}[H] \\
\theta: &=\theta+\tilde{d} \\
&=\theta+\frac{1}{\lambda} \mathcal{I}(\theta)^{-1} \nabla_{\theta} \ell(\theta) \\
&=\theta-\frac{1}{\lambda} \mathbb{E}_{y \sim p(y ; \theta)}[H]^{-1} \nabla_{\theta} \ell(\theta)
\end{aligned}
$$

## Problem 4 Semi-supervised EM

### a)

$$
\begin{aligned}
\ell_{\text {semi-sup }}\left(\theta^{(t+1)}\right) &=\ell_{\text {unsup }}\left(\theta^{(t+1)}\right)+\alpha \ell_{\text {sup }}\left(\theta^{(t+1)}\right) \\
& \geq \sum_{i=1}^{m}\left(\sum_{z^{(i)}} Q_{i}^{(t)}\left(z^{(i)}\right) \log \frac{p\left(x^{(i)}, z^{(i)} ; \theta^{(t+1)}\right)}{Q_{i}^{(t)}\left(z^{(i)}\right)}\right)+\alpha\left(\sum_{i=1}^{\tilde{m}} \log p\left(\tilde{x}^{(i)}, \tilde{z}^{(i)} ; \theta^{(t+1)}\right)\right) \\
& \geq \sum_{i=1}^{m}\left(\sum_{z^{(i)}} Q_{i}^{(t)}\left(z^{(i)}\right) \log \frac{p\left(x^{(i)}, z^{(i)} ; \theta^{(t)}\right)}{Q_{i}^{(t)}\left(z^{(i)}\right)}\right)+\alpha\left(\sum_{i=1}^{\tilde{m}} \log p\left(\tilde{x}^{(i)}, \tilde{z}^{(i)} ; \theta^{(t)}\right)\right) \\
&=\ell_{\text {unsup }}\left(\theta^{(t)}\right)+\alpha \ell_{\text {sup }}\left(\theta^{(t)}\right) \\
&=\ell_{\text {semi-sup }}\left(\theta^{(t)}\right)
\end{aligned}
$$

### b) Semi-supervised E-step

$$
\begin{aligned}
w_{j}^{(i)}&=p\left(z^{(i)}=j \mid x^{(i)} ; \phi, \mu, \Sigma\right) \\
&= \frac{p\left(x^{(i)} \mid z^{(i)}=j ; \mu, \Sigma\right) p\left(z^{(i)}=j ; \phi\right)}{\sum_{l=1}^{k} p\left(x^{(i)} \mid z^{(i)}=l ; \mu, \Sigma\right) p\left(z^{(i)}=l ; \phi\right)} \\
&= \frac {\frac{1}{(2 \pi)^{n / 2}|\Sigma_j|^{1 / 2}} \exp \left(-\frac{1}{2}(x-\mu_j)^{T} \Sigma_j^{-1}(x-\mu_j)\right) \phi_j}
{\sum_{l=1}^{k} \frac {1}{(2 \pi)^{n / 2}|\Sigma_l|^{1 / 2}} \exp \left(-\frac{1}{2}(x-\mu_l)^{T} \Sigma_l^{-1}(x-\mu_l)\right) \phi_l} \\
\end{aligned}
$$

Appendix:
$$
\begin{aligned}
p\left(x^{(i)} \mid z^{(i)}=j ; \mu, \Sigma\right) &= \frac{1}{(2 \pi)^{n / 2}|\Sigma|^{1 / 2}} \exp \left(-\frac{1}{2}(x-\mu)^{T} \Sigma^{-1}(x-\mu)\right)\\
p\left(z^{(i)}=j ; \phi\right) &= \phi _j
\end{aligned}
$$


### c) ==Semi-supervised M-step==

$$
\begin{aligned}
\phi_{j} & = \frac{\sum_{i = 1}^{m} w_{j}^{(i)}+\alpha \sum_{i = 1}^{\tilde{m}} \mathbb I\left\{\tilde{z}^{(i)}  = j\right\}}{m+\alpha \tilde{m}} \\
\mu_{j} &=\frac{\sum_{i=1}^{m} w_{j}^{(i)} x^{(i)} + \alpha \sum_{i=1}^{\tilde m} \mathbb I\{\tilde z^{(i)} = j \} \tilde x^{(i)}}{\sum_{i=1}^{m} w_{j}^{(i)} + \alpha \sum_{i=1}^{\tilde m} \mathbb I\{\tilde z^{(i)} = j \}} \\
\Sigma_{j} &=\frac{\sum_{i=1}^{m} w_{j}^{(i)}\left(x^{(i)}-\mu_{j}\right)\left(x^{(i)}-\mu_{j}\right)^{T} + \alpha \sum_{i=1}^{\tilde m} \mathbb I\{\tilde z^{(i)} = j \} \left(\tilde x^{(i)}-\mu_{j}\right)\left(\tilde x^{(i)}-\mu_{j}\right)^{T}}{\sum_{i=1}^{m} w_{j}^{(i)}+\alpha \sum_{i=1}^{\tilde m} \mathbb I\{\tilde z^{(i)} = j \}}\\ 
\end{aligned}
$$



### d) Classical (Unsupervised) EM Implementation

<img src="https://gitee.com/violets/typora--images/raw/main/imgs/202205031506833.png" alt="image-20220503150636726" style="zoom: 50%;" />

<img src="https://gitee.com/violets/typora--images/raw/main/imgs/202205031507505.png" alt="image-20220503150746391" style="zoom:50%;" />

<img src="https://gitee.com/violets/typora--images/raw/main/imgs/202205031508378.png" alt="image-20220503150836263" style="zoom:50%;" />



**Initialize Dataset**

```python
def main(is_semi_supervised, trial_num):
    """Problem 3: EM for Gaussian Mixture Models (unsupervised and semi-supervised)"""
    print('Running {} EM algorithm...'
          .format('semi-supervised' if is_semi_supervised else 'unsupervised'))

    # Load dataset
    train_path = os.path.join('..', 'data', 'ds3_train.csv')
    x, z = load_gmm_dataset(train_path)
    x_tilde = None

    if is_semi_supervised:
        # Split into labeled and unlabeled examples
        labeled_idxs = (z != UNLABELED).squeeze()
        x_tilde = x[labeled_idxs, :]   # Labeled examples
        z = z[labeled_idxs, :]         # Corresponding labels
        x = x[~labeled_idxs, :]        # Unlabeled examples

    # *** START CODE HERE ***
    # (1) Initialize mu and sigma by splitting the m data points uniformly at random
    # into K groups, then calculating the sample mean and covariance for each group
    # Split data
    m, n = x.shape
    row_indices = np.random.permutation(m)
    partitionLen = int(m / K)
    # initialize mu and sigma
    mu, sigma = np.zeros((K, n)), np.zeros((K, n, n))
    for i in range(K):
        x_samp = x[row_indices[i * partitionLen:(i + 1) * partitionLen]]
        mu[i] = np.mean(x_samp, axis=0)
        sigma[i] = np.cov(x_samp.T)
    # (2) Initialize phi to place equal probability on each Gaussian
    # phi should be a numpy array of shape (K,)
    phi = 1 / K * np.ones(K)
    # (3) Initialize the w values to place equal probability on each Gaussian
    # w should be a numpy array of shape (m, K)
    w = 1 / K * np.ones((m, K))
    # *** END CODE HERE ***

    if is_semi_supervised:
        w = run_semi_supervised_em(x, x_tilde, z, w, phi, mu, sigma)
    else:
        w = run_em(x, w, phi, mu, sigma)

    # Plot your predictions
    z_pred = np.zeros(m)
    if w is not None:  # Just a placeholder for the starter code
        for i in range(m):
            z_pred[i] = np.argmax(w[i])

    plot_gmm_preds(x, z_pred, is_semi_supervised, plot_id=trial_num)


```



**EM Algorithm**

```python
def run_em(x, w, phi, mu, sigma):
    """Problem 3(d): EM Algorithm (unsupervised).

    See inline comments for instructions.

    Args:
        x: Design matrix of shape (m, n).
        w: Initial weight matrix of shape (m, k).
        phi: Initial mixture prior, of shape (k,).
        mu: Initial cluster means, list of k arrays of shape (n,).
        sigma: Initial cluster covariances, list of k arrays of shape (n, n).

    Returns:
        Updated weight matrix of shape (m, k) resulting from EM algorithm.
        More specifically, w[i, j] should contain the probability of
        example x^(i) belonging to the j-th Gaussian in the mixture.
    """
    m, n = x.shape
    # No need to change any of these parameters
    eps = 1e-3  # Convergence threshold
    max_iter = 1000

    # Stop when the absolute change in log-likelihood is < eps
    # See below for explanation of the convergence criterion
    it = 0
    ll = prev_ll = None
    while it < max_iter and (prev_ll is None or np.abs(ll - prev_ll) >= eps):
        # Just a placeholder for the starter code
        # *** START CODE HERE
        # (1) E-step: Update your estimates in w
        # (2) M-step: Update the model parameters phi, mu, and sigma
        # (3) Compute the log-likelihood of the data to check for convergence.
        # By log-likelihood, we mean `ll = sum_x[log(sum_z[p(x|z) * p(z)])]`.
        # We define convergence by the first iteration where abs(ll - prev_ll) < eps.
        # Hint: For debugging, recall part (a). We showed that ll should be monotonically increasing.

        # (1) E-step:
        # it += 1
        prev_ll = ll
        for i in range(K):
            w[:, i] = np.exp(-0.5 * ((x - mu[i]).dot(np.linalg.inv(sigma[i]))*(x - mu[i])).sum(axis=1))*phi[i] / np.sqrt(np.linalg.det(sigma[i]))
        ll = np.sum(np.log(w))
        w /= np.sum(w, axis=1)[:, None]
        # (2) M-step
        phi = np.sum(w, axis=0)/m
        for i in range(K):
            mu[i] = x.T.dot(w[:, i]) / np.sum(w[:, i])
            sigma[i] = ((x-mu[i]).T*w[:, i]).dot(x-mu[i]) / np.sum(w[:, i])
        it += 1
        # *** END CODE HERE ***
    print(f'Number of iterations:{it}')
    return w

```

### e) Semi-supervised EM

<img src="https://gitee.com/violets/typora--images/raw/main/imgs/202205031511083.png" alt="image-20220503151118973" style="zoom:50%;" />

<img src="https://gitee.com/violets/typora--images/raw/main/imgs/202205031512520.png" alt="image-20220503151227392" style="zoom:50%;" />

<img src="https://gitee.com/violets/typora--images/raw/main/imgs/202205031512969.png" alt="image-20220503151245859" style="zoom:50%;" />

```python
def run_semi_supervised_em(x, x_tilde, z, w, phi, mu, sigma):
    """Problem 3(e): Semi-Supervised EM Algorithm.

    See inline comments for instructions.

    Args:
        x: Design matrix of unlabeled examples of shape (m, n).
        x_tilde: Design matrix of labeled examples of shape (m_tilde, n).
        z: Array of labels of shape (m_tilde, 1).
        w: Initial weight matrix of shape (m, k).
        phi: Initial mixture prior, of shape (k,).
        mu: Initial cluster means, list of k arrays of shape (n,).
        sigma: Initial cluster covariances, list of k arrays of shape (n, n).

    Returns:
        Updated weight matrix of shape (m, k) resulting from semi-supervised EM algorithm.
        More specifically, w[i, j] should contain the probability of
        example x^(i) belonging to the j-th Gaussian in the mixture.
    """
    # No need to change any of these parameters
    alpha = 20.  # Weight for the labeled examples
    eps = 1e-3   # Convergence threshold
    max_iter = 1000

    m, n = x.shape
    m_tilde = x_tilde.shape[0]
    # Stop when the absolute change in log-likelihood is < eps
    # See below for explanation of the convergence criterion
    it = 0
    ll = prev_ll = None
    while it < max_iter and (prev_ll is None or np.abs(ll - prev_ll) >= eps):
        pass  # Just a placeholder for the starter code
        # *** START CODE HERE ***
        # (1) E-step: Update your estimates in w
        # (2) M-step: Update the model parameters phi, mu, and sigma
        # (3) Compute the log-likelihood of the data to check for convergence.
        # Hint: Make sure to include alpha in your calculation of ll.
        # Hint: For debugging, recall part (a). We showed that ll should be monotonically increasing.
        # (1) E-step:
        # it += 1
        prev_ll = ll
        for i in range(K):
            w[:, i] = np.exp(-0.5 * ((x - mu[i]).dot(np.linalg.inv(sigma[i])) * (x - mu[i])).sum(axis=1)) * phi[
                i] / np.sqrt(np.linalg.det(sigma[i]))
        ll = np.sum(np.log(w))
        w /= np.sum(w, axis=1)[:, None]
        # (2) M-step
        z_tilde_indi = np.zeros((m_tilde, K))
        for i in range(K):
            z_tilde_indi[:, i] = [1. if z_ele == i else 0 for z_ele in z]
        phi = (np.sum(w, axis=0) + alpha*z_tilde_indi.sum(axis=0)) / (m+alpha*m_tilde)
        for i in range(K):
            mu[i] = (x.T.dot(w[:, i]) + alpha*x_tilde.T.dot(z_tilde_indi[:, i])) / (np.sum(w[:, i])+alpha*np.sum(z_tilde_indi[:, i]))
            sigma[i] = (((x - mu[i]).T * w[:, i]).dot(x - mu[i]) + alpha*((x_tilde - mu[i]).T * z_tilde_indi[:, i]).dot(x_tilde - mu[i])) \
                       / (np.sum(w[:, i])+alpha*np.sum(z_tilde_indi[:, i]))
        it += 1

    print(f'Number of iterations:{it}')
    # *** END CODE HERE ***

    return w


```

### f) Comparison

1. #converge_iterations

   `Running unsupervised EM algorithm...
   Number of iterations:1000
   Running semi-supervised EM algorithm...
   Number of iterations:52
   Running unsupervised EM algorithm...
   Number of iterations:1000
   Running semi-supervised EM algorithm...
   Number of iterations:58
   Running unsupervised EM algorithm...
   Number of iterations:1000
   Running semi-supervised EM algorithm...
   Number of iterations:53`

2. Stability

   ~~Semi-supervised EM is perfectly stable, whereas classical EM isn't so satisfactory.~~

   Semi-supervised EM are more stable than unsupervised EM. The assignments by unsupervised EM are random with different random initializations. But the assignments by semi-supervised EM are the same.

3. Overall quality of assignments

   ~~Classical EM performs poorly while high-variance Gaussian is mixed into dataset.~~

   The overall quality of assignments by semi-supervised EM are higher than unsupervised EM. 

   In the pictures of semi-supervised EM, there are three nearly the same low-variance Gaussian distributions, and a high-variance Gaussian distribution.

   In the pictures of unsupervised EM, there are four Gaussian distributions which variances are different.

***p03_gmm.py***

```python
import matplotlib.pyplot as plt
import numpy as np
import os

PLOT_COLORS = ['red', 'green', 'blue', 'orange']  # Colors for your plots
K = 4           # Number of Gaussians in the mixture model
NUM_TRIALS = 3  # Number of trials to run (can be adjusted for debugging)
UNLABELED = -1  # Cluster label for unlabeled data points (do not change)


def main(is_semi_supervised, trial_num):
    """Problem 3: EM for Gaussian Mixture Models (unsupervised and semi-supervised)"""
    print('Running {} EM algorithm...'
          .format('semi-supervised' if is_semi_supervised else 'unsupervised'))

    # Load dataset
    train_path = os.path.join('..', 'data', 'ds3_train.csv')
    x, z = load_gmm_dataset(train_path)
    x_tilde = None

    if is_semi_supervised:
        # Split into labeled and unlabeled examples
        labeled_idxs = (z != UNLABELED).squeeze()
        x_tilde = x[labeled_idxs, :]   # Labeled examples
        z = z[labeled_idxs, :]         # Corresponding labels
        x = x[~labeled_idxs, :]        # Unlabeled examples

    # *** START CODE HERE ***
    # (1) Initialize mu and sigma by splitting the m data points uniformly at random
    # into K groups, then calculating the sample mean and covariance for each group
    # Split data
    m, n = x.shape
    row_indices = np.random.permutation(m)
    partitionLen = int(m / K)
    # initialize mu and sigma
    mu, sigma = np.zeros((K, n)), np.zeros((K, n, n))
    for i in range(K):
        x_samp = x[row_indices[i * partitionLen:(i + 1) * partitionLen]]
        mu[i] = np.mean(x_samp, axis=0)
        sigma[i] = np.cov(x_samp.T)
    # (2) Initialize phi to place equal probability on each Gaussian
    # phi should be a numpy array of shape (K,)
    phi = 1 / K * np.ones(K)
    # (3) Initialize the w values to place equal probability on each Gaussian
    # w should be a numpy array of shape (m, K)
    w = 1 / K * np.ones((m, K))
    # *** END CODE HERE ***

    if is_semi_supervised:
        w = run_semi_supervised_em(x, x_tilde, z, w, phi, mu, sigma)
    else:
        w = run_em(x, w, phi, mu, sigma)

    # Plot your predictions
    z_pred = np.zeros(m)
    if w is not None:  # Just a placeholder for the starter code
        for i in range(m):
            z_pred[i] = np.argmax(w[i])

    plot_gmm_preds(x, z_pred, is_semi_supervised, plot_id=trial_num)


def run_em(x, w, phi, mu, sigma):
    """Problem 3(d): EM Algorithm (unsupervised).

    See inline comments for instructions.

    Args:
        x: Design matrix of shape (m, n).
        w: Initial weight matrix of shape (m, k).
        phi: Initial mixture prior, of shape (k,).
        mu: Initial cluster means, list of k arrays of shape (n,).
        sigma: Initial cluster covariances, list of k arrays of shape (n, n).

    Returns:
        Updated weight matrix of shape (m, k) resulting from EM algorithm.
        More specifically, w[i, j] should contain the probability of
        example x^(i) belonging to the j-th Gaussian in the mixture.
    """
    m, n = x.shape
    # No need to change any of these parameters
    eps = 1e-3  # Convergence threshold
    max_iter = 1000

    # Stop when the absolute change in log-likelihood is < eps
    # See below for explanation of the convergence criterion
    it = 0
    ll = prev_ll = None
    while it < max_iter and (prev_ll is None or np.abs(ll - prev_ll) >= eps):
        # Just a placeholder for the starter code
        # *** START CODE HERE
        # (1) E-step: Update your estimates in w
        # (2) M-step: Update the model parameters phi, mu, and sigma
        # (3) Compute the log-likelihood of the data to check for convergence.
        # By log-likelihood, we mean `ll = sum_x[log(sum_z[p(x|z) * p(z)])]`.
        # We define convergence by the first iteration where abs(ll - prev_ll) < eps.
        # Hint: For debugging, recall part (a). We showed that ll should be monotonically increasing.

        # (1) E-step:
        # it += 1
        prev_ll = ll
        for i in range(K):
            w[:, i] = np.exp(-0.5 * ((x - mu[i]).dot(np.linalg.inv(sigma[i]))*(x - mu[i])).sum(axis=1))*phi[i] / np.sqrt(np.linalg.det(sigma[i]))
        ll = np.sum(np.log(w))
        w /= np.sum(w, axis=1)[:, None]
        # (2) M-step
        phi = np.sum(w, axis=0)/m
        for i in range(K):
            mu[i] = x.T.dot(w[:, i]) / np.sum(w[:, i])
            sigma[i] = ((x-mu[i]).T*w[:, i]).dot(x-mu[i]) / np.sum(w[:, i])
        it += 1
        # *** END CODE HERE ***
    print(f'Number of iterations:{it}')
    return w


def run_semi_supervised_em(x, x_tilde, z, w, phi, mu, sigma):
    """Problem 3(e): Semi-Supervised EM Algorithm.

    See inline comments for instructions.

    Args:
        x: Design matrix of unlabeled examples of shape (m, n).
        x_tilde: Design matrix of labeled examples of shape (m_tilde, n).
        z: Array of labels of shape (m_tilde, 1).
        w: Initial weight matrix of shape (m, k).
        phi: Initial mixture prior, of shape (k,).
        mu: Initial cluster means, list of k arrays of shape (n,).
        sigma: Initial cluster covariances, list of k arrays of shape (n, n).

    Returns:
        Updated weight matrix of shape (m, k) resulting from semi-supervised EM algorithm.
        More specifically, w[i, j] should contain the probability of
        example x^(i) belonging to the j-th Gaussian in the mixture.
    """
    # No need to change any of these parameters
    alpha = 20.  # Weight for the labeled examples
    eps = 1e-3   # Convergence threshold
    max_iter = 1000

    m, n = x.shape
    m_tilde = x_tilde.shape[0]
    # Stop when the absolute change in log-likelihood is < eps
    # See below for explanation of the convergence criterion
    it = 0
    ll = prev_ll = None
    while it < max_iter and (prev_ll is None or np.abs(ll - prev_ll) >= eps):
        pass  # Just a placeholder for the starter code
        # *** START CODE HERE ***
        # (1) E-step: Update your estimates in w
        # (2) M-step: Update the model parameters phi, mu, and sigma
        # (3) Compute the log-likelihood of the data to check for convergence.
        # Hint: Make sure to include alpha in your calculation of ll.
        # Hint: For debugging, recall part (a). We showed that ll should be monotonically increasing.
        # (1) E-step:
        # it += 1
        prev_ll = ll
        for i in range(K):
            w[:, i] = np.exp(-0.5 * ((x - mu[i]).dot(np.linalg.inv(sigma[i])) * (x - mu[i])).sum(axis=1)) * phi[
                i] / np.sqrt(np.linalg.det(sigma[i]))
        ll = np.sum(np.log(w))
        w /= np.sum(w, axis=1)[:, None]
        # (2) M-step
        z_tilde_indi = np.zeros((m_tilde, K))
        for i in range(K):
            z_tilde_indi[:, i] = [1. if z_ele == i else 0 for z_ele in z]
        phi = (np.sum(w, axis=0) + alpha*z_tilde_indi.sum(axis=0)) / (m+alpha*m_tilde)
        for i in range(K):
            mu[i] = (x.T.dot(w[:, i]) + alpha*x_tilde.T.dot(z_tilde_indi[:, i])) / (np.sum(w[:, i])+alpha*np.sum(z_tilde_indi[:, i]))
            sigma[i] = (((x - mu[i]).T * w[:, i]).dot(x - mu[i]) + alpha*((x_tilde - mu[i]).T * z_tilde_indi[:, i]).dot(x_tilde - mu[i])) \
                       / (np.sum(w[:, i])+alpha*np.sum(z_tilde_indi[:, i]))
        it += 1

    print(f'Number of iterations:{it}')
    # *** END CODE HERE ***

    return w


# *** START CODE HERE ***
# Helper functions
# *** END CODE HERE ***


def plot_gmm_preds(x, z, with_supervision, plot_id):
    """Plot GMM predictions on a 2D dataset `x` with labels `z`.

    Write to the output directory, including `plot_id`
    in the name, and appending 'ss' if the GMM had supervision.

    NOTE: You do not need to edit this function.
    """
    plt.figure(figsize=(12, 8))
    plt.title('{} GMM Predictions'.format('Semi-supervised' if with_supervision else 'Unsupervised'))
    plt.xlabel('x_1')
    plt.ylabel('x_2')

    for x_1, x_2, z_ in zip(x[:, 0], x[:, 1], z):
        color = 'gray' if z_ < 0 else PLOT_COLORS[int(z_)]
        alpha = 0.25 if z_ < 0 else 0.75
        plt.scatter(x_1, x_2, marker='.', c=color, alpha=alpha)

    file_name = 'p03_pred{}_{}.pdf'.format('_ss' if with_supervision else '', plot_id)
    save_path = os.path.join('output', file_name)
    plt.savefig(save_path)


def load_gmm_dataset(csv_path):
    """Load dataset for Gaussian Mixture Model (problem 3).

    Args:
         csv_path: Path to CSV file containing dataset.

    Returns:
        x: NumPy array shape (m, n)
        z: NumPy array shape (m, 1)

    NOTE: You do not need to edit this function.
    """

    # Load headers
    with open(csv_path, 'r') as csv_fh:
        headers = csv_fh.readline().strip().split(',')

    # Load features and labels
    x_cols = [i for i in range(len(headers)) if headers[i].startswith('x')]
    z_cols = [i for i in range(len(headers)) if headers[i] == 'z']

    x = np.loadtxt(csv_path, delimiter=',', skiprows=1, usecols=x_cols, dtype=float)
    z = np.loadtxt(csv_path, delimiter=',', skiprows=1, usecols=z_cols, dtype=float)

    if z.ndim == 1:
        z = np.expand_dims(z, axis=-1)

    return x, z


if __name__ == '__main__':
    np.random.seed(229)
    # Run NUM_TRIALS trials to see how different initializations
    # affect the final predictions with and without supervision
    for t in range(NUM_TRIALS):
        main(is_semi_supervised=False, trial_num=t)

        # *** START CODE HERE ***
        # Once you've implemented the semi-supervised version,
        # uncomment the following line.
        # You do not need to add any other lines in this code block.
        # main(with_supervision=True, trial_num=t)
        main(is_semi_supervised=True, trial_num=t)
        # *** END CODE HERE ***

```

## Problem 5 K-means for compression

### a) K-Means Compression Implementation

<img src="https://gitee.com/violets/typora--images/raw/main/imgs/202205032141519.png" alt="image-20220503214107405" style="zoom: 50%;" />

<center>compressed image</center>



### b) Compression factor

~~Compression factor is approximately `2`~~

In the original image, we need 3 8=24 bits to represent a pixel. 

In the compressed image, we only need 4 bits (16 colors) to represent a pixel. 

So the image are compressed by factor 6.

*p05_kmeans.py*

```python
from matplotlib.image import imread
import matplotlib.pyplot as plt
import numpy as np

K = 16


# Helper functions
def rgb_img_vectorization(img_matrix):
    """

    Args:
        img_matrix: a list(size: [l,l,3]) represent a square image

    Returns: vectorized numpy array (size : [l**2,3] ) represent the image

    """
    return np.array(img_matrix).reshape(len(img_matrix) ** 2, 3)


# K-means
def k_means_rgbimg(x, mu):
    """
    Compress an image (use K colors instead).
    Args:
        x: unlabelled dataset of size (m,3)
        mu: cluster centroids size (K, 3)

    Returns: updated mu

    """
    m, n = x.shape
    k = mu.shape[0]
    max_iter = 1000

    prev_c = None
    it = 0
    norm_mu_x = np.zeros((m, k))
    c = np.zeros(m)
    while it < max_iter and (not (prev_c == c).all()):
        prev_c = c
        for i in range(K):
            norm_mu_x[:, i] = np.linalg.norm(mu[i] - x, axis=1)
        c = np.argmin(norm_mu_x, axis=1)
        c_indi = np.zeros((m, K))
        for i in range(K):
            c_indi[:, i] = [1. if c_ele == i else 0 for c_ele in c]
            mu[i] = x.T.dot(c_indi[:, i]) / np.sum(c_indi[:, i])
        it += 1
    print(f'Number of iterations:{it}')
    new_img = np.zeros(x.shape)
    for i in range(m):
        new_img[i] = mu[c[i]]
    img_array = new_img.reshape((int(np.sqrt(m + 1)), int(np.sqrt(m + 1)), 3))
    plt.imshow(img_array.astype(int))
    plt.show()
    return None


# Initialize mu to randomly chosen pixel in the image
A = imread('../data/peppers-large.tiff')
mu = np.zeros((K, 3))
x_t = rgb_img_vectorization(A)

idx = np.random.randint(0, len(A)**2, size=K)
for i in range(K):
    mu[i] = x_t[idx[i], :]

k_means_rgbimg(x_t, mu)

```

# PS 4

## Problem 1 CNN for MNIST

### a) Backward functions

---

**FYI :**

[Softmax in neural network](https://e2eml.school/softmax.html)

- Softmax is responsible for properly backpropagating the loss gradient so that upstream layers can learn from it.

---

**backward softmax**

```python
def backward_softmax(x, grad_outputs):
    """
    Compute the gradient of the loss with respect to x.

    grad_outputs is the gradient of the loss with respect to the outputs of the softmax.

    Args:
        x: A 1d numpy float array of shape number_of_classes
        grad_outputs: A 1d numpy float array of shape number_of_classes

    Returns:
        A 1d numpy float array of the same shape as x with the derivative of the loss with respect to x
    """

    # *** START CODE HERE ***
    # FYI: the input gradient is the output gradient multiplied by the softmax derivative
    # Calculate Softmax derivative
    s = forward_softmax(x)
    soft_deri = np.diag(s) - np.outer(s, s)
    return soft_deri.dot(grad_outputs)
    # *** END CODE HERE ***

```

**backward ReLU**

```python
def backward_relu(x, grad_outputs):
    """
    Compute the gradient of the loss with respect to x

    Args:
        x: A numpy array of arbitrary shape containing the input.
        grad_outputs: A numpy array of the same shape of x containing the gradient of the loss with respect
            to the output of relu

    Returns:
        A numpy array of the same shape as x containing the gradients with respect to x.
    """

    # *** START CODE HERE ***
    return (x > 0) * grad_outputs
    # *** END CODE HERE ***

```

**backward cross entropy loss**

```python
def backward_cross_entropy_loss(probabilities, labels):
    """
    Compute the gradient of the cross entropy loss with respect to the probabilities.

    probabilities is of the shape (# classes)
    labels is of the shape (# classes)

    The output should be the gradient with respect to the probabilities.

    Returns:
        The gradient of the loss with respect to the probabilities.
    """

    # *** START CODE HERE ***
    # loss_grad = np.zeros(probabilities.shape)
    # for i, label in enumerate(labels):
    #     if label == 1:
    #         loss_grad[i] = -1/probabilities[i]
    # return loss_grad
    return - labels/probabilities
    # *** END CODE HERE ***

```

**backward linear**

```python
def backward_linear(weights, bias, data, output_grad):
    """
    Compute the gradients of the loss with respect to the parameters of a linear layer.

    See forward_linear for information about the shapes of the variables.

    output_grad is the gradient of the loss with respect to the output of this layer.

    This should return a tuple with three elements:
    - The gradient of the loss with respect to the weights
    - The gradient of the loss with respect to the bias
    - The gradient of the loss with respect to the data
    """

    # *** START CODE HERE ***
    return np.outer(data, output_grad), output_grad, weights.dot(output_grad)
    # *** END CODE HERE ***
```

**backward convolution**

```python
def backward_convolution(conv_W, conv_b, data, output_grad):
    """
    Compute the gradient of the loss with respect to the parameters of the convolution.

    See forward_convolution for the sizes of the arguments.
    output_grad is the gradient of the loss with respect to the output of the convolution.

    Returns:
        A tuple containing 3 gradients.
        The first element is the gradient of the loss with respect to the convolution weights
        The second element is the gradient of the loss with respect to the convolution bias
        The third element is the gradient of the loss with respect to the input data
    """

    # *** START CODE HERE ***
    conv_channels, _, conv_width, conv_height = conv_W.shape

    input_channels, input_width, input_height = data.shape

    grad_bias = output_grad.sum(axis=(1, 2))
    grad_weight = np.zeros(conv_W.shape)
    grad_data = np.zeros(data.shape)

    for x in range(input_width - conv_width + 1):
        for y in range(input_height - conv_height + 1):
            for output_channel in range(conv_channels):
                grad_weight[output_channel, :, :, :] += data[:, x:(x + conv_width), y:(y + conv_height)] * output_grad[output_channel, x, y]
                grad_data[:, x:(x + conv_width), y:(y + conv_height)] += conv_W[output_channel, :, :, :] * output_grad[output_channel, x, y]

    return grad_weight, grad_bias, grad_data
    # *** END CODE HERE ***

```

**backward max pool**

```python
def backward_max_pool(data, pool_width, pool_height, output_grad):
    """
    Compute the gradient of the loss with respect to the data in the max pooling layer.

    data is of the shape (# channels, width, height)
    output_grad is of shape (# channels, width // pool_width, height // pool_height)

    output_grad is the gradient of the loss with respect to the output of the backward max
    pool layer.

    Returns:
        The gradient of the loss with respect to the data (of same shape as data)
    """
    
    # *** START CODE HERE ***
    input_channels, input_width, input_height = data.shape
    grad_data = np.zeros(data.shape)

    for i in range(input_channels):
        for x in range(0, input_width, pool_width):
            for y in range(0, input_height, pool_height):
                # Need the index of the max entry
                # Solution 1
                temp_index = data[i, x:(x + pool_width), y:(y + pool_height)].argmax()
                grad_data[i, x:(x + pool_width), y:(y + pool_height)].flat[temp_index] += output_grad[
                    i, x // pool_width, y // pool_height]
                # Solution 2
                # temp_matrix = data[i, x:(x + pool_width), y:(y + pool_height)]
                # grad_data[i, x:(x + pool_width), y:(y + pool_height)][i, np.unravel_index(temp_matrix.argmax(), temp_matrix.shape)] += output_grad[
                #     i, x // pool_width, y // pool_height]

    return grad_data
    # *** END CODE HERE ***

```

### b) Backward Propagation

```python
def backward_prop(data, labels, params):
    """
    Implement the backward propagation gradient computation step for a neural network
    
    Args:
        data: A numpy array containing the input for a single example
        labels: A 1d numpy array containing the labels for a single example
        params: A dictionary mapping parameter names to numpy arrays with the parameters.
            This numpy array will contain W1, b1, W2, and b2
            W1 and b1 represent the weights and bias for the convolutional layer
            W2 and b2 represent the weights and bias for the output layer of the network

    Returns:
        A dictionary of strings to numpy arrays where each key represents the name of a weight
        and the values represent the gradient of the loss with respect to that weight.
        
        In particular, it should have 4 elements:
            W1, W2, b1, and b2
    """

    # *** START CODE HERE ***
    W1 = params['W1']
    b1 = params['b1']
    W2 = params['W2']
    b2 = params['b2']

    first_convolution = forward_convolution(W1, b1, data)
    first_max_pool = forward_max_pool(first_convolution, MAX_POOL_SIZE, MAX_POOL_SIZE)
    first_after_relu = forward_relu(first_max_pool)
    flattened = np.reshape(first_after_relu, (-1))
    logits = forward_linear(W2, b2, flattened)
    y = forward_softmax(logits)

    dc_dp = backward_cross_entropy_loss(y, labels)
    d_softmax = backward_softmax(logits, dc_dp)
    d_W2, d_b2, dlin_dlinear = backward_linear(W2, b2, flattened, d_softmax)
    d_relu = backward_relu(first_max_pool, dlin_dlinear.reshape(first_max_pool.shape))
    d_maxpool = backward_max_pool(first_convolution, MAX_POOL_SIZE, MAX_POOL_SIZE, d_relu)
    d_W1, d_b1, _ = backward_convolution(W1, b1, data, d_maxpool)

    grad_dict = {'W1': d_W1,
                 'b1': d_b1,
                 'W2': d_W2,
                 'b2': d_b2}

    return grad_dict
    # *** END CODE HERE ***

```

## Problem 2 Off-policy evaluation for MDPs

### a) Importance Sampling Estimator

Wanted equation is equal to


$$
\begin{aligned}
\mathbb{E}_{\substack{s \sim p(s) \\ a \sim \pi_{0}(s, a)}} \frac{\pi_{1}(s, a)}{\hat{\pi}_{0}(s, a)} R(s, a) &= \sum_{(s,a)} p(s,a) \frac{\pi_{1}(s, a)}{\hat{\pi}_{0}(s, a)} R(s, a)\\
&= \sum_{(s,a)} p(s) \pi_{0}(s, a) \frac{\pi_{1}(s, a)}{\hat{\pi}_{0}(s, a)} R(s, a)
\end{aligned}
$$

Besides, we have $\hat \pi _0 = \pi _0$, thus
$$
\begin{aligned}
\mathbb{E}_{\substack{s \sim p(s) \\ a \sim \pi_{0}(s, a)}} \frac{\pi_{1}(s, a)}{\hat{\pi}_{0}(s, a)} R(s, a) &= \sum_{(s,a)} p(s) \pi_{0}(s, a) \frac{\pi_{1}(s, a)}{{\pi}_{0}(s, a)} R(s, a) \\
&= \sum_{(s,a)} p(s)  {\pi_{1}(s, a)} R(s, a)\\
&= \mathbb{E}_{\substack{s \sim p(s) \\ a \sim \pi_{1}(s, a)}} R(s, a)
\end{aligned}
$$

### b) Weighted Importance Sampling

When $\hat \pi _0 = \pi _0$, we have
$$
\begin{aligned}
\frac{\mathbb{E}_{\substack{s \sim p(s) \\ a \sim \pi_{0}(s, a)}} \frac{\pi_{1}(s, a)}{\hat{\pi}_{0}(s, a)} R(s, a)}{\mathbb{E}_{\substack{s \sim p(s) \\ a \sim \pi_{0}(s, a)}} \frac{\pi_{1}(s, a)}{\hat{\pi}_{0}(s, a)}} &= \frac{\sum_{(s,a)} p(s)  {\pi_{1}(s, a)} R(s, a)}{\sum_{(s,a)} p(s)  {\pi_{1}(s, a)}} \\
&= \frac{\sum_{(s,a)} p(s)  {\pi_{1}(s, a)} R(s, a)}{\sum_{(s,a)} p(s,a)} \\
&= \sum_{(s,a)} p(s)  {\pi_{1}(s, a)} R(s, a) \\
&= \mathbb{E}_{\substack{s \sim p(s) \\ a \sim \pi_{1}(s, a)}} R(s, a)
\end{aligned}
$$

### c) Weighted importance sampling estimator is biased

In our assumption, we have
$$
\begin{aligned}
\frac{\mathbb{E}_{\substack{s \sim p(s) \\ a \sim \pi_{0}(s, a)}} \frac{\pi_{1}(s, a)}{\hat{\pi}_{0}(s, a)} R(s, a)}{\mathbb{E}_{\substack{s \sim p(s) \\ a \sim \pi_{0}(s, a)}} \frac{\pi_{1}(s, a)}{\hat{\pi}_{0}(s, a)}} &= \mathbb{E}_{\substack{s \sim p(s) \\ a \sim \pi_{1}(s, a)}} R(s, a)
\end{aligned}
$$
If there's only one data element in observational dataset, we have
$$
\begin{aligned}
\frac{\mathbb{E}_{\substack{s \sim p(s) \\ a \sim \pi_{0}(s, a)}} \frac{\pi_{1}(s, a)}{\hat{\pi}_{0}(s, a)} R(s, a)}{\mathbb{E}_{\substack{s \sim p(s) \\ a \sim \pi_{0}(s, a)}} \frac{\pi_{1}(s, a)}{\hat{\pi}_{0}(s, a)}} &= \frac{\sum_{(s,a)} p(s,a) \frac{\pi_{1}(s, a)}{\hat{\pi}_{0}(s, a)} R(s, a)}{\sum_{(s,a)} p(s,a) \frac{\pi_{1}(s, a)}{\hat{\pi}_{0}(s, a)}} \\
&= \frac{p(s,a) \frac{\pi_{1}(s, a)}{\hat{\pi}_{0}(s, a)} R(s, a)}{p(s,a) \frac{\pi_{1}(s, a)}{\hat{\pi}_{0}(s, a)}} \\
&= R(s, a) \\
&= \mathbb{E}_{\substack{s \sim p(s) \\ a \sim \pi_{0}(s, a)}} R(s, a)
\end{aligned}
$$
If $\pi_1 \ne \pi_0$,
$$
\mathbb{E}_{\substack{s \sim p(s) \\ a \sim \pi_{0}(s, a)}} R(s, a) \ne \mathbb{E}_{\substack{s \sim p(s) \\ a \sim \pi_{1}(s, a)}} R(s, a)
$$


### d) Doubly Robust

**1)** When $\hat \pi _0 = \pi _0$, we have $\hat R (s,a) = R (s,a)$, thus
$$
\begin{aligned}
\mathbb{E}_{\substack{s \sim p(s) \\ a \sim \pi_{0}(s, a)}}\left(\left(\mathbb{E}_{a \sim \pi_{1}(s, a)} \hat{R}(s, a)\right)+\frac{\pi_{1}(s, a)}{\hat{\pi}_{0}(s, a)}(R(s, a)-\hat{R}(s, a))\right) 
&= {\mathbb{E}_{\substack{s \sim p(s) \\ a \sim \pi_{0}(s, a)}} \left(\mathbb{E}_{a \sim \pi_{1}(s, a)} \hat R(s, a)\right)}+{\mathbb{E}_{\substack{s \sim p(s) \\ a \sim \pi_{0}(s, a)}} \frac{\pi_{1}(s, a)}{{\pi}_{0}(s, a)}R(s, a)}\\
&-{\mathbb{E}_{\substack{s \sim p(s) \\ a \sim \pi_{0}(s, a)}} \frac{\pi_{1}(s, a)}{{\pi}_{0}(s, a)}\hat R(s, a)} \\
&= {\mathbb{E}_{\substack{s \sim p(s) \\ a \sim \pi_{1}(s, a)}} \hat R(s, a)} + {\mathbb{E}_{\substack{s \sim p(s) \\ a \sim \pi_{0}(s, a)}} \frac{\pi_{1}(s, a)}{{\pi}_{0}(s, a)}R(s, a)} \\
&- {\mathbb{E}_{\substack{s \sim p(s) \\ a \sim \pi_{1}(s, a)}} \hat R(s, a)} \\
&= {\mathbb{E}_{\substack{s \sim p(s) \\ a \sim \pi_{0}(s, a)}} \frac{\pi_{1}(s, a)}{{\pi}_{0}(s, a)}R(s, a)}\\
&= \mathbb{E}_{\substack{s \sim p(s) \\ a \sim \pi_{1}(s, a)}} R(s, a)
\end{aligned}
$$
**2)** When $\hat R (s,a) = R (s,a)$, thus
$$
\begin{aligned}
\mathbb{E}_{\substack{s \sim p(s) \\ a \sim \pi_{0}(s, a)}}\left(\left(\mathbb{E}_{a \sim \pi_{1}(s, a)} \hat{R}(s, a)\right)+\frac{\pi_{1}(s, a)}{\hat{\pi}_{0}(s, a)}(R(s, a)-\hat{R}(s, a))\right) 
&= \mathbb{E}_{\substack{s \sim p(s) \\ a \sim \pi_{0}(s, a)}} \left(\mathbb{E}_{a \sim \pi_{1}(s, a)} R(s, a)\right)\\
&= \frac{\mathbb{E}_{\substack{s \sim p(s) \\ a \sim \pi_{0}(s, a)}} \frac{\pi_{1}(s, a)}{\hat{\pi}_{0}(s, a)} R(s, a)}{\mathbb{E}_{\substack{s \sim p(s) \\ a \sim \pi_{0}(s, a)}} \frac{\pi_{1}(s, a)}{\hat{\pi}_{0}(s, a)}}\\
&= \mathbb{E}_{\substack{s \sim p(s) \\ a \sim \pi_{1}(s, a)}} R(s, a)\\
\end{aligned}
$$

### e) Choice between estimator

**i.**  Use importance sampling estimator. Because the interaction $R(s,a)$ is very complicated, and importance sampling estimator only need to model drug assignment policy $\pi _0$ using $R(s,a)$ as observation data. 

**ii.** Use regression estimator. Because interaction $R(s,a)$ is very simple. And for regression estimator, if $\hat R(s,a) = R(s,a)$, the estimator is trivially correct.

## Problem 3 ==PCA==

$$
\begin{aligned}
f_{u}(x) & = \arg \min _{v \in \mathcal{V}}\|x-v\|^{2} =\frac{uu^T x}{u^T u} = uu^T x \\ \\
\arg \min _{u: u^{T} u=1} \sum_{i=1}^{m}\left\|x^{(i)}-f_{u}\left(x^{(i)}\right)\right\|_{2}^{2} &=\arg \min _{u: u^{T} u=1} \sum_{i=1}^{m}\left\|x^{(i)}-u u^{T} x^{(i)}\right\|_{2}^{2} \\
&=\arg \min _{u: u^{T} u=1} \sum_{i=1}^{m}\left(x^{(i)}-u u^{T} x^{(i)}\right)^{T}\left(x^{(i)}-u u^{T} x^{(i)}\right) \\
&=\arg \min _{u: u^{T} u=1} \sum_{i=1}^{m} x^{(i)^{T}} x^{(i)}-x^{(i)^{T}} u u^{T} x^{(i)} \\
&=\arg \max _{u: u^{T} u=1} \sum_{i=1}^{m} x^{(i)^{T}} u u^{T} x^{(i)} \\
&=\arg \max _{u: u^{T} u=1} \sum_{i=1}^{m} u^{T} x^{(i)} x^{(i)^{T}} u \\
&=\arg \max _{u: u^{T} u=1} u^{T}\left(\sum_{i=1}^{m} x^{(i)} x^{(i)^{T}}\right) u
\end{aligned}
$$

## Problem 4  Independent Components Analysis

### a) Gaussian source

When $g\prime$ is standard normal distribution, we have
$$
\begin{aligned}
\ell (W) &= \sum_{i=1}^n \left(\log |W| + \sum_{j=1}^d \log {\frac{1}{\sqrt{2\pi}} \exp \frac{- \left(w_j^T x^{(i)} \right)^2}{2}} \right) \\
\nabla_W \ell &= n{(W^{-1})^T} + \left(\sum_{i=1}^n \nabla_W\sum_{j=1}^d \frac{- \left(w_j^T x^{(i)} \right)^2}{2} \right) \\
&=  n{(W^{-1})^T} + \sum_{i=1}^n W x^{(i)} x^{(i)T} \\
&=  n{(W^{-1})^T} - W X X^T \\
&= 0 \\ \\

n{(W^{-1})^T} &= W X X^T \\
W^T W &= n X^T X
\end{aligned}
$$
Let $u$ be an arbitrary orthogonal matrix, and let $W ^\prime = RW$. Then
$$
{W ^\prime}^T W ^\prime =  W^T R^T R W = W^T W
$$
i.e. if $W$ is a solution, any $W^ \prime$ is also a solution.

### b) ==Laplace source==

Derive update for $W$ when $s_i \sim \mathcal L (0,1)$
$$
\begin{aligned}
\nabla_{W} \ell(W) &=\nabla_{W}\left(\log |W|+\sum_{j=1}^{d} \log \frac{1}{2} \exp \left(-\left|w_{j}^{T} x^{(i)}\right|\right)\right) \\
&=\left(W^{-1}\right)^{T}-\nabla_{W} \sum_{j=1}^{d}\left|w_{j}^{T} x^{(i)}\right| \\
&=\left(W^{T}\right)^{-1}-\operatorname{sign}\left(W x^{(i)}\right) x^{(i)^{T}} \\
W &:=W+\alpha\left(\left(W^{T}\right)^{-1}-\operatorname{sign}\left(W x^{(i)}\right) x^{(i)^{T}}\right)
\end{aligned}
$$

*<u>Wrong Solution</u>*
$$
\begin{aligned}
\ell (W) &= \sum_{i=1}^n \left(\log |W| + \sum_{j=1}^d \log {\frac{1}{2} \exp \left({- w_j^T x^{(i)}} \right) }\right) \\
\nabla_W \ell &= n{(W^{-1})^T} + \left(\sum_{i=1}^n \nabla_W\sum_{j=1}^d {- w_j^T x^{(i)}} \right) \\
&= n{(W^{-1})^T} - nX \\
\\
W &:= W + \alpha n \left({(W^{-1})^T} - X \right)
\end{aligned}
$$

### c) Cocktail Party Problem

```python
def update_W(W, x, learning_rate):
    """
    Perform a gradient ascent update on W using data element x and the provided learning rate.

    This function should return the updated W.

    Use the laplace distribiution in this problem.

    Args:
        W: The W matrix for ICA
        x: A single data element
        learning_rate: The learning rate to use

    Returns:
        The updated W
    """
    
    # *** START CODE HERE ***
    updated_W = W + learning_rate * (np.linalg.inv(W.T) - np.outer(np.sign(W.dot(x)), x.T))
    # *** END CODE HERE ***

    return updated_W


def unmix(X, W):
    """
    Unmix an X matrix according to W using ICA.

    Args:
        X: The data matrix
        W: The W for ICA

    Returns:
        A numpy array S containing the split data
    """

    S = np.zeros(X.shape)


    # *** START CODE HERE ***
    S = X.dot(W.T)
    # *** END CODE HERE ***

    return S

```

## Problem 5 Markov decision processes

### a) 

$$
\begin{aligned}
\left\|B\left(V_{1}\right)-B\left(V_{2}\right)\right\|_{\infty} &=\gamma\left\|\max _{a \in A} \sum_{s^{\prime} \in S} P_{s a}\left(s^{\prime}\right)\left[V_{1}\left(s^{\prime}\right)-V_{2}\left(s^{\prime}\right)\right]\right\|_{\infty} \\
&=\gamma \max _{s^{\prime} \in S}\left|\max _{a \in A} \sum_{s^{\prime} \in S} P_{s a}\left(s^{\prime}\right)\left[V_{1}\left(s^{\prime}\right)-V_{2}\left(s^{\prime}\right)\right]\right| \\
& \leq \gamma\left\|V_{1}-V_{2}\right\|_{\infty}
\end{aligned}
$$

The inequality holds because for any $\alpha,x \in \mathbb R ^n$, if $\sum_i \alpha_i = 1$ and $\alpha_i \ge 0$, then $\sum_i \alpha_i x_i \le \max_i x_i$   

### b)

Assume there are two fixed points $V_1,V_2$, i.e. $B\left(V_{1}\right)=V_{1}, B\left(V_{2}\right)=V_{2}$
$$
\begin{array}{c}
\left\|V_{1}-V_{2}\right\|_{\infty} = \left\|B\left(V_{1}\right)-B\left(V_{2}\right)\right\|_{\infty} \leq \gamma\left\|V_{1}-V_{2}\right\|_{\infty} \\
\left\|V_{1}-V_{2}\right\|_{\infty} = 0 \\
V_{1} = V_{2}
\end{array}
$$
So $B$ have at most one fixed point.

## Problem 6 Reinforcement Learning: The inverted pendulum

- `simulate()` function for simulating the pole dynamics
- `get_state()` for discretizing
- `show_cart()` for display
- `NUM_STATES` 
- `time_steps_to_failure`  records the time for which the pole was balanced before each failure is in memory
- `num_failures`  stores the number of failures (pole drops / cart out of bounds) till now.

:warning: 

- Update the transition counts and rewards observed after each simulation cycle

```python
def choose_action(state, mdp_data):
    """
    Choose the next action (0 or 1) that is optimal according to your current
    mdp_data. When there is no optimal action, return a random action.

    Args:
        state: The current state in the MDP
        mdp_data: The parameters for your MDP. See initialize_mdp_data.

    Returns:
        0 or 1 that is optimal according to your current MDP
    """

    # *** START CODE HERE ***
    expected_value = mdp_data['value'].dot(mdp_data['transition_probs'][state])
    if expected_value[0] == expected_value[1]:
        return np.random.randint(2)
    elif expected_value[0] > expected_value[1]:
        return 0
    else:
        return 1

    # Plan B

    # else:
    #     return np.argmax(expect_value)

    # *** END CODE HERE ***

def update_mdp_transition_counts_reward_counts(mdp_data, state, action, new_state, reward):
    """
    Update the transition count and reward count information in your mdp_data.
    Do not change the other MDP parameters (those get changed later).

    Record the number of times `state, action, new_state` occurs.
    Record the rewards for every `new_state`
    (since rewards are -1 or 0, you just need to record number of times reward -1 is seen in 'reward_counts' index new_state,0)
    Record the number of time `new_state` was reached (in 'reward_counts' index new_state,1)

    Args:
        mdp_data: The parameters of your MDP. See initialize_mdp_data.
        state: The state that was observed at the start.
        action: The action you performed.
        new_state: The state after your action.
        reward: The reward after your action (i.e. reward corresponding to new_state).

    Returns:
        Nothing
    """

    # *** START CODE HERE ***
    mdp_data['transition_counts'][state, new_state, action] += 1

    mdp_data['reward_counts'][new_state, 1] += 1

    if reward != 0:
        mdp_data['reward_counts'][new_state, 0] += 1
    # *** END CODE HERE ***

    # This function does not return anything
    return


def update_mdp_transition_probs_reward(mdp_data):
    """
    Update the estimated transition probabilities and reward values in your MDP.

    Make sure you account for the case when a state-action pair has never
    been tried before, or the state has never been visited before. In that
    case, you must not change that component (and thus keep it at the
    initialized uniform distribution).

    Args:
        mdp_data: The data for your MDP. See initialize_mdp_data.

    Returns:
        Nothing

    """

    # *** START CODE HERE ***
    transition_counts = mdp_data['transition_counts']
    num_counts = transition_counts.sum(axis=1)
    for i in range(mdp_data['num_states']):
        for a in range(2):
            if num_counts[i, a] != 0:
                mdp_data['transition_probs'][i, :, a] = transition_counts[i, :, a] / num_counts[i, a]

    for state in range(mdp_data['num_states']):
        if mdp_data['reward_counts'][state][1] != 0:
            mdp_data['reward'][state] = - mdp_data['reward_counts'][state][0] / mdp_data['reward_counts'][state][1]
    # *** END CODE HERE ***

    # This function does not return anything
    return

def update_mdp_value(mdp_data, tolerance, gamma):
    """
    Update the estimated values in your MDP.

    Perform value iteration using the new estimated model for the MDP.
    The convergence criterion should be based on `TOLERANCE` as described
    at the top of the file.

    Return true if it converges within one iteration.
    
    Args:
        mdp_data: The data for your MDP. See initialize_mdp_data.
        tolerance: The tolerance to use for the convergence criterion.
        gamma: Your discount factor.

    Returns:
        True if the value iteration converged in one iteration

    """

    # *** START CODE HERE ***
    iters = 0

    while True:
        iters += 1

        value = mdp_data['value']
        new_value = mdp_data['reward'] + gamma * value.dot(mdp_data['transition_probs']).max(axis=1)
        mdp_data['value'] = new_value

        if np.max(np.abs(value - new_value)) < tolerance:
            break

    return iters == 1
    # *** END CODE HERE ***

```

