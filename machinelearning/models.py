import nn

class PerceptronModel(object):
    def __init__(self, dimensions):
        """
        Initialize a new Perceptron instance.

        A perceptron classifies data points as either belonging to a particular
        class (+1) or not (-1). `dimensions` is the dimensionality of the data.
        For example, dimensions=2 would mean that the perceptron must classify
        2D points.
        """
        self.w = nn.Parameter(1, dimensions)

    def get_weights(self):
        """
        Return a Parameter instance with the current weights of the perceptron.
        """
        return self.w

    def run(self, x):
        """
        Calculates the score assigned by the perceptron to a data point x.

        Inputs:
            x: a node with shape (1 x dimensions)
        Returns: a node containing a single number (the score)
        """
        return nn.DotProduct(x, self.w)

    def get_prediction(self, x):
        """
        Calculates the predicted class for a single data point `x`.

        Returns: 1 or -1
        """
        return 1.0 if nn.as_scalar(nn.DotProduct(x, self.w)) >= 0 else -1.0

    def train(self, dataset):
        """
        Train the perceptron until convergence.
        """
        conv = False
        while conv == False:
            conv = True
            for f, l in dataset.iterate_once(1):
                # print(self.get_prediction(f), nn.as_scalar(l))
                if self.get_prediction(f) != nn.as_scalar(l):
                    # print("Incorrectly classified!")
                    conv = False
                    self.w.update(f, nn.as_scalar(l))

class RegressionModel(object):
    """
    A neural network model for approximating a function that maps from real
    numbers to real numbers. The network should be sufficiently large to be able
    to approximate sin(x) on the interval [-2pi, 2pi] to reasonable precision.
    """
    def __init__(self):
        # Initialize your model parameters here
        self.w1 = 16
        self.w2 = 8

        self.m1 = nn.Parameter(1, self.w1)
        self.m2 = nn.Parameter(self.w1, self.w2)
        self.m3 = nn.Parameter(self.w2, 1)

        self.b1 = nn.Parameter(1, self.w1)
        self.b2 = nn.Parameter(1, self.w2)
        self.b3 = nn.Parameter(1, 1)

        self.t = 0.005
        self.b = 20

    def run(self, x):
        """
        Runs the model for a batch of examples.

        Inputs:
            x: a node with shape (batch_size x 1)
        Returns:
            A node with shape (batch_size x 1) containing predicted y-values
        """
        x = layer(x, self.m1, self.b1, True)
        x = layer(x, self.m2, self.b2, True)
        return layer(x, self.m3, self.b3, False)

    def get_loss(self, x, y):
        """
        Computes the loss for a batch of examples.

        Inputs:
            x: a node with shape (batch_size x 1)
            y: a node with shape (batch_size x 1), containing the true y-values
                to be used for training
        Returns: a loss node
        """
        pred = self.run(x)
        return nn.SquareLoss(pred, y)

    def train(self, dataset):
        """
        Trains the model.
        """
        i = 0
        sum_loss = 0
        for f, l in dataset.iterate_forever(self.b):
            # print("Iter: ", i)
            pred = self.run(f)
            loss = self.get_loss(f, l)

            step_loss = nn.as_scalar(loss)
            # print("Loss: ", step_loss)

            # Keep track of the avg loss
            sum_loss += step_loss

            if (i == 9):
                if (sum_loss / 10 < 0.019):
                    return
                i = 0
                sum_loss = 0
            else:
                i += 1

            # Loss gradient over parameters
            grad = nn.gradients(loss, [self.m1, self.m2, self.m3, \
                self.b1, self.b2, self.b3])

            # Update parameters
            self.m1.update(grad[0], -self.t)
            self.m2.update(grad[1], -self.t)
            self.m3.update(grad[2], -self.t)
            self.b1.update(grad[3], -self.t)
            self.b2.update(grad[4], -self.t)
            self.b3.update(grad[5], -self.t)

class DigitClassificationModel(object):
    """
    A model for handwritten digit classification using the MNIST dataset.

    Each handwritten digit is a 28x28 pixel grayscale image, which is flattened
    into a 784-dimensional vector for the purposes of this model. Each entry in
    the vector is a floating point number between 0 and 1.

    The goal is to sort each digit into one of 10 classes (number 0 through 9).

    (See RegressionModel for more information about the APIs of different
    methods here. We recommend that you implement the RegressionModel before
    working on this part of the project.)
    """
    def __init__(self):
        # Initialize your model parameters here
        self.w1 = 200
        # self.w3 = 64

        self.m1 = nn.Parameter(784, self.w1)
        self.m2 = nn.Parameter(self.w1, 10)
        # self.m3 = nn.Parameter(self.w2, 10)
        # self.m4 = nn.Parameter(self.w3, 10)

        self.b1 = nn.Parameter(1, self.w1)
        self.b2 = nn.Parameter(1, 10)
        # self.b3 = nn.Parameter(1, 10)
        # self.b4 = nn.Parameter(1, 10)

        self.t = 0.5
        self.b = 50

    def run(self, x):
        """
        Runs the model for a batch of examples.

        Your model should predict a node with shape (batch_size x 10),
        containing scores. Higher scores correspond to greater probability of
        the image belonging to a particular class.

        Inputs:
            x: a node with shape (batch_size x 784)
        Output:
            A node with shape (batch_size x 10) containing predicted scores
                (also called logits)
        """
        x = layer(x, self.m1, self.b1, True)
        return layer(x, self.m2, self.b2, False)
        # return layer(x, self.m3, self.b3, False)
        # return layer(x, self.m4, self.b4, False)

    def get_loss(self, x, y):
        """
        Computes the loss for a batch of examples.

        The correct labels `y` are represented as a node with shape
        (batch_size x 10). Each row is a one-hot vector encoding the correct
        digit class (0-9).

        Inputs:
            x: a node with shape (batch_size x 784)
            y: a node with shape (batch_size x 10)
        Returns: a loss node
        """
        pred = self.run(x)
        return nn.SoftmaxLoss(pred, y)

    def train(self, dataset):
        """
        Trains the model.
        """
        i = 0
        j = 1
        for f, l in dataset.iterate_forever(self.b):
            # print("Iter: ", i)
            pred = self.run(f)
            loss = self.get_loss(f, l)

            # print("Loss: ", nn.as_scalar(loss))

            if (i % (60000/self.b) == 0 and i > 0):
                va = dataset.get_validation_accuracy()
                print("Epoch ", j, " Accuracy: ", va)
                if (j >= 6):
                    return
                self.t /= 3
                j += 1
            i += 1

            # Loss gradient over parameters
            grad = nn.gradients(loss, [self.m1, self.m2, \
                self.b1, self.b2])

            # Update parameters
            self.m1.update(grad[0], -self.t)
            self.m2.update(grad[1], -self.t)
            # self.m3.update(grad[2], -self.t)
            # self.m4.update(grad[3], -self.t)

            self.b1.update(grad[2], -self.t)
            self.b2.update(grad[3], -self.t)
            # self.b3.update(grad[5], -self.t)
            # self.b4.update(grad[7], -self.t)

class LanguageIDModel(object):
    """
    A model for language identification at a single-word granularity.

    (See RegressionModel for more information about the APIs of different
    methods here. We recommend that you implement the RegressionModel before
    working on this part of the project.)
    """
    def __init__(self):
        # Our dataset contains words from five different languages, and the
        # combined alphabets of the five languages contain a total of 47 unique
        # characters.
        # You can refer to self.num_chars or len(self.languages) in your code
        self.num_chars = 47
        self.languages = ["English", "Spanish", "Finnish", "Dutch", "Polish"]

        # RNN Parameters

        self.w1 = 64

        self.m1 = nn.Parameter(47, self.w1)
        self.mh = nn.Parameter(self.w1, self.w1)

        self.b1 = nn.Parameter(1, self.w1)

        # Final NN
        self.w2 = 64
        self.tw = 5

        self.m3 = nn.Parameter(self.w1, self.w2)
        self.m4 = nn.Parameter(self.w2, self.tw)

        self.b3 = nn.Parameter(1, self.w2)
        self.b4 = nn.Parameter(1, self.tw)

        self.t = 0.5
        self.b = 100


    def run(self, xs):
        """
        Runs the model for a batch of examples.

        Although words have different lengths, our data processing guarantees
        that within a single batch, all words will be of the same length (L).

        Here `xs` will be a list of length L. Each element of `xs` will be a
        node with shape (batch_size x self.num_chars), where every row in the
        array is a one-hot vector encoding of a character. For example, if we
        have a batch of 8 three-letter words where the last word is "cat", then
        xs[1] will be a node that contains a 1 at position (7, 0). Here the
        index 7 reflects the fact that "cat" is the last word in the batch, and
        the index 0 reflects the fact that the letter "a" is the inital (0th)
        letter of our combined alphabet for this task.

        Your model should use a Recurrent Neural Network to summarize the list
        `xs` into a single node of shape (batch_size x hidden_size), for your
        choice of hidden_size. It should then calculate a node of shape
        (batch_size x 5) containing scores, where higher scores correspond to
        greater probability of the word originating from a particular language.

        Inputs:
            xs: a list with L elements (one per character), where each element
                is a node with shape (batch_size x self.num_chars)
        Returns:
            A node with shape (batch_size x 5) containing predicted scores
                (also called logits)
        """
        h = layer(xs[0], self.m1, self.b1, True)

        for x in xs[1:]:
            t1 = nn.Linear(x, self.m1)
            t2 = nn.Linear(h, self.mh)
            h = nn.Add(t1, t2)
            h = nn.ReLU(h)

        r = layer(h, self.m3, self.b3, True)
        return layer(r, self.m4, self.b4, False)

    def get_loss(self, xs, y):
        """
        Computes the loss for a batch of examples.

        The correct labels `y` are represented as a node with shape
        (batch_size x 5). Each row is a one-hot vector encoding the correct
        language.

        Inputs:
            xs: a list with L elements (one per character), where each element
                is a node with shape (batch_size x self.num_chars)
            y: a node with shape (batch_size x 5)
        Returns: a loss node
        """
        pred = self.run(xs)
        return nn.SoftmaxLoss(pred, y)

    def train(self, dataset):
        """
        Trains the model.
        """
        i = 0
        for f, l in dataset.iterate_forever(self.b):
            # print("Iter: ", i)
            pred = self.run(f)
            loss = self.get_loss(f, l)

            # print("Loss: ", nn.as_scalar(loss))

            if (i % 1000 == 0):
                if (dataset.get_validation_accuracy() >= 0.82):
                    return
                self.t /= 2
            i += 1

            # Loss gradient over parameters
            grad = nn.gradients(loss, [self.m1, self.mh, self.m3, self.m4, \
                self.b1, self.b3, self.b4])

            # Update parameters
            self.m1.update(grad[0], -self.t)
            self.mh.update(grad[1], -self.t)
            self.m3.update(grad[2], -self.t)
            self.m4.update(grad[3], -self.t)

            self.b1.update(grad[4], -self.t)
            self.b3.update(grad[5], -self.t)
            self.b4.update(grad[6], -self.t)

# Translates features x into features at the next layer,
#   applying nonlinearity if n is true
def layer(x, m, b, n):
    xm = nn.Linear(x, m)
    # print(xm)
    rv = nn.AddBias(xm, b)
    if n:
        rv = nn.ReLU(rv)

    # print(rv)
    return rv

