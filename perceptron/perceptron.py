import sys

# Initial weights
weights = [0.0, 0.0, 0.0]

# Test data
mat = []

# Max number of training iterations
epochs = 30

# Step function
def activate(net):

    return 1 if net >= 0 else 0

# Perceptron main function
def compute(x1, x2):

    # Sum inputs to weights and bias
    net = (x1 * weights[0]) + (x2 * weights[1]) + ((-1) * weights[2])

    # Execute the activation function on the result
    return activate(net)

# Optimize weights in case of errors
def optimizeWeights(i, output):

    weights[0] = weights[0] + (mat[i][2] - output) * mat[i][0]

    weights[1] = weights[1] + (mat[i][2] - output) * mat[i][1]

    weights[2] = weights[2] + (mat[i][2] - output) * -1

# Train your network
def training():

    count = 0

    trained = True

    while True:

        for i in range(4):

            output = compute(mat[i][0], mat[i][1])

            # Optimize weights if the output is not equal to the expected value
            if output != mat[i][2]:

                optimizeWeights(i, output)

                trained = False

        count += 1

        if trained or count >= epochs:

            break

# Predict according to a sample
def predict(sample):

    sample[2] = compute(sample[0], sample[1])

def main():

    op = input('Choose your operation [AND/OR]: ')

    global mat

    if op == 'AND':

        mat = [[0, 1, 0], \
               [1, 0, 0], \
               [0, 0, 0], \
               [1, 1, 1]]

    elif op == 'OR':

        mat = [[0, 1, 1], \
               [1, 0, 1], \
               [0, 0, 0], \
               [1, 1, 1]]

    else:

        sys.exit(1)

    print('Weights before training:')
    for i in weights:
        print(i, end=',')
    print()

    training()

    print('Weights after training:')
    for i in weights:
        print(i, end=',')
    print()

    samples = [[0, 1, -1], \
               [1, 0, -1], \
               [0, 0, -1], \
               [1, 1, -1]]

    for sample in samples:

        predict(sample)

    print('Result:')

    for sample in samples:
        for s in sample:
            print(s, end=',')
        print()

main()
