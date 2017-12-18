import numpy as np

def create_data2D(num_points = 200, data_type = 'linear', coef_a = 1, coef_b = 2, radius = 5):
    """Create 2D data with two classes. Data types are: linear, circle and check-board.
       Change coef_a, coef_b for linear data, and radius for circle and check-board data."""
    X = np.random.normal(0, 10, size=(num_points,2))
    Y = []
    for x in X:
        if data_type == 'linear':
            if x[0]*coef_a + x[1]*coef_b > 0:
                Y.append([0, 1])
            else:
                Y.append([1, 0])

        elif data_type == 'check-board':
            if x[0]*x[1] > radius:
                Y.append([0, 1])
            else:
                Y.append([1, 0])

        elif data_type == 'circle':
            if (x[0]**2 + x[1]**2)**0.5 > radius:
                Y.append([0, 1])
            else:
                Y.append([1, 0])

    Y = np.array(Y)
    return X,Y


def get_plot_data(min_v=-5, max_v=5, step=1):
    " Create mesh grid data used to plot the NN output"
    x,y = np.mgrid[min_v:max_v:step, min_v:max_v:step]
    return np.vstack((x.flatten(), y.flatten())).T


def shuffle_and_get_batch_data(X_data, Y_data, batch_size):
    "Shuffle data and return a random subset with batch_size length"
    assert len(X_data) == len(Y_data), "X_data and Y_data must have the same length"

    # Create an array with random indexes to select from X_data and Y_data
    rand_index = np.random.choice(len(X_data), batch_size)
    return X_data[rand_index], Y_data[rand_index]
