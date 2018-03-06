from __future__ import print_function
from itertools import count
import torch
import torch.autograd
import torch.nn.functional as F
from torch.autograd import Variable
import deepdish as dee
import numpy as np
from pypublish.common_fig import *

data = dee.io.load('data2.hdf5')
confs = data['X']
kappa = data['y']

def poly_desc(W, b):
    """Creates a string description of a polynomial."""
    result = 'y = '
    for i, w in enumerate(W):
        result += '{:+.2f} x^{} '.format(w, len(W) - i)
    result += '{:+.2f}'.format(b[0])
    return result

def get_batch_test(batch_size=32):
    """Builds a batch i.e. (x, f(x)) pair."""
    random = torch.randn(batch_size)
    indexes = np.random.permutation(int(len(kappa)/2))[0:batch_size]+int(len(kappa)/2)
    y = kappa[indexes]
    x = confs[indexes]
    x = torch.from_numpy(np.array(x,dtype=np.float32))
    y = torch.from_numpy(np.array(y,dtype=np.float32))
    return Variable(x), Variable(y)


def get_batch(batch_size=32):
    """Builds a batch i.e. (x, f(x)) pair."""
    random = torch.randn(batch_size)
    indexes = np.random.permutation(int(len(kappa)/2))[0:batch_size]
    y = kappa[indexes]
    x = confs[indexes]
    x = torch.from_numpy(np.array(x,dtype=np.float32))
    y = torch.from_numpy(np.array(y,dtype=np.float32))
    return Variable(x), Variable(y)


# Define model

ni = len(confs[0])

nu = ni

nu = 64
fc = torch.nn.Sequential(
            torch.nn.Linear(ni, nu),
            torch.nn.ReLU(False),
            torch.nn.Linear(nu, nu),
            torch.nn.ReLU(False),
            torch.nn.Linear(nu, 1),
            torch.nn.ReLU(False))

batch_x, batch_y = get_batch()
output = F.smooth_l1_loss(fc(batch_x), batch_y)
loss = output.data[0]

initial_loss = loss

loss_vec = []
for batch_idx in count(1):
    # Get data
    batch_x, batch_y = get_batch()

    # Reset gradients
    fc.zero_grad()

    # Forward pass
    output = F.smooth_l1_loss(fc(batch_x), batch_y)
    loss = output.data[0]
    loss_vec.append(loss)
    # Backward pass
    output.backward()

    # Apply gradients
    for param in fc.parameters():
        param.data.add_(-0.1 * param.grad.data)
    # Stop criterion
    if loss < 1e-4:
        break

print('Loss: {:.6f} after {} batches'.format(loss, batch_idx))

init_plotting(extra_x_padding = 0.05)

loss_cv = []
for n in range(50):
 batch_x, batch_y = get_batch_test()
 output = F.smooth_l1_loss(fc(batch_x), batch_y)
 loss = output.data[0]
 loss_cv.append(loss)

v = 1.0
plot(range(len(loss_vec)),np.array(loss_vec)/v,color=c1)
plot(np.array(range(len(loss_cv)))+len(loss_vec),np.array(loss_cv)/v,color=c2)
xlabel('Epochs')
ylabel('Error')
legend(['Training','Testing'])
yscale('log')
savefigure('error.png')
show()


