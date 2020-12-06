
"""
    Common operations on Tensors/Variables

    Version 3 Abhinav Kumar 2019-07-01 Flipping tensors added
    Version 2 Abhinav Kumar 2019-06-11
    Version 1 Abhinav Kumar 2019-06-09
"""

import torch
from torch.autograd import Variable

def expand_one_dimension_at_end(input, dim1):
    """
        Adds a single dimension to the end of the input
        input = batch_size x 68 x d
        output= batch_size x 68 x d x dim1
    """
    input = input.unsqueeze(-1)
    input = input.expand(-1, -1, -1, dim1)

    return input

def expand_two_dimensions_at_end(input, dim1, dim2):
    """
        Adds two more dimensions to the end of the input
        input = batch_size x 68
        output= batch_size x 68 x dim1 x dim2
    """
    input = input.unsqueeze(-1).unsqueeze(-1)
    input = input.expand(-1, -1, dim1, dim2)

    return input

def generate_grid(h, w):
    """
        Generates an equally spaced grid with coordinates as integers with the
        size same as the input heatmap.

        Convention of axis:
        |----> X
        |
        |
        V Y
    """
    x = torch.linspace(0, w - 1, steps = w)
    xv = x.repeat(h, 1)

    y = torch.linspace(0, h - 1, steps = h)
    yv = y.view(-1, 1).repeat(1, w)

    return xv, yv

def get_zero_variable_like(input):
    """
        Returns a zero variable which is similar in the shape as input and 
        with the same type as input
    """
    output = get_zero_variable(input.shape, input)
    
    return output

def get_zero_variable(shape, type_like_input):
    """
        Returns a zero variable whose shape is similar to the shape but type is 
        like the variable type_like_input.
    """
    output = Variable(torch.zeros(shape))

    if type_like_input.is_cuda:
        output = output.cuda()
    
    return output

def flip180_tensor(tensor):
    """
        Flips a 2D tensor by 180 degrees. This is equivalent to the function
        np.flipud(np.fliplr(input))

        Reference
        https://github.com/pytorch/pytorch/issues/229#issuecomment-299424875
    """
    inv_idx      = torch.arange(tensor.size(0)-1, -1, -1).long()
    if tensor.is_cuda:
        inv_idx  = inv_idx.cuda()
    inv_tensor   = tensor[inv_idx]

    inv_idx      = torch.arange(tensor.size(1)-1, -1, -1).long()
    if tensor.is_cuda:
        inv_idx  = inv_idx.cuda()
    inv_tensor_2 = inv_tensor[:, inv_idx]

    return inv_tensor_2

def is_empty(tensor):
    """
        Checks for the empty tensor

        Reference
        https://discuss.pytorch.org/t/how-to-judge-an-empty-tensor/14610/2
    """

    return len(tensor.size()) == 0
