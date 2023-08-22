import torch
import kornia

@torch.jit.script
def pyrdown(input_tensor: torch.Tensor, num_scales: int = 4):
    """ Creates a downscale pyramid for the input tensor. """
    output = [input_tensor]
    for _ in range(num_scales - 1):
        down = kornia.filters.blur_pool2d(output[-1], 3)
        output.append(down)
    return output


def batched_trace(mat_bNN):
    return mat_bNN.diagonal(offset=0, dim1=-1, dim2=-2).sum(-1)