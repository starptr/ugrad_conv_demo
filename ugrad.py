import torch

if __name__ == "__main__":
    # a 2D tensor of shape (1, 1, height, width)
    # first 2 entries of shape is count and channels
    xvel0 = torch.tensor([[[
          [0.0, 1.0, 2.0, 3.0, 4.0],
          [1.0, 2.0, 3.0, 4.0, 5.0],
          [2.0, 3.0, 4.0, 5.0, 6.0],
          [3.0, 4.0, 5.0, 6.0, 7.0],
          [4.0, 5.0, 6.0, 7.0, 8.0],
        ]]])
    # a 2D tensor of shape (1, 1, height, width)
    # first 2 entries of shape is out_channels and in_channels
    kernel = torch.tensor([[[
        [-1.0, 1.0],
        [-1.0, 1.0],
        ]]])

    ugrad = torch.nn.functional.conv2d(xvel0, kernel)
    print(ugrad)
