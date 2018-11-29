import torch

from os import path

WORLD_8X8 = './data/gridworld_8x8.npz', (8, 8)
WORLD_16X16 = './data/gridworld_16x16.npz', (16, 16)
WORLD_28X28 = './data/gridworld_28x28.npz', (28, 28)

k_zoo = {
    WORLD_8X8: 10,
    WORLD_16X16: 20,
    WORLD_28X28: 30
}

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

get_file_name = lambda world : path.splitext(path.basename(world[0]))[0]
get_model_path = lambda world:  'model-{}.pt'.format(get_file_name(world))