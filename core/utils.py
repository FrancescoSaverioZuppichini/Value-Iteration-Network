from random import randint
import seaborn as sns
import matplotlib.pyplot as plt

def torch_imshow(tensor, title):
    plt.title(title)
    img = tensor.detach().squeeze().cpu().numpy()
    sns.heatmap(img)


def get_random_data(ds, device, idx=None):
    idx  = randint(0, len(ds)) if idx == None else idx
    test = ds[0]
    (labels, s1, s2, obs) = test

    return labels.unsqueeze(0).to(device).long(), \
                      s1.unsqueeze(0).to(device), \
                      s2.unsqueeze(0).to(device), \
                      obs.unsqueeze(0).to(device)

def make_images(obs, r_img, v):
    fig = plt.figure()

    plt.subplot(2, 2, 1)
    torch_imshow(obs[0][0].squeeze(), 'world')

    plt.subplot(2, 2, 2)
    torch_imshow(obs[0][1].squeeze(), 'r')

    plt.subplot(2, 2, 3)
    torch_imshow(r_img[0].squeeze(), 'r from vin')

    plt.subplot(2, 2, 4)
    torch_imshow(v[0], 'V')

    fig.show()