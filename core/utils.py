import matplotlib.pyplot as plt

def torch_imshow(tensor, title):
    fig = plt.figure()
    plt.title(title)
    img = tensor.detach().squeeze().cpu().numpy()
    plt.imshow(img)
    fig.show()