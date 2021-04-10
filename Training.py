import torch
from HelperFunctions import *
import matplotlib.pyplot as plt


def train_network(network, train_loader, val_loader,
                  optimizer, loss_fun, epochs=50, save_every=1, plot_graph=False,
                  device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')):

    epochs = epochs
    save_every = save_every
    all_error = np.zeros(0)
    all_val_error = np.zeros(0)
    mean_error = np.zeros(0)
    device = device

    for epoch in range(epochs):

        # enumerate can be used to output iteration index i, as well as the data
        for i, item in enumerate(train_loader, 0):
            network.train()

            data, labels = item[DataEnum.Image], item[DataEnum.Label]

            data = data.to(device)
            labels = labels.to(device)
            batch_size_train = data.size()[0]

            # clear the gradient
            optimizer.zero_grad()

            # feed the input and acquire the output from network
            outputs = network(data)

            # calculating the predicted and the expected loss
            loss = loss_fun(outputs, labels)

            # compute the gradient
            loss.backward()

            # update the parameters
            optimizer.step()

            if i % save_every == 0 and i < len(val_loader):
                network.train(False)
                error = loss.item()
                all_error = np.append(all_error, error)

            # if i % save_every == 0:
            network.train(False)

            # print statistics
            ce_loss = loss.item()
            if i % 10 == 0:
                print('[%d, %5d] loss: %.3f' %
                      (epoch + 1, i + 1, ce_loss))

        network.train(False)
        for i, item in enumerate(val_loader, 0):
            data, labels = item[DataEnum.Image], item[DataEnum.Label]
            data = data.to(device)
            labels = labels.to(device)
            batch_size_val = data.size()[0]

            pred = network(data)

            loss = loss_fun(pred, labels)

            if i % save_every == 0:
                error = loss.item()
                mean_error = np.append(mean_error, error)
                all_val_error = np.append(all_val_error, np.mean(mean_error))

            # print(torch.stack((pred[:3], labels[:3]), dim=1))

        epochs_train = np.arange(0, all_error.size)
        epochs_val = np.arange(0, all_val_error.size)

    if plot_graph:
        plt.plot(epochs_train, all_error, label='Train error')
        plt.show()
        plt.plot(epochs_val, all_val_error, label='Val error')
        plt.show()