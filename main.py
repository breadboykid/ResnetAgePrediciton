import pandas as pd
import torch
from torch.utils.data import random_split
from Transforms import *
from torchvision import transforms
from Training import *
from Resnet import *
import torch.optim as optim


def format_data(data, labels):
    data_x = []
    data_y = []

    print(labels)
    # append features and target data, ordered by correct subject id for datset
    for index, row in labels.iterrows():
        subj_id = int(row['subj_id'])
        print(subj_id)
        data_x.append(data[subj_id])
        data_y.append(row['scan_ga'])

    return data_x, data_y


if __name__ == '__main__':
    data_path = 'Data'
    training_meta_path = data_path + '/Training_meta.pkl'
    features_meta_path = data_path + '/Features'

    labels = pd.read_pickle(training_meta_path)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print(f'label length is : {len(labels)}')

    features_data = load_numpy_files(features_meta_path, 'npy')
    print(type(features_data))
    data_x, data_y = format_data(features_data, labels)

    guass_noise = 50
    elas_control_points = 10
    elas_sigma = 2
    transform = transforms.Compose([
        ToTensor(),
        PermutateTransform(),
        GuassianNoiseTransform(guass_noise),
        ElasticDeformationTransform(elas_control_points, elas_sigma)
    ])

    dataset = CustomDataSet(data_x, data_y, transform=transform)

    # split in 8:2 ratio
    train_size = int(0.8 * dataset.__len__())
    val_size = dataset.__len__() - train_size

    train_dataset, val_dataset = random_split(dataset, [train_size, val_size],
                                              generator=torch.Generator().manual_seed(42))

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=2)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=val_size, shuffle=True, num_workers=2)

    # STUDENTS CODE - CREATE INSTANCE OF NETWORK & PASS TO NETWORK
    in_channels = 3
    num_features = [16, 32, 64, 96]
    strides = [2, 2, 2, 2]
    num_blocks = 2

    resnet = ResNet(ResidualBlock,num_blocks, strides, num_features, in_channels, 1)

    resnet = resnet.to(device)

    # CREATE LOSS OBJECT AND OPTIMISER
    loss_fun = nn.MSELoss()

    optimizer = optim.SGD(resnet.parameters(), lr=0.0002, momentum=0.9)

    train_network(resnet, train_loader, val_loader, optimizer, loss_fun, plot_graph=True)