import torch
from torchvision import datasets, transforms


def main():
    batch_size = 32
    test_batch_size = 32

    dataset_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    train_dataset = datasets.MNIST('../data', train=True, download=True, transform=dataset_transform)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    test_dataset = datasets.MNIST('../data', train=False, download=True, transform=dataset_transform)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=test_batch_size, shuffle=True)


if __name__ == '__main__':
    main()
