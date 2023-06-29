import torch


class NeRF(torch.nn.Module):
    def __init__(self, filter_size=128, encoding=6) -> None:
        super(NeRF, self).__init__()
        self.layer1 = torch.nn.Linear(3+3*2*encoding, filter_size)
        self.layer2 = torch.nn.Linear(filter_size, filter_size)
        self.layer3 = torch.nn.Linear(filter_size, 4)
        self.relu = torch.nn.functional.relu
        self.float()

    def forward(self, input_values):
        x = input_values.float()
        x = self.relu(self.layer1(x))
        x = self.relu(self.layer2(x))
        output_values = self.layer3(x)
        return output_values
