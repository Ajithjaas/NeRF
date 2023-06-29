import torch


def positional_encoding(tensor, num_encoding_functions = 6):
    encoding = [tensor]
    frequencyBands = 2.00 ** torch.linspace(
        0.0,
        num_encoding_functions - 1,
        num_encoding_functions
    )

    for frequency in frequencyBands:
        for function in [torch.sin, torch.cos]:
            encoding.append(function(tensor * frequency))
    
    return torch.cat(encoding, dim=-1)

