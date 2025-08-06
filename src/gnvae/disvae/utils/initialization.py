import torch
from torch import nn
from typing import Union, Optional


def get_activation_name(activation: Union[str, nn.Module, None]) -> str:
    """
    Get the standardized name of the activation function given either
    a string identifier or a PyTorch activation module instance.

    Parameters
    ----------
    activation : str or nn.Module or None
        Activation name or instance.

    Returns
    -------
    str
        Activation name compatible with `torch.nn.init.calculate_gain`.

    Raises
    ------
    ValueError
        If activation is unknown.
    """
    if activation is None:
        return "linear"

    if isinstance(activation, str):
        return activation.lower()

    mapper = {
        nn.LeakyReLU: "leaky_relu",
        nn.ReLU: "relu",
        nn.Tanh: "tanh",
        nn.Sigmoid: "sigmoid",
        nn.Softmax: "softmax"
    }

    for act_type, name in mapper.items():
        if isinstance(activation, act_type):
            return name

    raise ValueError(f"Unknown activation type: {activation}")


def get_gain(activation: Union[str, nn.Module, None]) -> float:
    """
    Get the appropriate initialization gain for a given activation.

    Parameters
    ----------
    activation : str or nn.Module or None
        Activation name or instance.

    Returns
    -------
    float
        Gain factor for weight initialization.
    """
    param = None
    activation_name = get_activation_name(activation)
    if activation_name == "leaky_relu" and isinstance(activation, nn.LeakyReLU):
        param = activation.negative_slope
    elif activation_name == "leaky_relu" and isinstance(activation, str):
        param = 0.01

    return nn.init.calculate_gain(activation_name, param)


def linear_init(layer: nn.Linear,
                activation: Union[str, nn.Module, None] = "relu") -> None:
    """
    Initialize the weights and bias of a linear layer according to the activation.

    Parameters
    ----------
    layer : nn.Linear
        The linear layer to initialize.
    activation : str or nn.Module or None
        Activation function after this layer (used to set gain). Defaults to 'relu'.
    """
    if not isinstance(layer, nn.Linear):
        raise TypeError('linear_init expects an nn.Linear layer')

    bias = layer.bias
    weight = layer.weight

    if activation is None or activation == "linear":
        nn.init.xavier_uniform_(weight)
    else:
        activation_name = get_activation_name(activation)
        if activation_name == "leaky_relu":
            a = 0.01 if isinstance(activation, str) else getattr(activation, 'negative_slope', 0.01)
            nn.init.kaiming_uniform_(weight, a=a, nonlinearity='leaky_relu')
        elif activation_name == "relu":
            nn.init.kaiming_uniform_(weight, nonlinearity='relu')
        elif activation_name in ["sigmoid", "tanh"]:
            gain = get_gain(activation)
            nn.init.xavier_uniform_(weight, gain=gain)
        else:
            # Default fallback to xavier uniform
            nn.init.xavier_uniform_(weight)

    if bias is not None:
        nn.init.zeros_(bias)


def conv_init(layer: nn.Conv2d,
              activation: Union[str, nn.Module, None] = "relu") -> None:
    """
    Initialize the weights and bias of a convolutional layer according to the
    activation.

    Parameters
    ----------
    layer : nn.Conv2d or nn.Conv1d or nn.Conv3d
        The conv layer to initialize.
    activation : str or nn.Module or None
        Activation function after this layer (used to set gain). Defaults to 'relu'.
    """
    bias = layer.bias
    weight = layer.weight

    if activation is None or activation == "linear":
        nn.init.xavier_uniform_(weight)
    else:
        activation_name = get_activation_name(activation)
        if activation_name == "leaky_relu":
            a = 0.01 if isinstance(activation, str) else getattr(activation, 'negative_slope', 0.01)
            nn.init.kaiming_uniform_(weight, a=a, nonlinearity='leaky_relu')
        elif activation_name == "relu":
            nn.init.kaiming_uniform_(weight, nonlinearity='relu')
        elif activation_name in ["sigmoid", "tanh"]:
            gain = get_gain(activation)
            nn.init.xavier_uniform_(weight, gain=gain)
        else:
            nn.init.xavier_uniform_(weight)

    if bias is not None:
        nn.init.zeros_(bias)


def weights_init(module: nn.Module,
                 activation: Union[str, nn.Module, None] = "relu") -> None:
    """
    Initialize weights for a given module, dispatching to linear or
    conv initialization.

    Parameters
    ----------
    module : nn.Module
        The module to initialize.
    activation : str or nn.Module or None
        Activation function to determine gain.
    """
    if isinstance(module, nn.Linear):
        linear_init(module, activation)
    elif isinstance(module, torch.nn.modules.conv._ConvNd):
        conv_init(module, activation)
