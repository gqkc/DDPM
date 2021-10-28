import argparse
import torchvision
import torch.nn.functional as F
import torch

from .unet import UNet
from .diffusion import (
    GaussianDiffusion,
    generate_linear_schedule,
    generate_cosine_schedule,
)


def cycle(dl):
    """
    https://github.com/lucidrains/denoising-diffusion-pytorch/
    """
    while True:
        for data in dl:
            yield data


class TransformDataset(torch.utils.data.Dataset):
    def __init__(self, dataset=None, transform=None):
        self.dataset = dataset
        self.transform = transform

    def __getitem__(self, index):
        data, target = self.dataset[index]

        if self.transform is not None:
            data = self.transform(data)
        return data, target

    def __len__(self):
        return self.dataset.__len__()


def get_transform():
    class RescaleChannels(object):
        def __call__(self, sample):
            return 2 * sample - 1

    return torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        RescaleChannels(),
    ])


class RescaleChannels(object):
    def __call__(self, sample):
        return 2 * sample - 1


class PermuteDetach(object):
    def __call__(self, sample):
        return sample.detach().permute(2, 0, 1)


class Exp(object):
    def __call__(self, sample):
        return sample.exp()


def get_transform_exp_mean(mean):
    class Mean(object):
        def __call__(self, sample):
            return (sample - mean)

    return torchvision.transforms.Compose([Exp(), Mean(), PermuteDetach()])


def get_transform_exp():
    return torchvision.transforms.Compose([Exp(), PermuteDetach()])


def get_transform_soft(mean):
    class Soft(object):
        def __call__(self, sample):
            return sample.softmax(-1) - mean

    return torchvision.transforms.Compose([Soft(), PermuteDetach()])


def get_transform_exp_minmax(min, max):
    class Minmax(object):
        def __call__(self, sample):
            return (sample - min) / (max - min)

    return torchvision.transforms.Compose([Exp(), Minmax(), PermuteDetach()])


def get_transform_norm(mean, std):
    class Norm(object):
        def __call__(self, sample):
            return (sample - mean) / std

    return torchvision.transforms.Compose([Exp(), Norm(), PermuteDetach()])


def get_transform_minmax(min, max):
    class Minmax(object):
        def __call__(self, sample):
            return (sample - min) / (max - min)

    return torchvision.transforms.Compose([Minmax(), RescaleChannels(), PermuteDetach()])


def str2bool(v):
    """
    https://stackoverflow.com/questions/15008758/parsing-boolean-values-with-argparse
    """
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("boolean value expected")


def add_dict_to_argparser(parser, default_dict):
    """
    https://github.com/openai/improved-diffusion/blob/main/improved_diffusion/script_util.py
    """
    for k, v in default_dict.items():
        v_type = type(v)
        if v is None:
            v_type = str
        elif isinstance(v, bool):
            v_type = str2bool
        parser.add_argument(f"--{k}", default=v, type=v_type)


def diffusion_defaults():
    defaults = dict(
        # num_timesteps=1000,
        schedule="linear",
        loss_type="l2",
        use_labels=False,

        # base_channels=128,
        # channel_mults=(1, 2, 2),
        num_res_blocks=2,
        time_emb_dim=128 * 4,
        norm="gn",
        dropout=0.1,
        activation="silu",
        attention_resolutions=(1,),

        ema_decay=0.9999,
        ema_update_rate=1,
    )

    return defaults


def get_diffusion_from_args(args):
    activations = {
        "relu": F.relu,
        "mish": F.mish,
        "silu": F.silu,
    }

    model = UNet(
        img_channels=args.channels,

        base_channels=args.base_channels,
        channel_mults=args.channel_mults,
        time_emb_dim=args.time_emb_dim,
        norm=args.norm,
        dropout=args.dropout,
        activation=activations[args.activation],
        attention_resolutions=args.attention_resolutions,

        num_classes=None if not args.use_labels else 10,
        initial_pad=0,
    )

    if args.schedule == "cosine":
        betas = generate_cosine_schedule(args.num_timesteps)
    else:
        betas = generate_linear_schedule(
            args.num_timesteps,
            1e-4 * 1000 / args.num_timesteps,
            0.02 * 1000 / args.num_timesteps,
        )

    diffusion = GaussianDiffusion(
        model, (args.img_size, args.img_size), args.channels, 10,
        betas,
        ema_decay=args.ema_decay,
        ema_update_rate=args.ema_update_rate,
        ema_start=2000,
        loss_type=args.loss_type,
    )

    return diffusion
