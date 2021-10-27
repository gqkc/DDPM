import argparse
import datetime
import torch
import wandb

from torch.utils.data import DataLoader
from torchvision import datasets
from ddpm import script_utils
from ddpm.script_utils import TransformDataset
from torchvision import transforms


def main():
    args = create_argparser().parse_args()
    device = args.device

    try:
        diffusion = script_utils.get_diffusion_from_args(args).to(device)
        optimizer = torch.optim.Adam(diffusion.parameters(), lr=args.learning_rate)

        if args.log_to_wandb:
            if args.project_name is None:
                raise ValueError("args.log_to_wandb set to True but args.project_name is None")

            run = wandb.init(
                project=args.project_name,
                entity='gkqc',
                config=vars(args),
                name=args.run_name,
            )
            wandb.watch(diffusion)

        batch_size = args.batch_size
        # load vq-vae model
        if args.model_path != None:
            model = torch.load(args.model_path, map_location=device)
            model.eval()

        # load datasets
        train_dataset = torch.load(args.train_dataset_path, map_location=device)
        val_dataset = torch.load(args.val_dataset_path, map_location=device)
        if args.rescale == "normalize":
            full_train = next(iter(DataLoader(train_dataset, batch_size=len(train_dataset))))[0]
            train_mean, train_std = full_train.mean(), full_train.std()
            transform = transforms.Compose([transforms.Normalize(train_mean, train_std)])
            # load datasets
            train_dataset = TransformDataset(train_dataset, transform=transform)
            val_dataset = TransformDataset(val_dataset, transform=transform)

        train_loader = script_utils.cycle(DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            drop_last=True,
            num_workers=0,
        ))
        test_loader = DataLoader(val_dataset, batch_size=batch_size, drop_last=True, num_workers=0)

        acc_train_loss = 0

        for iteration in range(1, args.iterations + 1):
            print(iteration)
            diffusion.train()

            x, y = next(train_loader)
            x = x.detach().to(device).permute(0, 3, 1, 2)
            y = y.to(device)

            if args.use_labels:
                loss = diffusion(x, y)
            else:
                loss = diffusion(x)

            acc_train_loss += loss.item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            diffusion.update_ema()

            if iteration % args.log_rate == 0:
                test_loss = 0
                with torch.no_grad():
                    diffusion.eval()
                    for x, y in test_loader:
                        x = x.detach().to(device).permute(0, 3, 1, 2)
                        y = y.to(device)

                        if args.use_labels:
                            loss = diffusion(x, y)
                        else:
                            loss = diffusion(x)

                        test_loss += loss.item()

                model_filename = f"{args.log_dir}/{args.project_name}-{args.run_name}-iteration-{iteration}-model.pth"
                optim_filename = f"{args.log_dir}/{args.project_name}-{args.run_name}-iteration-{iteration}-optim.pth"

                torch.save(diffusion.state_dict(), model_filename)
                torch.save(optimizer.state_dict(), optim_filename)

                if args.use_labels:
                    samples = diffusion.sample(10, device, y=torch.arange(10, device=device))
                else:
                    samples = diffusion.sample(10, device)

                img_samples = model.decode(samples.argmax(1))
                test_loss /= len(test_loader)
                acc_train_loss /= args.log_rate

                wandb.log({
                    "test_loss": test_loss,
                    "train_loss": acc_train_loss,
                    "samples": [wandb.Image(sample) for sample in img_samples],
                })

                acc_train_loss = 0

        if args.log_to_wandb:
            run.finish()
    except KeyboardInterrupt:
        if args.log_to_wandb:
            run.finish()
        print("Keyboard interrupt, run finished early")


def create_argparser():
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    run_name = datetime.datetime.now().strftime("ddpm-%Y-%m-%d-%H-%M")
    defaults = dict(
        learning_rate=2e-4,
        batch_size=128,
        iterations=1000,

        log_to_wandb=True,
        log_rate=100,
        log_dir="ddpm_logs",
        project_name="cifar",
        run_name=run_name,
        channels=64,
        device=device,
        rescale="normalize",
        model_path=None,
        train_dataset_path=None,
        val_dataset_path=None,
        img_size=8,
        base_channels=32
    )
    defaults.update(script_utils.diffusion_defaults())

    parser = argparse.ArgumentParser()
    script_utils.add_dict_to_argparser(parser, defaults)
    return parser


if __name__ == "__main__":
    main()
