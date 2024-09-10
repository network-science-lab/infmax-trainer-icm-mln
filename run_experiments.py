"""Main entrypoint to the experiments."""

# TODO: consider adding runners and defaulf configs for each method
# TODO: change print statements to logs
import argparse
import yaml

from misc.utils import set_seed


def parse_args(*args):
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "config",
        help="Experiment config file (default: config.yaml).",
        type=str,
    )
    return parser.parse_args(*args)


if __name__ == "__main__":

    # uncomment for debugging
    args = parse_args(["_configs/mn2vkma.yaml"])

    # comment this line while debugging
    # args = parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    if random_seed := config["train"].get("random_seed"):
        print(f"Setting randomness seed as {random_seed}!")
        set_seed(config["train"]["random_seed"])
    print(f"Loaded config: {config}")

    from trainers.train import train
    train(config)
