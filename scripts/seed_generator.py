import random
import argparse
from typing import Tuple

MASTER_SEED = 4     # https://imgs.xkcd.com/comics/random_number.png
RANDOM_RANGE = (1000, 1_000_000)


def get_n_seeds(
        n: int, 
        master_seed: int = MASTER_SEED,
        random_range: Tuple[int, int] = RANDOM_RANGE
    ):
    random.seed(master_seed)
    a, b = random_range
    return [random.randint(a, b) for _ in range(n)]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-n', type=int, required=True, help='Number of generated seeds')
    parser.add_argument('--seed', type=int, default=None, help=f'Master seed for reproducibility (default: {MASTER_SEED})')
    args = parser.parse_args()

    my_seeds = get_n_seeds(args.n) if args.seed is None else get_n_seeds(args.n, args.seed)
    print(f"Seeds: {my_seeds}")
#__main__

# 4 -> [248514, 319031, 109177, 757250, 416297]