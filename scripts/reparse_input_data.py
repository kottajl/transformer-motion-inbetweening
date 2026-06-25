import argparse

from utils.bvh_parser import parse_all_bvh_to_npz
from utils.utils import set_seed


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=None, help='Random seed for reproducibility (default: None)')
    args = parser.parse_args()
    
    if args.seed is not None:
        set_seed(args.seed)
        print(f"Random seed set to: {args.seed}")

    # Train
    success, errors = parse_all_bvh_to_npz(
        bvh_dir="datasets/lafan1/raw",
        npz_dir="datasets/lafan1/processed",
        mode="train",
        clear_npzs=True
    )
    print(f"Train dataset: success: {len(success)}, error: {len(errors)}: {errors}")

    # Test
    success, errors = parse_all_bvh_to_npz(
        bvh_dir="datasets/lafan1/raw",
        npz_dir="datasets/lafan1/test_processed",
        mode="test",
        clear_npzs=True
    )
    print(f"Test dataset: success: {len(success)}, error: {len(errors)}: {errors}")