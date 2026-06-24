from utils.bvh_parser import parse_all_bvh_to_npz


if __name__ == "__main__":

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