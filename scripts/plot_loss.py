import matplotlib.pyplot as plt
import argparse
import logging

logging.getLogger('matplotlib.font_manager').setLevel(logging.ERROR)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--log', type=str, default="generated_models/alpha12.log")
    parser.add_argument('--no_show', action='store_true', help='Do not show the plot to user (makes sense only without "--no_save" parameter)')
    parser.add_argument('--no_save', action='store_true', help='Disable saving the plot to a file')
    args = parser.parse_args()

    with open(args.log, "r") as f:
        lines = f.readlines()
    epochs = []
    train_losses = []
    val_losses = []

    for line in lines:
        if line.startswith("Epoch"):
            parts = line.split("|")
            epoch_num = int(parts[0].split()[1])
            train_loss = float(parts[1].split()[2])
            val_loss = float(parts[2].split()[2])

            epochs.append(epoch_num)
            train_losses.append(train_loss)
            val_losses.append(val_loss)


    with plt.xkcd():
        plt.figure(figsize=(8, 6))
        plt.plot(epochs, train_losses, label="Train Loss", color='blue')
        plt.plot(epochs, val_losses, label="Val Loss", color='orange')
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend()
        if not args.no_save:
            plt.savefig(f"{args.log.replace('.log', '.png')}")
        if not args.no_show:
            plt.show()