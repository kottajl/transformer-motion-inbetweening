import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import logging

logging.getLogger('matplotlib.font_manager').setLevel(logging.ERROR)

'''
Script to generate plots for the metrics (l2p, l2q, npss) against hole sizes.
The script reads the results from a text file and generates plots for each metric.

Example of the expected input format in the text file:

Using default window_step: 4
DATASET: Loading 13 animations from datasets/lafan1/test_processed/...
Number of joints: 22
Operating on hole size: 2 frames (from 10 to 11 in the window)
Testing completed.
Test Results for model generated_models\beta6_sin.pt (window step: 4):
  |- l2p: 0.7177
  |- l2q: 0.0136
  |- npss: 0.0002

'''


def get_data(file_path):
    data = {
        "l2p": {},
        "l2q": {},
        "npss": {}
    }
    with open(file_path, 'r') as f:
        for line in f:
            line = line.strip()
            if line.startswith("Test Results for model"):
                # Extract the model name
                model_name = line.split()[4].split("\\")[-1].split(".")[0]  # Get the model name without path and extension
                # print(f"Extracted model name: {model_name}")  # Debug print
                if model_name not in data['l2p']:
                    data['l2p'][model_name] = []
                    data['l2q'][model_name] = []
                    data['npss'][model_name] = []

                # Read the next three lines for metrics
                l2p_line = next(f).strip()
                l2q_line = next(f).strip()
                npss_line = next(f).strip()

                # Extract metric values
                l2p_value = float(l2p_line.split(":")[1].strip())
                l2q_value = float(l2q_line.split(":")[1].strip())
                npss_value = float(npss_line.split(":")[1].strip())
                data['l2p'][model_name].append(l2p_value)
                data['l2q'][model_name].append(l2q_value)
                data['npss'][model_name].append(npss_value)
    return data


def main():
    file_path = "wyniki_dziury2.txt"
    data = get_data(file_path)

    colors = {
        "beta6_sin": "orange",
        "beta6_sin_h15": "#FFD580",
        "beta6_rot": "blue",
        "beta6_rot_h15": "#ADD8E6",
    }

    x = list(range(2, 30, 2))  # Hole sizes from 2 to 28 with step of 2

    for metric in data.keys():
        with plt.xkcd():
            plt.figure(figsize=(10, 6))
            for model_name, values in data[metric].items():
                plt.plot(x, values, label=model_name, color=colors.get(model_name, 'black'))
            plt.title(f"{metric.upper()} vs Hole Size (lower is better)")
            plt.xlabel("Hole Size (frames)")
            plt.ylabel(metric.upper())
            plt.xticks(x)
            plt.grid()
            plt.legend()
            plt.savefig(f"temp/{metric}_vs_hole_size2.png")
            plt.close()


if __name__ == "__main__":
    main()