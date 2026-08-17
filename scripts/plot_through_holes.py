import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import logging

logging.getLogger('matplotlib.font_manager').setLevel(logging.ERROR)

'''
Script to generate plots for the metrics (l2p, l2q, npss) against hole sizes.
The script reads the results from a text file and generates plots for each metric.

Example of new expected input format in the text file:

...
s1_int_xxx_rot,20,5.498039,0.100148,0.013519
...
'''


def get_data(file_path):
    data = {
        "l2p": {},
        "l2q": {},
        "npss": {}
    }
    hole_sizes = []
    with open(file_path, 'r') as f:
        for line in f:
            line = line.strip()
            if line:
                parts = line.split(',')
                if len(parts) == 5:
                    model_name = parts[0].strip()
                    hole_size = int(parts[1].strip())
                    l2p_value = float(parts[2].strip())
                    l2q_value = float(parts[3].strip())
                    npss_value = float(parts[4].strip())

                    if hole_size not in hole_sizes:
                        hole_sizes.append(hole_size)

                    if model_name not in data['l2p']:
                        data['l2p'][model_name] = []
                        data['l2q'][model_name] = []
                        data['npss'][model_name] = []

                    data['l2p'][model_name].append(l2p_value)
                    data['l2q'][model_name].append(l2q_value)
                    data['npss'][model_name].append(npss_value)
    return data, hole_sizes


def data_concat(data, config_name_map: dict = None):
    '''
    from: data[metric][??_model_name] = list of values for different hole sizes
    to: data[metric][config_name] = list of averages and stds for different hole sizes
    '''

    cat_data = {}
    for metric, metric_data in data.items():
        cat_data[metric] = {}
        for model_name, values in metric_data.items():

            config_name_pos = model_name.find('_') + 1     # Remove the seed prefix (e.g., "s1_")
            config_name = model_name[config_name_pos:]

            if config_name_map is not None and config_name in config_name_map:
                config_name = config_name_map[config_name]

            if config_name not in cat_data[metric]:
                cat_data[metric][config_name] = []
            
            cat_data[metric][config_name].append(values)

        # Compute averages and stds for each config_name
        for config_name, values_list in cat_data[metric].items():
            # Transpose the list of lists to get values for each hole size
            transposed_values = list(zip(*values_list))
            averages = [sum(values) / len(values) for values in transposed_values]
            stds = [((sum((x - avg) ** 2 for x in values) / len(values)) ** 0.5) for values, avg in zip(transposed_values, averages)]
            cat_data[metric][config_name] = {
                "avg": averages,
                "std": stds
            }

    return cat_data


def main():
    file_path = "results/through_hole_results1.txt"
        
    data, x = get_data(file_path)
    if x is None:
        x = list(range(2, 30, 2))   # Hole sizes from 2 to 28 with step of 2

    config_name_map = {
        "int_vel_sin": "Sinusoidal PE (rozmiar dziury w treningu: [4, 16])",
        "int_vel_sin_h15": "Sinusoidal PE (rozmiar dziury w treningu: 15)",
        "int_vel_rel": "Relative Bias (rozmiar dziury w treningu: [4, 16])",
        "int_vel_rel_h15": "Relative Bias (rozmiar dziury w treningu: 15)",
        "int_vel_rot": "Rotary PE (rozmiar dziury w treningu: [4, 16])",
        "int_vel_rot_h15": "Rotary PE (rozmiar dziury w treningu: 15)"
    }

    data = data_concat(data, config_name_map=config_name_map)

    colors = [
        mcolors.CSS4_COLORS['orangered'],
        mcolors.CSS4_COLORS['lightcoral'],
        mcolors.CSS4_COLORS['blue'],
        mcolors.CSS4_COLORS['lightskyblue'],
        mcolors.CSS4_COLORS['green'],
        mcolors.CSS4_COLORS['lawngreen']
    ]

    # for metric in data.keys():
    #     with plt.xkcd():
    #         plt.figure(figsize=(10, 6))
    #         for i, (model_name, values) in enumerate(data[metric].items()):
    #             plt.plot(x, values, label=model_name)
    #         plt.title(f"{metric.upper()} vs Hole Size (lower is better)")
    #         plt.xlabel("Hole Size (frames)")
    #         plt.ylabel(metric.upper())
    #         plt.xticks(x)
    #         plt.grid()
    #         plt.legend()
    #         plt.savefig(f"results/{metric}_vs_hole_size.png")
    #         plt.close()

    a, b = 0, len(x)
    a, b = 4, 12
    print(f"Holes: {x[a:b]}")

    for metric in data.keys():
        plt.figure(figsize=(8, 6))
        # plt.xkcd()
        for i, (config_name, values) in enumerate(data[metric].items()):
            avg_values = values["avg"]
            std_values = values["std"]
            plt.errorbar(x[a:b], avg_values[a:b], yerr=std_values[a:b], label=config_name, capsize=5, color=colors[i % len(colors)])
        plt.title(f"{metric.upper()} vs Rozmiar dziury")
        plt.xlabel("Rozmiar dziury podczas ewaluacji (w klatkach)")
        plt.ylabel(metric.upper())
        plt.xticks(x[a:b])
        plt.grid()
        plt.legend()
        plt.savefig(f"results/figures/{metric}_vs_hole_size.png")
        plt.close()
        # plt.show()
        


if __name__ == "__main__":
    main()