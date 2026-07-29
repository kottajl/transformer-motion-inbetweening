import json

'''
Example config_key: "int_vel_sin": {
    5: ...,
    15: [
        {"l2p": ..., "l2q": ..., "npss": ...}, 
        {"l2p": ..., "l2q": ..., "npss": ...}, 
        {"l2p": ..., "l2q": ..., "npss": ...}
    ],
    20: ...,
    30: ...
}
'''


def read_data_from_file(file_path) -> dict:
    data = dict()

    with open(file_path, "r") as f:
        # Example line: "s1_int_vel_sin,15,0.1234,0.5678,0.9101"
        lines = f.readlines()
        # first_line = lines[0].strip().split(',')
        first_line = "...,...,l2p,l2q,npss".strip().split(',')
        metric_names = first_line[2:]  # e.g., ["l2p", "l2q", "npss"]
        print(metric_names)

        for line_idx, line in enumerate(lines):
            line = line.strip()
            if not line:
                continue

            line_results = line.split(',')
            assert len(line_results) == len(first_line), f"Line {line_idx} has unexpected number of values: {len(line_results)} (expected {len(first_line)})"

            config_name_pos = line_results[0].find('_') + 1     # Remove the seed prefix (e.g., "s1_")
            
            config_key = line_results[0][config_name_pos:]      # e.g., "int_vel_sin"
            h_size = int(line_results[1])                       # e.g., 15
            metric_results = line_results[2:]                   # e.g., ["0.1234", "0.5678", "0.9101"]

            if config_key not in data:
                data[config_key] = dict()
            if h_size not in data[config_key]:
                data[config_key][h_size] = list()
            
            metrics = {metric_names[i]: float(metric_results[i]) for i in range(len(metric_names))}
            data[config_key][h_size].append(metrics)

    # print(json.dumps(data, indent=4))
    return data
#read_data_from_file


def main():
    data = read_data_from_file("results/test_results.txt")
    
    cat_data = dict()
    for config_key, config_values in data.items():
        cat_data[config_key] = dict()
        for h_size, metrics_list in config_values.items():
            cat_data[config_key][h_size] = dict()
            for metric in metrics_list[0].keys():
                # Compute avg and std for each metric
                m_values = [v[metric] for v in metrics_list]
                m_avg = sum(m_values) / len(m_values)
                m_std = (sum((x - m_avg) ** 2 for x in m_values) / len(m_values)) ** 0.5
                cat_data[config_key][h_size][metric] = {"avg": round(m_avg, 6), "std": round(m_std, 6)}
    
    # cat_data = {5: ..., 15: {"l2p": {"avg": ..., "std": ...}, "l2q": ..., "npss": ...}, 20: ..., 30: ...}
    print(json.dumps(cat_data, indent=4))
#main


if __name__ == "__main__":
    main()
