import os
import glob
import pandas as pd
import matplotlib.pyplot as plt


def find_monitor_files(root='results'):
    matches = glob.glob(os.path.join(root, '**', 'monitor.csv'), recursive=True)
    return matches


def plot_monitor(file_path, out_dir='results/plots'):
    os.makedirs(out_dir, exist_ok=True)
    df = pd.read_csv(file_path)
    
    if 'r' in df.columns:
        rewards = df['r']
    elif 'reward' in df.columns:
        rewards = df['reward']
    else:
        print(f"Unknown monitor format for {file_path}; skipping")
        return
    # simple plot
    window = max(1, len(rewards) // 50)
    smooth = rewards.rolling(window=window).mean()
    plt.figure(figsize=(8,4))
    plt.plot(smooth)
    plt.title(os.path.basename(file_path))
    plt.xlabel('evaluation step')
    plt.ylabel('reward (rolling mean)')
    out_path = os.path.join(out_dir, os.path.basename(file_path).replace('/', '_') + '.png')
    plt.savefig(out_path)
    plt.close()
    print(f"Saved plot to {out_path}")


def main():
    files = find_monitor_files('results')
    print(f"Found {len(files)} monitor files")
    for f in files:
        try:
            plot_monitor(f)
        except Exception as ex:
            print(f"Failed plotting {f}: {ex}")


if __name__ == '__main__':
    main()
