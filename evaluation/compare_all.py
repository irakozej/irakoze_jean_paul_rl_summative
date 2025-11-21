import os
import glob
import pandas as pd
import subprocess


def find_models(root='models'):
    entries = []
    for algo_dir in os.listdir(root):
        algo_path = os.path.join(root, algo_dir)
        if not os.path.isdir(algo_path):
            continue
        if algo_dir.lower() == 'reinforce':
            
            for f in glob.glob(os.path.join(algo_path, '*.pt')):
                entries.append({'algo': 'reinforce', 'path': f})
        else:
            
            for run_dir in os.listdir(algo_path):
                final_model = os.path.join(algo_path, run_dir, 'final_model.zip')
                if os.path.exists(final_model):
                    entries.append({'algo': algo_dir.lower(), 'path': final_model})
    return entries


def evaluate_if_needed(entry, episodes=100):
    algo = entry['algo']
    model_path = entry['path']
    model_tag = os.path.splitext(os.path.basename(model_path))[0]
    out_csv = os.path.join('results', 'evaluation', f"{algo}_{model_tag}_eval.csv")
    if os.path.exists(out_csv):
        print(f"Skipping {model_path}, evaluation exists")
        return out_csv
    # call evaluate_agent.py
    cmd = ['python', 'evaluation/evaluate_agent.py', '--model-path', model_path, '--algo', algo, '--episodes', str(episodes), '--out-dir', 'results/evaluation']
    print('Running:', ' '.join(cmd))
    env = os.environ.copy()
    env['PYTHONPATH'] = '.'
    subprocess.run(cmd, check=True, env=env)
    return out_csv


def aggregate_results(out_dir='../results/evaluation'):
    all_files = glob.glob(os.path.join(out_dir, '*_eval.csv'))
    rows = []
    for f in all_files:
        df = pd.read_csv(f)
        mean_reward = df['reward'].mean()
        std_reward = df['reward'].std()
        success_rate = (df['outcome'] == 'mastery_reached').mean()
        rows.append({'file': os.path.basename(f), 'mean_reward': mean_reward, 'std_reward': std_reward, 'success_rate': success_rate, 'episodes': len(df)})
    summary = pd.DataFrame(rows)
    summary_path = os.path.join(out_dir, 'summary.csv')
    summary.to_csv(summary_path, index=False)
    print(f"Wrote summary to {summary_path}")
    print(summary)
    return summary_path


def main():
    entries = find_models('models')
    print(f"Found {len(entries)} models to evaluate")
    for e in entries:
        try:
            evaluate_if_needed(e, episodes=100)
        except Exception as ex:
            print(f"Failed evaluating {e['path']}: {ex}")
    aggregate_results('results/evaluation')


if __name__ == '__main__':
    main()


