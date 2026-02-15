import argparse
import json
import yaml
from pathlib import Path
import sys
from multiprocessing import Pool

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# Add the parent directory to sys.path to allow importing the hypothesis_helper module
sys.path.append(str(Path.cwd().parent))
import hypothesis_helper as hp_helper

# Parse arguments for data path and output
parser = argparse.ArgumentParser(description="Plot reasoning model comparison (o4 mini vs o3 mini)")
parser.add_argument(
    "--data-dir",
    default="/tmp/llm_data",
    help="Root directory containing experiment output (default: /tmp/llm_data on cluster)",
)
parser.add_argument(
    "--output",
    default="reasoning_models_o4_o3_comparison.png",
    help="Output PNG path for comparison plot",
)
parser.add_argument("--no-show", action="store_true", help="Do not display plot (save only)")
args, _ = parser.parse_known_args()

parent_path = Path(args.data_dir)
parser = argparse.ArgumentParser(description="Plot reasoning model comparison (o4 mini vs o3 mini)")
parser.add_argument(
    "--data-dir",
    default="/home/anthony/Project/rl-scm-tests/blicket_text/exp_output/",
    help="Root directory containing experiment output",
)
parser.add_argument(
    "--output",
    default="reasoning_models_o4_o3_comparison.png",
    help="Output PNG path for comparison plot",
)
parser.add_argument("--no-show", action="store_true", help="Do not display plot (save only)")
args, _ = parser.parse_known_args()
if not in_files:
    in_files = list(parent_path.rglob("**/action_log_trial-*.jsonl"))

parent_path = Path(args.data_dir)
exp_wcs = [
    "2025.03.26/144924/*/action_log_trial-*.jsonl",  # oracle
    "2025.05.31/185549/*/action_log_trial-*.jsonl",  # 
    "2025.05.31/185647/*/action_log_trial-*.jsonl",  # 
    "2025.05.31/185704/*/action_log_trial-*.jsonl",  # 
    "2025.05.27/015244/*/action_log_trial-*.jsonl",
    "2025.05.28/023113/*/action_log_trial-*.jsonl",
    "2025.05.31/130113/*/action_log_trial-*.jsonl",
]

in_files = []
for exp_wc in exp_wcs:
    in_files.extend(parent_path.rglob(exp_wc))

print(len(in_files))
sorted(in_files[:5])


OMPUTE_HYPOTHESIS_ENTROPY = True
STANDARD_MAX_TRAJ_LEN = 32  # set to None to not use this

# Helper function to flatten nested dictionaries
def flatten_dict(d, parent_key='', sep='.'):
    items = []
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)


# Function to read the CSV and YAML files and merge them into a single dataframe
def read_and_merge(path):
    # Read the trajectory file
    assert path.suffix == '.jsonl'
    data = []
    with open(path, 'r') as file:
        for line in file:
            cur_json = json.loads(line)
            # data = append_dict(data, cur_json)
            if "question" not in cur_json:  # NOTE: only loading non Q&A data
                data.append(flatten_dict(cur_json))
    
    if COMPUTE_HYPOTHESIS_ENTROPY:
        state_traj = [d['game_state.true_state'] for d in data if 'game_state.true_state' in d]
        state_traj = np.array(state_traj)  # [T, num_object + 1]

        for i in range(0, len(state_traj)):
            hypo_left = hp_helper.compute_num_valid_hypothesis(state_traj[:i+1])
            data[i]['n_hypothesis_left'] = hypo_left
    
    if STANDARD_MAX_TRAJ_LEN is not None:
        base_data_len = len(data)
        remain_len = max(0, STANDARD_MAX_TRAJ_LEN - len(data))
        for i in range(0, remain_len):
            data.append({"steps": base_data_len + i, "file_path": str(path)})

    # Read the YAML file
    yaml_path = path.parent / '.hydra' / 'config.yaml'
    with open(yaml_path, 'r') as file:
        config = yaml.safe_load(file)
    
    # Flatten the YAML config
    flat_config = flatten_dict(config, parent_key='cfg')
    flat_config["file_path"] = str(path)
    for key, value in flat_config.items():
        flat_config[key] = [value] * len(data)
    config_df = pd.DataFrame(flat_config)
    full_df = pd.DataFrame(data)
    full_df = full_df.merge(config_df, left_index=True, right_index=True, how='left')

    return full_df


    # Read
#with ThreadPoolExecutor(max_workers=32) as tpe:
#    results = list(tpe.map(read_and_merge, in_files))
#    all_data = pd.concat(results, ignore_index=True)

# Use multiprocessing for parallel processing
if True: 
    # Process files in parallel using multiprocessing.Pool
    with Pool(processes=32) as pool:
        all_data = list(pool.map(read_and_merge, in_files))
        all_data = pd.concat(all_data, ignore_index=True)
else:
    all_data = [read_and_merge(path) for path in in_files]
    all_data = pd.concat(all_data, ignore_index=True)
#all_data = [read_and_merge(path) for path in in_files]
#all_data = pd.concat(all_data, ignore_index=True)

print(all_data.shape)
all_data.head(2)

def process_data(in_df):
    df = in_df.copy()
    
    df['cfg.agent.model'].fillna("", inplace=True)
    df['cfg.agent.reasoning_effort'].fillna("", inplace=True)

    df = df[df['cfg.agent._target_'].str.contains(
        'RandomAgent|OracleAgent|OReasonPromptsAgent', na=True
    )]

    def _map_model(row):
        name = ""
        if '2025' in row['cfg.agent.model']:
            name += row['cfg.agent.model'].split('-2025')[0]
        elif "OracleAgent" in row['cfg.agent._target_']:
            name += 'infoGain oracle'
        elif "RandomAgent" in row['cfg.agent._target_']:
            name += "baseline random"
        else:
            name += row['cfg.agent.model']
        # Add reasoning effort if present and non-empty
        reasoning_effort = str(row.get('cfg.agent.reasoning_effort', '')).strip()
        if reasoning_effort and reasoning_effort.lower() != "nan":
            name += f" ({reasoning_effort} reasoning)"
        return name
    
    df['cfg.model'] = df.apply(_map_model, axis=1)
    df.drop(columns=['cfg.agent.model'], inplace=True)

    df = df.reset_index(drop=True)

    return df

all_data_processed = process_data(all_data)
print(all_data_processed.shape)
all_data_processed.head(2)

for col in all_data_processed.columns:
    if not col.startswith('cfg.'):
        continue
    
    if col == "file_path":
        continue
    
    unique_values = all_data_processed[col].unique()
    if len(unique_values) > 1:
        print(f"Column: {col}")
        print(f"Unique Values: {unique_values}")
        print()


# Display the mapping as actual colors
import matplotlib.patches as mpatches

def setup_model_colors(in_df):
    df = in_df.copy()
    # Initialize the model_colors dictionary with baseline colors
    model_colors = {
        'infoGain oracle': (0.0, 0.0, 0.0),  # Black
        'baseline random': (0.5, 0.5, 0.5)    # Grey
    }
    # Sort the models alphabetically before assigning colors
    sorted_models = sorted(df['cfg.model'].unique())

    # Get the remaining unique models excluding the baselines
    remaining_models = [model for model in sorted_models
                        if model not in model_colors]

    # Map the remaining models to a color palette
    # Options for discrete palettes in seaborn include: "Set1", "Set2", "Set3", "Pastel1", "Pastel2", "Dark2", "Accent", "tab10", "tab20", "tab20b", "tab20c", "Paired", "colorblind"
    color_palette = sns.color_palette("tab10", len(remaining_models))
def plot_o4_o3_comparison(df, output_path=None, show=True):
    """Plot comparison of o4 mini and o3 mini with low/medium/high reasoning effort."""
    df = df.copy()
    df['cfg.agent.react'].fillna("", inplace=True)
    df['cfg.agent.system_msg_path'].fillna("", inplace=True)
    df['n_hypothesis_left'] = df['n_hypothesis_left'].ffill()

    df = df[df['cfg.env_kwargs.num_objects'].isin([8])]
    # Filter for oracle + o4 mini + o3 mini (all reasoning effort levels)
    df = df[df['cfg.model'].str.contains('oracle|o4|o3', case=False, na=False)]

    plt_x = "steps"
    plt_y = "n_hypothesis_left"
    plt_hue = "cfg.model"
    plt_subplots = "cfg.env_kwargs.rule"
    plt_palette = MODEL_COLORS

    fig, axes = plt.subplots(1, 2, figsize=(8, 4), sharey=True, sharex=True)
    df.sort_values(by=[plt_hue, plt_x], inplace=True)

    for ax, (rule, group) in zip(axes, df.groupby(plt_subplots)):
        sns.lineplot(
            x=plt_x, y=plt_y, hue=plt_hue,
            estimator=np.mean, errorbar=('ci', 95), n_boot=200,
            palette=plt_palette,
            data=group, ax=ax,
            legend=(ax is axes[0])
        )
        ax.set_title(f'Rule: {rule}')
        ax.set_yscale('log')
        ax.axhline(y=1, color='gray', linestyle='--')
        ax.set_ylabel('N Hypotheses Remaining')
        ax.set_xlabel('Steps')

    handles, labels = axes[0].get_legend_handles_labels()
    for ax in axes:
        if ax.get_legend() is not None:
            ax.get_legend().remove()
    fig.legend(
        handles, labels, loc='lower center', bbox_to_anchor=(0.5, -0.12),
        ncol=4, frameon=False, fontsize=10
    )

    plt.tight_layout()

    if output_path:
        fig.savefig(output_path, bbox_inches='tight', dpi=150)
        print(f"Saved: {output_path}")

    if show:
        plt.show()
    else:
        plt.close(fig)


plot_o4_o3_comparison(
    all_data_processed,
    output_path=Path(args.output),
    show=not args.no_show,