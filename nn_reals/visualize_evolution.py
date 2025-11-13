import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

def load_data_from_json(json_file_path: str) -> pd.DataFrame:
    """
    Loads the evolution history from the specified JSON metrics file.
    
    Args:
        json_file_path: The path to the metrics.json file.

    Returns:
        A pandas DataFrame containing the full history, or an empty DataFrame if an error occurs.
    """
    path = Path(json_file_path)
    if not path.is_file():
        print(f"Error: File not found at '{json_file_path}'")
        return pd.DataFrame()

    try:
        with open(path, 'r') as f:
            data = json.load(f)
        
        # The actual time-series data is stored under the "metrics" key
        history_data = data.get('metrics', {})
        if not history_data:
            print("Error: JSON file does not contain a 'metrics' key.")
            return pd.DataFrame()

        return pd.DataFrame(history_data)
    
    except json.JSONDecodeError:
        print(f"Error: Could not decode JSON from '{json_file_path}'. The file may be corrupted.")
        return pd.DataFrame()
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        return pd.DataFrame()

def prepare_dataframe_for_plotting(df: pd.DataFrame, metric_name: str) -> pd.DataFrame:
    """
    Selects and renames the necessary columns for the generic plotting function.
    
    Args:
        df: The full DataFrame loaded from JSON.
        metric_name: The base name of the metric to visualize (e.g., 'euclidean').

    Returns:
        A new DataFrame with standardized column names for plotting.
    """
    # Define the column names based on the metric_name
    diversity_col = f'{metric_name}_diversity'
    correlation_col = f'{metric_name}_correlation'
    
    required_cols = ['generation', 'best_fitness', diversity_col, correlation_col]
    
    # Check if all required columns exist in the DataFrame
    for col in required_cols:
        if col not in df.columns:
            print(f"Error: Required column '{col}' not found in the JSON data.")
            return pd.DataFrame()
            
    # Select the relevant columns
    plot_df = df[required_cols].copy()
    
    # Rename them to the generic names the plotting function expects
    plot_df.rename(columns={
        'generation': 'Generation',
        'best_fitness': 'Best Fitness',
        diversity_col: 'Diversity',
        correlation_col: 'Correlation'
    }, inplace=True)
    
    return plot_df

def plot_evolution_metrics(df: pd.DataFrame, title: str, metric_name: str):
    """
    Generates and displays a multi-axis plot of the evolution metrics.
    """
    if df.empty:
        print("No data to plot. Check previous errors.")
        return

    # Set a nice theme for the plot
    sns.set_theme(style="whitegrid")

    fig, ax1 = plt.subplots(figsize=(16, 8))

    # --- Axis 1 (Left): Plot Fitness and Diversity ---
    ax1.set_xlabel('Generation', fontsize=14)
    ax1.set_ylabel('Fitness / Diversity', fontsize=14, color='darkblue')
    
    sns.lineplot(x='Generation', y='Best Fitness', data=df, ax=ax1, color='royalblue', label='Best Fitness')
    sns.lineplot(x='Generation', y='Diversity', data=df, ax=ax1, color='skyblue', linestyle='--', label='Diversity')
    ax1.tick_params(axis='y', labelcolor='darkblue')
    ax1.set_ylim(bottom=0) # Ensure y-axis starts at 0 for better perspective

    # --- Axis 2 (Right): Plot Correlation ---
    ax2 = ax1.twinx()
    ax2.set_ylabel('Correlation', fontsize=14, color='darkred')
    
    sns.lineplot(x='Generation', y='Correlation', data=df, ax=ax2, color='salmon', label='Correlation')
    ax2.tick_params(axis='y', labelcolor='darkred')
    ax2.axhline(0, color='grey', linestyle=':', linewidth=1)
    ax2.set_ylim(-1, 1) # Correlation is always between -1 and 1

    # --- Final Touches ---
    plt.title(title, fontsize=18, weight='bold')
    
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.get_legend().remove()
    ax2.get_legend().remove()
    fig.legend(lines1 + lines2, labels1 + labels2, loc='upper right', bbox_to_anchor=(0.88, 0.88))

    plt.tight_layout()

    plt.savefig(f'evolution_dynamics_{metric_name}.png', dpi=300, bbox_inches='tight')
    print(f"Heatmap saved as 'evolution_dynamics_{metric_name}.png'")
    plt.show()

if __name__ == "__main__":
    # 1. CONFIGURE: Set the path to your JSON file and the metric you want to see
    JSON_FILE_PATH = "metrics/run_3000gen_interval10_s42.json" 
    METRIC_TO_VISUALIZE = "euclidean"

    print(f"Loading evolution data from: '{JSON_FILE_PATH}'")
    
    # 2. LOAD: Read the full dataset from the JSON file.
    full_evolution_df = load_data_from_json(JSON_FILE_PATH)

    if not full_evolution_df.empty:
        # 3. PREPARE: Select and rename the columns for the specific metric.
        plot_df = prepare_dataframe_for_plotting(full_evolution_df, METRIC_TO_VISUALIZE)
        
        # 4. PLOT: Generate and display the graph.
        plot_title = f'Neuroevolution Performance Over Time ({METRIC_TO_VISUALIZE.capitalize()} Metric)'
        plot_evolution_metrics(plot_df, title=plot_title, metric_name=METRIC_TO_VISUALIZE)