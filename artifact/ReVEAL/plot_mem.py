import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.font_manager import FontProperties

def preprocess_csv(file_path):
    df = pd.read_csv(file_path)
    df = df[~df.iloc[:, 0].astype(str).str.startswith('128_128_S')]
    df = df.sort_values(by=df.columns[0])
    return df

# Read and preprocess the CSV files
df1 = preprocess_csv('./exp1_reveal/timing_report_resyn3_128_newp.csv')
df2 = preprocess_csv('./exp1_dynphase/resyn3_128_output.csv')
df3 = preprocess_csv('./exp1_revsca/resyn3_128_output.csv')
df4 = preprocess_csv('./exp1_amulet/resyn3_128_output.csv')  # New file

def process_memory_data(df, memory_column):
    memory_data = df[memory_column].replace({'None': 0, np.nan: 0, 'MO': 0})
    memory_data = memory_data.astype(float)
    used_memory = memory_data[memory_data > 0].sort_values().tolist()
    cumulative_counts = list(range(1, len(used_memory) + 1))
    return used_memory, cumulative_counts

def process_memory_data_with_status(df, memory_column, status_column, label):
    original_memory_data = df[memory_column].astype(str)
    status_data = df[status_column].str.lower()
    
    mo_count = original_memory_data.str.upper().eq('MO').sum()
    to_count = status_data.str.upper().eq('AIG FILE MISSING').sum()
    
    print(f"{label} - Number of MO circuits: {mo_count}")
    print(f"{label} - Number of TO circuits: {to_count}")
    
    memory_data = df[memory_column].replace({'None': 0, np.nan: 0, 'MO': 0}).astype(float)
    
    buggy_count = (status_data == 'buggy').sum()
    correct_count = (status_data == 'correct').sum()
    
    print(f"{label} - Number of buggy circuits: {buggy_count}")
    print(f"{label} - Number of correct circuits: {correct_count}")
    
    filtered_memory_data = memory_data[status_data != 'buggy']
    used_memory = filtered_memory_data[filtered_memory_data > 0].sort_values().tolist()
    cumulative_counts = list(range(1, len(used_memory) + 1))
    return used_memory, cumulative_counts

# Process memory data
memory1, count1 = process_memory_data(df1, 'Memory (KB)')
memory2, count2 = process_memory_data_with_status(df2, 'Memory Usage', 'Circuit Status', 'DynPhaseOrderOpt')
memory3, count3 = process_memory_data_with_status(df3, 'Memory Usage', 'Circuit Status', 'RevSCA-2.0')
memory4, count4 = process_memory_data_with_status(df4, 'Memory Usage', 'Circuit Status', 'Amulet2.2')  # New data

# Plotting
y_max_limit = 64

plt.figure(figsize=(14, 10))
plt.xlabel('Memory Usage (KB)', fontsize=50, fontweight='bold', labelpad=10)
plt.ylabel('Solved Cases', fontsize=50, fontweight='bold', labelpad=10)

plt.plot(memory1, count1, marker='o', linestyle='-', color=(0, 0, 1, 0.6), label='ReVEAL')
plt.plot(memory2, count2, marker='s', linestyle='-', color=(0, 0.5, 0, 0.6), label='DynPOO')
plt.plot(memory3, count3, marker='^', linestyle='-', color=(1, 0, 0, 0.6), label='RevSCA-2.0')
plt.plot(memory4, count4, marker='d', linestyle='-', color=(1, 0.5, 0, 0.6), label='AMulet2.2')  # Orange color

plt.xscale('log')

max_memory = max(max(memory1, default=1), max(memory2, default=1), max(memory3, default=1), max(memory4, default=1))
min_memory = min(min(memory1, default=1), min(memory2, default=1), min(memory3, default=1), min(memory4, default=1))

start_exp = int(np.floor(np.log10(min_memory))) if min_memory > 0 else 0
end_exp = int(np.ceil(np.log10(max_memory))) if max_memory > 0 else 6
exponents = range(start_exp, end_exp + 1)
x_ticks = [10**i for i in exponents]
x_labels = [f"$10^{i}$" for i in exponents]

plt.xticks(ticks=x_ticks, labels=x_labels, fontsize=30, fontweight='bold')
plt.yticks(fontsize=30, fontweight='bold')
plt.ylim(bottom=0, top=y_max_limit)

font_prop = FontProperties(weight='bold', size=30)
plt.legend(loc='lower right', framealpha=0.4, prop=font_prop)

plt.grid(True, which="both", ls="--", linewidth=0.5)
plt.tight_layout()

plt.savefig('resyn3_128_memory.pdf')
plt.show()