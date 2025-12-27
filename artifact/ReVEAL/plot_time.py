import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.font_manager import FontProperties

# CSV
def preprocess_csv(file_path):
    df = pd.read_csv(file_path)
    #  '128_128_S' 
    df = df[~df.iloc[:, 0].astype(str).str.startswith('128_128_S')]
    df = df.sort_values(by=df.columns[0])  # 
    return df

# CSV
df1 = preprocess_csv('./exp1_reveal/timing_report_resyn3_128_newp1.csv')
df2 = preprocess_csv('./exp1_dynphase/resyn3_128_output.csv')
df3 = preprocess_csv('./exp1_revsca/resyn3_128_output.csv')
df4 = preprocess_csv('./exp1_amulet/esyn3_128_output.csv')  # 

def process_time_data(df, time_column):
    time_data = df[time_column].replace({'TO': 3600, 'To': 3600, 'to': 3600, 'None': 3600, np.nan: 3600})
    time_data = time_data.astype(float)
    solved_times = time_data[time_data < 3600].sort_values().tolist()
    cumulative_counts = list(range(1, len(solved_times) + 1))
    return solved_times, cumulative_counts

def process_time_data_with_status(df, time_column, status_column, label):
    time_data = df[time_column].replace({'TO': 3600, 'To': 3600, 'to': 3600, 'None': 3600, np.nan: 3600})
    time_data = time_data.astype(float)

    status_data = df[status_column].str.lower()

    buggy_count = (status_data == 'buggy').sum()
    correct_count = (status_data == 'correct').sum()
    to_count = (time_data == 3600).sum()
    mo_count = (status_data == 'mo').sum()

    print(f"{label} - Number of buggy circuits: {buggy_count}")
    print(f"{label} - Number of correct circuits: {correct_count}")
    print(f"{label} - Number of TO circuits: {to_count}")
    print(f"{label} - Number of MO circuits: {mo_count}")

    filtered_time_data = time_data[status_data != 'buggy']
    solved_times = filtered_time_data[filtered_time_data < 3600].sort_values().tolist()
    cumulative_counts = list(range(1, len(solved_times) + 1))
    return solved_times, cumulative_counts

time1, count1 = process_time_data(df1, 'Time (seconds)')
time2, count2 = process_time_data_with_status(df2, 'Time Overall', 'Circuit Status', 'DynPhaseOrderOpt')
time3, count3 = process_time_data_with_status(df3, 'Time Overall', 'Circuit Status', 'RevSCA-2.0')
time4, count4 = process_time_data_with_status(df4, 'Time Overall', 'Circuit Status', 'Amulet2.2')  # 

#  Solved Cases
print(f"Solved cases for RevEAL: {len(time1)}")
print(f"Solved cases for DynPhaseOrderOpt: {len(time2)}")
print(f"Solved cases for RevSCA-2.0: {len(time3)}")
print(f"Solved cases for Amulet2.2: {len(time4)}")  # 

# Y128
y_max_limit = 64

plt.figure(figsize=(12, 8))
plt.xlabel('CPU Time (seconds)', fontsize=50, fontweight='bold', labelpad=10)
plt.ylabel('Solved Cases', fontsize=50, fontweight='bold', labelpad=10)

plt.plot(time1, count1, marker='o', linestyle='-', color=(0, 0, 1, 0.6), label='ReVEAL')
plt.plot(time2, count2, marker='s', linestyle='-', color=(0, 0.5, 0, 0.6), label='DynPOO')
plt.plot(time3, count3, marker='^', linestyle='-', color=(1, 0, 0, 0.6), label='RevSCA-2.0')
plt.plot(time4, count4, marker='D', linestyle='-', color=(1, 0.65, 0, 0.6), label='AMulet2.2')  # 

# x
plt.xscale('log')

# x10
max_time = max(max(time1, default=1), max(time2, default=1), max(time3, default=1), max(time4, default=1))
min_time = min(min(time1, default=1), min(time2, default=1), min(time3, default=1), min(time4, default=1))

start_exp = int(np.floor(np.log10(min_time))) if min_time > 0 else 0
end_exp = int(np.ceil(np.log10(max_time))) if max_time > 0 else 6
exponents = range(start_exp, end_exp + 1)
x_ticks = [10**i for i in exponents]
x_labels = [f"$10^{i}$" for i in exponents]

plt.xticks(ticks=x_ticks, labels=x_labels, fontsize=30, fontweight='bold')
plt.yticks(fontsize=30, fontweight='bold')

# y
plt.ylim(bottom=0, top=y_max_limit)

font_prop = FontProperties(weight='bold', size=30)
plt.legend(loc='lower right', framealpha=0.4, prop=font_prop)

plt.grid(True, which="both", ls="--", linewidth=0.5)

plt.tight_layout()

#  PDF 
plt.savefig('resyn3_128_time.pdf')
plt.show()