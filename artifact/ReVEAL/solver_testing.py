import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.font_manager import FontProperties

# CSV
def preprocess_csv(file_path):
    df = pd.read_csv(file_path)
    df.columns = df.columns.str.strip().str.lower()
    #  '128_128_S' 
    df = df[~df['filename'].astype(str).str.startswith('128_128_S')]
    df = df.sort_values(by='filename')  # 
    return df

def process_time_data(df, time_column):
    # TONone3600
    time_data = df[time_column].replace({'TO': 3600, 'To': 3600, 'to': 3600, 'None': 3600, np.nan: 3600})
    time_data = time_data.astype(float)
    # 3600
    solved_times = time_data[time_data < 3600].sort_values().tolist()
    cumulative_counts = list(range(1, len(solved_times) + 1))
    avg_time = np.mean(time_data)  #  3600 
    return solved_times, cumulative_counts, avg_time

# CSV
df1 = preprocess_csv('./exp2_solver1/dc2_new/dc2_64_result_cec_1.csv')
df2 = preprocess_csv('./exp2_solver1/dc2_new/dc2_64_result_cec.csv')
df3 = preprocess_csv('./exp2_solver1/dc2_new/dc2_64_result_fraig.csv')
df4 = preprocess_csv('./exp2_solver1/dc2_new/dc2_64_result_fraig1.csv')
df5 = preprocess_csv('./exp2_solver1/dc2_new/dc2_64_result_kissat.csv')

time1, count1, avg1 = process_time_data(df1, 'time')
time2, count2, avg2 = process_time_data(df2, 'time')
time3, count3, avg3 = process_time_data(df3, 'time')
time4, count4, avg4 = process_time_data(df4, 'time')
time5, count5, avg5 = process_time_data(df5, 'time')


#  Solved Cases 
print(f"Solved cases for CEC_1: {len(time1)}, Average Time (including TO): {avg1:.2f} seconds")
print(f"Solved cases for CEC: {len(time2)}, Average Time (including TO): {avg2:.2f} seconds")
print(f"Solved cases for FRAIG: {len(time3)}, Average Time (including TO): {avg3:.2f} seconds")
print(f"Solved cases for FRAIG1: {len(time4)}, Average Time (including TO): {avg4:.2f} seconds")
print(f"Solved cases for Kissat: {len(time5)}, Average Time (including TO): {avg5:.2f} seconds")

# Y64
y_max_limit = 64

plt.figure(figsize=(12, 8))
plt.xlabel('CPU Time (seconds)', fontsize=50, fontweight='bold', labelpad=10)
plt.ylabel('Solved Cases', fontsize=50, fontweight='bold', labelpad=10)

plt.plot(time1, count1, marker='o', linestyle='-', color=(0, 0, 1, 0.6), label='CEC_1')
plt.plot(time2, count2, marker='s', linestyle='-', color=(0, 0.5, 0, 0.6), label='CEC')
plt.plot(time3, count3, marker='^', linestyle='-', color=(1, 0, 0, 0.6), label='FRAIG')
plt.plot(time4, count4, marker='D', linestyle='-', color=(1, 0.65, 0, 0.6), label='FRAIG1')
plt.plot(time5, count5, marker='x', linestyle='-', color=(0.5, 0, 0.5, 0.6), label='Kissat')  # 

# x
plt.xscale('log')

# x10
max_time = max(max(time1, default=1), max(time2, default=1), max(time3, default=1), max(time4, default=1), max(time5, default=1))
min_time = min(min(time1, default=1), min(time2, default=1), min(time3, default=1), min(time4, default=1), min(time5, default=1))

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
plt.savefig('dc2_new_64_time.pdf')
plt.show()