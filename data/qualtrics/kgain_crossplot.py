import pandas as pd
import matplotlib.pyplot as plt

groups = ['A', 'B', 'C']
sc0_list = []
sc1_list = []
diff_list = []

for group in groups:
    df = pd.read_csv(f'testgroup{group}.csv', skiprows=[1])
    
    # Drop mapping row if present
    if df.iloc[0].astype(str).str.contains("ImportId").any():
        df = df.iloc[1:].reset_index(drop=True)
    
    # Extract SC0 and SC1 values from the first data row
    sc0 = float(df.loc[0, "SC0"])
    sc1 = float(df.loc[0, "SC1"])
    diff = sc1 - sc0  # compute sc1 - sc0
    
    sc0_list.append(sc0)
    sc1_list.append(sc1)
    diff_list.append(diff)
    
    print(f"Group {group}: SC0 = {sc0}, SC1 = {sc1}, Difference (SC1-SC0) = {diff}")

x = range(len(groups))
width = 0.25

plt.figure(figsize=(8, 6))
bars_sc0 = plt.bar([p - width for p in x], sc0_list, width=width, label='Part 1')
bars_sc1 = plt.bar(x, sc1_list, width=width, label='Part 2')
bars_diff = plt.bar([p + width for p in x], diff_list, width=width, label='Knowledge Gain (Part2 - Part 1)')

plt.xticks(x, groups)
plt.xlabel('Group')
plt.ylabel('Values')
#plt.title('Comparison of SC0, SC1 and their Difference for Each Group')
plt.legend()

# Annotate each bar with its value
def annotate_bars(bars):
    for bar in bars:
        height = bar.get_height()
        plt.annotate(f'{height}',
                     xy=(bar.get_x() + bar.get_width()/2, height),
                     xytext=(0, 3),
                     textcoords="offset points",
                     ha='center', va='bottom')

annotate_bars(bars_sc0)
annotate_bars(bars_sc1)
annotate_bars(bars_diff)

plt.show()
