import pandas as pd
import matplotlib.pyplot as plt

groups = ['A', 'B', 'C']
plt.figure()

for group in groups:
    df = pd.read_csv(f'testgroup{group}.csv', skiprows=[1])
    
    # Remove mapping row if it exists
    if df.iloc[0].astype(str).str.contains("ImportId").any():
        df = df.iloc[1:].reset_index(drop=True)
    
    # Extract Timer_Page Submit columns
    timer_submit_cols = [col for col in df.columns if "Timer_Page Submit" in col]
    
    # Use the first actual data row for plotting
    values = df.loc[0, timer_submit_cols].astype(float)
    
    # Create a sample index (1 to number of Timer_Page Submit columns)
    sample_indices = range(1, len(timer_submit_cols) + 1)
    
    plt.plot(sample_indices, values, marker='o', linestyle='-', label=f'Group {group}')

plt.xlabel("Sample Index")
plt.ylabel("Time (s)")
#plt.title("Reading Times for Groups A, B and C")
plt.xticks(sample_indices)
plt.legend()
plt.grid(True)
plt.show()
