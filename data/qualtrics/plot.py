import pandas as pd
import matplotlib.pyplot as plt

# Read CSV file (assumes the file is named "data.csv")
# Skip the second header row so only the first header is used
group = "B"
df = pd.read_csv(f'testgroup{group}.csv', skiprows=[1])

# Check if the first row is the mapping row and drop it if needed
if df.iloc[0].astype(str).str.contains("ImportId").any():
    df = df.iloc[1:].reset_index(drop=True)

# Extract columns that contain "Timer_Page Submit"
timer_submit_cols = [col for col in df.columns if "Timer_Page Submit" in col]

# Use the first actual data row for plotting
values = df.loc[0, timer_submit_cols].astype(float)

# Create a sample index (1 to number of Timer_Page Submit columns)
sample_indices = range(1, len(timer_submit_cols) + 1)

plt.figure()
plt.plot(sample_indices, values, marker='o', linestyle='-')
plt.xlabel("Sample Index")
plt.ylabel("Time")
plt.title(f"Reading Times for Group {group}")
plt.xticks(sample_indices)
plt.grid(True)
plt.show()