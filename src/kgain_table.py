import pandas as pd
import numpy as np
from pathlib import Path
from scipy import stats

# Config

mapping = {
    "A": "News",
    "B": "Abstract",
    "C": "Tweet/thread",
}

survey_files = {
    "A": ["a1", "a2", "a3"],
    "B": ["b1", "b2", "b3"],
    "C": ["c1", "c2", "c3"],
}

DATA_DIR = Path("../data/human_study/cleaned")

# If True: cells look like 18.17 $\pm$ 7.48
# If False: cells look like 18.17 [10.69, 25.65]
USE_PM_FORMAT = True

DIGITS = 2

# Helpers
def ci95(x):
    """
    Return mean and 95% CI half-width using the t distribution.
    """
    x = pd.Series(x).dropna().astype(float)
    n = len(x)

    if n == 0:
        return np.nan, np.nan

    mean = x.mean()

    if n == 1:
        return mean, np.nan

    sem = stats.sem(x, nan_policy="omit")
    h = sem * stats.t.ppf(0.975, n - 1)

    return mean, h


def fmt_mean_ci(x, digits=2):
    """
    Format mean with 95% CI.

    USE_PM_FORMAT=True:
        mean \\pm CI-half-width

    USE_PM_FORMAT=False:
        mean [CI-lower, CI-upper]
    """
    mean, h = ci95(x)

    if np.isnan(mean):
        return "--"

    if np.isnan(h):
        return f"{mean:.{digits}f}"

    if USE_PM_FORMAT:
        return f"{mean:.{digits}f} $\\pm$ {h:.{digits}f}"

    lo = mean - h
    hi = mean + h
    return f"{mean:.{digits}f} [{lo:.{digits}f}, {hi:.{digits}f}]"


def read_survey_file(fp):
    """
    Read one Qualtrics CSV file and return participant-level values:
    SC0, SC1, KG, Time_s, KG_per_min.
    """
    df = pd.read_csv(fp, skiprows=[1])

    # Drop Qualtrics ImportId row if it slipped through
    if len(df) > 0 and df.iloc[0].astype(str).str.contains("ImportId").any():
        df = df.iloc[1:].reset_index(drop=True)

    # Raw scores
    df[["SC0", "SC1"]] = df[["SC0", "SC1"]].apply(
        pd.to_numeric, errors="coerce"
    )

    df = df.dropna(subset=["SC0", "SC1"]).copy()

    df["KG"] = df["SC1"] - df["SC0"]

    # Reading time columns
    timer_cols = [c for c in df.columns if "Timer_Page Submit" in c]

    if len(timer_cols) == 0:
        df["Time_s"] = np.nan
    else:
        timers = df[timer_cols].apply(pd.to_numeric, errors="coerce")

        # Matches your old script's logic: average reading time across timer columns.
        # If you want total reading time instead, change mean(...) to sum(...).
        df["Time_s"] = timers.mean(axis=1, skipna=True)

    # KG per minute, participant-level
    df["KG_per_min"] = df["KG"] / (df["Time_s"] / 60)

    # Avoid infinities if a timer is zero
    df["KG_per_min"] = df["KG_per_min"].replace([np.inf, -np.inf], np.nan)

    return df[["SC0", "SC1", "KG", "KG_per_min", "Time_s"]]


 
# Collect results
 

rows = []

for group_id, subs in survey_files.items():
    group_dfs = []

    for sub in subs:
        fp = DATA_DIR / f"{sub}.csv"

        if not fp.exists():
            raise FileNotFoundError(f"Missing file: {fp}")

        sub_df = read_survey_file(fp)
        group_dfs.append(sub_df)

    group_df = pd.concat(group_dfs, ignore_index=True)

    rows.append({
        "Medium": mapping[group_id],
        "Pre Score": fmt_mean_ci(group_df["SC0"], DIGITS),
        "Post Score": fmt_mean_ci(group_df["SC1"], DIGITS),
        "KG": fmt_mean_ci(group_df["KG"], DIGITS),
        #"KG/min": fmt_mean_ci(group_df["KG_per_min"], DIGITS),
        "Time (s)": fmt_mean_ci(group_df["Time_s"], DIGITS),
        #"N": len(group_df),
    })


 
# Print LaTeX table
 

latex = r"""\begin{table*}[t]
\centering
\small
\begin{tabular}{lccccc r}
\toprule
Medium & Pre Score & Post Score & KG & Time (s) \\
\midrule
"""

for r in rows:
    latex += (
        f"{r['Medium']} & "
        f"{r['Pre Score']} & "
        f"{r['Post Score']} & "
        f"{r['KG']} & "
        #f"{r['KG/min']} & "
        f"{r['Time (s)']} \\\\\n "
        #f"{r['N']} \\\\\n"
    )

latex += r"""\bottomrule
\end{tabular}
\caption{Human validation results by medium. We report mean pre-reading score, post-reading score, {\sc KnowledgeGain} (KG), KG per minute, and reading time. Values are means with 95\% confidence intervals.}
\label{tab:human_kgain}
\end{table*}
"""

print(latex)