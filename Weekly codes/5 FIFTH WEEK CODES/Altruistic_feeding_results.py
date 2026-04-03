import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import numpy as np

# 1. Load the data
try:
    df = pd.read_csv("IdeaA_NutrientAltruism_Results.csv")
    print("Data loaded successfully!")
except FileNotFoundError:
    print("Error: Could not find 'IdeaA_NutrientAltruism_Results.csv'. Make sure it is in the same folder.")
    exit()

# Set up the plotting style
sns.set_theme(style="whitegrid", context="paper", font_scale=1.2)

# ==========================================
# --- PLOT 1: SURVIVABILITY BY DURATION ---
# ==========================================
# We want to see how the duration of apoptosis affects final survival.
# We will facet this by the leak rate.

g = sns.catplot(
    data=df, 
    x="Apoptosis_Duration", 
    y="Survivability_Pct", 
    hue="Strain", 
    col="Apoptosis_Leak_Rate", 
    kind="bar", 
    capsize=.1, 
    err_kws={'linewidth': 1.5},
    palette={"PCD+": "blue", "PCD-": "red"},
    height=5, 
    aspect=0.8
)

g.fig.subplots_adjust(top=0.85)
g.fig.suptitle("Colony Survivability vs. Apoptosis Duration\n(Faceted by Nutrient Leak Rate)", fontsize=16)
g.set_axis_labels("Apoptosis Duration (Hours)", "Final Survivability (%)")
g.set_titles("Leak Rate: {col_name} / hr")
plt.savefig("IdeaA_Survivability_Plot.png", dpi=300, bbox_inches='tight')
print("Saved Survivability Plot as 'IdeaA_Survivability_Plot.png'")

# ==========================================
# --- PLOT 2: TOTAL BIOMASS ADVANTAGE ---
# ==========================================
# Sometimes survival % is the same, but the surviving cells are healthier/larger.
# Let's look at Total Live Biomass.

plt.figure(figsize=(10, 6))
ax = sns.boxplot(
    data=df,
    x="Apoptosis_Duration",
    y="Total_Live_Biomass",
    hue="Strain",
    palette={"PCD+": "blue", "PCD-": "red"}
)
plt.title("Total Live Biomass at End of Simulation\n(Aggregated across all Leak Rates)", fontsize=16)
plt.xlabel("Apoptosis Duration (Hours)")
plt.ylabel("Total Live Biomass (Arbitrary Units)")
plt.legend(title="Strain")
plt.savefig("IdeaA_Biomass_Plot.png", dpi=300, bbox_inches='tight')
print("Saved Biomass Plot as 'IdeaA_Biomass_Plot.png'")

# ==========================================
# --- STATISTICAL ANALYSIS (T-TESTS) ---
# ==========================================
print("\n" + "="*50)
print("STATISTICAL ANALYSIS: PCD+ vs PCD-")
print("="*50)

# We want to find EXACTLY which conditions make PCD+ statistically better than PCD-
durations = df['Apoptosis_Duration'].unique()
leak_rates = df['Apoptosis_Leak_Rate'].unique()

results_stats = []

for dur in durations:
    for leak in leak_rates:
        # Filter data for this specific condition
        subset = df[(df['Apoptosis_Duration'] == dur) & (df['Apoptosis_Leak_Rate'] == leak)]
        
        plus_data = subset[subset['Strain'] == 'PCD+']['Total_Live_Biomass']
        minus_data = subset[subset['Strain'] == 'PCD-']['Total_Live_Biomass']
        
        # Only run t-test if we have enough data and variance
        if len(plus_data) > 1 and len(minus_data) > 1 and plus_data.var() > 0 and minus_data.var() > 0:
            t_stat, p_val = stats.ttest_ind(plus_data, minus_data, equal_var=False) # Welch's t-test
            
            mean_diff = plus_data.mean() - minus_data.mean()
            
            # Determine significance (alpha = 0.05)
            sig = "***" if p_val < 0.001 else "**" if p_val < 0.01 else "*" if p_val < 0.05 else "ns"
            
            results_stats.append({
                'Duration': dur,
                'Leak_Rate': leak,
                'PCD+_Mean': plus_data.mean(),
                'PCD-_Mean': minus_data.mean(),
                'Diff': mean_diff,
                'p_value': p_val,
                'Significance': sig
            })

stats_df = pd.DataFrame(results_stats)

if not stats_df.empty:
    stats_df = stats_df.sort_values(by=['Diff'], ascending=False) # Sort by biggest advantage to PCD+
    
    print("\nTop Conditions where PCD+ outperformed PCD- (By Biomass Difference):")
    print(stats_df[['Duration', 'Leak_Rate', 'Diff', 'p_value', 'Significance']].head(10).to_string(index=False))
    
    # Export the stats to a CSV for your records
    stats_df.to_csv("IdeaA_Statistical_Analysis.csv", index=False)
    print("\nFull statistical results saved to 'IdeaA_Statistical_Analysis.csv'")
else:
    print("\nNo statistical comparisons could be computed.")
    print("This happens if you only have 1 replicate per condition, or if all replicates had the exact same final biomass (0 variance).")
    print("Check your 'IdeaA_NutrientAltruism_Results.csv' to see the raw numbers.")