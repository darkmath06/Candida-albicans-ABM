import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 1. Load the data
df = pd.read_csv("Data/2026-03-25_Exp6_Resuscitation_Spatial_TimeSeries.csv")

# Set up visual style for the plots
sns.set_theme(style="whitegrid")

# ---------------------------------------------------------
# Plot 1: Overall Population Dynamics over Time
# ---------------------------------------------------------
plt.figure(figsize=(10, 6))
sns.lineplot(data=df, x='Time_Hours', y='Alive', hue='Genotype', style='Resuscitation_State', errorbar=None)
plt.title('Population Dynamics Over Time by Genotype and Resuscitation State')
plt.ylabel('Number of Alive Cells')
plt.xlabel('Time (Hours)')
plt.tight_layout()
plt.show()

# ---------------------------------------------------------
# Plot 2: Impact of Spatial Mode on Survival
# ---------------------------------------------------------
plt.figure(figsize=(10, 6))
sns.lineplot(data=df, x='Time_Hours', y='Alive', hue='Spatial_Mode', style='Genotype', errorbar=None)
plt.title('Impact of Spatial Mode on Alive Cells over Time')
plt.ylabel('Number of Alive Cells')
plt.xlabel('Time (Hours)')
plt.tight_layout()
plt.show()

# ---------------------------------------------------------
# Plot 3: Death Modes (Apoptosis vs Necrosis)
# ---------------------------------------------------------
# Melt the dataframe so we can plot both death types on the same axis easily
df_melted_death = pd.melt(df, id_vars=['Time_Hours', 'Genotype'], 
                          value_vars=['Dead_Apop', 'Dead_Necro'], 
                          var_name='Death_Type', value_name='Cell_Count')

plt.figure(figsize=(10, 6))
sns.lineplot(data=df_melted_death, x='Time_Hours', y='Cell_Count', hue='Death_Type', style='Genotype', errorbar=None)
plt.title('Cell Death Modes Over Time (Apoptosis vs Necrosis)')
plt.ylabel('Cumulative Dead Cells')
plt.xlabel('Time (Hours)')
plt.tight_layout()
plt.show()

# ---------------------------------------------------------
# Plot 4: Final Outcomes by Dose
# ---------------------------------------------------------
# Extract the final time point for each experimental run
max_time = df['Time_Hours'].max()
df_final = df[df['Time_Hours'] == max_time]

plt.figure(figsize=(10, 6))
sns.barplot(data=df_final, x='Dose', y='Alive', hue='Genotype')
plt.title('Final Number of Alive Cells by Dose and Genotype')
plt.ylabel('Alive Cells at End of Experiment')
plt.xlabel('Dose Level')
plt.tight_layout()
plt.show()