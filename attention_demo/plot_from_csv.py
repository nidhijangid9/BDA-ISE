# plot_from_csv.py
import pandas as pd
import matplotlib.pyplot as plt

CSV_LOG = "attention_log.csv"

# 1. Load data
df = pd.read_csv(CSV_LOG)
print(f"Loaded {len(df)} rows from {CSV_LOG}")
print(df.head())

# 2. Timeline plot for each student
plt.figure(figsize=(10, 6))
for oid, group in df.groupby("id"):
    plt.plot(group["frame"], group["score"], label=f"ID {oid}")
plt.xlabel("Frame")
plt.ylabel("Attention Score")
plt.title("Attention Score Over Time")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("attention_timeline.png")
plt.show()

# 3. Pie chart: attentive vs distracted
state_counts = df["state"].value_counts()
plt.figure(figsize=(5, 5))
plt.pie(state_counts, labels=state_counts.index, autopct="%1.1f%%", startangle=140)
plt.title("Overall Attention Distribution")
plt.savefig("attention_pie.png")
plt.show()

# 4. Average score per student (bar chart)
avg_scores = df.groupby("id")["score"].mean()
plt.figure(figsize=(8, 5))
avg_scores.plot(kind="bar", color="skyblue")
plt.ylabel("Average Attention Score")
plt.title("Average Attention Score per Student")
plt.grid(axis="y")
plt.tight_layout()
plt.savefig("attention_avg.png")
plt.show()

print("✅ Graphs saved as attention_timeline.png, attention_pie.png, attention_avg.png")
