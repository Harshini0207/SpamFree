import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load dataset
df = pd.read_csv("dataset/emails.csv")

print("========== DATASET OVERVIEW ==========")

# Dataset dimensions
print("\nDataset Shape:", df.shape)

# First five records
print("\nFirst 5 Emails:")
print(df.head())

# Check missing values
print("\nMissing Values:")
print(df.isnull().sum())

# Check duplicate emails
print("\nDuplicate Emails:", df.duplicated(subset=["email"]).sum())

# Class distribution
print("\nEmail Distribution:")
print(df["label"].value_counts())

# Add email length column
df["email_length"] = df["email"].astype(str).apply(len)

print("\nEmail Length Statistics:")
print(df.groupby("label")["email_length"].describe())

# Plot class distribution
plt.figure(figsize=(7, 5))
sns.countplot(data=df, x="label")
plt.title("Spam vs Ham Email Distribution")
plt.xlabel("Email Label (0 = Ham, 1 = Spam)")
plt.ylabel("Number of Emails")
plt.savefig("dataset/class_distribution.png")
plt.show()

# Plot email length distribution
plt.figure(figsize=(8, 5))
sns.histplot(
    data=df,
    x="email_length",
    hue="label",
    bins=50,
    element="step"
)
plt.title("Email Length Distribution")
plt.xlabel("Email Length (Characters)")
plt.ylabel("Frequency")
plt.xlim(0, 10000)
plt.savefig("dataset/email_length_distribution.png")
plt.show()

print("\nEDA completed successfully!")
