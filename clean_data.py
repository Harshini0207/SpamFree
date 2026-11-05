import pandas as pd

# Load original dataset
df = pd.read_csv("dataset/emails.csv")

print("Original dataset:", df.shape)

# Check duplicates with conflicting labels
label_counts = df.groupby("email")["label"].nunique()

conflicting_emails = label_counts[label_counts > 1].index

print("\nEmails with conflicting labels:",
      len(conflicting_emails))

# Remove emails whose duplicate copies have different labels
df = df[~df["email"].isin(conflicting_emails)]

# Remove exact duplicate emails
df = df.drop_duplicates(subset="email", keep="first")

# Reset index
df = df.reset_index(drop=True)

print("\nCleaned dataset:", df.shape)

print("\nFinal label distribution:")
print(df["label"].value_counts())

print("\nMissing values:")
print(df.isnull().sum())

# Save cleaned dataset
df.to_csv("dataset/cleaned_emails.csv", index=False)

print("\nCleaned dataset saved successfully!")
