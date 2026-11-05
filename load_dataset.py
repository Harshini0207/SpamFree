import os
import pandas as pd
from email import policy
from email.parser import BytesParser

# Dataset paths
HAM_PATH = "dataset/easy_ham_2"
SPAM_PATH = "dataset/spam_2"


def extract_email_text(file_path):
    try:
        with open(file_path, "rb") as file:
            msg = BytesParser(policy=policy.default).parse(file)

        parts = []

        for part in msg.walk():

            # Process only plain text
            if part.get_content_type() != "text/plain":
                continue

            # Ignore attachments
            if part.get_content_disposition() == "attachment":
                continue

            payload = part.get_payload(decode=True)

            if payload is None:
                content = part.get_payload()
                if isinstance(content, str):
                    parts.append(content)
                continue

            charset = part.get_content_charset()

            try:
                content = payload.decode(charset or "utf-8")
            except (LookupError, UnicodeDecodeError):
                # Fallback for unknown or invalid encodings
                try:
                    content = payload.decode("utf-8", errors="replace")
                except Exception:
                    content = payload.decode("latin-1", errors="replace")

            parts.append(content)

        return " ".join(parts)

    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        return ""



data = []
# Read Ham emails
for filename in os.listdir(HAM_PATH):
    file_path = os.path.join(HAM_PATH, filename)

    if os.path.isfile(file_path):
        text = extract_email_text(file_path)

        if text.strip():
            data.append({"email": text, "label": 0})

# Read Spam emails
for filename in os.listdir(SPAM_PATH):
    file_path = os.path.join(SPAM_PATH, filename)

    if os.path.isfile(file_path):
        text = extract_email_text(file_path)

        if text.strip():
            data.append({"email": text, "label": 1})

# Create DataFrame
df = pd.DataFrame(data)

print("\nDataset loaded successfully!")
print("\nDataset shape:", df.shape)
print("\nLabel distribution:")
print(df["label"].value_counts())

print("\nFirst email:")
print(df.iloc[0]["email"][:500])

# Save processed dataset
df.to_csv("dataset/emails.csv", index=False)

print("\nDataset saved to dataset/emails.csv")
