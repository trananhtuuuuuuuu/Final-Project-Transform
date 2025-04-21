import pandas as pd
import numpy as np
from datasets import load_dataset
from sklearn.model_selection import train_test_split

# Load IMDB dataset from Hugging Face
dataset = load_dataset("imdb")

# Convert to Pandas DataFrame
df = pd.DataFrame(dataset['train'])

# Explore dataset
print("Dataset Info:")
print(df.info())
print("\nFirst 5 rows:")
print(df.head())
print("\nLabel distribution:")
print(df['label'].value_counts())

# Check for missing values
print("\nMissing values:")
print(df.isnull().sum())

# Clean text (basic preprocessing)
df['text'] = df['text'].str.lower().str.replace(r'<br />', ' ', regex=True)

# Convert labels to numpy array
labels = np.array(df['label'])

# Split data: 80% train, 10% validation, 10% test
train_texts, temp_texts, train_labels, temp_labels = train_test_split(
    df['text'], labels, test_size=0.2, random_state=42
)
val_texts, test_texts, val_labels, test_labels = train_test_split(
    temp_texts, temp_labels, test_size=0.5, random_state=42
)

# Save splits to CSV for reproducibility
train_df = pd.DataFrame({'text': train_texts, 'label': train_labels})
val_df = pd.DataFrame({'text': val_texts, 'label': val_labels})
test_df = pd.DataFrame({'text': test_texts, 'label': test_labels})

train_df.to_csv('train.csv', index=False)
val_df.to_csv('val.csv', index=False)
test_df.to_csv('test.csv', index=False)

print("\nData splits saved as CSV files.")
print(f"Train size: {len(train_df)}")
print(f"Validation size: {len(val_df)}")
print(f"Test size: {len(test_df)}")