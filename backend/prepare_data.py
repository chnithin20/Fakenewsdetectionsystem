import os
import re
import pandas as pd
from sklearn.model_selection import train_test_split

# ── Configuration ──
DATA_DIR    = "e:/fakenews/backend/data"
DATASET_DIR = "e:/fakenews/backend/dataset"
SEED        = 42

def clean_text(text):
    """
    Remove URLs, HTML tags, special characters, and extra whitespace.
    """
    if not isinstance(text, str):
        return ""
    
    # 1. Lowercase
    text = text.lower()
    
    # 2. Remove URLs
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    
    # 3. Remove HTML tags
    text = re.sub(r'<.*?>', '', text)
    
    # 4. Remove special characters (keep punctuation for context in RoBERTa)
    text = re.sub(r'[^a-zA-Z0-9\s.,?!]', '', text)
    
    # 5. Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text

def prepare_pipeline():
    # 1. Verify files exist
    true_csv = os.path.join(DATASET_DIR, "True.csv")
    fake_csv = os.path.join(DATASET_DIR, "Fake.csv")
    
    if not os.path.exists(true_csv) or not os.path.exists(fake_csv):
        print(f"ERROR: Dataset files not found in {DATASET_DIR}. Please add True.csv and Fake.csv manually.")
        return

    print("Loading data...")
    df_true = pd.read_csv(true_csv)
    df_fake = pd.read_csv(fake_csv)

    # 2. Add labels: 0 = real, 1 = fake
    df_true['label'] = 0
    df_fake['label'] = 1

    # 3. Keep only relevant columns and combine
    df = pd.concat([df_true, df_fake], axis=0).reset_index(drop=True)

    # 4. Combine title + text
    print("Combining title and text...")
    df['content'] = df['title'] + " " + df['text']
    
    # 5. Clean text
    print("Cleaning text (URLs, HTML tags, special characters)...")
    df['content'] = df['content'].apply(clean_text)

    # Remove empty/too short contents
    df = df[df['content'].str.len() > 50].copy()

    # 6. Shuffle
    print("Shuffling dataset...")
    df = df.sample(frac=1, random_state=SEED).reset_index(drop=True)

    # 7. Split: 80% train, 10% val, 10% test (stratified)
    print("Splitting datasets (80/10/10)...")
    train_df, temp_df = train_test_split(
        df, 
        test_size=0.2, 
        random_state=SEED, 
        stratify=df['label']
    )
    val_df, test_df = train_test_split(
        temp_df, 
        test_size=0.5, 
        random_state=SEED, 
        stratify=temp_df['label']
    )

    # 8. Save outputs
    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR)
        
    print(f"Saving splits to {DATA_DIR}...")
    train_df[['content', 'label']].to_csv(os.path.join(DATA_DIR, "train.csv"), index=False)
    val_df[['content', 'label']].to_csv(os.path.join(DATA_DIR, "val.csv"), index=False)
    test_df[['content', 'label']].to_csv(os.path.join(DATA_DIR, "test.csv"), index=False)

    # 9. Verift Results
    print("\n" + "="*40)
    print("PREPARATION COMPLETE")
    print("="*40)
    print(f"Total Rows:     {len(df)}")
    print(f"Train Rows:     {len(train_df)}")
    print(f"Val Rows:       {len(val_df)}")
    print(f"Test Rows:      {len(test_df)}")
    print("\nClass Distribution (Train):")
    print(train_df['label'].value_counts(normalize=True))
    print("\nSample Rows:")
    print(train_df[['content', 'label']].head(3))

if __name__ == "__main__":
    prepare_pipeline()
