"""
Emotion Dataset Conversion Script
Converts train/test/val .txt files → single emotion_dataset.csv
"""

import pandas as pd
import os


def load_emotion_file(file_path):
    """Read a text file of 'text;emotion' lines into a DataFrame"""
    data = []
    with open(file_path, 'r') as f:
        for line in f:
            if ';' in line:
                text, emotion = line.strip().split(';')
                data.append([text, emotion])
    return pd.DataFrame(data, columns=["text", "emotion"])


def main():
    # Change this if your data folder path differs
    DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")

    # Load all datasets
    train_path = os.path.join(DATA_DIR, "train.txt")
    test_path  = os.path.join(DATA_DIR, "test.txt")
    val_path   = os.path.join(DATA_DIR, "val.txt")

    print("Loading dataset files from:", DATA_DIR)

    train_df = load_emotion_file(train_path)
    test_df  = load_emotion_file(test_path)
    val_df   = load_emotion_file(val_path)

    # Combine them
    df = pd.concat([train_df, test_df, val_df]).reset_index(drop=True)

    # Clean and inspect
    print("Combined dataset shape:", df.shape)
    print("Sample rows:")
    print(df.head())

    print("\nEmotion distribution:")
    print(df["emotion"].value_counts())

    # Save as CSV
    output_path = os.path.join(DATA_DIR, "emotion_dataset.csv")
    df.to_csv(output_path, index=False, encoding='utf-8')
    print("\nSaved combined dataset to:", output_path)


if __name__ == "__main__":
    main()

