# Emotion Analysis using Hugging Face's Transformers
# Author: Lewis Watson, Using "Natural Language Processing with Transformers" by Hugging Face
# Date: 27/02/2022

from datasets import load_dataset
import pandas as pd
import matplotlib.pyplot as plt

# Load the dataset
emotions = load_dataset("emotion")

# Split the dataset into training, test and validation sets
train_ds = emotions["train"]
valid_ds = emotions["validation"]
test_ds = emotions["test"]

print("Example dataset object:")
print(train_ds)

# Set emotions to pandas dataframe
emotions.set_format(type="pandas")
df = emotions["train"][:]
print(df.head())


# We can also obtain our string labels from the dataset
def label_int2str(row):
    return emotions["train"].features["label"].int2str(row)

df["label_name"] = df["label"].apply(label_int2str)
print(df.head())


# It is worth analysing the distribution of labels in the dataset
df["label_name"].value_counts(ascending=True).plot.barh()
plt.title("Frequency of Classes")
plt.show()

# We can see the dataset is unbalanced, to solve this we can
# a) Randomly oversample the minority class
# b) Randomly undersample the majority class
# c) Gather more labled data
# This is not covered in this chapter but more information can be found here: https://oreil.ly/5XBhb

