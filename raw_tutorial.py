# Emotion Analysis using Hugging Face's Transformers
# Comments are notes written by me, some code is also commented out for faster execution.
# Author: Lewis Watson, Using "Natural Language Processing with Transformers" by Hugging Face
# Date: 27/02/2022

from datasets import load_dataset
from transformers import AutoTokenizer
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F

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
#plt.show()

# We can see the dataset is unbalanced, to solve this we can
# a) Randomly oversample the minority class
# b) Randomly undersample the majority class
# c) Gather more labled data
# This is not covered in this chapter but more information can be found here: https://oreil.ly/5XBhb


## Maximum Context Size
# The maximum context size is the maximum input sequence length of the transformer model.
# In the case of DistilBERT, the maximum context size is 512.

# Lets have a look at the distribution of words per tweet in the emotions database.

df["Words Per Tweet"] = df["text"].str.split().apply(len)
df.boxplot("Words Per Tweet", by="label_name", grid=False,
          showfliers=False, color="black")
plt.suptitle("")
plt.xlabel("")
#plt.show()

# We can see that the majority of tweets are less than 20 words, and the longest are still under DistilBERTs maximum context size of 512.

# Reset formatting of the dataset as we dont need to visualise any more.
emotions.reset_format()


# Character Tokenization
text = "Tokenizing text is a core task of NLP."
tokenized_text = list(text)
print(tokenized_text)

# Our model expects each token to be represented by an integer, a simple way to do this is to encode each unique token with a unique integer.
token2idx = {ch: idx for idx, ch in enumerate(sorted(set(tokenized_text)))}
print(token2idx)
# Now we can map our tokens to the integers.
input_ids = [token2idx[token] for token in tokenized_text]
print(input_ids)

# We must now convert our input_ids to a 2D tensor of one-hot encoding vectors.
input_ids = torch.tensor(input_ids)
one_hot_encodings = F.one_hot(input_ids, num_classes=len(token2idx))
print(one_hot_encodings.shape)

# Lets Examine the first vector
print(f"Token: {tokenized_text[0]}")
print(f"Tensor index: {input_ids[0]}")
print(f"One-hot: {one_hot_encodings[0]}")

# This approach is not very good as it loses linguistic structures such as words, which could be learned this way
# however, greatly increases the complexity of the training process. Word tokenization is used to solve this.


# Word Tokenization

# Split the text into words.
tokenized_text = text.split()
print(tokenized_text)

# Following the previous example, we would now map each word to an integer. However, one problem with this is
# punctuation so, "NLP." is treated as a single token. Given words can often have deviations like this (or such as
# misspellings). This would leave us with a wasteful sized vocabulary.

# One option is to use only the top N most frequent words.
# and mapping unknown words to the same "unk" token.
# However, another option is subword tokenization.

# Subword Tokenization - combining character and word tokenization.

# BERT uses 'WordPiece' tokenization. Lets see it in action.
model_ckpt = "distilbert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_ckpt)

encoded_text = tokenizer(text) # Lets feed it our "Tokenizing text is a core task of NLP." example text.
print(encoded_text) # We get unique ids!

# Lets now decode the ids back to words.
tokens = tokenizer.convert_ids_to_tokens(encoded_text.input_ids)
print(tokens) # Tokenizing and NLP have been split, this is expected as they are not common words.
# the '##' prefix is used to indicate that the token is a subword.

# Lets see it as a string
print(tokenizer.convert_tokens_to_string(tokens))


# Tokenizing the whole dataset

