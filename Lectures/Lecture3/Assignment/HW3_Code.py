import pandas as pd
from tokenizers import Tokenizer, models, trainers, pre_tokenizers

# import train and test database
train_df = pd.read_csv("train.txt", header=None, names=["text"])
test_df = "he is unfair and unaware and unresponsive"

# Define the Hyper-parameters
vocab_sizes = [20, 50, 50]
min_freqs = [2, 2, 3]

# Define the tokenizers model
def token_model(min_freq, vocab_size):
    tokenizer = Tokenizer(models.BPE())
    # Trainer
    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        min_frequency=min_freq,
        special_tokens=["[UNK]", "[PAD]"]
    )
    # Train
    tokenizer.train_from_iterator(train_df["text"].tolist(), trainer)
    return tokenizer

for min_freq, vocab_size in zip(min_freqs, vocab_sizes):
    tokenizer = token_model(min_freq, vocab_size)
    output = tokenizer.encode(test_df)
    print(f"\n### min_freq={min_freq}, vocab_size={vocab_size}")
    print("Tokens:", output.tokens)
    print("IDs:", output.ids)

