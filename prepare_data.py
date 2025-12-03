from transformers import AutoTokenizer
from datasets import load_dataset, DatasetDict
import multiprocessing as mp

BLOCK_SIZE = 512
NUM_PROCS = mp.cpu_count()
SAVE_DIR = "tokenized_wiki"

def tokenize_function(batch):
    return tokenizer(
        batch['text'],
        truncation=True,
        max_length=512
    )

# def group_texts(examples):
#     # Convert tensors -> lists if needed
#     input_ids = examples["input_ids"]
#     input_ids = [
#         ids.tolist() if hasattr(ids, "tolist") else ids
#         for ids in input_ids
#     ]

#     # Flatten all tokens from the batch into one long list
#     concatenated = sum(input_ids, [])
#     total_length = len(concatenated)

#     # Drop the remainder that doesn't fit block_size
#     total_length = (total_length // BLOCK_SIZE) * BLOCK_SIZE

#     # Split into equal-sized blocks
#     result = {
#         "input_ids": [
#             concatenated[i:i + BLOCK_SIZE]
#             for i in range(0, total_length, BLOCK_SIZE)
#         ]
#     }

#     return result

tokenizer = AutoTokenizer.from_pretrained('gpt2')

if __name__=='__main__':

    print('Loading data...')
    data = load_dataset('wikimedia/wikipedia', '20231101.en')

    print('Tokenizing data...')
    tokenized = data.map(
        tokenize_function,
        batched=True,
        batch_size=256,
        num_proc=NUM_PROCS,
        remove_columns=['text']
        )
    
    # print("Grouping into blocks...")
    # tokenized = tokenized.map(
    #     group_texts,
    #     batched=True,
    #     batch_size=1000,
    #     num_proc=NUM_PROCS,
    # )

    print('Saving the data...')
    tokenized.save_to_disk('tokenized_wiki')

    print('Done')