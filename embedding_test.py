import torch
import torch.nn.functional as F
from transformers import AutoTokenizer
from models.transformer.transformer import Transformer
# from models.kat.kat import KAT
from models.mamba.mamba import MambaModel


def get_word_embedding(word, tokenizer, embedding_layer, device):
    'Get the embedding vector for a single word (uses first token if word splits into multiple).'
    token_ids = tokenizer.encode(word, add_special_tokens=False)
    if len(token_ids) > 1:
        print(f'  Warning: "{word}" tokenizes into {len(token_ids)} tokens, using first token only.')
    token_id = torch.tensor([token_ids[0]], device=device)
    with torch.no_grad():
        emb = embedding_layer(token_id)  # (1, embed_dim)
    return emb.squeeze(0)  # (embed_dim,)


def find_closest(query_vec, embedding_layer, tokenizer, top_k=5, exclude_words=None):
    'Find the closest tokens to query_vec by cosine similarity.'
    exclude_words = exclude_words or []
    exclude_ids = set()
    for w in exclude_words:
        ids = tokenizer.encode(w, add_special_tokens=False)
        exclude_ids.update(ids)

    with torch.no_grad():
        all_embeddings = embedding_layer.weight  # (vocab_size, embed_dim)
        query_norm = F.normalize(query_vec.unsqueeze(0), dim=-1)
        all_norm = F.normalize(all_embeddings, dim=-1)
        similarities = (all_norm @ query_norm.T).squeeze(1)  # (vocab_size,)

    # Mask out excluded tokens
    for idx in exclude_ids:
        similarities[idx] = -1.0

    top_scores, top_ids = similarities.topk(top_k)
    results = []
    for score, token_id in zip(top_scores, top_ids):
        word = tokenizer.decode([token_id.item()])
        results.append((word.strip(), score.item()))
    return results


def analogy(a, b, c, embedding_layer, tokenizer, device, top_k=5):
    '''
    Solve: a - b + c = ?
    Example: king - man + woman = queen
    '''
    print(f'\n  "{a}" - "{b}" + "{c}" = ?')
    va = get_word_embedding(a, tokenizer, embedding_layer, device)
    vb = get_word_embedding(b, tokenizer, embedding_layer, device)
    vc = get_word_embedding(c, tokenizer, embedding_layer, device)

    query = va - vb + vc

    results = find_closest(query, embedding_layer, tokenizer, top_k=top_k, exclude_words=[a, b, c])
    for word, score in results:
        print(f'    {word:<20} cosine sim: {score:.4f}')


if __name__ == '__main__':
    CHECKPOINT = 'models/mamba/training/checkpoint.pt'
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

    tokenizer = AutoTokenizer.from_pretrained('gpt2')

    # model = Transformer(depth=6, num_heads=8).to(DEVICE)
    model = MambaModel(d_model=512, d_hidden=2048, n_blocks=6).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters())

    ckpt = torch.load(CHECKPOINT, map_location=DEVICE)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()

    embedding_layer = model.embedding.embed

    print('=== Embedding Analogy Tests ===')

    analogies = [
        # (a,        b,        c)           expected: a - b + c
        ('Berlin', 'Germany','France'),   # Paris
    ]

    for a, b, c in analogies:
        analogy(a, b, c, embedding_layer, tokenizer, DEVICE, top_k=5)