from datasets import load_dataset

ds = load_dataset('stanfordnlp/imdb', split='train')
print(f'IMDB train: {len(ds)} examples')

ds = load_dataset('stanfordnlp/imdb', split='test')
print(f'IMDB test: {len(ds)} examples')
print(f'Features: {ds.features}')
example = ds[0]
print(f'Example label: {example["label"]}')
print(f'Text preview: {example["text"][:80]}...')