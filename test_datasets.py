from datasets import load_dataset

# Test WikiText-2
try:
    ds = load_dataset('wikitext', 'wikitext-2-raw-v1', split='train')
    print(f'WikiText-2: SUCCESS - {len(ds)} examples')
    print(f'  Example: {ds[0]["text"][:100]}...')
except Exception as e:
    print(f'WikiText-2: FAILED - {e}')

# Test GLUE/SST-2
try:
    ds = load_dataset('glue', 'sst2', split='train')
    print(f'GLUE SST-2: SUCCESS - {len(ds)} examples')
    print(f'  Example: {ds[0]}')
except Exception as e:
    print(f'GLUE SST-2: FAILED - {e}')