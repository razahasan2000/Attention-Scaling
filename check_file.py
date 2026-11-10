with open(r'C:\Users\Raza-SEAGUARD\Downloads\Attention\attention_scaling_fresh_study\src\data\prepare.py', 'rb') as f:
    content = f.read()
idx = content.find(b'vocab = {')
if idx >= 0:
    print(f'Found vocab at byte {idx}: {content[idx:idx+60]}')
else:
    print('Not found')