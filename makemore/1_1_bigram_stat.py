import os
import torch

script_dir = os.path.dirname(os.path.abspath(__file__))
names_path = os.path.join(script_dir, "names.txt")

with open(names_path, "r") as f:
    names = f.read().splitlines()

print(names[:16])
print(len(names))
print(min(len(x) for x in names))
print(max(len(x) for x in names))

chars = sorted(set("".join(names) + "."))
stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for i, ch in enumerate(chars)}
bigram_counts = torch.zeros((len(chars), len(chars)), dtype=torch.int32)
for name in names:
    chs = ["."] + list(name) + ["."]
    for a, b in zip(chs, chs[1:]):
        bigram_counts[stoi[a], stoi[b]] += 1

bigram_probs = bigram_counts.float() / bigram_counts.sum(dim=1, keepdim=True)

print(f"Bigram counts shape: {bigram_counts.shape}")
print("Bigram counts[:8, :8]:")
print(bigram_counts[:8, :8])
print(f"Bigram probs shape: {bigram_probs.shape}")
print("Bigram probs[:8, :8]:")
print(bigram_probs[:8, :8])

generated = []
for _ in range(32):
    name = []
    ix = stoi["."]
    while True:
        p = bigram_probs[ix]
        ix = int(torch.multinomial(p, num_samples=1).item())
        if ix == stoi["."]:
            break
        name.append(itos[ix])
    generated.append("".join(name))

print("Generated names:")
for name in generated:
    print(name)


def calculate_avg_nll(names, bigram_probs, stoi):
    nll = 0.0
    count = 0
    for name in names:
        chs = ["."] + list(name) + ["."]
        for a, b in zip(chs, chs[1:]):
            prob = bigram_probs[stoi[a], stoi[b]]
            nll -= prob.log().item()
            count += 1
    return nll / count


avg_nll = calculate_avg_nll(names, bigram_probs, stoi)
print(f"Average negative log-likelihood: {avg_nll:.4f}")

bigram_counts_smooth = bigram_counts + 1
bigram_probs_smooth = bigram_counts_smooth.float() / (
    bigram_counts_smooth.sum(dim=1, keepdim=True) + len(chars)
)
avg_nll_smooth = calculate_avg_nll(names, bigram_probs_smooth, stoi)
print(f"Average negative log-likelihood (smoothed): {avg_nll_smooth:.4f}")
