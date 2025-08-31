import os
from collections import defaultdict

script_dir = os.path.dirname(os.path.abspath(__file__))
names_path = os.path.join(script_dir, "names.txt")

with open(names_path, "r") as f:
    names = f.read().splitlines()

print(names[:16])
print(len(names))
print(min(len(x) for x in names))
print(max(len(x) for x in names))

bigram_counts = defaultdict(int)
for name in names:
    chs = ["."] + list(name) + ["."]
    for a, b in zip(chs, chs[1:]):
        bigram_counts[(a, b)] += 1

for k, v in list(bigram_counts.items())[:8]:
    print(f"{k}: {v}")
