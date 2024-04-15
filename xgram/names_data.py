# Read Karpathy's names file
names = open("data/names.txt").read().splitlines()

# Create character set
chars = sorted(list(set("".join(names))))
chtoi = {ch: i + 1 for i, ch in enumerate(chars)}
chtoi["."] = 0

itoch = {i: ch for ch, i in chtoi.items()}