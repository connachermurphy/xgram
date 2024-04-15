import torch

import names_data as names_data

# Cast as a 2D array
N = torch.zeros((len(names_data.chtoi), len(names_data.chtoi)), dtype=torch.int32)

for n in names_data.names:
    chs = ["."] + list(n) + ["."]
    for ch1, ch2 in zip(chs, chs[1:]):
        ich1 = names_data.chtoi[ch1]
        ich2 = names_data.chtoi[ch2]
        N[ich1, ich2] += 1

P = N.float()
P /= P.sum(dim=1, keepdim=True)


def draw_single(g):
    out = []
    ix = 0

    while True:
        p = P[ix]
        ix = torch.multinomial(p, num_samples=1, replacement=True, generator=g).item()

        if ix == 0:
            break
        else:
            out.append(names_data.itoch[ix])

    print("".join(out))


def draw_multiple(N, seed):
    g = torch.Generator().manual_seed(seed)

    for _ in range(N):
        draw_single(g)

if __name__ == "__main__":
    draw_multiple(N=20, seed=1234)
