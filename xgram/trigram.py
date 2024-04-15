import torch

import names_data as names_data

# Cast as a 3D array
N = torch.zeros((len(names_data.chtoi), len(names_data.chtoi), len(names_data.chtoi)), dtype=torch.int32)

for n in names_data.names:
    chs = ["."] + list(n) + ["."]
    for ch1, ch2, ch3 in zip(chs, chs[1:], chs[2:]):
        ich1 = names_data.chtoi[ch1]
        ich2 = names_data.chtoi[ch2]
        ich3 = names_data.chtoi[ch3]
        N[ich1, ich2, ich3] += 1

P = N.float()

P_first = N[0].float().view(-1)
P_first /= P_first.sum()

P /= P.sum(dim=2, keepdim=True)

def draw_single(g):
    out = []

    ixflat = torch.multinomial(P_first, num_samples=1, replacement=True, generator=g).item()

    ixi, ixj = divmod(ixflat, len(names_data.chtoi))

    out.extend([names_data.itoch[ixi], names_data.itoch[ixj]])

    while True:
        p = P[ixi, ixj]
        ixk = torch.multinomial(p, num_samples=1, replacement=True, generator=g).item()

        if ixk == 0:
            break
        else:
            out.append(names_data.itoch[ixk])
            ixi, ixj = ixj, ixk
    
    print("".join(out))

def draw_multiple(N, seed):
    g = torch.Generator().manual_seed(seed)

    for _ in range(N):
        draw_single(g)

if __name__ == "__main__":
    draw_multiple(N=20, seed=1234)
