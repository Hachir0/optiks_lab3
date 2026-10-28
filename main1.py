import numpy as np
import matplotlib.pyplot as plt
from collections import deque


def simulate(lam, M, T=200_000, p0=1.0, seed=None, warmup=0.1):
    rng = np.random.default_rng(seed)
    queues = [deque() for _ in range(M)]
    active = set()          # абоненты с непустой очередью
    p = p0
    total = 0               # сообщений в системе
    delays, sumN, out, cnt = [], 0.0, 0, 0
    t0 = int(T * warmup)

    for t in range(T):
        # состояние в начале окна 
        if t >= t0:
            sumN += total
            cnt += 1
        ready = list(active)
        if ready:
            tx = [i for i in ready if rng.random() < p]
        else:
            tx = []

        if len(tx) == 1:                       # успех
            i = tx[0]   
            a = queues[i].popleft()
            total -= 1
            if not queues[i]:
                active.discard(i)
            if t >= t0:
                delays.append(t + 1 - a)
                out += 1
        elif len(tx) > 1:                      # конфликт
            p = max(1.0 / M, p / 2)
        else:                                  # пусто
            p = min(1.0, 2 * p)

        # появление сообщений в течение окна
        for _ in range(rng.poisson(lam)):
            i = int(rng.integers(M))
            queues[i].append(t + rng.random())
            active.add(i)
            total += 1

    d = float(np.mean(delays)) if delays else float('nan')
    return d, sumN / cnt, out / cnt


def theory_sync(lam):
    N = lam * (2 - lam) / (2 * (1 - lam))
    return N, N / lam + 0.5


if __name__ == "__main__":
    # Проверка 1: M=1 должен совпасть с синхронной M|D|1 
    lams = np.arange(0.05, 0.96, 0.05)
    res = [simulate(l, 1, T=100_000, seed=1) for l in lams]
    fig, ax = plt.subplots(1, 3, figsize=(15, 4))
    ax[0].plot(lams, [r[0] for r in res], 'o', label='моделирование (M=1)')
    ax[0].plot(lams, [theory_sync(l)[1] for l in lams], '-', label='синхронная M|D|1')
    ax[0].set_title('d(λ)'); ax[0].set_ylim(0, 15)
    ax[1].plot(lams, [r[1] for r in res], 'o', label='моделирование (M=1)')
    ax[1].plot(lams, [theory_sync(l)[0] for l in lams], '-', label='синхронная M|D|1')
    ax[1].set_title('N(λ)'); ax[1].set_ylim(0, 15)
    ax[2].plot(lams, [r[2] for r in res], 'o', label='λ_вых')
    ax[2].plot(lams, lams, '-', label='λ')
    ax[2].set_title('λ_вых(λ)')
    for a in ax:
        a.set_xlabel('λ'); a.grid(); a.legend()
    plt.tight_layout(); plt.savefig('check_M1.png', dpi=130)

    # Основные графики: разные M 
    Ms = [2, 5, 10, 20]
    lams = np.arange(0.02, 0.50, 0.02)
    fig, ax = plt.subplots(1, 3, figsize=(15, 4))
    for M in Ms:
        r = [simulate(l, M, T=100_000, seed=2) for l in lams]
        ax[0].plot(lams, [x[0] for x in r], 'o-', ms=3, label=f'M={M}')
        ax[1].plot(lams, [x[1] for x in r], 'o-', ms=3, label=f'M={M}')
        ax[2].plot(lams, [x[2] for x in r], 'o-', ms=3, label=f'M={M}')
    ax[2].plot(lams, lams, 'k--', label='λ (входная)')
    ax[0].set_title('d(λ)'); ax[0].set_ylim(0, 40)
    ax[1].set_title('N(λ)'); ax[1].set_ylim(0, 40)
    ax[2].set_title('λ_вых(λ)')
    for a in ax:
        a.set_xlabel('λ'); a.grid(); a.legend()
    plt.tight_layout(); plt.savefig('adaptive_aloha.png', dpi=130)
    plt.show()