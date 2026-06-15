# OpenRain

**Decentralized training of open AI models with formal security guarantees.**

Anyone with a GPU can contribute to training powerful, unrestricted AI models. No geofencing. No corporate control. No permission needed.

## What is this?

OpenRain is a P2P network where volunteers contribute compute to train open-source language models. Paid verifiers (earning $RAIN tokens) ensure training integrity and catch hallucinations. The resulting models are released under Apache 2.0 with no restrictions.

## Key Results

| | |
|---|---|
| **Security guarantee** | 93% proven (path to 99%) |
| **Adversary tolerance** | 10% malicious nodes |
| **Key finding** | Security is independent of adversary's computational budget |
| **Headline property** | System security improves monotonically with operation time |

## Read the Whitepaper

[OpenRain.md](./OpenRain.md)

## Core Components

| Component | Purpose |
|-----------|---------|
| Adaptive DiLoCo | Distributed training with sparse communication (works over home internet) |
| $RAIN Token | Pays verifiers for ensuring model quality |
| NeurASM | Neural Assembly Language for auditable reasoning traces |
| 4-Layer Defense | Randomized filters + effect verification + temporal detection + audits |
| Cross-Model Voting | Independent models verify each other (low error correlation) |

## Participate

```bash
# Coming soon
docker run openrain/node
```

## Author

Renan Zapelini

## License

Code: Apache 2.0  
Whitepaper: CC-BY-SA 4.0  
Models: Apache 2.0
