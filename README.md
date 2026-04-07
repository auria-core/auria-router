# auria-router

Deterministic routing for expert selection in AURIA Runtime Core.

## Overview

Implements routing logic that selects which experts to activate for each inference step.

## Routing Strategy

- Nano → Top-2 experts
- Standard → Top-4 experts
- Pro → Top-8 experts
- Max → Top-16 experts

## Usage

```rust
use auria_router::{DeterministicRouter, Router};
use auria_core::Tier;

let router = DeterministicRouter;
let decision = router.route(Tier::Standard, 0);
```
