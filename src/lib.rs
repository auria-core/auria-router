// File: lib.rs - This file is part of AURIA
// Copyright (c) 2026 AURIA Developers and Contributors
// Description:
//     Deterministic routing logic for expert selection in AURIA Runtime Core.
//     Implements routing strategies that select which experts to activate
//     for each inference step based on tier and token position.
//
use auria_core::{ExpertId, RoutingDecision, Tier};

pub trait Router: Send + Sync {
    fn route(&self, tier: Tier, token_index: u64) -> RoutingDecision;
}

pub struct DeterministicRouter;

impl Router for DeterministicRouter {
    fn route(&self, tier: Tier, token_index: u64) -> RoutingDecision {
        let k = match tier {
            Tier::Nano => 2,
            Tier::Standard => 4,
            Tier::Pro => 8,
            Tier::Max => 16,
        };
        let mut ids = Vec::with_capacity(k as usize);
        for i in 0..k {
            let val = ((token_index as u32) + i) % 1024;
            let mut bytes = [0u8; 32];
            bytes[0..4].copy_from_slice(&val.to_le_bytes());
            ids.push(ExpertId(bytes));
        }
        RoutingDecision { expert_ids: ids }
    }
}
