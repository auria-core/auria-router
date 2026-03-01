// File: lib.rs - This file is part of AURIA
// Copyright (c) 2026 AURIA Developers and Contributors
// Description:
//     Deterministic routing logic for expert selection in AURIA Runtime Core.
//     Implements routing strategies that select which experts to activate
//     for each inference step based on tier, token position, and gating weights.
//
use auria_core::{ExpertId, RoutingDecision, Tier};
use std::collections::HashMap;

pub trait Router: Send + Sync {
    fn route(&self, tier: Tier, token_index: u64) -> RoutingDecision;
    fn route_with_weights(
        &self,
        tier: Tier,
        token_index: u64,
        weights: &HashMap<ExpertId, f32>,
    ) -> RoutingDecision;
}

pub struct DeterministicRouter {
    expert_count: u32,
}

impl DeterministicRouter {
    pub fn new(expert_count: u32) -> Self {
        Self { expert_count }
    }

    fn get_top_k_experts(&self, token_index: u64, k: u32) -> Vec<ExpertId> {
        let mut ids = Vec::with_capacity(k as usize);
        for i in 0..k {
            let val = ((token_index as u32) + i) % self.expert_count.max(1);
            let mut bytes = [0u8; 32];
            bytes[0..4].copy_from_slice(&val.to_le_bytes());
            ids.push(ExpertId(bytes));
        }
        ids
    }
}

impl Router for DeterministicRouter {
    fn route(&self, tier: Tier, token_index: u64) -> RoutingDecision {
        let k = match tier {
            Tier::Nano => 2,
            Tier::Standard => 4,
            Tier::Pro => 8,
            Tier::Max => 16,
        };
        let ids = self.get_top_k_experts(token_index, k);
        RoutingDecision { expert_ids: ids }
    }

    fn route_with_weights(
        &self,
        tier: Tier,
        token_index: u64,
        weights: &HashMap<ExpertId, f32>,
    ) -> RoutingDecision {
        let k = match tier {
            Tier::Nano => 2,
            Tier::Standard => 4,
            Tier::Pro => 8,
            Tier::Max => 16,
        };

        let mut sorted: Vec<_> = weights.iter().collect();
        sorted.sort_by(|a, b| b.1.partial_cmp(a.1).unwrap_or(std::cmp::Ordering::Equal));

        let ids: Vec<ExpertId> = sorted.iter().take(k as usize).map(|(id, _)| **id).collect();

        RoutingDecision { expert_ids: ids }
    }
}

pub struct GatingRouter {
    gate_weights: HashMap<ExpertId, f32>,
    temperature: f32,
}

impl GatingRouter {
    pub fn new(temperature: f32) -> Self {
        Self {
            gate_weights: HashMap::new(),
            temperature: temperature.max(0.01),
        }
    }

    pub fn set_gate_weight(&mut self, expert_id: ExpertId, weight: f32) {
        self.gate_weights.insert(expert_id, weight);
    }

    pub fn set_gate_weights(&mut self, weights: HashMap<ExpertId, f32>) {
        self.gate_weights = weights;
    }

    fn softmax(weights: &HashMap<ExpertId, f32>, temperature: f32) -> HashMap<ExpertId, f32> {
        let max_weight = weights.values().cloned().fold(f32::NEG_INFINITY, f32::max);

        let exp_weights: Vec<(ExpertId, f32)> = weights
            .iter()
            .map(|(id, w)| (*id, ((w - max_weight) / temperature).exp()))
            .collect();

        let sum: f32 = exp_weights.iter().map(|(_, e)| e).sum();

        exp_weights.iter().map(|(id, e)| (*id, e / sum)).collect()
    }
}

impl Router for GatingRouter {
    fn route(&self, tier: Tier, _token_index: u64) -> RoutingDecision {
        let k = match tier {
            Tier::Nano => 2,
            Tier::Standard => 4,
            Tier::Pro => 8,
            Tier::Max => 16,
        };

        let probs = Self::softmax(&self.gate_weights, self.temperature);

        let mut sorted: Vec<_> = probs.iter().collect();
        sorted.sort_by(|a, b| b.1.partial_cmp(a.1).unwrap_or(std::cmp::Ordering::Equal));

        let ids: Vec<ExpertId> = sorted.iter().take(k as usize).map(|(id, _)| **id).collect();

        RoutingDecision { expert_ids: ids }
    }

    fn route_with_weights(
        &self,
        tier: Tier,
        token_index: u64,
        _weights: &HashMap<ExpertId, f32>,
    ) -> RoutingDecision {
        self.route(tier, token_index)
    }
}

pub struct RoundRobinRouter {
    experts: Vec<ExpertId>,
    current: std::sync::atomic::AtomicUsize,
}

impl RoundRobinRouter {
    pub fn new(experts: Vec<ExpertId>) -> Self {
        Self {
            experts,
            current: std::sync::atomic::AtomicUsize::new(0),
        }
    }
}

impl Router for RoundRobinRouter {
    fn route(&self, tier: Tier, _token_index: u64) -> RoutingDecision {
        let k = match tier {
            Tier::Nano => 2,
            Tier::Standard => 4,
            Tier::Pro => 8,
            Tier::Max => 16,
        };

        if self.experts.is_empty() {
            return RoutingDecision {
                expert_ids: Vec::new(),
            };
        }

        let start = self
            .current
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        let ids: Vec<ExpertId> = (0..k)
            .map(|i| self.experts[(start + i as usize) % self.experts.len()])
            .collect();

        RoutingDecision { expert_ids: ids }
    }

    fn route_with_weights(
        &self,
        tier: Tier,
        token_index: u64,
        _weights: &HashMap<ExpertId, f32>,
    ) -> RoutingDecision {
        self.route(tier, token_index)
    }
}

pub enum AnyRouter {
    Deterministic(DeterministicRouter),
    Gating(GatingRouter),
    RoundRobin(RoundRobinRouter),
}

impl Router for AnyRouter {
    fn route(&self, tier: Tier, token_index: u64) -> RoutingDecision {
        match self {
            AnyRouter::Deterministic(r) => r.route(tier, token_index),
            AnyRouter::Gating(r) => r.route(tier, token_index),
            AnyRouter::RoundRobin(r) => r.route(tier, token_index),
        }
    }

    fn route_with_weights(
        &self,
        tier: Tier,
        token_index: u64,
        weights: &HashMap<ExpertId, f32>,
    ) -> RoutingDecision {
        match self {
            AnyRouter::Deterministic(r) => r.route_with_weights(tier, token_index, weights),
            AnyRouter::Gating(r) => r.route_with_weights(tier, token_index, weights),
            AnyRouter::RoundRobin(r) => r.route_with_weights(tier, token_index, weights),
        }
    }
}

pub fn create_default_router() -> DeterministicRouter {
    DeterministicRouter::new(1024)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_deterministic_router() {
        let router = DeterministicRouter::new(1024);
        let decision = router.route(Tier::Standard, 0);
        assert_eq!(decision.expert_ids.len(), 4);
    }

    #[test]
    fn test_gating_router() {
        let router = GatingRouter::new(1.0);
        let mut weights = HashMap::new();
        weights.insert(ExpertId([1u8; 32]), 0.5);
        weights.insert(ExpertId([2u8; 32]), 0.3);
        weights.insert(ExpertId([3u8; 32]), 0.2);

        let decision = router.route(Tier::Nano, 0);
        assert_eq!(decision.expert_ids.len(), 2);
    }
}
