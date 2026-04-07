#![no_main]

use auria_router::*;
use libfuzzer_sys::fuzz_target;

fuzz_target!(|data: &[u8]| {
    if data.len() < 2 {
        return;
    }

    let num_experts = (data[0] as usize % 128).max(1);
    let tier_byte = data[1] % 4;
    let tier = match tier_byte {
        0 => Tier::Nano,
        1 => Tier::Standard,
        2 => Tier::Pro,
        _ => Tier::Max,
    };

    let router = DeterministicRouter::new(num_experts);
    let _decision = router.route(tier, 0);

    if data.len() >= 4 {
        let token = u32::from_le_bytes([data[0], data[1], data[2], data[3]]) as u64;
        let _decision2 = router.route(tier, token);
    }
});
