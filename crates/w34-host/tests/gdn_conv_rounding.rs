//! Native causal-conv product materialization, including split continuation.
#![cfg(feature = "cuda")]
use core::ffi::c_void;
use serde::Deserialize;
use w34_host::cuda::{CudaDevice, CudaStream, DeviceBuffer};
use w34_kernels::ffi::{self, W34GdnBatch};

#[derive(Deserialize)]
struct Fixture {
    tokens: usize,
    channels: usize,
    kernel_size: usize,
    input: Vec<u16>,
    weight: Vec<u16>,
    output: Vec<u16>,
}

#[test]
fn native_conv_products_match_for_prefill_and_split_continuation() {
    let f: Fixture = serde_json::from_str(include_str!(
        "../../../tools/oracle/goldens/gdn_conv_real.json"
    ))
    .unwrap();
    assert_eq!(f.input.len(), f.tokens * f.channels);
    assert_eq!(f.output.len(), f.input.len());
    assert_eq!(f.weight.len(), f.channels * f.kernel_size);
    let device = CudaDevice::set_current(0).unwrap();
    let stream = CudaStream::new(device.device_id).unwrap();
    let ex = stream.exec(0, 0);
    let x = DeviceBuffer::alloc(device.device_id, f.input.len() * 2).unwrap();
    let w = DeviceBuffer::alloc(device.device_id, f.weight.len() * 2).unwrap();
    let y = DeviceBuffer::alloc(device.device_id, f.output.len() * 2).unwrap();
    x.copy_from_host(&f.input, &stream).unwrap();
    w.copy_from_host(&f.weight, &stream).unwrap();
    let state_elements = f.channels * (f.kernel_size - 1);
    let state = DeviceBuffer::alloc(device.device_id, state_elements * 2).unwrap();
    let loc = DeviceBuffer::alloc(device.device_id, 8).unwrap();
    let slot = DeviceBuffer::alloc(device.device_id, 4).unwrap();
    slot.copy_from_host(&[0i32], &stream).unwrap();
    let has_init = DeviceBuffer::alloc(device.device_id, 1).unwrap();
    for chunk in [f.tokens, 1, 3, 8] {
        y.copy_from_host(&vec![0x7BFFu16; f.output.len()], &stream)
            .unwrap();
        state
            .copy_from_host(&vec![0u16; state_elements], &stream)
            .unwrap();
        for start in (0..f.tokens).step_by(chunk) {
            let count = chunk.min(f.tokens - start);
            loc.copy_from_host(&[start as i32, (start + count) as i32], &stream)
                .unwrap();
            has_init
                .copy_from_host(&[u8::from(start > 0)], &stream)
                .unwrap();
            let batch = W34GdnBatch {
                query_start_loc: loc.addr() as *const i32,
                state_in_indices: slot.addr() as *const i32,
                state_out_indices: slot.addr() as *const i32,
                spec_state_indices: std::ptr::null(),
                checkpoint_dst_indices: std::ptr::null(),
                checkpoint_token_offsets: std::ptr::null(),
                has_initial_state: has_init.addr() as *const u8,
                valid_req_mask: std::ptr::null(),
                num_reqs: 1,
                total_tokens: f.tokens as i32,
                max_verify_tokens: 0,
                max_checkpoints_per_req: 0,
            };
            assert_eq!(
                // SAFETY: buffers cover their declared extents and remain live
                // until synchronization; each request uses the sole state slot.
                unsafe {
                    ffi::w34_gdn_causal_conv1d_f16(
                        &ex,
                        &batch,
                        x.addr() as *const c_void,
                        w.addr() as *const c_void,
                        std::ptr::null(),
                        state.addr() as *mut c_void,
                        y.addr() as *mut c_void,
                        f.channels as i32,
                        f.kernel_size as i32,
                        1,
                        1,
                    )
                },
                0
            );
        }
        let mut actual = vec![0u16; f.output.len()];
        let mut final_state = vec![0u16; state_elements];
        y.copy_to_host(&mut actual, &stream).unwrap();
        state.copy_to_host(&mut final_state, &stream).unwrap();
        stream.synchronize().unwrap();
        for (i, (&a, &e)) in actual.iter().zip(&f.output).enumerate() {
            assert_eq!(
                a,
                e,
                "native conv chunk={chunk}, token={}, channel={}",
                i / f.channels,
                i % f.channels
            );
        }
        for c in 0..f.channels {
            for j in 0..f.kernel_size - 1 {
                assert_eq!(
                    final_state[c * (f.kernel_size - 1) + j],
                    f.input[(f.tokens - (f.kernel_size - 1) + j) * f.channels + c]
                );
            }
        }
    }
}
