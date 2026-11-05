#![cfg(feature = "cuda")]
//! Host/GPU equivalence tests for the token-wise rejection seam, not full gates.
use w34_host::cuda::{CudaDevice, CudaStream, DeviceBuffer, check_rc};
use w34_host::speculative::RejectionSampler;
use w34_kernels::ffi;

fn upload<T: Copy>(values: &[T], stream: &CudaStream) -> DeviceBuffer {
    let buffer = DeviceBuffer::alloc(0, std::mem::size_of_val(values).max(8)).unwrap();
    if !values.is_empty() {
        buffer.copy_from_host(values, stream).unwrap();
    }
    buffer
}

#[test]
fn greedy_gpu_matches_host_at_each_rejection_and_masks_padding() {
    let _device = CudaDevice::set_current(0).unwrap();
    let stream = CudaStream::new(0).unwrap();
    let ex = stream.exec(0, 0);
    for k in [0, 1, 2, 3, 7, 15, 16] {
        let reqs = k + 2;
        let mut drafts = vec![0; reqs * k];
        let mut targets = vec![0; reqs * (k + 1)];
        let mut valid = vec![1u8; reqs];
        valid[reqs - 1] = 0;
        for r in 0..reqs {
            for i in 0..k {
                drafts[r * k + i] = 100000 + i as i32;
                targets[r * (k + 1) + i] = 100000 + i as i32;
            }
            targets[r * (k + 1) + k] = 200000;
            if r <= k {
                targets[r * (k + 1) + r] = 200000;
            }
        }
        targets[(reqs - 1) * (k + 1)..].fill(-2);
        let d = upload(&drafts, &stream);
        let t = upload(&targets, &stream);
        let mask = upload(&valid, &stream);
        let out = upload(&vec![42i32; reqs * (k + 1)], &stream);
        let counts = upload(&vec![42i32; reqs], &stream);
        let status = upload(&vec![42i32; reqs], &stream);
        // SAFETY: buffers match the contiguous R,k dimensions for this launch.
        unsafe {
            check_rc(
                ffi::w34_rejection_greedy(
                    &ex,
                    d.addr() as *const i32,
                    t.addr() as *const i32,
                    mask.addr() as *const u8,
                    out.addr() as *mut i32,
                    counts.addr() as *mut i32,
                    status.addr() as *mut i32,
                    reqs as i32,
                    k as i32,
                ),
                "rejection greedy",
            )
            .unwrap();
        }
        let mut got = vec![0i32; reqs * (k + 1)];
        let mut accepted = vec![0i32; reqs];
        let mut errors = vec![0i32; reqs];
        out.copy_to_host(&mut got, &stream).unwrap();
        counts.copy_to_host(&mut accepted, &stream).unwrap();
        status.copy_to_host(&mut errors, &stream).unwrap();
        assert_eq!(errors, vec![0; reqs]);
        for r in 0..reqs - 1 {
            let expected = RejectionSampler::greedy(
                &drafts[r * k..(r + 1) * k],
                &targets[r * (k + 1)..(r + 1) * (k + 1)],
            )
            .unwrap();
            assert_eq!(accepted[r], expected.accepted_drafts() as i32);
            assert_eq!(
                &got[r * (k + 1)..r * (k + 1) + expected.processed_rows()],
                expected.emitted_tokens()
            );
            assert!(
                got[r * (k + 1) + expected.processed_rows()..(r + 1) * (k + 1)]
                    .iter()
                    .all(|&x| x == -1)
            );
        }
        assert_eq!(accepted[reqs - 1], -1);
        assert!(got[(reqs - 1) * (k + 1)..].iter().all(|&x| x == -1));
    }
}

#[test]
fn stochastic_gpu_matches_host_on_residuals_bonus_and_invalid_requests() {
    let _device = CudaDevice::set_current(0).unwrap();
    let stream = CudaStream::new(0).unwrap();
    let ex = stream.exec(0, 0);
    // Vocabulary spans multiple threads, including empty partitions at small V.
    for vocab in [2usize, 513] {
        for k in [0usize, 1, 3, 7, 15] {
            let reqs = k + 4;
            let mut target = vec![0.0f32; reqs * (k + 1) * vocab];
            let mut proposal = vec![0.0f32; reqs * k * vocab];
            let drafts = vec![0i32; reqs * k];
            let mut uniforms = vec![0.0f64; reqs * k];
            let recovery = vec![0.0f64; reqs];
            let mut valid = vec![1u8; reqs];
            for r in 0..reqs {
                for i in 0..=k {
                    let p = (r * (k + 1) + i) * vocab;
                    target[p] = 0.25;
                    target[p + vocab - 1] = 0.75;
                    if i < k {
                        let q = (r * k + i) * vocab;
                        proposal[q] = 0.75;
                        proposal[q + vocab - 1] = 0.25;
                        // Requests 0..k reject at row r; request k fully accepts.
                        if i == r {
                            uniforms[r * k + i] = 0.9;
                        }
                    }
                }
            }
            let invalid = reqs - 2;
            target[invalid * (k + 1) * vocab] = f32::NAN;
            valid[reqs - 1] = 0;
            target[(reqs - 1) * (k + 1) * vocab..].fill(f32::NAN);
            let d = upload(&drafts, &stream);
            let p = upload(&target, &stream);
            let q = upload(&proposal, &stream);
            let u = upload(&uniforms, &stream);
            let recover = upload(&recovery, &stream);
            let mask = upload(&valid, &stream);
            let out = upload(&vec![42i32; reqs * (k + 1)], &stream);
            let counts = upload(&vec![42i32; reqs], &stream);
            let status = upload(&vec![42i32; reqs], &stream);
            // SAFETY: all allocations cover declared rows and vocabulary widths;
            // invalid probability contents are checked by the kernel, not dereferenced as IDs.
            unsafe {
                check_rc(
                    ffi::w34_rejection_stochastic(
                        &ex,
                        d.addr() as *const i32,
                        p.addr() as *const f32,
                        q.addr() as *const f32,
                        u.addr() as *const f64,
                        recover.addr() as *const f64,
                        mask.addr() as *const u8,
                        out.addr() as *mut i32,
                        counts.addr() as *mut i32,
                        status.addr() as *mut i32,
                        reqs as i32,
                        k as i32,
                        vocab as i32,
                    ),
                    "rejection stochastic",
                )
                .unwrap();
            }
            let mut got = vec![0i32; reqs * (k + 1)];
            let mut accepted = vec![0i32; reqs];
            let mut errors = vec![0i32; reqs];
            out.copy_to_host(&mut got, &stream).unwrap();
            counts.copy_to_host(&mut accepted, &stream).unwrap();
            status.copy_to_host(&mut errors, &stream).unwrap();
            for r in 0..invalid {
                let p_rows: Vec<Vec<f32>> = target[r * (k + 1) * vocab..(r + 1) * (k + 1) * vocab]
                    .chunks(vocab)
                    .map(<[f32]>::to_vec)
                    .collect();
                let q_rows: Vec<Vec<f32>> = proposal[r * k * vocab..(r + 1) * k * vocab]
                    .chunks(vocab)
                    .map(<[f32]>::to_vec)
                    .collect();
                let expected = RejectionSampler::stochastic(
                    &drafts[r * k..(r + 1) * k],
                    &p_rows,
                    &q_rows,
                    &uniforms[r * k..(r + 1) * k],
                    recovery[r],
                )
                .unwrap();
                assert_eq!(errors[r], 0);
                assert_eq!(accepted[r], expected.accepted_drafts() as i32);
                assert_eq!(
                    &got[r * (k + 1)..r * (k + 1) + expected.processed_rows()],
                    expected.emitted_tokens()
                );
                assert!(
                    got[r * (k + 1) + expected.processed_rows()..(r + 1) * (k + 1)]
                        .iter()
                        .all(|&x| x == -1)
                );
            }
            assert_eq!(accepted[invalid], -1);
            assert_eq!(errors[invalid], 1);
            assert_eq!(accepted[reqs - 1], -1);
            assert_eq!(errors[reqs - 1], 0);
            assert!(got[invalid * (k + 1)..].iter().all(|&x| x == -1));
        }
    }
}
