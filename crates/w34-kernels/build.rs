// Incremental parallel nvcc build for w34-kernels.
// Target architecture defaults to sm_86 (override via W34_ARCH).
use std::env::{var, var_os};
use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::thread;

fn main() {
    if var_os("CARGO_FEATURE_CUDA").is_none() {
        return;
    }

    let out = PathBuf::from(var("OUT_DIR").expect("OUT_DIR"));
    let cuda_home = var_os("CUDA_HOME")
        .map(PathBuf::from)
        .unwrap_or_else(|| "/opt/cuda".into());
    let nvcc = cuda_home.join("bin/nvcc");
    assert!(
        nvcc.exists(),
        "nvcc not found at {} (set CUDA_HOME or add /opt/cuda/bin to PATH)",
        nvcc.display()
    );

    let csrc = PathBuf::from(var("CARGO_MANIFEST_DIR").expect("CARGO_MANIFEST_DIR")).join("csrc");
    let mut sources = Vec::new();
    let mut headers = Vec::new();
    walk_csrc(&csrc, &mut sources, &mut headers);
    sources.sort();
    headers.sort();

    let header_epoch = newest_mtime(&headers).unwrap_or(0);
    let arch = var("W34_ARCH").unwrap_or_else(|_| "sm_86".into());
    let valid = arch.strip_prefix("sm_").is_some_and(|cc| {
        let mut chars = cc.chars();
        chars.next().is_some_and(|c| c.is_ascii_digit())
            && chars.all(|c| c.is_ascii_digit() || c.is_ascii_lowercase())
    });
    assert!(
        valid,
        "W34_ARCH must look like sm_86 / sm_89 / sm_90, got {arch:?}"
    );

    let extra = var("W34_NVCC_EXTRA").unwrap_or_default();
    let flags = format!(
        "-arch={arch} -O3 -lineinfo -std=c++17 --expt-relaxed-constexpr \
         -use_fast_math -extended-lambda {extra}"
    );

    let stamps = out.join("stamps");
    fs::create_dir_all(&stamps).expect("create stamps dir");

    let mut plan = Vec::new();
    for src in &sources {
        let stem = src.file_stem().unwrap().to_str().unwrap();
        let obj = out.join(format!("{stem}.o"));
        let stamp = stamps.join(format!("{stem}.stamp"));
        let content = format!("{flags}\n{}\n{header_epoch}\n", mtime(src).unwrap_or(0));
        let up_to_date = obj.exists()
            && fs::read_to_string(&stamp)
                .map(|s| s == content)
                .unwrap_or(false);
        if !up_to_date {
            plan.push((src.clone(), obj, stamp, content));
        }
    }

    if !plan.is_empty() {
        let jobs = var("W34_NVCC_JOBS")
            .ok()
            .and_then(|j| j.parse::<usize>().ok())
            .filter(|&j| j > 0)
            .unwrap_or(16)
            .min(plan.len());
        println!(
            "cargo:warning=[w34-kernels] nvcc: {} file(s), {jobs} job(s)",
            plan.len()
        );
        let next = AtomicUsize::new(0);
        thread::scope(|s| {
            for _ in 0..jobs {
                s.spawn(|| {
                    loop {
                        let i = next.fetch_add(1, Ordering::SeqCst);
                        let Some((src, obj, stamp, content)) = plan.get(i) else {
                            break;
                        };
                        compile_one(&nvcc, &flags, &csrc, src, obj);
                        fs::write(stamp, content).expect("write stamp");
                    }
                });
            }
        });
    }

    let lib = out.join("libw34_kernels.a");
    let mut ar = Command::new("ar");
    ar.arg("rcs").arg(&lib);
    for src in &sources {
        let stem = src.file_stem().unwrap().to_str().unwrap();
        ar.arg(out.join(format!("{stem}.o")));
    }
    let st = ar
        .status()
        .unwrap_or_else(|e| panic!("failed to run ar: {e}"));
    assert!(st.success(), "ar failed");

    println!("cargo:rustc-link-search=native={}", out.display());
    println!("cargo:rustc-link-lib=static=w34_kernels");
    println!(
        "cargo:rustc-link-search=native={}",
        cuda_home.join("lib64").display()
    );
    println!("cargo:rustc-link-lib=cudart");
    println!("cargo:rustc-link-lib=cublas");
    println!("cargo:rustc-link-lib=cuda");
    println!("cargo:rustc-link-lib=stdc++");

    println!("cargo:rerun-if-changed={}", csrc.display());
    for p in sources.iter().chain(headers.iter()) {
        println!("cargo:rerun-if-changed={}", p.display());
    }
    println!("cargo:rerun-if-env-changed=CUDA_HOME");
    println!("cargo:rerun-if-env-changed=W34_ARCH");
    println!("cargo:rerun-if-env-changed=W34_NVCC_JOBS");
    println!("cargo:rerun-if-env-changed=W34_NVCC_EXTRA");
}

fn compile_one(nvcc: &Path, flags: &str, csrc: &Path, src: &Path, obj: &Path) {
    let st = Command::new(nvcc)
        .args(flags.split_whitespace())
        .arg("-c")
        .arg("-I")
        .arg(csrc)
        .arg(src)
        .arg("-o")
        .arg(obj)
        .status()
        .unwrap_or_else(|e| panic!("failed to run nvcc: {e}"));
    assert!(st.success(), "nvcc failed on {}", src.display());
}

fn walk_csrc(dir: &Path, sources: &mut Vec<PathBuf>, headers: &mut Vec<PathBuf>) {
    let Ok(entries) = fs::read_dir(dir) else {
        return;
    };
    for entry in entries.flatten() {
        let p = entry.path();
        if p.is_dir() {
            walk_csrc(&p, sources, headers);
        } else if let Some(ext) = p.extension().and_then(|e| e.to_str()) {
            match ext {
                "cu" => sources.push(p),
                "h" | "cuh" => headers.push(p),
                _ => {}
            }
        }
    }
}

fn mtime(p: &Path) -> Option<u64> {
    use std::time::UNIX_EPOCH;
    fs::metadata(p)
        .ok()?
        .modified()
        .ok()?
        .duration_since(UNIX_EPOCH)
        .ok()
        .map(|d| d.as_nanos() as u64)
}

fn newest_mtime(paths: &[PathBuf]) -> Option<u64> {
    paths.iter().filter_map(|p| mtime(p)).max()
}
