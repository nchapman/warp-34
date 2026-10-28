fn main() {
    #[cfg(feature = "cuda")]
    {
        match w34_host::ffi::cublas_runtime_version() {
            Ok(v) => println!("warp-34 (cuda enabled, cuBLAS runtime v{v})"),
            Err(rc) => eprintln!("warp-34: cuBLAS version query failed with rc={rc}"),
        }
    }
    #[cfg(not(feature = "cuda"))]
    {
        println!("warp-34 (built without --features cuda)");
    }
}
