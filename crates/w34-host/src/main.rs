mod cli;

use std::env;
use std::process::ExitCode;

fn main() -> ExitCode {
    let args: Vec<String> = env::args().skip(1).collect();
    #[cfg(feature = "cuda")]
    {
        match cli::run_cuda_cli(&args) {
            Ok(code) => code,
            Err(e) => {
                eprintln!("w34 error: {e}");
                ExitCode::FAILURE
            }
        }
    }
    #[cfg(not(feature = "cuda"))]
    {
        let _ = args;
        let _ = cli::print_usage;
        println!("w34 v{} (cuda=false)", env!("CARGO_PKG_VERSION"));
        ExitCode::SUCCESS
    }
}
