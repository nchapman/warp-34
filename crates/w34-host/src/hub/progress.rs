use hf_hub::progress::{DownloadEvent, Progress, ProgressEvent, ProgressHandler};
use std::{
    io,
    sync::mpsc,
    thread,
    time::{Duration, Instant},
};

#[derive(Clone)]
struct Events(Option<mpsc::SyncSender<ProgressEvent>>);
impl ProgressHandler for Events {
    fn on_progress(&self, event: &ProgressEvent) {
        // Never block the client's network/Xet tasks on a slow terminal.
        if let Some(sender) = &self.0 {
            let _ = sender.try_send(event.clone());
        }
    }
}

pub(super) struct Reporter {
    events: Events,
    worker: Option<thread::JoinHandle<()>>,
}
impl Reporter {
    pub fn new(enabled: bool) -> io::Result<Self> {
        if !enabled {
            return Ok(Self {
                events: Events(None),
                worker: None,
            });
        }
        let (tx, rx) = mpsc::sync_channel(8);
        let worker = thread::Builder::new()
            .name("hf-progress".into())
            .spawn(move || {
                let mut last = Instant::now() - Duration::from_secs(1);
                for event in rx {
                    if last.elapsed() < Duration::from_secs(1) {
                        continue;
                    }
                    match event {
                        ProgressEvent::Download(DownloadEvent::Start {
                            total_files,
                            total_bytes,
                        }) => {
                            eprintln!(
                                "  Hub: {total_files} files, {:.1} MiB to transfer",
                                total_bytes as f64 / 1048576.0
                            );
                        }
                        ProgressEvent::Download(DownloadEvent::AggregateProgress {
                            bytes_completed,
                            total_bytes,
                            ..
                        }) => {
                            eprintln!(
                                "  Hub Xet: {:.1} / {:.1} MiB",
                                bytes_completed as f64 / 1048576.0,
                                total_bytes as f64 / 1048576.0
                            );
                        }
                        ProgressEvent::Download(DownloadEvent::Progress { files }) => {
                            if let Some(file) = files.iter().find(|f| f.total_bytes > 0) {
                                eprintln!(
                                    "  Hub {}: {:.1} / {:.1} MiB",
                                    crate::loader::sanitize_diag(&file.filename),
                                    file.bytes_completed as f64 / 1048576.0,
                                    file.total_bytes as f64 / 1048576.0
                                );
                            }
                        }
                        _ => continue,
                    }
                    last = Instant::now();
                }
            })?;
        Ok(Self {
            events: Events(Some(tx)),
            worker: Some(worker),
        })
    }

    pub fn handler(&self) -> Progress {
        self.events.clone().into()
    }
}
impl Drop for Reporter {
    fn drop(&mut self) {
        self.events.0.take();
        if let Some(worker) = self.worker.take() {
            let _ = worker.join();
        }
    }
}
