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

#[cfg(test)]
mod tests {
    use super::*;
    use hf_hub::progress::{FileProgress, FileStatus};

    #[test]
    fn disabled_reporter_ignores_events_and_joins_cleanly() {
        let reporter = Reporter::new(false).unwrap();
        let handler = reporter.handler();
        handler.on_progress(&ProgressEvent::Download(DownloadEvent::Start {
            total_files: 2,
            total_bytes: 2048,
        }));
    }

    #[test]
    fn enabled_reporter_processes_all_download_event_variants() {
        let reporter = Reporter::new(true).unwrap();
        let handler = reporter.handler();
        handler.on_progress(&ProgressEvent::Download(DownloadEvent::Start {
            total_files: 1,
            total_bytes: 1_048_576,
        }));
        handler.on_progress(&ProgressEvent::Download(DownloadEvent::AggregateProgress {
            bytes_completed: 524_288,
            total_bytes: 1_048_576,
            bytes_per_sec: None,
        }));
        handler.on_progress(&ProgressEvent::Download(DownloadEvent::Progress {
            files: vec![FileProgress {
                filename: "model.safetensors\n".into(),
                bytes_completed: 1_048_576,
                total_bytes: 1_048_576,
                status: FileStatus::Complete,
            }],
        }));
        handler.on_progress(&ProgressEvent::Download(DownloadEvent::Complete));
        drop(handler);
        drop(reporter);
    }
}
