//! Compiled pipeline states kept across runs in an `MTLBinaryArchive` (one file in the library
//! cache's folder, `disk::use_dir`). A pipeline is looked up in the archive first (failing on a
//! miss); a miss is compiled as before and added. The archive is written back at most every few
//! seconds while pipelines are being added, and when the backend is dropped, so a run that is
//! killed loses at most the last few seconds' pipelines. A file Metal refuses (another GPU or OS)
//! is replaced by an empty archive.

use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_foundation::{NSArray, NSString, NSURL};
use objc2_metal::*;
use std::cell::Cell;
use std::time::{Duration, Instant};

const SAVE_EVERY: Duration = Duration::from_secs(5);

/// `VICTORY_TRACE_PIPE_TIMES=1`: time spent making compute pipelines, by step (library load,
/// archive lookup, archive add, made from the archive, all), logged every 64 pipelines.
pub(crate) fn note_time(step: usize, since: Instant) {
    use std::sync::atomic::{AtomicU64, Ordering::Relaxed};
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    static US: [AtomicU64; 5] = [const { AtomicU64::new(0) }; 5];
    static N: AtomicU64 = AtomicU64::new(0);
    if !*ON.get_or_init(|| std::env::var_os("VICTORY_TRACE_PIPE_TIMES").is_some()) {
        return;
    }
    US[step].fetch_add(since.elapsed().as_micros() as u64, Relaxed);
    if step == 4 && (N.fetch_add(1, Relaxed) + 1).is_multiple_of(64) {
        let ms = |i: usize| US[i].load(Relaxed) / 1000;
        tracing::info!(n = N.load(Relaxed), load_ms = ms(0), lookup_ms = ms(1), add_ms = ms(2), from_archive_ms = ms(3), all_ms = ms(4), "compute pipeline times");
    }
}

pub(crate) struct Archive {
    archive: Retained<ProtocolObject<dyn MTLBinaryArchive>>,
    list: Retained<NSArray<ProtocolObject<dyn MTLBinaryArchive>>>,
    path: std::path::PathBuf,
    added: Cell<u32>,
    saved: Cell<Instant>,
}

type Dev = Retained<ProtocolObject<dyn MTLDevice>>;

impl Archive {
    /// The archive for this GPU, if libraries are kept on disk.
    pub(crate) fn open(dev: &Dev) -> Option<Archive> {
        let dir = crate::disk::dir()?;
        let gpu = blake3::hash(dev.name().to_string().as_bytes()).to_hex();
        let path = dir.join(format!("pipelines-{}.metalarchive", &gpu[..8]));
        let make = |load: bool| {
            let d = MTLBinaryArchiveDescriptor::new();
            if load {
                d.setUrl(Some(&url(&path)));
            }
            dev.newBinaryArchiveWithDescriptor_error(&d)
        };
        let archive = match path.exists().then(|| make(true)) {
            Some(Ok(a)) => a,
            _ => {
                let _ = std::fs::remove_file(&path);
                make(false).ok()?
            }
        };
        let list = NSArray::from_retained_slice(&[archive.clone()]);
        Some(Archive { archive, list, path, added: Cell::new(0), saved: Cell::new(Instant::now()) })
    }

    /// The pipeline of `d`: from the archive, else compiled and added.
    pub(crate) fn render(
        &self,
        dev: &Dev,
        d: &MTLRenderPipelineDescriptor,
    ) -> Result<Retained<ProtocolObject<dyn MTLRenderPipelineState>>, String> {
        d.setBinaryArchives(Some(&self.list));
        if let Ok(p) = dev.newRenderPipelineStateWithDescriptor_options_reflection_error(
            d,
            MTLPipelineOption::FailOnBinaryArchiveMiss,
            None,
        ) {
            return Ok(p);
        }
        // A miss is compiled once: into the archive, then the pipeline is made from it (compiling
        // it for the pipeline and again for the archive cost each new pipeline twice).
        if self.archive.addRenderPipelineFunctionsWithDescriptor_error(d).is_ok() {
            self.added_one();
            if let Ok(p) = dev.newRenderPipelineStateWithDescriptor_options_reflection_error(
                d,
                MTLPipelineOption::FailOnBinaryArchiveMiss,
                None,
            ) {
                return Ok(p);
            }
        }
        dev.newRenderPipelineStateWithDescriptor_error(d)
            .map_err(|e| e.localizedDescription().to_string())
    }

    /// The pipeline of `d` if the archive has it (compiled in an earlier run), else nothing.
    pub(crate) fn cached_render(
        &self,
        dev: &Dev,
        d: &MTLRenderPipelineDescriptor,
    ) -> Option<Retained<ProtocolObject<dyn MTLRenderPipelineState>>> {
        d.setBinaryArchives(Some(&self.list));
        dev.newRenderPipelineStateWithDescriptor_options_reflection_error(
            d,
            MTLPipelineOption::FailOnBinaryArchiveMiss,
            None,
        )
        .ok()
    }

    /// As [`Self::render`], for a compute function.
    pub(crate) fn compute(
        &self,
        dev: &Dev,
        f: &ProtocolObject<dyn MTLFunction>,
    ) -> Result<Retained<ProtocolObject<dyn MTLComputePipelineState>>, String> {
        let d = MTLComputePipelineDescriptor::new();
        d.setComputeFunction(Some(f));
        d.setBinaryArchives(Some(&self.list));
        let t = Instant::now();
        let hit = dev.newComputePipelineStateWithDescriptor_options_reflection_error(
            &d,
            MTLPipelineOption::FailOnBinaryArchiveMiss,
            None,
        );
        note_time(1, t);
        if let Ok(p) = hit {
            return Ok(p);
        }
        let t = Instant::now();
        let added = self.archive.addComputePipelineFunctionsWithDescriptor_error(&d).is_ok();
        note_time(2, t);
        if added {
            self.added_one();
            let t = Instant::now();
            let made = dev.newComputePipelineStateWithDescriptor_options_reflection_error(
                &d,
                MTLPipelineOption::FailOnBinaryArchiveMiss,
                None,
            );
            note_time(3, t);
            if let Ok(p) = made {
                return Ok(p);
            }
        }
        dev.newComputePipelineStateWithDescriptor_options_reflection_error(&d, MTLPipelineOption::None, None)
            .map_err(|e| e.localizedDescription().to_string())
    }

    fn added_one(&self) {
        self.added.set(self.added.get() + 1);
        if self.saved.get().elapsed() >= SAVE_EVERY {
            self.save();
        }
    }

    /// Writes the archive back (to a temporary name, then renamed) if pipelines were added.
    fn save(&self) {
        if self.added.replace(0) == 0 {
            return;
        }
        self.saved.set(Instant::now());
        let tmp = self.path.with_extension(format!("{}.tmp", std::process::id()));
        if self.archive.serializeToURL_error(&url(&tmp)).is_err() || std::fs::rename(&tmp, &self.path).is_err() {
            let _ = std::fs::remove_file(&tmp);
        }
    }
}

impl Drop for Archive {
    fn drop(&mut self) {
        self.save();
    }
}

fn url(p: &std::path::Path) -> Retained<NSURL> {
    NSURL::fileURLWithPath(&NSString::from_str(&p.to_string_lossy()))
}
