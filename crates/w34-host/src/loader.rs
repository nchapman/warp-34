//! Unified Safetensors & HuggingFace Hub checkpoint loader (`w34-host::loader`).
//!
//! Supports:
//! - Single-file (`model.safetensors`) and sharded (`model.safetensors.index.json`) archives,
//!   plus standalone `.safetensors` fixture files.
//! - HuggingFace Hub cache resolution (`refs/main` -> `snapshots/<sha>`).
//! - Quantization formats: `compressed-tensors` (`pack-quantized` INT4/INT8), `gptq` / `gptq_marlin`,
//!   `awq`, `auto-round`, and unquantized `FP16` / `BF16` / `FP32`.
//! - Model families: Qwen3.5 dense (`4B`, `9B`), Qwen3.8 dense (`27B`), Qwen3.6 MoE (`35B-A3B` + MTP),
//!   and `Qwen3DFlashDraftModel` (`DFlash2`).

use crate::cuda::{
    bf16_to_f16_slice_inplace, bf16_to_f32, f16_to_f32, f16_to_f32_slice, f32_to_bf16, f32_to_f16,
};
use serde_json::Value as Json;
use std::collections::HashMap;
use std::fs::{self, File};
use std::io;
use std::os::unix::fs::FileExt;
use std::path::{Path, PathBuf};

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TensorInfo {
    pub name: String,
    pub shard_file: String,
    pub offset: u64,
    pub len_bytes: u64,
    pub dtype: String,
    pub shape: Vec<usize>,
}

impl TensorInfo {
    pub fn num_elems(&self) -> usize {
        self.shape.iter().copied().product()
    }
}

pub struct SafetensorsArchive {
    base_dir: Option<PathBuf>,
    files: HashMap<String, File>,
    entries: HashMap<String, TensorInfo>,
}

impl SafetensorsArchive {
    /// Directory containing this archive, if opened from a path on disk.
    pub fn base_dir(&self) -> Option<&Path> {
        self.base_dir.as_deref()
    }
    /// Opens either a checkpoint directory (`model.safetensors` or `model.safetensors.index.json`)
    /// or a standalone `.safetensors` file directly.
    pub fn open(path: &Path) -> io::Result<Self> {
        if path.is_file() {
            return Self::open_single_file(path);
        }
        let index_path = path.join("model.safetensors.index.json");
        if index_path.exists() {
            Self::open_indexed_dir(path, &index_path)
        } else {
            let single = path.join("model.safetensors");
            if !single.exists() {
                return Err(io::Error::new(
                    io::ErrorKind::NotFound,
                    format!("No model.safetensors or index in {}", path.display()),
                ));
            }
            Self::open_single_file(&single)
        }
    }

    fn parse_safetensors_header(file: &File) -> io::Result<(u64, u64, Json)> {
        let file_len = file.metadata()?.len();
        if file_len < 8 {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "safetensors file shorter than 8 bytes",
            ));
        }
        let mut len_bytes = [0u8; 8];
        file.read_exact_at(&mut len_bytes, 0)?;
        let header_len = u64::from_le_bytes(len_bytes);
        let data_base = 8u64.checked_add(header_len).ok_or_else(|| {
            io::Error::new(io::ErrorKind::InvalidData, "safetensors header overflow")
        })?;
        if data_base > file_len {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "safetensors header exceeds file length",
            ));
        }
        let mut header_buf = vec![0u8; header_len as usize];
        file.read_exact_at(&mut header_buf, 8)?;
        let header: Json = serde_json::from_slice(&header_buf).map_err(|e| {
            io::Error::new(
                io::ErrorKind::InvalidData,
                format!("invalid safetensors JSON header: {e}"),
            )
        })?;
        Ok((data_base, file_len, header))
    }

    fn insert_entries_from_header(
        shard_key: &str,
        data_base: u64,
        file_len: u64,
        header: &Json,
        entries: &mut HashMap<String, TensorInfo>,
    ) -> io::Result<()> {
        let obj = header.as_object().ok_or_else(|| {
            io::Error::new(
                io::ErrorKind::InvalidData,
                "safetensors header is not an object",
            )
        })?;
        for (name, meta) in obj {
            if name == "__metadata__" || meta.get("dtype").is_none() {
                continue;
            }
            let dtype = meta["dtype"]
                .as_str()
                .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidData, "missing dtype"))?
                .to_string();
            let shape_arr = meta["shape"]
                .as_array()
                .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidData, "missing shape"))?;
            let mut shape = Vec::with_capacity(shape_arr.len());
            for dim in shape_arr {
                let d = dim
                    .as_u64()
                    .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidData, "invalid dim"))?;
                shape.push(d as usize);
            }
            let offsets = meta["data_offsets"].as_array().ok_or_else(|| {
                io::Error::new(io::ErrorKind::InvalidData, "missing data_offsets")
            })?;
            let start = offsets.first().and_then(|v| v.as_u64()).ok_or_else(|| {
                io::Error::new(io::ErrorKind::InvalidData, "invalid start offset")
            })?;
            let end = offsets
                .get(1)
                .and_then(|v| v.as_u64())
                .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidData, "invalid end offset"))?;
            if end < start {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidData,
                    "safetensors end < start",
                ));
            }
            let abs_off = data_base.checked_add(start).ok_or_else(|| {
                io::Error::new(io::ErrorKind::InvalidData, "safetensors offset overflow")
            })?;
            let len_bytes = end - start;
            if abs_off.saturating_add(len_bytes) > file_len {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidData,
                    format!("tensor {name} exceeds file end"),
                ));
            }
            entries.insert(
                name.clone(),
                TensorInfo {
                    name: name.clone(),
                    shard_file: shard_key.to_string(),
                    offset: abs_off,
                    len_bytes,
                    dtype,
                    shape,
                },
            );
        }
        Ok(())
    }

    fn open_single_file(file_path: &Path) -> io::Result<Self> {
        let f = File::open(file_path)?;
        let (data_base, file_len, header) = Self::parse_safetensors_header(&f)?;
        let shard_key = file_path
            .file_name()
            .and_then(|s| s.to_str())
            .unwrap_or("model.safetensors")
            .to_string();
        let mut entries = HashMap::new();
        Self::insert_entries_from_header(&shard_key, data_base, file_len, &header, &mut entries)?;
        let mut files = HashMap::new();
        files.insert(shard_key, f);
        Ok(Self {
            base_dir: file_path.parent().map(Path::to_path_buf),
            files,
            entries,
        })
    }

    fn open_indexed_dir(dir: &Path, index_path: &Path) -> io::Result<Self> {
        let index_text = fs::read_to_string(index_path)?;
        let index_json: Json = serde_json::from_str(&index_text).map_err(|e| {
            io::Error::new(
                io::ErrorKind::InvalidData,
                format!("bad model.safetensors.index.json: {e}"),
            )
        })?;
        let weight_map = index_json["weight_map"].as_object().ok_or_else(|| {
            io::Error::new(
                io::ErrorKind::InvalidData,
                "missing weight_map in safetensors index",
            )
        })?;
        let mut files = HashMap::new();
        let mut entries = HashMap::new();
        for shard_val in weight_map.values() {
            let shard_name = shard_val.as_str().ok_or_else(|| {
                io::Error::new(io::ErrorKind::InvalidData, "shard filename not a string")
            })?;
            if files.contains_key(shard_name) {
                continue;
            }
            let shard_path = dir.join(shard_name);
            let f = File::open(&shard_path)?;
            let (data_base, file_len, header) = Self::parse_safetensors_header(&f)?;
            Self::insert_entries_from_header(
                shard_name,
                data_base,
                file_len,
                &header,
                &mut entries,
            )?;
            files.insert(shard_name.to_string(), f);
        }
        let extra_name = "model_extra_tensors.safetensors";
        let extra_path = dir.join(extra_name);
        if !files.contains_key(extra_name) && extra_path.exists() {
            let f = File::open(&extra_path)?;
            let (data_base, file_len, header) = Self::parse_safetensors_header(&f)?;
            Self::insert_entries_from_header(
                extra_name,
                data_base,
                file_len,
                &header,
                &mut entries,
            )?;
            files.insert(extra_name.to_string(), f);
        }
        Ok(Self {
            base_dir: Some(dir.to_path_buf()),
            files,
            entries,
        })
    }

    pub fn contains(&self, name: &str) -> bool {
        self.entries.contains_key(name)
    }

    pub fn len(&self) -> usize {
        self.entries.len()
    }

    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    pub fn tensor_names(&self) -> impl Iterator<Item = &str> {
        self.entries.keys().map(String::as_str)
    }

    pub fn tensor_info(&self, name: &str) -> io::Result<&TensorInfo> {
        self.entries.get(name).ok_or_else(|| {
            io::Error::new(
                io::ErrorKind::NotFound,
                format!("tensor '{name}' not found in archive"),
            )
        })
    }

    pub fn iter_entries(&self) -> impl Iterator<Item = (&str, &TensorInfo)> {
        self.entries.iter().map(|(k, v)| (k.as_str(), v))
    }

    pub fn shard_file(&self, shard_name: &str) -> Option<&File> {
        self.files.get(shard_name)
    }

    pub fn shard_files(&self) -> &HashMap<String, File> {
        &self.files
    }

    /// Reads the raw bytes of `info` directly into `dst` via `pread` (`read_exact_at`)
    /// without repeating a `HashMap` lookup or allocating an intermediate `Vec<u8>`.
    pub fn read_exact_info(&self, info: &TensorInfo, dst: &mut [u8]) -> io::Result<()> {
        if dst.len() as u64 != info.len_bytes {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                format!(
                    "tensor {} byte length mismatch: expected {}, got {}",
                    info.name,
                    info.len_bytes,
                    dst.len()
                ),
            ));
        }
        let f = self.files.get(&info.shard_file).ok_or_else(|| {
            io::Error::new(
                io::ErrorKind::NotFound,
                format!("shard '{}' not open", info.shard_file),
            )
        })?;
        f.read_exact_at(dst, info.offset)
    }

    /// Reads the raw bytes of `name` directly into `dst` via `pread` (`read_exact_at`)
    /// without allocating an intermediate `Vec<u8>`.
    pub fn read_exact_into(&self, name: &str, dst: &mut [u8]) -> io::Result<()> {
        let info = self.tensor_info(name)?;
        self.read_exact_info(info, dst)
    }

    /// Resolves either `<prefix>.<suffix>` or `model.language_model.<stripped>.<suffix>`.
    pub fn resolve_name(&self, canonical: &str) -> Option<String> {
        if self.entries.contains_key(canonical) {
            return Some(canonical.to_string());
        }
        if let Some(rest) = canonical.strip_prefix("model.") {
            let alt = format!("model.language_model.{rest}");
            if self.entries.contains_key(&alt) {
                return Some(alt);
            }
        }
        None
    }

    pub fn read_bytes(&self, name: &str) -> io::Result<Vec<u8>> {
        let info = self.tensor_info(name)?;
        let f = self.files.get(&info.shard_file).ok_or_else(|| {
            io::Error::new(
                io::ErrorKind::NotFound,
                format!("shard '{}' not open", info.shard_file),
            )
        })?;
        let mut buf = vec![0u8; info.len_bytes as usize];
        f.read_exact_at(&mut buf, info.offset)?;
        Ok(buf)
    }

    fn read_le_u16_vec(&self, name: &str, info: &TensorInfo) -> io::Result<Vec<u16>> {
        let len_bytes = info.len_bytes as usize;
        if !len_bytes.is_multiple_of(2) {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!("tensor {name} byte length not a multiple of 2"),
            ));
        }
        let f = self.files.get(&info.shard_file).ok_or_else(|| {
            io::Error::new(
                io::ErrorKind::NotFound,
                format!("shard '{}' not open", info.shard_file),
            )
        })?;
        let num_elems = len_bytes / 2;
        let mut out = vec![0u16; num_elems];
        if len_bytes < 8 * 1024 * 1024 {
            // SAFETY: `out` is an initialized contiguous `u16` buffer of `len_bytes` bytes,
            // and any bit pattern is valid for `u16`.
            let byte_slice =
                unsafe { std::slice::from_raw_parts_mut(out.as_mut_ptr().cast::<u8>(), len_bytes) };
            f.read_exact_at(byte_slice, info.offset)?;
            #[cfg(target_endian = "big")]
            for w in &mut out {
                *w = u16::from_le(*w);
            }
        } else {
            let num_threads = std::thread::available_parallelism()
                .map(|n| n.get().clamp(1, 8))
                .unwrap_or(4);
            let chunk_elems = num_elems.div_ceil(num_threads).max(1);
            let base_offset = info.offset;
            std::thread::scope(|s| -> io::Result<()> {
                let mut handles = Vec::with_capacity(num_threads);
                for (chunk_idx, dst_chunk) in out.chunks_mut(chunk_elems).enumerate() {
                    let elem_offset = chunk_idx * chunk_elems;
                    let byte_offset = base_offset + (elem_offset as u64) * 2;
                    handles.push(s.spawn(move || -> io::Result<()> {
                        let chunk_bytes = dst_chunk.len() * 2;
                        // SAFETY: `dst_chunk` is a disjoint mutable `[u16]` slice of `chunk_bytes` bytes.
                        let byte_slice = unsafe {
                            std::slice::from_raw_parts_mut(
                                dst_chunk.as_mut_ptr().cast::<u8>(),
                                chunk_bytes,
                            )
                        };
                        f.read_exact_at(byte_slice, byte_offset)?;
                        #[cfg(target_endian = "big")]
                        for w in dst_chunk.iter_mut() {
                            *w = u16::from_le(*w);
                        }
                        Ok(())
                    }));
                }
                for h in handles {
                    h.join().map_err(|_| {
                        io::Error::other("safetensors u16 reader thread panicked")
                    })??;
                }
                Ok(())
            })?;
        }
        Ok(out)
    }

    fn read_le_u32_vec(&self, name: &str, info: &TensorInfo) -> io::Result<Vec<u32>> {
        let len_bytes = info.len_bytes as usize;
        if !len_bytes.is_multiple_of(4) {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!("tensor {name} byte length not a multiple of 4"),
            ));
        }
        let f = self.files.get(&info.shard_file).ok_or_else(|| {
            io::Error::new(
                io::ErrorKind::NotFound,
                format!("shard '{}' not open", info.shard_file),
            )
        })?;
        let mut out = vec![0u32; len_bytes / 4];
        // SAFETY: `out` is an initialized contiguous `u32` buffer of `len_bytes` bytes,
        // and any bit pattern is valid for `u32`.
        let byte_slice =
            unsafe { std::slice::from_raw_parts_mut(out.as_mut_ptr().cast::<u8>(), len_bytes) };
        f.read_exact_at(byte_slice, info.offset)?;
        #[cfg(target_endian = "big")]
        for w in &mut out {
            *w = u32::from_le(*w);
        }
        Ok(out)
    }

    pub fn read_u32(&self, name: &str) -> io::Result<Vec<u32>> {
        let info = self.tensor_info(name)?;
        self.read_le_u32_vec(name, info)
    }

    pub fn read_i32(&self, name: &str) -> io::Result<Vec<i32>> {
        let info = self.tensor_info(name)?;
        let len_bytes = info.len_bytes as usize;
        if !len_bytes.is_multiple_of(4) {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!("tensor {name} byte length not a multiple of 4"),
            ));
        }
        let f = self.files.get(&info.shard_file).ok_or_else(|| {
            io::Error::new(
                io::ErrorKind::NotFound,
                format!("shard '{}' not open", info.shard_file),
            )
        })?;
        let mut out = vec![0i32; len_bytes / 4];
        // SAFETY: `out` is an initialized contiguous `i32` buffer of `len_bytes` bytes,
        // and any bit pattern is valid for `i32`.
        let byte_slice =
            unsafe { std::slice::from_raw_parts_mut(out.as_mut_ptr().cast::<u8>(), len_bytes) };
        f.read_exact_at(byte_slice, info.offset)?;
        #[cfg(target_endian = "big")]
        for w in &mut out {
            *w = i32::from_le(*w);
        }
        Ok(out)
    }

    pub fn read_i64(&self, name: &str) -> io::Result<Vec<i64>> {
        let info = self.tensor_info(name)?;
        let bytes = self.read_bytes(name)?;
        match info.dtype.as_str() {
            "I64" | "U64" => {
                if !bytes.len().is_multiple_of(8) {
                    return Err(io::Error::new(
                        io::ErrorKind::InvalidData,
                        format!("tensor {name} byte length not a multiple of 8"),
                    ));
                }
                Ok(bytes
                    .as_chunks::<8>()
                    .0
                    .iter()
                    .copied()
                    .map(i64::from_le_bytes)
                    .collect())
            }
            "I32" => {
                if !bytes.len().is_multiple_of(4) {
                    return Err(io::Error::new(
                        io::ErrorKind::InvalidData,
                        format!("tensor {name} byte length not a multiple of 4"),
                    ));
                }
                Ok(bytes
                    .as_chunks::<4>()
                    .0
                    .iter()
                    .copied()
                    .map(|c| i32::from_le_bytes(c) as i64)
                    .collect())
            }
            other => Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!("unsupported dtype {other} for read_i64({name})"),
            )),
        }
    }

    pub fn read_f32(&self, name: &str) -> io::Result<Vec<f32>> {
        let info = self.tensor_info(name)?;
        match info.dtype.as_str() {
            "F32" => {
                let words = self.read_le_u32_vec(name, info)?;
                Ok(words.into_iter().map(f32::from_bits).collect())
            }
            "F16" => {
                let words = self.read_le_u16_vec(name, info)?;
                let mut out = vec![0.0f32; words.len()];
                f16_to_f32_slice(&words, &mut out);
                Ok(out)
            }
            "BF16" => {
                let words = self.read_le_u16_vec(name, info)?;
                Ok(words.into_iter().map(bf16_to_f32).collect())
            }
            other => Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!("unsupported dtype {other} for read_f32({name})"),
            )),
        }
    }

    /// Reads a single row `row` (`[cols]`) of a 2-D floating-point tensor (`F32`, `F16`, `BF16`)
    /// via `pread` without reading the entire tensor into host memory.
    pub fn read_row_f32(&self, name: &str, row: usize) -> io::Result<Vec<f32>> {
        let info = self.tensor_info(name)?;
        if info.shape.len() != 2 {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!("tensor {name} is not 2-D: {:?}", info.shape),
            ));
        }
        let rows = info.shape[0];
        let cols = info.shape[1];
        if row >= rows {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                format!("row {row} out of range for {name} ([{rows}, {cols}])"),
            ));
        }
        let elem_bytes: usize = match info.dtype.as_str() {
            "F32" => 4,
            "F16" | "BF16" => 2,
            other => {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidData,
                    format!("unsupported dtype {other} for read_row_f32({name})"),
                ));
            }
        };
        let row_bytes = cols.checked_mul(elem_bytes).ok_or_else(|| {
            io::Error::new(io::ErrorKind::InvalidData, "row byte length overflow")
        })?;
        let rel_off = (row as u64)
            .checked_mul(row_bytes as u64)
            .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidData, "row offset overflow"))?;
        if rel_off.saturating_add(row_bytes as u64) > info.len_bytes {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!("row {row} of {name} exceeds tensor byte slice"),
            ));
        }
        let f = self.files.get(&info.shard_file).ok_or_else(|| {
            io::Error::new(
                io::ErrorKind::NotFound,
                format!("shard '{}' not open", info.shard_file),
            )
        })?;
        let mut buf = vec![0u8; row_bytes];
        f.read_exact_at(&mut buf, info.offset + rel_off)?;
        match info.dtype.as_str() {
            "F32" => Ok(buf
                .as_chunks::<4>()
                .0
                .iter()
                .copied()
                .map(f32::from_le_bytes)
                .collect()),
            "F16" => Ok(buf
                .as_chunks::<2>()
                .0
                .iter()
                .copied()
                .map(|c| f16_to_f32(u16::from_le_bytes(c)))
                .collect()),
            "BF16" => Ok(buf
                .as_chunks::<2>()
                .0
                .iter()
                .copied()
                .map(|c| bf16_to_f32(u16::from_le_bytes(c)))
                .collect()),
            _ => unreachable!(),
        }
    }

    /// Reads any floating-point tensor (`F16`, `BF16`, `F32`) and converts to IEEE-754 `FP16` bits.
    pub fn read_f16_bits(&self, name: &str) -> io::Result<Vec<u16>> {
        let info = self.tensor_info(name)?;
        match info.dtype.as_str() {
            "F16" => self.read_le_u16_vec(name, info),
            "BF16" => {
                let mut words = self.read_le_u16_vec(name, info)?;
                if words.len() < 4 * 1024 * 1024 {
                    bf16_to_f16_slice_inplace(&mut words);
                } else {
                    let num_threads = std::thread::available_parallelism()
                        .map(|n| n.get().clamp(1, 8))
                        .unwrap_or(4);
                    let chunk_elems = words.len().div_ceil(num_threads).max(1);
                    std::thread::scope(|s| {
                        for chunk in words.chunks_mut(chunk_elems) {
                            s.spawn(move || {
                                bf16_to_f16_slice_inplace(chunk);
                            });
                        }
                    });
                }
                Ok(words)
            }
            "F32" => {
                let words = self.read_le_u32_vec(name, info)?;
                Ok(words
                    .into_iter()
                    .map(|bits| f32_to_f16(f32::from_bits(bits)))
                    .collect())
            }
            other => Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!("unsupported dtype {other} for read_f16_bits({name})"),
            )),
        }
    }

    /// Reads any floating-point tensor (`F16`, `BF16`, `F32`) described by `info` into `dst`
    /// as IEEE-754 `FP16` bits without repeating a `HashMap` lookup.
    pub fn read_f16_bits_info_into(&self, info: &TensorInfo, dst: &mut [u16]) -> io::Result<()> {
        match info.dtype.as_str() {
            "F16" | "BF16" => {
                let len_bytes = dst.len() * 2;
                if info.len_bytes != len_bytes as u64 {
                    return Err(io::Error::new(
                        io::ErrorKind::InvalidInput,
                        format!(
                            "tensor {} byte length mismatch: expected {}, got {len_bytes}",
                            info.name, info.len_bytes
                        ),
                    ));
                }
                let f = self.files.get(&info.shard_file).ok_or_else(|| {
                    io::Error::new(
                        io::ErrorKind::NotFound,
                        format!("shard '{}' not open", info.shard_file),
                    )
                })?;
                // SAFETY: `dst` is a valid mutable `[u16]` slice of `len_bytes` bytes.
                let byte_slice = unsafe {
                    std::slice::from_raw_parts_mut(dst.as_mut_ptr().cast::<u8>(), len_bytes)
                };
                f.read_exact_at(byte_slice, info.offset)?;
                #[cfg(target_endian = "big")]
                for w in dst.iter_mut() {
                    *w = u16::from_le(*w);
                }
                if info.dtype == "BF16" {
                    bf16_to_f16_slice_inplace(dst);
                }
                Ok(())
            }
            "F32" => {
                let raw_u32 = self.read_le_u32_vec(&info.name, info)?;
                if raw_u32.len() != dst.len() {
                    return Err(io::Error::new(
                        io::ErrorKind::InvalidInput,
                        format!(
                            "tensor {} element count mismatch: expected {}, got {}",
                            info.name,
                            raw_u32.len(),
                            dst.len()
                        ),
                    ));
                }
                for (out, &bits) in dst.iter_mut().zip(&raw_u32) {
                    *out = f32_to_f16(f32::from_bits(bits));
                }
                Ok(())
            }
            other => Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!(
                    "unsupported dtype {other} for read_f16_bits_into({})",
                    info.name
                ),
            )),
        }
    }

    /// Reads any floating-point tensor (`F16`, `BF16`, `F32`) into `dst` as IEEE-754 `FP16` bits
    /// without allocating an intermediate `Vec<u16>` when `dtype` is `F16` or `BF16`.
    pub fn read_f16_bits_into(&self, name: &str, dst: &mut [u16]) -> io::Result<()> {
        let info = self.tensor_info(name)?;
        self.read_f16_bits_info_into(info, dst)
    }

    /// Reads any floating-point tensor (`BF16`, `F16`, `F32`) and converts to `BF16` bits.
    pub fn read_bf16_bits(&self, name: &str) -> io::Result<Vec<u16>> {
        let info = self.tensor_info(name)?;
        match info.dtype.as_str() {
            "BF16" => self.read_le_u16_vec(name, info),
            "F16" => {
                let mut words = self.read_le_u16_vec(name, info)?;
                for w in &mut words {
                    *w = f32_to_bf16(f16_to_f32(*w));
                }
                Ok(words)
            }
            "F32" => {
                let words = self.read_le_u32_vec(name, info)?;
                Ok(words
                    .into_iter()
                    .map(|bits| f32_to_bf16(f32::from_bits(bits)))
                    .collect())
            }
            other => Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!("unsupported dtype {other} for read_bf16_bits({name})"),
            )),
        }
    }

    /// Reads a 2-element `I64` or `I32` `.weight_shape` tensor (`[N, K]`).
    pub fn read_weight_shape_2d(&self, name: &str) -> io::Result<[usize; 2]> {
        let info = self.tensor_info(name)?;
        if info.num_elems() != 2 {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!(
                    "expected 2 elements in '{name}', got shape {:?}",
                    info.shape
                ),
            ));
        }
        let expected_bytes: u64 = match info.dtype.as_str() {
            "I64" | "U64" => 16,
            "I32" | "U32" => 8,
            other => {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidData,
                    format!("unsupported dtype {other} for read_weight_shape_2d({name})"),
                ));
            }
        };
        if info.len_bytes != expected_bytes {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!(
                    "expected {expected_bytes} bytes in '{name}' ({}), got {}",
                    info.dtype, info.len_bytes
                ),
            ));
        }
        let bytes = self.read_bytes(name)?;
        if bytes.len() as u64 != expected_bytes {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!(
                    "short read on '{name}': expected {expected_bytes} bytes, got {}",
                    bytes.len()
                ),
            ));
        }
        let vals: [i64; 2] = match info.dtype.as_str() {
            "I64" | "U64" => [
                i64::from_le_bytes(bytes[0..8].try_into().unwrap()),
                i64::from_le_bytes(bytes[8..16].try_into().unwrap()),
            ],
            "I32" | "U32" => [
                i32::from_le_bytes(bytes[0..4].try_into().unwrap()) as i64,
                i32::from_le_bytes(bytes[4..8].try_into().unwrap()) as i64,
            ],
            _ => unreachable!(),
        };
        if vals[0] <= 0 || vals[1] <= 0 {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!("non-positive dimensions {:?} in '{name}'", vals),
            ));
        }
        Ok([vals[0] as usize, vals[1] as usize])
    }
}

/// Resolves a model path or `<org>/<repo>` HuggingFace Hub identifier to a local snapshot directory.
///
/// Checks `refs/main` first when resolving from `~/.cache/huggingface/hub/models--<org>--<repo>/`
/// so multi-snapshot cache directories always pick the active revision.
pub fn resolve_hf_model_dir(model_spec: &str) -> io::Result<PathBuf> {
    let direct = Path::new(model_spec);
    if direct.exists() {
        return Ok(direct.to_path_buf());
    }
    let hub_root = crate::hub::cache_dir();

    let folder_name = if model_spec.starts_with("models--") {
        model_spec.to_string()
    } else {
        format!("models--{}", model_spec.replace('/', "--"))
    };
    let repo_dir = hub_root.join(&folder_name);
    if !repo_dir.exists() {
        return Err(io::Error::new(
            io::ErrorKind::NotFound,
            format!(
                "HF model '{}' not found at {}",
                model_spec,
                repo_dir.display()
            ),
        ));
    }
    let snapshots_dir = repo_dir.join("snapshots");
    let ref_main = repo_dir.join("refs/main");
    if ref_main.exists()
        && let Ok(rev) = fs::read_to_string(&ref_main)
    {
        let rev = rev.trim();
        let candidate = snapshots_dir.join(rev);
        if candidate.join("config.json").exists() {
            return Ok(candidate);
        }
    }
    let mut candidates = Vec::new();
    if snapshots_dir.exists() {
        for entry in fs::read_dir(&snapshots_dir)? {
            let entry = entry?;
            let p = entry.path();
            if p.is_dir() && p.join("config.json").exists() {
                candidates.push(p);
            }
        }
    }
    candidates.sort();
    candidates.pop().ok_or_else(|| {
        io::Error::new(
            io::ErrorKind::NotFound,
            format!(
                "no valid snapshot with config.json under {}",
                repo_dir.display()
            ),
        )
    })
}

pub fn sanitize_diag(s: &str) -> String {
    let mut out = String::with_capacity(s.len().min(64));
    for c in s.chars().take(64) {
        if c.is_ascii_graphic() || c == ' ' {
            out.push(c);
        } else {
            out.push_str(&format!("\\u{{{:04x}}}", c as u32));
        }
    }
    out
}

mod config;
pub use config::*;

mod quant;
pub use quant::*;

#[cfg(test)]
mod tests;
