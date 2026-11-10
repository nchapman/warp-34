//! `SGLang`-donor `MambaRadixCache` (`mamba_radix_cache.py:73-1428`) and
//! `HybridReqToTokenPool` (`memory_pool.py:35-130`) adapted to logical `KvPool` `page_id`s
//! (`B = 16`) and chunk-aligned (`C = k * 64`) GDN `StateBundle` checkpoints.
//!
//! Enforces the core `SGLang` invariants:
//! 1. Dual lock reference counts (`full_lock_ref >= mamba_lock_ref` at every node):
//!    `inc_lock_ref(node)` locks `mamba_lock_ref` on `node` itself (if `mamba_value.is_some()`)
//!    and `full_lock_ref` on every node along `[node .. root)`.
//! 2. Dual LRU lists (`full_lru` vs `mamba_lru`) with independent recency stamps:
//!    prefix hits refresh `full_lru` along `[node .. root)` and refresh `mamba_lru` only on `node`.
//! 3. Internal nodes may be Mamba-tombstoned (`mamba_value = None`) when `evict_mamba` reclaims
//!    their `StateBundle` slot while keeping KV pages alive for longer descendants; leaf nodes
//!    are never tombstones (`_iteratively_delete_tombstone_leaf`).
//! 4. 2-slot ping-pong `extra_buffer` checkpoint tracking (`ReqRadixState`): snapshots GDN state
//!    at `C = 64` boundaries via `w34_gdn_batch::checkpoint_dst_indices` and donates only
//!    **committed** (`emitted_len`) snapshots into `MambaRadixCache`.

use std::collections::BTreeMap;

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct RadixNodeId(pub usize);

impl RadixNodeId {
    pub const ROOT: Self = Self(0);
}

#[derive(Clone, Debug)]
pub struct RadixNode {
    pub id: RadixNodeId,
    pub parent: Option<RadixNodeId>,
    /// Keyed by the first page (`page_size` tokens) of the child's `key`.
    pub children: BTreeMap<Vec<i32>, RadixNodeId>,
    /// Page-aligned token sequence (`key.len() == value.len() * page_size`).
    pub key: Vec<i32>,
    /// Logical KV `page_id`s (`W34_KV_NAMESPACE_TARGET_FULL_ATTN`).
    pub value: Vec<i32>,
    /// Optional `DFlashDraftContext` logical KV `page_id`s for warm speculative proposer hits.
    pub draft_pages: Vec<i32>,
    /// GDN `StateBundle` snapshot slot at the end of `key` (`None` if tombstoned or split parent).
    pub mamba_value: Option<i32>,
    pub full_lock_ref: usize,
    pub mamba_lock_ref: usize,
    pub last_access_tick: u64,
    pub mamba_last_access_tick: u64,
    pub is_decode_tail: bool,
    pub alive: bool,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RadixMatchResult {
    /// Number of matched tokens up to `last_node` (always a multiple of `page_size`).
    pub matched_tokens: usize,
    /// Matched logical KV `page_id`s (`kv_pages.len() == matched_tokens / page_size`).
    pub kv_pages: Vec<i32>,
    /// Matched draft-context KV `page_id`s (if present).
    pub draft_pages: Vec<i32>,
    /// Restored GDN `StateBundle` snapshot slot at `matched_tokens` (`None` only when `matched_tokens == 0`).
    pub mamba_slot: Option<i32>,
    /// Deepest matched node holding a valid `mamba_value` (`RadixNodeId::ROOT` if no match).
    pub last_node: RadixNodeId,
    /// When a longer KV prefix was matched past a Mamba tombstone, records the chunk-aligned
    /// branching sequence length (`mamba_radix_cache.py:1174-1182`).
    pub mamba_branching_seqlen: Option<usize>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RadixInsertResult {
    /// Number of tokens along `token_ids` that already existed in the radix tree.
    pub existing_prefix_tokens: usize,
    /// Whether the target node already had a live `mamba_value` (`true` => caller should free its donated `mamba_slot`).
    pub mamba_already_existed: bool,
    /// Duplicate KV pages from `kv_pages[prev_protected_pages .. existing_prefix_pages]` that were
    /// superseded by existing tree pages and must be freed back to `KvPool`.
    pub freed_duplicate_kv_pages: Vec<i32>,
    /// Duplicate draft KV pages superseded by existing tree pages.
    pub freed_duplicate_draft_pages: Vec<i32>,
    /// Node corresponding to the end of the inserted sequence.
    pub terminal_node: RadixNodeId,
}

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct RadixEvictResult {
    pub freed_kv_pages: Vec<i32>,
    pub freed_draft_pages: Vec<i32>,
    pub freed_mamba_slots: Vec<i32>,
}

pub struct MambaRadixCache {
    pub page_size: usize,
    pub mamba_chunk_size: usize,
    pub disable: bool,
    nodes: Vec<RadixNode>,
    /// Ordered from LRU (index 0) to MRU (last). Contains all alive non-root nodes.
    full_lru: Vec<RadixNodeId>,
    /// Ordered from LRU (index 0) to MRU (last). Contains all alive non-root nodes with `mamba_value.is_some()`.
    mamba_lru: Vec<RadixNodeId>,
    tick_counter: u64,
    full_evictable_pages: usize,
    full_protected_pages: usize,
    mamba_evictable_slots: usize,
    mamba_protected_slots: usize,
    pub total_lookups: u64,
    pub total_lookup_tokens: u64,
    pub total_hit_tokens: u64,
    pub total_mamba_hits: u64,
}

impl MambaRadixCache {
    pub fn new(page_size: usize, mamba_chunk_size: usize) -> Self {
        assert!(page_size > 0, "page_size must be > 0");
        assert!(
            mamba_chunk_size >= page_size && mamba_chunk_size.is_multiple_of(page_size),
            "mamba_chunk_size ({mamba_chunk_size}) must be a multiple of page_size ({page_size})"
        );
        let mut cache = Self {
            page_size,
            mamba_chunk_size,
            disable: false,
            nodes: Vec::new(),
            full_lru: Vec::new(),
            mamba_lru: Vec::new(),
            tick_counter: 1,
            full_evictable_pages: 0,
            full_protected_pages: 0,
            mamba_evictable_slots: 0,
            mamba_protected_slots: 0,
            total_lookups: 0,
            total_lookup_tokens: 0,
            total_hit_tokens: 0,
            total_mamba_hits: 0,
        };
        cache.reset();
        cache
    }

    fn next_tick(&mut self) -> u64 {
        self.tick_counter += 1;
        self.tick_counter
    }

    pub fn reset(&mut self) {
        self.nodes.clear();
        self.full_lru.clear();
        self.mamba_lru.clear();
        self.tick_counter = 1;
        self.full_evictable_pages = 0;
        self.full_protected_pages = 0;
        self.mamba_evictable_slots = 0;
        self.mamba_protected_slots = 0;
        self.nodes.push(RadixNode {
            id: RadixNodeId::ROOT,
            parent: None,
            children: BTreeMap::new(),
            key: Vec::new(),
            value: Vec::new(),
            draft_pages: Vec::new(),
            mamba_value: None,
            full_lock_ref: 1,
            mamba_lock_ref: 1,
            last_access_tick: 1,
            mamba_last_access_tick: 1,
            is_decode_tail: false,
            alive: true,
        });
    }

    #[inline]
    pub fn node(&self, id: RadixNodeId) -> &RadixNode {
        &self.nodes[id.0]
    }

    /// Collects the canonical `kv_pages` along `[ROOT ..= node_id]` without modifying LRU
    /// or incrementing hit-rate lookup counters.
    pub fn collect_node_kv_pages(&self, node_id: RadixNodeId) -> Vec<i32> {
        let mut chain = Vec::new();
        let mut cur = node_id;
        while cur != RadixNodeId::ROOT {
            chain.push(cur);
            cur = self.nodes[cur.0].parent.unwrap_or(RadixNodeId::ROOT);
        }
        let mut pages = Vec::new();
        for &nid in chain.iter().rev() {
            pages.extend_from_slice(&self.nodes[nid.0].value);
        }
        pages
    }

    #[inline]
    pub fn full_evictable_pages(&self) -> usize {
        self.full_evictable_pages
    }

    #[inline]
    pub fn full_protected_pages(&self) -> usize {
        self.full_protected_pages
    }

    #[inline]
    pub fn mamba_evictable_slots(&self) -> usize {
        self.mamba_evictable_slots
    }

    #[inline]
    pub fn mamba_protected_slots(&self) -> usize {
        self.mamba_protected_slots
    }

    pub fn hit_rate(&self) -> f64 {
        if self.total_lookup_tokens == 0 {
            0.0
        } else {
            (self.total_hit_tokens as f64) / (self.total_lookup_tokens as f64)
        }
    }

    fn full_lru_insert_mru(&mut self, id: RadixNodeId) {
        debug_assert!(id != RadixNodeId::ROOT);
        debug_assert!(!self.full_lru.contains(&id));
        let tick = self.next_tick();
        self.nodes[id.0].last_access_tick = tick;
        self.full_lru.push(id);
    }

    fn full_lru_reset_mru(&mut self, id: RadixNodeId) {
        if id == RadixNodeId::ROOT {
            return;
        }
        if let Some(pos) = self.full_lru.iter().position(|&x| x == id) {
            self.full_lru.remove(pos);
        }
        let tick = self.next_tick();
        self.nodes[id.0].last_access_tick = tick;
        self.full_lru.push(id);
    }

    fn full_lru_remove(&mut self, id: RadixNodeId) {
        if let Some(pos) = self.full_lru.iter().position(|&x| x == id) {
            self.full_lru.remove(pos);
        }
    }

    fn mamba_lru_insert_mru(&mut self, id: RadixNodeId) {
        debug_assert!(id != RadixNodeId::ROOT);
        debug_assert!(self.nodes[id.0].mamba_value.is_some());
        debug_assert!(!self.mamba_lru.contains(&id));
        let tick = self.next_tick();
        self.nodes[id.0].mamba_last_access_tick = tick;
        self.mamba_lru.push(id);
    }

    fn mamba_lru_reset_mru(&mut self, id: RadixNodeId) {
        if id == RadixNodeId::ROOT || self.nodes[id.0].mamba_value.is_none() {
            return;
        }
        if let Some(pos) = self.mamba_lru.iter().position(|&x| x == id) {
            self.mamba_lru.remove(pos);
        }
        let tick = self.next_tick();
        self.nodes[id.0].mamba_last_access_tick = tick;
        self.mamba_lru.push(id);
    }

    fn mamba_lru_remove(&mut self, id: RadixNodeId) {
        if let Some(pos) = self.mamba_lru.iter().position(|&x| x == id) {
            self.mamba_lru.remove(pos);
        }
    }

    /// Refreshes `[node .. root)` in `full_lru` so child is more recently used than parent
    /// (`mamba_radix_cache.py:249-263`).
    fn full_lru_reset_node_and_parents_mru(&mut self, node: RadixNodeId) {
        let mut chain = Vec::new();
        let mut cur = node;
        while cur != RadixNodeId::ROOT {
            chain.push(cur);
            cur = self.nodes[cur.0].parent.unwrap_or(RadixNodeId::ROOT);
        }
        // Refresh from closest-to-root down to `node` so `node` ends up at the very back (MRU).
        for &n in chain.iter().rev() {
            self.full_lru_reset_mru(n);
        }
    }

    fn page_aligned_common_prefix_len(&self, a: &[i32], b: &[i32]) -> usize {
        let max_pages = a.len().min(b.len()) / self.page_size;
        let mut matched_pages = 0;
        while matched_pages < max_pages {
            let start = matched_pages * self.page_size;
            let end = start + self.page_size;
            if a[start..end] != b[start..end] {
                break;
            }
            matched_pages += 1;
        }
        matched_pages * self.page_size
    }

    /// Splits `child_id` at `split_token_len` (`0 < split_token_len < child.key.len()`).
    /// Returns the newly created intermediate node `new_node_id`, which has `mamba_value = None`
    /// and inherits `child.full_lock_ref` (`mamba_radix_cache.py:1210-1244`).
    fn split_node(&mut self, child_id: RadixNodeId, split_token_len: usize) -> RadixNodeId {
        let split_pages = split_token_len / self.page_size;
        let parent_id = self.nodes[child_id.0]
            .parent
            .expect("cannot split root node");
        let child_full_lock = self.nodes[child_id.0].full_lock_ref;
        let prefix_key = self.nodes[child_id.0].key[..split_token_len].to_vec();
        let suffix_key = self.nodes[child_id.0].key[split_token_len..].to_vec();
        let prefix_val = self.nodes[child_id.0].value[..split_pages].to_vec();
        let suffix_val = self.nodes[child_id.0].value[split_pages..].to_vec();
        let (prefix_draft, suffix_draft) =
            if self.nodes[child_id.0].draft_pages.len() >= split_pages {
                (
                    self.nodes[child_id.0].draft_pages[..split_pages].to_vec(),
                    self.nodes[child_id.0].draft_pages[split_pages..].to_vec(),
                )
            } else {
                (Vec::new(), Vec::new())
            };

        let first_page_prefix = prefix_key[..self.page_size].to_vec();
        let first_page_suffix = suffix_key[..self.page_size].to_vec();

        let new_node_id = RadixNodeId(self.nodes.len());
        let mut new_children = BTreeMap::new();
        new_children.insert(first_page_suffix, child_id);

        let tick = self.next_tick();
        self.nodes.push(RadixNode {
            id: new_node_id,
            parent: Some(parent_id),
            children: new_children,
            key: prefix_key,
            value: prefix_val,
            draft_pages: prefix_draft,
            mamba_value: None, // Mamba state cannot be split mid-node
            full_lock_ref: child_full_lock,
            mamba_lock_ref: 0,
            last_access_tick: tick,
            mamba_last_access_tick: tick,
            is_decode_tail: false,
            alive: true,
        });

        self.full_lru_remove(child_id);
        {
            let child = &mut self.nodes[child_id.0];
            child.parent = Some(new_node_id);
            child.key = suffix_key;
            child.value = suffix_val;
            child.draft_pages = suffix_draft;
        }
        self.nodes[parent_id.0]
            .children
            .insert(first_page_prefix, new_node_id);

        // Insert parent first so child is more recent than parent in `full_lru`.
        self.full_lru_insert_mru(new_node_id);
        self.full_lru_insert_mru(child_id);
        new_node_id
    }

    /// Finds the longest prefix of `token_ids` in the radix tree that possesses a valid
    /// GDN `StateBundle` snapshot (`mamba_value.is_some()`).
    pub fn match_prefix(&mut self, token_ids: &[i32]) -> RadixMatchResult {
        self.total_lookups += 1;
        self.total_lookup_tokens += token_ids.len() as u64;

        let aligned_len = (token_ids.len() / self.page_size) * self.page_size;
        if self.disable || aligned_len == 0 {
            return RadixMatchResult {
                matched_tokens: 0,
                kv_pages: Vec::new(),
                draft_pages: Vec::new(),
                mamba_slot: None,
                last_node: RadixNodeId::ROOT,
                mamba_branching_seqlen: None,
            };
        }

        let mut cur_node = RadixNodeId::ROOT;
        let mut key_offset = 0usize;
        let mut path_nodes: Vec<RadixNodeId> = Vec::new();
        let mut best_path_len = 0usize;
        let mut best_last_node = RadixNodeId::ROOT;

        while key_offset < aligned_len {
            let first_page = &token_ids[key_offset..key_offset + self.page_size];
            let Some(&child_id) = self.nodes[cur_node.0].children.get(first_page) else {
                break;
            };
            if self.nodes[cur_node.0].mamba_value.is_some() {
                best_path_len = path_nodes.len();
                best_last_node = cur_node;
            }

            let rem_key = &token_ids[key_offset..aligned_len];
            let child_key_len = self.nodes[child_id.0].key.len();
            let prefix_len =
                self.page_aligned_common_prefix_len(&self.nodes[child_id.0].key, rem_key);
            debug_assert!(prefix_len >= self.page_size);

            if prefix_len < child_key_len {
                let split_id = self.split_node(child_id, prefix_len);
                path_nodes.push(split_id);
                cur_node = split_id;
                break;
            } else {
                path_nodes.push(child_id);
                cur_node = child_id;
                key_offset += prefix_len;
            }
        }

        if self.nodes[cur_node.0].mamba_value.is_some() {
            best_path_len = path_nodes.len();
            best_last_node = cur_node;
        }

        // Refresh full LRU along `[best_last_node .. root)` and mamba LRU only on `best_last_node`.
        self.full_lru_reset_node_and_parents_mru(best_last_node);
        if best_last_node != RadixNodeId::ROOT && self.nodes[best_last_node.0].mamba_value.is_some()
        {
            self.mamba_lru_reset_mru(best_last_node);
        }

        let total_kv_matched_tokens: usize = path_nodes
            .iter()
            .map(|&nid| self.nodes[nid.0].key.len())
            .sum();
        let mut kv_pages = Vec::new();
        let mut draft_pages = Vec::new();
        for &nid in &path_nodes[..best_path_len] {
            self.nodes[nid.0].is_decode_tail = false;
            kv_pages.extend_from_slice(&self.nodes[nid.0].value);
            draft_pages.extend_from_slice(&self.nodes[nid.0].draft_pages);
        }
        let matched_tokens = kv_pages.len() * self.page_size;
        let mamba_slot = self.nodes[best_last_node.0].mamba_value;

        let mamba_branching_seqlen = if path_nodes.len() > best_path_len {
            let aligned = (total_kv_matched_tokens / self.mamba_chunk_size) * self.mamba_chunk_size;
            if aligned > 0 { Some(aligned) } else { None }
        } else {
            None
        };

        self.total_hit_tokens += matched_tokens as u64;
        if mamba_slot.is_some() {
            self.total_mamba_hits += 1;
        }

        RadixMatchResult {
            matched_tokens,
            kv_pages,
            draft_pages,
            mamba_slot,
            last_node: best_last_node,
            mamba_branching_seqlen,
        }
    }

    /// Increments `mamba_lock_ref` on `node_id` (if `mamba_value.is_some()`) and `full_lock_ref`
    /// on `[node_id .. root)` (`mamba_radix_cache.py:922-948`).
    pub fn inc_lock_ref(&mut self, node_id: RadixNodeId) {
        if self.disable || node_id == RadixNodeId::ROOT {
            return;
        }
        if self.nodes[node_id.0].mamba_value.is_some() {
            if self.nodes[node_id.0].mamba_lock_ref == 0 {
                self.mamba_evictable_slots = self.mamba_evictable_slots.saturating_sub(1);
                self.mamba_protected_slots += 1;
            }
            self.nodes[node_id.0].mamba_lock_ref += 1;
        }
        let mut cur = node_id;
        while cur != RadixNodeId::ROOT {
            let pages = self.nodes[cur.0].value.len();
            if self.nodes[cur.0].full_lock_ref == 0 {
                self.full_evictable_pages = self.full_evictable_pages.saturating_sub(pages);
                self.full_protected_pages += pages;
            }
            self.nodes[cur.0].full_lock_ref += 1;
            cur = self.nodes[cur.0].parent.unwrap_or(RadixNodeId::ROOT);
        }
    }

    /// Decrements `mamba_lock_ref` on `node_id` (if `mamba_value.is_some()`) and `full_lock_ref`
    /// on `[node_id .. root)` (`mamba_radix_cache.py:950-980`).
    pub fn dec_lock_ref(&mut self, node_id: RadixNodeId) {
        if self.disable || node_id == RadixNodeId::ROOT {
            return;
        }
        if self.nodes[node_id.0].mamba_value.is_some() {
            assert!(
                self.nodes[node_id.0].mamba_lock_ref > 0,
                "dec_lock_ref on node {:?} with mamba_lock_ref == 0",
                node_id
            );
            if self.nodes[node_id.0].mamba_lock_ref == 1 {
                self.mamba_evictable_slots += 1;
                self.mamba_protected_slots = self.mamba_protected_slots.saturating_sub(1);
            }
            self.nodes[node_id.0].mamba_lock_ref -= 1;
        }
        let mut cur = node_id;
        while cur != RadixNodeId::ROOT {
            assert!(
                self.nodes[cur.0].full_lock_ref > 0,
                "dec_lock_ref on node {:?} with full_lock_ref == 0",
                cur
            );
            let pages = self.nodes[cur.0].value.len();
            if self.nodes[cur.0].full_lock_ref == 1 {
                self.full_evictable_pages += pages;
                self.full_protected_pages = self.full_protected_pages.saturating_sub(pages);
            }
            self.nodes[cur.0].full_lock_ref -= 1;
            cur = self.nodes[cur.0].parent.unwrap_or(RadixNodeId::ROOT);
        }
    }

    /// Inserts a page-aligned `(token_ids, kv_pages, mamba_slot)` prefix into the radix tree
    /// (`mamba_radix_cache.py:1246-1317`).
    pub fn insert(
        &mut self,
        token_ids: &[i32],
        kv_pages: &[i32],
        draft_pages: &[i32],
        mamba_slot: i32,
        prev_protected_pages: usize,
    ) -> RadixInsertResult {
        assert!(
            token_ids.len().is_multiple_of(self.page_size),
            "insert token_ids length ({}) must be a multiple of page_size ({})",
            token_ids.len(),
            self.page_size
        );
        assert_eq!(
            token_ids.len() / self.page_size,
            kv_pages.len(),
            "kv_pages length must equal token_ids.len() / page_size"
        );
        assert!(
            draft_pages.is_empty() || draft_pages.len() == kv_pages.len(),
            "draft_pages must be empty or match kv_pages.len()"
        );

        if self.disable || token_ids.is_empty() {
            return RadixInsertResult {
                existing_prefix_tokens: 0,
                mamba_already_existed: true,
                freed_duplicate_kv_pages: Vec::new(),
                freed_duplicate_draft_pages: Vec::new(),
                terminal_node: RadixNodeId::ROOT,
            };
        }

        let mut cur_node = RadixNodeId::ROOT;
        let mut key_offset = 0usize;
        let mut page_offset = 0usize;
        let mut freed_duplicate_kv_pages = Vec::new();
        let mut freed_duplicate_draft_pages = Vec::new();

        while key_offset < token_ids.len() {
            let first_page = &token_ids[key_offset..key_offset + self.page_size];
            let Some(&child_id) = self.nodes[cur_node.0].children.get(first_page) else {
                break;
            };
            self.full_lru_reset_mru(child_id);
            let rem_key = &token_ids[key_offset..];
            let child_key_len = self.nodes[child_id.0].key.len();
            let prefix_len =
                self.page_aligned_common_prefix_len(&self.nodes[child_id.0].key, rem_key);
            let prefix_pages = prefix_len / self.page_size;

            if prev_protected_pages < page_offset + prefix_pages {
                let start_page = prev_protected_pages.saturating_sub(page_offset);
                freed_duplicate_kv_pages.extend_from_slice(
                    &kv_pages[page_offset + start_page..page_offset + prefix_pages],
                );
                if !draft_pages.is_empty() {
                    freed_duplicate_draft_pages.extend_from_slice(
                        &draft_pages[page_offset + start_page..page_offset + prefix_pages],
                    );
                }
            }

            key_offset += prefix_len;
            page_offset += prefix_pages;

            if prefix_len < child_key_len {
                cur_node = self.split_node(child_id, prefix_len);
                break;
            } else {
                self.nodes[child_id.0].is_decode_tail = false;
                cur_node = child_id;
            }
        }

        let mamba_already_existed;
        let terminal_node;
        if key_offset < token_ids.len() {
            let new_key = token_ids[key_offset..].to_vec();
            let new_val = kv_pages[page_offset..].to_vec();
            let new_draft = if draft_pages.is_empty() {
                Vec::new()
            } else {
                draft_pages[page_offset..].to_vec()
            };
            let first_page = new_key[..self.page_size].to_vec();
            let added_pages = new_val.len();
            let new_node_id = RadixNodeId(self.nodes.len());
            let tick = self.next_tick();
            self.nodes.push(RadixNode {
                id: new_node_id,
                parent: Some(cur_node),
                children: BTreeMap::new(),
                key: new_key,
                value: new_val,
                draft_pages: new_draft,
                mamba_value: Some(mamba_slot),
                full_lock_ref: 0,
                mamba_lock_ref: 0,
                last_access_tick: tick,
                mamba_last_access_tick: tick,
                is_decode_tail: false,
                alive: true,
            });
            self.nodes[cur_node.0]
                .children
                .insert(first_page, new_node_id);
            self.full_lru_insert_mru(new_node_id);
            self.mamba_lru_insert_mru(new_node_id);
            self.full_evictable_pages += added_pages;
            self.mamba_evictable_slots += 1;
            mamba_already_existed = false;
            terminal_node = new_node_id;
        } else if self.nodes[cur_node.0].mamba_value.is_none() {
            // Revive Mamba tombstone on an existing internal/split node
            self.nodes[cur_node.0].mamba_value = Some(mamba_slot);
            self.full_lru_reset_mru(cur_node);
            self.mamba_lru_insert_mru(cur_node);
            self.mamba_evictable_slots += 1;
            mamba_already_existed = false;
            terminal_node = cur_node;
        } else {
            mamba_already_existed = true;
            self.full_lru_reset_mru(cur_node);
            terminal_node = cur_node;
        }

        RadixInsertResult {
            existing_prefix_tokens: key_offset,
            mamba_already_existed,
            freed_duplicate_kv_pages,
            freed_duplicate_draft_pages,
            terminal_node,
        }
    }

    fn delete_leaf_and_cleanup_tombstones(
        &mut self,
        leaf_id: RadixNodeId,
        out: &mut RadixEvictResult,
    ) {
        assert_eq!(self.nodes[leaf_id.0].full_lock_ref, 0);
        assert_eq!(self.nodes[leaf_id.0].mamba_lock_ref, 0);
        assert!(self.nodes[leaf_id.0].children.is_empty());
        assert!(
            self.nodes[leaf_id.0].mamba_value.is_some(),
            "leaf node {:?} must not be a tombstone prior to eviction",
            leaf_id
        );

        let pages = self.nodes[leaf_id.0].value.len();
        out.freed_kv_pages
            .extend_from_slice(&self.nodes[leaf_id.0].value);
        out.freed_draft_pages
            .extend_from_slice(&self.nodes[leaf_id.0].draft_pages);
        if let Some(ms) = self.nodes[leaf_id.0].mamba_value.take() {
            out.freed_mamba_slots.push(ms);
            self.mamba_evictable_slots = self.mamba_evictable_slots.saturating_sub(1);
        }
        self.full_evictable_pages = self.full_evictable_pages.saturating_sub(pages);

        self.full_lru_remove(leaf_id);
        self.mamba_lru_remove(leaf_id);

        let first_page = self.nodes[leaf_id.0].key[..self.page_size].to_vec();
        let parent_id = self.nodes[leaf_id.0]
            .parent
            .expect("leaf node must have parent");
        self.nodes[parent_id.0].children.remove(&first_page);
        self.nodes[leaf_id.0].alive = false;

        // Iteratively delete exposed tombstone leaves (`mamba_radix_cache.py:1319-1340`).
        let mut cur_parent = parent_id;
        while cur_parent != RadixNodeId::ROOT {
            if !self.nodes[cur_parent.0].children.is_empty()
                || self.nodes[cur_parent.0].mamba_value.is_some()
                || self.nodes[cur_parent.0].full_lock_ref > 0
            {
                break;
            }
            assert_eq!(self.nodes[cur_parent.0].mamba_lock_ref, 0);
            let p_pages = self.nodes[cur_parent.0].value.len();
            out.freed_kv_pages
                .extend_from_slice(&self.nodes[cur_parent.0].value);
            out.freed_draft_pages
                .extend_from_slice(&self.nodes[cur_parent.0].draft_pages);
            self.full_evictable_pages = self.full_evictable_pages.saturating_sub(p_pages);
            self.full_lru_remove(cur_parent);
            let p_first_page = self.nodes[cur_parent.0].key[..self.page_size].to_vec();
            let grand_parent = self.nodes[cur_parent.0]
                .parent
                .expect("non-root must have parent");
            self.nodes[grand_parent.0].children.remove(&p_first_page);
            self.nodes[cur_parent.0].alive = false;
            cur_parent = grand_parent;
        }
    }

    pub fn mark_node_decode_tail(&mut self, nid: RadixNodeId) {
        if nid != RadixNodeId::ROOT && self.nodes[nid.0].children.is_empty() {
            self.nodes[nid.0].is_decode_tail = true;
        }
    }

    fn has_non_decode_mamba_descendant(&self, nid: RadixNodeId) -> bool {
        let n = &self.nodes[nid.0];
        for &cid in n.children.values() {
            let c = &self.nodes[cid.0];
            if (c.mamba_value.is_some() && !c.is_decode_tail)
                || self.has_non_decode_mamba_descendant(cid)
            {
                return true;
            }
        }
        false
    }

    fn is_reclaimable_decode_tail_leaf(&self, nid: RadixNodeId) -> bool {
        let n = &self.nodes[nid.0];
        n.children.is_empty()
            && n.is_decode_tail
            && n.full_lock_ref == 0
            && n.mamba_lock_ref == 0
            && n.parent
                .is_some_and(|pid| self.nodes[pid.0].mamba_value.is_some())
    }

    /// Evicts up to `num_mamba_slots` unlocked GDN `StateBundle` slots in `mamba_lru` order
    /// (`mamba_radix_cache.py:865-899`). Internal nodes are tombstoned; leaf nodes are deleted
    /// along with any exposed tombstone ancestors.
    pub fn has_evictable_internal_mamba(&self) -> bool {
        !self.disable
            && self.mamba_lru.iter().any(|&nid| {
                let n = &self.nodes[nid.0];
                (n.mamba_lock_ref == 0
                    && n.children.len() == 1
                    && self.has_non_decode_mamba_descendant(nid))
                    || self.is_reclaimable_decode_tail_leaf(nid)
            })
    }

    pub fn evict_mamba(&mut self, num_mamba_slots: usize) -> RadixEvictResult {
        let mut out = RadixEvictResult::default();
        if self.disable || num_mamba_slots == 0 {
            return out;
        }

        while out.freed_mamba_slots.len() < num_mamba_slots {
            // Priority 1: Superseded single-child internal nodes that have a non-decode-tail
            // Mamba descendant (`has_non_decode_mamba_descendant`), freeing 0 KV pages while
            // never tombstoning a prompt frontier whose only child is an unverified decode tail.
            // Priority 2: Unverified decode-tail leaf nodes (`is_reclaimable_decode_tail_leaf`)
            // whose parent prompt frontier still holds a live Mamba snapshot.
            // Priority 3: Standard `mamba_lru` order.
            let candidate = self
                .mamba_lru
                .iter()
                .copied()
                .find(|&nid| {
                    let n = &self.nodes[nid.0];
                    n.mamba_lock_ref == 0
                        && n.children.len() == 1
                        && self.has_non_decode_mamba_descendant(nid)
                })
                .or_else(|| {
                    self.mamba_lru
                        .iter()
                        .copied()
                        .find(|&nid| self.is_reclaimable_decode_tail_leaf(nid))
                })
                .or_else(|| {
                    self.mamba_lru.iter().copied().find(|&nid| {
                        let n = &self.nodes[nid.0];
                        n.mamba_lock_ref == 0 && (!n.children.is_empty() || n.full_lock_ref == 0)
                    })
                });
            let Some(nid) = candidate else {
                break;
            };
            if !self.nodes[nid.0].children.is_empty() {
                // Internal node: tombstone its Mamba state while preserving KV pages for children.
                if let Some(ms) = self.nodes[nid.0].mamba_value.take() {
                    out.freed_mamba_slots.push(ms);
                    self.mamba_evictable_slots = self.mamba_evictable_slots.saturating_sub(1);
                }
                self.mamba_lru_remove(nid);
            } else {
                self.delete_leaf_and_cleanup_tombstones(nid, &mut out);
            }
        }
        out
    }

    /// Evicts unlocked leaf nodes in `full_lru` order until at least `num_pages` KV pages are freed
    /// (`mamba_radix_cache.py:901-920`).
    pub fn evict_full_pages(&mut self, num_pages: usize) -> RadixEvictResult {
        let mut out = RadixEvictResult::default();
        if self.disable || num_pages == 0 {
            return out;
        }

        while out.freed_kv_pages.len() < num_pages {
            let candidate = self
                .full_lru
                .iter()
                .copied()
                .find(|&nid| {
                    let n = &self.nodes[nid.0];
                    n.children.is_empty()
                        && n.is_decode_tail
                        && n.full_lock_ref == 0
                        && n.mamba_lock_ref == 0
                })
                .or_else(|| {
                    self.full_lru.iter().copied().find(|&nid| {
                        let n = &self.nodes[nid.0];
                        n.children.is_empty() && n.full_lock_ref == 0 && n.mamba_lock_ref == 0
                    })
                });
            let Some(nid) = candidate else {
                break;
            };
            self.delete_leaf_and_cleanup_tombstones(nid, &mut out);
        }
        out
    }

    /// Verifies `SGLang`'s structural invariants across the tree and both LRU lists.
    pub fn sanity_check(&self) -> Result<(), String> {
        let mut calc_evictable_pages = 0usize;
        let mut calc_protected_pages = 0usize;
        let mut calc_evictable_mamba = 0usize;
        let mut calc_protected_mamba = 0usize;

        for node in &self.nodes {
            if !node.alive || node.id == RadixNodeId::ROOT {
                continue;
            }
            if node.full_lock_ref < node.mamba_lock_ref {
                return Err(format!(
                    "node {:?} violated full_lock_ref ({}) >= mamba_lock_ref ({})",
                    node.id, node.full_lock_ref, node.mamba_lock_ref
                ));
            }
            if node.children.is_empty() && node.mamba_value.is_none() && node.full_lock_ref == 0 {
                return Err(format!("node {:?} is an unlocked tombstone leaf", node.id));
            }
            if node.full_lock_ref == 0 {
                calc_evictable_pages += node.value.len();
            } else {
                calc_protected_pages += node.value.len();
            }
            if node.mamba_value.is_some() {
                if node.mamba_lock_ref == 0 {
                    calc_evictable_mamba += 1;
                } else {
                    calc_protected_mamba += 1;
                }
            }
        }

        if calc_evictable_pages != self.full_evictable_pages {
            return Err(format!(
                "full_evictable_pages mismatch: tracked {} vs actual {}",
                self.full_evictable_pages, calc_evictable_pages
            ));
        }
        if calc_protected_pages != self.full_protected_pages {
            return Err(format!(
                "full_protected_pages mismatch: tracked {} vs actual {}",
                self.full_protected_pages, calc_protected_pages
            ));
        }
        if calc_evictable_mamba != self.mamba_evictable_slots {
            return Err(format!(
                "mamba_evictable_slots mismatch: tracked {} vs actual {}",
                self.mamba_evictable_slots, calc_evictable_mamba
            ));
        }
        if calc_protected_mamba != self.mamba_protected_slots {
            return Err(format!(
                "mamba_protected_slots mismatch: tracked {} vs actual {}",
                self.mamba_protected_slots, calc_protected_mamba
            ));
        }
        Ok(())
    }
}

/// Planned `C = 64` checkpoint within a forward/verify step.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PlannedCheckpoint {
    /// 1-based query token offset (`1..=query_len`) passed to `w34_gdn_batch::checkpoint_token_offsets`.
    pub trigger_offset_1based: i32,
    /// Destination `StateBundle` slot (`ping_pong_slots[track_idx]`) passed to `w34_gdn_batch::checkpoint_dst_indices`.
    pub dst_slot: i32,
    /// Ping-pong buffer index (`0` or `1`) used for this checkpoint.
    pub track_idx: usize,
    /// Absolute sequence length (`ctx_len_before + trigger_offset_1based`) at the checkpoint boundary.
    pub abs_seq_len: usize,
}

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct RadixDonationOutcome {
    pub donated_seq_len: usize,
    pub freed_kv_pages: Vec<i32>,
    pub freed_draft_pages: Vec<i32>,
    pub freed_mamba_slots: Vec<i32>,
}

/// Per-request `HybridReqToTokenPool` state (`SGLang` `extra_buffer` ping-pong checkpoint tracking).
#[derive(Clone, Debug)]
pub struct ReqRadixState {
    pub page_size: usize,
    pub mamba_chunk_size: usize,
    /// Committed token IDs (`processed_len == emitted_len` outside active verify).
    pub token_ids: Vec<i32>,
    /// Logical KV `page_id`s (`block_table`). The first `protected_pages` are locked in `MambaRadixCache`.
    pub kv_pages: Vec<i32>,
    /// Optional draft-context KV `page_id`s.
    pub draft_pages: Vec<i32>,
    /// Number of prefix KV pages currently shared with and locked in `MambaRadixCache`.
    pub protected_pages: usize,
    /// Currently locked node in `MambaRadixCache` (`RadixNodeId::ROOT` if none).
    pub last_locked_node: RadixNodeId,
    /// Private working `StateBundle` slot for this request.
    pub active_state_slot: i32,
    /// When restored from a `MambaRadixCache` prefix hit, holds the cached `StateBundle` slot
    /// to read from on the first continuation step (`state_in_indices`), while writing to
    /// `active_state_slot` (`state_out_indices`) for zero-copy one-pass COW!
    pub cow_src_state_slot: Option<i32>,
    ///Whether the request has a valid initial GDN state (`true` after prefix hit or first prefill chunk).
    pub has_initial_state: bool,
    /// 2-slot ping-pong `StateBundle` track buffer (`SGLang` `mamba_ping_pong_track_buffer`).
    pub ping_pong_slots: [i32; 2],
    /// Next ping-pong index (`0` or `1`) to write on the next `C = 64` boundary.
    pub next_track_idx: usize,
    /// Sequence length of the most recently **committed** `C = 64` checkpoint stored at
    /// `ping_pong_slots[next_track_idx ^ 1]`.
    pub last_track_seqlen: Option<usize>,
}

impl ReqRadixState {
    pub fn new(
        page_size: usize,
        mamba_chunk_size: usize,
        active_state_slot: i32,
        ping_pong_slots: [i32; 2],
    ) -> Self {
        Self {
            page_size,
            mamba_chunk_size,
            token_ids: Vec::new(),
            kv_pages: Vec::new(),
            draft_pages: Vec::new(),
            protected_pages: 0,
            last_locked_node: RadixNodeId::ROOT,
            active_state_slot,
            cow_src_state_slot: None,
            has_initial_state: false,
            ping_pong_slots,
            next_track_idx: 0,
            last_track_seqlen: None,
        }
    }

    /// Binds a warm prefix hit from `MambaRadixCache::match_prefix` (locking `match_res.last_node`).
    pub fn restore_prefix_hit(
        &mut self,
        radix: &mut MambaRadixCache,
        prompt_tokens: &[i32],
        match_res: &RadixMatchResult,
    ) {
        if match_res.matched_tokens == 0 || match_res.mamba_slot.is_none() {
            return;
        }
        radix.inc_lock_ref(match_res.last_node);
        self.token_ids.clear();
        self.token_ids
            .extend_from_slice(&prompt_tokens[..match_res.matched_tokens]);
        self.kv_pages = match_res.kv_pages.clone();
        self.draft_pages = match_res.draft_pages.clone();
        self.protected_pages = self.kv_pages.len();
        self.last_locked_node = match_res.last_node;
        self.cow_src_state_slot = match_res.mamba_slot;
        self.has_initial_state = true;
    }

    /// Returns the `state_in_slot` for the next forward/verify launch (`cow_src_state_slot` if set,
    /// otherwise `active_state_slot`).
    #[inline]
    pub fn current_state_in_slot(&self) -> i32 {
        self.cow_src_state_slot.unwrap_or(self.active_state_slot)
    }

    /// Plans `C = 64` checkpoints crossed when advancing `query_len` tokens starting from
    /// `self.token_ids.len()`. Up to 2 checkpoints fit in the ping-pong buffer per launch.
    pub fn plan_checkpoints(&self, query_len: usize) -> Vec<PlannedCheckpoint> {
        let seq_before = self.token_ids.len();
        let mut out = Vec::new();
        let mut track_idx = self.next_track_idx;
        for step in 0..query_len {
            let abs_seq = seq_before + step + 1;
            if abs_seq.is_multiple_of(self.mamba_chunk_size) {
                out.push(PlannedCheckpoint {
                    trigger_offset_1based: (step + 1) as i32,
                    dst_slot: self.ping_pong_slots[track_idx],
                    track_idx,
                    abs_seq_len: abs_seq,
                });
                track_idx ^= 1;
            }
        }
        // If > 2 checkpoints were crossed in a single multi-chunk prefill call, only the last 2
        // need to be captured into the 2-slot ping-pong buffer (or callers chunk at `C = 64`).
        if out.len() > 2 {
            out.drain(0..out.len() - 2);
        }
        out
    }

    /// Commits `accepted_tokens` after a prefill chunk, decode step, or speculative verify step.
    /// Crucially, only planned checkpoints at `abs_seq_len <= new_committed_len` are committed!
    /// Any speculative checkpoint past a mid-block rejection is ignored so `last_track_seqlen`
    /// and `next_track_idx` remain strictly consistent with committed tokens.
    pub fn commit_tokens(
        &mut self,
        accepted_tokens: &[i32],
        planned_checkpoints: &[PlannedCheckpoint],
    ) {
        if accepted_tokens.is_empty() {
            return;
        }
        self.token_ids.extend_from_slice(accepted_tokens);
        self.cow_src_state_slot = None;
        self.has_initial_state = true;
        let committed_len = self.token_ids.len();
        for ckpt in planned_checkpoints {
            if ckpt.abs_seq_len <= committed_len {
                self.last_track_seqlen = Some(ckpt.abs_seq_len);
                self.next_track_idx = ckpt.track_idx ^ 1;
            }
        }
    }

    /// Truncates any speculative tail KV pages beyond `ceil(self.token_ids.len() / page_size)`,
    /// returning the freed `page_id`s to be returned to `KvPool`. Never frees `protected_pages`.
    pub fn rollback_uncommitted_kv_pages(&mut self) -> Vec<i32> {
        let needed_pages = self
            .token_ids
            .len()
            .div_ceil(self.page_size)
            .max(self.protected_pages);
        if self.kv_pages.len() > needed_pages {
            self.kv_pages.split_off(needed_pages)
        } else {
            Vec::new()
        }
    }

    /// Donates the most recent committed `C = 64` checkpoint (`ping_pong_slots[next_track_idx ^ 1]`)
    /// into `MambaRadixCache` (`mamba_radix_cache.py:570-820`).
    ///
    /// - When `finish_req == false` (chunked prefill / mid-flight donation), `replacement_slot`
    ///   must be `Some(new_slot)` to replenish the donated ping-pong slot, and the new terminal
    ///   node in `MambaRadixCache` is locked for continued execution.
    /// - When `finish_req == true`, releases the request's radix lock and returns all non-cached
    ///   tail KV pages (`kv_pages[donated_pages..]`) and private GDN state slots so the caller can
    ///   return them to `KvPool` and `StateBundlePool`.
    pub fn donate_committed_checkpoint(
        &mut self,
        radix: &mut MambaRadixCache,
        replacement_slot: Option<i32>,
        finish_req: bool,
    ) -> RadixDonationOutcome {
        assert!(
            finish_req || replacement_slot.is_some(),
            "replacement_slot must be Some(...) when finish_req == false"
        );
        let mut outcome = RadixDonationOutcome::default();

        if let Some(cache_len) = self.last_track_seqlen
            && cache_len > self.protected_pages * self.page_size
            && cache_len <= self.token_ids.len()
            && cache_len.is_multiple_of(self.page_size)
        {
            let cache_pages = cache_len / self.page_size;
            let keep_idx = self.next_track_idx ^ 1;
            let donated_mamba_slot = self.ping_pong_slots[keep_idx];
            let draft_slice = if self.draft_pages.len() >= cache_pages {
                &self.draft_pages[..cache_pages]
            } else {
                &[]
            };
            let had_protected_prompt = self.protected_pages > 0;
            let ins = radix.insert(
                &self.token_ids[..cache_len],
                &self.kv_pages[..cache_pages],
                draft_slice,
                donated_mamba_slot,
                self.protected_pages,
            );
            if finish_req && had_protected_prompt && !ins.mamba_already_existed {
                radix.mark_node_decode_tail(ins.terminal_node);
            }
            outcome.donated_seq_len = cache_len;
            outcome
                .freed_kv_pages
                .extend_from_slice(&ins.freed_duplicate_kv_pages);
            outcome
                .freed_draft_pages
                .extend_from_slice(&ins.freed_duplicate_draft_pages);

            // If existing tree pages replaced some of our newly allocated pages, update `self.kv_pages`
            // to point to the canonical tree pages (`mamba_radix_cache.py:766-803`).
            let canonical_pages = radix.collect_node_kv_pages(ins.terminal_node);
            if canonical_pages.len() == cache_pages {
                self.kv_pages[..cache_pages].copy_from_slice(&canonical_pages);
            }

            if ins.mamba_already_existed {
                // Keep `donated_mamba_slot` in our ping-pong buffer (or free `replacement_slot` if provided).
                if let Some(rep) = replacement_slot {
                    outcome.freed_mamba_slots.push(rep);
                }
            } else if let Some(rep) = replacement_slot {
                self.ping_pong_slots[keep_idx] = rep;
            } else {
                // `finish_req == true`: mark donated slot as consumed (-1) so cleanup doesn't free it.
                self.ping_pong_slots[keep_idx] = -1;
            }

            radix.dec_lock_ref(self.last_locked_node);
            if finish_req {
                self.last_locked_node = RadixNodeId::ROOT;
                self.protected_pages = cache_pages;
            } else {
                radix.inc_lock_ref(ins.terminal_node);
                self.last_locked_node = ins.terminal_node;
                self.protected_pages = cache_pages;
            }
            self.last_track_seqlen = None;
        } else if finish_req {
            radix.dec_lock_ref(self.last_locked_node);
            self.last_locked_node = RadixNodeId::ROOT;
        }

        if finish_req {
            if self.kv_pages.len() > self.protected_pages {
                outcome
                    .freed_kv_pages
                    .extend_from_slice(&self.kv_pages[self.protected_pages..]);
            }
            if self.draft_pages.len() > self.protected_pages {
                outcome
                    .freed_draft_pages
                    .extend_from_slice(&self.draft_pages[self.protected_pages..]);
            }
            if self.active_state_slot >= 0 {
                outcome.freed_mamba_slots.push(self.active_state_slot);
                self.active_state_slot = -1;
            }
            for s in &mut self.ping_pong_slots {
                if *s >= 0 {
                    outcome.freed_mamba_slots.push(*s);
                    *s = -1;
                }
            }
        }

        outcome
    }

    /// Aborts an in-flight request without donating any uncommitted checkpoint, unlocking
    /// `last_locked_node` and returning all private KV pages and GDN state slots.
    pub fn abort_and_release(&mut self, radix: &mut MambaRadixCache) -> RadixDonationOutcome {
        self.last_track_seqlen = None;
        self.donate_committed_checkpoint(radix, None, true)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_radix_cache_initialization_and_sanity() {
        let cache = MambaRadixCache::new(16, 64);
        assert_eq!(cache.page_size, 16);
        assert_eq!(cache.mamba_chunk_size, 64);
        assert_eq!(cache.total_lookups, 0);
        assert_eq!(cache.total_hit_tokens, 0);
        assert!(cache.sanity_check().is_ok());
    }

    #[test]
    fn test_radix_insert_exact_and_match() {
        let mut cache = MambaRadixCache::new(16, 64);
        let tokens: Vec<i32> = (0..64).collect();
        let kv_pages = vec![101, 102, 103, 104];
        let draft_pages = vec![201, 202, 203, 204];

        let ins = cache.insert(&tokens, &kv_pages, &draft_pages, 42, 0);
        assert_eq!(ins.existing_prefix_tokens, 0);
        assert!(!ins.mamba_already_existed);
        assert!(ins.freed_duplicate_kv_pages.is_empty());
        assert!(cache.sanity_check().is_ok());

        // Exact match
        let m = cache.match_prefix(&tokens);
        assert_eq!(m.matched_tokens, 64);
        assert_eq!(m.kv_pages, kv_pages);
        assert_eq!(m.draft_pages, draft_pages);
        assert_eq!(m.mamba_slot, Some(42));
        assert_eq!(m.last_node, ins.terminal_node);

        // Query with extra tokens should still match the 64-token prefix
        let mut query_longer = tokens.clone();
        query_longer.extend_from_slice(&[999, 1000]);
        let m_longer = cache.match_prefix(&query_longer);
        assert_eq!(m_longer.matched_tokens, 64);
        assert_eq!(m_longer.mamba_slot, Some(42));

        // Query with fewer tokens (e.g. 48 tokens = 3 pages)
        let query_shorter: Vec<i32> = (0..48).collect();
        let m_shorter = cache.match_prefix(&query_shorter);
        // Prefix of 48 tokens splits the 64-token node into a 48-token prefix (tombstone) and 16-token leaf.
        // Since the 48-token node has no Mamba checkpoint (checkpoint was at token 64), match returns 0 tokens
        // with mamba_branching_seqlen recording the KV branch.
        assert_eq!(m_shorter.matched_tokens, 0);
        assert_eq!(m_shorter.mamba_slot, None);
    }

    #[test]
    fn test_radix_split_and_branch_invariants() {
        let mut cache = MambaRadixCache::new(16, 64);

        // Insert prompt A: 64 tokens [0..64]
        let tokens_a: Vec<i32> = (0..64).collect();
        let kv_a = vec![1, 2, 3, 4];
        cache.insert(&tokens_a, &kv_a, &[], 10, 0);
        assert!(cache.sanity_check().is_ok());

        // Insert prompt B: sharing first 32 tokens (2 pages), diverging in last 32 tokens
        let mut tokens_b: Vec<i32> = (0..32).collect();
        tokens_b.extend(100..132);
        let kv_b = vec![1, 2, 5, 6];
        let ins_b = cache.insert(&tokens_b, &kv_b, &[], 20, 0);

        assert_eq!(ins_b.existing_prefix_tokens, 32);
        // Pages 1 and 2 were duplicate of existing tree pages
        assert_eq!(ins_b.freed_duplicate_kv_pages, vec![1, 2]);
        assert!(cache.sanity_check().is_ok());

        // Verify match A
        let match_a = cache.match_prefix(&tokens_a);
        assert_eq!(match_a.matched_tokens, 64);
        assert_eq!(match_a.mamba_slot, Some(10));
        assert_eq!(match_a.kv_pages, vec![1, 2, 3, 4]);

        // Verify match B
        let match_b = cache.match_prefix(&tokens_b);
        assert_eq!(match_b.matched_tokens, 64);
        assert_eq!(match_b.mamba_slot, Some(20));
        assert_eq!(match_b.kv_pages, vec![1, 2, 5, 6]);
    }

    #[test]
    fn test_radix_dual_lock_and_eviction() {
        let mut cache = MambaRadixCache::new(16, 64);
        let tokens: Vec<i32> = (0..64).collect();
        let kv_pages = vec![10, 20, 30, 40];
        let ins = cache.insert(&tokens, &kv_pages, &[], 99, 0);

        // Lock node
        cache.inc_lock_ref(ins.terminal_node);
        assert!(cache.sanity_check().is_ok());

        // Eviction while locked must free nothing
        let ev_locked = cache.evict_full_pages(4);
        assert!(ev_locked.freed_kv_pages.is_empty());

        let ev_mamba_locked = cache.evict_mamba(1);
        assert!(ev_mamba_locked.freed_mamba_slots.is_empty());

        // Unlock node
        cache.dec_lock_ref(ins.terminal_node);
        assert!(cache.sanity_check().is_ok());

        // Evict leaf via evict_full_pages
        let ev_full = cache.evict_full_pages(4);
        assert_eq!(ev_full.freed_kv_pages, vec![10, 20, 30, 40]);
        assert_eq!(ev_full.freed_mamba_slots, vec![99]);
        assert!(cache.sanity_check().is_ok());
    }

    #[test]
    fn test_req_radix_state_planning_and_donation() {
        let mut radix = MambaRadixCache::new(16, 64);
        let mut req = ReqRadixState::new(16, 64, 5, [6, 7]);

        assert_eq!(req.current_state_in_slot(), 5);
        assert_eq!(req.plan_checkpoints(32), vec![]); // doesn't cross 64 boundary

        let ckpts = req.plan_checkpoints(64);
        assert_eq!(ckpts.len(), 1);
        assert_eq!(ckpts[0].trigger_offset_1based, 64);
        assert_eq!(ckpts[0].dst_slot, 6);
        assert_eq!(ckpts[0].abs_seq_len, 64);

        let tokens: Vec<i32> = (0..64).collect();
        req.kv_pages = vec![100, 101, 102, 103];
        req.commit_tokens(&tokens, &ckpts);
        assert_eq!(req.last_track_seqlen, Some(64));
        assert_eq!(req.token_ids.len(), 64);

        // Donate committed checkpoint into radix cache with a replacement slot
        let outcome = req.donate_committed_checkpoint(&mut radix, Some(8), false);
        assert_eq!(outcome.donated_seq_len, 64);
        assert!(radix.sanity_check().is_ok());

        // Match the donated prefix
        let match_res = radix.match_prefix(&tokens);
        assert_eq!(match_res.matched_tokens, 64);
        assert_eq!(match_res.mamba_slot, Some(6));
    }

    #[test]
    fn test_radix_multi_checkpoint_drain_rollback_and_multi_node_lru() {
        // 1. Multi-chunk prefill crossing > 2 checkpoints (256 tokens = 4 * 64) retains only the last 2.
        let mut req = ReqRadixState::new(16, 64, 5, [6, 7]);
        let ckpts_256 = req.plan_checkpoints(256);
        assert_eq!(ckpts_256.len(), 2);
        assert_eq!(ckpts_256[0].abs_seq_len, 192);
        assert_eq!(ckpts_256[0].dst_slot, 6);
        assert_eq!(ckpts_256[0].track_idx, 0);
        assert_eq!(ckpts_256[1].abs_seq_len, 256);
        assert_eq!(ckpts_256[1].dst_slot, 7);
        assert_eq!(ckpts_256[1].track_idx, 1);

        // 2. Partial commit (mid-block speculative rejection at 200 tokens < 256) commits only <= 200.
        let tokens_200: Vec<i32> = (0..200).collect();
        req.kv_pages = (100..116).collect(); // 16 pages (256 capacity), only 13 pages needed for 200 tokens
        req.protected_pages = 4;
        req.commit_tokens(&tokens_200, &ckpts_256);
        assert_eq!(req.last_track_seqlen, Some(192));
        assert_eq!(req.next_track_idx, 1);
        let freed = req.rollback_uncommitted_kv_pages();
        assert_eq!(freed, vec![113, 114, 115]);
        assert_eq!(req.kv_pages.len(), 13);

        // 3. Multi-node LRU eviction order: insert A, B, C, touch A, then evict one by one.
        let mut cache = MambaRadixCache::new(16, 64);
        let tok_a: Vec<i32> = (0..64).collect();
        let tok_b: Vec<i32> = (100..164).collect();
        let tok_c: Vec<i32> = (200..264).collect();
        cache.insert(&tok_a, &[1, 2, 3, 4], &[], 11, 0);
        cache.insert(&tok_b, &[5, 6, 7, 8], &[], 22, 0);
        cache.insert(&tok_c, &[9, 10, 11, 12], &[], 33, 0);
        // Touch A so LRU order becomes B (oldest), C, A (newest).
        assert_eq!(cache.match_prefix(&tok_a).mamba_slot, Some(11));
        assert_eq!(cache.mamba_evictable_slots(), 3);

        let ev1 = cache.evict_mamba(1);
        assert_eq!(ev1.freed_mamba_slots, vec![22]);
        let ev2 = cache.evict_mamba(1);
        assert_eq!(ev2.freed_mamba_slots, vec![33]);
        let ev3 = cache.evict_mamba(1);
        assert_eq!(ev3.freed_mamba_slots, vec![11]);
        assert_eq!(cache.mamba_evictable_slots(), 0);
        assert!(cache.sanity_check().is_ok());
    }

    #[test]
    fn test_evict_mamba_tombstones_superseded_linear_turns_before_waiting_session_leaves() {
        let mut cache = MambaRadixCache::new(16, 64);

        // Wave 1: Session 0 finishes Turn 1 (64 tokens, leaf node holding mamba slot 10).
        let s0_t1: Vec<i32> = (0..64).collect();
        cache.insert(&s0_t1, &[1, 2, 3, 4], &[], 10, 0);

        // Wave 2: Session 1 advances through Chunk 1 (64 tokens, mamba slot 20) and then
        // Chunk 2 (128 tokens, mamba slot 21), making Chunk 1 a superseded single-child
        // internal node with a newer timestamp than Session 0's waiting leaf.
        let s1_c2: Vec<i32> = (1000..1128).collect();
        cache.insert(&s1_c2[..64], &[10, 11, 12, 13], &[], 20, 0);
        cache.insert(&s1_c2, &[10, 11, 12, 13, 14, 15, 16, 17], &[], 21, 4);

        // Reclaiming 1 Mamba slot under pool pressure must tombstone Session 1's superseded
        // linear intermediate node (slot 20, freeing 0 KV pages) rather than deleting
        // Session 0's waiting turn frontier leaf (slot 10 and its 4 KV pages).
        let ev = cache.evict_mamba(1);
        assert_eq!(ev.freed_mamba_slots, vec![20]);
        assert!(ev.freed_kv_pages.is_empty());
        let hit_s0 = cache.match_prefix(&s0_t1);
        assert_eq!(hit_s0.matched_tokens, 64);
        assert_eq!(hit_s0.mamba_slot, Some(10));
        assert!(cache.sanity_check().is_ok());
    }

    #[test]
    fn test_evict_mamba_reclaims_decode_tail_leaf_preserving_parent_prompt_checkpoint() {
        let mut radix = MambaRadixCache::new(16, 64);
        let mut req = ReqRadixState::new(16, 64, 1, [2, 3]);

        // Prefill 64 prompt tokens and donate prompt checkpoint (mamba slot 2).
        let prompt: Vec<i32> = (0..64).collect();
        req.kv_pages = vec![10, 11, 12, 13];
        let ckpts_prefill = req.plan_checkpoints(64);
        req.commit_tokens(&prompt, &ckpts_prefill);
        let prefill_don = req.donate_committed_checkpoint(&mut radix, Some(4), false);
        assert_eq!(prefill_don.donated_seq_len, 64);

        // Decode 64 generated tokens and donate finish_req decode tail (mamba slot 3).
        let generated: Vec<i32> = (1000..1064).collect();
        req.kv_pages.extend_from_slice(&[14, 15, 16, 17]);
        let ckpts_decode = req.plan_checkpoints(64);
        req.commit_tokens(&generated, &ckpts_decode);
        let finish_don = req.donate_committed_checkpoint(&mut radix, None, true);
        assert_eq!(finish_don.donated_seq_len, 128);

        // Reclaiming 1 Mamba slot must delete the unverified decode-tail leaf (slot 3 and pages
        // 14..17) rather than tombstoning the parent prompt frontier (slot 2), so a subsequent
        // turn that diverges from the generated tail still matches the 64-token prompt prefix.
        let ev = radix.evict_mamba(1);
        assert_eq!(ev.freed_mamba_slots, vec![3]);
        assert_eq!(ev.freed_kv_pages, vec![14, 15, 16, 17]);
        let mut next_turn = prompt.clone();
        next_turn.extend(2000..2064);
        let hit = radix.match_prefix(&next_turn);
        assert_eq!(hit.matched_tokens, 64);
        assert_eq!(hit.mamba_slot, Some(2));
        assert!(radix.sanity_check().is_ok());
    }
}
