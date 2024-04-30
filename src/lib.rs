// offset-allocator/src/lib.rs

use std::fmt::{Debug, Display, Formatter, Result as FmtResult};

use log::debug;
use nonmax::{NonMaxU16, NonMaxU32};

mod small_float;

pub const NUM_TOP_BINS: usize = 32;
pub const BINS_PER_LEAF: usize = 8;
pub const TOP_BINS_INDEX_SHIFT: u32 = 3;
pub const LEAF_BINS_INDEX_MASK: u32 = 7;
pub const NUM_LEAF_BINS: usize = NUM_TOP_BINS * BINS_PER_LEAF;

pub trait NodeIndex: Clone + Copy + Default {
    type NonMax: NodeIndexNonMax + TryFrom<Self> + Into<Self>;

    fn from_u32(val: u32) -> Self;
    fn to_usize(self) -> usize;
}

pub trait NodeIndexNonMax: Clone + Copy + PartialEq + Default + Debug + Display {
    fn to_usize(self) -> usize;
}

pub struct Allocation<NI = u32>
where
    NI: NodeIndex,
{
    pub offset: Option<NonMaxU32>,
    pub metadata: Option<NI::NonMax>,
}

#[derive(Debug)]
pub struct StorageReport {
    pub total_free_space: u32,
    pub largest_free_region: u32,
}

pub struct StorageReportFull {
    pub free_regions: [StorageReportFullRegion; NUM_LEAF_BINS],
}

#[derive(Clone, Copy, Default)]
pub struct StorageReportFullRegion {
    pub size: u32,
    pub count: u32,
}

pub struct Allocator<NI = u32>
where
    NI: NodeIndex,
{
    size: u32,
    max_allocs: u32,
    free_storage: u32,

    used_bins_top: u32,
    used_bins: [u8; NUM_TOP_BINS],
    bin_indices: [Option<NI::NonMax>; NUM_LEAF_BINS],

    nodes: Vec<Node<NI>>,
    free_nodes: Vec<NI::NonMax>,
    free_offset: u32,
}

#[derive(Clone, Copy, Default)]
struct Node<NI = u32>
where
    NI: NodeIndex,
{
    data_offset: u32,
    data_size: u32,
    bin_list_prev: Option<NI::NonMax>,
    bin_list_next: Option<NI::NonMax>,
    neighbor_prev: Option<NI::NonMax>,
    neighbor_next: Option<NI::NonMax>,
    used: bool, // TODO: Merge as bit flag
}

// Utility functions
pub fn find_lowest_bit_set_after(bit_mask: u32, start_bit_index: u32) -> Option<NonMaxU32> {
    let mask_before_start_index = (1 << start_bit_index) - 1;
    let mask_after_start_index = !mask_before_start_index;
    let bits_after = bit_mask & mask_after_start_index;
    if bits_after == 0 {
        None
    } else {
        NonMaxU32::try_from(bits_after.trailing_zeros()).ok()
    }
}

impl<NI> Allocator<NI>
where
    NI: NodeIndex,
{
    // Allocator…
    pub fn new(size: u32, max_allocs: u32) -> Self {
        #[cfg(not(feature = "32_bit_node_indices"))]
        assert!(max_allocs < u16::MAX as u32);

        let mut this = Self {
            size,
            max_allocs,
            free_storage: 0,
            used_bins_top: 0,
            free_offset: 0,
            used_bins: [0; NUM_TOP_BINS],
            bin_indices: [None; NUM_LEAF_BINS],
            nodes: vec![],
            free_nodes: vec![],
        };
        this.reset();
        this
    }

    pub fn reset(&mut self) {
        self.free_storage = 0;
        self.used_bins_top = 0;
        self.free_offset = self.max_allocs - 1;

        self.used_bins.iter_mut().for_each(|bin| *bin = 0);

        self.bin_indices.iter_mut().for_each(|index| *index = None);

        self.nodes = vec![Node::default(); self.max_allocs as usize];

        // Freelist is a stack. Nodes in inverse order so that [0] pops first.
        self.free_nodes = (0..self.max_allocs)
            .map(|i| {
                NI::NonMax::try_from(NI::from_u32(self.max_allocs - i - 1)).unwrap_or_default()
            })
            .collect();

        // Start state: Whole storage as one big node
        // Algorithm will split remainders and push them back as smaller nodes
        self.insert_node_into_bin(self.size, 0);
    }

    pub fn allocate(&mut self, size: u32) -> Option<Allocation<NI>> {
        debug!("Allocator::allocate({})", size);

        // Out of allocations?
        if self.free_offset == 0 {
            return None;
        }

        // Round up to bin index to ensure that alloc >= bin
        // Gives us min bin index that fits the size
        let min_bin_index = small_float::uint_to_float_round_up(size);
        debug!("min_bin_index={}", min_bin_index);

        let min_top_bin_index = min_bin_index >> TOP_BINS_INDEX_SHIFT;
        let min_leaf_bin_index = min_bin_index & LEAF_BINS_INDEX_MASK;

        let mut top_bin_index = min_top_bin_index;
        let mut leaf_bin_index = None;

        // If top bin exists, scan its leaf bin. This can fail (NO_SPACE).
        if (self.used_bins_top & (1 << top_bin_index)) != 0 {
            leaf_bin_index = find_lowest_bit_set_after(
                self.used_bins[top_bin_index as usize] as _,
                min_leaf_bin_index,
            );
        }

        // If we didn't find space in top bin, we search top bin from +1
        let leaf_bin_index = match leaf_bin_index {
            Some(leaf_bin_index) => leaf_bin_index,
            None => {
                debug!("used_bins_top={:x}", self.used_bins_top);
                top_bin_index =
                    find_lowest_bit_set_after(self.used_bins_top, min_top_bin_index + 1)?.into();

                // All leaf bins here fit the alloc, since the top bin was
                // rounded up. Start leaf search from bit 0.
                //
                // NOTE: This search can't fail since at least one leaf bit was
                // set because the top bit was set.
                NonMaxU32::try_from(self.used_bins[top_bin_index as usize].trailing_zeros())
                    .unwrap()
            }
        };

        let bin_index = (top_bin_index << TOP_BINS_INDEX_SHIFT) | u32::from(leaf_bin_index);

        // Pop the top node of the bin. Bin top = node.next.
        debug!("bin_index = {}", bin_index);
        let node_index = self.bin_indices[bin_index as usize].unwrap();
        let node = &mut self.nodes[node_index.to_usize()];
        let node_total_size = node.data_size;
        node.data_size = size;
        node.used = true;
        self.bin_indices[bin_index as usize] = node.bin_list_next;
        if let Some(bin_list_next) = node.bin_list_next {
            self.nodes[bin_list_next.to_usize()].bin_list_prev = None;
        }
        self.free_storage -= node_total_size;
        debug!(
            "Free storage: {} (-{}) (allocate)",
            self.free_storage, node_total_size
        );

        // Bin empty?
        if self.bin_indices[bin_index as usize].is_none() {
            // Remove a leaf bin mask bit
            self.used_bins[top_bin_index as usize] &= !(1 << u32::from(leaf_bin_index));

            // All leaf bins empty?
            if self.used_bins[top_bin_index as usize] == 0 {
                // Remove a top bin mask bit
                self.used_bins_top &= !(1 << top_bin_index);
            }
        }

        // Push back remainder N elements to a lower bin
        let remainder_size = node_total_size - size;
        if remainder_size > 0 {
            let Node {
                data_offset,
                neighbor_next,
                ..
            } = self.nodes[node_index.to_usize()];

            let new_node_index = self.insert_node_into_bin(remainder_size, data_offset + size);

            // Link nodes next to each other so that we can merge them later if both are free
            // And update the old next neighbor to point to the new node (in middle)
            let node = &mut self.nodes[node_index.to_usize()];
            if let Some(neighbor_next) = node.neighbor_next {
                self.nodes[neighbor_next.to_usize()].neighbor_prev = Some(new_node_index);
            }
            self.nodes[new_node_index.to_usize()].neighbor_prev = Some(node_index);
            self.nodes[new_node_index.to_usize()].neighbor_next = neighbor_next;
            self.nodes[node_index.to_usize()].neighbor_next = Some(new_node_index);
        }

        let node = &mut self.nodes[node_index.to_usize()];
        Some(Allocation {
            offset: NonMaxU32::try_from(node.data_offset).ok(),
            metadata: Some(node_index),
        })
    }

    pub fn free(&mut self, allocation: Allocation<NI>) {
        let Some(node_index) = allocation.metadata else {
            panic!("Invalid allocation: no metadata");
        };

        // Merge with neighbors…
        let Node {
            data_offset: mut offset,
            data_size: mut size,
            used,
            ..
        } = self.nodes[node_index.to_usize()];

        // Double delete check
        assert!(used);

        if let Some(neighbor_prev) = self.nodes[node_index.to_usize()].neighbor_prev {
            if !self.nodes[neighbor_prev.to_usize()].used {
                // Previous (contiguous) free node: Change offset to previous
                // node offset. Sum sizes
                let prev_node = &self.nodes[neighbor_prev.to_usize()];
                offset = prev_node.data_offset;
                size += prev_node.data_size;

                // Remove node from the bin linked list and put it in the
                // freelist
                self.remove_node_from_bin(neighbor_prev);

                let prev_node = &self.nodes[neighbor_prev.to_usize()];
                debug_assert_eq!(prev_node.neighbor_next, Some(node_index));
                self.nodes[node_index.to_usize()].neighbor_prev = prev_node.neighbor_prev;
            }
        }

        if let Some(neighbor_next) = self.nodes[node_index.to_usize()].neighbor_next {
            if !self.nodes[neighbor_next.to_usize()].used {
                // Next (contiguous) free node: Offset remains the same. Sum
                // sizes.
                let next_node = &self.nodes[neighbor_next.to_usize()];
                size += next_node.data_size;

                // Remove node from the bin linked list and put it in the
                // freelist
                self.remove_node_from_bin(neighbor_next);

                let next_node = &self.nodes[neighbor_next.to_usize()];
                debug_assert_eq!(next_node.neighbor_prev, Some(node_index));
                self.nodes[node_index.to_usize()].neighbor_next = next_node.neighbor_next;
            }
        }

        let Node {
            neighbor_next,
            neighbor_prev,
            ..
        } = self.nodes[node_index.to_usize()];

        // Insert the removed node to freelist
        debug!(
            "Putting node {} into freelist[{}] (free)",
            node_index,
            self.free_offset + 1
        );
        self.free_offset += 1;
        self.free_nodes[self.free_offset as usize] = node_index;

        // Insert the (combined) free node to bin
        let combined_node_index = self.insert_node_into_bin(size, offset);

        // Connect neighbors with the new combined node
        if let Some(neighbor_next) = neighbor_next {
            self.nodes[combined_node_index.to_usize()].neighbor_next = Some(neighbor_next);
            self.nodes[neighbor_next.to_usize()].neighbor_prev = Some(combined_node_index);
        }
        if let Some(neighbor_prev) = neighbor_prev {
            self.nodes[combined_node_index.to_usize()].neighbor_prev = Some(neighbor_prev);
            self.nodes[neighbor_prev.to_usize()].neighbor_next = Some(combined_node_index);
        }
    }

    fn insert_node_into_bin(&mut self, size: u32, data_offset: u32) -> NI::NonMax {
        // Round down to bin index to ensure that bin >= alloc
        let bin_index = small_float::uint_to_float_round_down(size);

        let top_bin_index = bin_index >> TOP_BINS_INDEX_SHIFT;
        let leaf_bin_index = bin_index & LEAF_BINS_INDEX_MASK;

        // Bin was empty before?
        if self.bin_indices[bin_index as usize].is_none() {
            // Set bin mask bits
            self.used_bins[top_bin_index as usize] |= 1 << leaf_bin_index;
            self.used_bins_top |= 1 << top_bin_index;
        }

        // Take a freelist node and insert on top of the bin linked list (next = old top)
        let top_node_index = self.bin_indices[bin_index as usize];
        let free_offset = self.free_offset;
        let node_index = self.free_nodes[free_offset as usize];
        self.free_offset -= 1;
        debug!(
            "Getting node {} from freelist[{}]",
            node_index,
            self.free_offset + 1
        );
        self.nodes[node_index.to_usize()] = Node {
            data_offset,
            data_size: size,
            bin_list_next: top_node_index,
            ..Node::default()
        };
        if let Some(top_node_index) = top_node_index {
            self.nodes[top_node_index.to_usize()].bin_list_prev = Some(node_index);
        }
        self.bin_indices[bin_index as usize] = Some(node_index);
        debug!("loading bin {}", bin_index);

        self.free_storage += size;
        debug!(
            "Free storage: {} (+{}) (insert_node_into_bin)",
            self.free_storage, size
        );
        node_index
    }

    fn remove_node_from_bin(&mut self, node_index: NI::NonMax) {
        // Copy the node to work around borrow check.
        let node = self.nodes[node_index.to_usize()];

        match node.bin_list_prev {
            Some(bin_list_prev) => {
                // Easy case: We have previous node. Just remove this node from the middle of the list.
                self.nodes[bin_list_prev.to_usize()].bin_list_next = node.bin_list_next;
                if let Some(bin_list_next) = node.bin_list_next {
                    self.nodes[bin_list_next.to_usize()].bin_list_prev = node.bin_list_prev;
                }
            }
            None => {
                // Hard case: We are the first node in a bin. Find the bin.

                // Round down to bin index to ensure that bin >= alloc
                let bin_index = small_float::uint_to_float_round_down(node.data_size);

                let top_bin_index = (bin_index >> TOP_BINS_INDEX_SHIFT) as usize;
                let leaf_bin_index = (bin_index & LEAF_BINS_INDEX_MASK) as usize;

                self.bin_indices[bin_index as usize] = node.bin_list_next;
                if let Some(bin_list_next) = node.bin_list_next {
                    self.nodes[bin_list_next.to_usize()].bin_list_prev = None;
                }

                // Bin empty?
                if self.bin_indices[bin_index as usize].is_none() {
                    // Remove a leaf bin mask bit
                    self.used_bins[top_bin_index as usize] &= !(1 << leaf_bin_index);

                    // All leaf bins empty?
                    if self.used_bins[top_bin_index as usize] == 0 {
                        // Remove a top bin mask bit
                        self.used_bins_top &= !(1 << top_bin_index);
                    }
                }
            }
        }

        // Insert the node to freelist
        debug!(
            "Putting node {} into freelist[{}] (remove_node_from_bin)",
            node_index,
            self.free_offset + 1
        );
        self.free_offset += 1;
        self.free_nodes[self.free_offset as usize] = node_index;

        self.free_storage -= node.data_size;
        debug!(
            "Free storage: {} (-{}) (remove_node_from_bin)",
            self.free_storage, node.data_size
        );
    }

    pub fn allocation_size(&self, allocation: Allocation<NI>) -> u32 {
        allocation
            .metadata
            .and_then(|metadata| self.nodes.get(metadata.to_usize()))
            .map(|node| node.data_size)
            .unwrap_or_default()
    }

    pub fn storage_report(&self) -> StorageReport {
        let mut largest_free_region = 0;
        let mut free_storage = 0;

        // Out of allocations? -> Zero free space
        if self.free_offset > 0 {
            free_storage = self.free_storage;
            if self.used_bins_top > 0 {
                let top_bin_index = 31 - self.used_bins_top.leading_zeros();
                let leaf_bin_index = 31 - self.used_bins[top_bin_index as usize].leading_zeros();
                largest_free_region = small_float::float_to_uint(
                    (top_bin_index << TOP_BINS_INDEX_SHIFT) | leaf_bin_index,
                );
                debug_assert!(free_storage >= largest_free_region);
            }
        }

        StorageReport {
            total_free_space: free_storage,
            largest_free_region,
        }
    }

    pub fn storage_report_full(&self) -> StorageReportFull {
        let mut report = StorageReportFull::default();
        for i in 0..NUM_LEAF_BINS {
            let mut count = 0;
            let mut maybe_node_index = self.bin_indices[i];
            while let Some(node_index) = maybe_node_index {
                maybe_node_index = self.nodes[node_index.to_usize()].bin_list_next;
                count += 1;
            }
            report.free_regions[i] = StorageReportFullRegion {
                size: small_float::float_to_uint(i as u32),
                count,
            }
        }
        report
    }
}

impl Default for StorageReportFull {
    fn default() -> Self {
        Self {
            free_regions: [Default::default(); NUM_LEAF_BINS],
        }
    }
}

impl<NI> Debug for Allocator<NI>
where
    NI: NodeIndex,
{
    fn fmt(&self, f: &mut Formatter<'_>) -> FmtResult {
        self.storage_report().fmt(f)
    }
}

impl NodeIndex for u32 {
    type NonMax = NonMaxU32;

    fn from_u32(val: u32) -> Self {
        val
    }

    fn to_usize(self) -> usize {
        self as usize
    }
}

impl NodeIndex for u16 {
    type NonMax = NonMaxU16;

    fn from_u32(val: u32) -> Self {
        val as u16
    }

    fn to_usize(self) -> usize {
        self as usize
    }
}

impl NodeIndexNonMax for NonMaxU32 {
    fn to_usize(self) -> usize {
        u32::from(self) as usize
    }
}

impl NodeIndexNonMax for NonMaxU16 {
    fn to_usize(self) -> usize {
        u16::from(self) as usize
    }
}
