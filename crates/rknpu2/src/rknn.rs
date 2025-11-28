use std::mem::MaybeUninit;

use rknpu2_sys::{
    _rknn_core_mask::{
        RKNN_NPU_CORE_0, RKNN_NPU_CORE_0_1, RKNN_NPU_CORE_0_1_2, RKNN_NPU_CORE_1, RKNN_NPU_CORE_2,
        RKNN_NPU_CORE_ALL, RKNN_NPU_CORE_AUTO,
    },
    _rknn_mem_alloc_flags::{
        RKNN_FLAG_MEMORY_CACHEABLE, RKNN_FLAG_MEMORY_FLAGS_DEFAULT, RKNN_FLAG_MEMORY_NON_CACHEABLE,
        RKNN_FLAG_MEMORY_TRY_ALLOC_SRAM,
    },
    rknn_context,
};

#[cfg(any(feature = "rk3576", feature = "rk35xx"))]
use crate::io::{input::IntoInputs, output::Output};
use {
    crate::{
        Error,
        api::RKNNAPI,
        mem::TensorMem,
        query::{InputAttr, Io, OutputAttr, Query, QueryWithInput},
    },
    std::{ffi::c_void, ptr},
};

/// Main rknn struct with ability to query the model and run inference.
pub struct RKNN<A: RKNNAPI> {
    pub(crate) ctx: rknn_context,
    pub(crate) api: A,
}

impl<A: RKNNAPI> RKNN<A> {
    pub fn query<T: Query>(&self) -> Result<T, Error> {
        let mut result = std::mem::MaybeUninit::<T::Output>::uninit();
        let ret = unsafe {
            self.api.query(
                self.ctx,
                T::QUERY_TYPE,
                &mut result as *mut _ as *mut c_void,
                std::mem::size_of::<T::Output>() as u32,
            )?
        };
        if ret != 0 {
            return Err(ret.into());
        }
        unsafe { Ok(result.assume_init().into()) }
    }

    pub fn query_with_input<T: QueryWithInput>(&self, input: T::Input) -> Result<T, Error> {
        let mut result = std::mem::MaybeUninit::<T::Output>::uninit();

        // SAFETY: we are immediately initializing the memory via `prepare`.
        T::prepare(input, unsafe { &mut *result.as_mut_ptr() });

        let ret = unsafe {
            self.api.query(
                self.ctx,
                T::QUERY_TYPE,
                result.as_mut_ptr() as *mut _ as *mut c_void,
                std::mem::size_of::<T::Output>() as u32,
            )?
        };
        if ret != 0 {
            return Err(ret.into());
        }
        unsafe { Ok(result.assume_init().into()) }
    }

    pub fn run(&self) -> Result<(), Error> {
        let ret = unsafe { self.api.run(self.ctx, ptr::null_mut())? };
        if ret != 0 {
            return Err(ret.into());
        }
        Ok(())
    }

    pub fn create_mem2(&self, size: u64, flags: MemAllocFlags) -> Result<TensorMem, Error> {
        let flags_u32: u32 = flags.into();

        let ret: *mut crate::rknpu2_sys::rknn_tensor_mem =
            unsafe { self.api.create_mem2(self.ctx, size, flags_u32.into())? };

        if ret.is_null() {
            return Err(Error::NullPointer);
        }

        Ok(TensorMem { mem: ret })
    }

    pub fn set_io_mem(&self, mem: &TensorMem, io: Io, index: u32) -> Result<(), Error> {
        match io {
            Io::Input => {
                let mut attr = self.query_with_input::<InputAttr>(index)?;
                let ret = unsafe { self.api.set_io_mem(self.ctx, mem.mem, &mut attr.inner)? };
                if ret != 0 {
                    return Err(ret.into());
                }
            }
            Io::Output => {
                let mut attr = self.query_with_input::<OutputAttr>(index)?;
                let ret = unsafe { self.api.set_io_mem(self.ctx, mem.mem, &mut attr.inner)? };
                if ret != 0 {
                    return Err(ret.into());
                }
            }
        };

        Ok(())
    }

    #[cfg(any(feature = "rk3576", feature = "rk35xx"))]
    #[cfg_attr(
        feature = "docs",
        doc(cfg(any(feature = "rk35xx", feature = "rk3576")))
    )]
    pub fn set_inputs<'a, I: IntoInputs<'a>>(&self, inputs: I) -> Result<(), Error> {
        let mut tensors = inputs.into_inputs();

        let mut ffi_inputs: Vec<rknpu2_sys::rknn_input> =
            tensors.iter_mut().map(|t| t.as_sys_input()).collect();

        let ret = unsafe {
            self.api
                .inputs_set(self.ctx, ffi_inputs.len() as u32, ffi_inputs.as_mut_ptr())?
        };

        if ret != 0 {
            return Err(ret.into());
        }

        Ok(())
    }

    #[cfg(any(feature = "rk3576", feature = "rk35xx"))]
    #[cfg_attr(
        feature = "docs",
        doc(cfg(any(feature = "rk35xx", feature = "rk3576")))
    )]
    pub fn get_outputs<'a>(&self, outputs: &mut [Output<'a>]) -> Result<(), Error> {
        let mut outputs_ffi = outputs
            .iter_mut()
            .map(|t| t.as_sys_output())
            .collect::<Vec<_>>();

        let ret = unsafe {
            self.api.outputs_get(
                self.ctx,
                outputs_ffi.len() as u32,
                outputs_ffi.as_mut_ptr(),
                std::ptr::null_mut(),
            )?
        };

        for (output, ffi) in outputs.iter_mut().zip(outputs_ffi.iter_mut()) {
            output.from_sys_output(ffi);
        }

        if ret != 0 {
            return Err(ret.into());
        }

        Ok(())
    }

    #[cfg(feature = "rk3576")]
    #[cfg_attr(feature = "docs", doc(cfg(feature = "rk3576")))]
    pub fn set_core_mask(&self, mask: NpuCores) -> Result<(), Error> {
        unsafe {
            self.api.set_core_mask(self.ctx, mask.into())?;
        }

        Ok(())
    }
}

impl<A: RKNNAPI> Drop for RKNN<A> {
    fn drop(&mut self) {
        unsafe {
            self.api.destroy(self.ctx).unwrap();
        }
    }
}

use bitflags::bitflags;

bitflags! {
    /// Flags for specifying which NPU cores to use.
    #[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
    pub struct NpuCores: u32 {
        // Single cores
        const CORE0 = RKNN_NPU_CORE_0;
        const CORE1 = RKNN_NPU_CORE_1;
        const CORE2 = RKNN_NPU_CORE_2;

        // Predefined combinations from the C API
        const CORE0_1    = RKNN_NPU_CORE_0_1;
        const CORE0_1_2  = RKNN_NPU_CORE_0_1_2;

        // "All cores" bitmask that C defines as 0xFFFF (not just 0b111).
        const ALL        = RKNN_NPU_CORE_ALL;
    }
}

impl NpuCores {
    /// Let the driver choose cores automatically.
    pub const fn auto() -> Self {
        // This is equivalent to `Self::empty()`
        // but documents the intent.
        NpuCores::from_bits_truncate(RKNN_NPU_CORE_AUTO)
    }

    /// Start building from "auto"/empty.
    pub const fn builder() -> Self {
        Self::auto()
    }

    /// Add core 0.
    pub const fn with_core0(self) -> Self {
        self.union(Self::CORE0)
    }

    /// Add core 1.
    pub const fn with_core1(self) -> Self {
        self.union(Self::CORE1)
    }

    /// Add core 2.
    pub const fn with_core2(self) -> Self {
        self.union(Self::CORE2)
    }

    /// Convenience: all three cores (0,1,2) only.
    /// (This is 0b111, not 0xFFFF).
    pub const fn cores_0_1_2() -> Self {
        Self::CORE0_1_2
    }

    /// Is this the "auto" / "no specific cores requested" mode?
    pub const fn is_auto(self) -> bool {
        self.bits() == RKNN_NPU_CORE_AUTO
    }
}

// Optional: ergonomic conversions
impl From<NpuCores> for u32 {
    #[inline]
    fn from(mask: NpuCores) -> u32 {
        mask.bits()
    }
}

impl From<u32> for NpuCores {
    #[inline]
    fn from(bits: u32) -> NpuCores {
        // Truncate unknown bits instead of panicking.
        NpuCores::from_bits_truncate(bits)
    }
}

impl From<MemAllocFlags> for u32 {
    #[inline]
    fn from(flags: MemAllocFlags) -> u32 {
        flags.bits()
    }
}

impl From<u32> for MemAllocFlags {
    #[inline]
    fn from(bits: u32) -> MemAllocFlags {
        // Truncate unknown bits instead of panicking.
        MemAllocFlags::from_bits_truncate(bits)
    }
}

bitflags! {
    /// Flags controlling RKNN memory allocation behavior.
    ///
    /// Safe wrapper over:
    /// - `RKNN_FLAG_MEMORY_CACHEABLE`
    /// - `RKNN_FLAG_MEMORY_NON_CACHEABLE`
    /// - `RKNN_FLAG_MEMORY_FLAGS_DEFAULT`
    /// - `RKNN_FLAG_MEMORY_TRY_ALLOC_SRAM`
    #[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
    pub struct MemAllocFlags: u32 {
        /// Use cacheable memory.
        const CACHEABLE      = RKNN_FLAG_MEMORY_CACHEABLE;
        /// Use non-cacheable memory.
        const NON_CACHEABLE  = RKNN_FLAG_MEMORY_NON_CACHEABLE;
        /// RKNN's default memory behavior.
        const DEFAULT        = RKNN_FLAG_MEMORY_FLAGS_DEFAULT;
        /// Try to allocate from on-chip SRAM if possible.
        const TRY_ALLOC_SRAM = RKNN_FLAG_MEMORY_TRY_ALLOC_SRAM;
    }
}

impl MemAllocFlags {
    /// Start from RKNN's default memory flags.
    ///
    /// If `RKNN_FLAG_MEMORY_FLAGS_DEFAULT == 0`, this is the same as `empty()`
    /// but documents intent.
    pub const fn builder() -> Self {
        MemAllocFlags::DEFAULT
    }

    /// Explicitly select "default" cache policy, clearing conflicting bits.
    pub const fn with_default(self) -> Self {
        Self::set_cache_policy(self, CachePolicy::Default)
    }

    /// Select cacheable memory, clearing NON_CACHEABLE/DEFAULT cache-policy bits.
    pub const fn with_cacheable(self) -> Self {
        Self::set_cache_policy(self, CachePolicy::Cacheable)
    }

    /// Select non-cacheable memory, clearing CACHEABLE/DEFAULT cache-policy bits.
    pub const fn with_non_cacheable(self) -> Self {
        Self::set_cache_policy(self, CachePolicy::NonCacheable)
    }

    /// Request SRAM-backed allocations in addition to current policy.
    pub const fn with_try_alloc_sram(self) -> Self {
        self.union(MemAllocFlags::TRY_ALLOC_SRAM)
    }

    /// Is SRAM requested?
    pub const fn wants_sram(self) -> bool {
        self.intersects(MemAllocFlags::TRY_ALLOC_SRAM)
    }

    /// Introspect cache policy in a nice typed way.
    pub const fn cache_policy(self) -> CachePolicy {
        if self.intersects(MemAllocFlags::CACHEABLE) {
            CachePolicy::Cacheable
        } else if self.intersects(MemAllocFlags::NON_CACHEABLE) {
            CachePolicy::NonCacheable
        } else {
            // Either DEFAULT bit is set, or neither is set (if DEFAULT == 0)
            CachePolicy::Default
        }
    }

    // --- private helpers ---

    /// Clear all cache-policy-related bits (CACHEABLE/NON_CACHEABLE/DEFAULT).
    const fn clear_cache_policy_bits(self) -> Self {
        self.difference(
            MemAllocFlags::CACHEABLE
                .union(MemAllocFlags::NON_CACHEABLE)
                .union(MemAllocFlags::DEFAULT),
        )
    }

    const fn set_cache_policy(self, policy: CachePolicy) -> Self {
        let base = self.clear_cache_policy_bits();
        match policy {
            CachePolicy::Default => base.union(MemAllocFlags::DEFAULT),
            CachePolicy::Cacheable => base.union(MemAllocFlags::CACHEABLE),
            CachePolicy::NonCacheable => base.union(MemAllocFlags::NON_CACHEABLE),
        }
    }
}

/// High-level cache policy enum.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CachePolicy {
    Default,
    Cacheable,
    NonCacheable,
}
