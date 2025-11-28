/// Trait for RKNN API operations.

#[cfg(any(feature = "rk35xx", feature = "rk3576"))]
use rknpu2_sys::{
    rknn_core_mask, rknn_custom_op, rknn_custom_op_attr, rknn_custom_op_context, rknn_input,
    rknn_matmul_ctx, rknn_matmul_info, rknn_matmul_io_attr, rknn_matmul_shape,
    rknn_matmul_tensor_attr, rknn_output, rknn_output_extend, rknn_quant_params,
};
#[cfg(any(feature = "rk35xx", feature = "rk3576"))]
use std::ffi::c_char;
use {
    rknpu2_sys::{
        RKNN_FLAG_ASYNC_MASK, RKNN_FLAG_COLLECT_MODEL_INFO_ONLY, RKNN_FLAG_COLLECT_PERF_MASK,
        RKNN_FLAG_DISABLE_FLUSH_INPUT_MEM_CACHE, RKNN_FLAG_DISABLE_FLUSH_OUTPUT_MEM_CACHE,
        RKNN_FLAG_DISABLE_PROC_HIGH_PRIORITY, RKNN_FLAG_ENABLE_SRAM,
        RKNN_FLAG_EXECUTE_FALLBACK_PRIOR_DEVICE_GPU, RKNN_FLAG_FENCE_IN_OUTSIDE,
        RKNN_FLAG_FENCE_OUT_OUTSIDE, RKNN_FLAG_INTERNAL_ALLOC_OUTSIDE, RKNN_FLAG_MEM_ALLOC_OUTSIDE,
        RKNN_FLAG_MODEL_BUFFER_ZERO_COPY, RKNN_FLAG_PRIOR_HIGH, RKNN_FLAG_PRIOR_LOW,
        RKNN_FLAG_PRIOR_MEDIUM, RKNN_FLAG_SHARE_SRAM, RKNN_FLAG_SHARE_WEIGHT_MEM, rknn_context,
        rknn_mem_sync_mode, rknn_query_cmd, rknn_run_extend, rknn_tensor_attr, rknn_tensor_mem,
    },
    std::ffi::{c_int, c_void},
};

#[cfg(not(feature = "libloading"))]
pub mod linked;

#[cfg(feature = "libloading")]
pub mod runtime;

use crate::Error;

pub trait RKNNAPI {
    // ───── core runtime ──────────────────────────────────────────────────────
    #[cfg_attr(
        feature = "docs",
        doc(cfg(any(feature = "rk35xx", feature = "rk3576")))
    )]
    #[cfg(any(feature = "rk35xx", feature = "rk3576"))]
    unsafe fn dup_context(
        &self,
        context_in: *mut rknn_context,
        context_out: *mut rknn_context,
    ) -> Result<c_int, Error>;

    unsafe fn destroy(&self, context: rknn_context) -> Result<c_int, Error>;

    unsafe fn query(
        &self,
        context: rknn_context,
        cmd: rknn_query_cmd,
        info: *mut c_void,
        size: u32,
    ) -> Result<c_int, Error>;

    #[cfg_attr(
        feature = "docs",
        doc(cfg(any(feature = "rk35xx", feature = "rk3576")))
    )]
    #[cfg(any(feature = "rk35xx", feature = "rk3576"))]
    unsafe fn inputs_set(
        &self,
        context: rknn_context,
        n_inputs: u32,
        inputs: *mut rknn_input,
    ) -> Result<c_int, Error>;

    #[cfg_attr(
        feature = "docs",
        doc(cfg(any(feature = "rk35xx", feature = "rk3576")))
    )]
    #[cfg(any(feature = "rk35xx", feature = "rk3576"))]
    unsafe fn set_batch_core_num(
        &self,
        context: rknn_context,
        core_num: c_int,
    ) -> Result<c_int, Error>;

    #[cfg_attr(
        feature = "docs",
        doc(cfg(any(feature = "rk35xx", feature = "rk3576")))
    )]
    #[cfg(any(feature = "rk35xx", feature = "rk3576"))]
    unsafe fn set_core_mask(
        &self,
        context: rknn_context,
        core_mask: rknn_core_mask,
    ) -> Result<c_int, Error>;

    unsafe fn run(
        &self,
        context: rknn_context,
        extend: *mut rknn_run_extend,
    ) -> Result<c_int, Error>;

    #[cfg_attr(
        feature = "docs",
        doc(cfg(any(feature = "rk35xx", feature = "rk3576")))
    )]
    #[cfg(any(feature = "rk35xx", feature = "rk3576"))]
    unsafe fn outputs_get(
        &self,
        context: rknn_context,
        n_outputs: u32,
        outputs: *mut rknn_output,
        extend: *mut rknn_output_extend,
    ) -> Result<c_int, Error>;

    #[cfg_attr(
        feature = "docs",
        doc(cfg(any(feature = "rk35xx", feature = "rk3576")))
    )]
    #[cfg(any(feature = "rk35xx", feature = "rk3576"))]
    unsafe fn outputs_release(
        &self,
        context: rknn_context,
        n_outputs: u32,
        outputs: *mut rknn_output,
    ) -> Result<c_int, Error>;

    // ───── memory helpers ────────────────────────────────────────────────────
    unsafe fn create_mem_from_phys(
        &self,
        ctx: rknn_context,
        phys_addr: u64,
        virt_addr: *mut c_void,
        size: u32,
    ) -> Result<*mut rknn_tensor_mem, Error>;

    #[cfg_attr(
        feature = "docs",
        doc(cfg(any(feature = "rk35xx", feature = "rk3576", feature = "rv110x")))
    )]
    #[cfg(any(feature = "rk35xx", feature = "rk3576", feature = "rv110x"))]
    unsafe fn create_mem_from_fd(
        &self,
        ctx: rknn_context,
        fd: i32,
        virt_addr: *mut c_void,
        size: u32,
        offset: i32,
    ) -> Result<*mut rknn_tensor_mem, Error>;

    unsafe fn create_mem(
        &self,
        ctx: rknn_context,
        size: u32,
    ) -> Result<*mut rknn_tensor_mem, Error>;

    unsafe fn create_mem2(
        &self,
        ctx: rknn_context,
        size: u64,
        alloc_flags: u64,
    ) -> Result<*mut rknn_tensor_mem, Error>;

    unsafe fn destroy_mem(
        &self,
        ctx: rknn_context,
        mem: *mut rknn_tensor_mem,
    ) -> Result<c_int, Error>;

    unsafe fn set_weight_mem(
        &self,
        ctx: rknn_context,
        mem: *mut rknn_tensor_mem,
    ) -> Result<c_int, Error>;

    unsafe fn set_internal_mem(
        &self,
        ctx: rknn_context,
        mem: *mut rknn_tensor_mem,
    ) -> Result<c_int, Error>;

    unsafe fn set_io_mem(
        &self,
        ctx: rknn_context,
        mem: *mut rknn_tensor_mem,
        attr: *mut rknn_tensor_attr,
    ) -> Result<c_int, Error>;

    unsafe fn set_input_shape(
        &self,
        ctx: rknn_context,
        attr: *mut rknn_tensor_attr,
    ) -> Result<c_int, Error>;

    #[cfg_attr(
        feature = "docs",
        doc(cfg(any(feature = "rk35xx", feature = "rk3576")))
    )]
    #[cfg(any(feature = "rk35xx", feature = "rk3576"))]
    unsafe fn set_input_shapes(
        &self,
        ctx: rknn_context,
        n_inputs: u32,
        attr: *mut rknn_tensor_attr,
    ) -> Result<c_int, Error>;

    unsafe fn mem_sync(
        &self,
        context: rknn_context,
        mem: *mut rknn_tensor_mem,
        mode: rknn_mem_sync_mode,
    ) -> Result<c_int, Error>;

    // ───── MatMul accelerator sub-API ────────────────────────────────────────

    #[cfg_attr(
        feature = "docs",
        doc(cfg(any(feature = "rk35xx", feature = "rk3576")))
    )]
    #[cfg(any(feature = "rk35xx", feature = "rk3576"))]
    unsafe fn matmul_create(
        &self,
        ctx: *mut rknn_matmul_ctx,
        info: *mut rknn_matmul_info,
        io_attr: *mut rknn_matmul_io_attr,
    ) -> Result<c_int, Error>;

    #[cfg_attr(
        feature = "docs",
        doc(cfg(any(feature = "rk35xx", feature = "rk3576")))
    )]
    #[cfg(any(feature = "rk35xx", feature = "rk3576"))]
    unsafe fn matmul_create_dynamic_shape(
        &self,
        ctx: *mut rknn_matmul_ctx,
        info: *mut rknn_matmul_info,
        shape_num: c_int,
        dynamic_shapes: *mut rknn_matmul_shape,
        io_attrs: *mut rknn_matmul_io_attr,
    ) -> Result<c_int, Error>;

    #[cfg_attr(
        feature = "docs",
        doc(cfg(any(feature = "rk35xx", feature = "rk3576")))
    )]
    #[cfg(any(feature = "rk35xx", feature = "rk3576"))]
    unsafe fn matmul_set_io_mem(
        &self,
        ctx: rknn_matmul_ctx,
        mem: *mut rknn_tensor_mem,
        attr: *mut rknn_matmul_tensor_attr,
    ) -> Result<c_int, Error>;

    #[cfg_attr(
        feature = "docs",
        doc(cfg(any(feature = "rk35xx", feature = "rk3576")))
    )]
    #[cfg(any(feature = "rk35xx", feature = "rk3576"))]
    unsafe fn matmul_set_core_mask(
        &self,
        ctx: rknn_matmul_ctx,
        core_mask: rknn_core_mask,
    ) -> Result<c_int, Error>;

    #[cfg_attr(
        feature = "docs",
        doc(cfg(any(feature = "rk35xx", feature = "rk3576")))
    )]
    #[cfg(any(feature = "rk35xx", feature = "rk3576"))]
    unsafe fn matmul_set_quant_params(
        &self,
        ctx: rknn_matmul_ctx,
        params: *mut rknn_quant_params,
    ) -> Result<c_int, Error>;

    #[cfg_attr(
        feature = "docs",
        doc(cfg(any(feature = "rk35xx", feature = "rk3576")))
    )]
    #[cfg(any(feature = "rk35xx", feature = "rk3576"))]
    unsafe fn matmul_get_quant_params(
        &self,
        ctx: rknn_matmul_ctx,
        params: *mut rknn_quant_params,
        scale: *mut f32,
    ) -> Result<c_int, Error>;

    #[cfg_attr(
        feature = "docs",
        doc(cfg(any(feature = "rk35xx", feature = "rk3576")))
    )]
    #[cfg(any(feature = "rk35xx", feature = "rk3576"))]
    unsafe fn matmul_set_dynamic_shape(
        &self,
        ctx: rknn_matmul_ctx,
        shape: *mut rknn_matmul_shape,
    ) -> Result<c_int, Error>;

    #[cfg_attr(
        feature = "docs",
        doc(cfg(any(feature = "rk35xx", feature = "rk3576")))
    )]
    #[cfg(any(feature = "rk35xx", feature = "rk3576"))]
    unsafe fn matmul_run(&self, ctx: rknn_matmul_ctx) -> Result<c_int, Error>;

    #[cfg_attr(
        feature = "docs",
        doc(cfg(any(feature = "rk35xx", feature = "rk3576")))
    )]
    #[cfg(any(feature = "rk35xx", feature = "rk3576"))]
    unsafe fn matmul_destroy(&self, ctx: rknn_matmul_ctx) -> Result<c_int, Error>;

    #[cfg_attr(
        feature = "docs",
        doc(cfg(any(feature = "rk35xx", feature = "rk3576")))
    )]
    #[cfg(any(feature = "rk35xx", feature = "rk3576"))]
    unsafe fn B_normal_layout_to_native_layout(
        &self,
        B_input: *mut c_void,
        B_output: *mut c_void,
        K: c_int,
        N: c_int,
        info: *mut rknn_matmul_info,
    ) -> Result<c_int, Error>;

    // ───── custom operator extension ────────────────────────────────────────

    #[cfg_attr(
        feature = "docs",
        doc(cfg(any(feature = "rk35xx", feature = "rk3576")))
    )]
    #[cfg(any(feature = "rk35xx", feature = "rk3576"))]
    unsafe fn register_custom_ops(
        &self,
        ctx: rknn_context,
        ops: *mut rknn_custom_op,
        custom_op_num: u32,
    ) -> Result<c_int, Error>;

    #[cfg_attr(
        feature = "docs",
        doc(cfg(any(feature = "rk35xx", feature = "rk3576")))
    )]
    #[cfg(any(feature = "rk35xx", feature = "rk3576"))]
    unsafe fn custom_op_get_op_attr(
        &self,
        op_ctx: *mut rknn_custom_op_context,
        attr_name: *const c_char,
        op_attr: *mut rknn_custom_op_attr,
    ) -> Result<(), Error>;
}

use bitflags::bitflags;

bitflags! {
    /// Flags passed to `rknn_init` controlling execution behavior.
    #[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
    pub struct RknnInitFlags: u32 {
        const ASYNC_MASK                     = RKNN_FLAG_ASYNC_MASK;
        const COLLECT_MODEL_INFO_ONLY        = RKNN_FLAG_COLLECT_MODEL_INFO_ONLY;
        const COLLECT_PERF_MASK              = RKNN_FLAG_COLLECT_PERF_MASK;
        const DISABLE_FLUSH_INPUT_MEM_CACHE  = RKNN_FLAG_DISABLE_FLUSH_INPUT_MEM_CACHE;
        const DISABLE_FLUSH_OUTPUT_MEM_CACHE = RKNN_FLAG_DISABLE_FLUSH_OUTPUT_MEM_CACHE;
        const DISABLE_PROC_HIGH_PRIORITY     = RKNN_FLAG_DISABLE_PROC_HIGH_PRIORITY;
        const ENABLE_SRAM                    = RKNN_FLAG_ENABLE_SRAM;
        const EXECUTE_FALLBACK_PRIOR_DEVICE_GPU = RKNN_FLAG_EXECUTE_FALLBACK_PRIOR_DEVICE_GPU;
        const FENCE_IN_OUTSIDE               = RKNN_FLAG_FENCE_IN_OUTSIDE;
        const FENCE_OUT_OUTSIDE              = RKNN_FLAG_FENCE_OUT_OUTSIDE;
        const INTERNAL_ALLOC_OUTSIDE         = RKNN_FLAG_INTERNAL_ALLOC_OUTSIDE;
        const MEM_ALLOC_OUTSIDE              = RKNN_FLAG_MEM_ALLOC_OUTSIDE;
        const MODEL_BUFFER_ZERO_COPY         = RKNN_FLAG_MODEL_BUFFER_ZERO_COPY;
        const PRIOR_HIGH                     = RKNN_FLAG_PRIOR_HIGH;
        const PRIOR_MEDIUM                   = RKNN_FLAG_PRIOR_MEDIUM;
        const PRIOR_LOW                      = RKNN_FLAG_PRIOR_LOW;
        const SHARE_SRAM                     = RKNN_FLAG_SHARE_SRAM;
        const SHARE_WEIGHT_MEM               = RKNN_FLAG_SHARE_WEIGHT_MEM;
    }
}

impl From<RknnInitFlags> for u32 {
    fn from(flags: RknnInitFlags) -> Self {
        flags.bits()
    }
}

impl From<u32> for RknnInitFlags {
    fn from(bits: u32) -> Self {
        Self::from_bits_truncate(bits)
    }
}

impl RknnInitFlags {
    /// Start from "no flags".
    pub const fn builder() -> Self {
        Self::empty()
    }

    pub const fn with_async(self) -> Self {
        self.union(Self::ASYNC_MASK)
    }

    pub const fn with_model_info_only(self) -> Self {
        self.union(Self::COLLECT_MODEL_INFO_ONLY)
    }

    pub const fn with_perf_collection(self) -> Self {
        self.union(Self::COLLECT_PERF_MASK)
    }

    pub const fn with_zero_copy_model_buffer(self) -> Self {
        self.union(Self::MODEL_BUFFER_ZERO_COPY)
    }

    pub const fn with_share_sram(self) -> Self {
        self.union(Self::SHARE_SRAM)
    }

    pub const fn with_share_weight_mem(self) -> Self {
        self.union(Self::SHARE_WEIGHT_MEM)
    }

    pub const fn with_external_mem_alloc(self) -> Self {
        self.union(Self::MEM_ALLOC_OUTSIDE)
    }

    pub const fn with_external_internal_alloc(self) -> Self {
        self.union(Self::INTERNAL_ALLOC_OUTSIDE)
    }

    pub const fn with_no_input_cache_flush(self) -> Self {
        self.union(Self::DISABLE_FLUSH_INPUT_MEM_CACHE)
    }

    pub const fn with_no_output_cache_flush(self) -> Self {
        self.union(Self::DISABLE_FLUSH_OUTPUT_MEM_CACHE)
    }

    pub const fn with_no_proc_high_priority(self) -> Self {
        self.union(Self::DISABLE_PROC_HIGH_PRIORITY)
    }

    pub const fn with_enable_sram(self) -> Self {
        self.union(Self::ENABLE_SRAM)
    }

    pub const fn with_gpu_fallback(self) -> Self {
        self.union(Self::EXECUTE_FALLBACK_PRIOR_DEVICE_GPU)
    }

    pub const fn with_fence_in_outside(self) -> Self {
        self.union(Self::FENCE_IN_OUTSIDE)
    }

    pub const fn with_fence_out_outside(self) -> Self {
        self.union(Self::FENCE_OUT_OUTSIDE)
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Priority {
    Low,
    Medium,
    High,
}

impl Priority {
    pub const fn as_flags(self) -> RknnInitFlags {
        match self {
            Priority::Low => RknnInitFlags::PRIOR_LOW,
            Priority::Medium => RknnInitFlags::PRIOR_MEDIUM,
            Priority::High => RknnInitFlags::PRIOR_HIGH,
        }
    }
}

impl RknnInitFlags {
    /// Clear existing priority bits and set a new one.
    pub fn with_priority(self, priority: Priority) -> Self {
        let cleared = self
            & !(RknnInitFlags::PRIOR_LOW | RknnInitFlags::PRIOR_MEDIUM | RknnInitFlags::PRIOR_HIGH);

        cleared | priority.as_flags()
    }
}
