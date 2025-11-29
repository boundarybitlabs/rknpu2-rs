use rknpu2_sys::_rknn_tensor_memory;

pub struct TensorMem {
    pub mem: *mut _rknn_tensor_memory,
}
