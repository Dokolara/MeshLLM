//! AETHER-LINK: Zero-Copy Neural Fabric Transport Subsystem
//! Implements a 32-byte hardware-aligned binary wire header and streaming protocol.

#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct WireHeader {
    pub magic: u32,         // 0x4D455348 ('MESH')
    pub token_idx: u32,
    pub request_id: u64,
    pub seq_pos: u32,
    pub num_elements: u32,
    pub scale: f32,         // Dynamic quantization scale
    pub layer_idx: u16,
    pub dtype: u8,          // 0 = FP32, 1 = FP16, 2 = INT8
    pub is_finished: u8,    // 1 if generation complete
}

pub const WIRE_HEADER_LEN: usize = std::mem::size_of::<WireHeader>();
pub const MESH_MAGIC: u32 = 0x4D455348;

impl WireHeader {
    pub fn new_fp32(request_id: u64, token_idx: u32, seq_pos: u32, layer_idx: u16, num_elements: u32) -> Self {
        Self {
            magic: MESH_MAGIC,
            token_idx,
            request_id,
            seq_pos,
            num_elements,
            scale: 1.0,
            layer_idx,
            dtype: 0,
            is_finished: 0,
        }
    }

    pub fn new_int8(request_id: u64, token_idx: u32, seq_pos: u32, layer_idx: u16, num_elements: u32, scale: f32) -> Self {
        Self {
            magic: MESH_MAGIC,
            token_idx,
            request_id,
            seq_pos,
            num_elements,
            scale,
            layer_idx,
            dtype: 2,
            is_finished: 0,
        }
    }

    pub fn to_bytes(&self) -> [u8; WIRE_HEADER_LEN] {
        unsafe { std::mem::transmute_copy(self) }
    }

    pub fn from_bytes(bytes: &[u8]) -> Result<Self, String> {
        if bytes.len() < WIRE_HEADER_LEN {
            return Err("Buffer too short for WireHeader".to_string());
        }
        let header: WireHeader = unsafe { std::ptr::read_unaligned(bytes.as_ptr() as *const _) };
        if header.magic != MESH_MAGIC {
            return Err(format!("Bad magic: 0x{:08X}", header.magic));
        }
        Ok(header)
    }

    pub fn payload_byte_size(&self) -> usize {
        match self.dtype {
            0 => self.num_elements as usize * 4,
            1 => self.num_elements as usize * 2,
            2 => self.num_elements as usize * 1,
            _ => 0,
        }
    }
}
