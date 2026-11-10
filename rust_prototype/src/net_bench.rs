use std::time::Instant;
use tokio::net::{TcpListener, TcpStream};
use tokio::io::{AsyncReadExt, AsyncWriteExt};

#[repr(C, packed)]
#[derive(Clone, Copy, Debug)]
pub struct WireHeader {
    pub magic: u32,         // 0x4D455348 ('MESH')
    pub request_id: u64,
    pub token_idx: u32,
    pub seq_pos: u32,
    pub layer_idx: u16,
    pub dtype: u8,          // 0 = FP32, 1 = FP16, 2 = INT8
    pub reserved: u8,
    pub num_elements: u32,
    pub scale: f32,         // for INT8 quant
}

const WIRE_HEADER_LEN: usize = std::mem::size_of::<WireHeader>();

async fn run_server(addr: &str, ready_tx: tokio::sync::oneshot::Sender<()>) {
    let listener = TcpListener::bind(addr).await.unwrap();
    ready_tx.send(()).unwrap();

    let (mut socket, _) = listener.accept().await.unwrap();
    socket.set_nodelay(true).unwrap();

    let mut header_buf = [0u8; WIRE_HEADER_LEN];
    let mut payload_buf = vec![0u8; 64 * 1024];

    loop {
        // Read header
        if socket.read_exact(&mut header_buf).await.is_err() {
            break;
        }
        let header: WireHeader = unsafe { std::ptr::read_unaligned(header_buf.as_ptr() as *const _) };
        if header.magic != 0x4D455348 {
            panic!("Bad magic!");
        }

        let payload_len = match header.dtype {
            0 => header.num_elements as usize * 4,
            1 => header.num_elements as usize * 2,
            2 => header.num_elements as usize * 1,
            _ => panic!("Unknown dtype"),
        };

        // Read payload directly into buffer (zero-copy view)
        socket.read_exact(&mut payload_buf[..payload_len]).await.unwrap();

        // Echo back a 16-byte ACK
        socket.write_all(b"OK_STEP_FINISHED").await.unwrap();
    }
}

#[tokio::main]
async fn main() {
    println!("=== 3.1 & 3.2 Native Rust Zero-Copy TCP Transport Benchmark ===");
    let addr = "127.0.0.1:19999";
    let (ready_tx, ready_rx) = tokio::sync::oneshot::channel();

    tokio::spawn(run_server(addr, ready_tx));
    ready_rx.await.unwrap();

    let mut stream = TcpStream::connect(addr).await.unwrap();
    stream.set_nodelay(true).unwrap();

    let dims = [288, 2048, 4096, 5120];

    for &dim in &dims {
        // Benchmark A: Raw FP32
        let num_elements = dim as u32;
        let tensor_data = vec![1.234f32; dim];
        let raw_bytes: &[u8] = unsafe {
            std::slice::from_raw_parts(tensor_data.as_ptr() as *const u8, dim * 4)
        };

        let header = WireHeader {
            magic: 0x4D455348,
            request_id: 42,
            token_idx: 1,
            seq_pos: 15,
            layer_idx: 16,
            dtype: 0, // FP32
            reserved: 0,
            num_elements,
            scale: 1.0,
        };
        let header_bytes: &[u8] = unsafe {
            std::slice::from_raw_parts(&header as *const _ as *const u8, WIRE_HEADER_LEN)
        };

        let iters = 2000;
        let mut ack_buf = [0u8; 16];

        let start = Instant::now();
        for _ in 0..iters {
            stream.write_all(header_bytes).await.unwrap();
            stream.write_all(raw_bytes).await.unwrap();
            stream.read_exact(&mut ack_buf).await.unwrap();
        }
        let elapsed_fp32 = start.elapsed().as_micros() as f64 / iters as f64;

        // Benchmark B: INT8 Quantized
        let int8_data = vec![42i8; dim];
        let int8_bytes: &[u8] = unsafe {
            std::slice::from_raw_parts(int8_data.as_ptr() as *const u8, dim)
        };
        let mut q_header = header;
        q_header.dtype = 2; // INT8
        let q_header_bytes: &[u8] = unsafe {
            std::slice::from_raw_parts(&q_header as *const _ as *const u8, WIRE_HEADER_LEN)
        };

        let start_q = Instant::now();
        for _ in 0..iters {
            stream.write_all(q_header_bytes).await.unwrap();
            stream.write_all(int8_bytes).await.unwrap();
            stream.read_exact(&mut ack_buf).await.unwrap();
        }
        let elapsed_int8 = start_q.elapsed().as_micros() as f64 / iters as f64;

        println!("Dim {:5} | FP32 Wire: {:5} B (RTT: {:5.2} µs) | INT8 Wire: {:5} B (RTT: {:5.2} µs | {:.2}x speedup)",
                 dim, WIRE_HEADER_LEN + dim * 4, elapsed_fp32, WIRE_HEADER_LEN + dim, elapsed_int8, elapsed_fp32 / elapsed_int8);
    }
}
