//! 真实多进程集合通信：TCP 上的环形算法（gloo 模式）。
//!
//! [`crate::distributed::World`] 用单进程模拟证明了算法，这里把"memcpy 换 send/recv"
//! 落地成真传输：每个 rank 一个进程，每对 rank 之间建一条 TCP 连接（全连接拓扑），
//! 环形 allreduce / reduce-scatter / all-gather / broadcast 每轮只跟真实邻居交换数据。
//! 在没有 NCCL 的环境里，PyTorch 的 gloo CPU 后端就是这么干的。
//!
//! 两个设计要点：
//!
//! - **每个连接配一条独立读线程**：环形算法里"先发后收"是常态，若收发都走同一条
//!   阻塞 socket，环上很容易互相等待到死。读线程永远在把收到的帧塞进内存队列，
//!   发送方写满 TCP 缓冲也只意味着对端读线程在消化，不会卡住环的推进。
//! - **消息按 (来源 rank, 通信组 tag) 匹配**：多维并行下同一对 rank 可能同时属于
//!   TP 组、DP 组，帧头带组 tag 避免串包。SPMD 保证各组内集合调用顺序一致，
//!   TCP 又保证单连接内有序，所以按 tag 取队首即可，不需要全局序号。

use std::collections::{HashMap, VecDeque};
use std::io::{Read, Write};
use std::net::{TcpListener, TcpStream};
use std::sync::mpsc::{self, Receiver, RecvTimeoutError};
use std::sync::{Mutex, MutexGuard};
use std::thread;
use std::time::{Duration, Instant};

use crate::distributed::{chunk_range, rotate_chunks, wrap, DistConfig};

/// 通信组 tag（帧头的组号）：三个维度各占一个，同一对 rank 上的
/// TP / DP / PP 流量互不串包。值本身无大小含义，只要互不相同。
pub const GROUP_TP: u32 = 1;
pub const GROUP_DP: u32 = 2;
pub const GROUP_PP: u32 = 3;

/// 单帧头长度：组 tag（4 字节）+ 载荷字节数（4 字节），大端。
const HEADER_LEN: usize = 8;

/// 默认接收超时（秒）。rank0 单独做 eval / 存 checkpoint 时会短暂领先，
/// 超时要放得足够宽，只用于把"真的断了"和"慢"区分开。可用 `DIST_RECV_TIMEOUT_SECS` 覆盖。
const DEFAULT_RECV_TIMEOUT_SECS: u64 = 600;

/// f32 → 小端字节。逐元素 `to_le_bytes` 是安全代码，比 cast 字节数组慢一点，
/// 但梯度传输本来就不是瓶颈（TCP loopback 上瓶颈在 RTT 而非拷贝）。
fn encode(v: &[f32]) -> Vec<u8> {
    let mut out = Vec::with_capacity(v.len() * 4);
    for x in v {
        out.extend_from_slice(&x.to_le_bytes());
    }
    out
}

/// 小端字节 → f32。`chunks_exact(4)` 保证不会读到半截元素。
fn decode(b: &[u8]) -> Vec<f32> {
    b.chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}

/// 从某个连接上读出的一帧。
struct Frame {
    group: u32,
    payload: Vec<u8>,
}

/// 每条连接的读线程：循环读「头 + 载荷」，塞进 mpsc 通道。
///
/// 对端关闭或读到一半断开时线程静默退出——后续等待方会收到 `Disconnected`
/// 并带着明确的 rank 信息 panic，而不是永远挂住。
fn spawn_reader(stream: TcpStream) -> Receiver<Frame> {
    let (tx, rx) = mpsc::channel();
    thread::spawn(move || {
        let mut s = stream;
        let mut head = [0u8; HEADER_LEN];
        loop {
            if s.read_exact(&mut head).is_err() {
                break;
            }
            let group = u32::from_be_bytes([head[0], head[1], head[2], head[3]]);
            let len = u32::from_be_bytes([head[4], head[5], head[6], head[7]]) as usize;
            let mut payload = vec![0u8; len];
            if s.read_exact(&mut payload).is_err() {
                break;
            }
            if tx.send(Frame { group, payload }).is_err() {
                break;
            }
        }
    });
    rx
}

/// 本进程在通信拓扑里的节点：持有到其余每个 rank 的连接。
///
/// - `tx[peer]`：发往 peer 的写端（互斥锁保护——多个通信组可能共用同一条连接，
///   两条大消息交错写入会让帧格式彻底损坏）。
/// - `rx[peer]`：读线程的接收端。
/// - `queued[peer]`：已读出但还没被消费的帧，按组 tag 分队列暂存。
pub struct Comm {
    rank: usize,
    world: usize,
    tx: Vec<Option<Mutex<TcpStream>>>,
    rx: Vec<Option<Receiver<Frame>>>,
    queued: Vec<Mutex<HashMap<u32, VecDeque<Vec<f32>>>>>,
    recv_timeout: Duration,
}

impl Comm {
    /// 建立全连接拓扑。
    ///
    /// 端口约定：rank `r` 监听 `base_port + r`，所以连接目标就是 `base_port + peer`，
    /// 不需要额外的地址服务。`addr` 是所有 rank 的可达地址（单机 loopback 传
    /// `127.0.0.1`；多机传调度器分到的同一网段地址）。
    ///
    /// 连接方向由 rank 大小决定（小 rank 拨大 rank），保证每对 rank 恰好一条连接、
    /// 不会出现"互相等对方先拨"的僵局；拨号带重试，先启动的进程会等监听就绪。
    pub fn new(rank: usize, world: usize, addr: &str, base_port: u16) -> Self {
        assert!(rank < world, "rank {rank} 超出 world {world}");
        assert!(
            base_port as usize + world < 65535,
            "base_port {base_port} 加上 world {world} 会溢出端口号"
        );
        let recv_timeout = std::env::var("DIST_RECV_TIMEOUT_SECS")
            .ok()
            .and_then(|s| s.parse().ok())
            .map(Duration::from_secs)
            .unwrap_or(Duration::from_secs(DEFAULT_RECV_TIMEOUT_SECS));
        let mut comm = Comm {
            rank,
            world,
            tx: (0..world).map(|_| None).collect(),
            rx: (0..world).map(|_| None).collect(),
            queued: (0..world)
                .map(|_| Mutex::new(HashMap::new()))
                .collect(),
            recv_timeout,
        };
        if world > 1 {
            comm.connect_all(addr, base_port);
        }
        comm
    }

    /// 从环境变量读取进程坐标（torch 约定）：`RANK`、`WORLD_SIZE`、
    /// `MASTER_ADDR`、`MASTER_PORT`。缺省时退化成 world=1 的单进程空实现，
    /// 所以不设环境变量的普通训练完全不受影响。
    pub fn from_env() -> Self {
        let rank: usize = std::env::var("RANK").ok().and_then(|s| s.parse().ok()).unwrap_or(0);
        let world: usize = std::env::var("WORLD_SIZE")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(1);
        let addr = std::env::var("MASTER_ADDR").unwrap_or_else(|_| "127.0.0.1".to_string());
        let port: u16 = std::env::var("MASTER_PORT")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(29611);
        Self::new(rank, world, &addr, port)
    }

    fn connect_all(&mut self, addr: &str, base_port: u16) {
        // 监听本 rank 的端口，收下所有"比我小"的 rank 的拨号
        let listener = TcpListener::bind(("0.0.0.0", base_port + self.rank as u16))
            .unwrap_or_else(|e| panic!("rank {} 监听端口 {} 失败：{e}", self.rank, base_port + self.rank as u16));

        // 拨号所有"比我大"的 rank（小拨大，方向唯一）
        for peer in self.rank + 1..self.world {
            let stream = dial(addr, base_port + peer as u16, self.rank, peer);
            self.install(peer, stream);
        }
        // 接受所有比我小的 rank 的拨号；帧头前先握手报上自己的 rank
        for _ in 0..self.rank {
            let (stream, _from) = listener
                .accept()
                .unwrap_or_else(|e| panic!("rank {} 接受连接失败：{e}", self.rank));
            let mut stream = stream;
            let mut id_buf = [0u8; 4];
            stream
                .read_exact(&mut id_buf)
                .unwrap_or_else(|e| panic!("rank {} 握手读取失败：{e}", self.rank));
            let peer = u32::from_be_bytes(id_buf) as usize;
            assert!(peer < self.world && peer != self.rank, "握手收到非法 rank {peer}");
            self.install(peer, stream);
        }
    }

    /// 把一条已握手的连接装进 tx/rx 槽位并启动读线程。
    fn install(&mut self, peer: usize, stream: TcpStream) {
        stream.set_nodelay(true).ok();
        let read_end = stream.try_clone().expect("克隆连接失败");
        self.tx[peer] = Some(Mutex::new(stream));
        self.rx[peer] = Some(spawn_reader(read_end));
    }

    /// 发一帧给 `dst`。帧 = 头（组 tag + 字节数）+ 小端 f32 载荷，一次写入。
    fn send_frame(&self, dst: usize, group: u32, data: &[f32]) {
        assert!(dst < self.world && dst != self.rank, "非法发送目标 {dst}");
        let payload = encode(data);
        assert!(payload.len() <= u32::MAX as usize, "单帧超过 4GiB");
        let mut buf = Vec::with_capacity(HEADER_LEN + payload.len());
        buf.extend_from_slice(&group.to_be_bytes());
        buf.extend_from_slice(&(payload.len() as u32).to_be_bytes());
        buf.extend_from_slice(&payload);
        let guard: MutexGuard<TcpStream> = self.tx[dst]
            .as_ref()
            .unwrap_or_else(|| panic!("rank {} 没有到 rank {dst} 的连接", self.rank))
            .lock()
            .unwrap();
        (&*guard)
            .write_all(&buf)
            .unwrap_or_else(|e| panic!("rank {} 发往 rank {dst} 失败：{e}", self.rank));
    }

    /// 从 `src` 收一帧，且必须属于通信组 `group`。
    ///
    /// 本地队列里已有就直接取；否则从读线程的通道里等。等来的帧若属于别的组
    /// （别的通信组抢先到了），先归档进它自己的队列再继续等——SPMD 下每个组
    /// 最终都会轮到，不会丢帧。
    fn recv_frame(&self, src: usize, group: u32) -> Vec<f32> {
        loop {
            {
                let mut q = self.queued[src].lock().unwrap();
                if let Some(deque) = q.get_mut(&group) {
                    if let Some(v) = deque.pop_front() {
                        return v;
                    }
                }
            }
            let rx = self.rx[src]
                .as_ref()
                .unwrap_or_else(|| panic!("rank {} 没有到 rank {src} 的连接", self.rank));
            match rx.recv_timeout(self.recv_timeout) {
                Ok(f) => {
                    if f.group == group {
                        return decode(&f.payload);
                    }
                    self.queued[src]
                        .lock()
                        .unwrap()
                        .entry(f.group)
                        .or_default()
                        .push_back(decode(&f.payload));
                }
                Err(RecvTimeoutError::Timeout) => panic!(
                    "rank {} 等待 rank {src}（组 {group}）超时（{} 秒）——对端可能已退出",
                    self.rank,
                    self.recv_timeout.as_secs()
                ),
                Err(RecvTimeoutError::Disconnected) => panic!(
                    "rank {} 与 rank {src} 的连接断开（组 {group}）",
                    self.rank
                ),
            }
        }
    }

    pub fn rank(&self) -> usize {
        self.rank
    }

    pub fn world(&self) -> usize {
        self.world
    }

    /// 取一个通信组句柄。`members` 按环序排列（约定升序），环上前后邻居
    /// 由这份列表决定——3D 并行里 TP/DP/PP 组就是用不同的 `members` 圈出来的。
    pub fn group(&self, id: u32, members: Vec<usize>) -> CommGroup<'_> {
        for &m in &members {
            assert!(m < self.world, "通信组成员 {m} 超出 world {}", self.world);
        }
        let n = members.len();
        assert!(n > 0, "通信组不能为空");
        let mut sorted = members;
        sorted.sort_unstable();
        sorted.dedup();
        assert_eq!(sorted.len(), n, "通信组成员重复");
        assert!(sorted.contains(&self.rank), "本 rank {} 不在通信组内", self.rank);
        CommGroup { comm: self, id, members: sorted }
    }
}

/// 拨号：带重试与总时限。先启动的进程会等对方监听就绪，超时则明确报错。
fn dial(addr: &str, port: u16, self_rank: usize, peer: usize) -> TcpStream {
    let deadline = Instant::now() + Duration::from_secs(120);
    loop {
        match TcpStream::connect((addr, port)) {
            Ok(s) => {
                let mut s = s;
                s.write_all(&(self_rank as u32).to_be_bytes())
                    .unwrap_or_else(|e| panic!("rank {self_rank} 握手发送失败：{e}"));
                return s;
            }
            Err(e) => {
                assert!(
                    Instant::now() < deadline,
                    "rank {self_rank} 连接 rank {peer}（{addr}:{port}）超时：{e}"
                );
                thread::sleep(Duration::from_millis(50));
            }
        }
    }
}

/// 一个通信组上的操作面。轻量句柄（组 id + 成员环），可随时重建。
///
/// 所有集合算法与 [`crate::distributed::World`] **逐位同构**：同一批输入
/// 模拟器和 TCP 版给出完全相同的浮点结果（加法顺序一致），测试里直接对拍。
pub struct CommGroup<'c> {
    comm: &'c Comm,
    id: u32,
    members: Vec<usize>,
}

impl CommGroup<'_> {
    pub fn size(&self) -> usize {
        self.members.len()
    }

    /// 本 rank 在环上的位置（members 升序后的下标）
    fn pos(&self) -> usize {
        self.members
            .iter()
            .position(|&r| r == self.comm.rank)
            .expect("本 rank 不在通信组内")
    }

    /// 环形一步：把 `payload` 发给右邻居，从左邻居收一帧回来。
    ///
    /// 先发后收是安全的——对端有独立读线程，阻塞的写只会被对端读线程消化，
    /// 不会出现环上互相等待。
    fn exchange(&self, payload: &[f32]) -> Vec<f32> {
        let n = self.members.len();
        let pos = self.pos();
        let right = self.members[(pos + 1) % n];
        let left = self.members[(pos + n - 1) % n];
        self.comm.send_frame(right, self.id, payload);
        self.comm.recv_frame(left, self.id)
    }

    /// 求和 allreduce，返回完整结果，组内各 rank 逐位相同。
    ///
    /// 与 `World::all_reduce_sum` 同构的两阶段环形：
    /// reduce-scatter（n-1 轮）→ all-gather（n-1 轮）→ [`rotate_chunks`] 块序换算。
    /// 每 rank 总流量 ≈ `2(n-1)/n · len`，与组大小几乎无关。
    pub fn all_reduce_sum(&self, local: &[f32]) -> Vec<f32> {
        let n = self.members.len();
        if n == 1 {
            return local.to_vec();
        }
        let (_chunk, owned) = self.reduce_scatter_inner(local);
        let parts = self.all_gather_inner(&owned);
        let gathered: Vec<f32> = parts.into_iter().flatten().collect();
        rotate_chunks(&gathered, n)
    }

    /// 环形 allreduce 第一阶段：求和结果切开留在各 rank。
    ///
    /// 返回 `(块下标, 该块的全和)`，rank 拿到第 `(pos+1) mod n` 块——
    /// 与 `World::reduce_scatter_sum` 的块归属约定完全一致（ZeRO 分片依赖它）。
    pub fn reduce_scatter_sum(&self, local: &[f32]) -> (usize, Vec<f32>) {
        if self.members.len() == 1 {
            return (0, local.to_vec());
        }
        self.reduce_scatter_inner(local)
    }

    fn reduce_scatter_inner(&self, local: &[f32]) -> (usize, Vec<f32>) {
        let n = self.members.len();
        let pos = self.pos();
        let len = local.len();
        let mut buf = local.to_vec();
        // rank pos 第 s 轮：发第 (pos-s) 块，收第 (pos-s-1) 块并累加
        for s in 0..n - 1 {
            let s = s as isize;
            let (ss, se) = chunk_range(wrap(pos as isize - s, n), len, n);
            let got = self.exchange(&buf[ss..se]);
            let (rs, re) = chunk_range(wrap(pos as isize - s - 1, n), len, n);
            assert_eq!(
                re - rs,
                got.len(),
                "环形通信两端块大小不一致：rank {} 收 {} 个，期望 {} 个",
                self.comm.rank,
                got.len(),
                re - rs
            );
            for (d, v) in buf[rs..re].iter_mut().zip(&got) {
                *d += v;
            }
        }
        // n-1 轮后，本 rank 完整的和是第 (pos+1) mod n 块
        let c = (pos + 1) % n;
        let (s, e) = chunk_range(c, len, n);
        (c, buf[s..e].to_vec())
    }

    /// 收集组内每个 rank 的贡献，返回 `按成员升序` 的分块列表。
    ///
    /// 与 `World::all_gather` 同构的环形（n-1 轮），支持不等长分块（ZeRO 需要）。
    /// 注意入参与模拟器不同：真实进程手里只有**自己**那份。
    pub fn all_gather(&self, local: &[f32]) -> Vec<Vec<f32>> {
        if self.members.len() == 1 {
            return vec![local.to_vec()];
        }
        self.all_gather_inner(local)
    }

    fn all_gather_inner(&self, local: &[f32]) -> Vec<Vec<f32>> {
        let n = self.members.len();
        let pos = self.pos();
        // blocks[i] = 手上"来自成员 i"的那块；起初只有自己
        let mut blocks: Vec<Option<Vec<f32>>> = vec![None; n];
        blocks[pos] = Some(local.to_vec());
        for s in 0..n - 1 {
            let s = s as isize;
            // 快照语义：本轮转发刚到手的 (pos-s) 块
            let send_i = wrap(pos as isize - s, n);
            let payload = blocks[send_i].clone().expect("本轮待转发的块缺失");
            let recv_i = wrap(pos as isize - s - 1, n);
            let got = self.exchange(&payload);
            blocks[recv_i] = Some(got);
        }
        blocks
            .into_iter()
            .map(|b| b.expect("all_gather 结束后每块都应就位"))
            .collect()
    }

    /// 从组内 `src`（全局 rank）广播 `local`，返回广播值。
    /// 只有源 rank 的数据被采用，其余 rank 传什么无所谓。
    pub fn broadcast(&self, src: usize, local: &[f32]) -> Vec<f32> {
        let n = self.members.len();
        if n == 1 {
            return local.to_vec();
        }
        let pos = self.pos();
        let src_pos = self
            .members
            .iter()
            .position(|&r| r == src)
            .unwrap_or_else(|| panic!("广播源 rank {src} 不在通信组内"));
        if pos == src_pos {
            // 源只发一次：沿环 src → src+1 → … 一圈
            let right = self.members[(pos + 1) % n];
            self.comm.send_frame(right, self.id, local);
            return local.to_vec();
        }
        let left = self.members[(pos + n - 1) % n];
        let v = self.comm.recv_frame(left, self.id);
        // 环上源的前一个位置是链的末端，不再往下转发
        let last = (src_pos + n - 1) % n;
        if pos != last {
            let right = self.members[(pos + 1) % n];
            self.comm.send_frame(right, self.id, &v);
        }
        v
    }

    /// 同步屏障：组内每人贡献一个元素并 allgather，人人都拿齐才算过。
    /// 集合算法本身已是同步的，这里用于给阶段边界（如"梯度同步完才 step"）打标。
    pub fn barrier(&self) {
        let _ = self.all_gather(&[0.0]);
    }

    /// 点对点发送（流水线并行的 stage 间传激活/梯度用）。
    /// 组 tag 隔离不同用途的传输，与集合通信互不串包。
    pub fn send(&self, dst: usize, data: &[f32]) {
        assert!(dst != self.comm.rank, "不能发给自己");
        self.comm.send_frame(dst, self.id, data);
    }

    /// 点对点接收，与 [`Self::send`] 按 (来源, 组 tag) 配对。
    pub fn recv(&self, src: usize) -> Vec<f32> {
        assert!(src != self.comm.rank, "不能收自己");
        self.comm.recv_frame(src, self.id)
    }
}

/// 一个训练进程的分布式运行时：把 env 坐标、三维切分配置、传输层绑在一起。
///
/// 进程坐标来自 torch 约定的环境变量（启动器负责设置，见 `DistRuntime::from_env`）：
///
/// | 变量 | 含义 |
/// |------|------|
/// | `RANK` | 全局 rank（0..world） |
/// | `WORLD_SIZE` | 总进程数，必须等于 `dp · tp · pp` |
/// | `LOCAL_RANK` | 本机内的序号（选 GPU adapter 用） |
/// | `MASTER_ADDR` / `MASTER_PORT` | 建连地址与基准端口 |
///
/// 三个组句柄（[`CommGroup`]）按 [`DistConfig`] 的 rank 编号约定圈定成员，
/// 组 tag 固定为 [`GROUP_TP`] / [`GROUP_DP`] / [`GROUP_PP`]：
/// - TP 组内做每层两次的 allreduce（通信最密）；
/// - DP 组内做每步一次的梯度 allreduce；
/// - PP 组只用点对点 `send`/`recv` 在相邻 stage 间传激活与梯度。
pub struct DistRuntime {
    pub comm: Comm,
    pub cfg: DistConfig,
    /// 本机内的卡号（`LOCAL_RANK`）：多进程同机时选 GPU adapter 用
    pub local_rank: usize,
}

impl DistRuntime {
    /// 从环境变量构造。`cfg` 的 `dp·tp·pp` 必须与 `WORLD_SIZE` 一致——
    /// 不一致说明启动器传参和各进程的 CLI 参数没对上，宁可起跑前 panic
    /// 也不要让 rank 各自跑到一半才发现组对不上。
    pub fn from_env(cfg: DistConfig) -> Self {
        let world: usize = std::env::var("WORLD_SIZE")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(1);
        assert_eq!(
            world,
            cfg.world_size(),
            "WORLD_SIZE={world} 与 dp·tp·pp={} 不一致",
            cfg.world_size()
        );
        let comm = Comm::from_env();
        assert_eq!(comm.world(), world, "Comm 的 world 与配置不一致");
        let local_rank: usize = std::env::var("LOCAL_RANK")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(0);
        DistRuntime { comm, cfg, local_rank }
    }

    /// 单进程退化（无环境变量时 world=1）：普通训练路径零改动地复用同一套代码。
    pub fn single(cfg: DistConfig) -> Self {
        assert_eq!(cfg.world_size(), 1, "single() 只接受 world=1 的配置");
        DistRuntime { comm: Comm::new(0, 1, "127.0.0.1", 0), cfg, local_rank: 0 }
    }

    /// 多进程入口：`WORLD_SIZE > 1` 时按 [`DistConfig::from_env`] 建运行时，否则返回
    /// `None` 让调用方走单进程路径。
    ///
    /// 这里刻意用 `None` 而不是 `single(...)` 来代表"单进程"：单进程时训练循环里
    /// **一次集合通信都不该发生**（连 `dp_group()` 的组句柄都不建），
    /// 这样"接分布式"这件事对普通训练的影响在代码层面就是零。
    pub fn from_env_if_distributed() -> Option<Self> {
        let world: usize = std::env::var("WORLD_SIZE")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(1);
        if world <= 1 {
            return None;
        }
        Some(Self::from_env(DistConfig::from_env()))
    }

    pub fn is_primary(&self) -> bool {
        self.comm.rank() == 0
    }

    /// TP 组：层内权重通信（组 tag [`GROUP_TP`]）
    pub fn tp_group(&self) -> CommGroup<'_> {
        self.comm.group(GROUP_TP, self.cfg.tp_group(self.comm.rank()))
    }

    /// DP 组：梯度同步（组 tag [`GROUP_DP`]）
    pub fn dp_group(&self) -> CommGroup<'_> {
        self.comm.group(GROUP_DP, self.cfg.dp_group(self.comm.rank()))
    }

    /// PP 组：stage 间点对点传激活（组 tag [`GROUP_PP`]）。
    /// 组句柄用于 `send`/`recv`；成员是同一条流水线上的全部 stage。
    pub fn pp_group(&self) -> CommGroup<'_> {
        self.comm.group(GROUP_PP, self.cfg.pp_group(self.comm.rank()))
    }

    /// 上游 stage（`None` = 第一段）
    pub fn prev_stage(&self) -> Option<usize> {
        self.cfg.prev_stage(self.comm.rank())
    }

    /// 下游 stage（`None` = 最后一段）
    pub fn next_stage(&self) -> Option<usize> {
        self.cfg.next_stage(self.comm.rank())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::distributed::{World, allreduce_mean_grads, allreduce_mean_moe_bias, sync_overflow};

    /// 拼一个测试用输入集：`n` 个 rank，长度 `len`
    fn inputs(n: usize, len: usize) -> Vec<Vec<f32>> {
        (0..n)
            .map(|r| (0..len).map(|i| (r * 1000 + i) as f32 * 0.25 - 7.0).collect())
            .collect()
    }

    /// 对拍：3 个线程各起一个真 Comm 跑集合通信，结果与单进程 World 逐位比对。
    /// 选长度 10（不能被 3 整除）同时压测不均匀分块。
    #[test]
    fn test_tcp_collectives_match_simulator_world() {
        let (n, len, base) = (3usize, 10usize, 47321u16);
        let data = inputs(n, len);

        // 模拟器侧：全部期望结果算好
        let mut w = World::new(n);
        let exp_ar = w.all_reduce_sum(&data);
        let exp_rs = w.reduce_scatter_sum(&data);
        let exp_ag = w.all_gather(&data);
        let exp_bc = w.broadcast(1, &data);

        let errs: Vec<_> = (0..n)
            .map(|r| {
                let data = data.clone();
                thread::spawn(move || {
                    let comm = Comm::new(r, n, "127.0.0.1", base);
                    let g = comm.group(7, (0..n).collect());
                    g.barrier();
                    let ar = g.all_reduce_sum(&data[r]);
                    let rs = g.reduce_scatter_sum(&data[r]);
                    let ag_parts = g.all_gather(&data[r]);
                    let ag: Vec<f32> = ag_parts.into_iter().flatten().collect();
                    let bc = g.broadcast(1, &data[r]);
                    (ar, rs, ag, bc)
                })
            })
            .collect();
        for (r, h) in errs.into_iter().enumerate() {
            let (ar, rs, ag, bc) = h.join().expect("工作线程 panic");
            assert_eq!(ar, exp_ar[r], "rank {r} allreduce 与模拟器不一致");
            assert_eq!(rs, exp_rs[r], "rank {r} reduce-scatter 与模拟器不一致");
            assert_eq!(ag, exp_ag[r], "rank {r} all-gather 与模拟器不一致");
            assert_eq!(bc, exp_bc[r], "rank {r} broadcast 与模拟器不一致");
        }
    }

    /// 2 rank 基础路径 + 点对点双向互发
    #[test]
    fn test_tcp_two_ranks_and_p2p() {
        let (n, len, base) = (2usize, 6usize, 47331u16);
        let data = inputs(n, len);
        let mut w = World::new(n);
        let exp = w.all_reduce_sum(&data);

        let hs: Vec<_> = (0..n)
            .map(|r| {
                let data = data.clone();
                let exp = exp[r].clone();
                thread::spawn(move || {
                    let comm = Comm::new(r, n, "127.0.0.1", base);
                    let g = comm.group(1, (0..n).collect());
                    let ar = g.all_reduce_sum(&data[r]);
                    assert_eq!(ar, exp, "rank {r} allreduce 与模拟器不一致");
                    // 双向 p2p：每人发自己那份给对方，再收对方的
                    let p2p = comm.group(2, (0..n).collect());
                    let peer = 1 - r;
                    p2p.send(peer, &data[r]);
                    let got = p2p.recv(peer);
                    assert_eq!(got, data[peer], "rank {r} p2p 收到的数据不对");
                })
            })
            .collect();
        for h in hs {
            h.join().expect("工作线程 panic");
        }
    }

    /// world=1 退化：所有集合通信都是恒等操作，不起任何连接
    #[test]
    fn test_tcp_single_rank_is_identity() {
        let comm = Comm::new(0, 1, "127.0.0.1", 47341);
        let g = comm.group(0, vec![0]);
        assert_eq!(g.all_reduce_sum(&[1.0, 2.0]), vec![1.0, 2.0]);
        let (c, d) = g.reduce_scatter_sum(&[3.0]);
        assert_eq!((c, d), (0, vec![3.0]));
        assert_eq!(g.all_gather(&[9.0]), vec![vec![9.0]]);
        g.barrier();
    }

    /// 编解码往返不丢精度
    #[test]
    fn test_tcp_encode_decode_roundtrip() {
        let v: Vec<f32> = (0..17).map(|i| i as f32 * -0.125 + 3.5).collect();
        assert_eq!(decode(&encode(&v)), v);
    }

    /// DistRuntime 的三维组圈定 + 各组独立通信：4 进程（dp=2 × tp=2 × pp=1），
    /// 每个 rank 先在 TP 组内 allreduce、再在 DP 组内 allreduce，
    /// 结果必须等于"按 DistConfig 圈出的成员求和"，且两种切法互不串包。
    #[test]
    fn test_tcp_dist_runtime_3d_groups() {
        let cfg = DistConfig::new(2, 2, 1);
        let (n, base) = (cfg.world_size(), 47351u16);
        let hs: Vec<_> = (0..n)
            .map(|r| {
                let cfg = cfg;
                thread::spawn(move || {
                    let rt = DistRuntime {
                        comm: Comm::new(r, n, "127.0.0.1", base),
                        cfg,
                        local_rank: r,
                    };
                    // 组成员必须与纯函数划分一致
                    let tp = rt.tp_group();
                    let dp = rt.dp_group();
                    assert_eq!(tp.size(), cfg.tp, "rank {r} TP 组大小不对");
                    assert_eq!(dp.size(), cfg.dp, "rank {r} DP 组大小不对");
                    // TP 组 allreduce：输入带 rank 标记，期望 = TP 成员之和
                    let mine = [10.0 + r as f32];
                    let tp_sum = tp.all_reduce_sum(&mine);
                    let want: f32 =
                        cfg.tp_group(r).iter().map(|&x| 10.0 + x as f32).sum();
                    assert_eq!(tp_sum, vec![want], "rank {r} TP 组 allreduce 不对");
                    // DP 组 allreduce：同一 (tp,pp) 坐标上的 dp 个 rank 求和
                    let dp_sum = dp.all_reduce_sum(&mine);
                    let want: f32 =
                        cfg.dp_group(r).iter().map(|&x| 10.0 + x as f32).sum();
                    assert_eq!(dp_sum, vec![want], "rank {r} DP 组 allreduce 不对");
                    // rank 编号约定：tp 在最低位
                    assert_eq!(r % cfg.tp, cfg.coord(r).tp, "rank {r} 坐标换算不对");
                })
            })
            .collect();
        for h in hs {
            h.join().expect("工作线程 panic");
        }
    }

    // ---------- DP 接线的三个通信点 ----------

    /// DP 梯度同步：各 rank 的扁平梯度求平均后写回，结果必须等于手工平均，
    /// 且各 rank 从此**逐位相同**（分块错位、少除 N 这类错误在这里就会露出来）。
    #[test]
    fn test_tcp_dp_gradient_average_matches_manual_mean() {
        let cfg = DistConfig::new(2, 1, 1);
        let (n, base, len) = (cfg.world_size(), 47401u16, 7usize);
        let hs: Vec<_> = (0..n)
            .map(|r| {
                thread::spawn(move || {
                    let rt = DistRuntime { comm: Comm::new(r, n, "127.0.0.1", base), cfg, local_rank: r };
                    let p = crate::tensor::Tensor::param(vec![0.0; len], vec![len]);
                    *p.grad.borrow_mut() = (0..len).map(|i| (r * 100 + i) as f32).collect();
                    let dp = rt.dp_group();
                    allreduce_mean_grads(&dp, std::slice::from_ref(&p), 1.0);
                    p.grad.borrow().clone()
                })
            })
            .collect();
        let got: Vec<Vec<f32>> =
            hs.into_iter().map(|h| h.join().expect("工作线程 panic")).collect();
        let want: Vec<f32> = (0..len)
            .map(|i| (0..n).map(|r| (r * 100 + i) as f32).sum::<f32>() * (1.0 / n as f32))
            .collect();
        for (r, g) in got.iter().enumerate() {
            assert_eq!(g, &want, "rank {r} 的 DP 平均梯度不等于各副本的手工平均");
            assert_eq!(g, &got[0], "DP 同步后各 rank 的梯度必须逐位相同");
        }
    }

    /// AMP 溢出标志的跨 DP 汇总：**任一**副本溢出，全体都必须看到"溢出"；
    /// 只有全都干净时才允许更新参数（否则 scale 会分叉）。
    #[test]
    fn test_tcp_sync_overflow_any_rank_wins() {
        for (i, (local_flags, want)) in
            [([false, false], false), ([true, false], true), ([false, true], true), ([true, true], true)]
                .into_iter()
                .enumerate()
        {
            let cfg = DistConfig::new(2, 1, 1);
            // 每轮换一组端口：上一轮的监听套接字可能还在 TIME_WAIT 里
            let (n, base) = (cfg.world_size(), 47411u16 + (i as u16) * 8);
            let hs: Vec<_> = (0..n)
                .map(|r| {
                    let local = local_flags[r];
                    thread::spawn(move || {
                        let rt = DistRuntime { comm: Comm::new(r, n, "127.0.0.1", base), cfg, local_rank: r };
                        let dp = rt.dp_group();
                        sync_overflow(&dp, local)
                    })
                })
                .collect();
            let got: Vec<bool> =
                hs.into_iter().map(|h| h.join().expect("工作线程 panic")).collect();
            for (r, g) in got.iter().enumerate() {
                assert_eq!(
                    *g, want,
                    "本地标志 {local_flags:?} 时 rank {r} 看到的溢出标志应为 {want}（各副本必须一致）"
                );
            }
        }
    }

    /// MoE 均衡偏置的跨 DP 同步：各副本先各自持有不同的偏置（负载不同必然如此），
    /// 同步后必须人人相同且等于平均值。
    ///
    /// 偏置不同步的后果不是精度而是**挂死**：偏置参与下一次前向的硬路由，
    /// 各副本的 val_loss 因此不同，最优/早停判据分叉，一部分 rank 退出循环之后
    /// 另一部分还在等它做集合通信。
    #[test]
    fn test_tcp_moe_balance_bias_averages_across_dp() {
        use crate::model::{Transformer, TransformerConfig};
        use crate::rng::Rng;

        let cfg = DistConfig::new(2, 1, 1);
        let (n, base) = (cfg.world_size(), 47451u16);
        let hs: Vec<_> = (0..n)
            .map(|r| {
                thread::spawn(move || {
                    let rt = DistRuntime { comm: Comm::new(r, n, "127.0.0.1", base), cfg, local_rank: r };
                    let mcfg = TransformerConfig {
                        n_embd: 16,
                        n_head: 2,
                        n_layer: 2,
                        block_size: 16,
                        n_expert: 2,
                        moe_top_k: 1,
                        moe_bias_balance: true,
                        ..TransformerConfig::tiny(24)
                    };
                    // 每个副本建**自己的**模型（同种子 = 同结构），这正是 SPMD 的形态
                    let model = Transformer::new(mcfg, &mut Rng::new(7));
                    let layers = model.moe_bias_layer_count();
                    assert!(layers > 0, "本测试需要开了 aux-loss-free 均衡的 MoE 配置");
                    let flat = model.moe_balance_flat();
                    let mine: Vec<f32> =
                        flat.iter().enumerate().map(|(i, _)| (r * 10 + i) as f32).collect();
                    model.set_moe_balance_flat(&mine);

                    let dp = rt.dp_group();
                    assert_eq!(allreduce_mean_moe_bias(&dp, &model), layers);
                    model.moe_balance_flat()
                })
            })
            .collect();
        let got: Vec<Vec<f32>> =
            hs.into_iter().map(|h| h.join().expect("工作线程 panic")).collect();
        let len = got[0].len();
        assert!(len > 0, "均衡偏置不该是空表");
        let want: Vec<f32> = (0..len)
            .map(|i| (0..n).map(|r| (r * 10 + i) as f32).sum::<f32>() * (1.0 / n as f32))
            .collect();
        for (r, g) in got.iter().enumerate() {
            assert_eq!(g, &want, "rank {r} 同步后的均衡偏置应为各副本的平均");
        }
    }
}
