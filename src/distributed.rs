//! 分布式训练（第 38 课）：集合通信、数据并行、ZeRO、张量并行、流水线并行、3D 并行
//!
//! 模型大到单卡放不下、或者训练慢到不可接受时，唯一的出路是把工作铺到多张卡上。
//! 但"并行"不是一个开关，而是**一套切分方式 + 一套配套通信**：切在哪里，就要在哪里
//! 把缺的部分补回来。本模块把这套对应关系完整实现出来。
//!
//! 实现方式是**单进程模拟**：一个"rank"就是一个普通的数据结构，集合通信按真实的
//! 环形算法逐个通信轮推进（每轮只在相邻 rank 之间搬一块数据），于是
//! "切分 → 通信 → 拼回"的每一步都可读、可跑、可与单卡逐位对拍。
//!
//! 为什么不真的起 N 个进程：本课要教的是**通信模式本身**——谁在什么时候把哪一块
//! 发给谁、发完之后谁手里有什么。这个模式一旦用模拟器写清楚，换成 NCCL / MPI
//! 只是把 [`ring_round`] 里的 memcpy 换成 `send`/`recv` 系统调用。而"梯度平均到底
//! 对不对""每个 rank 是否真的只持有 1/N 的状态"这些**真正会出错的地方**，
//! 在单进程里就能逐位验证，不需要集群。
//!
//! 六节：
//! 1. **集合通信**：[`World`]（环形 allreduce = reduce-scatter + all-gather、
//!    [`World::all_gather`]、[`World::broadcast`]、[`World::barrier`]）与通信量统计 [`CommLog`]
//! 2. **数据并行**：[`DataParallel`]（本地梯度 → allreduce 平均 → 各副本完全同步）
//! 3. **ZeRO**：[`ZeroOptimizer`]（优化器状态分片、梯度分片，单卡状态量降到 1/N）
//! 4. **张量并行**：[`ColumnParallelLinear`] / [`RowParallelLinear`]（层内按列/行切分 + 通信，
//!    拼成 [`TensorParallelMlp`]；注意力的 QKV 按 head 切分见 [`qkv_head_columns`]）
//! 5. **流水线并行**：[`Pipeline`] 与两种调度 [`Schedule::gpipe`] /
//!    [`Schedule::one_forward_one_backward`]（micro-batch 与激活驻留）
//! 6. **3D 并行**：[`DistConfig`]（dp × tp × pp 正交切分与 rank 分工报告）

use crate::layers::Linear;
use crate::optim::{AdamW, Optimizer};
use crate::tensor::Tensor;

// ==================== 1. 集合通信 ====================

/// 一次集合通信的通信量统计。
///
/// 只统计**发出去**的元素数：集合通信里每个 rank 的收发量相等（发多少就收多少），
/// 记一份就够。`per_rank` 让"各 rank 负载是否均衡"变成可断言的事实——
/// 这正是环形算法与主从广播的分水岭。
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct CommLog {
    /// 通信轮数（环上每走一跳算一轮）
    pub rounds: usize,
    /// 每个 rank 累计发送的**元素个数**
    pub per_rank: Vec<usize>,
}

impl CommLog {
    fn new(ranks: usize) -> Self {
        CommLog { rounds: 0, per_rank: vec![0; ranks] }
    }

    /// 所有 rank 累计发送的元素总数
    pub fn total_sent(&self) -> usize {
        self.per_rank.iter().sum()
    }

    /// 单个元素 4 字节，折算成字节数
    pub fn per_rank_bytes(&self) -> Vec<usize> {
        self.per_rank.iter().map(|e| e * std::mem::size_of::<f32>()).collect()
    }

    /// 发得最多的那个 rank 的字节数——瓶颈就是它
    pub fn max_rank_bytes(&self) -> usize {
        self.per_rank.iter().max().copied().unwrap_or(0) * std::mem::size_of::<f32>()
    }
}

/// 把 `len` 个元素切成 `n` 份，返回第 `i` 份的 `[start, end)`。
///
/// 除不尽时**前 `len % n` 份各多 1 个**（余数均匀摊在前面），而不是把余数全丢给
/// 最后一份：后者会让最后一块大出一截，环上"每轮搬一块"的通信量就不再均衡，
/// 而通信是同步的，最慢的那一轮决定整体耗时。
pub fn chunk_range(i: usize, len: usize, n: usize) -> (usize, usize) {
    let base = len / n;
    let rem = len % n;
    let start = i * base + i.min(rem);
    (start, start + base + if i < rem { 1 } else { 0 })
}

/// 环形下标：把可能为负的 `a` 折回 `[0, n)`。环上"左邻居"就是 `-1`，
/// 直接写 `(r - 1) % n` 在 usize 上会向下溢出 panic。
fn wrap(a: isize, n: usize) -> usize {
    (((a % n as isize) + n as isize) as usize) % n
}

/// 一个 N 进程通信域的**单进程模拟**：所有 rank 的通信缓冲都由本结构持有。
///
/// 唯一的通信原语是 [`ring_round`]。它按"同步轮"语义执行——所有发送都取本轮开始时
/// 的**快照**：真实网络里收发确实是并发的，但环形算法每轮只依赖上一轮的收包，
/// 快照语义与之一致，而且能杜绝"某个 rank 读到了邻居本轮的中间结果"这种
/// 并行程序里最难查的错。
pub struct World {
    size: usize,
    log: CommLog,
    barriers: usize,
}

/// 一轮环形邻居通信：rank `r` 把第 `send_of(r)` 块发给右邻居 `r + 1`，
/// 同时从左邻居 `r - 1` 收下一块，落到本地的第 `recv_of(r)` 块。
///
/// `accumulate = true` 累加（reduce-scatter 阶段），`false` 覆盖（all-gather 阶段）。
/// 调用方通过 `send_of` / `recv_of` 决定"这一轮谁发哪块、谁收哪块"——
/// 两个阶段只是这两组下标不同，通信动作本身完全一样。
fn ring_round(
    bufs: &mut [Vec<f32>],
    send_of: impl Fn(usize) -> usize,
    recv_of: impl Fn(usize) -> usize,
    accumulate: bool,
    log: &mut CommLog,
) {
    let n = bufs.len();
    if n == 1 {
        return; // 单 rank：没有邻居，集合通信退化成恒等操作
    }
    let len = bufs[0].len();
    // 本轮的发件箱：必须**先全部取快照**再投递。若边发边收，rank r 覆盖自己的块时
    // 就可能改掉 rank r-1 本轮要发的内容（同一块内存被两个角色引用）。
    let outbox: Vec<Vec<f32>> = (0..n)
        .map(|r| {
            let (s, e) = chunk_range(send_of(r), len, n);
            bufs[r][s..e].to_vec()
        })
        .collect();
    for r in 0..n {
        let from = wrap(r as isize - 1, n);
        let (s, e) = chunk_range(recv_of(r), len, n);
        let src = &outbox[from];
        assert_eq!(
            e - s,
            src.len(),
            "环形通信两端块大小不一致：rank {r} 收 {} 个，rank {from} 发 {} 个",
            e - s,
            src.len()
        );
        if accumulate {
            for (d, v) in bufs[r][s..e].iter_mut().zip(src) {
                *d += v;
            }
        } else {
            bufs[r][s..e].copy_from_slice(src);
        }
        log.per_rank[r] += src.len();
    }
    log.rounds += 1;
}

/// 把"按 rank 顺序拼接"的缓冲重排成"按块下标顺序"。
///
/// reduce-scatter 结束时 **rank `r` 手里是第 `(r+1) mod N` 块**的全和（这是环上
/// 轮流累加的必然结果，不是可以随便选的约定）。所以 all-gather 之后拿到的
/// `rank0 块 ++ rank1 块 ++ ...` 恰好是块序整体左移一格，必须转回来。
/// 少了这一步，结果"看起来"是一堆数字，只是每个位置都错位了——不会 panic。
///
/// 注意**长度必须按目标块下标取**：`len` 不能被 N 整除时各块大小不等，
/// 拼接缓冲里的第 `r` 段长度是块 `(r+1) mod N` 的大小（7、6 这样的差别），
/// 而不是第 `r` 块的大小。用错下标会让 `copy_from_slice` 长度不匹配直接 panic。
fn rotate_chunks(gathered: &[f32], n: usize) -> Vec<f32> {
    let len = gathered.len();
    let mut out = vec![0.0f32; len];
    let mut src_off = 0usize;
    for r in 0..n {
        let dst_chunk = (r + 1) % n;
        let (ds, de) = chunk_range(dst_chunk, len, n);
        let size = de - ds;
        out[ds..de].copy_from_slice(&gathered[src_off..src_off + size]);
        src_off += size;
    }
    debug_assert_eq!(src_off, len, "重排必须刚好覆盖整段缓冲");
    out
}

impl World {
    pub fn new(size: usize) -> Self {
        assert!(size > 0, "通信域至少要有 1 个 rank");
        World { size, log: CommLog::new(size), barriers: 0 }
    }

    pub fn size(&self) -> usize {
        self.size
    }

    /// 最近一次集合通信的通信量（每次集合调用开始时会清零）
    pub fn log(&self) -> &CommLog {
        &self.log
    }

    /// 已经经过的屏障次数
    pub fn barriers(&self) -> usize {
        self.barriers
    }

    /// 同步屏障：所有 rank 都到齐后才继续。
    ///
    /// 单进程模拟里没有东西会真的阻塞，所以它只做两件**可观测**的事：记一个同步点、
    /// 把通信日志清零。它真实的作用是划出"通信阶段"的边界——没有屏障，
    /// A 组的第二次 allreduce 可能和 B 组的第一次撞进同一个缓冲（真实网络里
    /// 这不会报错，只会把两批数据混在一起，静默算错）。
    pub fn barrier(&mut self) {
        self.barriers += 1;
        self.log = CommLog::new(self.size);
    }

    /// 求和 allreduce：返回每个 rank 上的完整结果，N 份**逐位相同**。
    ///
    /// 两阶段环形算法（NCCL / Horovod 真正在用的那个）：
    ///
    /// - **reduce-scatter**（N-1 轮）：rank `r` 第 `s` 轮把第 `r-s` 块发给右邻居，
    ///   把收到的第 `r-s-1` 块累加进本地。走完 N-1 轮，rank `r` 恰好有**一块完整的和**。
    /// - **all-gather**（N-1 轮）：把这块已算好的和沿环转一圈，每轮覆盖一格。
    ///
    /// 每轮每 rank 只发一块 ≈ `len/N`，所以每 rank 总量 ≈ `2(N-1)/N · len`，
    /// 与 N 几乎无关——这是 allreduce 的带宽下界。主从广播（[`Self::all_reduce_sum_naive`]）
    /// 则是"所有 rank 发给 rank 0，rank 0 再发回来"，rank 0 的收发量是别人的 N 倍，
    /// 它一个人就是瓶颈，所以只留作对照，不作为实现。
    pub fn all_reduce_sum(&mut self, locals: &[Vec<f32>]) -> Vec<Vec<f32>> {
        let len = self.check_locals(locals);
        self.log = CommLog::new(self.size);
        let n = self.size;
        if n == 1 {
            return vec![locals[0].clone()];
        }
        // 阶段 1：每 rank 拿到自己负责那块的全和
        let owned = self.reduce_scatter_inner(locals);
        // 阶段 2：按 rank 顺序收集（rank r 贡献自己那块）
        let blocks_by_rank: Vec<Vec<f32>> = owned.iter().map(|(_, d)| d.clone()).collect();
        let gathered = self.all_gather_inner(&blocks_by_rank);
        // 阶段 3：纯粹的下标换算，把块序转回原始顺序（见 rotate_chunks）
        let assembled = rotate_chunks(&gathered[0], n);
        debug_assert_eq!(assembled.len(), len);
        vec![assembled; n]
    }

    /// 只做环形 allreduce 的第一阶段：求和结果被**切开**留在各 rank 上。
    ///
    /// 返回 `(块下标, 该块的全和)`，即 rank `r` 拿到的是第 `(r+1) mod N` 块。
    /// ZeRO 的梯度分片靠它：[`World::all_reduce_sum`] 的后两阶段（收集 + 重排）
    /// 在这里是**纯浪费**——每个 rank 只需要自己那 1/N 的梯度，没必要把它们
    /// 拼成完整的向量再人人存一份，那等于把省下来的显存又还回去了。
    pub fn reduce_scatter_sum(&mut self, locals: &[Vec<f32>]) -> Vec<(usize, Vec<f32>)> {
        self.check_locals(locals);
        self.log = CommLog::new(self.size);
        self.reduce_scatter_inner(locals)
    }

    fn reduce_scatter_inner(&mut self, locals: &[Vec<f32>]) -> Vec<(usize, Vec<f32>)> {
        let n = self.size;
        let mut bufs = locals.to_vec();
        if n == 1 {
            return vec![(0, bufs.pop().unwrap())];
        }
        // rank r 在第 s 轮：发第 (r-s) 块，收第 (r-s-1) 块并累加
        for s in 0..n - 1 {
            let s = s as isize;
            ring_round(
                &mut bufs,
                |r| wrap(r as isize - s, n),
                |r| wrap(r as isize - s - 1, n),
                true,
                &mut self.log,
            );
        }
        // N-1 轮之后，rank r 手里完整的和是第 (r+1) mod N 块
        let len = bufs[0].len();
        (0..n)
            .map(|r| {
                let c = (r + 1) % n;
                let (s, e) = chunk_range(c, len, n);
                (c, bufs[r][s..e].to_vec())
            })
            .collect()
    }

    /// 收集所有 rank 的数据并**按 rank 顺序首尾拼接**，每个 rank 得到同样的完整结果。
    ///
    /// 环形实现（N-1 轮）：rank `r` 起初只有自己那块；第 `s` 轮它把刚拿到的
    /// `r-s` 块转发给右邻居，同时收下 `r-s-1` 块。因为每一轮把"刚到手的东西"
    /// 继续往右推，N-1 轮后所有块绕环一周、人人都拿全。
    ///
    /// 各 rank 的块**允许不等长**（ZeRO 按参数区间分片时就会出现 7、6 这样的差别）。
    /// NCCL 的原生 all_gather 要求等长，不等长要用 all_gatherv；这里实现的是后者，
    /// 拼接顺序仍严格按 rank 升序。
    pub fn all_gather(&mut self, locals: &[Vec<f32>]) -> Vec<Vec<f32>> {
        self.check_rank_count(locals);
        self.log = CommLog::new(self.size);
        self.all_gather_inner(locals)
    }

    fn all_gather_inner(&mut self, locals: &[Vec<f32>]) -> Vec<Vec<f32>> {
        let n = self.size;
        if n == 1 {
            return vec![locals[0].clone()];
        }
        // blocks[r][i] = rank r 目前手上"来自 rank i"的那块
        let mut blocks: Vec<Vec<Vec<f32>>> = (0..n)
            .map(|r| {
                let mut b = vec![Vec::new(); n];
                b[r] = locals[r].clone();
                b
            })
            .collect();
        for s in 0..n - 1 {
            let s = s as isize;
            // 快照：rank r 本轮转发的是它手上最新的那块 (r-s)
            let outbox: Vec<Vec<f32>> = (0..n)
                .map(|r| blocks[r][wrap(r as isize - s, n)].clone())
                .collect();
            for r in 0..n {
                let origin = wrap(r as isize - s - 1, n);
                let src = outbox[wrap(r as isize - 1, n)].clone();
                assert_eq!(src.len(), locals[origin].len(), "all_gather 要求各 rank 的块等长");
                blocks[r][origin] = src;
                self.log.per_rank[r] += blocks[r][origin].len();
            }
            self.log.rounds += 1;
        }
        blocks.iter().map(|b| b.iter().flatten().copied().collect()).collect()
    }

    /// 从 `src` 广播到所有 rank。只有 `locals[src]` 的内容会被采用，
    /// 其余 rank 传什么无所谓（真实场景里它们此时是未初始化内存）。
    ///
    /// 环形逐跳传递（N-1 轮）：`src → src+1 → …` 一圈。真实实现里广播常走树形
    /// （`log N` 轮），但环形的通信模式与 allreduce 一致、不需要额外的拓扑假设，
    /// 而且每轮每 rank 的收发量相同——本课演示"每跳只跟邻居说话"这件事已经够了。
    pub fn broadcast(&mut self, src: usize, locals: &[Vec<f32>]) -> Vec<Vec<f32>> {
        let n = self.size;
        assert!(src < n, "广播源 rank {src} 超出通信域大小 {n}");
        self.log = CommLog::new(n);
        let mut have: Vec<Option<Vec<f32>>> = vec![None; n];
        have[src] = Some(locals[src].clone());
        for _ in 0..n - 1 {
            let snapshot = have.clone();
            for r in 0..n {
                if have[r].is_none() {
                    let from = wrap(r as isize - 1, n);
                    if let Some(v) = &snapshot[from] {
                        self.log.per_rank[from] += v.len();
                        have[r] = Some(v.clone());
                    }
                }
            }
            self.log.rounds += 1;
        }
        have.into_iter().map(|v| v.expect("广播结束时所有 rank 都应收到数据")).collect()
    }

    /// 主从广播式 allreduce：所有 rank 发给 rank 0，rank 0 求和后再发回给所有人。
    /// **仅作对照**——结果与环形一致，但 rank 0 的通信量是其他 rank 的 N 倍。
    pub fn all_reduce_sum_naive(&mut self, locals: &[Vec<f32>]) -> Vec<Vec<f32>> {
        let len = self.check_locals(locals);
        let n = self.size;
        self.log = CommLog::new(n);
        // 第一轮：各 rank 上传（rank 0 收 n-1 份）
        for r in 1..n {
            self.log.per_rank[r] += len;
        }
        let mut sum = locals[0].clone();
        for l in &locals[1..] {
            for (a, b) in sum.iter_mut().zip(l) {
                *a += b;
            }
        }
        // 第二轮：rank 0 下发（n-1 份）
        self.log.per_rank[0] += (n - 1) * len;
        self.log.rounds = if n > 1 { 2 } else { 0 };
        vec![sum; n]
    }

    /// 校验 rank 数量与通信域一致
    fn check_rank_count(&self, locals: &[Vec<f32>]) {
        assert_eq!(
            locals.len(),
            self.size,
            "通信域有 {} 个 rank，却收到 {} 份本地数据",
            self.size,
            locals.len()
        );
    }

    /// 校验各 rank 的本地缓冲长度一致，返回公共长度（分块类集合通信需要等长）
    fn check_locals(&self, locals: &[Vec<f32>]) -> usize {
        self.check_rank_count(locals);
        let len = locals[0].len();
        assert!(locals.iter().all(|l| l.len() == len), "各 rank 的数据长度必须一致");
        len
    }
}

// ==================== 2. 数据并行 ====================

/// 把一个参数列表的梯度拍平成一条**连续缓冲**：`[p0 梯度, p1 梯度, ...]`。
///
/// 集合通信只认连续内存。真实实现（NCCL、以及各家框架手写的 gradient bucketing）
/// 也是这么做的：每层梯度各占一段，拼成一条发出去，收回来再切回去。
/// 拼接顺序必须**所有 rank 完全一致**——否则第 i 段的梯度会被加到第 j 个参数上，
/// 训练照样跑得下去，只是每个人都在学错的东西，而且不会报错。
pub fn flatten_grads(params: &[Tensor]) -> Vec<f32> {
    let mut out = Vec::with_capacity(params.iter().map(|p| p.numel()).sum());
    for p in params {
        out.extend_from_slice(&p.grad.borrow());
    }
    out
}

/// [`flatten_grads`] 的逆操作：按各参数长度把缓冲切回去，乘 `scale` 后**覆盖**写入梯度槽。
///
/// 覆盖而不是累加：同步之后的梯度代表"这一轮算出来的梯度"，不是"再叠一层"。
/// 若这里写成累加，每同步一次梯度就翻一倍，训练会在几步之内飞出去。
pub fn write_grads(params: &[Tensor], flat: &[f32], scale: f32) {
    let total: usize = params.iter().map(|p| p.numel()).sum();
    assert_eq!(flat.len(), total, "扁平梯度长度与参数不匹配：{} vs {total}", flat.len());
    let mut off = 0;
    for p in params {
        let n = p.numel();
        let mut slot = p.grad.borrow_mut();
        for (j, v) in slot.iter_mut().enumerate() {
            *v = flat[off + j] * scale;
        }
        off += n;
    }
}

/// 把一个参数列表的**数值**拍平成一条连续缓冲。与 [`flatten_grads`] 严格同序，
/// 因此"扁平参数第 j 个元素"与"扁平梯度第 j 个元素"永远指同一个参数的同一位。
///
/// ZeRO 全程在这条扁平缓冲上工作：切分、更新、拼回都只认下标区间。
pub fn flatten_params(params: &[Tensor]) -> Vec<f32> {
    let mut out = Vec::with_capacity(params.iter().map(|p| p.numel()).sum());
    for p in params {
        out.extend_from_slice(&p.data_ref());
    }
    out
}

/// 把扁平参数写回参数张量（[`flatten_params`] 的逆操作，逐张量 `set_data`）。
///
/// 用 `set_data` 就地改数值而不是换一个新张量：优化器和模型的句柄都指向同一个参数，
/// 换张量会变成"模型用新的、优化器更新旧的"两本账（见 [`Tensor::set_data`] 的说明）。
pub fn write_params(params: &[Tensor], flat: &[f32]) {
    let total: usize = params.iter().map(|p| p.numel()).sum();
    assert_eq!(flat.len(), total, "扁平参数长度与参数不匹配：{} vs {total}", flat.len());
    let mut off = 0;
    for p in params {
        let n = p.numel();
        p.set_data(flat[off..off + n].to_vec());
        off += n;
    }
}

/// 校验各副本的参数结构完全一致（个数与形状）。
///
/// 这是 DP 最容易踩的坑：副本本应是"同一份模型的不同拷贝"，一旦某个 rank 的模型
/// 结构不同（比如只在这个 rank 上加了 LoRA），扁平缓冲的分段就会错位，
/// 通信结果仍然"有值"，只是每个参数拿到的都是别人的梯度。
fn check_same_structure(replicas: &[Vec<Tensor>]) {
    let first = &replicas[0];
    for (r, params) in replicas.iter().enumerate() {
        assert_eq!(params.len(), first.len(), "rank {r} 的参数个数与其他 rank 不一致");
        for (i, (a, b)) in first.iter().zip(params).enumerate() {
            assert_eq!(
                a.shape(),
                b.shape(),
                "rank {r} 的第 {i} 个参数形状不一致：{:?} vs {:?}",
                b.shape(),
                a.shape()
            );
        }
    }
}

/// 数据并行（DP）：每个 rank 持有一份**完整模型副本**，各吃一部分数据，
/// 反向得到本地梯度，再把所有 rank 的梯度求平均，于是所有副本的参数永远一致。
///
/// 它换来的是**吞吐**而不是容量：模型仍必须单卡放得下，切开的只是数据。
///
/// 梯度平均为什么是对的：设全局 batch 被均分成 N 份、每份 m 个样本，本地用的是
/// **本份的平均损失** `L_r = (1/m)·Σ_{i∈r} ℓ_i`，同步时求平均：
///
/// ```text
/// (1/N)·Σ_r ∇L_r = (1/(N·m))·Σ_i ∇ℓ_i
/// ```
///
/// 正是全局 batch 的平均损失梯度。两处"平均"缺一不可：本地不取平均（改成求和）
/// 会让梯度放大 m 倍，同步后不除以 N 会让梯度放大 N 倍——两者都只是把学习率
/// 悄悄改了，训练照样跑，loss 曲线看起来甚至更"漂亮"，所以必须靠单测钉死。
pub struct DataParallel {
    world: World,
}

impl DataParallel {
    pub fn new(world_size: usize) -> Self {
        DataParallel { world: World::new(world_size) }
    }

    pub fn world_size(&self) -> usize {
        self.world.size()
    }

    /// 最近一次梯度同步的通信量
    pub fn log(&self) -> &CommLog {
        self.world.log()
    }

    /// 同步梯度：`replicas[r]` 是第 r 个 rank 的参数列表（**同序、同形状**），
    /// 调用前各 rank 的梯度槽里已经放好本地梯度。
    ///
    /// `local_scale` 是本地缩放系数，用于梯度累加：如果本地攒了 k 个 micro-step
    /// 才同步一次，槽里是 k 份梯度之和，传 `1/k` 把"和"还原成"平均"，
    /// 于是"本地累加 k 次再同步"与"逐个 micro-step 同步"在数学上完全等价。
    ///
    /// 只做通信、不做参数更新：更新交给各 rank 自己的优化器（同构 DP 下各副本
    /// 更新完全一样；ZeRO 则要求把优化器状态也切开，那一步在下一节）。
    pub fn sync_gradients(&mut self, replicas: &[Vec<Tensor>], local_scale: f32) {
        let n = self.world.size();
        assert_eq!(replicas.len(), n, "副本数应与通信域大小一致：{} vs {n}", replicas.len());
        check_same_structure(replicas);
        let locals: Vec<Vec<f32>> = replicas.iter().map(|ps| flatten_grads(ps)).collect();
        // 求和 allreduce；每个 rank 拿到的是一份**逐位相同**的完整结果
        let summed = self.world.all_reduce_sum(&locals);
        let scale = local_scale / n as f32;
        for (r, params) in replicas.iter().enumerate() {
            write_grads(params, &summed[r], scale);
        }
    }
}

// ==================== 3. ZeRO：切分优化器状态与梯度 ====================

/// ZeRO 的阶段（术语沿用论文）
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ZeroStage {
    /// 只切**优化器状态**（m/v）：梯度照旧完整 allreduce，参数每个 rank 也完整。
    /// 省下的却是最大的一块——Adam 每个参数要常驻 m、v 各 4 字节，
    /// 而参数本身才 4 字节、梯度 4 字节，**全量状态里有整整一半是优化器状态**。
    One,
    /// 再切**梯度**：用 reduce-scatter 只算出各自那一段的和，
    /// 完整梯度缓冲从头到尾不出现，显存再降一份。
    Two,
}

/// ZeRO-1/2：把 Adam 的 m/v（以及 stage 2 的梯度）按 rank 切开，每个 rank 只养 1/N。
///
/// 数据并行有个不显眼的开销：**每个 rank 都存一份完整的优化器状态**。
/// 可 m、v 只在"更新那一步"用得到，平时就是躺着占显存，
/// 而且在每个 rank 上都躺着——N 份完全一样的东西。
///
/// ZeRO 的做法：参数分片，每个 rank 只负责其中一段的 m/v；反向后把各段梯度
/// **求和送到它的主人那里**（reduce-scatter 天然就是这个形状），主人更新自己那段，
/// 再把更新后的参数段 all-gather 给所有人。每步只多一次通信，显存少一份状态。
///
/// 数值上它与"每 rank 完整状态"的 vanilla DP **完全等价**：Adam 的更新逐元素独立，
/// 第 j 个元素由谁算都不影响结果。这条等价性必须由单测钉死——分片最常出的错是
/// **边界错位**（第 j 个元素的 m/v 配到了第 j+1 个元素的梯度），
/// 那种错不会崩也不会 NaN，只会让 loss 曲线比之前"抖一点"，光看曲线根本看不出来。
///
/// 分片归属取 `chunk_range((r+1) mod N)`：这与 [`World::reduce_scatter_sum`] 的
/// 天然产出**完全对齐**，于是 stage 2 的梯度落到谁手里不需要任何额外搬运
/// （代价是最后拼回完整参数时要按块下标转回来，见 [`rotate_chunks`]）。
pub struct ZeroOptimizer {
    stage: ZeroStage,
    world: World,
    /// 每 rank 负责的扁平参数区间 `[start, end)`
    ranges: Vec<(usize, usize)>,
    /// 每 rank 那一段自己的 AdamW——m/v 就藏在它内部，长度只有 1/N
    opts: Vec<AdamW>,
    /// 每 rank 那一段参数的句柄。它承担两个角色：给 AdamW 提供真实的 θ（权重衰减要用），
    /// 以及作为"更新后的参数段"的出口。
    shards: Vec<Tensor>,
    total: usize,
}

impl ZeroOptimizer {
    pub fn new(
        stage: ZeroStage,
        world_size: usize,
        total: usize,
        lr: f32,
        weight_decay: f32,
    ) -> Self {
        assert!(world_size > 0 && total > 0, "ZeRO 需要一个非空的参数集");
        let ranges: Vec<(usize, usize)> = (0..world_size)
            .map(|r| chunk_range((r + 1) % world_size, total, world_size))
            .collect();
        let shards: Vec<Tensor> = ranges
            .iter()
            .map(|(s, e)| Tensor::param(vec![0.0; e - s], vec![e - s]))
            .collect();
        // 每 rank 一个只装"自己那段"的 AdamW：状态量随分片一起缩小到 1/N
        let opts = shards.iter().map(|t| AdamW::new(lr, vec![t.clone()], weight_decay)).collect();
        ZeroOptimizer { stage, world: World::new(world_size), ranges, opts, shards, total }
    }

    pub fn stage(&self) -> ZeroStage {
        self.stage
    }

    pub fn total_params(&self) -> usize {
        self.total
    }

    /// 每 rank 负责的扁平参数区间
    pub fn ranges(&self) -> &[(usize, usize)] {
        &self.ranges
    }

    pub fn log(&self) -> &CommLog {
        self.world.log()
    }

    /// 每 rank 实际常驻的优化器状态字节数（m + v 各 4 字节/元素）
    pub fn state_bytes_per_rank(&self) -> Vec<usize> {
        self.opts
            .iter()
            .map(|o| {
                let (_, m, v) = o.state();
                let e = m.iter().map(Vec::len).sum::<usize>() + v.iter().map(Vec::len).sum::<usize>();
                e * std::mem::size_of::<f32>()
            })
            .collect()
    }

    /// 不切分时**每个** rank 都要常驻的状态字节数（m + v 各一份完整长度）
    pub fn full_state_bytes(&self) -> usize {
        2 * self.total * std::mem::size_of::<f32>()
    }

    /// 一步 ZeRO 更新。
    ///
    /// - `params[r]`：rank `r` 手上的**完整**扁平参数（各 rank 相同）
    /// - `grads[r]`：rank `r` 用自己那份数据算出的**完整**扁平梯度
    /// - `local_scale`：梯度累加的本地缩放（语义同 [`DataParallel::sync_gradients`]）
    ///
    /// 返回每个 rank 上更新后的完整扁平参数（N 份逐位相同），由调用方
    /// [`write_params`] 写回模型继续下一步。
    pub fn step(&mut self, params: &[Vec<f32>], grads: &[Vec<f32>], local_scale: f32) -> Vec<Vec<f32>> {
        let n = self.world.size();
        assert_eq!(params.len(), n, "参数份数应与 rank 数一致");
        assert_eq!(grads.len(), n, "梯度份数应与 rank 数一致");
        assert_eq!(grads[0].len(), self.total, "扁平梯度长度应为 {}", self.total);
        assert!(
            params.iter().all(|p| p.len() == self.total && p == &params[0]),
            "ZeRO 假定各 rank 手上的完整参数当前是一致的；不一致说明上一步的 \
             all-gather 没写回，或者某个 rank 偷偷自己更新了参数"
        );
        let scale = local_scale / n as f32;

        // 1) 梯度同步：每 rank 只拿到**自己那一段**的平均梯度
        let shard_grads: Vec<Vec<f32>> = match self.stage {
            ZeroStage::One => {
                // 完整 allreduce：每 rank 都得到完整平均梯度……但只保留自己那段，
                // 剩下的立刻丢掉（这就是 ZeRO-1 与 ZeRO-2 的差别所在）
                let summed = self.world.all_reduce_sum(grads);
                let full = &summed[0];
                self.ranges
                    .iter()
                    .map(|(s, e)| full[*s..*e].iter().map(|v| v * scale).collect())
                    .collect()
            }
            ZeroStage::Two => {
                // reduce-scatter：完整梯度缓冲**从未存在过**，显存里只有 1/N 段
                let owned = self.world.reduce_scatter_sum(grads);
                owned
                    .iter()
                    .enumerate()
                    .map(|(r, (chunk, d))| {
                        debug_assert_eq!(
                            *chunk,
                            (r + 1) % n,
                            "reduce-scatter 的块归属必须与本结构的分片规划对齐"
                        );
                        d.iter().map(|v| v * scale).collect()
                    })
                    .collect()
            }
        };

        // 2) 各 rank 用自己那片状态、更新自己那段参数
        let mut segments: Vec<Vec<f32>> = Vec::with_capacity(n);
        for r in 0..n {
            let (s, e) = self.ranges[r];
            let shard = &self.shards[r];
            // 先把真实 θ 灌进去：AdamW 的权重衰减项是 lr·wd·θ，
            // 若让分片张量保持初始的 0，衰减就变成了空转（有 wd 没衰减）。
            shard.set_data(params[0][s..e].to_vec());
            shard.zero_grad();
            shard.accumulate_grad(&shard_grads[r], 1.0);
            self.opts[r].step();
            segments.push(shard.data());
        }

        // 3) 把各 rank 更新好的参数段拼回完整参数
        let gathered = self.world.all_gather(&segments);
        let full = rotate_chunks(&gathered[0], n);
        vec![full; n]
    }
}

// ==================== 4. 张量并行 ====================

/// 连续的列切分：把 `out` 列按 [`chunk_range`] 均分给 `world_size` 个 rank。
pub fn contiguous_columns(out: usize, world_size: usize) -> Vec<Vec<usize>> {
    (0..world_size)
        .map(|r| {
            let (s, e) = chunk_range(r, out, world_size);
            (s..e).collect()
        })
        .collect()
}

/// 连续的行切分：与 [`contiguous_columns`] 是同一个划分，只是以区间形式给出。
///
/// 列并行的"输出列"就是下一层行并行的"输入行"——两者必须落在同一个 rank 上，
/// 两层的接缝才能不通信（这是 TP 最核心的一条设计，见 [`TensorParallelMlp`]）。
pub fn contiguous_ranges(len: usize, world_size: usize) -> Vec<(usize, usize)> {
    (0..world_size).map(|r| chunk_range(r, len, world_size)).collect()
}

/// 融合 QKV 权重 `[d_model, 3·d_model]`（列序 `[q | k | v]`）的列切分：
/// **按 head 分**，每个 rank 拿到 `n_head / world_size` 个**完整**的 head。
///
/// 为什么不"把 3d 列拉平后按 3d/N 均分"：那样切点会落在 head 内部——`n_head = 4`、
/// `d_model = 16`、`N = 2` 时均分线在 24 列处，正好切开 v 段的第 2 个 head。
/// 而 attention 的每一步（`Q·Kᵀ`、softmax、`·V`）都以 **head 为单位**，
/// 半个 head 在手意味着这个 head 的 q 在这张卡、k/v 在另一张卡，
/// 每算一个 head 都要插一次通信，"层内通信只发生在层的接缝上"这条前提就没了。
///
/// 切好之后每张卡能在本地算完自己那几个 head 的注意力，**不需要任何通信**，
/// 直到输出投影（行并行）才 allreduce 一次。
pub fn qkv_head_columns(d_model: usize, n_head: usize, world_size: usize) -> Vec<Vec<usize>> {
    assert!(
        n_head % world_size == 0,
        "head 数 {n_head} 必须能被 rank 数 {world_size} 整除，否则 head 会被切到两张卡上"
    );
    assert!(d_model % n_head == 0, "d_model {d_model} 必须能被 head 数 {n_head} 整除");
    let head_dim = d_model / n_head;
    let per_rank = n_head / world_size;
    (0..world_size)
        .map(|r| {
            let mut cols = Vec::with_capacity(3 * per_rank * head_dim);
            for seg in 0..3 {
                for h in r * per_rank..(r + 1) * per_rank {
                    let base = seg * d_model + h * head_dim;
                    cols.extend(base..base + head_dim);
                }
            }
            cols
        })
        .collect()
}

/// 校验列切分：每列恰好归一个 rank、且都在 `[0, out_features)` 内。
///
/// 漏掉任何一列，对应参数就会**静默地永不获得梯度**——训练跑得下去，
/// 只是那一部分权重从头到尾没动过，光看 loss 曲线看不出来。
fn check_columns(columns: &[Vec<usize>], out_features: usize) {
    let mut seen = vec![false; out_features];
    for (r, cols) in columns.iter().enumerate() {
        for &c in cols {
            assert!(c < out_features, "rank {r} 负责的列 {c} 超出输出宽度 {out_features}");
            assert!(!seen[c], "第 {c} 列被分给了多个 rank：它的梯度会被算两遍");
            seen[c] = true;
        }
    }
    assert!(seen.iter().all(|s| *s), "有列没有被任何 rank 负责");
}

/// 校验行切分：区间首尾相接、从 0 铺到 `in_features`（不重不漏）
fn check_input_ranges(ranges: &[(usize, usize)], in_features: usize) {
    let mut next = 0usize;
    for (s, e) in ranges {
        assert_eq!(*s, next, "行切分必须首尾相接、从 0 开始：期望 {next}，实际 {s}");
        assert!(e > s, "空的输入分片会让这个 rank 什么都算不出来");
        next = *e;
    }
    assert_eq!(next, in_features, "行切分没有铺满输入维度：只到 {next}，应为 {in_features}");
}

/// 把 all_gather 收上来的**按 rank 顺序拼接**的缓冲，按列索引放回完整输出里
/// （行优先，`rows` = 前面各维之积）。
///
/// 列切分可以**不连续**（QKV 就是），所以不能简单当作结果：必须把第 r 段
/// 的第 j 列放回完整缓冲的第 `cols_of[r][j]` 列。
fn scatter_columns(
    concat: &[f32],
    cols_of: &[Vec<usize>],
    rows: usize,
    out_features: usize,
) -> Vec<f32> {
    let mut out = vec![0.0f32; rows * out_features];
    let mut off = 0usize;
    for cols in cols_of {
        let local = cols.len();
        for row in 0..rows {
            for (j, &c) in cols.iter().enumerate() {
                out[row * out_features + c] = concat[off + row * local + j];
            }
        }
        off += rows * local;
    }
    debug_assert_eq!(off, concat.len(), "拼接缓冲必须刚好被用完");
    out
}

/// [`scatter_columns`] 的逆运算（反向传播用）：从完整梯度里切出每 rank 那一份
fn gather_columns(
    grad: &[f32],
    cols_of: &[Vec<usize>],
    rows: usize,
    out_features: usize,
) -> Vec<Vec<f32>> {
    cols_of
        .iter()
        .map(|cols| {
            let mut out = Vec::with_capacity(rows * cols.len());
            for row in 0..rows {
                for &c in cols {
                    out.push(grad[row * out_features + c]);
                }
            }
            out
        })
        .collect()
}

/// 列并行线性层：`y = x @ W + b`，把**输出维**按列切开，每个 rank 只算自己那几列。
///
/// 前向：每个 rank 拿**完整输入**、算出自己那部分输出，**不需要任何通信**——
/// 输入是复制的、输出是切开的。需要完整输出时（后面接的不是行并行层）用
/// [`Self::forward`] 补一次 all_gather。
///
/// 反向的两条规则与 Megatron 一致：
/// - `dW`：各 rank 只算自己那几列，天然就是切开的，**无需通信**；
/// - `dx`：各 rank 只算出了"自己那几列对 x 的贡献"，必须**求和**才是完整的 `dx`。
///   本模块里这一步由自动微分对共享父张量 `x` 的累加天然完成（数值与一次 allreduce
///   完全相同），真机上就是 `f` 的反向 allreduce。
pub struct ColumnParallelLinear {
    /// 每 rank 负责的输出列（升序）。列可以**不连续**：QKV 靠这个按 head 对齐
    columns: Vec<Vec<usize>>,
    /// 每 rank 一片 `[in_features, columns[r].len()]`
    w_shards: Vec<Tensor>,
    /// 每 rank 一片 `[columns[r].len()]`
    b_shards: Vec<Tensor>,
    in_features: usize,
    out_features: usize,
}

impl ColumnParallelLinear {
    /// 用**单卡权重**切片构造：`w_full` 是 `[in_features, out_features]` 行优先。
    ///
    /// 真机上各 rank 只初始化自己那一片，但必须保证"拼起来 == 单卡那份权重"：
    /// 要么从同一份 checkpoint 切着加载，要么用同一随机种子按 rank 偏移取数。
    /// 各卡各随各的种子初始化，模型从第一步就已经不是同一个模型了。
    pub fn from_full(
        w_full: &[f32],
        b_full: &[f32],
        in_features: usize,
        out_features: usize,
        columns: Vec<Vec<usize>>,
    ) -> Self {
        assert_eq!(w_full.len(), in_features * out_features, "权重元素数应为 in×out");
        assert_eq!(b_full.len(), out_features, "偏置长度应为 out");
        check_columns(&columns, out_features);
        let w_shards = columns
            .iter()
            .map(|cols| {
                let mut data = Vec::with_capacity(in_features * cols.len());
                for row in 0..in_features {
                    for &c in cols {
                        data.push(w_full[row * out_features + c]);
                    }
                }
                Tensor::param(data, vec![in_features, cols.len()])
            })
            .collect();
        let b_shards = columns
            .iter()
            .map(|cols| {
                let data: Vec<f32> = cols.iter().map(|&c| b_full[c]).collect();
                Tensor::param(data, vec![cols.len()])
            })
            .collect();
        ColumnParallelLinear { columns, w_shards, b_shards, in_features, out_features }
    }

    /// 融合 QKV 的列并行层：`w_full` 为 `[d_model, 3·d_model]`（列序 q|k|v），
    /// 按 head 切成 `n_head / world_size` 份（见 [`qkv_head_columns`]）。
    pub fn qkv(
        d_model: usize,
        n_head: usize,
        world_size: usize,
        w_full: &[f32],
        b_full: &[f32],
    ) -> Self {
        let columns = qkv_head_columns(d_model, n_head, world_size);
        Self::from_full(w_full, b_full, d_model, 3 * d_model, columns)
    }

    pub fn world_size(&self) -> usize {
        self.columns.len()
    }

    /// rank `r` 负责的输出列
    pub fn columns(&self, r: usize) -> &[usize] {
        &self.columns[r]
    }

    /// rank `r` 这一片的输出宽度（= 它负责的列数）
    pub fn local_out(&self, r: usize) -> usize {
        self.columns[r].len()
    }

    pub fn weight_shard(&self, r: usize) -> &Tensor {
        &self.w_shards[r]
    }

    pub fn bias_shard(&self, r: usize) -> &Tensor {
        &self.b_shards[r]
    }

    /// 各 rank 的本地输出 `[.., local_out(r)]`，**不做任何通信**。
    ///
    /// 这是与行并行配对时的用法：下一层要的正好是"切开"的输入。
    pub fn forward_local(&self, x: &Tensor) -> Vec<Tensor> {
        let last = *x.shape().last().expect("输入至少要有 1 维");
        assert_eq!(last, self.in_features, "输入宽度应为 {}", self.in_features);
        (0..self.world_size())
            .map(|r| x.matmul(&self.w_shards[r]).add(&self.b_shards[r]))
            .collect()
    }

    /// 需要**完整输出**时的用法：本地算完再 all_gather 一次（N-1 轮）。
    ///
    /// 反向里这一步对应真机的 reduce-scatter（各 rank 把完整输出梯度里属于别人的那部分
    /// 送出去）；本模块里完整输出只有一份，于是退化成按列切片，数值完全一致。
    pub fn forward(&self, world: &mut World, x: &Tensor) -> Tensor {
        assert_eq!(world.size(), self.world_size(), "通信域大小与切分份数必须一致");
        let locals = self.forward_local(x);
        let data: Vec<Vec<f32>> = locals.iter().map(|t| t.data()).collect();
        let gathered = world.all_gather(&data);
        let rows = locals[0].numel() / self.local_out(0);
        let full = scatter_columns(&gathered[0], &self.columns, rows, self.out_features);
        let mut shape = locals[0].shape().to_vec();
        *shape.last_mut().unwrap() = self.out_features;

        let back: Vec<Tensor> = locals.clone();
        let columns = self.columns.clone();
        let out_features = self.out_features;
        Tensor::external(full, shape, locals, move |grad| {
            Box::new(move || {
                let g = grad.borrow();
                let parts = gather_columns(&g, &columns, rows, out_features);
                for (p, part) in back.iter().zip(&parts) {
                    p.accumulate_grad(part, 1.0);
                }
            })
        })
    }
}

/// 行并行线性层：`y = x @ W + b`，把**输入维**按行切开。
///
/// 它生来就是跟列并行配对的：上一层的列并行输出（已经按 rank 切开）正好是本层要的
/// 输入分片，于是**两层的接缝上不需要任何通信**。
///
/// 前向：各 rank 算 `x_r @ W_r` 得到**部分和**，必须 allreduce 求和才是完整输出——
/// 这是 TP 里唯一一次不可省的通信。反向则不需要通信：`y = Σ_r y_r`，
/// 所以 `∂L/∂y_r = ∂L/∂y`，上游梯度各 rank 各拿一份即可（真机上输出是 N 份复制品，
/// 上游梯度本来就已经人人一份）。
///
/// `b` 在 allreduce **之后**加：真机上各 rank 都加同一份复制品，数值与梯度都一致，
/// 本模块里只有一份，所以只加一次。
pub struct RowParallelLinear {
    /// 输入维切分，必须与上一层列并行的 `columns` 落在同一个 rank 上
    ranges: Vec<(usize, usize)>,
    /// 每 rank 一片 `[ranges[r].len(), out_features]`
    w_shards: Vec<Tensor>,
    bias: Tensor,
    out_features: usize,
}

impl RowParallelLinear {
    /// 用单卡权重按行区间切片构造
    pub fn from_full(
        w_full: &[f32],
        b_full: &[f32],
        in_features: usize,
        out_features: usize,
        ranges: Vec<(usize, usize)>,
    ) -> Self {
        assert_eq!(w_full.len(), in_features * out_features, "权重元素数应为 in×out");
        assert_eq!(b_full.len(), out_features, "偏置长度应为 out");
        check_input_ranges(&ranges, in_features);
        let w_shards = ranges
            .iter()
            .map(|(s, e)| {
                let rows = e - s;
                let mut data = Vec::with_capacity(rows * out_features);
                for row in *s..*e {
                    data.extend_from_slice(&w_full[row * out_features..(row + 1) * out_features]);
                }
                Tensor::param(data, vec![rows, out_features])
            })
            .collect();
        RowParallelLinear {
            ranges,
            w_shards,
            bias: Tensor::param(b_full.to_vec(), vec![out_features]),
            out_features,
        }
    }

    /// 输入维连续均分（常规用法）
    pub fn contiguous(
        w_full: &[f32],
        b_full: &[f32],
        in_features: usize,
        out_features: usize,
        world_size: usize,
    ) -> Self {
        Self::from_full(
            w_full,
            b_full,
            in_features,
            out_features,
            contiguous_ranges(in_features, world_size),
        )
    }

    pub fn world_size(&self) -> usize {
        self.ranges.len()
    }

    pub fn input_range(&self, r: usize) -> (usize, usize) {
        self.ranges[r]
    }

    pub fn weight_shard(&self, r: usize) -> &Tensor {
        &self.w_shards[r]
    }

    pub fn bias(&self) -> &Tensor {
        &self.bias
    }

    /// 前向：`x_shards[r]` 是 rank `r` 那份输入分片 `[.., width_r]`，
    /// 返回**完整**输出 `[.., out_features]`（N 份复制品，内容相同）
    pub fn forward(&self, world: &mut World, x_shards: &[Tensor]) -> Tensor {
        let n = self.world_size();
        assert_eq!(world.size(), n, "通信域大小与切分份数必须一致");
        assert_eq!(x_shards.len(), n, "输入分片数应为 {n}");
        let mut partials = Vec::with_capacity(n);
        for (r, x) in x_shards.iter().enumerate() {
            let (s, e) = self.ranges[r];
            let width = *x.shape().last().expect("输入至少要有 1 维");
            assert_eq!(width, e - s, "rank {r} 的输入宽度应为 {}", e - s);
            partials.push(x.matmul(&self.w_shards[r]));
        }
        // 各 rank 手里只有"部分和"，这一步就是**唯一一次不可省的 allreduce**
        let locals: Vec<Vec<f32>> = partials.iter().map(|p| p.data()).collect();
        let summed = world.all_reduce_sum(&locals);
        let shape = partials[0].shape().to_vec();
        debug_assert_eq!(*shape.last().unwrap(), self.out_features);

        let back: Vec<Tensor> = partials.clone();
        let y = Tensor::external(summed[0].clone(), shape, partials, move |grad| {
            Box::new(move || {
                let g = grad.borrow();
                for p in &back {
                    p.accumulate_grad(&g, 1.0);
                }
            })
        });
        y.add(&self.bias)
    }
}

/// 张量并行的 MLP：`z = gelu(x W1 + b1) W2 + b2`，两层分别是列并行与行并行。
///
/// 整块只通信一次：
///
/// ```text
/// x（每 rank 一份完整复制）
///   └─ W1 列并行 ──→ 各 rank 算出自己的 hidden 分片      （接缝：零通信）
///                     └─ gelu ──→ ─ W2 行并行 ──→ allreduce 求和 ──→ z（完整，N 份复制品）
/// ```
pub struct TensorParallelMlp {
    fc1: ColumnParallelLinear,
    fc2: RowParallelLinear,
    world: World,
}

impl TensorParallelMlp {
    /// `w1`/`b1` 是 `[d_model, hidden]` 的第一层权重（列并行），
    /// `w2`/`b2` 是 `[hidden, d_model]` 的第二层权重（行并行）
    pub fn from_full(
        w1: &[f32],
        b1: &[f32],
        w2: &[f32],
        b2: &[f32],
        d_model: usize,
        hidden: usize,
        world_size: usize,
    ) -> Self {
        assert_eq!(b1.len(), hidden, "隐藏层偏置长度应为 hidden");
        let fc1 = ColumnParallelLinear::from_full(
            w1,
            b1,
            d_model,
            hidden,
            contiguous_columns(hidden, world_size),
        );
        let fc2 = RowParallelLinear::contiguous(w2, b2, hidden, d_model, world_size);
        // 两层必须切在**同一处**：列并行输出的第 j 列就是行并行输入的第 j 行。
        // 这里错开一格不会 panic，只会让每张卡都用别人的激活去乘自己的权重——
        // 前向照样出数、loss 照样下降，只是收敛到一个错的模型。
        for r in 0..world_size {
            let (s, e) = fc2.input_range(r);
            assert_eq!(
                fc1.local_out(r),
                e - s,
                "rank {r} 上列并行的输出分片（{}）与行并行的输入分片（{}）不一样大",
                fc1.local_out(r),
                e - s
            );
        }
        TensorParallelMlp { fc1, fc2, world: World::new(world_size) }
    }

    pub fn log(&self) -> &CommLog {
        self.world.log()
    }

    pub fn fc1(&self) -> &ColumnParallelLinear {
        &self.fc1
    }

    pub fn fc2(&self) -> &RowParallelLinear {
        &self.fc2
    }

    /// 返回完整输出（各 rank 上都有相同的一份）
    pub fn forward(&mut self, x: &Tensor) -> Tensor {
        let local = self.fc1.forward_local(x);
        let act: Vec<Tensor> = local.iter().map(|t| t.gelu()).collect();
        self.fc2.forward(&mut self.world, &act)
    }
}

// ==================== 5. 流水线并行 ====================

/// 一个 micro-batch 在某一台上做的一件事
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Step {
    /// 前向：`(micro_batch, stage)`
    Forward(usize, usize),
    /// 反向：`(micro_batch, stage)`
    Backward(usize, usize),
}

/// 调度计划，以及它真正决定的东西——**显存里同时压着多少激活**。
///
/// GPipe 与 1F1B 算出来的**梯度完全一样**，区别只在这两个峰值上：
///
/// - GPipe：把所有 micro-batch 的前向跑完才开始反向，于是 M 个 micro-batch 的激活
///   从头到尾都压在显存里，峰值 = M；
/// - 1F1B（one-forward-one-backward）：填满流水线后，每进来一个新的前向就送走一个
///   **最老**的反向，峰值只由**段数** p 决定（≈ p）。
///
/// 这就是"流水线并行必须配 1F1B"的全部理由：micro-batch 数 M 是**必须**开大的
/// （要填满流水线、要让首尾气泡的占比可接受），而 GPipe 的显存正比于 M。
/// 段数 p 由模型大小决定、micro-batch 由 batch 大小决定，都改不动——
/// 能改的只有调度。
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Schedule {
    pub micro_batches: usize,
    pub stages: usize,
    /// 事件表（执行顺序）：同一时隙内不同 stage 的事件可以并行
    pub steps: Vec<Step>,
    /// 峰值：同时"前向已算、反向未算"的 micro-batch 数
    pub peak_in_flight: usize,
    /// 峰值：同时驻留的**跨阶段激活缓冲**数 = in_flight × (段数 - 1)
    pub peak_boundary_buffers: usize,
}

impl Schedule {
    /// GPipe：先跑完所有 micro-batch 的前向，再依次反向
    pub fn gpipe(micro_batches: usize, stages: usize) -> Self {
        assert!(micro_batches > 0 && stages > 0, "micro-batch 数与段数都必须为正");
        let mut steps = Vec::with_capacity(2 * micro_batches * stages);
        for i in 0..micro_batches {
            for s in 0..stages {
                steps.push(Step::Forward(i, s));
            }
        }
        for i in (0..micro_batches).rev() {
            for s in (0..stages).rev() {
                steps.push(Step::Backward(i, s));
            }
        }
        Self::analyze(micro_batches, stages, steps)
    }

    /// 1F1B：先填 `p-1` 个 micro-batch 的前向（暖机），之后每做一个新的前向
    /// 就送走一个最老的反向（稳态），最后把剩下的反向做完（收尾）。
    ///
    /// 稳态里的"前向进来一个、反向送走一个"就是它的名字。注意 `M <= p-1` 时
    /// 没有可重叠的活，1F1B 退化成 GPipe——所以用 1F1B 的前提是 M 至少和 p 同量级。
    pub fn one_forward_one_backward(micro_batches: usize, stages: usize) -> Self {
        assert!(micro_batches > 0 && stages > 0, "micro-batch 数与段数都必须为正");
        let mut steps = Vec::with_capacity(2 * micro_batches * stages);
        let warm = (stages - 1).min(micro_batches);
        let mut oldest = 0usize; // 下一个该被送走的（最老的）micro-batch
        for i in 0..warm {
            for s in 0..stages {
                steps.push(Step::Forward(i, s));
            }
        }
        for i in warm..micro_batches {
            for s in 0..stages {
                steps.push(Step::Forward(i, s));
            }
            if i >= stages - 1 {
                for s in (0..stages).rev() {
                    steps.push(Step::Backward(oldest, s));
                }
                oldest += 1;
            }
        }
        for i in oldest..micro_batches {
            for s in (0..stages).rev() {
                steps.push(Step::Backward(i, s));
            }
        }
        Self::analyze(micro_batches, stages, steps)
    }

    /// 从事件表推出资源画像。两种调度共用这一份口径，所以"谁更省"是可比的。
    fn analyze(micro_batches: usize, stages: usize, steps: Vec<Step>) -> Self {
        // 计入/释放都发生在**第一个**前向事件与**第一个**反向事件上：
        // 一个 micro-batch 的激活从它进流水线开始占显存，到它的反向开始为止
        let mut live = 0usize;
        let (mut peak_in_flight, mut peak_boundary_buffers) = (0usize, 0usize);
        let mut in_flight = vec![false; micro_batches];
        let mut seen_fwd = vec![0usize; micro_batches];
        let mut seen_bwd = vec![0usize; micro_batches];
        for step in &steps {
            match *step {
                Step::Forward(i, s) => {
                    assert!(s < stages, "段下标 {s} 越界");
                    if seen_fwd[i] == 0 {
                        in_flight[i] = true;
                        live += 1;
                    }
                    seen_fwd[i] += 1;
                }
                Step::Backward(i, s) => {
                    assert!(s < stages, "段下标 {s} 越界");
                    if seen_bwd[i] == 0 {
                        assert!(in_flight[i], "第 {i} 个 micro-batch 还没进流水线就先反向");
                        in_flight[i] = false;
                        live -= 1;
                    }
                    seen_bwd[i] += 1;
                }
            }
            peak_in_flight = peak_in_flight.max(live);
            peak_boundary_buffers = peak_boundary_buffers.max(live * stages.saturating_sub(1));
        }
        // 每个 micro-batch 在每段上前向、反向各**恰好一次**：少一次就是有段没算，
        // 多一次就是重复累加梯度，两种都不会报错，只会把模型悄悄训歪
        for i in 0..micro_batches {
            assert_eq!(seen_fwd[i], stages, "第 {i} 个 micro-batch 的前向次数应等于段数");
            assert_eq!(seen_bwd[i], stages, "第 {i} 个 micro-batch 的反向次数应等于段数");
        }
        assert_eq!(live, 0, "调度跑完时所有 micro-batch 的激活都应已释放");
        Schedule { micro_batches, stages, steps, peak_in_flight, peak_boundary_buffers }
    }
}

/// 一段流水线 = 一层。
///
/// 真实系统里一段是"模型里连续的一大块层"（stage 0 前 12 层、stage 1 后 12 层），
/// 但**调度**的行为跟段内有多少层无关，所以这里一段只留一层，让事件表一眼能读懂。
pub struct Pipeline {
    stages: Vec<Linear>,
    schedule: Schedule,
}

/// 一趟流水线执行的结果
pub struct RunReport {
    /// 各 micro-batch 最后一段的输出，按 micro-batch 原顺序
    pub outputs: Vec<Tensor>,
    /// 实测峰值：同时驻留的 micro-batch 数
    pub peak_in_flight: usize,
    /// 实测峰值：同时驻留的跨阶段激活缓冲数
    pub peak_boundary_buffers: usize,
}

impl Pipeline {
    pub fn new(stages: Vec<Linear>, schedule: Schedule) -> Self {
        assert_eq!(stages.len(), schedule.stages, "层数与调度里的段数不一致");
        Pipeline { stages, schedule }
    }

    pub fn schedule(&self) -> &Schedule {
        &self.schedule
    }

    pub fn stages(&self) -> &[Linear] {
        &self.stages
    }

    /// 按调度表跑一遍。
    ///
    /// `inputs[i]` 是第 i 个 micro-batch 的输入，`loss_of` 在该 micro-batch 走到最后
    /// 一段时被调用一次，返回**已经按比例缩好的标量损失**（各 micro-batch 的损失之
    /// 和应等于整个 batch 的损失，见单测）。
    ///
    /// 反向上有个单进程模拟的边界：自动微分一次 `backward()` 就把整条链走完，
    /// 没法真按段切开。所以这里在某个 micro-batch 的**第一个**反向事件上执行整条链
    /// 的反向，并立刻释放它的全部激活。驻留量的**峰值**两种口径一致
    /// （真机是逐段释放，开始释放的时刻相同），而峰值正是本课要比的东西。
    pub fn run(&self, inputs: &[Tensor], loss_of: impl Fn(&Tensor) -> Tensor) -> RunReport {
        let (m, p) = (self.schedule.micro_batches, self.schedule.stages);
        assert_eq!(inputs.len(), m, "micro-batch 个数应为 {m}");
        // acts[i][s] = 第 i 个 micro-batch 从第 s 段出去的那块激活
        let mut acts: Vec<Vec<Option<Tensor>>> = vec![vec![None; p]; m];
        let mut loss: Vec<Option<Tensor>> = vec![None; m];
        let mut out: Vec<Option<Tensor>> = vec![None; m];
        let mut in_flight = vec![false; m];
        let mut live = 0usize;
        let (mut peak_in_flight, mut peak_boundary_buffers) = (0usize, 0usize);

        for step in &self.schedule.steps {
            match *step {
                Step::Forward(i, s) => {
                    if !in_flight[i] {
                        in_flight[i] = true;
                        live += 1;
                    }
                    let x = if s == 0 {
                        inputs[i].clone()
                    } else {
                        acts[i][s - 1].clone().expect("上一段的激活还没算出来")
                    };
                    let y = self.stages[s].forward(&x);
                    if s + 1 == p {
                        loss[i] = Some(loss_of(&y));
                        out[i] = Some(y.clone());
                    }
                    acts[i][s] = Some(y);
                }
                Step::Backward(i, s) => {
                    if in_flight[i] {
                        // 调度保证一个 micro-batch 的第一个反向事件来自最后一段
                        assert_eq!(s + 1, p, "反向必须从最后一段开始，却收到第 {s} 段");
                        loss[i].clone().expect("还没前向就先反向").backward();
                        acts[i] = vec![None; p];
                        in_flight[i] = false;
                        live -= 1;
                    }
                    // 后续的 (i, s < p-1) 事件在单进程里没有对应动作：
                    // 整条链的反向上面已经一次做完了
                }
            }
            peak_in_flight = peak_in_flight.max(live);
            peak_boundary_buffers = peak_boundary_buffers.max(live * (p - 1));
        }

        RunReport {
            outputs: out.into_iter().map(|o| o.expect("每个 micro-batch 都应跑完")).collect(),
            peak_in_flight,
            peak_boundary_buffers,
        }
    }
}

// ==================== 6. 3D 并行 ====================

/// 3D 并行的切分配置：三个轴各切多少份。
///
/// 三个轴**正交**，各自解决一个不同的问题，谁都不替代谁：
///
/// | 轴 | 切什么 | 省什么 | 代价 |
/// |----|--------|--------|------|
/// | `dp` | 数据（每个副本吃不同的样本） | 时间：N 份数据并行算 | 每步一次梯度 allreduce |
/// | `tp` | 层内的权重（列/行） | 单层放不下的显存 | 每层两次 allreduce，通信最频繁 |
/// | `pp` | 层（前几层在这张卡、后几层在那张卡） | 整个模型放不下的显存 | 首尾气泡 + micro-batch 调度 |
///
/// 一个 rank 同时属于三个组：TP 组（层内通信）、DP 组（梯度同步）、PP 组（传激活）。
/// 真实集群里 `dp · tp · pp = 卡数`，谁是几取决于**网络**：TP 通信最密，优先放在
/// 同一台机器（NVLink）里；pp 只传边界激活；dp 每步一次、可以走最慢的链路。
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct DistConfig {
    pub dp: usize,
    pub tp: usize,
    pub pp: usize,
}

/// rank 在三维网格里的坐标
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub struct RankCoord {
    pub dp: usize,
    pub tp: usize,
    pub pp: usize,
}

impl DistConfig {
    pub fn new(dp: usize, tp: usize, pp: usize) -> Self {
        assert!(dp > 0 && tp > 0 && pp > 0, "三个轴都必须至少切 1 份");
        DistConfig { dp, tp, pp }
    }

    pub fn world_size(&self) -> usize {
        self.dp * self.tp * self.pp
    }

    /// rank 编号 → 三维坐标。
    ///
    /// 约定与 Megatron 一致：**tp 放在最低位**（`rank % tp`）。于是同一个 TP 组
    /// （需要做最频繁通信的那几个 rank）的编号是连着的，调度器把它们放到同一台机器
    /// 或同一个 NVLink 域里就行；把 dp 放低位的话，TP 队友会散在整个集群里，
    /// 每层那两次 allreduce 都要跨机，通信立刻成为瓶颈。
    pub fn coord(&self, rank: usize) -> RankCoord {
        assert!(rank < self.world_size(), "rank {rank} 超出通信域大小 {}", self.world_size());
        RankCoord {
            tp: rank % self.tp,
            dp: (rank / self.tp) % self.dp,
            pp: rank / (self.tp * self.dp),
        }
    }

    /// 坐标 → rank 编号（[`Self::coord`] 的逆）
    pub fn rank_of(&self, c: RankCoord) -> usize {
        (c.pp * self.dp + c.dp) * self.tp + c.tp
    }

    /// 与本 rank 同属一个 TP 组的 rank：dp/pp 坐标相同、tp 坐标走满一圈。
    /// **层内通信只在这一组里发生**。
    pub fn tp_group(&self, rank: usize) -> Vec<usize> {
        let c = self.coord(rank);
        (0..self.tp).map(|tp| self.rank_of(RankCoord { tp, ..c })).collect()
    }

    /// 与本 rank 同属一个 DP 组的 rank：tp/pp 坐标相同、dp 坐标走满一圈。
    /// **梯度 allreduce 只在这一组里发生**：它们持有完全相同的模型分片，
    /// 吃的是不同的数据。
    pub fn dp_group(&self, rank: usize) -> Vec<usize> {
        let c = self.coord(rank);
        (0..self.dp).map(|dp| self.rank_of(RankCoord { dp, ..c })).collect()
    }

    /// 本 rank 流水线里的上游 stage（`None` = 第一段，输入直接来自数据）
    pub fn prev_stage(&self, rank: usize) -> Option<usize> {
        let c = self.coord(rank);
        if c.pp == 0 {
            None
        } else {
            Some(self.rank_of(RankCoord { pp: c.pp - 1, ..c }))
        }
    }

    /// 本 rank 流水线里的下游 stage（`None` = 最后一段，输出直接接损失）
    pub fn next_stage(&self, rank: usize) -> Option<usize> {
        let c = self.coord(rank);
        if c.pp + 1 == self.pp {
            None
        } else {
            Some(self.rank_of(RankCoord { pp: c.pp + 1, ..c }))
        }
    }

    /// 某个 rank 到底负责什么——把"3D 切分"落成一张可读可断言的清单。
    ///
    /// `layers` / `columns` / `batch` 是这次切分的三个虚拟维度：`layers` 是流水线要
    /// 切的层数，`columns` 是某个权重矩阵的输出维，`batch` 是全局 batch 大小。
    /// 真实模型里对每个权重张量各算一次即可（`layers` 换成"参数列表的区间"）。
    ///
    /// 三个轴各自切一刀，`plan` 就是把这三刀的结果放在一起：
    /// - `layers` ← pp 轴（[`chunk_range`]）
    /// - `columns` ← tp 轴（[`contiguous_columns`]）
    /// - `batch` ← dp 轴（[`chunk_range`]）
    ///
    /// 注意**参数不按 dp 切**：同一 (tp, pp) 坐标上的 dp 个 rank 持有的是
    /// 同一份参数分片（复制），它们只是各自吃一段数据。3D 并行的"不重不漏"
    /// 因此要说清楚是**哪个轴上的**：参数在 (tp, pp) 网格上不重不漏，
    /// 数据在 dp 轴上不重不漏。
    pub fn plan(&self, rank: usize, layers: usize, columns: usize, batch: usize) -> RankPlan {
        let c = self.coord(rank);
        let layer_range = chunk_range(c.pp, layers, self.pp);
        let col_split = contiguous_columns(columns, self.tp);
        RankPlan {
            rank,
            coord: c,
            batch: chunk_range(c.dp, batch, self.dp),
            layers: layer_range,
            columns: col_split[c.tp].clone(),
        }
    }
}

/// 一个 rank 的分工清单
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RankPlan {
    pub rank: usize,
    pub coord: RankCoord,
    /// 本 rank 吃的样本区间（dp 轴切分出来的那一份）
    pub batch: (usize, usize),
    /// 本 rank 持有的层区间（pp 轴切分）
    pub layers: (usize, usize),
    /// 本 rank 在本层里负责的输出列（tp 轴切分）
    pub columns: Vec<usize>,
}

impl RankPlan {
    /// 一行式分工报告
    pub fn summary(&self, cfg: &DistConfig) -> String {
        let c = self.coord;
        let cols = match (self.columns.first(), self.columns.last()) {
            (Some(a), Some(b)) => format!("{a}..={b}（{} 列）", self.columns.len()),
            _ => "无".to_string(),
        };
        format!(
            "rank {}/{} [dp {} tp {} pp {}]：层 [{}, {})、列 {}、样本 [{}, {})；\
             TP 组 {} 个、DP 组 {} 个、上游 stage {}",
            self.rank,
            cfg.world_size(),
            c.dp,
            c.tp,
            c.pp,
            self.layers.0,
            self.layers.1,
            cols,
            self.batch.0,
            self.batch.1,
            cfg.tp_group(self.rank).len(),
            cfg.dp_group(self.rank).len(),
            match cfg.prev_stage(self.rank) {
                Some(r) => r.to_string(),
                None => "无（第一段）".to_string(),
            }
        )
    }
}

// ==================== 单测 ====================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::loss::cross_entropy_loss_masked;
    use crate::model::{GPT, GPTConfig};
    use crate::module::Module;
    use crate::optim::{AdamW, Optimizer, SGD};
    use crate::rng::Rng;

    /// 造 `n` 份长度为 `len` 的随机本地数据
    fn random_locals(n: usize, len: usize, seed: u64) -> Vec<Vec<f32>> {
        let mut rng = Rng::new(seed);
        (0..n).map(|_| (0..len).map(|_| rng.randn()).collect()).collect()
    }

    /// 按位求和
    fn sum_of(locals: &[Vec<f32>]) -> Vec<f32> {
        let mut acc = locals[0].clone();
        for l in &locals[1..] {
            for (a, b) in acc.iter_mut().zip(l) {
                *a += b;
            }
        }
        acc
    }

    fn assert_close(got: &[f32], want: &[f32], tol: f32, what: &str) {
        assert_eq!(got.len(), want.len(), "{what}：长度 {} vs {}", got.len(), want.len());
        for (i, (g, w)) in got.iter().zip(want).enumerate() {
            assert!(
                (g - w).abs() <= tol,
                "{what} 第 {i} 个元素不符：{g} vs {w}（容差 {tol}）"
            );
        }
    }

    #[test]
    fn test_ring_all_reduce_matches_rank_sum() {
        // 覆盖 1、2 的幂（2/4/8）与非 2 的幂（3/5）rank 数；len 取不能整除的值，
        // 顺带验证"余数摊在前面"的切块在两个阶段里始终对齐。
        for n in [1usize, 2, 3, 4, 5, 8] {
            for len in [len_for(n), 13, 12] {
                let locals = random_locals(n, len, 1000 + n as u64 * 31 + len as u64);
                let want = sum_of(&locals);
                let mut world = World::new(n);
                let got = world.all_reduce_sum(&locals);
                assert_eq!(got.len(), n);
                for (r, g) in got.iter().enumerate() {
                    assert_close(g, &want, 1e-6, &format!("n={n} len={len} rank {r} 的 allreduce"));
                    // 各 rank 的结果必须**逐位**相同，否则后续参数更新会分叉
                    assert_eq!(g, &got[0], "n={n} len={len} 时 rank {r} 的结果与 rank 0 不一致");
                }
            }
        }
    }

    /// 选一个能制造余数的长度
    fn len_for(n: usize) -> usize {
        if n % 2 == 0 { 10 } else { 11 }
    }

    #[test]
    fn test_ring_all_reduce_traffic_is_balanced_naive_is_not() {
        let n = 4usize;
        let len = 400usize;
        let locals = random_locals(n, len, 7);

        let mut world = World::new(n);
        let ring = world.all_reduce_sum(&locals);
        let ring_log = world.log().clone();

        // 环上每轮每 rank 发一块 ≈ len/N，两阶段各 N-1 轮
        let chunk = len / n;
        assert_eq!(
            ring_log.per_rank,
            vec![2 * (n - 1) * chunk; n],
            "环形 allreduce 各 rank 的发送量必须完全相等"
        );
        assert_eq!(ring_log.rounds, 2 * (n - 1));
        assert_eq!(ring_log.max_rank_bytes(), 2 * (n - 1) * chunk * 4);

        let naive = world.all_reduce_sum_naive(&locals);
        let naive_log = world.log().clone();
        assert_eq!(naive_log.per_rank[0], (n - 1) * len, "主从广播里 rank 0 要下发 N-1 份");
        assert_eq!(naive_log.per_rank[1], len);
        assert_eq!(
            naive_log.max_rank_bytes(),
            (n - 1) * len * 4,
            "主从广播的瓶颈 rank 承担了 (N-1) 倍通信量"
        );
        // 倾斜倍数：(N-1)·len ÷ (2(N-1)/N·len) = N/2
        assert_eq!(
            naive_log.max_rank_bytes(),
            ring_log.max_rank_bytes() * n / 2,
            "主从广播的瓶颈 rank 通信量应是环形的 N/2 倍：{} vs {}",
            naive_log.max_rank_bytes(),
            ring_log.max_rank_bytes()
        );

        // 两种实现结果一致
        assert_close(&ring[0], &naive[0], 1e-6, "环形与主从广播的 allreduce");
        assert_close(&ring[0], &sum_of(&locals), 1e-6, "环形 allreduce 与直接求和");
    }

    #[test]
    fn test_reduce_scatter_keeps_one_chunk_per_rank() {
        let n = 3usize;
        let len = 12usize;
        let locals = random_locals(n, len, 99);
        let want = sum_of(&locals);
        let mut world = World::new(n);
        let owned = world.reduce_scatter_sum(&locals);
        assert_eq!(owned.len(), n);
        let mut seen = Vec::new();
        for (r, (c, data)) in owned.iter().enumerate() {
            assert_eq!(*c, (r + 1) % n, "rank {r} 应持有第 {} 块", (r + 1) % n);
            let (s, e) = chunk_range(*c, len, n);
            assert_close(data, &want[s..e], 1e-6, &format!("rank {r} 持有的第 {c} 块"));
            seen.push(*c);
        }
        seen.sort();
        assert_eq!(seen, vec![0, 1, 2], "N 块必须不重不漏地分给 N 个 rank");
        // 只有 N-1 轮（没有 all-gather 那一段）
        assert_eq!(world.log().rounds, n - 1);
    }

    #[test]
    fn test_all_gather_concatenates_in_rank_order() {
        let n = 4usize;
        let locals = random_locals(n, 5, 5);
        let mut world = World::new(n);
        let got = world.all_gather(&locals);
        let want: Vec<f32> = locals.iter().flatten().copied().collect();
        for (r, g) in got.iter().enumerate() {
            assert_close(g, &want, 1e-6, &format!("rank {r} 的 all_gather"));
            assert_eq!(g, &got[0]);
        }
        assert_eq!(world.log().rounds, n - 1);
    }

    #[test]
    fn test_broadcast_delivers_source_payload() {
        for n in [1usize, 2, 5] {
            let mut locals = random_locals(n, 7, 3);
            // 非 src 的 rank 传垃圾数据，验证广播只认 src
            for r in 0..n {
                if r != n - 1 {
                    locals[r] = vec![f32::NAN; 7];
                }
            }
            let src = n - 1;
            let mut world = World::new(n);
            let got = world.broadcast(src, &locals);
            for (r, g) in got.iter().enumerate() {
                assert_eq!(g, &locals[src], "rank {r} 收到的广播内容应等于 rank {src} 的数据");
            }
            if n > 1 {
                assert_eq!(world.log().rounds, n - 1);
                // 逐跳传递：每轮只有一趟发送，总量 = (N-1) 份
                assert_eq!(world.log().total_sent(), (n - 1) * 7);
            }
        }
    }

    #[test]
    fn test_barrier_marks_sync_point_and_resets_log() {
        let mut world = World::new(2);
        let locals = random_locals(2, 4, 1);
        world.all_reduce_sum(&locals);
        assert!(world.log().total_sent() > 0);
        world.barrier();
        assert_eq!(world.barriers(), 1);
        assert_eq!(world.log(), &CommLog::new(2), "屏障之后是新阶段，通信量重新计数");
    }

    // ---------- 数据并行 ----------

    /// 小 GPT，同种子 → 同初值（DP 的多份"副本"靠这个构造）
    fn tiny_gpt(vocab: usize, seed: u64) -> GPT {
        let cfg = GPTConfig {
            n_embd: 16,
            n_head: 2,
            n_layer: 2,
            block_size: 32,
            ..GPTConfig::tiny(vocab)
        };
        GPT::new(cfg, &mut Rng::new(seed))
    }

    /// 一批随机 token id（语言模型的损失只要求 id 合法）
    fn random_ids(count: usize, vocab: usize, seed: u64) -> Vec<usize> {
        let mut rng = Rng::new(seed);
        (0..count).map(|_| rng.choice(vocab)).collect()
    }

    /// 每个位置的目标：错位一位取下一个 token（最后一位回绕到首位）。
    ///
    /// 刻意做成**整体可切分**的纯函数：`targets[i]` 只由 `ids` 决定，不依赖"自己在第几片"。
    /// 若写成"每片最后一位复用自身"（分片内部自洽、但和整批不一致），DP 与单卡的对拍
    /// 会在分片边界上差出固定偏差——不是浮点误差，而是两边的损失函数根本不是同一个。
    fn next_targets(ids: &[usize]) -> Vec<usize> {
        (0..ids.len()).map(|i| ids[(i + 1) % ids.len()]).collect()
    }

    /// 一个 batch 的平均交叉熵
    fn batch_loss(model: &GPT, ids: &[usize], targets: &[usize], b: usize, t: usize) -> Tensor {
        assert_eq!(ids.len(), b * t, "ids 个数应与 b×t 一致");
        assert_eq!(targets.len(), b * t, "targets 个数应与 b×t 一致");
        let logits = model.forward(ids, b, t, None, false);
        cross_entropy_loss_masked(&logits, targets, None)
    }

    #[test]
    fn test_data_parallel_matches_single_process_full_batch() {
        let (vocab, t, total_b, world_size) = (24usize, 8usize, 4usize, 2usize);
        let shard_b = total_b / world_size;
        let ids = random_ids(total_b * t, vocab, 42);
        let targets = next_targets(&ids);
        let seed = 7u64;

        // 参照组：单进程，整个 batch 一次算完、更新一步
        let full = tiny_gpt(vocab, seed);
        let full_params = full.parameters();
        let mut opt_full = SGD::new(0.05, full_params.clone());
        opt_full.zero_grad();
        batch_loss(&full, &ids, &targets, total_b, t).backward();
        opt_full.step();

        // DP 组：两份**同初值**副本，各吃一半数据
        let replicas: Vec<GPT> = (0..world_size).map(|_| tiny_gpt(vocab, seed)).collect();
        for r in 1..world_size {
            assert_eq!(
                replicas[r].parameters()[0].data(),
                replicas[0].parameters()[0].data(),
                "副本 {r} 的初值与 rank 0 不同，后续对拍无从谈起"
            );
        }
        let per_rank: Vec<Vec<Tensor>> = replicas.iter().map(|m| m.parameters()).collect();
        for r in 0..world_size {
            let lo = r * shard_b * t;
            let hi = lo + shard_b * t;
            batch_loss(&replicas[r], &ids[lo..hi], &targets[lo..hi], shard_b, t).backward();
        }

        let mut dp = DataParallel::new(world_size);
        dp.sync_gradients(&per_rank, 1.0);
        assert_eq!(dp.log().rounds, 2 * (world_size - 1), "一次梯度同步 = 两阶段环形 allreduce");

        // 同步后的梯度应与"单卡全 batch 一次算完"的梯度一致
        assert_close(
            &flatten_grads(&per_rank[0]),
            &flatten_grads(&full_params),
            1e-5,
            "DP 平均后的梯度 vs 单卡全 batch 的梯度",
        );

        // 各 rank 用自己的优化器更新一步（同构 DP 下更新完全一样）
        for params in &per_rank {
            let mut opt = SGD::new(0.05, params.clone());
            opt.step();
        }
        for (i, (a, b)) in full_params.iter().zip(&per_rank[0]).enumerate() {
            assert_close(&b.data(), &a.data(), 1e-5, &format!("一步更新后第 {i} 个参数"));
        }
        for r in 1..world_size {
            for (i, p) in per_rank[r].iter().enumerate() {
                assert_eq!(
                    p.data(),
                    per_rank[0][i].data(),
                    "更新之后副本 {r} 的第 {i} 个参数与 rank 0 不一致——模型已经分叉"
                );
            }
        }
    }

    #[test]
    fn test_data_parallel_local_scale_matches_gradient_accumulation() {
        // `local_scale` 的语义：本地累加 k 个 micro-step 的梯度后，传 1/k 应等于
        // "把这 k 份数据当成一个大 batch 直接算"的梯度。用单 rank 通信域验证
        // （通信是恒等操作），从而把这一条与 allreduce 的数值行为解耦开。
        let (vocab, t) = (24usize, 8usize);
        let ids = random_ids(4 * t, vocab, 11);
        let targets = next_targets(&ids);
        let seed = 5u64;

        let merged = tiny_gpt(vocab, seed);
        batch_loss(&merged, &ids, &targets, 4, t).backward();
        let want = flatten_grads(&merged.parameters());

        let acc = tiny_gpt(vocab, seed);
        let params = acc.parameters();
        for half in 0..2 {
            let lo = half * 2 * t;
            batch_loss(&acc, &ids[lo..lo + 2 * t], &targets[lo..lo + 2 * t], 2, t).backward();
        }
        let mut dp = DataParallel::new(1);
        dp.sync_gradients(std::slice::from_ref(&params), 0.5);
        assert_close(
            &flatten_grads(&params),
            &want,
            1e-5,
            "本地累加 2 次（scale = 1/2）vs 合并成一个大 batch",
        );
    }

    // ---------- ZeRO ----------

    #[test]
    fn test_zero_matches_vanilla_dp_trajectory_and_shrinks_state() {
        let (vocab, t, total_b, n) = (24usize, 8usize, 4usize, 2usize);
        let steps = 4usize;
        let shard_b = total_b / n;
        // 打开权重衰减：分片实现里若忘了把真实 θ 灌给分片优化器，
        // 衰减项就会静默失效，只有带 wd 的轨迹对拍才能抓到
        let (lr, wd, seed) = (0.05f32, 0.01f32, 21u64);
        let ids = random_ids(total_b * t, vocab, 123);
        let targets = next_targets(&ids);

        // 参照组：vanilla DP（每个 rank 一份完整优化器状态）
        let ref_losses: Vec<f32> = {
            let replicas: Vec<GPT> = (0..n).map(|_| tiny_gpt(vocab, seed)).collect();
            let per_rank: Vec<Vec<Tensor>> = replicas.iter().map(|m| m.parameters()).collect();
            let mut opts: Vec<AdamW> =
                per_rank.iter().map(|p| AdamW::new(lr, p.clone(), wd)).collect();
            let mut dp = DataParallel::new(n);
            let mut losses = Vec::new();
            for _ in 0..steps {
                for r in 0..n {
                    for p in &per_rank[r] {
                        p.zero_grad();
                    }
                    let lo = r * shard_b * t;
                    batch_loss(
                        &replicas[r],
                        &ids[lo..lo + shard_b * t],
                        &targets[lo..lo + shard_b * t],
                        shard_b,
                        t,
                    )
                    .backward();
                }
                dp.sync_gradients(&per_rank, 1.0);
                for o in opts.iter_mut() {
                    o.step();
                }
                losses.push(batch_loss(&replicas[0], &ids, &targets, total_b, t).item());
            }
            losses
        };

        for stage in [ZeroStage::One, ZeroStage::Two] {
            let replicas: Vec<GPT> = (0..n).map(|_| tiny_gpt(vocab, seed)).collect();
            let per_rank: Vec<Vec<Tensor>> = replicas.iter().map(|m| m.parameters()).collect();
            let total: usize = per_rank[0].iter().map(|p| p.numel()).sum();
            let mut zero = ZeroOptimizer::new(stage, n, total, lr, wd);
            let mut flat: Vec<Vec<f32>> = (0..n).map(|_| flatten_params(&per_rank[0])).collect();

            let mut losses = Vec::new();
            for _ in 0..steps {
                let mut grads = Vec::with_capacity(n);
                for r in 0..n {
                    for p in &per_rank[r] {
                        p.zero_grad();
                    }
                    let lo = r * shard_b * t;
                    batch_loss(
                        &replicas[r],
                        &ids[lo..lo + shard_b * t],
                        &targets[lo..lo + shard_b * t],
                        shard_b,
                        t,
                    )
                    .backward();
                    grads.push(flatten_grads(&per_rank[r]));
                }
                flat = zero.step(&flat, &grads, 1.0);
                // 把更新后的完整参数写回模型，下一步才能算出正确的 loss
                for r in 0..n {
                    write_params(&per_rank[r], &flat[r]);
                }
                losses.push(batch_loss(&replicas[0], &ids, &targets, total_b, t).item());
            }

            for (i, (a, b)) in losses.iter().zip(&ref_losses).enumerate() {
                assert!(
                    (a - b).abs() < 1e-5,
                    "{stage:?} 第 {i} 步的 loss 与 vanilla DP 不一致：{a} vs {b}"
                );
            }

            // 状态量：N 份分片加起来恰好是一份完整状态（不重不漏），每 rank 只 ≈ 1/N
            let per_rank_bytes = zero.state_bytes_per_rank();
            let full_bytes = zero.full_state_bytes();
            assert_eq!(
                per_rank_bytes.iter().sum::<usize>(),
                full_bytes,
                "各 rank 分片加起来应恰好是一份完整状态，没有重叠也没有缺口"
            );
            for (r, b) in per_rank_bytes.iter().enumerate() {
                assert!(
                    *b <= (full_bytes / n) * 3 / 2,
                    "rank {r} 常驻的优化器状态应约为 1/N：{b} 字节 vs 完整 {full_bytes} 字节"
                );
            }
        }
    }

    // ---------- 张量并行 ----------

    /// Xavier 初始化（与 [`crate::layers::Linear::new`] 同分布）
    fn xavier(rows: usize, cols: usize, rng: &mut Rng) -> Vec<f32> {
        let std = (2.0 / (rows + cols) as f32).sqrt();
        (0..rows * cols).map(|_| rng.randn() * std).collect()
    }

    /// 位置相关的偏置：**不能**全用同一个常数，否则"列对错位"这类错误在
    /// 偏置梯度上完全看不出来（所有元素都一样，错位与不错位的结果相同）
    fn ramp(len: usize, step: f32, base: f32) -> Vec<f32> {
        (0..len).map(|i| base + step * i as f32).collect()
    }

    /// 从完整权重的梯度里按列取出一片（列并行 `dW` 对拍用）
    fn pick_columns(w: &[f32], rows: usize, cols: usize, idx: &[usize]) -> Vec<f32> {
        let mut out = Vec::with_capacity(rows * idx.len());
        for r in 0..rows {
            for &c in idx {
                out.push(w[r * cols + c]);
            }
        }
        out
    }

    /// 从完整权重的梯度里按行区间取出一片（行并行 `dW` 对拍用）
    fn pick_rows(w: &[f32], cols: usize, range: (usize, usize)) -> Vec<f32> {
        w[range.0 * cols..range.1 * cols].to_vec()
    }

    /// 单卡参照 MLP（TP 版要逐位复现它）
    struct FullMlp {
        w1: Tensor,
        b1: Tensor,
        w2: Tensor,
        b2: Tensor,
    }

    impl FullMlp {
        fn forward(&self, x: &Tensor) -> Tensor {
            x.matmul(&self.w1).add(&self.b1).gelu().matmul(&self.w2).add(&self.b2)
        }
    }

    #[test]
    fn test_tensor_parallel_mlp_matches_single_card_forward_and_backward() {
        let (d, hidden, b, n) = (16usize, 32usize, 4usize, 2usize);
        let mut rng = Rng::new(2024);
        let (w1, w2) = (xavier(d, hidden, &mut rng), xavier(hidden, d, &mut rng));
        let (b1, b2) = (ramp(hidden, 0.01, 0.03), ramp(d, -0.02, 0.05));
        let x_data: Vec<f32> = (0..b * d).map(|_| rng.randn()).collect();

        // 参照组：单卡一次算完整个 MLP
        let full = FullMlp {
            w1: Tensor::param(w1.clone(), vec![d, hidden]),
            b1: Tensor::param(b1.clone(), vec![hidden]),
            w2: Tensor::param(w2.clone(), vec![hidden, d]),
            b2: Tensor::param(b2.clone(), vec![d]),
        };
        let x_full = Tensor::param(x_data.clone(), vec![b, d]);
        let y_full = full.forward(&x_full);
        y_full.mul(&y_full).sum().backward();

        // TP 组：同一份权重切给 N 个 rank
        let mut tp = TensorParallelMlp::from_full(&w1, &b1, &w2, &b2, d, hidden, n);
        let x_tp = Tensor::param(x_data.clone(), vec![b, d]);
        let y_tp = tp.forward(&x_tp);
        assert_close(&y_tp.data(), &y_full.data(), 1e-5, "TP MLP 前向 vs 单卡 MLP 前向");
        y_tp.mul(&y_tp).sum().backward();

        // 整个 MLP 只有行并行那一次 allreduce：列并行的接缝上是**零通信**
        let chunk = b * d / n;
        assert_eq!(tp.log().rounds, 2 * (n - 1), "前向通信 = 一次环形 allreduce");
        assert_eq!(
            tp.log().per_rank,
            vec![2 * (n - 1) * chunk; n],
            "各 rank 的通信量必须完全相等（环形的意义就在这里）"
        );

        // 权重梯度：每 rank 那一片 == 单卡梯度的对应切片
        let (dw1, db1, dw2, db2) = (
            full.w1.grad(),
            full.b1.grad(),
            full.w2.grad(),
            full.b2.grad(),
        );
        for r in 0..n {
            let cols = tp.fc1().columns(r);
            assert_eq!(cols.len(), hidden / n, "rank {r} 应负责 {} 列", hidden / n);
            assert_close(
                &tp.fc1().weight_shard(r).grad(),
                &pick_columns(&dw1, d, hidden, cols),
                1e-5,
                &format!("rank {r} 的 dW1"),
            );
            assert_close(
                &tp.fc1().bias_shard(r).grad(),
                &cols.iter().map(|&c| db1[c]).collect::<Vec<f32>>(),
                1e-5,
                &format!("rank {r} 的 db1"),
            );
            let range = tp.fc2().input_range(r);
            assert_close(
                &tp.fc2().weight_shard(r).grad(),
                &pick_rows(&dw2, d, range),
                1e-5,
                &format!("rank {r} 的 dW2"),
            );
        }
        assert_close(&tp.fc2().bias().grad(), &db2, 1e-5, "行并行的 db2（allreduce 之后只加一次）");
        // dx：每个 rank 只算出了"自己那几列对 x 的贡献"，自动微分把它们累加成完整 dx
        // ——这一步就是真机上列并行的**反向 allreduce**
        assert_close(&x_tp.grad(), &x_full.grad(), 1e-5, "dx（TP 的反向 allreduce 结果）");
    }

    #[test]
    fn test_tensor_parallel_qkv_split_keeps_heads_whole() {
        let (d, n_head, n) = (16usize, 4usize, 2usize);
        let head_dim = d / n_head;
        let columns = qkv_head_columns(d, n_head, n);

        // 每列恰好归一个 rank：不重不漏
        let mut owner = vec![usize::MAX; 3 * d];
        for (r, cols) in columns.iter().enumerate() {
            assert_eq!(cols.len(), 3 * d / n, "各 rank 分到的列数应相同（3d/N）");
            for &j in cols {
                assert_eq!(owner[j], usize::MAX, "第 {j} 列被分给了多个 rank");
                owner[j] = r;
            }
        }
        assert!(owner.iter().all(|o| *o != usize::MAX), "有列没分出去");
        // 整 head 不跨 rank：attention 的每一步都以 head 为单位算
        for seg in 0..3 {
            for h in 0..n_head {
                let base = seg * d + h * head_dim;
                assert!(
                    owner[base..base + head_dim].iter().all(|o| *o == owner[base]),
                    "第 {seg} 段第 {h} 个 head 被切到了两张卡上"
                );
            }
        }
        // 每 rank 恰好拿到 n_head/N 组完整的 {q, k, v}
        for r in 0..n {
            let heads: std::collections::BTreeSet<usize> =
                columns[r].iter().map(|c| (c % d) / head_dim).collect();
            assert_eq!(heads.len(), n_head / n, "rank {r} 应负责 {} 个 head", n_head / n);
        }

        // 前向：融合 QKV 的输出（all_gather 按列拼回）必须等于单卡一次 matmul
        let (b, out3) = (3usize, 3 * d);
        let mut rng = Rng::new(77);
        let (w_full, b_full) = (xavier(d, out3, &mut rng), ramp(out3, 0.001, -0.05));
        let x_data: Vec<f32> = (0..b * d).map(|_| rng.randn()).collect();

        let w_ref = Tensor::param(w_full.clone(), vec![d, out3]);
        let b_ref = Tensor::param(b_full.clone(), vec![out3]);
        let x_ref = Tensor::param(x_data.clone(), vec![b, d]);
        let y_ref = x_ref.matmul(&w_ref).add(&b_ref);
        y_ref.mul(&y_ref).sum().backward();

        let qkv = ColumnParallelLinear::qkv(d, n_head, n, &w_full, &b_full);
        let x_tp = Tensor::param(x_data.clone(), vec![b, d]);
        let mut world = World::new(n);
        let y_tp = qkv.forward(&mut world, &x_tp);
        assert_close(
            &y_tp.data(),
            &y_ref.data(),
            1e-5,
            "QKV 列并行（all_gather 拼回）vs 单卡融合 QKV",
        );
        y_tp.mul(&y_tp).sum().backward();

        // 通信：非连续的列切分照样只走一次 all_gather（N-1 轮），每 rank 送自己那份
        let local = 3 * d / n;
        assert_eq!(world.log().rounds, n - 1);
        assert_eq!(world.log().per_rank, vec![(n - 1) * b * local; n]);

        let (dw, db) = (w_ref.grad(), b_ref.grad());
        for r in 0..n {
            let cols = qkv.columns(r);
            assert_close(
                &qkv.weight_shard(r).grad(),
                &pick_columns(&dw, d, out3, cols),
                1e-5,
                &format!("rank {r} 的 dW_qkv"),
            );
            assert_close(
                &qkv.bias_shard(r).grad(),
                &cols.iter().map(|&c| db[c]).collect::<Vec<f32>>(),
                1e-5,
                &format!("rank {r} 的 db_qkv"),
            );
        }
    }

    // ---------- 流水线并行 ----------

    /// 同一随机种子造 `p` 层 `d -> d` 的链（两条链必须逐位一样，对拍才有意义）
    fn layer_chain(p: usize, d: usize, seed: u64) -> Vec<Linear> {
        let mut rng = Rng::new(seed);
        (0..p).map(|_| Linear::new(d, d, &mut rng)).collect()
    }

    #[test]
    fn test_pipeline_gpipe_matches_single_card_and_1f1b_holds_less() {
        let (d, m, per_micro, p) = (16usize, 4usize, 3usize, 2usize);
        let total_rows = m * per_micro;
        let (seed, np) = (99u64, 17u64);

        // 参照组：单卡把 p 层串起来，整个 batch 一次算完
        let ref_layers = layer_chain(p, d, seed);
        let mut rng = Rng::new(np);
        let x_data: Vec<f32> = (0..total_rows * d).map(|_| rng.randn()).collect();
        let x_ref = Tensor::param(x_data.clone(), vec![total_rows, d]);
        let mut y_ref = x_ref.clone();
        for l in &ref_layers {
            y_ref = l.forward(&y_ref);
        }
        // 全 batch 的均方误差：1/B · Σ y²（B = 总行数）
        y_ref.mul(&y_ref).sum().mul_scalar(1.0 / total_rows as f32).backward();

        // 每个 micro-batch 的损失之和 = 全 batch 的损失，缩放因子也按同一个分母给
        let loss_of = |y: &Tensor| y.mul(y).sum().mul_scalar(1.0 / total_rows as f32);

        // GPipe 组：p 段，M 个 micro-batch
        let gpipe_layers = layer_chain(p, d, seed);
        for (a, b) in ref_layers.iter().zip(&gpipe_layers) {
            assert_eq!(a.weight.data(), b.weight.data(), "流水线与单卡的初值必须一致");
        }
        let gpipe = Pipeline::new(gpipe_layers, Schedule::gpipe(m, p));
        let gpipe_inputs: Vec<Tensor> = (0..m)
            .map(|i| {
                let lo = i * per_micro * d;
                Tensor::param(x_data[lo..lo + per_micro * d].to_vec(), vec![per_micro, d])
            })
            .collect();
        let gpipe_report = gpipe.run(&gpipe_inputs, loss_of);

        // 前向：各 micro-batch 的输出按原顺序拼起来，必须等于单卡整批的输出
        let got: Vec<f32> = gpipe_report.outputs.iter().flat_map(|t| t.data()).collect();
        assert_close(&got, &y_ref.data(), 1e-5, "GPipe 输出（micro-batch 拼回）vs 单卡整批");

        // 反向：每层的权重/偏置梯度 == 单卡梯度（micro-batch 的梯度自动累加）
        for (s, (a, b)) in ref_layers.iter().zip(gpipe.stages()).enumerate() {
            let (pa, pb) = (a.parameters(), b.parameters());
            assert_close(&pb[0].grad(), &pa[0].grad(), 1e-5, &format!("第 {s} 段权重梯度"));
            assert_close(&pb[1].grad(), &pa[1].grad(), 1e-5, &format!("第 {s} 段偏置梯度"));
        }

        // 1F1B 组：同一份初值、同一批输入，只换调度
        let onef1b = Pipeline::new(layer_chain(p, d, seed), Schedule::one_forward_one_backward(m, p));
        let onef1b_inputs: Vec<Tensor> = (0..m)
            .map(|i| {
                let lo = i * per_micro * d;
                Tensor::param(x_data[lo..lo + per_micro * d].to_vec(), vec![per_micro, d])
            })
            .collect();
        let onef1b_report = onef1b.run(&onef1b_inputs, loss_of);

        // 两种调度算出来的东西必须**完全一样**——调度只影响"什么时候算"，不影响"算成什么"
        let got2: Vec<f32> = onef1b_report.outputs.iter().flat_map(|t| t.data()).collect();
        assert_close(&got2, &got, 1e-5, "1F1B 输出 vs GPipe 输出");
        assert_close(&got2, &y_ref.data(), 1e-5, "1F1B 输出 vs 单卡整批");
        for (s, (a, b)) in gpipe.stages().iter().zip(onef1b.stages()).enumerate() {
            let (pa, pb) = (a.parameters(), b.parameters());
            assert_close(&pb[1].grad(), &pa[1].grad(), 1e-5, &format!("第 {s} 段偏置梯度：1F1B vs GPipe"));
        }

        // 差别在驻留：GPipe 压着全部 M 个 micro-batch，1F1B 只压 p 个
        assert_eq!(gpipe_report.peak_in_flight, m, "GPipe 的驻留峰值就是 micro-batch 数");
        assert_eq!(onef1b_report.peak_in_flight, p, "1F1B 的驻留峰值由段数决定");
        assert!(
            onef1b_report.peak_in_flight < gpipe_report.peak_in_flight,
            "1F1B 必须比 GPipe 省（{} vs {}）",
            onef1b_report.peak_in_flight,
            gpipe_report.peak_in_flight
        );
        // 跨阶段激活缓冲：每个在飞的 micro-batch 在 p-1 条边界上各占一块
        assert_eq!(gpipe_report.peak_boundary_buffers, m * (p - 1));
        assert_eq!(onef1b_report.peak_boundary_buffers, p * (p - 1));
        // 调度表自身给出的画像与实测一致（说明"省"是计划里就算出来的，不是碰巧）
        assert_eq!(gpipe.schedule().peak_in_flight, gpipe_report.peak_in_flight);
        assert_eq!(gpipe.schedule().peak_boundary_buffers, gpipe_report.peak_boundary_buffers);
        assert_eq!(onef1b.schedule().peak_in_flight, onef1b_report.peak_in_flight);
        assert_eq!(onef1b.schedule().peak_boundary_buffers, onef1b_report.peak_boundary_buffers);

        // 事件表的完整性：每个 (micro-batch, 段) 前向、反向各恰好一次
        for sch in [gpipe.schedule(), onef1b.schedule()] {
            assert_eq!(sch.steps.len(), 2 * m * p);
            let count = |target: Step| sch.steps.iter().filter(|s| **s == target).count();
            for i in 0..m {
                for s in 0..p {
                    assert_eq!(count(Step::Forward(i, s)), 1, "({i}, {s}) 的前向应恰好一次");
                    assert_eq!(count(Step::Backward(i, s)), 1, "({i}, {s}) 的反向应恰好一次");
                }
            }
        }
    }

    // ---------- 3D 并行 ----------

    #[test]
    fn test_3d_parallel_partitions_are_disjoint_and_complete() {
        let (layers, columns, batch) = (6usize, 12usize, 8usize);
        // 覆盖：纯一个轴、两个轴、三个轴；含 3/6 这样除不尽的切分
        for (dp, tp, pp) in [
            (1usize, 1usize, 1usize),
            (4, 1, 1),
            (1, 4, 1),
            (1, 1, 4),
            (2, 2, 2),
            (3, 2, 2),
            (2, 3, 1),
        ] {
            let cfg = DistConfig::new(dp, tp, pp);
            let n = cfg.world_size();
            assert_eq!(n, dp * tp * pp);

            // 1) rank ↔ 坐标 是双射：n 个 rank 必须落在 n 个互不相同的坐标上
            let mut coords = std::collections::BTreeSet::new();
            for r in 0..n {
                let c = cfg.coord(r);
                assert!(c.dp < dp && c.tp < tp && c.pp < pp, "坐标越界");
                assert_eq!(cfg.rank_of(c), r, "rank {r} 的坐标还原不回去");
                assert!(coords.insert(c), "两个 rank 落在同一个坐标上");
            }
            assert_eq!(coords.len(), n);

            // 2) 参数在 (tp, pp) 网格上不重不漏（dp 轴是复制，不参与参数切分）
            let mut owner: Vec<Option<usize>> = vec![None; layers * columns];
            for tp_i in 0..tp {
                for pp_i in 0..pp {
                    let r = cfg.rank_of(RankCoord { dp: 0, tp: tp_i, pp: pp_i });
                    let plan = cfg.plan(r, layers, columns, batch);
                    assert_eq!(plan.coord, RankCoord { dp: 0, tp: tp_i, pp: pp_i });
                    for l in plan.layers.0..plan.layers.1 {
                        for &c in &plan.columns {
                            let cell = l * columns + c;
                            assert!(owner[cell].is_none(), "层 {l} 第 {c} 列被两个 rank 同时持有");
                            owner[cell] = Some(r);
                        }
                    }
                }
            }
            assert!(
                owner.iter().all(|o| o.is_some()),
                "dp{dp}tp{tp}pp{pp}：有参数格子没人负责，那部分权重永远不会被训练"
            );

            // 3) dp 轴是**复制**：同一 (tp, pp)、不同 dp 的 rank 持有同一份参数分片
            for tp_i in 0..tp {
                for pp_i in 0..pp {
                    let plans: Vec<RankPlan> = (0..dp)
                        .map(|d| {
                            cfg.plan(
                                cfg.rank_of(RankCoord { dp: d, tp: tp_i, pp: pp_i }),
                                layers,
                                columns,
                                batch,
                            )
                        })
                        .collect();
                    for p in &plans[1..] {
                        assert_eq!(p.layers, plans[0].layers, "dp 副本的层切分必须相同");
                        assert_eq!(p.columns, plans[0].columns, "dp 副本的列切分必须相同");
                    }
                }
            }

            // 4) 数据在 dp 轴上不重不漏，且在 (tp, pp) 内是复制的
            let mut covered = vec![0usize; batch];
            for d in 0..dp {
                let r0 = cfg.rank_of(RankCoord { dp: d, tp: 0, pp: 0 });
                let b0 = cfg.plan(r0, layers, columns, batch).batch;
                for tp_i in 0..tp {
                    for pp_i in 0..pp {
                        let r = cfg.rank_of(RankCoord { dp: d, tp: tp_i, pp: pp_i });
                        assert_eq!(
                            cfg.plan(r, layers, columns, batch).batch,
                            b0,
                            "同一 dp 坐标上的 rank 必须吃同一段数据"
                        );
                    }
                }
                for i in b0.0..b0.1 {
                    covered[i] += 1;
                }
            }
            assert!(
                covered.iter().all(|c| *c == 1),
                "dp{dp}tp{tp}pp{pp}：样本必须恰好被一个 dp 分片覆盖，实际 {covered:?}"
            );

            // 5) 三个组：TP 组管层内通信、DP 组管梯度同步
            for r in 0..n {
                let tp_g = cfg.tp_group(r);
                let dp_g = cfg.dp_group(r);
                assert_eq!(tp_g.len(), tp);
                assert_eq!(dp_g.len(), dp);
                for &q in &tp_g {
                    assert_eq!(cfg.coord(q).dp, cfg.coord(r).dp);
                    assert_eq!(cfg.coord(q).pp, cfg.coord(r).pp);
                }
                for &q in &dp_g {
                    assert_eq!(cfg.coord(q).tp, cfg.coord(r).tp);
                    assert_eq!(cfg.coord(q).pp, cfg.coord(r).pp);
                }
            }

            // 6) 流水线沿 pp 方向是一条链：首段没有上游、末段没有下游
            if pp > 1 {
                let first = cfg.rank_of(RankCoord { dp: 0, tp: 0, pp: 0 });
                let last = cfg.rank_of(RankCoord { dp: 0, tp: 0, pp: pp - 1 });
                assert_eq!(cfg.prev_stage(first), None);
                assert_eq!(cfg.next_stage(last), None);
                let mut cur = first;
                let mut hops = 0;
                while let Some(nx) = cfg.next_stage(cur) {
                    cur = nx;
                    hops += 1;
                }
                assert_eq!(cur, last);
                assert_eq!(hops, pp - 1);
            }

            // 7) 分工报告能打印出来，且与坐标、切分对得上
            for r in 0..n {
                let plan = cfg.plan(r, layers, columns, batch);
                let s = plan.summary(&cfg);
                assert!(s.contains(&format!("rank {r}/")), "报告里应含 rank 编号：{s}");
                let c = plan.coord;
                assert!(s.contains(&format!("[dp {} tp {} pp {}]", c.dp, c.tp, c.pp)), "{s}");
                assert!(s.contains(&format!("层 [{}, {})", plan.layers.0, plan.layers.1)), "{s}");
                assert!(s.contains(&format!("样本 [{}, {})", plan.batch.0, plan.batch.1)), "{s}");
            }
        }
    }
}
