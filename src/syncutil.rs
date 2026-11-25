//! 并发小工具：**中毒免疫**的互斥锁获取。
//!
//! `Mutex::lock().unwrap()` 的问题是：**持锁线程一旦 panic，锁就被标记为中毒，
//! 之后所有调用 `unwrap()` 的地方都会跟着 panic**——一次偶发 panic 会永久废掉整个子系统。
//! 本项目里这些锁保护的是诊断计数、buffer 池、常驻缓存，数据在 panic 之后依然可用
//! （最坏只是某次统计少记一笔），因此正确做法是恢复内层数据而不是拒绝服务。

use std::sync::{Mutex, MutexGuard};

/// 取锁；即使锁中毒也返回可用的守卫（见模块文档）。
pub fn lock<T>(m: &Mutex<T>) -> MutexGuard<'_, T> {
    m.lock().unwrap_or_else(|e| e.into_inner())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;

    /// 前提验证 + 恢复：故意让锁中毒，`lock()` 仍能拿到 panic 前的数据。
    #[test]
    fn test_lock_recovers_from_poison() {
        let m = Arc::new(Mutex::new(vec![1, 2, 3]));
        let m2 = Arc::clone(&m);
        let _ = std::thread::spawn(move || {
            let mut g = m2.lock().unwrap();
            g.push(4);
            panic!("故意 panic，让锁中毒");
        })
        .join();

        assert!(m.lock().is_err(), "前提：锁应已中毒");
        let g = lock(&m);
        assert_eq!(*g, vec![1, 2, 3, 4], "恢复后应拿到 panic 前的数据");
    }

    /// 并发累加不丢更新（与标准 `Mutex` 行为一致）。
    #[test]
    fn test_lock_concurrent_increments_are_atomic() {
        let m = Arc::new(Mutex::new(0usize));
        let mut hs = Vec::new();
        for _ in 0..8 {
            let m = Arc::clone(&m);
            hs.push(std::thread::spawn(move || {
                for _ in 0..1000 {
                    *lock(&m) += 1;
                }
            }));
        }
        for h in hs {
            h.join().unwrap();
        }
        assert_eq!(*lock(&m), 8000);
    }
}
