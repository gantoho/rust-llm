# 训练启动脚本：固定开启整叠 Block 常驻路径（实测 0.44 vs 0.53 s/步，约快 17%）。
# 用法与直接 cargo run 相同，参数原样透传：
#   ./train.ps1                                  # 用默认 config 训练
#   ./train.ps1 --resume checkpoints/zh/latest.ckpt
$ErrorActionPreference = 'Stop'
$env:LLM_GPU_STACK = '1'
cargo run --release --features gpu -- train @args
