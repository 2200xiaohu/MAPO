#!/bin/bash
#
# 使用 tmux 在后台运行完整的 SOTA 模型测试（30个案例）
# 即使断开 SSH 连接或关闭终端，测试也会继续运行
#

# 设置 tmux 会话名称
SESSION_NAME="benchmark_sota"

# 检查是否已经存在同名会话
if tmux has-session -t $SESSION_NAME 2>/dev/null; then
    echo "⚠️  tmux 会话 '$SESSION_NAME' 已存在"
    echo ""
    echo "选择操作："
    echo "  1. 连接到现有会话查看进度：tmux attach -t $SESSION_NAME"
    echo "  2. 终止旧会话并创建新的：tmux kill-session -t $SESSION_NAME && bash $0"
    exit 1
fi

# 创建新的 tmux 会话并运行测试
echo "🚀 创建 tmux 会话: $SESSION_NAME"
echo "📋 将运行 30 个 SOTA 模型测试案例（预计需要数小时）"
echo "🤖 测试模型: google/gemini-2.5-pro"
echo ""

# 获取当前时间戳
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_FILE="benchmark_sota_${TIMESTAMP}.log"

# 创建 detached 会话并执行命令
tmux new-session -d -s $SESSION_NAME -c "$(pwd)" "python3 runner/run_benchmark_sota_model.py 2>&1 | tee $LOG_FILE"

echo "✅ tmux 会话已启动！"
echo ""
echo "📌 常用命令："
echo "  查看运行状态：tmux attach -t $SESSION_NAME"
echo "  或简写：tmux a -t $SESSION_NAME"
echo "  断开连接（保持运行）：按 Ctrl+B 然后按 D"
echo "  终止运行：tmux kill-session -t $SESSION_NAME"
echo ""
echo "💡 重要提示："
echo "  ✅ 退出会话时使用：Ctrl+B 然后 D（程序继续运行）"
echo "  ❌ 千万别用：Ctrl+C 或 exit（会杀死进程）"
echo ""
echo "📁 输出位置："
echo "  - 实时日志：$LOG_FILE"
echo "  - 结果文件：results/benchmark_runs/gemini-2.5-pro_*/"
echo ""
echo "🔍 现在连接到会话查看运行状态..."
sleep 2
tmux attach -t $SESSION_NAME
