# Utils Scripts - 工具脚本

本目录包含辅助工具脚本。

---

## 📊 脚本列表（1个）

### `export_trajectory_data.py`
**功能**: 导出轨迹数据为JSON格式

**描述**:
- 从对话记录中提取轨迹数据
- 转换为标准JSON格式
- 用于前端可视化或数据分析

**输入**: 
- `results/benchmark_runs/{run_id}/conversation_*.json`

**输出**: 
- `visualization/data/trajectories.json`

**输出格式**:
```json
{
  "metadata": {
    "total_cases": 30,
    "success_count": 23,
    "failure_count": 7,
    "export_time": "2025-11-14T10:30:00"
  },
  "trajectories": [
    {
      "script_id": "script_001",
      "success": true,
      "turns": 15,
      "points": [
        {"turn": 0, "C": -15, "A": -18, "P": -16},
        {"turn": 1, "C": -14, "A": -17, "P": -15},
        ...
      ],
      "final_distance": 5.2,
      "victory_condition": "distance"
    }
  ]
}
```

**使用方法**:
```bash
python3 scripts/utils/export_trajectory_data.py
```

---

## 🔧 使用场景

### 数据导出
使用`export_trajectory_data.py`：
- 为前端可视化准备数据
- 导出数据用于外部分析
- 生成数据快照

---

## 📝 开发新工具脚本

### 脚本模板

```python
#!/usr/bin/env python3
"""
工具脚本名称

功能描述：
- 功能点1
- 功能点2

输入：
- 输入文件或数据

输出：
- 输出文件或结果

使用方法：
    python3 scripts/utils/your_script.py

作者: EPJ项目组
日期: YYYY-MM-DD
"""

import sys
from pathlib import Path

# 添加项目根目录到路径
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

def main():
    """主函数"""
    print("=" * 70)
    print("工具脚本名称")
    print("=" * 70)
    
    # 实现功能
    pass

if __name__ == "__main__":
    main()
```

---

## 🐛 常见问题

### Q: 如何批量处理文件？
A: 使用`Path.glob()`遍历文件：
```python
from pathlib import Path

for file in Path("data/").glob("*.md"):
    process_file(file)
```

### Q: 如何处理不同编码的文件？
A: 明确指定编码：
```python
with open(file, 'r', encoding='utf-8') as f:
    content = f.read()
```

### Q: 如何避免覆盖原文件？
A: 先备份：
```python
import shutil
shutil.copy(original_file, f"{original_file}.backup")
```

---

**最后更新**: 2025-11-14  
**维护者**: EPJ项目组

