#!/usr/bin/env python3
"""
将EPM轨迹数据导出为标准JSON格式
供前端可视化直接加载使用
"""

import json
import numpy as np
from pathlib import Path

def export_trajectories(results_dir, output_file):
    """提取所有轨迹数据并导出为JSON"""
    results_dir = Path(results_dir)
    
    trajectories_data = {
        "metadata": {
            "model_name": "default_20251106_233640",
            "total_cases": 0,
            "success_count": 0,
            "failure_count": 0,
            "max_turns": 0
        },
        "trajectories": []
    }
    
    for json_file in sorted(results_dir.glob('script_*_result.json')):
        with open(json_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        script_id = data['script_id']
        total_turns = data['total_turns']
        termination_reason = data.get('termination_reason', '')
        
        # 判断成功/失败
        if 'EPM' in termination_reason and ('失败' in termination_reason or '负能量' in termination_reason):
            status = 'failure'
        elif total_turns >= 45:
            status = 'max_turns'
        else:
            status = 'success'
        
        # 提取轨迹点
        if 'epj' in data and 'trajectory' in data['epj']:
            trajectory = data['epj']['trajectory']
            points = [[point['P_t'][0], point['P_t'][1], point['P_t'][2]] 
                     for point in trajectory]
            
            trajectories_data["trajectories"].append({
                'script_id': script_id,
                'points': points,
                'status': status,
                'total_turns': total_turns
            })
            
            # 更新统计
            trajectories_data["metadata"]["total_cases"] += 1
            if status == 'success':
                trajectories_data["metadata"]["success_count"] += 1
            else:
                trajectories_data["metadata"]["failure_count"] += 1
            trajectories_data["metadata"]["max_turns"] = max(
                trajectories_data["metadata"]["max_turns"], 
                len(points)
            )
    
    # 保存
    output_file = Path(output_file)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(trajectories_data, f, ensure_ascii=False, indent=2)
    
    print(f"✅ 数据导出完成:")
    print(f"   📁 输出文件: {output_file}")
    print(f"   📊 案例总数: {trajectories_data['metadata']['total_cases']}")
    print(f"   💙 成功: {trajectories_data['metadata']['success_count']}")
    print(f"   ❤️ 失败: {trajectories_data['metadata']['failure_count']}")
    print(f"   📏 最大轮次: {trajectories_data['metadata']['max_turns']}")

def main():
    results_dir = "results/benchmark_runs/default_20251106_233640"
    output_file = "visualization/data/trajectories.json"
    
    print("🔄 开始导出轨迹数据...\n")
    export_trajectories(results_dir, output_file)
    print("\n✅ 完成！前端可直接加载此JSON文件")

if __name__ == "__main__":
    main()

