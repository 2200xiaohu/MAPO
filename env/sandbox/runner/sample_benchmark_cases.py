#!/usr/bin/env python3
"""
三维分层抽样脚本（新数据集）

目标：
- 基于新的 IEDR 结果（iedr_batch_results_new.json）
- 结合新数据的话题分类（Benchmark/topics/new_data/classification_results.json）
- 按 C/A/P 主导轴、难度分布、情境标签 做 30 个案例的三维分层抽样
"""

import json
import math
from collections import defaultdict
from typing import Dict, List
import random
from pathlib import Path


def calculate_euclidean_distance(c: float, a: float, p: float) -> float:
    """计算欧氏距离作为总赤字"""
    return math.sqrt(c**2 + a**2 + p**2)


def determine_dominant_axis(c: float, a: float, p: float) -> str:
    """确定主导轴"""
    abs_c, abs_a, abs_p = abs(c), abs(a), abs(p)
    max_deficit = max(abs_c, abs_a, abs_p)

    if abs_c == max_deficit:
        return "C"
    elif abs_a == max_deficit:
        return "A"
    else:
        return "P"


def determine_difficulty(total_deficit: float) -> str:
    """
    根据总赤字（欧氏距离）确定难度等级
    参考分析报告的分布：
    - 较易: < 28
    - 中等: 28-32
    - 困难: 32-36
    - 极难: >= 36
    """
    if total_deficit < 28:
        return "较易"
    elif total_deficit < 32:
        return "中等"
    elif total_deficit < 36:
        return "困难"
    else:
        return "极难"


def extract_primary_category(labels: str) -> str:
    """
    提取主要情境分类
    标签格式: "大分类-小分类,情感类型-具体情感"
    我们取第一个逗号前的大分类
    """
    if not labels:
        return "未分类"

    parts = labels.split(",")
    if parts:
        category = parts[0].split("-")[0]
        return category or "未分类"
    return "未分类"


def load_data() -> tuple[list[Dict], Dict]:
    """加载新数据集的IEDR和分类数据"""
    project_root = Path(__file__).parent.parent.parent

    iedr_file = project_root / "scripts/results/iedr_batch_results_new.json"
    cls_file = project_root / "Benchmark/topics/new_data/classification_results.json"

    print(f"  - 读取IEDR结果: {iedr_file}")
    with open(iedr_file, "r", encoding="utf-8") as f:
        iedr_data = json.load(f)

    print(f"  - 读取分类结果: {cls_file}")
    with open(cls_file, "r", encoding="utf-8") as f:
        classification_data = json.load(f)

    classification_dict = {item["script_id"]: item.get("labels", "") for item in classification_data}

    return iedr_data, classification_dict


def prepare_case_metadata(iedr_data: List[Dict], classification_dict: Dict) -> List[Dict]:
    """根据IEDR和分类信息准备案例元数据"""
    cases: List[Dict] = []

    for item in iedr_data:
        script_id = item.get("script_id")
        if not script_id:
            continue

        if item.get("status") != "success":
            continue

        p0 = item.get("P_0")
        if not p0:
            continue

        c_deficit = p0.get("C", 0)
        a_deficit = p0.get("A", 0)
        p_deficit = p0.get("P", 0)
        total_deficit = p0.get("total", 0.0)

        if not total_deficit:
            total_deficit = calculate_euclidean_distance(c_deficit, a_deficit, p_deficit)

        dominant_axis = determine_dominant_axis(c_deficit, a_deficit, p_deficit)
        difficulty = determine_difficulty(total_deficit)

        labels = classification_dict.get(script_id, "")
        category = extract_primary_category(labels)

        cases.append(
            {
                "script_id": script_id,
                "c_deficit": c_deficit,
                "a_deficit": a_deficit,
                "p_deficit": p_deficit,
                "total_deficit": total_deficit,
                "dominant_axis": dominant_axis,
                "difficulty": difficulty,
                "category": category,
                "labels": labels,
            }
        )

    return cases


def stratified_sampling(cases: List[Dict], target_count: int = 30) -> List[Dict]:
    """
    三维分层抽样
    维度一: C-A-P轴均衡 (各1/3)
    维度二: 难度分布 (较易20%, 中等40%, 困难30%, 极难10%)
    维度三: 情境标签尽可能多样
    """
    axis_quota = {"C": 10, "A": 10, "P": 10}

    difficulty_ratios = {"较易": 0.20, "中等": 0.40, "困难": 0.30, "极难": 0.10}

    cases_by_axis: dict[str, list[Dict]] = defaultdict(list)
    for case in cases:
        cases_by_axis[case["dominant_axis"]].append(case)

    print("\n=== 各轴难度分布统计（新数据集） ===")
    for axis in ["C", "A", "P"]:
        axis_cases = cases_by_axis[axis]
        print(f"\n{axis}轴案例数: {len(axis_cases)}")
        difficulty_counts: dict[str, int] = defaultdict(int)
        for case in axis_cases:
            difficulty_counts[case["difficulty"]] += 1
        for diff in ["较易", "中等", "困难", "极难"]:
            print(f"  {diff}: {difficulty_counts[diff]}个")

    print("\n=== 情境分布统计（新数据集） ===")
    category_counts: dict[str, int] = defaultdict(int)
    for case in cases:
        category_counts[case["category"]] += 1
    for cat, count in sorted(category_counts.items(), key=lambda x: x[1], reverse=True):
        print(f"{cat}: {count}个")

    selected_cases: list[Dict] = []

    for axis in ["C", "A", "P"]:
        print(f"\n=== 从{axis}轴抽取 {axis_quota[axis]} 个案例 ===")
        axis_cases = cases_by_axis[axis]

        cases_by_difficulty: dict[str, list[Dict]] = defaultdict(list)
        for case in axis_cases:
            cases_by_difficulty[case["difficulty"]].append(case)

        difficulty_targets = {
            "较易": round(axis_quota[axis] * difficulty_ratios["较易"]),
            "中等": round(axis_quota[axis] * difficulty_ratios["中等"]),
            "困难": round(axis_quota[axis] * difficulty_ratios["困难"]),
            "极难": round(axis_quota[axis] * difficulty_ratios["极难"]),
        }
        total = sum(difficulty_targets.values())
        if total != axis_quota[axis]:
            difficulty_targets["中等"] += axis_quota[axis] - total

        print(f"难度目标分配: {difficulty_targets}")

        axis_selected: list[Dict] = []
        selected_categories: set[str] = set()

        for difficulty, target in difficulty_targets.items():
            available = cases_by_difficulty[difficulty]
            print(f"\n{difficulty}难度: 可用{len(available)}个，目标{target}个")

            if len(available) == 0 or target == 0:
                if target > 0:
                    print(f"  ⚠️ {difficulty}难度没有可用案例，跳过")
                continue

            if len(available) <= target:
                selected = available
                print(f"  案例不足，全选{len(selected)}个")
            else:
                selected = []
                for case in available:
                    if len(selected) >= target:
                        break
                    if case["category"] not in selected_categories:
                        selected.append(case)
                        selected_categories.add(case["category"])

                if len(selected) < target:
                    remaining = [c for c in available if c not in selected]
                    random.shuffle(remaining)
                    needed = target - len(selected)
                    selected.extend(remaining[:needed])

                print(f"  成功抽取{len(selected)}个")

            axis_selected.extend(selected)

        if len(axis_selected) < axis_quota[axis]:
            needed = axis_quota[axis] - len(axis_selected)
            print(f"\n需要补充{needed}个案例（{axis}轴）")
            all_available = [c for c in axis_cases if c not in axis_selected]
            random.shuffle(all_available)
            axis_selected.extend(all_available[:needed])

        if len(axis_selected) > axis_quota[axis]:
            random.shuffle(axis_selected)
            axis_selected = axis_selected[: axis_quota[axis]]

        print(f"{axis}轴最终抽取: {len(axis_selected)}个")
        for case in axis_selected:
            print(
                f"  - {case['script_id']}: {case['difficulty']}, {case['category']}, "
                f"总赤字={case['total_deficit']:.2f}"
            )

        selected_cases.extend(axis_selected)

    return selected_cases


def save_results(selected_cases: List[Dict], output_file: Path):
    """保存抽样结果（JSON + MD + ID 列表）"""
    selected_cases.sort(key=lambda x: int(x["script_id"].replace("script_", "")))

    report = {
        "sampling_method": "三维分层抽样 (Multi-Dimensional Stratified Sampling)",
        "total_cases": len(selected_cases),
        "sampling_dimensions": {
            "dimension_1": "C-A-P轴均衡 (各占1/3)",
            "dimension_2": "难度分布 (较易20%, 中等40%, 困难30%, 极难10%)",
            "dimension_3": "情境标签多样性",
        },
        "statistics": {
            "axis_distribution": {},
            "difficulty_distribution": {},
            "category_distribution": {},
        },
        "selected_cases": [],
    }

    axis_counts: dict[str, int] = defaultdict(int)
    difficulty_counts: dict[str, int] = defaultdict(int)
    category_counts: dict[str, int] = defaultdict(int)

    for case in selected_cases:
        axis_counts[case["dominant_axis"]] += 1
        difficulty_counts[case["difficulty"]] += 1
        category_counts[case["category"]] += 1

        report["selected_cases"].append(
            {
                "script_id": case["script_id"],
                "dominant_axis": case["dominant_axis"],
                "difficulty": case["difficulty"],
                "category": case["category"],
                "labels": case["labels"],
                "deficits": {
                    "C": case["c_deficit"],
                    "A": case["a_deficit"],
                    "P": case["p_deficit"],
                    "total": round(case["total_deficit"], 2),
                },
            }
        )

    report["statistics"]["axis_distribution"] = dict(axis_counts)
    report["statistics"]["difficulty_distribution"] = dict(difficulty_counts)
    report["statistics"]["category_distribution"] = dict(category_counts)

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    md_report = generate_markdown_report(report)
    md_file = output_file.with_suffix(".md")
    with open(md_file, "w", encoding="utf-8") as f:
        f.write(md_report)

    id_list_file = output_file.with_name(output_file.stem + "_ids.txt")
    with open(id_list_file, "w", encoding="utf-8") as f:
        for case in selected_cases:
            f.write(f"{case['script_id']}\n")

    print(f"\n✅ 结果已保存:")
    print(f"  - 详细JSON: {output_file}")
    print(f"  - 报告MD: {md_file}")
    print(f"  - ID列表: {id_list_file}")


def generate_markdown_report(report: Dict) -> str:
    """生成Markdown格式的报告"""
    lines: list[str] = [
        "# EPJ Benchmark（三维分层抽样结果）- 新数据集",
        "",
        f"**抽样方法**: {report['sampling_method']}",
        f"**样本数量**: {report['total_cases']}个案例",
        "",
        "---",
        "",
        "## 📊 抽样维度",
        "",
        "### 维度一: C-A-P轴均衡",
        f"- {report['sampling_dimensions']['dimension_1']}",
        "",
        "### 维度二: 难度分布",
        f"- {report['sampling_dimensions']['dimension_2']}",
        "",
        "### 维度三: 情境标签",
        f"- {report['sampling_dimensions']['dimension_3']}",
        "",
        "---",
        "",
        "## 📈 分布统计",
        "",
        "### 轴分布",
        "",
    ]

    stats = report["statistics"]
    total_cases = report["total_cases"] or 1
    for axis, count in sorted(stats["axis_distribution"].items()):
        lines.append(f"- **{axis}轴**: {count}个 ({count/total_cases*100:.1f}%)")

    lines.extend(["", "### 难度分布", ""])
    difficulty_order = ["较易", "中等", "困难", "极难"]
    for diff in difficulty_order:
        count = stats["difficulty_distribution"].get(diff, 0)
        lines.append(f"- **{diff}**: {count}个 ({count/total_cases*100:.1f}%)")

    lines.extend(["", "### 情境分布", ""])
    for category, count in sorted(
        stats["category_distribution"].items(), key=lambda x: x[1], reverse=True
    ):
        lines.append(f"- **{category}**: {count}个 ({count/total_cases*100:.1f}%)")

    lines.extend(
        [
            "",
            "---",
            "",
            "## 📋 抽取的案例列表",
            "",
            "| Script ID | 主导轴 | 难度 | 情境分类 | 总赤字 | 完整标签 |",
            "|-----------|--------|------|----------|--------|----------|",
        ]
    )

    for case in report["selected_cases"]:
        lines.append(
            f"| {case['script_id']} | {case['dominant_axis']} | {case['difficulty']} | "
            f"{case['category']} | {case['deficits']['total']:.2f} | {case['labels']} |"
        )

    lines.extend(
        [
            "",
            "---",
            "",
            "## ✅ 抽样质量评估",
            "",
            "### 轴均衡性",
            "",
        ]
    )

    axis_dist = stats["axis_distribution"]
    target_per_axis = report["total_cases"] / 3 if report["total_cases"] else 1
    for axis in ["C", "A", "P"]:
        count = axis_dist.get(axis, 0)
        deviation = (
            abs(count - target_per_axis) / target_per_axis * 100 if target_per_axis > 0 else 0
        )
        status = "✅" if deviation < 10 else "⚠️"
        lines.append(
            f"{status} {axis}轴: 目标{target_per_axis:.0f}个，实际{count}个，偏差{deviation:.1f}%"
        )

    lines.extend(["", "### 难度分布质量", ""])
    difficulty_targets = {"较易": 0.20, "中等": 0.40, "困难": 0.30, "极难": 0.10}
    for diff, target_ratio in difficulty_targets.items():
        actual_count = stats["difficulty_distribution"].get(diff, 0)
        actual_ratio = actual_count / total_cases if total_cases > 0 else 0
        deviation = (
            abs(actual_ratio - target_ratio) / target_ratio * 100 if target_ratio > 0 else 0
        )
        status = "✅" if deviation < 30 else "⚠️"
        lines.append(
            f"{status} {diff}: 目标{target_ratio*100:.0f}%，实际{actual_ratio*100:.1f}%，偏差{deviation:.1f}%"
        )

    lines.extend(
        [
            "",
            "### 情境多样性",
            f"- 覆盖了 **{len(stats['category_distribution'])}** 种不同情境分类",
            "",
            "---",
            "",
            "**报告生成完成** ✅",
        ]
    )

    return "\n".join(lines)


def main():
    print("=== EPJ Benchmark 三维分层抽样系统（新数据集） ===\n")

    random.seed(42)

    print("📂 正在加载数据...")
    iedr_data, classification_dict = load_data()
    print(f"✅ 已加载 {len(iedr_data)} 个剧本的IEDR数据")
    print(f"✅ 已加载 {len(classification_dict)} 个剧本的分类数据")

    print("\n📊 正在准备案例元数据...")
    cases = prepare_case_metadata(iedr_data, classification_dict)
    print(f"✅ 成功处理 {len(cases)} 个案例")

    print("\n🎯 开始三维分层抽样...")
    selected_cases = stratified_sampling(cases, target_count=30)

    project_root = Path(__file__).parent.parent.parent
    # 抽样结果作为“官方基准案例配置”，统一放到 runner/benchmark_cases 目录
    output_file = project_root / "runner/benchmark_cases/sampled_benchmark_30_new.json"

    print("\n💾 正在保存结果...")
    save_results(selected_cases, output_file)

    print("\n" + "=" * 50)
    print("✅ 新数据集抽样完成！")
    print("=" * 50)


if __name__ == "__main__":
    main()


