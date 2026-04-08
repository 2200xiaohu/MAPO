#!/usr/bin/env python3
"""
Case-Level Model Comparison Generator
Aggregates results from all resampled model runs into a single comprehensive Excel table.

Features:
1. Rows: 30 Benchmark Cases (with metadata like Axis, Difficulty, Category, SP Features).
2. Columns: For each model, shows "Success/Failure", "Turns", and "Failure Reason".
3. Output: A single Excel file suitable for paper publication.
"""

import json
import pandas as pd
from pathlib import Path
import sys
import re

# Define project paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
RESAMPLED_RESULTS_DIR = PROJECT_ROOT / "results/benchmark_runs/resampled_results"
METADATA_FILE = PROJECT_ROOT / "runner/benchmark_cases/sampled_benchmark_30_new.json"
SP_DIR = PROJECT_ROOT / "Benchmark/topics/new_data/character_setting"
OUTPUT_FILE = RESAMPLED_RESULTS_DIR / "MODEL_COMPARISON_BY_CASE.xlsx"

# Define models to include (folder names)
MODELS = [
    "gemini-2.5-pro_resampled",
    "qwen3-235b-a22b-2507_resampled",
    "kimi-k2-0905_resampled",
    "Echo-N1",
    "doubao-1.5-character_resampled",
    "qwen3-32b_resampled"
]

# Display names for columns
MODEL_DISPLAY_NAMES = {
    "gemini-2.5-pro_resampled": "Gemini 2.5 Pro",
    "qwen3-235b-a22b-2507_resampled": "Qwen 3 235B",
    "kimi-k2-0905_resampled": "Kimi k2-0905",
    "Echo-N1": "Echo-N1",
    "doubao-1.5-character_resampled": "Doubao 1.5 Character",
    "qwen3-32b_resampled": "Qwen 3 32B"
}

def load_case_metadata():
    """Load case metadata to form the backbone of the table."""
    with open(METADATA_FILE, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    cases_list = data.get('selected_cases', data.get('sampled_cases', []))
    metadata_map = {}
    
    for case in cases_list:
        sid = case['script_id']
        
        # Translation Maps
        diff_map = {
            "较易": "Easy", "中等": "Medium", "困难": "Hard", "极难": "Extreme",
            "Easy": "Easy", "Medium": "Medium", "Hard": "Hard", "Extreme": "Extreme"
        }
        cat_map = {
            "人际关系": "Interpersonal", "观念认同": "Values", 
            "生涯发展": "Career", "身心健康": "Health",
            "休闲娱乐": "Leisure", "生活状况": "Lifestyle",
            "Interpersonal": "Interpersonal", "Values": "Values",
            "Career": "Career", "Health": "Health",
            "Leisure": "Leisure", "Lifestyle": "Lifestyle"
        }
        
        diff_val = case.get('difficulty', 'N/A')
        cat_val = case.get('category', 'N/A')
        
        metadata_map[sid] = {
            "Dominant Axis": case.get('dominant_axis', 'N/A'),
            "Difficulty": diff_map.get(diff_val, diff_val),
            "Category": cat_map.get(cat_val, cat_val)
        }
    return metadata_map

def extract_sp_features(script_id):
    """Extract SP features directly from markdown files."""
    sp_file = SP_DIR / f"{script_id}.md"
    info = {
        "Empathy Threshold": "N/A",
        "Affective Priority": "N/A",
        "Proactive Priority": "N/A",
        "Cognitive Priority": "N/A"
    }
    
    if not sp_file.exists():
        return info
        
    try:
        text = sp_file.read_text(encoding="utf-8")
    except:
        return info

    # 1. Threshold
    m = re.search(r"共情阈值【(.+?)】", text)
    if m:
        val = m.group(1).strip()
        # Map Chinese level to English
        cn_val = val[0] if val else "中"
        mapping = {"高": "High", "中": "Mid", "低": "Low"}
        info["Empathy Threshold"] = mapping.get(cn_val, "Mid")

    # 2. Priority Extraction
    def _extract_priority(label):
        val = "N/A"
        # Method 1: [优先级：High]
        m_local = re.search(label + r"：\[优先级：(.+?)\]", text)
        if m_local:
            val = m_local.group(1).strip()[0]
        else:
            # Method 2: High Priority ...
            m_line = re.search(label + r"：([^\n]+)", text)
            if m_line:
                seg = m_line.group(1).strip()
                if "高" in seg: val = "高"
                elif "中" in seg: val = "中"
                elif "低" in seg: val = "低"
        
        mapping = {"高": "High", "中": "Mid", "低": "Low", "N/A": "N/A"}
        return mapping.get(val, "N/A")

    emo = _extract_priority("情感共情")
    mot = _extract_priority("动机共情")
    cog = _extract_priority("认知共情")

    # Fallback logic
    if emo == "N/A" and mot == "N/A" and cog == "N/A":
        m_order = re.search(r"当下共情需求优先级[：:]\s*([^\n。]+)", text)
        if m_order:
            order_str = m_order.group(1)
            parts = re.split(r"[>＞]", order_str)
            level_map = {0: "High", 1: "Mid", 2: "Low"}
            for idx, part in enumerate(parts):
                lv = level_map.get(idx, "Low")
                if "情感共情" in part: emo = lv
                elif "动机共情" in part: mot = lv
                elif "认知共情" in part: cog = lv

    # Default to Mid if still N/A to avoid blanks
    info["Affective Priority"] = emo if emo != "N/A" else "Mid"
    info["Proactive Priority"] = mot if mot != "N/A" else "Mid"
    info["Cognitive Priority"] = cog if cog != "N/A" else "Mid"
    
    return info

def determine_success(result_data):
    """
    Determine success based on epm_victory_analysis conditions.
    This replicates logic from generate_descriptive_statistics.py
    """
    if 'epm_victory_analysis' in result_data:
        # Sometimes it's directly in root
        victory_data = result_data['epm_victory_analysis']
    elif 'epj' in result_data and 'epm_victory_analysis' in result_data['epj']:
        # Sometimes nested in epj
        victory_data = result_data['epj']['epm_victory_analysis']
    else:
        # Fallback: Check termination reason text
        term_reason = result_data.get('termination_reason', '')
        if '成功' in term_reason or 'SUCCESS' in term_reason.upper():
            return True
        return False

    if not victory_data:
        return False

    conditions = victory_data.get('conditions', {})
    spatial_achieved = (
        conditions.get('geometric', {}).get('achieved', False) or
        conditions.get('positional', {}).get('achieved', False)
    )
    energy_achieved = conditions.get('energetic', {}).get('achieved', False)
    
    return spatial_achieved and energy_achieved

def get_termination_reason(result_data, is_success):
    """
    Get termination reason.
    For success: Returns 'Success' or specific victory type.
    For failure: Returns simplified failure category.
    """
    if is_success:
        # Extract all achieved conditions
        victory_data = None
        if 'epm_victory_analysis' in result_data:
            victory_data = result_data['epm_victory_analysis']
        elif 'epj' in result_data and 'epm_victory_analysis' in result_data['epj']:
            victory_data = result_data['epj']['epm_victory_analysis']
            
        if victory_data:
            conditions = victory_data.get('conditions', {})
            achieved = []
            
            if conditions.get('geometric', {}).get('achieved', False):
                achieved.append("Geometric")
            if conditions.get('positional', {}).get('achieved', False):
                achieved.append("Positional")
            if conditions.get('energetic', {}).get('achieved', False):
                achieved.append("Energetic")
                
            if achieved:
                return f"Success ({' & '.join(achieved)})"
        
        return "Success"
        
    reason = result_data.get('termination_reason', '')
    if not reason:
        return "Unknown"
        
    if '方向崩溃' in reason:
        return 'Directional Collapse'
    elif '停滞' in reason:
        return 'Stagnation'
    elif '持续倒退' in reason:
        return 'Regression'
    elif '超时' in reason or 'MAX_TURNS' in reason:
        return 'Timeout'
    elif '能量不足' in reason:
        return 'Energy Depletion'
    else:
        # If no keyword matched, return truncated reason
        return reason[:30] + "..." if len(reason) > 30 else reason

def load_model_results(model_folder):
    """Load results directly from summary.json for reliability."""
    summary_file = RESAMPLED_RESULTS_DIR / model_folder / "summary.json"
    results = {}
    
    if not summary_file.exists():
        print(f"Warning: Summary file not found for {model_folder}")
        return results
        
    try:
        with open(summary_file, 'r', encoding='utf-8') as f:
            summary = json.load(f)
            
        for res in summary.get('results', []):
            sid = res.get('script_id', 'N/A')
            if sid == 'N/A': continue
            
            # Calculate success status
            is_success = determine_success(res)
            
            # Get turns
            turns = res.get('total_turns', 0)
            
            # Get simplified reason (success or failure)
            reason_str = get_termination_reason(res, is_success)
            
            results[sid] = {
                "Result": "✅ Success" if is_success else "❌ Failure",
                "Turns": turns,
                "Reason": reason_str
            }
            
    except Exception as e:
        print(f"Error loading results for {model_folder}: {e}")
        
    return results

def main():
    print("🔄 Starting aggregation (from summary.json)...")
    metadata_map = load_case_metadata()
    
    # Initialize DataFrame rows
    rows = []
    
    # Pre-load results for all models
    all_model_results = {}
    for model in MODELS:
        print(f"   Loading {model}...")
        all_model_results[model] = load_model_results(model)
        
    # Sort script IDs
    sorted_ids = sorted(metadata_map.keys())
    
    for sid in sorted_ids:
        meta = metadata_map[sid]
        sp_feats = extract_sp_features(sid)
        
        # Base Row Data (Case Info)
        row = {
            "Case ID": sid,
            "Dominant Axis": meta["Dominant Axis"],
            "Difficulty": meta["Difficulty"],
            "Category": meta["Category"],
            "Empathy Threshold": sp_feats["Empathy Threshold"],
            "Affective Priority": sp_feats["Affective Priority"],
            "Proactive Priority": sp_feats["Proactive Priority"],
            "Cognitive Priority": sp_feats["Cognitive Priority"]
        }
        
        # Append Model Results
        for model in MODELS:
            disp_name = MODEL_DISPLAY_NAMES.get(model, model)
            res = all_model_results[model].get(sid, {"Result": "N/A", "Turns": "N/A", "Reason": "N/A"})
            
            row[f"{disp_name} - Result"] = res["Result"]
            row[f"{disp_name} - Turns"] = res["Turns"]
            row[f"{disp_name} - Reason"] = res["Reason"]
            
        rows.append(row)
        
    df_out = pd.DataFrame(rows)
    
    # Create Multi-level Header for Excel
    new_columns = []
    case_info_cols = [
        "Case ID", "Dominant Axis", "Difficulty", "Category", 
        "Empathy Threshold", "Affective Priority", "Proactive Priority", "Cognitive Priority"
    ]
    
    header_tuples = []
    for col in case_info_cols:
        header_tuples.append(("Case Metadata", col))
        
    for model in MODELS:
        disp_name = MODEL_DISPLAY_NAMES.get(model, model)
        header_tuples.append((disp_name, "Result"))
        header_tuples.append((disp_name, "Turns"))
        header_tuples.append((disp_name, "Reason")) # Changed from "Failure Reason"
        
    # Reorder data
    final_rows = []
    for _, row_data in df_out.iterrows():
        new_row = []
        for col in case_info_cols:
            new_row.append(row_data[col])
        for model in MODELS:
            disp_name = MODEL_DISPLAY_NAMES.get(model, model)
            new_row.append(row_data[f"{disp_name} - Result"])
            new_row.append(row_data[f"{disp_name} - Turns"])
            new_row.append(row_data[f"{disp_name} - Reason"])
        final_rows.append(new_row)
        
    df_final = pd.DataFrame(final_rows)
    df_final.columns = pd.MultiIndex.from_tuples(header_tuples)
    
    # Save to Excel
    print(f"💾 Saving to {OUTPUT_FILE}...")
    with pd.ExcelWriter(OUTPUT_FILE, engine='openpyxl') as writer:
        df_final.to_excel(writer, sheet_name="Comparison Table")
        
        # Auto-adjust column widths
        worksheet = writer.sheets['Comparison Table']
        from openpyxl.utils import get_column_letter
        
        for i, col in enumerate(worksheet.columns, 1):
            max_length = 0
            column_letter = get_column_letter(i)
            for cell in col:
                try:
                    if cell.value:
                        cell_len = len(str(cell.value))
                        if cell_len > max_length:
                            max_length = cell_len
                except:
                    pass
            adjusted_width = (max_length + 2) * 1.1
            worksheet.column_dimensions[column_letter].width = min(max(adjusted_width, 10), 40)

    print("✅ Done!")

if __name__ == "__main__":
    main()
