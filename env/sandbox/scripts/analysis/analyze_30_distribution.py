import json
import re
from collections import Counter

# File paths
json_path = 'Benchmark/topics/new_data/classification_results.json'
txt_path = 'runner/benchmark_cases/sampled_benchmark_30_new_ids.txt'

# Load IDs
with open(txt_path, 'r') as f:
    target_ids = {line.strip() for line in f if line.strip()}

# Load Classification Data
with open(json_path, 'r') as f:
    all_data = json.load(f)

# Filter and Analyze
scenario_counter = Counter()
emotion_full_counter = Counter()
emotion_category_counter = Counter()

found_ids = []

for item in all_data:
    sid = item['script_id']
    if sid in target_ids:
        found_ids.append(sid)
        categories = item['BASIC_CATEGORIES']
        
        # Extract Primary Scenario
        scenario_match = re.search(r'主要场景-\[(.*?)\]', categories)
        if scenario_match:
            scenarios = scenario_match.group(1).split(',')
            for s in scenarios:
                s = s.strip()
                if s: scenario_counter[s] += 1

        # Extract Primary Emotion
        emotion_match = re.search(r'主要情感-\[(.*?)\]', categories)
        if emotion_match:
            # Handle cases where commas might be inside parentheses (though seemingly not in this dataset)
            # Simple split by comma for now as the data looks clean
            emotions = emotion_match.group(1).split(',')
            for e in emotions:
                e = e.strip()
                if e:
                    emotion_full_counter[e] += 1
                    # Extract category (e.g., "负向情感" from "负向情感(xxx)")
                    if '(' in e:
                        cat = e.split('(')[0]
                        emotion_category_counter[cat] += 1
                    else:
                        emotion_category_counter[e] += 1

print(f"统计案例数: {len(found_ids)} / {len(target_ids)}")

print("\n### 场景分布 (Scenario Distribution)")
print("| 场景类型 | 数量 | 占比 |")
print("|---|---|---|")
total_scenarios = sum(scenario_counter.values())
for k, v in scenario_counter.most_common():
    print(f"| {k} | {v} | {v/total_scenarios:.1%} |")

print("\n### 情感分布 - 细分 (Emotion Distribution - Detailed)")
print("| 情感类型 (细分) | 数量 | 占比 |")
print("|---|---|---|")
total_emotions = sum(emotion_full_counter.values())
for k, v in emotion_full_counter.most_common():
    print(f"| {k} | {v} | {v/total_emotions:.1%} |")

print("\n### 情感分布 - 大类 (Emotion Distribution - Category)")
print("| 情感大类 | 数量 | 占比 |")
print("|---|---|---|")
total_cats = sum(emotion_category_counter.values())
for k, v in emotion_category_counter.most_common():
    print(f"| {k} | {v} | {v/total_cats:.1%} |")

