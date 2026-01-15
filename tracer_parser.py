"""
To get the traces use 

```
with torch.profiler.profile(
    activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
    record_shapes=True
) as prof
```
as context manager.
To better annotate the functions use

```
with torch.autograd.profiler.emit_nvtx():
```
as the outer context manager in combination with 
```
with torch.profiler.record_function("Function Label"):
```
to record the functions both in CPU and CUDA.
"""

import json
import argparse
import csv
import os
from collections import defaultdict

def load_list_file(file_path):
    """Helper to load a list of strings from a file (one per line)."""
    if not file_path:
        return set()
    
    if not os.path.exists(file_path):
        print(f"Warning: File '{file_path}' not found. Ignoring.")
        return set()

    with open(file_path, 'r') as f:
        return {line.strip() for line in f if line.strip()}

def parse_and_export_filtered(args):
    file_path = args.file_path
    output_dir = args.output_dir
    
    # Metadata
    model_name = args.model_name
    ml_task = args.ml_task
    is_training = args.is_training

    print(f"Loading trace: {file_path}...")
    
    # Load Lists
    blacklist = load_list_file(args.blacklist)
    whitelist = load_list_file(args.whitelist)
    
    if blacklist:
        print(f"Loaded {len(blacklist)} operations to blacklist (exclude).")
    if whitelist:
        print(f"Loaded {len(whitelist)} custom operations to whitelist (force opaque leaf).")

    with open(file_path, 'r') as f:
        try:
            trace_data = json.load(f)
        except json.JSONDecodeError:
            print("Error: Invalid JSON.")
            return

    events = trace_data.get('traceEvents', [])
    if not events and isinstance(trace_data, list):
        events = trace_data

    # --- 1. Filter for CPU events ---
    cpu_events = []
    for e in events:
        if e.get('ph') != 'X':
            continue
            
        cat = e.get('cat', '').lower()
        name = e.get('name', '')
        
        # 1. Skip GPU Kernels (unless specifically whitelisted)
        if (cat == 'kernel' or 'cuda' in name.lower()) and name not in whitelist:
            continue
            
        # 2. INCLUSION LOGIC
        # Standard CPU op OR Whitelisted op
        is_standard_op = (cat == 'cpu_op' or name.startswith('aten::') or 'nn.module' in name.lower())
        is_whitelisted = (name in whitelist)
        
        if is_standard_op or is_whitelisted:
            cpu_events.append(e)

    # --- 2. Identify Leaf Operations (With Whitelist Exception) ---
    all_leaves = []
    thread_groups = defaultdict(list)
    
    for e in cpu_events:
        key = (e.get('pid'), e.get('tid'))
        thread_groups[key].append(e)

    print(f"Processing {len(cpu_events)} candidate events to find leaves...")

    for key, thread_events in thread_groups.items():
        # Sort by Start Time (asc), then Duration (desc)
        thread_events.sort(key=lambda x: (x['ts'], -x.get('dur', 0)))

        stack = []
        for e in thread_events:
            e['_has_children'] = False

        for e in thread_events:
            start = e['ts']
            
            # A. Pop finished parents from stack
            while stack:
                top = stack[-1]
                top_end = top['ts'] + top.get('dur', 0)
                if top_end <= start:
                    stack.pop()
                else:
                    break
            
            # B. Whitelist "Opaque" Check
            # If the immediate parent (top of stack) is a whitelisted op,
            # we treat it as an opaque leaf. We must IGNORE this current child event.
            if stack:
                top_parent_name = stack[-1].get('name', '')
                if top_parent_name in whitelist:
                    # Skip this child entirely. 
                    # Do not add to stack. Do not mark parent as having children.
                    continue

            # C. Standard Hierarchy Logic
            if stack:
                parent = stack[-1]
                parent['_has_children'] = True
            
            stack.append(e)

        # D. Collect Leaves
        for e in thread_events:
            # If an event was skipped in step B, it never entered the stack, 
            # so we won't encounter it here? 
            # wait, we are iterating `thread_events` (the original list).
            # We need to filter out events that we decided to "skip" in step B.
            # 
            # The easiest way: Only collect events where `_has_children` is False
            # AND (Crucially) ensure we don't pick up the "internal" children we wanted to ignore.
            
            # Problem: The logic above iterated `thread_events` to build hierarchy, 
            # but `e` in `thread_events` still exists. 
            # If we skipped processing it, `e['_has_children']` is still False (default).
            # So the child would be reported as a leaf! 
            
            # Fix: We need to know which events were effectively "consumed" by an opaque parent.
            pass

    # --- REVISED STEP 2: Strict Opaque Handling ---
    all_leaves = []
    
    for key, thread_events in thread_groups.items():
        thread_events.sort(key=lambda x: (x['ts'], -x.get('dur', 0)))
        
        stack = []
        # We'll use a set to track events that are "shadowed" by a whitelist op
        shadowed_indices = set()

        # Pre-pass: initialize flags
        for i, e in enumerate(thread_events):
            e['_has_children'] = False
            e['_index'] = i 

        for e in thread_events:
            start = e['ts']
            
            # Pop finished
            while stack:
                top = stack[-1]
                top_end = top['ts'] + top.get('dur', 0)
                if top_end <= start:
                    stack.pop()
                else:
                    break
            
            # Check Opaque
            is_shadowed = False
            if stack:
                top_parent = stack[-1]
                if top_parent.get('name') in whitelist:
                    is_shadowed = True
                    # Mark this current event as shadowed/ignored
                    shadowed_indices.add(e['_index'])
                else:
                    # Normal parent
                    top_parent['_has_children'] = True

            # If this event is shadowed, we don't push it to stack 
            # (because we don't want its children to reference it)
            if not is_shadowed:
                stack.append(e)

        # Collect valid leaves
        for i, e in enumerate(thread_events):
            if i in shadowed_indices:
                continue # It was inside a custom op
            if not e['_has_children']:
                all_leaves.append(e)


    # --- 3. Filter & Prepare CSV Data ---
    csv_rows = []
    skipped_empty = 0
    skipped_blacklist = 0

    all_leaves.sort(key=lambda x: x['ts'])

    for e in all_leaves:
        name = e.get('name', 'unknown')

        # Blacklist Check (High Priority)
        if name in blacklist:
            skipped_blacklist += 1
            continue
        
        event_args = e.get('args', {})
        raw_dims = event_args.get('Input Dims') or event_args.get('Input sizes') or event_args.get('shapes')
        
        # Empty Shape Check
        if isinstance(raw_dims, list):
            if not raw_dims: 
                skipped_empty += 1
                continue
            
            has_content = False
            for d in raw_dims:
                if d: 
                    has_content = True
                    break
            if not has_content:
                skipped_empty += 1
                continue

        # Prepare Data
        input_dims_str = str(raw_dims) if raw_dims is not None else "N/A"
        input_type = event_args.get('Input type') or event_args.get('Input dtypes') or event_args.get('dtypes') or "N/A"
        if not isinstance(input_type, str):
            input_type = str(input_type)

        csv_rows.append({
            'Model Name': model_name,
            'ML Task': ml_task,
            'Is Training': is_training,
            'Operation': name,
            'Input Dims': input_dims_str,
            'Input Type': input_type,
            'Duration': e.get('dur', 0)
        })

    # --- 4. Export ---
    base_name = os.path.basename(file_path)
    root_name = os.path.splitext(base_name)[0]
    csv_filename = f"{root_name}_ops.csv"
    
    if output_dir:
        if not os.path.exists(output_dir):
            try:
                os.makedirs(output_dir)
            except OSError as e:
                print(f"Error creating directory {output_dir}: {e}")
                return
        output_csv = os.path.join(output_dir, csv_filename)
    else:
        output_csv = csv_filename 

    print(f"\nExporting {len(csv_rows)} operations to {output_csv}...")

    fieldnames = ['Model Name', 'ML Task', 'Is Training', 'Operation', 'Input Dims', 'Input Type']

    with open(output_csv, 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        
        for row in csv_rows:
            writer.writerow({k: v for k, v in row.items() if k in fieldnames})
            
    # --- 5. Summary ---
    print(f"\n=== Summary of Operations in CSV ===")
    
    csv_stats = defaultdict(lambda: {'count': 0})
    for row in csv_rows:
        csv_stats[row['Operation']]['count'] += 1

    sorted_csv_ops = sorted(csv_stats.items(), key=lambda x: x[1]['count'], reverse=True)

    print(f"{'Operation':<60} | {'Count':<8} | {'% of CSV Rows':<12}")
    print("-" * 90)

    total_rows = len(csv_rows)
    if total_rows > 0:
        for name, data in sorted_csv_ops:
            count = data['count']
            percent = (count / total_rows) * 100
            print(f"{name[:58]:<60} | {count:<8} | {percent:<12.1f}")
    else:
        print("No operations exported.")

    print("\n--- Statistics ---")
    print(f"Total Leaves Found:    {len(all_leaves)}")
    print(f"Skipped (Blacklist):   {skipped_blacklist}")
    print(f"Skipped (Empty Shape): {skipped_empty}")
    print(f"Written to CSV:        {len(csv_rows)}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Parse PyTorch Traces to CSV with Metadata")
    
    parser.add_argument("file_path", type=str, help="Path to input trace .json file")
    parser.add_argument("--output-dir", type=str, default=None, help="Directory to save the output CSV")
    
    parser.add_argument("--blacklist", type=str, default=None, help="Path to text file with ops to EXCLUDE")
    parser.add_argument("--whitelist", type=str, default=None, help="Path to text file with custom ops to FORCE INCLUDE (Opaque)")

    parser.add_argument("--model-name", type=str, required=True, help="Name of the model")
    parser.add_argument("--ml-task", type=str, required=True, help="Task solved by the model")
    parser.add_argument("--is-training", type=str, required=True, help="Flag: True if training, False otherwise")

    args = parser.parse_args()
    parse_and_export_filtered(args)
