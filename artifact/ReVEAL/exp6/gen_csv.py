#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import csv
import os
import sys

STAGE2_TXT = "pred_stage2_resyn3_U.txt"
STAGE3_TXT = "pred_stage3_resyn3_U.txt"
OUTPUT_CSV = "pred_stage3_dc2_U_64.csv"

# stage2 id -> 
id_to_part2 = {
    0: "4to2",
    1: "DT",
    2: "WT",
    3: "CWT",
    4: "AR",
}

# stage3 id ->  reverse_mapping
id_to_part3 = {
    0: "RC",
    1: "SE",
    2: "CL",
    3: "CK",
    4: "HCA",
    5: "LF",
    6: "KS",
    7: "BK",
    8: "JCA",
}

def read_stage2(path):
    m = {}
    with open(path, "r", encoding="utf-8") as f:
        for ln, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            if "," not in line:
                print(f"[WARN][stage2] Line {ln} malformed: {line}", file=sys.stderr)
                continue
            name, sid = line.split(",", 1)
            name = name.strip()
            try:
                sid = int(sid.strip())
            except ValueError:
                print(f"[WARN][stage2] Line {ln} invalid id: {line}", file=sys.stderr)
                continue
            m[name] = sid
    return m

def read_stage3(path):
    m = {}
    with open(path, "r", encoding="utf-8") as f:
        for ln, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            parts = line.split(",")
            if len(parts) != 4:
                print(f"[WARN][stage3] Line {ln} malformed: {line}", file=sys.stderr)
                continue
            name = parts[0].strip()
            try:
                t1 = int(parts[1].strip())
                t2 = int(parts[2].strip())
                t3 = int(parts[3].strip())
            except ValueError:
                print(f"[WARN][stage3] Line {ln} invalid ids: {line}", file=sys.stderr)
                continue
            m[name] = (t1, t2, t3)
    return m

def locate_indices(parts):
    """
    bits_bits_U*_<part2>_<part3>_Multgen|GenMul_<suffix>
     'Multgen'  'GenMul'  m_idx
     part3  m_idx-1part2  m_idx-2
    """
    m_idx = None
    for key in ("Multgen", "GenMul"):
        if key in parts:
            m_idx = parts.index(key)
            break
    if m_idx is None:
        return None, None, None
    p3_idx = m_idx - 1
    p2_idx = m_idx - 2
    if p2_idx < 0 or p3_idx < 0:
        return None, None, m_idx
    return p2_idx, p3_idx, m_idx

def replace_part2(name, new_part2):
    parts = name.split("_")
    p2_idx, p3_idx, m_idx = locate_indices(parts)
    if p2_idx is None:
        return name
    parts[p2_idx] = new_part2
    return "_".join(parts)

def replace_part3(name, new_part3):
    parts = name.split("_")
    p2_idx, p3_idx, m_idx = locate_indices(parts)
    if p3_idx is None:
        return name
    parts[p3_idx] = new_part3
    return "_".join(parts)

def swap_suffix(name, new_suffix):
    parts = name.split("_")
    if not parts:
        return name
    parts[-1] = new_suffix
    return "_".join(parts)

def main():
    if not os.path.exists(STAGE2_TXT):
        print(f"[ERROR] Missing {STAGE2_TXT}", file=sys.stderr)
        sys.exit(1)
    if not os.path.exists(STAGE3_TXT):
        print(f"[ERROR] Missing {STAGE3_TXT}", file=sys.stderr)
        sys.exit(1)

    stage2 = read_stage2(STAGE2_TXT)   # {basename: stage2_id}
    stage3 = read_stage3(STAGE3_TXT)   # {basename: (top1, top2, top3)}

    rows = []
    for base, (t1, t2, t3) in stage3.items():
        col1 = base

        #  stage2  id
        sid = stage2.get(base, None)
        if sid is None:
            print(f"[WARN] {base} not found in stage2 file, skip.", file=sys.stderr)
            continue
        if sid not in id_to_part2:
            print(f"[WARN] stage2 id {sid} not in mapping for {base}, skip.", file=sys.stderr)
            continue
        part2_txt = id_to_part2[sid]

        # 1 part2  stage2 
        base_with_stage2 = replace_part2(col1, part2_txt)

        #  stage3  top1/top2/top3 id 
        try:
            p3_1 = id_to_part3[t1]
            p3_2 = id_to_part3[t2]
            p3_3 = id_to_part3[t3]
        except KeyError:
            print(f"[WARN] stage3 ids not in mapping for {base}: {t1},{t2},{t3}", file=sys.stderr)
            continue

        #  2/3/4 part3 default
        col2 = swap_suffix(replace_part3(base_with_stage2, p3_1), "default")
        col3 = swap_suffix(replace_part3(base_with_stage2, p3_2), "default")
        col4 = swap_suffix(replace_part3(base_with_stage2, p3_3), "default")

        rows.append([col1, col2, col3, col4])

    with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["Filename", "Column2", "Column3", "Column4"])
        writer.writerows(rows)

    print(f"Done. Wrote {len(rows)} rows to {OUTPUT_CSV}")

if __name__ == "__main__":
    main()