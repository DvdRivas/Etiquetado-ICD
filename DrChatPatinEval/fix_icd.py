import pandas as pd
import ast
import re
import sys

def truncate_icd(code: str) -> str:
    return code.split(".")[0] if isinstance(code, str) else code

def flatten(lst) -> list:
    result = []
    for item in lst:
        if isinstance(item, list):
            result.extend(flatten(item))
        else:
            result.append(str(item))
    return result

def parse_gt(raw) -> str:
    """Extract and truncate ICD code from ground_truth_icd (bare string, quoted, or list)."""
    if pd.isna(raw):
        return raw
    s = str(raw).strip()
    try:
        parsed = ast.literal_eval(s)
        if isinstance(parsed, list):
            code = str(parsed[0]) if parsed else ""
        else:
            code = str(parsed)
    except Exception:
        code = s.strip("[]\"' ")
    return '"' + truncate_icd(code) + '"'

def clean_icd_list(raw) -> list:
    """Parse icd_codes list and truncate each code."""
    if pd.isna(raw) or str(raw).strip() == "":
        return []
    try:
        codes = ast.literal_eval(str(raw))
        if not isinstance(codes, list):
            codes = [codes]
        codes = flatten(codes)
    except Exception:
        codes = re.findall(r'"([^"]+)"', str(raw))
    return [truncate_icd(c) for c in codes]

def clean_chapters(raw) -> list:
    """Parse icd_chapters list and remove 'Unknown' entries."""
    if pd.isna(raw) or str(raw).strip() == "":
        return []
    try:
        items = ast.literal_eval(str(raw))
        if not isinstance(items, list):
            items = [items]
        items = flatten(items)
    except Exception:
        items = re.findall(r'"([^"]+)"', str(raw))
    return [i for i in items if i.strip().lower() != "unknown"]

def unique_ordered(lst: list) -> list:
    seen = set()
    result = []
    for item in lst:
        if item not in seen:
            seen.add(item)
            result.append(item)
    return result

def to_json_list(lst: list) -> str:
    return str(lst).replace("'", '"') if lst else "[]"

def process(input_path: str, output_path: str):
    df = pd.read_excel(input_path, dtype=str)

    # 1. Fix ground_truth_icd
    if "ground_truth_icd" in df.columns:
        df["ground_truth_icd"] = df["ground_truth_icd"].apply(parse_gt)

    # 2. Truncate icd_codes and rewrite icd_categories
    if "icd_codes" in df.columns:
        cleaned_lists = df["icd_codes"].apply(clean_icd_list)
        df["icd_codes"] = cleaned_lists.apply(to_json_list)

        if "icd_categories" in df.columns:
            df["icd_categories"] = cleaned_lists.apply(
                lambda lst: to_json_list(unique_ordered(lst))
            )

    # 3. Truncate codes inside icd_raw_response JSON
    if "icd_raw_response" in df.columns:
        import json
        def clean_raw_response(v):
            if pd.isna(v) or str(v).strip() == "":
                return v
            try:
                obj = json.loads(str(v))
                if "icd_codes" in obj and isinstance(obj["icd_codes"], list):
                    obj["icd_codes"] = [truncate_icd(c) for c in obj["icd_codes"]]
                return json.dumps(obj)
            except Exception:
                return re.sub(r'("[\w]+)\.\d+(")', r'\1\2', str(v))
        df["icd_raw_response"] = df["icd_raw_response"].apply(clean_raw_response)

    # 4. Remove "Unknown" from icd_chapters
    if "icd_chapters" in df.columns:
        df["icd_chapters"] = df["icd_chapters"].apply(
            lambda v: to_json_list(clean_chapters(v)) if pd.notna(v) else v
        )

    df.to_excel(output_path, index=False)
    print(f"Done → {output_path}")

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python fix_icd.py <input.xlsx> <output.xlsx>")
        sys.exit(1)
    process(sys.argv[1], sys.argv[2])