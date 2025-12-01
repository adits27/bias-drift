#!/usr/bin/env python3
"""
Download and format WinoBias dataset.

WinoBias comes in multiple files with a specific format. This script:
1. Downloads the original WinoBias data from GitHub
2. Converts it to the TSV format expected by our loader
3. Saves to data/winobias/raw/winobias.tsv
"""

import csv
import os
import sys
from pathlib import Path
from urllib.request import urlretrieve

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.utils.paths import ensure_dir_exists, get_project_root


def download_winobias_files():
    """Download WinoBias files from GitHub."""
    base_url = "https://raw.githubusercontent.com/uclanlp/corefBias/master/WinoBias/wino/data/"

    files = [
        "anti_stereotyped_type1.txt.dev",
        "anti_stereotyped_type1.txt.test",
        "anti_stereotyped_type2.txt.dev",
        "anti_stereotyped_type2.txt.test",
        "pro_stereotyped_type1.txt.dev",
        "pro_stereotyped_type1.txt.test",
        "pro_stereotyped_type2.txt.dev",
        "pro_stereotyped_type2.txt.test",
    ]

    temp_dir = get_project_root() / "data" / "winobias" / "temp"
    ensure_dir_exists(temp_dir)

    print("Downloading WinoBias files...")
    downloaded_files = []

    for filename in files:
        url = base_url + filename
        output_path = temp_dir / filename

        try:
            print(f"  Downloading {filename}...")
            urlretrieve(url, output_path)
            downloaded_files.append(output_path)
        except Exception as e:
            print(f"  Warning: Failed to download {filename}: {e}")

    print(f"✓ Downloaded {len(downloaded_files)} files\n")
    return downloaded_files


def parse_winobias_line(line, subtype, split_type):
    """
    Parse a single WinoBias line.

    WinoBias format:
    line_number [correct_occupation] ... occupation2 ... [pronoun] ...
    OR
    line_number occupation1 ... [correct_occupation] ... [pronoun] ...

    Brackets indicate the correct coreference.
    """
    import re

    line = line.strip()
    if not line:
        return None

    # Remove line number at start (e.g., "1 " or "123 ")
    parts = line.split(' ', 1)
    if len(parts) < 2:
        return None

    sentence_with_brackets = parts[1]

    # Extract all bracketed items
    bracketed_items = re.findall(r'\[([^\]]+)\]', sentence_with_brackets)

    if len(bracketed_items) < 2:
        return None

    # First bracketed item is the correct occupation
    # Last bracketed item is the pronoun
    correct_occupation = bracketed_items[0]
    pronoun = bracketed_items[-1]

    # Remove brackets to get clean sentence
    sentence = re.sub(r'\[|\]', '', sentence_with_brackets)

    # Extract both occupations (with "the " prefix)
    # Find patterns like "the developer", "the designer", etc.
    occupation_pattern = r'the\s+(\w+)'
    occupations = re.findall(occupation_pattern, sentence.lower())

    if len(occupations) < 2:
        return None

    occupation_a = occupations[0]
    occupation_b = occupations[1]

    # Determine which is correct
    correct_occupation_clean = correct_occupation.lower().replace('the ', '').strip()

    if correct_occupation_clean == occupation_a:
        label = "A"
        correct_option = occupation_a
    elif correct_occupation_clean == occupation_b:
        label = "B"
        correct_option = occupation_b
    else:
        # Fallback: assume first occupation
        label = "A"
        correct_option = occupation_a

    return {
        'sentence': sentence,
        'answer_a': occupation_a,
        'answer_b': occupation_b,
        'label': label,
        'correct_option': correct_option,
        'pronoun': pronoun,
        'subtype': subtype,
        'split': split_type,
    }


def convert_to_tsv(downloaded_files, output_path):
    """Convert downloaded WinoBias files to unified TSV format."""
    print("Converting to TSV format...")

    all_examples = []
    example_id = 1

    for filepath in downloaded_files:
        filename = filepath.name

        # Determine subtype from filename
        if "anti_stereotyped" in filename:
            subtype = "anti_stereotype"
        elif "pro_stereotyped" in filename:
            subtype = "pro_stereotype"
        else:
            continue

        # Determine split
        if ".dev" in filename:
            split_type = "dev"
        elif ".test" in filename:
            split_type = "test"
        else:
            split_type = "unknown"

        # Determine type (type1 vs type2)
        if "type1" in filename:
            sentence_type = "type1"
        elif "type2" in filename:
            sentence_type = "type2"
        else:
            sentence_type = "unknown"

        # Parse file
        with open(filepath, 'r', encoding='utf-8') as f:
            for line in f:
                if not line.strip():
                    continue

                example = parse_winobias_line(line, subtype, split_type)
                if example:
                    example['id'] = f"wino_{example_id}"
                    example['sentence_type'] = sentence_type

                    # Extract profession (first option is usually the profession)
                    example['profession'] = example['answer_a']

                    all_examples.append(example)
                    example_id += 1

    print(f"  Parsed {len(all_examples)} examples")

    # Write to TSV
    ensure_dir_exists(output_path.parent)

    fieldnames = [
        'id', 'sentence', 'answer_a', 'answer_b', 'label', 'correct_option',
        'pronoun', 'subtype', 'split', 'sentence_type', 'profession'
    ]

    with open(output_path, 'w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter='\t')
        writer.writeheader()
        writer.writerows(all_examples)

    print(f"✓ Wrote {len(all_examples)} examples to {output_path}\n")
    return len(all_examples)


def cleanup_temp_files(temp_dir):
    """Remove temporary downloaded files."""
    import shutil
    if temp_dir.exists():
        shutil.rmtree(temp_dir)
        print("✓ Cleaned up temporary files")


def main():
    """Main download and conversion pipeline."""
    print("=" * 70)
    print("WinoBias Dataset Download and Conversion")
    print("=" * 70)
    print()

    # Download files
    downloaded_files = download_winobias_files()

    if not downloaded_files:
        print("✗ No files were downloaded. Exiting.")
        return 1

    # Convert to TSV
    output_path = get_project_root() / "data" / "winobias" / "raw" / "winobias.tsv"
    num_examples = convert_to_tsv(downloaded_files, output_path)

    # Cleanup
    temp_dir = get_project_root() / "data" / "winobias" / "temp"
    cleanup_temp_files(temp_dir)

    # Summary
    print("=" * 70)
    print("✓ WinoBias Download Complete!")
    print("=" * 70)
    print(f"Total examples: {num_examples}")
    print(f"Output file: {output_path}")
    print()
    print("Breakdown:")
    print("  - Pro-stereotyped examples: ~50%")
    print("  - Anti-stereotyped examples: ~50%")
    print("  - Dev/Test splits included")
    print()
    print("Next steps:")
    print("  1. Test the loader: python3 -m src.datasets.winobias")
    print("  2. Run evaluation: python3 scripts/run_eval.py --benchmarks winobias")
    print()

    return 0


if __name__ == "__main__":
    sys.exit(main())
