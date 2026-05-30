#!/usr/bin/env python3
"""
Filter CVE database to extract only entries that have patch code.
"""

import os
import argparse


def filter_cves_with_patches(input_file: str, output_file: str) -> int:
    """
    Read CVE database and write only entries with patch code to output file.

    Args:
        input_file: Path to input CVE database text file
        output_file: Path to write filtered output

    Returns:
        Number of CVEs with patch code found
    """
    with open(input_file, 'r', encoding='utf-8') as f:
        content = f.read()

    entries = content.split('=' * 80)
    header = entries[0]

    filtered_entries = []
    cves_with_patches = 0

    for entry in entries[1:]:
        if not entry.strip():
            continue
        if '--- Code from' in entry:
            filtered_entries.append(entry)
            cves_with_patches += 1

    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("CVE Database Export (Filtered - With Patches Only)\n")
        try:
            date_str = header.split('Generated: ')[1].split()[0] if 'Generated:' in header else 'N/A'
        except IndexError:
            date_str = 'N/A'
        f.write(f"Generated: {date_str}\n")
        f.write(f"Total CVEs with patches: {cves_with_patches}\n")
        f.write("=" * 80 + "\n")
        for entry in filtered_entries:
            f.write(entry)
            f.write("=" * 80 + "\n")

    return cves_with_patches


if __name__ == '__main__':
    _script_dir = os.path.dirname(os.path.abspath(__file__))

    parser = argparse.ArgumentParser(description="Filter CVE database to entries with patch code")
    parser.add_argument(
        "--input", "-i",
        default=os.path.join(_script_dir, "cve_database.txt"),
        help="Input CVE database file (default: CVE/cve_database.txt)"
    )
    parser.add_argument(
        "--output", "-o",
        default=os.path.join(_script_dir, "cve_database_with_patches.txt"),
        help="Output file (default: CVE/cve_database_with_patches.txt)"
    )
    args = parser.parse_args()

    print(f"Filtering CVEs from {args.input}...")
    count = filter_cves_with_patches(args.input, args.output)
    print(f"Found {count} CVEs with patch code")
    print(f"Saved to {args.output}")
