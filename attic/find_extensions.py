#!/usr/bin/env python3
import os
from collections import defaultdict
from pathlib import Path
import argparse


def main():
    parser = argparse.ArgumentParser(description='Find all file extensions in a directory')
    parser.add_argument('path', nargs='?', default='.',
                        help='Directory path to scan (default: current directory)')
    parser.add_argument('-a', '--all', action='store_true',
                        help='Include hidden files and directories')
    parser.add_argument('-c', '--count', action='store_true',
                        help='Show count of files per extension')
    parser.add_argument('-s', '--sort-count', action='store_true',
                        help='Sort by count instead of alphabetically')

    args = parser.parse_args()

    # Validate directory
    if not os.path.isdir(args.path):
        print(f"Error: '{args.path}' is not a valid directory")
        return 1

    # Initialize counter
    extensions = defaultdict(int)

    # Walk through directory
    for root, dirs, files in os.walk(args.path):
        # Skip hidden directories unless -a flag is used
        if not args.all:
            dirs[:] = [d for d in dirs if not d.startswith('.')]
            files = [f for f in files if not f.startswith('.')]

        for file in files:
            ext = Path(file).suffix.lower()
            if ext:
                extensions[ext[1:]] += 1  # Remove the leading dot
            else:
                extensions['(no extension)'] += 1

    # No files found
    if not extensions:
        print("No files found in the specified directory.")
        return 0

    # Prepare for display
    if args.sort_count:
        # Sort by count (descending) and then by extension name
        items = sorted(extensions.items(), key=lambda x: (-x[1], x[0]))
    else:
        # Sort alphabetically by extension
        items = sorted(extensions.items())

    # Display results
    print(f"\nExtensions found in: {os.path.abspath(args.path)}")
    print("-" * 40)

    if args.count:
        # Show with counts
        max_ext_len = max(len(ext) for ext in extensions.keys())
        for ext, count in items:
            print(f"{ext:<{max_ext_len}} : {count:>5} files")
    else:
        # Show just extensions
        for ext, _ in items:
            print(ext)

    # Print summary
    total_files = sum(extensions.values())
    total_extensions = len(extensions)
    print("-" * 40)
    print(f"Total: {total_files} files, {total_extensions} unique extensions")


if __name__ == "__main__":
    exit(main())
