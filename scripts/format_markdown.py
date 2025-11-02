#!/usr/bin/env python3
"""
Format markdown files to 80 character line length.

Usage:
    ./scripts/format_markdown.py docs/aureus.md
    ./scripts/format_markdown.py docs/*.md
"""

import sys
import re
import textwrap
from pathlib import Path


def format_markdown(content: str, max_length: int = 80) -> str:
    """
    Format markdown content to max line length.

    Preserves:
    - Code blocks (```)
    - Block quotes (>)
    - Lists (* or -)
    - Headers (#)
    - Tables (|)
    - Horizontal rules (---)
    """
    lines = content.split("\n")
    formatted = []
    in_code_block = False
    in_table = False

    i = 0
    while i < len(lines):
        line = lines[i]

        # Toggle code block state
        if line.strip().startswith("```"):
            in_code_block = not in_code_block
            formatted.append(line)
            i += 1
            continue

        # Preserve code blocks as-is
        if in_code_block:
            formatted.append(line)
            i += 1
            continue

        # Detect table
        if "|" in line and line.strip().startswith("|"):
            in_table = True
        elif in_table and "|" not in line:
            in_table = False

        # Preserve tables as-is
        if in_table:
            formatted.append(line)
            i += 1
            continue

        # Preserve horizontal rules
        if re.match(r"^-{3,}$", line.strip()):
            formatted.append(line)
            i += 1
            continue

        # Preserve empty lines
        if not line.strip():
            formatted.append("")
            i += 1
            continue

        # Format different line types
        if line.lstrip().startswith("#"):
            # Headers - don't wrap, but warn if too long
            if len(line) > max_length:
                print(f"Warning: Header too long: {line[:50]}...", file=sys.stderr)
            formatted.append(line)

        elif line.lstrip().startswith(("* ", "- ", "+ ")):
            # List items - preserve indent, wrap content
            match = re.match(r"^(\s*[-*+]\s+)", line)
            if match:
                indent = match.group(1)
                content = line[len(indent) :]
                wrapped = textwrap.fill(
                    content,
                    width=max_length,
                    initial_indent=indent,
                    subsequent_indent=" " * len(indent),
                    break_long_words=False,
                    break_on_hyphens=False,
                )
                formatted.extend(wrapped.split("\n"))
            else:
                formatted.append(line)

        elif line.lstrip().startswith(">"):
            # Block quotes - preserve '>', wrap content
            match = re.match(r"^(\s*>\s*)", line)
            if match:
                indent = match.group(1)
                content = line[len(indent) :]
                wrapped = textwrap.fill(
                    content,
                    width=max_length,
                    initial_indent=indent,
                    subsequent_indent=indent,
                    break_long_words=False,
                    break_on_hyphens=False,
                )
                formatted.extend(wrapped.split("\n"))
            else:
                formatted.append(line)

        elif re.match(r"^\d+\.", line.lstrip()):
            # Numbered lists - preserve number, wrap content
            match = re.match(r"^(\s*\d+\.\s+)", line)
            if match:
                indent = match.group(1)
                content = line[len(indent) :]
                wrapped = textwrap.fill(
                    content,
                    width=max_length,
                    initial_indent=indent,
                    subsequent_indent=" " * len(indent),
                    break_long_words=False,
                    break_on_hyphens=False,
                )
                formatted.extend(wrapped.split("\n"))
            else:
                formatted.append(line)

        else:
            # Regular paragraph - wrap at 80 chars
            if len(line) <= max_length:
                formatted.append(line)
            else:
                # Check if line is indented
                indent_match = re.match(r"^(\s+)", line)
                if indent_match:
                    indent = indent_match.group(1)
                    content = line[len(indent) :]
                    wrapped = textwrap.fill(
                        content,
                        width=max_length,
                        initial_indent=indent,
                        subsequent_indent=indent,
                        break_long_words=False,
                        break_on_hyphens=False,
                    )
                    formatted.extend(wrapped.split("\n"))
                else:
                    wrapped = textwrap.fill(
                        line,
                        width=max_length,
                        break_long_words=False,
                        break_on_hyphens=False,
                    )
                    formatted.extend(wrapped.split("\n"))

        i += 1

    return "\n".join(formatted)


def main():
    if len(sys.argv) < 2:
        print("Usage: format_markdown.py <file.md> [file2.md ...]")
        print("   or: format_markdown.py docs/*.md")
        sys.exit(1)

    for filepath in sys.argv[1:]:
        path = Path(filepath)

        if not path.exists():
            print(f"Error: {filepath} not found", file=sys.stderr)
            continue

        if not path.suffix == ".md":
            print(f"Skipping non-markdown file: {filepath}", file=sys.stderr)
            continue

        print(f"Formatting {filepath}...")

        # Read file
        with open(path, "r") as f:
            content = f.read()

        # Format
        formatted = format_markdown(content, max_length=80)

        # Write back
        with open(path, "w") as f:
            f.write(formatted)

        print(f"  ✓ Formatted {filepath}")


if __name__ == "__main__":
    main()
