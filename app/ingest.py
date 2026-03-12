#!/usr/bin/env python3
"""
CLI script to ingest .txt and .md files from a folder into a JSONL file.
"""
import argparse
import hashlib
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Optional


MAX_FILE_SIZE = 200 * 1024  # 200 KB


def clean_text(text: str) -> str:
    """Clean text by removing extra whitespace and normalizing line breaks."""
    # Replace multiple spaces with single space
    text = " ".join(text.split())
    return text.strip()


def generate_doc_id(file_path: str) -> str:
    """Generate a unique document ID by hashing the file path."""
    return hashlib.sha256(file_path.encode('utf-8')).hexdigest()


def extract_title(text: str) -> str:
    """Extract title from the first line of text."""
    lines = text.strip().split('\n')
    if not lines:
        return "Untitled"
    
    title = lines[0].strip()
    # Remove markdown heading markers
    title = title.lstrip('#').strip()
    return title if title else "Untitled"


def process_file(file_path: Path) -> Optional[dict]:
    """Process a single file and return document data."""
    try:
        # Check file size
        file_size = file_path.stat().st_size
        if file_size > MAX_FILE_SIZE:
            print(f"Skipping {file_path} (size: {file_size / 1024:.1f} KB > 200 KB)")
            return None
        
        # Read file content
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        if not content.strip():
            print(f"Skipping {file_path} (empty file)")
            return None
        
        # Extract title and clean text
        title = extract_title(content)
        cleaned_text = clean_text(content)
        
        # Create document record
        doc = {
            "doc_id": generate_doc_id(str(file_path.absolute())),
            "title": title,
            "text": cleaned_text,
            "source": str(file_path.absolute()),
            "created_at": datetime.utcnow().isoformat()
        }
        
        return doc
    
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return None


def ingest_documents(input_dir: Path, output_file: Path) -> int:
    """
    Walk through input directory and ingest .txt and .md files.
    
    Args:
        input_dir: Directory to search for files
        output_file: Path to output JSONL file
    
    Returns:
        Number of documents processed
    """
    if not input_dir.exists():
        raise ValueError(f"Input directory does not exist: {input_dir}")
    
    if not input_dir.is_dir():
        raise ValueError(f"Input path is not a directory: {input_dir}")
    
    # Find all .txt and .md files
    file_patterns = ['**/*.txt', '**/*.md']
    all_files = []
    for pattern in file_patterns:
        all_files.extend(input_dir.glob(pattern))
    
    print(f"Found {len(all_files)} files to process")
    
    processed_count = 0
    
    # Process files and write to JSONL
    with open(output_file, 'w', encoding='utf-8') as out_f:
        for file_path in all_files:
            doc = process_file(file_path)
            if doc:
                out_f.write(json.dumps(doc) + '\n')
                processed_count += 1
                print(f"Processed: {file_path.name}")
    
    return processed_count


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Ingest .txt and .md files from a directory into a JSONL file"
    )
    parser.add_argument(
        "input_dir",
        type=str,
        help="Directory to search for .txt and .md files"
    )
    parser.add_argument(
        "output_file",
        type=str,
        help="Output JSONL file path"
    )
    
    args = parser.parse_args()
    
    input_dir = Path(args.input_dir)
    output_file = Path(args.output_file)
    
    print(f"Starting ingestion from: {input_dir}")
    print(f"Output file: {output_file}")
    print(f"Max file size: {MAX_FILE_SIZE / 1024} KB")
    print("-" * 50)
    
    try:
        count = ingest_documents(input_dir, output_file)
        print("-" * 50)
        print(f"✓ Successfully processed {count} documents")
        print(f"✓ Output written to: {output_file}")
    except Exception as e:
        print(f"✗ Error: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
