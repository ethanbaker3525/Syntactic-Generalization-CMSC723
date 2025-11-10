#!/bin/bash

# Directory containing the TSV files
INPUT_DIR="eval/baseline"

# Output directory
OUTPUT_DIR="output/results"

# Create output directory if it doesn't exist
mkdir -p "$OUTPUT_DIR"

# Loop through all TSV files in the input directory
for tsv_file in "$INPUT_DIR"/*.tsv; do
    # Check if any TSV files exist
    if [ ! -e "$tsv_file" ]; then
        echo "No TSV files found in $INPUT_DIR"
        exit 1
    fi
    
    # Get the filename for logging
    filename=$(basename "$tsv_file")
    
    echo "Processing: $filename"
    
    # Run the Python script
    python filler_gap_effect_analysis.py \
        --eval_file_path "$tsv_file" \
        --output_path "$OUTPUT_DIR"
    
    echo "Completed: $filename"
    echo "---"
done

echo "All files processed successfully!"