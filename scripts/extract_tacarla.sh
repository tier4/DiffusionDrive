#!/bin/bash
# One-time extraction of TaCarla sensor tar.gz archives to NAS.
# Usage: bash scripts/extract_tacarla.sh [--subset N] [--town Town12|Town13]
#
# Extracts each tar.gz into its own directory under DEST_DIR.
# Skips already-extracted routes.

set -euo pipefail

SRC_DIR="/mnt/nas/private_workspace/chenglin/dataset/TaCarla"
DEST_DIR="/mnt/nas/private_workspace/chenglin/dataset/TaCarla_extracted"
SUBSET=""
TOWN=""

while [[ $# -gt 0 ]]; do
    case $1 in
        --subset) SUBSET="$2"; shift 2 ;;
        --town) TOWN="$2"; shift 2 ;;
        --src) SRC_DIR="$2"; shift 2 ;;
        --dest) DEST_DIR="$2"; shift 2 ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

mkdir -p "$DEST_DIR"

count=0
total_extracted=0

for town_dir in "$SRC_DIR"/Town*_sensors; do
    town_name=$(basename "$town_dir" | sed 's/_sensors//')

    # Filter by town if specified
    if [[ -n "$TOWN" && "$town_name" != "$TOWN" ]]; then
        continue
    fi

    echo "Processing $town_name..."
    tar_files=("$town_dir"/*.tar.gz)
    echo "  Found ${#tar_files[@]} archives"

    for tar_file in "${tar_files[@]}"; do
        route_name=$(basename "$tar_file" .tar.gz)
        route_dir="$DEST_DIR/$route_name"

        # Skip if already extracted
        if [[ -d "$route_dir" ]]; then
            continue
        fi

        echo "  Extracting: $route_name"
        mkdir -p "$route_dir"
        tar -xzf "$tar_file" -C "$route_dir"
        total_extracted=$((total_extracted + 1))

        count=$((count + 1))
        if [[ -n "$SUBSET" && $count -ge $SUBSET ]]; then
            echo "Reached subset limit of $SUBSET routes"
            break 2
        fi
    done
done

echo ""
echo "Done. Extracted $total_extracted new routes to $DEST_DIR"
echo "Total routes in destination: $(find "$DEST_DIR" -maxdepth 1 -type d | wc -l)"
