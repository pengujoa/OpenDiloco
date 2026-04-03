#!/bin/bash
# Clone allenai/c4 under a chosen parent directory (creates .../c4).
#
# Use HTTP:
#   bash scripts/pull-c4.sh
#   bash scripts/pull-c4.sh /data/datasets
#   TARGET=/data/datasets bash scripts/pull-c4.sh
# Use GIT:
#   USE_GIT=1 bash scripts/pull-c4.sh /data/datasets
#
# Parent directory: TARGET env overrides first argument; both default to current directory.
set -e

DEST="${TARGET:-${1:-.}}"
mkdir -p "$DEST"
cd "$DEST"
echo "Destination parent: $(pwd) (repo will be ./c4)"

# The git server seems to be faster than the http server
# However, it requires ssh keys to be set up
if [ "$USE_GIT" = "1" ]; then
    echo "Using git"
    GIT_LFS_SKIP_SMUDGE=1 git clone git@hf.co:datasets/allenai/c4
else
    echo "Using http"
    GIT_LFS_SKIP_SMUDGE=1 git clone https://huggingface.co/datasets/allenai/c4
fi

echo "Pulling LFS files"
cd c4
git lfs pull -I 'en/*.json.gz'
