#!/bin/bash
# Copy the committed code of HEAD to LUMI as /scratch/project_462001534/feature_geometry/code/<short commit>/ (git archive,
# so only tracked files; CODE_COMMIT holds the full hash) and compare the sha256 of every file with the local commit.
# Refuses a dirty working tree and an existing target folder. One ssh connection per step (login nodes rotate).
#
# Usage: scripts/local/sync_code_to_lumi.sh            (from the repository root, ssh agent loaded)
set -euo pipefail
HOST=muhamkha@lumi.csc.fi
BASE=/scratch/project_462001534/feature_geometry/code
[ -z "$(git status --porcelain --untracked-files=no)" ] || { echo "uncommitted changes; commit first" >&2; exit 1; }
FULL=$(git rev-parse HEAD)
SHORT=$(git rev-parse --short HEAD)
DEST=$BASE/$SHORT
ssh -o BatchMode=yes $HOST "test ! -e $DEST" || { echo "$DEST exists" >&2; exit 1; }
git archive --format=tar HEAD | ssh -o BatchMode=yes $HOST "mkdir -p $DEST && tar -x -C $DEST && echo $FULL > $DEST/CODE_COMMIT"
LOCAL=$(git ls-files -z | xargs -0 sha256sum | sort -k2 | sha256sum | cut -d' ' -f1)
REMOTE=$(ssh -o BatchMode=yes $HOST "cd $DEST && git_files=\$(find . -type f ! -name CODE_COMMIT | sed 's|^\./||' | sort) && \
    echo \"\$git_files\" | tr '\n' '\0' | xargs -0 sha256sum | sort -k2 | sha256sum | cut -d' ' -f1")
N=$(git ls-files | wc -l)
if [ "$LOCAL" = "$REMOTE" ]; then echo "synced $SHORT to $DEST: $N files, sha256 list identical"; else
    echo "MISMATCH: local $LOCAL remote $REMOTE" >&2; exit 1; fi
