#!/usr/bin/env bash
# Step 065 link integrity — mdBook renders only what SUMMARY.md names and
# does NOT check inline links, so a moved or misspelled page 404s
# silently. Two checks over docs/:
#
#   1. Every relative link target in every .md file exists on disk
#      (http(s)/mailto and pure #anchors are skipped; a target's own
#      #fragment is stripped before the existence check).
#   2. Every page SUMMARY.md names exists.
#
# Run from the repo root: scripts/check-book-links.sh
set -euo pipefail

DOCS=docs
fail=0

# 1. Inline links.
while IFS= read -r file; do
  dir=$(dirname "$file")
  # markdown links: capture the (...) target of ](...) pairs.
  while IFS= read -r target; do
    case "$target" in
      http://*|https://*|mailto:*|\#*) continue ;;
    esac
    path="${target%%#*}"
    [ -z "$path" ] && continue
    if [ ! -e "$dir/$path" ]; then
      echo "MISSING: $file -> $target"
      fail=1
    fi
  done < <(grep -o '](\([^)]*\))' "$file" 2>/dev/null | sed 's/^](//;s/)$//')
done < <(find "$DOCS" -name '*.md' -not -path "$DOCS/book/*")

# 2. SUMMARY entries.
while IFS= read -r page; do
  if [ ! -e "$DOCS/$page" ]; then
    echo "SUMMARY names a missing page: $page"
    fail=1
  fi
done < <(grep -o '](\([^)]*\.md\))' "$DOCS/SUMMARY.md" | sed 's/^](//;s/)$//')

if [ "$fail" -ne 0 ]; then
  echo "book link check FAILED"
  exit 1
fi
echo "book links OK"
