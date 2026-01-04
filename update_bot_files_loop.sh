#!/usr/bin/env bash
# SPDX-License-Identifier: MIT
#
# Update loop for selected tracked files only.
# No restart, no reset, no clean, no sqlite touch.
#

set -euo pipefail

# ---------------- CONFIG ----------------
REPO_DIR="$(pwd)"
REMOTE="origin"
BRANCH="main"

# FILES DA AGGIORNARE (SOLO QUESTI)
FILES=(
	"ldl.py"
)

SLEEP_SECONDS=10
# ---------------------------------------

log() {
	printf '[%s] %s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "$*"
}

die() {
	log "ERROR: $*"
	exit 1
}

cd "$REPO_DIR" || die "cannot cd into repo"

[[ -d .git ]] || die "not a git repository"
command -v git >/dev/null || die "git not installed"

log "starting update loop (every ${SLEEP_SECONDS}s)"

while true; do
	log "fetching from ${REMOTE}/${BRANCH}"
	git fetch "$REMOTE" "$BRANCH" >/dev/null 2>&1 || {
		log "fetch failed, retrying later"
		sleep "$SLEEP_SECONDS"
		continue
	}

	for file in "${FILES[@]}"; do
		# file must exist on remote
		if ! git cat-file -e "${REMOTE}/${BRANCH}:${file}" 2>/dev/null; then
			log "remote file missing: $file (skipped)"
			continue
		fi

		# if local file exists and is identical -> skip
		if [[ -f "$file" ]]; then
			if git show "${REMOTE}/${BRANCH}:${file}" | cmp -s - "$file"; then
				continue
			fi
		fi

		log "updating $file"
		tmp="$(mktemp)"
		git show "${REMOTE}/${BRANCH}:${file}" > "$tmp"
		chmod --reference="$file" "$tmp" 2>/dev/null || true
		mv "$tmp" "$file"
	done

	sleep "$SLEEP_SECONDS"
done
