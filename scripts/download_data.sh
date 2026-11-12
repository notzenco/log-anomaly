#!/usr/bin/env bash
#
# Fetch the public datasets used by log-anomaly and verify their checksums.
#
# Usage:
#   scripts/download_data.sh [DEST_DIR] [DATASET]
#
#   DEST_DIR   where to put the archives and extracted logs (default: data/raw)
#   DATASET    linux | openssh | proxifier | all   (default: all)
#
# The script is deliberately dependency-light: curl plus md5sum/md5. It is the
# canonical download path and does not require the Python package to be
# installed. The checksums below are duplicated in
# src/log_anomaly/data/datasets.py; a test asserts that the two agree.
#
# Raw data is never committed to the repository.

set -euo pipefail

DEST="${1:-data/raw}"
WHICH="${2:-all}"

ZENODO_BASE="https://zenodo.org/api/records/8196385/files"

# name:archive:md5:member
DATASETS="linux:Linux.tar.gz:6d1802d7778126f21c001c6aa7b6b106:Linux.log
openssh:SSH.tar.gz:a44e40b4348697dabf2bba56885e0a38:SSH.log
proxifier:Proxifier.tar.gz:2612fce12cc3d16599ddb3db8e9c477a:Proxifier.log"

md5_of() {
  if command -v md5sum >/dev/null 2>&1; then
    md5sum "$1" | awk '{print $1}'
  elif command -v md5 >/dev/null 2>&1; then
    md5 -q "$1"
  else
    echo "error: no md5sum or md5 available" >&2
    exit 1
  fi
}

mkdir -p "${DEST}/archives" "${DEST}/logs"

while IFS=: read -r name archive expected member; do
  [ -n "${name}" ] || continue
  if [ "${WHICH}" != "all" ] && [ "${WHICH}" != "${name}" ]; then
    continue
  fi

  archive_path="${DEST}/archives/${archive}"

  if [ -f "${archive_path}" ]; then
    actual="$(md5_of "${archive_path}")"
    if [ "${actual}" = "${expected}" ]; then
      echo "[${name}] archive already present and verified, skipping download"
    else
      echo "[${name}] checksum mismatch on cached archive, refetching"
      rm -f "${archive_path}"
    fi
  fi

  if [ ! -f "${archive_path}" ]; then
    echo "[${name}] downloading ${archive} ..."
    curl -fsSL --retry 3 --retry-delay 2 \
      -o "${archive_path}.part" \
      "${ZENODO_BASE}/${archive}/content"
    mv "${archive_path}.part" "${archive_path}"
  fi

  actual="$(md5_of "${archive_path}")"
  if [ "${actual}" != "${expected}" ]; then
    echo "[${name}] CHECKSUM MISMATCH: expected ${expected}, got ${actual}" >&2
    rm -f "${archive_path}"
    exit 1
  fi
  echo "[${name}] checksum verified: ${actual}"

  if [ ! -f "${DEST}/logs/${member}" ]; then
    tar -xzf "${archive_path}" -C "${DEST}/logs" "${member}"
  fi
  echo "[${name}] extracted to ${DEST}/logs/${member}"
done <<EOF
${DATASETS}
EOF

echo "done: raw data under ${DEST}"
