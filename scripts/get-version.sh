#!/bin/sh

set -eu

SCRIPT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
VERSION_HEADER="${SCRIPT_DIR}/../NVEncCore/rgy_version.h"

MATCH_COUNT=$(sed -n 's/^#define VER_STR_FILEVERSION[ \t]*"\([^"]*\)".*$/\1/p' "$VERSION_HEADER" | wc -l | tr -d ' ')
if [ "$MATCH_COUNT" -ne 1 ]; then
    echo "ERROR: VER_STR_FILEVERSION must be defined exactly once in $VERSION_HEADER." >&2
    exit 1
fi

VERSION=$(sed -n 's/^#define VER_STR_FILEVERSION[ \t]*"\([^"]*\)".*$/\1/p' "$VERSION_HEADER" | tr -d '\r')
if ! printf '%s\n' "$VERSION" | grep -Eq '^[0-9]+\.[0-9]+$'; then
    echo "ERROR: VER_STR_FILEVERSION ('$VERSION') must match <major>.<minor>." >&2
    exit 1
fi

printf '%s\n' "$VERSION"
