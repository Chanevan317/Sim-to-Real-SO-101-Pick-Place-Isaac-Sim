#!/usr/bin/env bash
# Add "SO-Arm Data Studio" to the applications menu and the desktop of the current user.
# Run it from your own terminal (not over ssh), so the desktop icon can be marked as trusted.
set -euo pipefail
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
chmod +x "$DIR/run_studio.sh"
NAME="soarm-data-studio.desktop"
ENTRY="[Desktop Entry]
Type=Application
Name=SO-Arm Data Studio
Comment=Record, review and convert robot demonstrations
Exec=$DIR/run_studio.sh
Icon=$DIR/web/icon.svg
Terminal=false
Categories=Science;
StartupWMClass=so101-data-studio"
APPS="$HOME/.local/share/applications"
DESKTOP="$(xdg-user-dir DESKTOP 2>/dev/null || echo "$HOME/Desktop")"
# Older versions used another file name: remove those entries so the old name does not linger.
rm -f "$APPS/so101-data-studio.desktop" "$DESKTOP/so101-data-studio.desktop"

mkdir -p "$APPS"
echo "$ENTRY" > "$APPS/$NAME"
# Make the app drawer pick the new entry up right away.
update-desktop-database "$APPS" 2>/dev/null || true

if [[ -d "$DESKTOP" ]]; then
  echo "$ENTRY" > "$DESKTOP/$NAME"
  chmod +x "$DESKTOP/$NAME"
  if ! gio set "$DESKTOP/$NAME" metadata::trusted true 2>/dev/null; then
    echo "Note: right-click the desktop icon and choose 'Allow Launching' once."
  fi
fi
echo "Installed: look for 'SO-Arm Data Studio' in the applications menu${DESKTOP:+ and on the desktop}."
