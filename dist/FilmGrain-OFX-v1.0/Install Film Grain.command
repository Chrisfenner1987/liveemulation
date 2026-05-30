#!/bin/bash
# Film Grain (Stochastic) OFX - macOS installer
BUNDLE="FilmGrain.ofx.bundle"
DIR="$(cd "$(dirname "$0")" && pwd)"
SRC="$DIR/$BUNDLE"

echo "==============================================="
echo " Film Grain (Stochastic) - OFX plugin installer"
echo "==============================================="
echo

if [ ! -d "$SRC" ]; then
  echo "ERROR: '$BUNDLE' was not found next to this installer."
  echo "Keep this script in the same folder as the .ofx.bundle."
  echo; echo "Press Return to close."; read; exit 1
fi

# Remove the macOS download 'quarantine' flag so Resolve will load the plugin.
xattr -dr com.apple.quarantine "$SRC" 2>/dev/null

echo "Installing to:  /Library/OFX/Plugins/$BUNDLE"
echo "You'll be asked for your Mac password (required to write there)."
echo

osascript <<EOF
do shell script "mkdir -p /Library/OFX/Plugins && rm -rf '/Library/OFX/Plugins/$BUNDLE' && cp -R '$SRC' '/Library/OFX/Plugins/' && xattr -dr com.apple.quarantine '/Library/OFX/Plugins/$BUNDLE'" with administrator privileges
EOF

if [ $? -eq 0 ]; then
  echo
  echo "Installed successfully."
  echo "Now QUIT and REOPEN DaVinci Resolve (it scans OFX plugins only at launch)."
  echo "Find it under:  OpenFX  >  Fenner  >  Film Grain (Stochastic)"
else
  echo
  echo "Install was cancelled or failed."
fi
echo; echo "Press Return to close."; read
