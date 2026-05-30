#!/bin/bash
# Film Grain (Stochastic) OFX - macOS uninstaller
BUNDLE="FilmGrain.ofx.bundle"
echo "Removing /Library/OFX/Plugins/$BUNDLE ..."
echo "You'll be asked for your Mac password."
echo
osascript <<EOF
do shell script "rm -rf '/Library/OFX/Plugins/$BUNDLE'" with administrator privileges
EOF
if [ $? -eq 0 ]; then echo "Removed. Restart DaVinci Resolve."; else echo "Cancelled or failed."; fi
echo; echo "Press Return to close."; read
