# Regenerate docs/assets/demo.gif with `make demo` from the repository root.
# Requires: vhs, ttyd, jq, figlet, ffmpeg, and the qwen2.5-0.5b model
# registered locally (vipera model search, or vipera model import <path>).
#
# The recording is supersampled: 1800x1200 at 32px, then `make demo` downscales
# it to 900x600 with lanczos and a 128-colour palette. Rendering at twice the
# final size and resampling removes the grey fringes GIF text otherwise gets.
#
# The opening wordmark is figlet's "block" font (shipped with the base figlet
# package on Debian/Ubuntu and Homebrew). It is generated, not a bitmap: the
# same command reproduces it on any machine, which is why the GIF stays small.
# No `cd` here on purpose: vhs inherits the invoker's working directory, so the
# tape works on any clone instead of one hard-coded home path.
Output docs/assets/demo-raw.mp4
Set Shell "bash"
# VHS resolves "JetBrains Mono" to whatever mono font happens to be installed;
# name the Nerd Font explicitly so the glyphs match the documented dependency.
Set FontFamily "JetBrainsMono Nerd Font Mono"
Set FontSize 32
Set Width 1800
Set Height 1200
Set Theme "Catppuccin Mocha"
Set Padding 32
Set TypingSpeed 35ms
Set PlaybackSpeed 1.5
Set Framerate 15
Set CursorBlink false

Type "figlet -f block Vipera"
Enter
Sleep 800ms

Type "vipera --version"
Enter
Sleep 500ms

Type "vipera system doctor"
Enter
Sleep 3s

Type "vipera serve start --model qwen2.5-0.5b-instruct-q4_k_m --ctx 4096 >/tmp/vhs-serve.log 2>&1 &"
Enter
Sleep 5s

Type "vipera serve status"
Enter
Sleep 500ms

Type "bash docs/assets/demo-request.sh"
Enter
Sleep 4s

Type "vipera serve stop --yes"
Enter
Sleep 500ms
