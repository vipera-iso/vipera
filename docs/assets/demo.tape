# Regenerate docs/assets/demo.gif with: make assets
# Requires: vhs, ttyd, jq, and the qwen2.5-0.5b model registered locally
# (vipera model search, or vipera model import <path>).
Output docs/assets/demo.gif
Set Shell "bash"
Set FontSize 16
Set Width 900
Set Height 600
Set Theme "Catppuccin Mocha"
Set Padding 16
Set TypingSpeed 35ms
Set PlaybackSpeed 1.5
Set Framerate 15
Set CursorBlink false

Hide
Type "cd ~/vipera"
Enter
Show

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
