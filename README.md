## This amazing project figures out what would be the best shape for lamb light shade!

Use: `uv sync` to install depencies. Then:

`uv run python bulb_shade.py --num-rays 300 --max-reflection-angle-deg 20 --light-height 5 --max-mirror-angle 110 --monitor`

Make video:

`ffmpeg -framerate 30 -pattern_type glob -i 'figs/*.png' out.mp4`

![Look at this](https://github.com/topiko/plantlightshade/blob/main/bulbshade.png)
