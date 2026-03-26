## This amazing project figures out what would be the best shape for light bulb shade if light intensity distribution is given!

Use: `uv sync` to install depencies. Then:

`source .venn/bin/activate`

followed by:

`uv run python bulb_shade.py --num-rays 300 --max-reflection-angle-deg 20 --light-height 5 --max-mirror-angle 110 --monitor`

where:

- `num-rays` is number of rays to be traced.
- `max-reflection-angle-deg` is how "wide" the cone is e.g. 0 -> only rays in the same direction 20 -> cone with opening angl (per side(?)) of 20 degrees.
- `light-height` how high the source is from the mirror bottom.
- `max-mirror-angle` when starting from the bottom of the mirror how much in deg we have mirror before the traces "fly out" (e.g., 90 -> mirror is a "half sphere")..
- `monitor` see the iteration progress.

Make video (use flag (`--make-movie`) and after building run:

`ffmpeg -framerate 30 -pattern_type glob -i 'figs/*.png' out.mp4`

![Look at this](https://github.com/topiko/plantlightshade/blob/main/bulbshade.png)
