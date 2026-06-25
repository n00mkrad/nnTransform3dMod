## nnTransform3dMod

Fork features compared to original implementation:
- Built-in chroma decoder with the option to directly output YUV video to file or stdout (pipe)
- TBC input can be read from stdin for streaming decode pipelines
- TBC or raw video can also be piped out (only one pipe, so either specify Y/C or interleaving mode)
- Automatic metadata loading from `.tbc.json` if available
- Options to specify start/end frame numbers
- Additional QoL stuff and more control via CLI arguments (e.g. specify location of metadata JSON or model file or set output lines/height)

#### Basic Usage:  
`nnTransform3D.exe [--out-mode tbc|raw_y|raw_yc|y4m] [--first-line <num>] [--out <path|->] [input.tbc|-]`

**Basic Example:** `nnTransform3D --out-mode y4m --first-line 42 --out decoded.y4m input.tbc`  
-> Loads `input.tbc` along with the metadata (expected at `input.tbc.json`), with active image starting at line 42 with a default height of 480, outputs `decoded.y4m` which is raw/lossless YUV 4:4:4 16-bit, interlaced 760x480 video (exact width depends on metadata or CLI args).

#### Full Usage:  
`nnTransform3D.exe [--input <path|->] [--model <path>] [--gpu <num>] [--trt_mpi <num>] [--trt_mss <num>] [--start-frame <num>] [--end-frame <num>] [--av-start <num>] [--av-end <num>] [--width <num>] [--out-mode tbc|raw_y|raw_yc|y4m] [--tbc-pipe-mode <y|c|yc_alt|yc_stack>] [--json <path>] [--no-meta] [--no-field-meta] [--full-frame] [--force-limited] [--levels <black:white|ntsc|ntscj>] [--chroma-gain <number>] [--first-line <num>] [--last-line <num>] [--lines <num>] [-q] [--out <path|->] [input.tbc|-]`

**Options:**  
`--av-start`: Active video area start (in pixels, horizontal).  
`--av-end`: Active video area end (in pixels, horizontal).  
`--width`: Active video width. Used to derive `av-end` from `av-start` when `--av-end` is omitted.  
`--start-frame`: Source frame index to start decoding from (0-based, default `0`). When `> 0`, the decoder internally pre-rolls one frame for temporal context and drops that pre-roll output.  
`--end-frame`: Inclusive source frame index to stop output at (0-based, default `0` for unlimited). When active, one extra frame may still be decoded for temporal lookahead.
`--out-mode`: Output mode, either `tbc`, `raw_y`, `raw_yc`, or `y4m`. Default: `tbc`.  
`--tbc-pipe-mode`: TBC stdout layout for `--out-mode tbc`: `y`, `c`, `yc_alt`, or `yc_stack` (Luma, Chroma, Luma/Chroma alternating at 2x frame rate, Luma/Chroma stacked top-bottom). Requires `--out -`.  
`--out`: Output path, or `-` for binary stdout. In TBC mode, `--out -` is only valid when `--tbc-pipe-mode` is set.  
`--json`: Metadata JSON path, defaults to `<input>.json` for file input. Required for Y4M output when input is stdin unless `--no-meta` is used.
`--no-meta`: Ignore any existing metadata files and construct built-in NTSC metadata and field entries on runtime. Cannot be combined with `--json`; combining it with `--no-field-meta` is allowed but redundant.
`--no-field-meta`: Ignore `fields` metadata and generate generic metadata on runtime. Allows metadata JSON without a `fields` array; all other metadata is still loaded from the file.
`--full-frame`: For `raw_y`, `raw_yc`, and `y4m`, output full frame geometry including blanking regions.  
`--force-limited`: For `y4m`, clamp all output samples to legal limited range (`Y` 16..235, `Cb/Cr` 16..240 in 8-bit-equivalent terms).
`--levels`: For `y4m`, override metadata black/white scaling levels. Use `<black>:<white>` raw 16-bit `black16bIre` / `white16bIre` values, `ntsc` (`18048:51200`), or `ntscj` (`15360:51200`).
`--chroma-gain`: For `y4m`, gain factor applied to chroma components. CLI overrides metadata `videoParameters.chromaGain`; default `1.0` when neither is provided. Use `0` for neutral chroma.
`--first-line`: First output line for active-area output (default `40`).  
`--last-line`: Last output line for active-area output (exclusive).  
`--lines`: Active output height in lines. Used to derive `last-line` from `first-line` when `--last-line` is omitted. Default: `480`.  
`-q`: Disable the progress message (`[Info] Processed n frames...`).  

#### Advanced:  
`--input`: Input TBC file, or `-` for binary stdin. Explicit/keyword form of positional `[input.tbc|-]`.
`--model`: ONNX model path. Default: `chroma_net.onnx` in the executable directory or working directory.  
`--gpu`: GPU index used for TensorRT and CUDA providers. Default: `0`.  
`--trt_mpi`: TensorRT max partition iterations (`trt_max_partition_iterations`). Default: `1000`.  
`--trt_mss`: TensorRT minimum subgraph size (`trt_min_subgraph_size`). Default: `1`.  


### Piped TBC Input

Use `--input -` or positional `-` to read headerless field-sequential TBC data from stdin. The stream must contain little-endian `uint16` samples with fixed NTSC geometry: two `910x263` fields per source frame, or exactly `957320` bytes per frame.

- Input is consumed sequentially with one frame of temporal lookahead.
- `--start-frame` discards preceding input frames and retains frame `start-frame - 1` as temporal pre-roll.
- `--end-frame` is inclusive. The decoder reads one additional complete frame for lookahead when available and then stops without draining stdin.
- Exact frame-boundary EOF is accepted. Empty input, an incomplete trailing frame, or EOF before a requested start/end frame returns an error.
- Metadata is not auto-detected for stdin. Y4M output requires `--json <path>` or `--no-meta`.
- When file output is selected without `--out`, stdin-derived names are `stdin_Y.tbc`, `stdin_C.tbc`, `stdin_Y.raw`, `stdin_YC.raw`, or `stdin.y4m`.

Examples:

```bat
producer.exe | nnTransform3D.exe --input - --out-mode raw_y --full-frame --out decoded.raw
producer.exe | nnTransform3D.exe - --out-mode y4m --json input.tbc.json --out decoded.y4m
producer.exe | nnTransform3D.exe --input - --out-mode raw_y --full-frame --out - | consumer.exe
```

Windows PowerShell 5.1 pipelines are not binary-safe and can alter raw stream bytes. Run binary pipelines through `cmd.exe` or connect native processes directly instead.


### Metadata and Active Video Area

For file input, metadata is attempted in all output modes from `--json` or auto-detected `<input>.json`. For stdin, metadata is only loaded from an explicit `--json` path.
If metadata loads and `--av-start` / `--av-end` are not set, `activeVideoStart` / `activeVideoEnd` from JSON are used.  
For `tbc`, `raw_y`, and `raw_yc`, missing/invalid metadata falls back to defaults (`132..896`) unless AV bounds are explicitly set.  
For `y4m`, valid JSON metadata is required unless `--no-meta` is used.
With `--no-field-meta`, the JSON `fields` array is ignored. File input pre-generates entries for every frame before decoding; stdin generates entries incrementally. **Generated metadata is only used in-memory and is never written to disk.** This applies to `--no-meta` as well.
With `--no-meta`, no sidecar is opened. Built-in NTSC metadata uses active area `147..905`, black/white levels `18048..51200`, burst area `92..128`, `910x263` field geometry, sample rate `14318181.81818182`, system `NTSC`, and 4:3 output. CLI active-area, level, and chroma-gain options retain precedence. This mode includes the behavior of `--no-field-meta`, passing both flags is not needed.

Range derivation precedence:
- `--av-end` overrides `--width`.
- `--last-line` overrides `--lines`.
- If `--last-line` is omitted, resolved vertical end is `last-line = first-line + lines` (default lines `480`).

### Y4M Output Mode

- `--out-mode y4m` writes YUV4MPEG2 `YUV444P16` limited-range frames. By default, nominal levels are limited-range while headroom/footroom samples are preserved.
- `--force-limited` clips all Y4M pixels to limited range (`Y` 16..235, `Cb/Cr` 16..240 in 8-bit-equivalent terms).
- `--levels` overrides the Y4M luma/chroma scaling reference levels from metadata. The custom form uses raw 16-bit `black16bIre` / `white16bIre` values, not 8-bit video code values.
- `--chroma-gain` applies chroma saturation gain after Y/C separation and before Y4M chroma mapping. Overrides `videoParameters.chromaGain` metadata; default is `1.0` (neutral).
- Video is merged from separated luma and chroma using minimal `mono`/`ntsc1d` decoders. More advanced comb filters are not needed as Y/C separation is done by the neural network.
- `--start-frame` keeps Y4M phase continuity aligned to absolute source-frame position.
- Default is active-area output, using horizontal metadata bounds (or CLI overrides) and `--first-line` / `--last-line`.
- `--full-frame` outputs full metadata geometry (`fieldWidth x ((fieldHeight * 2) - 1)`).
- `--out -` is supported for piping Y4M to stdout.

Examples:

```bash
# Default Y4M file output (auto metadata from input.tbc.json)
nnTransform3D --input input.tbc --out-mode y4m

# Y4M output using built-in NTSC metadata without a sidecar
nnTransform3D --input input.tbc --out-mode y4m --no-meta

# Explicit metadata file and active-area output to stdout using first-line + lines
nnTransform3D --input input.tbc --out-mode y4m --json tbc-example.json --first-line 40 --lines 480 --out - > output.y4m

# Explicit metadata file using av-start + width shorthand (av-end derived)
nnTransform3D --input input.tbc --out-mode y4m --json tbc-example.json --av-start 147 --width 758 --out output_active.y4m

# Full-frame Y4M output
nnTransform3D --input input.tbc --out-mode y4m --json tbc-example.json --full-frame --out output_full.y4m
```

### Default TBC File Output

Default output mode is `tbc`, which writes two files into the source directory: `input_Y.tbc` (luma) and `input_C.tbc` (chroma).
For stdin input, the corresponding default names are `stdin_Y.tbc` and `stdin_C.tbc`.

### TBC Stdout Pipe Modes

All TBC pipe modes emit headerless `uint16` little-endian samples and preserve current field-sequential TBC frame chunk packing.

- `--tbc-pipe-mode y`: Emit only luma TBC frame chunks.
- `--tbc-pipe-mode c`: Emit only chroma TBC frame chunks.
- `--tbc-pipe-mode yc_alt`: Emit luma chunk then chroma chunk for each source frame. Interpret as `910x526` with doubled frame cadence. (60000/1001 FPS with alternating Y/C)
- `--tbc-pipe-mode yc_stack`: Emit the same byte order as `yc_alt`, but interpret as vertically stacked `910x1052` frames (Y top, C bottom). This is intentionally "out-of-spec" TBC geometry.

**Both `yc_alt` and `yc_stack` are "out-of-spec" methods of storing Y+C in a single TBC while keeping them separated - Any downstream software needs to explicitly handle this.** Alternatively, both methods allow reconstruction of either Y or C via ffmpeg (select even/odd frames for `yc_alt` or crop top/bottom for `yc_stack`).

Examples:

```bash
# Y-only TBC stream to file
nnTransform3D --input input.tbc --out-mode tbc --tbc-pipe-mode y --out - > input_Y.tbc

# C-only TBC stream to file
nnTransform3D --input input.tbc --out-mode tbc --tbc-pipe-mode c --out - > input_C.tbc

# YC alternating mode interpreted as 910x526 at 2x frame cadence
nnTransform3D --input input.tbc --out-mode tbc --tbc-pipe-mode yc_alt --out - | ffmpeg -f rawvideo -pixel_format gray16le -video_size 910x526 -framerate 60000/1001 -i - -c:v ffv1 input_YC_alt.mkv

# YC stacked interpretation (910x1052), split back into Y and C
nnTransform3D --input input.tbc --out-mode tbc --tbc-pipe-mode yc_stack --out - | ffmpeg -f rawvideo -pixel_format gray16le -video_size 910x1052 -framerate 30000/1001 -i - -filter_complex "[0:v]split=2[yall][call];[yall]crop=910:526:0:0[y];[call]crop=910:526:0:526[c]" -map "[y]" -c:v ffv1 input_Y_from_stack.mkv -map "[c]" -c:v ffv1 input_C_from_stack.mkv
```

### Raw Video Output Mode

- `--out-mode raw_y` writes one luma stream (`uint16` little-endian).
- `--out-mode raw_yc` writes luma then chroma planes (`uint16` LE) for each frame.
- Default raw output is active-area cropped:
  - Horizontal: `[activeVideoStart, activeVideoEnd)` from metadata/defaults/AV overrides.
  - Vertical: `[first-line, last-line)` where `last-line` defaults to `first-line + lines` (`40 + 480 = 520`) unless explicitly set.
- `--av-end` overrides `--width`.
- `--last-line` overrides `--lines`.
- `--full-frame` keeps raw geometry at `910x526` (`raw_y`) or `910x1052` stacked (`raw_yc`).
- Raw default names are `input_Y.raw` (`raw_y`) and `input_YC.raw` (`raw_yc`) unless `--out` is provided.
- `--out -` can be used in raw mode to pipe the data directly into another process (e.g. ffmpeg) without writing to disk.
- `--start-frame` and `--end-frame` are applied before all raw output modes using 0-based source-frame indexing.

FFmpeg and mpv decode examples (replace `30000/1001` with your actual frame rate if needed):

```bash
# Decode full-frame raw luma output (input_Y.raw) to a lossless FFV1 MKV
nnTransform3D --input input.tbc --out-mode raw_y --full-frame
ffmpeg -f rawvideo -pixel_format gray16le -video_size 910x526 -framerate 30000/1001 -i input_Y.raw -c:v ffv1 input_Y.mkv

# Pipe to ffmpeg and decode full-frame YC raw output while splitting into separate Y and C videos
nnTransform3D --input input.tbc --out-mode raw_yc --full-frame --out - | ffmpeg -f rawvideo -pixel_format gray16le -video_size 910x1052 -framerate 30000/1001 -i - -filter_complex "[0:v]split=2[yall][call];[yall]crop=910:526:0:0[y];[call]crop=910:526:0:526[c]" -map "[y]" -c:v ffv1 input_Y_from_YC.mkv -map "[c]" -c:v ffv1 input_C_from_YC.mkv

# Pipe to mpv to preview full-frame luma output in real time
nnTransform3D --input input.tbc --out-mode raw_y --full-frame --out - | mpv --demuxer=rawvideo --demuxer-rawvideo-mp-format=gray16le --demuxer-rawvideo-w=910 --demuxer-rawvideo-h=526 --demuxer-rawvideo-fps=30000/1001 -
```
