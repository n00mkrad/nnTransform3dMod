# nnTransform3D Y/C Separation

This document explains the neural-network Y/C separator used by `nnTransform3D`. It focuses only on the comb-filtering / chroma-separation path: how the composite LaserDisc signal is split into luma and chroma. It does not cover output formats, command-line workflow, or other surrounding processing features.

The current implementation is best understood as a learned 3D frequency-domain comb filter. It does not ask the neural network to draw a new image. Instead, it uses the network to decide which local frequency components of the original composite signal belong to chroma. Those selected components are transformed back into a chroma estimate, then subtracted from the original composite signal to obtain luma.

## Basic

### The Problem

A LaserDisc TBC capture is still a composite video signal. The brightness signal and color signal are mixed together:

```text
Composite = Luma + Chroma
```

The difficulty is that the two signals are not cleanly separated by frequency. Luma detail can live near the chroma subcarrier, and chroma can look like fine brightness detail. A simple filter can remove some color or some dot crawl, but it usually damages something else at the same time.

A traditional comb filter uses the repeating phase pattern of the color subcarrier. It compares neighboring lines, fields, or frames and tries to cancel one part of the signal while keeping the other. That works well when the picture is static and regular, but it can fail on motion, diagonal edges, fine texture, or unusual color/luma combinations.

`nnTransform3D` uses the same general idea, but replaces the hand-written decision logic with a small neural network.

### The Core Idea

The decoder looks at small overlapping chunks of video, not at the whole frame at once. Each chunk contains:

- a small 16 by 16 area of the picture
- field-aware information from two neighboring frames
- enough local context to see line, field, and frame relationships

For each chunk, the program asks a different question than a normal image neural network would:

```text
In this local piece of composite video, which frequency components look like chroma?
```

The network answers by producing a soft mask. A value near 1 means "keep this frequency component as chroma." A value near 0 means "reject it." Values in between allow partial retention.

The original frequency component is still used. The network only decides how much of it should pass through. This is important: the neural network is not inventing chroma from scratch. It is acting like a learned, adaptive filter over the original signal.

### How One Area Is Separated

For each overlapping video chunk, the separator does this:

1. It removes the local average brightness so the filter can focus on the changing part of the composite signal.
2. It applies a smooth window so the chunk can later be blended with neighboring chunks without hard seams.
3. It converts the chunk into frequency space with a 3D FFT. The three dimensions are horizontal position, vertical position, and field/frame time.
4. It measures the strength of each frequency component and a related mirrored component.
5. It sends those measurements to the neural network.
6. The neural network returns a chroma mask.
7. The mask is applied to the original complex FFT data.
8. The filtered chunk is converted back to normal sample space.
9. All overlapping chunks are blended together to produce a chroma estimate for the frame.

After that, luma is simple:

```text
Luma = Composite - Chroma
```

The chroma output is the separated chroma signal stored around a neutral offset, while the luma output is the original composite signal with that chroma estimate removed.

### Why This Is Still a Comb Filter

The separator uses several clues that conventional comb filters also use:

- Chroma has a predictable subcarrier phase pattern.
- Adjacent lines can help distinguish color from brightness detail.
- Fields and nearby frames can help when the picture is not changing too much.
- Static areas and moving areas need different behavior.

The difference is that the decisions are not hard-coded as a fixed 1D, 2D, or 3D rule. The neural network sees the local frequency pattern and produces a per-frequency mask. In easy areas, it can behave like a strong 3D comb. In difficult areas, it can become more conservative.

### Why It Uses Two Frames

The filter processes frame pairs internally. Each network input contains four temporal/field slices:

```text
current frame, even lines
current frame, odd lines
next frame, even lines
next frame, odd lines
```

This gives the model both line-to-line and frame-to-frame evidence. A frame is also processed from both sides during the normal stream: once with the previous neighbor and once with the next neighbor. The accumulated result is blended before the frame is written.

This is why the separator is better described as field-aware frame-pair processing, not single-frame or single-field processing.

### What the Neural Network Learns to Do

At a high level, the network learns what chroma energy tends to look like in the local 3D frequency domain. It can use patterns such as:

- where chroma energy tends to appear around the subcarrier
- how that energy relates to mirrored frequency components
- whether a component is more likely to be color or fine luma detail
- whether local temporal evidence supports a 3D decision

The network output is a confidence map for chroma, not a finished image. The original signal still supplies the actual sample values and phase.

### Practical Consequences

The separator can preserve more luma detail than a blunt low-pass or notch filter because it does not simply remove everything near chroma. It can also reduce cross-color and dot crawl better than a fixed comb rule because its decisions vary by local content.

It is still limited by the information in the composite signal. When luma and chroma are genuinely ambiguous, no separator can always know the original components perfectly. The neural network can make a better guess, but it cannot recover information that the composite encoding made mathematically indistinguishable.

## Advanced

### Signal Model

The implementation treats the input as 16-bit composite samples with fixed NTSC TBC geometry:

```text
FIELD_WIDTH  = 910
FIELD_HEIGHT = 263
FRAME_WIDTH  = 910
FRAME_HEIGHT = 526
```

Two fields are interlaced into a frame buffer by alternating lines:

```text
frame line 0 = field 0 line 0
frame line 1 = field 1 line 0
frame line 2 = field 0 line 1
frame line 3 = field 1 line 1
...
```

The separator estimates chroma `C` directly from composite `S`. Luma is then computed as:

```text
Y = S - C
```

The chroma product is not demodulated into I/Q or U/V inside this separator. It remains a separated composite-domain chroma waveform.

### Temporal Packing

The core routine is `processSplit3D(f0, f1, session, ...)`. It always works on two frame buffers. For each 3D block, the temporal dimension has four slices:

```text
t = 0: f0 even frame lines
t = 1: f0 odd frame lines
t = 2: f1 even frame lines
t = 3: f1 odd frame lines
```

For each `t`, only lines whose parity belongs to that field slice are populated. The opposite parity is zero-filled. Padding frames are also zero-filled.

Streaming uses this pairwise routine as an overlap in time:

- At startup, a padding frame plus the first real frame primes the first frame.
- During the main loop, each pair of adjacent real frames contributes chroma to both frames.
- At the end, the final real frame is paired with padding to finish its accumulation.

Therefore, most frames receive contributions from both the previous-pair pass and the next-pair pass before final normalization.

### Block Grid

Each block is:

```text
Nt = 4
Ny = 16
Nx = 16
block_size = Nt * Ny * Nx = 1024 samples
```

Spatial block origins advance by:

```text
STEP_X = 8
STEP_Y = 8
```

This gives 50 percent overlap in both spatial dimensions. The block grid starts half a block before the filtered region so that edge pixels still get coverage from centered windows. Coordinates for all blocks are stored in `d_ledger_y` and `d_ledger_x`.

### DC Removal and Analysis Window

For each block, `calcDCKernel` computes the mean of all valid samples across the four temporal/field slices. Padding, vertical blanking, and samples outside the filtered picture region are ignored.

`packAndWindowKernel` then packs each block into a complex cuFFT input buffer:

```text
x[t, y, x] = (sample - block_dc) * winT[t] * winY[y] * winX[x]
```

The windows are separable sine windows:

```text
winX[i] = sin(pi * (i + 0.5) / Nx)
winY[i] = sin(pi * (i + 0.5) / Ny)
winT[i] = sin(pi * (i + 0.5) / Nt)
```

The imaginary part is initialized to zero. The same window factors are later used during synthesis overlap-add.

### 3D FFT

All blocks are transformed as a batch using `cufftPlanMany` with dimensions:

```text
{ Nt, Ny, Nx } = { 4, 16, 16 }
```

The transform type is double-precision complex-to-complex:

```text
CUFFT_Z2Z
```

The forward FFT converts each local composite block into a 3D spectrum. At this point, the axes are temporal frequency, vertical frequency, and horizontal frequency bins.

### Neural Network Input Features

The model does not receive complex FFT coefficients directly. `calcMagnitudeKernel` converts each coefficient into two float32 feature channels:

```text
channel 0 = abs(X[k]) / 128
channel 1 = abs(X[ref(k)]) / 128
```

The reference coordinate is:

```text
ref_t = (2 - t) mod 4
ref_y = (16 - y) mod 16
ref_x = (8 - x) mod 16
```

The first channel tells the network how much energy exists at the current bin. The second channel gives a deliberately mirrored subcarrier-related comparison point. This lets the network evaluate a bin together with a frequency counterpart that is meaningful for the encoded chroma pattern, instead of treating each bin as independent scalar energy.

The ONNX Runtime binding uses CUDA memory directly. The input tensor is:

```text
name:  input
type:  float32
shape: [num_blocks, 2, 4, 16, 16]
```

The dynamic first dimension is the number of blocks in the current frame pair.

### Shipped Model Topology

The shipped `chroma_net.onnx` model has this interface:

```text
input:  [batch_size, 2, 4, 16, 16]
output: [batch_size, 1, 4, 16, 16]
```

The graph is a small 3D convolutional residual network:

```text
1x1x1 Conv, 2 -> 32 channels
LeakyReLU

Residual block 0:
    3x3x3 Conv, 32 -> 32
    LeakyReLU
    3x3x3 Conv, 32 -> 32
    Add skip connection
    LeakyReLU

Residual block 1:
    3x3x3 Conv, 32 -> 32
    LeakyReLU
    3x3x3 Conv, 32 -> 32
    Add skip connection
    LeakyReLU

Residual block 2:
    3x3x3 Conv, 32 -> 32
    LeakyReLU
    3x3x3 Conv, 32 -> 32
    Add skip connection
    LeakyReLU

1x1x1 Conv, 32 -> 1 channel
Sigmoid
```

The model contains 8 convolution nodes, 7 LeakyReLU nodes, 3 residual adds, and a final sigmoid. The weights total about 166k float parameters.

Because the final activation is sigmoid, the output is a soft mask:

```text
M[k] in [0, 1]
```

### Spectral Masking

The predicted mask is bound as:

```text
name:  output
type:  float32
shape: [num_blocks, 1, 4, 16, 16]
```

`applyMaskKernel` applies the mask to the original complex spectrum:

```text
Xc[k].real = X[k].real * M[k]
Xc[k].imag = X[k].imag * M[k]
```

The network therefore controls amplitude only. It does not predict phase. Phase is preserved from the original composite block, which makes the operation closer to a learned spectral gate or Wiener-style mask than to direct waveform synthesis.

This design is important for Y/C separation. The desired chroma waveform is already present in the composite input. The model only needs to decide how much of each local 3D frequency bin should be considered chroma.

### Inverse FFT and Synthesis

After masking, a batched inverse `CUFFT_Z2Z` transform converts the chroma spectrum back to sample space. cuFFT's inverse transform is unnormalized, so `olaKernel` divides each reconstructed sample by `block_size`.

The synthesis accumulation for a sample is:

```text
accChroma[p] += reconstructed_block[p] * w[p]
weightSum[p] += w[p] * w[p]
```

where:

```text
w[p] = winT[t] * winY[y] * winX[x]
```

The analysis stage already multiplied the sample by the same window. Accumulating `w * w` allows final normalization to compensate for overlap and tapering:

```text
C[p] = accChroma[p] / weightSum[p]
```

If no valid block contributed to a pixel, the chroma estimate falls back to zero.

The overlap-add uses atomic additions because many CUDA threads can contribute to the same output pixel from different overlapping blocks.

### Final Y/C Equations

Once all relevant frame-pair passes have contributed to a frame, the final separator result is:

```text
C[p] = weightSum[p] > epsilon ? accChroma[p] / weightSum[p] : 0
Y[p] = S[p] - C[p]
```

The stored chroma sample is offset around neutral gray:

```text
C_stored[p] = C[p] + 32768
```

Both outputs are clamped to the 16-bit range.

### Why the Mask Is Predicted in Frequency Space

Composite chroma is structured. In NTSC, chroma lives around the color subcarrier and has predictable phase relationships across horizontal samples, adjacent lines, fields, and frames. Luma detail can occupy overlapping bandwidth, but its local 3D spectral pattern is often different.

The 3D FFT exposes those patterns compactly:

- horizontal frequency separates fine detail and subcarrier-related energy
- vertical frequency captures line comb relationships
- temporal frequency captures field/frame consistency and motion
- mirrored reference magnitudes expose paired spectral evidence

A convolutional network over this small spectral cube can make local decisions using neighboring bins. That is more expressive than independent thresholding and less brittle than a fixed comb heuristic.

### Boundary and Motion Behavior

The first and last frames use padding on the missing temporal side. The filter still runs, but temporal evidence is weaker at the stream boundaries.

Motion is handled implicitly through the local spectral evidence. The model is not performing optical flow or motion compensation. If motion breaks the frame-to-frame chroma relationship, the mask must learn to rely less on temporal evidence and more on line/local-frequency evidence. This is one of the main reasons the output is a soft mask rather than a hard 3D decision.

### Precision and Runtime Path

The separator keeps the FFT data in double-precision complex buffers. The neural network features and mask are float32 tensors, and the ONNX Runtime setup prefers TensorRT with FP16 enabled, with CUDA as a fallback provider.

The data path is designed to avoid unnecessary CPU/GPU transfers during the separator itself:

```text
CVBS frame upload
GPU block packing and windowing
batched cuFFT forward
GPU magnitude feature generation
ONNX Runtime inference via CUDA I/O binding
GPU spectral masking
batched cuFFT inverse
GPU overlap-add accumulation
final chroma and weight buffers copied back for frame finalization
```

This makes the neural network part a per-block spectral mask generator inside a mostly GPU-resident comb-filter pipeline.

### Summary

`nnTransform3D` separates Y/C by estimating the chroma part of the original composite signal. It does that through overlapping 3D FFT blocks, a learned spectral mask, inverse FFT reconstruction, and normalized overlap-add. The neural network does not replace the signal-processing pipeline; it replaces the brittle hand-written decision rule that decides which local 3D frequency components should be chroma.
