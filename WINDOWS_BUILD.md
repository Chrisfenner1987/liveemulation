# Building Film Grain (Stochastic) OFX on Windows (NVIDIA / CUDA)

The Windows build uses the **CUDA** GPU path (NVIDIA), via CMake. The grain +
halation math is shared with the macOS/Metal build through `src/GrainCudaCore.cuh`
and has been verified bit-identical to the CPU reference model.

> Covers NVIDIA GPUs (the large majority of DaVinci Resolve Windows systems).
> AMD/Intel GPUs would use an OpenCL path (not yet built) — on those, the plugin
> still loads and runs on the CPU fallback. Ask if you need the OpenCL path.

## Prerequisites
- **Visual Studio 2019/2022** with the "Desktop development with C++" workload.
- **CUDA Toolkit 11.2 or newer** (for `cudaMallocAsync`). 11.8 / 12.x recommended.
  Install *after* Visual Studio so the VS integration registers.
- **CMake 3.20+** (bundled with recent VS, or install separately).
- The repo, including the vendored `openfx/` SDK folder.

## Build
From a "x64 Native Tools Command Prompt for VS" (or any shell with CMake + CUDA on PATH):

```bat
cd FilmGrainOFX
cmake -B build -G "Visual Studio 17 2022" -A x64
cmake --build build --config Release
```

This produces:

```
build\FilmGrain.ofx.bundle\Contents\Win64\FilmGrain.ofx
build\FilmGrain.ofx.bundle\Contents\Info.plist
```

(If CMake can't find CUDA, pass `-DCUDAToolkit_ROOT="C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.4"`.)

## Install (Windows)
OFX plugins load from:

```
C:\Program Files\Common Files\OFX\Plugins\
```

Copy the whole bundle there:

```bat
xcopy /E /I /Y build\FilmGrain.ofx.bundle "C:\Program Files\Common Files\OFX\Plugins\FilmGrain.ofx.bundle"
```

(Needs an Administrator command prompt.) Then restart DaVinci Resolve. It appears
under **OpenFX > Fenner > Film Grain (Stochastic)**.

## Notes
- The same `.ofx.bundle` can hold all platforms at once
  (`Contents/MacOS`, `Contents/Win64`, `Contents/Linux-x86-64`). To ship one
  universal bundle, build on each OS and merge the `Contents/<arch>` folders.
- Linux (NVIDIA) builds from the same CMake: `cmake -B build && cmake --build build`
  → `Contents/Linux-x86-64/FilmGrain.ofx`.
- The CUDA path requires the host to provide a CUDA stream (Resolve does). If a
  host enables neither CUDA nor (on mac) Metal, the multi-threaded CPU path runs.
