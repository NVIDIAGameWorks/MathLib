# MathLib (ML)

*ML* is a cross-platform header-only *SSE/AVX/NEON*-accelerated math library, designed for computer graphics. It serves to goals:
- accelerate performance using *SSE/AVX/NEON* intrinsics
- be HLSL compatible and deliver functionality to both CPU and shader code without code duplication

Features:
- compile-time optimization level specialization: SSE3 (and below), +SSE4, +AVX1, +AVX2 (or NEON on ARM via [*sse2neon*](https://github.com/DLTcollab/sse2neon))
- `int2`, `int3` and `int4` types
- `uint2`, `uint3` and `uint4` types
- `float2`, `float3`, `float4` and `float4x4` types
- `double2`, `double3`, `double4` and `double4x4` types
- `bool2`, `bool3` and `bool4` types
- overloaded operators
- vector swizzling
- common functions: `all`, `any`, `sign`, `abs`, `floor`, `round`, `ceil`, `fmod`, `frac`, `min`, `max`, `clamp`, `saturate`, `lerp`, `step`, `smoothstep` and `linearstep`
- transcendental functions: `sin`, `cos`, `tan`, `asin`, `acos`, `atan`, `atan2`, `sqrt`, `rsqrt`, `rcp`, `pow`, `log`, `log2`, `exp` and `exp2`
- data conversion and packing functionality - FP32, FP16, SNORM and UNORM (with any number of bits per component)
- vectors and matrices
- linear algebra miscellaneous functionality
- projective math miscellaneous functionality
- frustum & AABB primitives
- random numbers generation
- sorting

Important:
- `sizeof(int3/uint3/float3) == sizeof(float4)` on CPU
- `sizeof(double3) == sizeof(double4)` on CPU
- `using namespace std` can lead to name collisions
- inclusion of `cmath` and/or `cstdlib` (even implicitly) after `ml.h` leads to name collisions

Also includes `ml.hlsli` file which is a standalone HLSL math library usable in C++ code.

## License

*ML* is licensed under the MIT License.
