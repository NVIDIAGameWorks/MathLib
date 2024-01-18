# MathLib

*MathLib* is a cross-platform header-only *SSE/AVX/NEON*-accelerated math library, designed for computer graphics.

Features:
- compile-time optimization level specialization: SSE3, SSE4, AVX1, AVX2 (or NEON on ARM via [*sse2neon*](https://github.com/DLTcollab/sse2neon))
- `float3`, `float4` and `float4x4` types
- `double3`, `double4` and `double4x4` types
- `int2`, `uint2`, `int4` and `uint4` types
- common functions: `abs`, `round`, `floor`, `mod`, `clamp`, `saturate`, `min`, `max`, and more...
- transcendental functions: `sin`, `cos`, `tan`, `sqrt`, `rsqrt`, `asin`, `acos`, and more...
- data conversion and packing functionality - FP32, FP16, SNORM and UNORM (with any number of bits per component)
- vectors and matrices
- linear algebra miscellaneous functionality
- projective math miscellaneous functionality
- frustum & AABB primitives
- random numbers generation
- sorting

Also includes `STL.hlsli` file which is a standalone HLSL math library.

## License

MathLib is licensed under the MIT License.
