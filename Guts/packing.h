// © 2021 NVIDIA Corporation

#pragma once

#define UF11_M_BITS 6
#define UF11_E_BITS 5
#define UF11_S_MASK 0x0

#define UF10_M_BITS 5
#define UF10_E_BITS 5
#define UF10_S_MASK 0x0

namespace Packing {

template <uint32_t Rbits, uint32_t Gbits, uint32_t Bbits, uint32_t Abits>
ML_INLINE uint32_t float4_to_unorm(const float4& v) {
    ML_StaticAssertMsg(Rbits + Gbits + Bbits + Abits <= 32, "Sum of all bit must be <= 32");

    const uint32_t Rmask = (1 << Rbits) - 1;
    const uint32_t Gmask = (1 << Gbits) - 1;
    const uint32_t Bmask = (1 << Bbits) - 1;
    const uint32_t Amask = (1 << Abits) - 1;

    const uint32_t Gshift = Rbits;
    const uint32_t Bshift = Gshift + Gbits;
    const uint32_t Ashift = Bshift + Bbits;

    const v4f scale = v4f_set(float(Rmask), float(Gmask), float(Bmask), float(Amask));

    v4f t = _mm_mul_ps(v.xmm, scale);
    v4i i = _mm_cvtps_epi32(t);

    uint32_t p = _mm_cvtsi128_si32(i);
    p |= _mm_cvtsi128_si32(_mm_shuffle_epi32(i, _MM_SHUFFLE(1, 1, 1, 1))) << Gshift;
    p |= _mm_cvtsi128_si32(_mm_shuffle_epi32(i, _MM_SHUFFLE(2, 2, 2, 2))) << Bshift;
    p |= _mm_cvtsi128_si32(_mm_shuffle_epi32(i, _MM_SHUFFLE(3, 3, 3, 3))) << Ashift;

    return p;
}

template <>
ML_INLINE uint32_t float4_to_unorm<8, 8, 8, 8>(const float4& v) {
    v4f t = _mm_mul_ps(v.xmm, _mm_set1_ps(255.0f));
    v4i i = _mm_cvtps_epi32(t);
    i = _mm_shuffle_epi8(i, _mm_set1_epi32(0x0C080400));

    return _mm_cvtsi128_si32(i);
}

ML_INLINE uint32_t float2_to_unorm_16_16(const float2& v) {
    v4f t = v4f_set(v.x, v.y, 0.0f, 0.0f);
    t = _mm_mul_ps(t, _mm_set1_ps(65535.0f));
    v4i i = _mm_cvtps_epi32(t);

    uint32_t p = _mm_cvtsi128_si32(i);
    p |= _mm_cvtsi128_si32(_mm_shuffle_epi32(i, _MM_SHUFFLE(1, 1, 1, 1))) << 16;

    return p;
}

ML_INLINE uint32_t float3_to_ufloat_11_11_10(const float3& v) {
    ML_Assert(v.x >= 0.0f && v.y >= 0.0f && v.z >= 0.0f);

    uint32_t r = ToSmallFloat<UF11_M_BITS, UF11_E_BITS, UF11_S_MASK>(v.x);
    r |= ToSmallFloat<UF11_M_BITS, UF11_E_BITS, UF11_S_MASK>(v.y) << 11;
    r |= ToSmallFloat<UF10_M_BITS, UF10_E_BITS, UF10_S_MASK>(v.z) << 22;

    return r;
}

template <uint32_t Rbits, uint32_t Gbits, uint32_t Bbits, uint32_t Abits>
ML_INLINE uint32_t float4_to_snorm(const float4& v) {
    ML_StaticAssertMsg(Rbits + Gbits + Bbits + Abits <= 32, "Sum of all bit must be <= 32");

    const uint32_t Rmask = (1 << Rbits) - 1;
    const uint32_t Gmask = (1 << Gbits) - 1;
    const uint32_t Bmask = (1 << Bbits) - 1;
    const uint32_t Amask = (1 << Abits) - 1;

    const uint32_t Gshift = Rbits;
    const uint32_t Bshift = Gshift + Gbits;
    const uint32_t Ashift = Bshift + Bbits;

    const uint32_t Rrange = (1 << (Rbits - 1)) - 1;
    const uint32_t Grange = (1 << (Gbits - 1)) - 1;
    const uint32_t Brange = (1 << (Bbits - 1)) - 1;
    const uint32_t Arange = (1 << (Abits - 1)) - 1;

    const v4f scale = v4f_set(float(Rrange), float(Grange), float(Brange), float(Arange));
    const v4i mask = _mm_setr_epi32(Rmask, Gmask, Bmask, Amask);

    v4f t = _mm_mul_ps(v.xmm, scale);
    v4i i = _mm_cvtps_epi32(t);
    i = _mm_and_si128(i, mask);

    uint32_t p = _mm_cvtsi128_si32(i);
    p |= _mm_cvtsi128_si32(_mm_shuffle_epi32(i, _MM_SHUFFLE(1, 1, 1, 1))) << Gshift;
    p |= _mm_cvtsi128_si32(_mm_shuffle_epi32(i, _MM_SHUFFLE(2, 2, 2, 2))) << Bshift;
    p |= _mm_cvtsi128_si32(_mm_shuffle_epi32(i, _MM_SHUFFLE(3, 3, 3, 3))) << Ashift;

    return p;
}

template <>
ML_INLINE uint32_t float4_to_snorm<8, 8, 8, 8>(const float4& v) {
    v4f t = _mm_mul_ps(v.xmm, _mm_set1_ps(127.0f));
    v4i i = _mm_cvtps_epi32(t);
    i = _mm_shuffle_epi8(i, _mm_set1_epi32(0x0C080400));

    return _mm_cvtsi128_si32(i);
}

ML_INLINE uint32_t float2_to_snorm_16_16(const float2& v) {
    v4f t = v4f_set(v.x, v.y, 0.0f, 0.0f);
    t = _mm_mul_ps(t, _mm_set1_ps(32767.0f));
    v4i i = _mm_cvtps_epi32(t);
    i = _mm_and_si128(i, _mm_setr_epi32(65535, 65535, 0, 0));

    uint32_t p = _mm_cvtsi128_si32(i);
    p |= _mm_cvtsi128_si32(_mm_shuffle_epi32(i, _MM_SHUFFLE(1, 1, 1, 1))) << 16;

    return p;
}

ML_INLINE uint32_t float2_to_sfloat_16_16(const float2& v) {
#if (ML_INTRINSIC_LEVEL >= ML_INTRINSIC_AVX1)
    v4f t = v4f_set(v.x, v.y, 0.0f, 0.0f);
    v4i p = v4f_to_h4(t);

    uint32_t r = _mm_cvtsi128_si32(p);
#else
    uint32_t r = ToSmallFloat<F16_M_BITS, F16_E_BITS, F16_S_MASK>(v.x);
    r |= ToSmallFloat<F16_M_BITS, F16_E_BITS, F16_S_MASK>(v.y) << 16;
#endif

    return r;
}

ML_INLINE uint2 float4_to_sfloat_16_16_x2(const float4& v) {
    uint2 r;

#if (ML_INTRINSIC_LEVEL >= ML_INTRINSIC_AVX1)
    v4i p = v4f_to_h4(v.xmm);
    *((uint64_t*)&r) = _mm_extract_epi64(p, 0);
#else
    r.x = float2_to_sfloat_16_16(v.xy);
    r.y = float2_to_sfloat_16_16(v.zw);
#endif

    return r;
}

template <uint32_t Rbits, uint32_t Gbits, uint32_t Bbits, uint32_t Abits>
ML_INLINE float4 unorm_to_float4(uint32_t p) {
    ML_StaticAssertMsg(Rbits + Gbits + Bbits + Abits <= 32, "Sum of all bit must be <= 32");

    const uint32_t Rmask = (1 << Rbits) - 1;
    const uint32_t Gmask = (1 << Gbits) - 1;
    const uint32_t Bmask = (1 << Bbits) - 1;
    const uint32_t Amask = (1 << Abits) - 1;

    const uint32_t Gshift = Rbits;
    const uint32_t Bshift = Gshift + Gbits;
    const uint32_t Ashift = Bshift + Bbits;

    const v4f scale = v4f_set(1.0f / Rmask, 1.0f / Gmask, 1.0f / Bmask, 1.0f / Amask);

    v4i i = _mm_setr_epi32(p & Rmask, (p >> Gshift) & Gmask, (p >> Bshift) & Bmask, (p >> Ashift) & Amask);
    v4f t = _mm_cvtepi32_ps(i);
    t = _mm_mul_ps(t, scale);

    return t;
}

template <>
ML_INLINE float4 unorm_to_float4<8, 8, 8, 8>(uint32_t p) {
#if (ML_INTRINSIC_LEVEL >= ML_INTRINSIC_SSE4)
    v4i i = _mm_cvtepu8_epi32(_mm_cvtsi32_si128(p));
#else
    v4i i = _mm_set_epi32(p >> 24, (p >> 16) & 0xFF, (p >> 8) & 0xFF, p & 0xFF);
#endif

    v4f t = _mm_cvtepi32_ps(i);
    t = _mm_mul_ps(t, _mm_set1_ps(1.0f / 255.0f));

    return t;
}

ML_INLINE float3 ufloat_11_11_10_to_float3(uint32_t p) {
    float3 v;
    v.x = FromSmallFloat<UF11_M_BITS, UF11_E_BITS, UF11_S_MASK>(p & ((1 << 11) - 1));
    v.y = FromSmallFloat<UF11_M_BITS, UF11_E_BITS, UF11_S_MASK>((p >> 11) & ((1 << 11) - 1));
    v.z = FromSmallFloat<UF10_M_BITS, UF10_E_BITS, UF10_S_MASK>((p >> 22) & ((1 << 10) - 1));

    return v;
}

template <uint32_t Rbits, uint32_t Gbits, uint32_t Bbits, uint32_t Abits>
ML_INLINE float4 snorm_to_float4(uint32_t p) {
    ML_StaticAssertMsg(Rbits + Gbits + Bbits + Abits <= 32, "Sum of all bit must be <= 32");

    const uint32_t Rmask = (1 << Rbits) - 1;
    const uint32_t Gmask = (1 << Gbits) - 1;
    const uint32_t Bmask = (1 << Bbits) - 1;
    const uint32_t Amask = (1 << Abits) - 1;

    const uint32_t Gshift = Rbits;
    const uint32_t Bshift = Gshift + Gbits;
    const uint32_t Ashift = Bshift + Bbits;

    const uint32_t Rsign = (1 << (Rbits - 1));
    const uint32_t Gsign = (1 << (Gbits - 1));
    const uint32_t Bsign = (1 << (Bbits - 1));
    const uint32_t Asign = (1 << (Abits - 1));

    const v4i vsign = _mm_setr_epi32(Rsign, Gsign, Bsign, Asign);
    const v4i vor = _mm_setr_epi32(~(Rsign - 1), ~(Gsign - 1), ~(Bsign - 1), ~(Asign - 1));
    const v4f vscale = v4f_set(1.0f / (Rsign - 1), 1.0f / (Gsign - 1), 1.0f / (Bsign - 1), 1.0f / (Asign - 1));

    v4i i = _mm_setr_epi32(p & Rmask, (p >> Gshift) & Gmask, (p >> Bshift) & Bmask, (p >> Ashift) & Amask);

    v4i mask = _mm_and_si128(i, vsign);
    v4i ii = _mm_or_si128(i, vor);
    i = v4i_select(i, ii, _mm_cmpeq_epi32(mask, _mm_setzero_si128()));

    v4f t = _mm_cvtepi32_ps(i);
    t = _mm_mul_ps(t, vscale);
    t = _mm_max_ps(t, _mm_set1_ps(-1.0f));

    return t;
}

template <>
ML_INLINE float4 snorm_to_float4<8, 8, 8, 8>(uint32_t p) {
#if (ML_INTRINSIC_LEVEL >= ML_INTRINSIC_SSE4)
    v4i i = _mm_cvtepi8_epi32(_mm_cvtsi32_si128(p));
#else
    v4i i = _mm_set_epi32(int8_t(p >> 24), int8_t((p >> 16) & 0xFF), int8_t((p >> 8) & 0xFF), int8_t(p & 0xFF));
#endif

    v4f t = _mm_cvtepi32_ps(i);
    t = _mm_mul_ps(t, _mm_set1_ps(1.0f / 127.0f));
    t = _mm_max_ps(t, _mm_set1_ps(-1.0f));

    return t;
}

ML_INLINE float2 sfloat_16_16_to_float2(uint32_t p) {
    float2 r;

#if (ML_INTRINSIC_LEVEL >= ML_INTRINSIC_AVX1)
    v4i t = _mm_cvtsi32_si128(p);
    v4f f = _mm_cvtph_ps(t);

    _mm_storel_pi(&r.mm, f);
#else
    r.x = FromSmallFloat<F16_M_BITS, F16_E_BITS, F16_S_MASK>(p & 0xFFFF);
    r.y = FromSmallFloat<F16_M_BITS, F16_E_BITS, F16_S_MASK>(p >> 16);
#endif

    return r;
}

ML_INLINE float4 sfloat_16_16_x2_to_float4(const uint2& p) {
    float4 f;

#if (ML_INTRINSIC_LEVEL >= ML_INTRINSIC_AVX1)
    v4i t = _mm_loadu_si64(&p);
    f.xmm = _mm_cvtph_ps(t);
#else
    f.x = FromSmallFloat<F16_M_BITS, F16_E_BITS, F16_S_MASK>(p.x & 0xFFFF);
    f.y = FromSmallFloat<F16_M_BITS, F16_E_BITS, F16_S_MASK>(p.x >> 16);
    f.z = FromSmallFloat<F16_M_BITS, F16_E_BITS, F16_S_MASK>(p.y & 0xFFFF);
    f.w = FromSmallFloat<F16_M_BITS, F16_E_BITS, F16_S_MASK>(p.y >> 16);
#endif

    return f;
}

} // namespace Packing
