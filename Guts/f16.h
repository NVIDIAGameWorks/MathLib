// © 2021 NVIDIA Corporation

#pragma once

#define F16_M_BITS 10
#define F16_E_BITS 5
#define F16_S_MASK 0x8000

template <uint32_t M_BITS, uint32_t E_BITS, uint32_t S_MASK>
ML_INLINE uint32_t ToSmallFloat(float x) {
    const int32_t E_MASK = (1 << E_BITS) - 1;
    const uint32_t INF = uint32_t(E_MASK) << uint32_t(M_BITS);
    const int32_t BIAS = E_MASK >> 1;
    const int32_t ROUND = 1 << (23 - M_BITS - 1);

    // decompose float
    uint32_t f32 = *(uint32_t*)&x;
    uint32_t packed = (f32 >> 16) & S_MASK;
    int32_t e = ((f32 >> 23) & 0xFF) - 127 + BIAS;
    int32_t m = f32 & 0x007FFFFF;

    if (e == 128 + BIAS) {
        // Inf
        packed |= INF;

        if (m) {
            // NaN
            m >>= 23 - M_BITS;
            packed |= m | (m == 0);
        }
    } else if (e > 0) {
        // round to nearest, round "0.5" up
        if (m & ROUND) {
            m += ROUND << 1;

            if (m & 0x00800000) {
                // mantissa overflow
                m = 0;
                e++;
            }
        }

        if (e >= E_MASK) {
            // exponent overflow - flush to Inf
            packed |= INF;
        } else {
            // representable value
            m >>= 23 - M_BITS;
            packed |= (e << M_BITS) | m;
        }
    } else {
        // denormalized or zero
        m = ((m | 0x00800000) >> (1 - e)) + ROUND;
        m >>= 23 - M_BITS;
        packed |= m;
    }

    return packed;
}

template <int32_t M_BITS, int32_t E_BITS, int32_t S_MASK>
ML_INLINE float FromSmallFloat(uint32_t x) {
    const uint32_t E_MASK = (1 << E_BITS) - 1;
    const int32_t BIAS = E_MASK >> 1;
    const float DENORM_SCALE = 1.0f / (1 << (14 + M_BITS));
    const float NORM_SCALE = 1.0f / float(1 << M_BITS);

    int32_t s = (x & S_MASK) << 15;
    int32_t e = (x >> M_BITS) & E_MASK;
    int32_t m = x & ((1 << M_BITS) - 1);

    uFloat f;
    if (e == 0)
        f.f = DENORM_SCALE * m;
    else if (e == E_MASK)
        f.i = s | 0x7F800000 | (m << (23 - M_BITS));
    else {
        f.f = 1.0f + float(m) * NORM_SCALE;

        if (e < BIAS)
            f.f /= float(1 << (BIAS - e));
        else
            f.f *= float(1 << (e - BIAS));
    }

    if (s)
        f.f = -f.f;

    return f.f;
}

ML_INLINE uint32_t f32tof16(float x) {
#if (ML_INTRINSIC_LEVEL >= ML_INTRINSIC_AVX1)
    v4f v = v4f_set(x, 0.0f, 0.0f, 0.0f);
    v4i p = v4f_to_h4(v);

    uint32_t r = _mm_cvtsi128_si32(p);
#else
    uint32_t r = ToSmallFloat<F16_M_BITS, F16_E_BITS, F16_S_MASK>(x);
#endif

    return r;
}

ML_INLINE float f16tof32(uint32_t x) {
#if (ML_INTRINSIC_LEVEL >= ML_INTRINSIC_AVX1)
    v4i p = _mm_cvtsi32_si128(x);
    v4f f = _mm_cvtph_ps(p);

    return _mm_cvtss_f32(f);
#else
    return FromSmallFloat<F16_M_BITS, F16_E_BITS, F16_S_MASK>(x);
#endif
}

struct float16_t {
    uint16_t us;

    ML_INLINE float16_t() = default;

    ML_INLINE float16_t(uint16_t x) {
        us = x;
    }

    ML_INLINE float16_t(float x) {
        us = (uint16_t)f32tof16(x);
    }

    ML_INLINE operator float() const {
        return f16tof32(us);
    }
};
