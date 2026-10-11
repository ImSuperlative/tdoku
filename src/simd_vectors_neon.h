#ifndef TDOKU_SIMD_VECTORS_NEON_H
#define TDOKU_SIMD_VECTORS_NEON_H

// The API of simd_vectors.h implemented with Arm NEON intrinsics, so tdoku can be measured
// natively on arm64 instead of through SIMDe's x86 emulation. Same semantics as the SSE4.2
// path of simd_vectors.h, which is what tdoku uses without AVX2.

#include <arm_neon.h>
#include <cstdint>
#include <cstring>

struct TwoBy64 {
    uint64_t x0;
    uint64_t x1;
};

struct FourBy64 {
    uint64_t x0;
    uint64_t x1;
    uint64_t x2;
    uint64_t x3;
};

struct Bitvec08x16 {
    uint16x8_t vec;

    Bitvec08x16() : vec{vdupq_n_u16(0)} {}

    Bitvec08x16(const uint16x8_t &v) noexcept : vec{v} {}

    Bitvec08x16(const Bitvec08x16 &other) noexcept = default;

    Bitvec08x16(uint16_t x00, uint16_t x01, uint16_t x02, uint16_t x03,
                uint16_t x04, uint16_t x05, uint16_t x06, uint16_t x07) {
        const uint16_t lanes[8] = {x00, x01, x02, x03, x04, x05, x06, x07};
        vec = vld1q_u16(lanes);
    }

    static inline Bitvec08x16 All(uint16_t value) {
        return vdupq_n_u16(value);
    }

    static inline Bitvec08x16
    X_Y_and_Z_or(const Bitvec08x16 &x, const Bitvec08x16 &y, const Bitvec08x16 &z) {
        return (x & y) | z;
    }

    static inline Bitvec08x16
    X_Y_or_Z_or(const Bitvec08x16 &x, const Bitvec08x16 &y, const Bitvec08x16 &z) {
        return x | y | z;
    }

    inline Bitvec08x16 &operator=(const Bitvec08x16 &other) = default;

    inline bool operator==(const Bitvec08x16 &other) const {
        return (*this ^ other).AllZero();
    }

    inline bool operator!=(const Bitvec08x16 &other) const {
        return !(*this == other);
    }

    inline TwoBy64 As_2x64() const {
        TwoBy64 out{};
        vst1q_u16((uint16_t *) &out, vec);
        return out;
    }

    inline Bitvec08x16 WhichEqual(const Bitvec08x16 &other) const {
        return vceqq_u16(vec, other.vec);
    }

    // signed, as _mm_cmpgt_epi16 against zero
    inline Bitvec08x16 WhichNonZero() const {
        return vcgtq_s16(vreinterpretq_s16_u16(vec), vdupq_n_s16(0));
    }

    inline bool AllZero() const {
        return vmaxvq_u16(vec) == 0;
    }

    inline bool AnyZero() const {
        return vminvq_u16(vec) == 0;
    }

    // signed, as _mm_cmpgt_epi16
    inline bool AnyLessThan(const Bitvec08x16 &other) const {
        return vmaxvq_u16(vcltq_s16(vreinterpretq_s16_u16(vec),
                                    vreinterpretq_s16_u16(other.vec))) != 0;
    }

    inline bool Intersects(const Bitvec08x16 &other) const {
        return vmaxvq_u16(vandq_u16(vec, other.vec)) != 0;
    }

    inline bool SubsetOf(const Bitvec08x16 &other) const {
        return vmaxvq_u16(vbicq_u16(vec, other.vec)) == 0;
    }

    inline Bitvec08x16 GetLowBit() const {
        return vandq_u16(vec, vreinterpretq_u16_s16(vnegq_s16(vreinterpretq_s16_u16(vec))));
    }

    // clears the lowest set bit of the vector read as two 64-bit words, low word first
    inline Bitvec08x16 ClearLowBit() const {
        uint64x2_t words = vreinterpretq_u64_u16(vec);
        uint64_t lo = vgetq_lane_u64(words, 0);
        uint64_t hi = vgetq_lane_u64(words, 1);
        if (lo != 0) {
            lo &= lo - 1;
        } else {
            hi &= hi - 1;
        }
        uint64_t out[2] = {lo, hi};
        return vreinterpretq_u16_u64(vld1q_u64(out));
    }

    inline Bitvec08x16 Popcounts9() const {
        return vpaddlq_u8(vcntq_u8(vreinterpretq_u8_u16(vec)));
    }

    inline int Popcount() const {
        return vaddvq_u8(vcntq_u8(vreinterpretq_u8_u16(vec)));
    }

    // as _mm_minpos_epu16 on vec - min_val: (index << 16) | minimum, lowest index on ties
    inline uint32_t MinPosGreaterThanOrEqual(uint16_t min_val) {
        uint16x8_t shifted = vsubq_u16(vec, vdupq_n_u16(min_val));
        uint16_t min = vminvq_u16(shifted);
        uint16x8_t at_min = vceqq_u16(shifted, vdupq_n_u16(min));
        uint64_t nibbles = vget_lane_u64(vreinterpret_u64_u8(vshrn_n_u16(at_min, 4)), 0);
        uint32_t pos = __builtin_ctzll(nibbles) / 8;
        return (pos << 16u) | min;
    }

    // as _mm_shuffle_epi8: out-of-range control bytes (high bit set) give zero
    inline Bitvec08x16 Shuffle(const Bitvec08x16 &control) const {
        return vreinterpretq_u16_u8(
                vqtbl1q_u8(vreinterpretq_u8_u16(vec), vreinterpretq_u8_u16(control.vec)));
    }

    // rotate each 4-lane row left by one lane
    inline Bitvec08x16 RotateRows() const {
        static const uint8_t control[16] = {2, 3, 4, 5, 6, 7, 0, 1, 10, 11, 12, 13, 14, 15, 8, 9};
        return vreinterpretq_u16_u8(vqtbl1q_u8(vreinterpretq_u8_u16(vec), vld1q_u8(control)));
    }

    inline Bitvec08x16 RotateRows2() const {
        return vreinterpretq_u16_u32(vrev64q_u32(vreinterpretq_u32_u16(vec)));
    }

    inline Bitvec08x16 RotateCols() const {
        return vextq_u16(vec, vec, 4);
    }

    inline uint16_t Extract(int index) const {
        uint16_t lanes[8];
        vst1q_u16(lanes, vec);
        return lanes[index];
    }

    void Insert(int index, uint16_t value) {
        uint16_t lanes[8];
        vst1q_u16(lanes, vec);
        lanes[index] = value;
        vec = vld1q_u16(lanes);
    }

    inline Bitvec08x16 operator|(const Bitvec08x16 &other) const {
        return vorrq_u16(vec, other.vec);
    }

    inline void operator|=(const Bitvec08x16 &other) {
        vec = vorrq_u16(vec, other.vec);
    }

    inline Bitvec08x16 operator^(const Bitvec08x16 &other) const {
        return veorq_u16(vec, other.vec);
    }

    inline void operator^=(const Bitvec08x16 &other) {
        vec = veorq_u16(vec, other.vec);
    }

    inline Bitvec08x16 operator&(const Bitvec08x16 &other) const {
        return vandq_u16(vec, other.vec);
    }

    inline void operator&=(const Bitvec08x16 &other) {
        vec = vandq_u16(vec, other.vec);
    }

    inline Bitvec08x16 and_not(const Bitvec08x16 &other) const {
        return vbicq_u16(vec, other.vec);
    }
};

struct Bitvec16x16 {
    Bitvec08x16 lo_;
    Bitvec08x16 hi_;

    Bitvec16x16() noexcept : lo_{}, hi_{} {}

    Bitvec16x16(const Bitvec16x16 &other) noexcept = default;

    Bitvec16x16(const Bitvec08x16 &lo, const Bitvec08x16 &hi) noexcept : lo_(lo), hi_(hi) {}

    Bitvec16x16(uint16_t x00, uint16_t x01, uint16_t x02, uint16_t x03,
                uint16_t x04, uint16_t x05, uint16_t x06, uint16_t x07,
                uint16_t x08, uint16_t x09, uint16_t x10, uint16_t x11,
                uint16_t x12, uint16_t x13, uint16_t x14, uint16_t x15) noexcept :
            lo_{x00, x01, x02, x03, x04, x05, x06, x07},
            hi_{x08, x09, x10, x11, x12, x13, x14, x15} {}

    static inline Bitvec16x16 All(uint16_t value) {
        return Bitvec16x16{Bitvec08x16::All(value), Bitvec08x16::All(value)};
    }

    static inline Bitvec16x16
    X_Y_and_Z_or(const Bitvec16x16 &x, const Bitvec16x16 &y, const Bitvec16x16 &z) {
        return Bitvec16x16{(x.lo_ & y.lo_) | z.lo_, (x.hi_ & y.hi_) | z.hi_};
    }

    static inline Bitvec16x16
    X_Y_andnot_Z_or(const Bitvec16x16 &x, const Bitvec16x16 &y, const Bitvec16x16 &z) {
        return Bitvec16x16{x.lo_.and_not(y.lo_) | z.lo_, x.hi_.and_not(y.hi_) | z.hi_};
    }

    static inline Bitvec16x16
    X_Y_or_Z_or(const Bitvec16x16 &x, const Bitvec16x16 &y, const Bitvec16x16 &z) {
        return Bitvec16x16{x.lo_ | y.lo_ | z.lo_, x.hi_ | y.hi_ | z.hi_};
    }

    static inline Bitvec16x16
    X_Y_xor_Z_or(const Bitvec16x16 &x, const Bitvec16x16 &y, const Bitvec16x16 &z) {
        return Bitvec16x16{(x.lo_ ^ y.lo_) | z.lo_, (x.hi_ ^ y.hi_) | z.hi_};
    }

    inline Bitvec16x16 &operator=(const Bitvec16x16 &other) = default;

    inline bool operator==(const Bitvec16x16 &other) const {
        return lo_ == other.lo_ && hi_ == other.hi_;
    }

    inline bool operator!=(const Bitvec16x16 &other) const {
        return lo_ != other.lo_ || hi_ != other.hi_;
    }

    inline const Bitvec08x16 GetLo() const { return lo_; }

    inline const Bitvec08x16 GetHi() const { return hi_; }

    inline FourBy64 As_4x64() const {
        FourBy64 out{};
        TwoBy64 lo_2x64 = lo_.As_2x64();
        TwoBy64 hi_2x64 = hi_.As_2x64();
        memcpy(&out.x0, &lo_2x64, sizeof(lo_2x64));
        memcpy(&out.x2, &hi_2x64, sizeof(hi_2x64));
        return out;
    }

    inline Bitvec16x16 WhichEqual(const Bitvec16x16 &other) const {
        return Bitvec16x16{lo_.WhichEqual(other.lo_), hi_.WhichEqual(other.hi_)};
    }

    inline Bitvec16x16 WhichNonZero() const {
        return Bitvec16x16{lo_.WhichNonZero(), hi_.WhichNonZero()};
    }

    inline bool AllZero() const {
        return vmaxvq_u16(vorrq_u16(lo_.vec, hi_.vec)) == 0;
    }

    inline bool AnyZero() const {
        return vminvq_u16(vminq_u16(lo_.vec, hi_.vec)) == 0;
    }

    inline bool AnyLessThan(const Bitvec16x16 &other) const {
        uint16x8_t lo = vcltq_s16(vreinterpretq_s16_u16(lo_.vec), vreinterpretq_s16_u16(other.lo_.vec));
        uint16x8_t hi = vcltq_s16(vreinterpretq_s16_u16(hi_.vec), vreinterpretq_s16_u16(other.hi_.vec));
        return vmaxvq_u16(vorrq_u16(lo, hi)) != 0;
    }

    inline bool Intersects(const Bitvec16x16 &other) const {
        return vmaxvq_u16(vorrq_u16(vandq_u16(lo_.vec, other.lo_.vec),
                                    vandq_u16(hi_.vec, other.hi_.vec))) != 0;
    }

    inline bool SubsetOf(const Bitvec16x16 &other) const {
        return vmaxvq_u16(vorrq_u16(vbicq_u16(lo_.vec, other.lo_.vec),
                                    vbicq_u16(hi_.vec, other.hi_.vec))) == 0;
    }

    inline Bitvec16x16 Popcounts9() const {
        return Bitvec16x16{lo_.Popcounts9(), hi_.Popcounts9()};
    }

    inline Bitvec16x16 Shuffle(const Bitvec16x16 &control) const {
        return Bitvec16x16{lo_.Shuffle(control.lo_), hi_.Shuffle(control.hi_)};
    }

    inline Bitvec16x16 RotateRows() const {
        return Bitvec16x16{lo_.RotateRows(), hi_.RotateRows()};
    }

    inline Bitvec16x16 RotateRows2() const {
        return Bitvec16x16{lo_.RotateRows2(), hi_.RotateRows2()};
    }

    // as _mm_alignr_epi8(hi, lo, 8) and _mm_alignr_epi8(lo, hi, 8)
    inline Bitvec16x16 RotateCols() const {
        return Bitvec16x16{vextq_u16(lo_.vec, hi_.vec, 4), vextq_u16(hi_.vec, lo_.vec, 4)};
    }

    inline Bitvec16x16 RotateCols2() const {
        return Bitvec16x16{hi_, lo_};
    }

    inline uint16_t Extract(int index) const {
        return index < 8 ? lo_.Extract(index) : hi_.Extract(index - 8);
    }

    inline void Insert(int index, uint16_t value) {
        if (index < 8) {
            lo_.Insert(index, value);
        } else {
            hi_.Insert(index - 8, value);
        }
    }

    inline Bitvec16x16 operator|(const Bitvec16x16 &other) const {
        return Bitvec16x16{lo_ | other.lo_, hi_ | other.hi_};
    }

    inline void operator|=(const Bitvec16x16 &other) {
        lo_ |= other.lo_;
        hi_ |= other.hi_;
    }

    inline Bitvec16x16 operator^(const Bitvec16x16 &other) const {
        return Bitvec16x16{lo_ ^ other.lo_, hi_ ^ other.hi_};
    }

    inline void operator^=(const Bitvec16x16 &other) {
        lo_ ^= other.lo_;
        hi_ ^= other.hi_;
    }

    inline Bitvec16x16 operator&(const Bitvec16x16 &other) const {
        return Bitvec16x16{lo_ & other.lo_, hi_ & other.hi_};
    }

    inline void operator&=(const Bitvec16x16 &other) {
        lo_ &= other.lo_;
        hi_ &= other.hi_;
    }

    inline Bitvec16x16 and_not(const Bitvec16x16 &other) const {
        return Bitvec16x16{lo_.and_not(other.lo_), hi_.and_not(other.hi_)};
    }
};

// bit i set where x[i] is '.', as _mm_movemask_epi8(_mm_cmpeq_epi8(src, dots))
inline uint32_t WhichDots16(const char *x) {
    static const uint8_t weights[16] = {1, 2, 4, 8, 16, 32, 64, 128, 1, 2, 4, 8, 16, 32, 64, 128};
    uint8x16_t dots = vceqq_u8(vld1q_u8((const uint8_t *) x), vdupq_n_u8('.'));
    uint8x16_t bits = vandq_u8(dots, vld1q_u8(weights));
    return (uint32_t) vaddv_u8(vget_low_u8(bits)) | ((uint32_t) vaddv_u8(vget_high_u8(bits)) << 8u);
}

inline uint32_t WhichDots32(const char *x) {
    return WhichDots16(x) | (WhichDots16(x + 16) << 16u);
}

inline uint64_t WhichDots64(const char *x) {
    return (uint64_t) WhichDots32(x) | ((uint64_t) WhichDots32(x + 32) << 32u);
}

#endif //TDOKU_SIMD_VECTORS_NEON_H
