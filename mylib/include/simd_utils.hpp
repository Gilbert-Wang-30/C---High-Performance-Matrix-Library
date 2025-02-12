#ifndef SIMD_UTILS_HPP
#define SIMD_UTILS_HPP

#include <iostream>

// Detect OS and Architecture
#if defined(__ARM_NEON) || defined(__aarch64__)  // NEON (macOS ARM)
    #include <arm_neon.h>  // Use NEON intrinsics
#elif defined(_MSC_VER)  // Windows (MSVC)
    #include <intrin.h>  // MSVC intrinsics (AVX, AVX2, AVX-512)
#elif defined(__x86_64__) || defined(__i386__)  // Linux/macOS x86
    #include <immintrin.h>  // AVX, AVX2, AVX-512
    #include <x86intrin.h>
#endif


inline bool hasAVX512() {
#ifdef __AVX512F__
    return true;
#else
    int info[4] = {0};

    #ifdef _MSC_VER
        __cpuidex(info, 7, 0);  
    #endif

    return (info[1] & (1 << 16)) != 0;  // Check AVX512F flag
#endif
}

inline bool hasAVX2() {
#ifdef __AVX2__
    return true;
#else
    int info[4] = {0};

    #ifdef _MSC_VER
        __cpuidex(info, 7, 0);  
    #endif

    return (info[1] & (1 << 5)) != 0;  // Check AVX2 flag
#endif
}


inline bool hasNEON() {
    #ifdef __ARM_NEON
        return true;
    #else
        return false;
    #endif
    }
#endif // SIMD_UTILS_HPP
