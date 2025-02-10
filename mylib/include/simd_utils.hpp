#ifndef SIMD_UTILS_HPP
#define SIMD_UTILS_HPP

#include <iostream>

#ifdef _MSC_VER
    #include <intrin.h>  // AVX, AVX512 intrinsics
#else
    #include <x86intrin.h>
#endif

#ifdef __ARM_NEON
    #include <arm_neon.h>  // NEON for macOS (ARM)
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
