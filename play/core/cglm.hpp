#pragma once
/***
 * Amalgamated Version of CGLM from https://github.com/recp/cglm
 * Edits made:
 *  - Namespaced into local library for convenience.
 *  - Some bugs with include order due to amalgamation
 *  - TODO: Operator overloads for common structs
 */
/*** Start of inlined file: struct.h ***/
#include <stdint.h>
#include <stddef.h>
#include <stdlib.h>
#include <math.h>
#include <float.h>
#include <stdbool.h>



#ifndef NDEBUG
#include <assert.h>
#endif


#if (defined(__STDC_VERSION__) && __STDC_VERSION__ >= 201112L)
#include <stdalign.h>
#endif

#if defined(_MSC_VER) && !defined(_M_ARM64EC)
#  if (defined(_M_AMD64) || defined(_M_X64)) || _M_IX86_FP == 2
#    ifndef __SSE__
#      define __SSE__
#    endif
#    ifndef __SSE2__
#      define __SSE2__
#    endif
#  elif _M_IX86_FP == 1
#    ifndef __SSE__
#      define __SSE__
#    endif
#  endif
/* do not use alignment for older visual studio versions */
/* also ARM32 also causes similar error, disable it for now on ARM32 too */
#  if _MSC_VER < 1913 || _M_ARM     /* Visual Studio 2017 version 15.6 */
#    define PLAY_CGLM_ALL_UNALIGNED
#  endif
#endif

#ifdef __AVX__
#include <immintrin.h>
#  define PLAY_CGLM_AVX_FP 1
#    ifndef __SSE2__
#      define __SSE2__
#    endif
#    ifndef __SSE3__
#      define __SSE3__
#    endif
#    ifndef __SSE4__
#      define __SSE4__
#    endif
#    ifndef __SSE4_1__
#      define __SSE4_1__
#    endif
#    ifndef __SSE4_2__
#      define __SSE4_2__
#    endif
#  ifndef PLAY_CGLM_SIMD_x86
#    define PLAY_CGLM_SIMD_x86
#  endif
#endif

#if defined(__SSE__)
#include <xmmintrin.h>
#  define PLAY_CGLM_SSE_FP 1
#  ifndef PLAY_CGLM_SIMD_x86
#    define PLAY_CGLM_SIMD_x86
#  endif
#endif

#if defined(__SSE2__)
#include <emmintrin.h>
#  define PLAY_CGLM_SSE2_FP 1
#  ifndef PLAY_CGLM_SIMD_x86
#    define PLAY_CGLM_SIMD_x86
#  endif
#endif

#if defined(__SSE3__)
#include <pmmintrin.h>
#  ifndef PLAY_CGLM_SIMD_x86
#    define PLAY_CGLM_SIMD_x86
#  endif
#endif

#if defined(__SSE4_1__)
#include <smmintrin.h>
#  ifndef PLAY_CGLM_SIMD_x86
#    define PLAY_CGLM_SIMD_x86
#  endif
#endif

#if defined(__SSE4_2__)
#include <nmmintrin.h>
#  ifndef PLAY_CGLM_SIMD_x86
#    define PLAY_CGLM_SIMD_x86
#  endif
#endif

/* ARM Neon */
#if defined(_WIN32) && defined(_MSC_VER)

#  if defined(_M_ARM64) || defined(_M_HYBRID_X86_ARM64) || defined(_M_ARM64EC)
#include <arm64intr.h>
#include <arm64_neon.h>
#    ifndef PLAY_CGLM_NEON_FP
#      define PLAY_CGLM_NEON_FP  1
#    endif
#    ifndef PLAY_CGLM_SIMD_ARM
#      define PLAY_CGLM_SIMD_ARM
#    endif
#  elif defined(_M_ARM)
#include <armintr.h>
#include <arm_neon.h>
#    ifndef PLAY_CGLM_NEON_FP
#      define PLAY_CGLM_NEON_FP 1
#    endif
#    ifndef PLAY_CGLM_SIMD_ARM
#      define PLAY_CGLM_SIMD_ARM
#    endif
#  endif

#else /* non-windows */
#  if defined(__ARM_NEON) || defined(__ARM_NEON__)
#include <arm_neon.h>
#    if defined(__ARM_NEON_FP) || defined(__ARM_FP)
#      define PLAY_CGLM_NEON_FP 1
#    endif
#    ifndef PLAY_CGLM_SIMD_ARM
#      define PLAY_CGLM_SIMD_ARM
#    endif
#  endif
#endif

/* WebAssembly */
#if defined(__wasm__) && defined(__wasm_simd128__)
#  ifndef PLAY_CGLM_SIMD_WASM
#    define PLAY_CGLM_SIMD_WASM
#include <wasm_simd128.h>
#  endif
#endif

#if defined(PLAY_CGLM_SIMD_x86) || defined(PLAY_CGLM_SIMD_ARM) || defined(PLAY_CGLM_SIMD_WASM)
#  ifndef PLAY_CGLM_SIMD
#    define PLAY_CGLM_SIMD
#  endif
#endif


/*** Start of inlined file: cglm.h ***/
#ifndef cglmath_h
#define cglmath_h


/*** Start of inlined file: common.h ***/



#define __cglmath__ 1

#ifndef _USE_MATH_DEFINES
#  define _USE_MATH_DEFINES       /* for windows */
#endif

#ifndef _CRT_SECURE_NO_WARNINGS
#  define _CRT_SECURE_NO_WARNINGS /* for windows */
#endif


#if defined(_MSC_VER)
#  ifdef PLAY_CGLM_STATIC
#    define PLAY_CGLM_EXPORT
#  elif defined(PLAY_CGLM_EXPORTS)
#    define PLAY_CGLM_EXPORT __declspec(dllexport)
#  else
#    define PLAY_CGLM_EXPORT __declspec(dllimport)
#  endif
#  define PLAY_CGLM_INLINE __forceinline
#else
#  define PLAY_CGLM_EXPORT __attribute__((visibility("default")))
#  define PLAY_CGLM_INLINE static inline __attribute((always_inline))
#endif

#if defined(__GNUC__) || defined(__clang__)
#  define PLAY_CGLM_UNLIKELY(expr) __builtin_expect(!!(expr), 0)
#  define PLAY_CGLM_LIKELY(expr)   __builtin_expect(!!(expr), 1)
#else
#  define PLAY_CGLM_UNLIKELY(expr) (expr)
#  define PLAY_CGLM_LIKELY(expr)   (expr)
#endif

#if defined(_M_FP_FAST) || defined(__FAST_MATH__)
#  define PLAY_CGLM_FAST_MATH
#endif

#define PLAY_CGLM_SHUFFLE4(z, y, x, w) (((z) << 6) | ((y) << 4) | ((x) << 2) | (w))
#define PLAY_CGLM_SHUFFLE3(z, y, x)    (((z) << 4) | ((y) << 2) | (x))
#define PLAY_CGLM_SHUFFLE2(y, x)       (((y) << 2) | (x))


/*** Start of inlined file: types.h ***/




#if defined(_MSC_VER)
/* do not use alignment for older visual studio versions */
/* also ARM32 also causes similar error, disable it for now on ARM32 too */
#  if _MSC_VER < 1913 || _M_ARM /*  Visual Studio 2017 version 15.6  */
#    define PLAY_CGLM_ALL_UNALIGNED
#    define PLAY_CGLM_ALIGN(X) /* no alignment */
#  else
#    define PLAY_CGLM_ALIGN(X) __declspec(align(X))
#  endif
#else
#  define PLAY_CGLM_ALIGN(X) __attribute((aligned(X)))
#endif

#ifndef PLAY_CGLM_ALL_UNALIGNED
#  define PLAY_CGLM_ALIGN_IF(X) PLAY_CGLM_ALIGN(X)
#else
#  define PLAY_CGLM_ALIGN_IF(X) /* no alignment */
#endif

#ifdef __AVX__
#  define PLAY_CGLM_ALIGN_MAT PLAY_CGLM_ALIGN(32)
#else
#  define PLAY_CGLM_ALIGN_MAT PLAY_CGLM_ALIGN(16)
#endif

#ifndef PLAY_CGLM_HAVE_BUILTIN_ASSUME_ALIGNED

#  if defined(__has_builtin)
#    if __has_builtin(__builtin_assume_aligned)
#      define PLAY_CGLM_HAVE_BUILTIN_ASSUME_ALIGNED 1
#    endif
#  elif defined(__GNUC__) && defined(__GNUC_MINOR__)
#    if __GNUC__ >= 4 && __GNUC_MINOR__ >= 7
#      define PLAY_CGLM_HAVE_BUILTIN_ASSUME_ALIGNED 1
#    endif
#  endif

#  ifndef PLAY_CGLM_HAVE_BUILTIN_ASSUME_ALIGNED
#    define PLAY_CGLM_HAVE_BUILTIN_ASSUME_ALIGNED 0
#  endif

#endif

#if PLAY_CGLM_HAVE_BUILTIN_ASSUME_ALIGNED
#  define PLAY_CGLM_ASSUME_ALIGNED(expr, alignment) \
     __builtin_assume_aligned((expr), (alignment))
#else
#  define PLAY_CGLM_ASSUME_ALIGNED(expr, alignment) (expr)
#endif

#if (defined(__STDC_VERSION__) && __STDC_VERSION__ >= 201112L)
# define PLAY_CGLM_CASTPTR_ASSUME_ALIGNED(expr, type) \
   ((type*)PLAY_CGLM_ASSUME_ALIGNED((expr), alignof(type)))
#elif defined(_MSC_VER)
# define PLAY_CGLM_CASTPTR_ASSUME_ALIGNED(expr, type) \
   ((type*)PLAY_CGLM_ASSUME_ALIGNED((expr), __alignof(type)))
#else
# define PLAY_CGLM_CASTPTR_ASSUME_ALIGNED(expr, type) \
   ((type*)PLAY_CGLM_ASSUME_ALIGNED((expr), __alignof__(type)))
#endif

#ifdef __cplusplus
namespace play
{
#endif



typedef int                     ivec2[2];
typedef int                     ivec3[3];
typedef int                     ivec4[4];

typedef float                   vec2[2];
typedef float                   vec3[3];
typedef PLAY_CGLM_ALIGN_IF(16) float vec4[4];
typedef vec4                    versor;     /* |x, y, z, w| -> w is the last */
typedef vec3                    mat3[3];
typedef vec2                    mat3x2[3];  /* [col (3), row (2)] */
typedef vec4                    mat3x4[3];  /* [col (3), row (4)] */
typedef PLAY_CGLM_ALIGN_IF(16) vec2  mat2[2];
typedef vec3                    mat2x3[2];  /* [col (2), row (3)] */
typedef vec4                    mat2x4[2];  /* [col (2), row (4)] */
typedef PLAY_CGLM_ALIGN_MAT    vec4  mat4[4];
typedef vec2                    mat4x2[4];  /* [col (4), row (2)] */
typedef vec3                    mat4x3[4];  /* [col (4), row (3)] */

/*
  Important: cglm stores quaternion as [x, y, z, w] in memory since v0.4.0
  it was [w, x, y, z] before v0.4.0 ( v0.3.5 and earlier ). w is real part.
*/

#define PLAY_CGLM_E         2.71828182845904523536028747135266250   /* e           */
#define PLAY_CGLM_LOG2E     1.44269504088896340735992468100189214   /* log2(e)     */
#define PLAY_CGLM_LOG10E    0.434294481903251827651128918916605082  /* log10(e)    */
#define PLAY_CGLM_LN2       0.693147180559945309417232121458176568  /* loge(2)     */
#define PLAY_CGLM_LN10      2.30258509299404568401799145468436421   /* loge(10)    */
#define PLAY_CGLM_PI        3.14159265358979323846264338327950288   /* pi          */
#define PLAY_CGLM_PI_2      1.57079632679489661923132169163975144   /* pi/2        */
#define PLAY_CGLM_PI_4      0.785398163397448309615660845819875721  /* pi/4        */
#define PLAY_CGLM_1_PI      0.318309886183790671537767526745028724  /* 1/pi        */
#define PLAY_CGLM_2_PI      0.636619772367581343075535053490057448  /* 2/pi        */
#define PLAY_CGLM_TAU       6.283185307179586476925286766559005768  /* tau         */
#define PLAY_CGLM_TAU_2     PLAY_CGLM_PI                                  /* tau/2       */
#define PLAY_CGLM_TAU_4     PLAY_CGLM_PI_2                                /* tau/4       */
#define PLAY_CGLM_1_TAU     0.159154943091895335768883763372514362  /* 1/tau       */
#define PLAY_CGLM_2_TAU     0.318309886183790671537767526745028724  /* 2/tau       */
#define PLAY_CGLM_2_SQRTPI  1.12837916709551257389615890312154517   /* 2/sqrt(pi)  */
#define PLAY_CGLM_SQRTTAU   2.506628274631000502415765284811045253  /* sqrt(tau)   */
#define PLAY_CGLM_SQRT2     1.41421356237309504880168872420969808   /* sqrt(2)     */
#define PLAY_CGLM_SQRT1_2   0.707106781186547524400844362104849039  /* 1/sqrt(2)   */

#define PLAY_CGLM_Ef         ((float)PLAY_CGLM_E)
#define PLAY_CGLM_LOG2Ef     ((float)PLAY_CGLM_LOG2E)
#define PLAY_CGLM_LOG10Ef    ((float)PLAY_CGLM_LOG10E)
#define PLAY_CGLM_LN2f       ((float)PLAY_CGLM_LN2)
#define PLAY_CGLM_LN10f      ((float)PLAY_CGLM_LN10)
#define PLAY_CGLM_PIf        ((float)PLAY_CGLM_PI)
#define PLAY_CGLM_PI_2f      ((float)PLAY_CGLM_PI_2)
#define PLAY_CGLM_PI_4f      ((float)PLAY_CGLM_PI_4)
#define PLAY_CGLM_1_PIf      ((float)PLAY_CGLM_1_PI)
#define PLAY_CGLM_2_PIf      ((float)PLAY_CGLM_2_PI)
#define PLAY_CGLM_TAUf       ((float)PLAY_CGLM_TAU)
#define PLAY_CGLM_TAU_2f     ((float)PLAY_CGLM_TAU_2)
#define PLAY_CGLM_TAU_4f     ((float)PLAY_CGLM_TAU_4)
#define PLAY_CGLM_1_TAUf     ((float)PLAY_CGLM_1_TAU)
#define PLAY_CGLM_2_TAUf     ((float)PLAY_CGLM_2_TAU)
#define PLAY_CGLM_2_SQRTPIf  ((float)PLAY_CGLM_2_SQRTPI)
#define PLAY_CGLM_2_SQRTTAUf ((float)PLAY_CGLM_SQRTTAU)
#define PLAY_CGLM_SQRT2f     ((float)PLAY_CGLM_SQRT2)
#define PLAY_CGLM_SQRT1_2f   ((float)PLAY_CGLM_SQRT1_2)



/*** End of inlined file: types.h ***/


/*** Start of inlined file: intrin.h ***/

#if defined(PLAY_CGLM_SIMD_x86) && !defined(PLAY_CGLM_SIMD_WASM)

/*** Start of inlined file: x86.h ***/



#ifdef PLAY_CGLM_SIMD_x86

#ifdef PLAY_CGLM_ALL_UNALIGNED
#  define glmm_load(p)      _mm_loadu_ps(p)
#  define glmm_store(p, a)  _mm_storeu_ps(p, a)
#else
#  define glmm_load(p)      _mm_load_ps(p)
#  define glmm_store(p, a)  _mm_store_ps(p, a)
#endif

#define glmm_128     __m128

#ifdef __AVX__
#  define glmm_shuff1(xmm, z, y, x, w)                                        \
     _mm_permute_ps((xmm), _MM_SHUFFLE(z, y, x, w))
#else
#  if !defined(PLAY_CGLM_NO_INT_DOMAIN) && defined(__SSE2__)
#    define glmm_shuff1(xmm, z, y, x, w)                                      \
       _mm_castsi128_ps(_mm_shuffle_epi32(_mm_castps_si128(xmm),              \
                                          _MM_SHUFFLE(z, y, x, w)))
#  else
#    define glmm_shuff1(xmm, z, y, x, w)                                      \
       _mm_shuffle_ps(xmm, xmm, _MM_SHUFFLE(z, y, x, w))
#  endif
#endif

#define glmm_splat(x, lane) glmm_shuff1(x, lane, lane, lane, lane)

#ifdef __AVX__
#  define glmm_set1(x)      _mm_broadcast_ss(&x)
#  define glmm_set1_ptr(x)  _mm_broadcast_ss(x)
#  define glmm_set1_rval(x) _mm_set1_ps(x)
#  ifdef __AVX2__
#    define glmm_splat_x(x) _mm_broadcastss_ps(x)
#  else
#    define glmm_splat_x(x) _mm_permute_ps(x, _MM_SHUFFLE(0, 0, 0, 0))
#  endif
#  define glmm_splat_y(x)   _mm_permute_ps(x, _MM_SHUFFLE(1, 1, 1, 1))
#  define glmm_splat_z(x)   _mm_permute_ps(x, _MM_SHUFFLE(2, 2, 2, 2))
#  define glmm_splat_w(x)   _mm_permute_ps(x, _MM_SHUFFLE(3, 3, 3, 3))
#else
#  define glmm_set1(x)      _mm_set1_ps(x)
#  define glmm_set1_ptr(x)  _mm_set1_ps(*x)
#  define glmm_set1_rval(x) _mm_set1_ps(x)

#  define glmm_splat_x(x)   glmm_splat(x, 0)
#  define glmm_splat_y(x)   glmm_splat(x, 1)
#  define glmm_splat_z(x)   glmm_splat(x, 2)
#  define glmm_splat_w(x)   glmm_splat(x, 3)
#endif

#ifdef __AVX__
#  ifdef PLAY_CGLM_ALL_UNALIGNED
#    define glmm_load256(p)      _mm256_loadu_ps(p)
#    define glmm_store256(p, a)  _mm256_storeu_ps(p, a)
#  else
#    define glmm_load256(p)      _mm256_load_ps(p)
#    define glmm_store256(p, a)  _mm256_store_ps(p, a)
#  endif
#endif

/* Note that `0x80000000` corresponds to `INT_MIN` for a 32-bit int. */

#if defined(__SSE2__)
#  define GLMM_NEGZEROf ((int)0x80000000) /*  0x80000000 ---> -0.0f  */
#  define GLMM_POSZEROf ((int)0x00000000) /*  0x00000000 ---> +0.0f  */
#else
#  ifdef PLAY_CGLM_FAST_MATH
union { int i; float f; } static GLMM_NEGZEROf_TU = { .i = (int)0x80000000 };
#    define GLMM_NEGZEROf GLMM_NEGZEROf_TU.f
#    define GLMM_POSZEROf 0.0f
#  else
#    define GLMM_NEGZEROf -0.0f
#    define GLMM_POSZEROf  0.0f
#  endif
#endif

#if defined(__SSE2__)
#  define GLMM__SIGNMASKf(X, Y, Z, W)                                         \
   _mm_castsi128_ps(_mm_set_epi32(X, Y, Z, W))
/* _mm_set_ps(X, Y, Z, W); */
#else
#  define GLMM__SIGNMASKf(X, Y, Z, W)  _mm_set_ps(X, Y, Z, W)
#endif

#define glmm_float32x4_SIGNMASK_PNPN GLMM__SIGNMASKf(GLMM_POSZEROf, GLMM_NEGZEROf, GLMM_POSZEROf, GLMM_NEGZEROf)
#define glmm_float32x4_SIGNMASK_NPNP GLMM__SIGNMASKf(GLMM_NEGZEROf, GLMM_POSZEROf, GLMM_NEGZEROf, GLMM_POSZEROf)
#define glmm_float32x4_SIGNMASK_NPPN GLMM__SIGNMASKf(GLMM_NEGZEROf, GLMM_POSZEROf, GLMM_POSZEROf, GLMM_NEGZEROf)

/* fasth math prevents -0.0f to work */
#if defined(__SSE2__)
#  define glmm_float32x4_SIGNMASK_NEG _mm_castsi128_ps(_mm_set1_epi32(GLMM_NEGZEROf)) /* _mm_set1_ps(-0.0f) */
#else
#  define glmm_float32x4_SIGNMASK_NEG glmm_set1(GLMM_NEGZEROf)
#endif

#define glmm_float32x8_SIGNMASK_NEG _mm256_castsi256_ps(_mm256_set1_epi32(GLMM_NEGZEROf))

static inline
__m128
glmm_abs(__m128 x)
{
    return _mm_andnot_ps(glmm_float32x4_SIGNMASK_NEG, x);
}

static inline __m128
glmm_min(__m128 a, __m128 b) { return _mm_min_ps(a, b); }
static inline __m128
glmm_max(__m128 a, __m128 b) { return _mm_max_ps(a, b); }

static inline
__m128
glmm_vhadd(__m128 v)
{
    __m128 x0;
    x0 = _mm_add_ps(v,  glmm_shuff1(v, 0, 1, 2, 3));
    x0 = _mm_add_ps(x0, glmm_shuff1(x0, 1, 0, 0, 1));
    return x0;
}

static inline
__m128
glmm_vhadds(__m128 v)
{
#if defined(__SSE3__)
    __m128 shuf, sums;
    shuf = _mm_movehdup_ps(v);
    sums = _mm_add_ps(v, shuf);
    shuf = _mm_movehl_ps(shuf, sums);
    sums = _mm_add_ss(sums, shuf);
    return sums;
#else
    __m128 shuf, sums;
    shuf = glmm_shuff1(v, 2, 3, 0, 1);
    sums = _mm_add_ps(v, shuf);
    shuf = _mm_movehl_ps(shuf, sums);
    sums = _mm_add_ss(sums, shuf);
    return sums;
#endif
}

static inline
float
glmm_hadd(__m128 v)
{
    return _mm_cvtss_f32(glmm_vhadds(v));
}

static inline
__m128
glmm_vhmin(__m128 v)
{
    __m128 x0, x1, x2;
    x0 = _mm_movehl_ps(v, v);     /* [2, 3, 2, 3] */
    x1 = _mm_min_ps(x0, v);       /* [0|2, 1|3, 2|2, 3|3] */
    x2 = glmm_splat(x1, 1);       /* [1|3, 1|3, 1|3, 1|3] */
    return _mm_min_ss(x1, x2);
}

static inline
float
glmm_hmin(__m128 v)
{
    return _mm_cvtss_f32(glmm_vhmin(v));
}

static inline
__m128
glmm_vhmax(__m128 v)
{
    __m128 x0, x1, x2;
    x0 = _mm_movehl_ps(v, v);     /* [2, 3, 2, 3] */
    x1 = _mm_max_ps(x0, v);       /* [0|2, 1|3, 2|2, 3|3] */
    x2 = glmm_splat(x1, 1);       /* [1|3, 1|3, 1|3, 1|3] */
    return _mm_max_ss(x1, x2);
}

static inline
float
glmm_hmax(__m128 v)
{
    return _mm_cvtss_f32(glmm_vhmax(v));
}

static inline
__m128
glmm_vdots(__m128 a, __m128 b)
{
#if (defined(__SSE4_1__) || defined(__SSE4_2__)) && defined(PLAY_CGLM_SSE4_DOT)
    return _mm_dp_ps(a, b, 0xFF);
#elif defined(__SSE3__) && defined(PLAY_CGLM_SSE3_DOT)
    __m128 x0, x1;
    x0 = _mm_mul_ps(a, b);
    x1 = _mm_hadd_ps(x0, x0);
    return _mm_hadd_ps(x1, x1);
#else
    return glmm_vhadds(_mm_mul_ps(a, b));
#endif
}

static inline
__m128
glmm_vdot(__m128 a, __m128 b)
{
#if (defined(__SSE4_1__) || defined(__SSE4_2__)) && defined(PLAY_CGLM_SSE4_DOT)
    return _mm_dp_ps(a, b, 0xFF);
#elif defined(__SSE3__) && defined(PLAY_CGLM_SSE3_DOT)
    __m128 x0, x1;
    x0 = _mm_mul_ps(a, b);
    x1 = _mm_hadd_ps(x0, x0);
    return _mm_hadd_ps(x1, x1);
#else
    __m128 x0;
    x0 = _mm_mul_ps(a, b);
    x0 = _mm_add_ps(x0, glmm_shuff1(x0, 1, 0, 3, 2));
    return _mm_add_ps(x0, glmm_shuff1(x0, 0, 1, 0, 1));
#endif
}

static inline
float
glmm_dot(__m128 a, __m128 b)
{
    return _mm_cvtss_f32(glmm_vdots(a, b));
}

static inline
float
glmm_norm(__m128 a)
{
    return _mm_cvtss_f32(_mm_sqrt_ss(glmm_vhadds(_mm_mul_ps(a, a))));
}

static inline
float
glmm_norm2(__m128 a)
{
    return _mm_cvtss_f32(glmm_vhadds(_mm_mul_ps(a, a)));
}

static inline
float
glmm_norm_one(__m128 a)
{
    return _mm_cvtss_f32(glmm_vhadds(glmm_abs(a)));
}

static inline
float
glmm_norm_inf(__m128 a)
{
    return _mm_cvtss_f32(glmm_vhmax(glmm_abs(a)));
}

#if defined(__SSE2__)
static inline
__m128
glmm_load3(float v[3])
{
    __m128i xy;
    __m128  z;

    xy = _mm_loadl_epi64(PLAY_CGLM_CASTPTR_ASSUME_ALIGNED(v, const __m128i));
    z  = _mm_load_ss(&v[2]);

    return _mm_movelh_ps(_mm_castsi128_ps(xy), z);
}

static inline
void
glmm_store3(float v[3], __m128 vx)
{
    _mm_storel_pi(PLAY_CGLM_CASTPTR_ASSUME_ALIGNED(v, __m64), vx);
    _mm_store_ss(&v[2], glmm_shuff1(vx, 2, 2, 2, 2));
}
#endif

static inline
__m128
glmm_div(__m128 a, __m128 b)
{
    return _mm_div_ps(a, b);
}

/* enable FMA macro for MSVC? */
#if defined(_MSC_VER) && !defined(__FMA__) && defined(__AVX2__)
#  define __FMA__ 1
#endif

static inline
__m128
glmm_fmadd(__m128 a, __m128 b, __m128 c)
{
#ifdef __FMA__
    return _mm_fmadd_ps(a, b, c);
#else
    return _mm_add_ps(c, _mm_mul_ps(a, b));
#endif
}

static inline
__m128
glmm_fnmadd(__m128 a, __m128 b, __m128 c)
{
#ifdef __FMA__
    return _mm_fnmadd_ps(a, b, c);
#else
    return _mm_sub_ps(c, _mm_mul_ps(a, b));
#endif
}

static inline
__m128
glmm_fmsub(__m128 a, __m128 b, __m128 c)
{
#ifdef __FMA__
    return _mm_fmsub_ps(a, b, c);
#else
    return _mm_sub_ps(_mm_mul_ps(a, b), c);
#endif
}

static inline
__m128
glmm_fnmsub(__m128 a, __m128 b, __m128 c)
{
#ifdef __FMA__
    return _mm_fnmsub_ps(a, b, c);
#else
    return _mm_xor_ps(_mm_add_ps(_mm_mul_ps(a, b), c),
                      glmm_float32x4_SIGNMASK_NEG);
#endif
}

#if defined(__AVX__)
static inline
__m256
glmm256_fmadd(__m256 a, __m256 b, __m256 c)
{
#ifdef __FMA__
    return _mm256_fmadd_ps(a, b, c);
#else
    return _mm256_add_ps(c, _mm256_mul_ps(a, b));
#endif
}

static inline
__m256
glmm256_fnmadd(__m256 a, __m256 b, __m256 c)
{
#ifdef __FMA__
    return _mm256_fnmadd_ps(a, b, c);
#else
    return _mm256_sub_ps(c, _mm256_mul_ps(a, b));
#endif
}

static inline
__m256
glmm256_fmsub(__m256 a, __m256 b, __m256 c)
{
#ifdef __FMA__
    return _mm256_fmsub_ps(a, b, c);
#else
    return _mm256_sub_ps(_mm256_mul_ps(a, b), c);
#endif
}

static inline
__m256
glmm256_fnmsub(__m256 a, __m256 b, __m256 c)
{
#ifdef __FMA__
    return _mm256_fmsub_ps(a, b, c);
#else
    return _mm256_xor_ps(_mm256_sub_ps(_mm256_mul_ps(a, b), c),
                         glmm_float32x8_SIGNMASK_NEG);
#endif
}
#endif

#endif


/*** End of inlined file: x86.h ***/


#endif

#if defined(PLAY_CGLM_SIMD_ARM)

/*** Start of inlined file: arm.h ***/



#ifdef PLAY_CGLM_SIMD_ARM

#if defined(_M_ARM64) || defined(_M_HYBRID_X86_ARM64) || defined(_M_ARM64EC) || defined(__aarch64__)
# define PLAY_CGLM_ARM64 1
#else
# define PLAY_CGLM_ARM64 0
#endif

#define glmm_load(p)      vld1q_f32(p)
#define glmm_store(p, a)  vst1q_f32(p, a)

#define glmm_set1(x)      vdupq_n_f32(x)
#define glmm_set1_ptr(x)  vdupq_n_f32(*x)
#define glmm_set1_rval(x) vdupq_n_f32(x)
#define glmm_128          float32x4_t

#define glmm_splat_x(x) vdupq_lane_f32(vget_low_f32(x),  0)
#define glmm_splat_y(x) vdupq_lane_f32(vget_low_f32(x),  1)
#define glmm_splat_z(x) vdupq_lane_f32(vget_high_f32(x), 0)
#define glmm_splat_w(x) vdupq_lane_f32(vget_high_f32(x), 1)

#define glmm_xor(a, b)                                                        \
  vreinterpretq_f32_s32(veorq_s32(vreinterpretq_s32_f32(a),                   \
                                  vreinterpretq_s32_f32(b)))

#define glmm_swplane(v) vextq_f32(v, v, 2)
#define glmm_low(x)     vget_low_f32(x)
#define glmm_high(x)    vget_high_f32(x)

#define glmm_combine_ll(x, y) vcombine_f32(vget_low_f32(x),  vget_low_f32(y))
#define glmm_combine_hl(x, y) vcombine_f32(vget_high_f32(x), vget_low_f32(y))
#define glmm_combine_lh(x, y) vcombine_f32(vget_low_f32(x),  vget_high_f32(y))
#define glmm_combine_hh(x, y) vcombine_f32(vget_high_f32(x), vget_high_f32(y))

#if defined(_WIN32) && defined(_MSC_VER)
/* #  define glmm_float32x4_init(x, y, z, w) { .n128_f32 = { x, y, z, w } } */
PLAY_CGLM_INLINE
float32x4_t
glmm_float32x4_init(float x, float y, float z, float w)
{
    PLAY_CGLM_ALIGN(16) float v[4] = {x, y, z, w};
    return vld1q_f32(v);
}
#else
#  define glmm_float32x4_init(x, y, z, w) { x, y, z, w }
#endif

#define glmm_float32x4_SIGNMASK_PNPN glmm_float32x4_init( 0.f, -0.f,  0.f, -0.f)
#define glmm_float32x4_SIGNMASK_NPNP glmm_float32x4_init(-0.f,  0.f, -0.f,  0.f)
#define glmm_float32x4_SIGNMASK_NPPN glmm_float32x4_init(-0.f,  0.f,  0.f, -0.f)

static inline float32x4_t
glmm_abs(float32x4_t v)                { return vabsq_f32(v);    }
static inline float32x4_t
glmm_min(float32x4_t a, float32x4_t b) { return vminq_f32(a, b); }
static inline float32x4_t
glmm_max(float32x4_t a, float32x4_t b) { return vmaxq_f32(a, b); }

static inline
float32x4_t
glmm_vhadd(float32x4_t v)
{
#if PLAY_CGLM_ARM64
    float32x4_t p;
    p = vpaddq_f32(v, v); /* [a+b, c+d, a+b, c+d] */
    return vpaddq_f32(p, p); /* [t, t, t, t] */;
#else
    return vaddq_f32(vaddq_f32(glmm_splat_x(v), glmm_splat_y(v)),
                     vaddq_f32(glmm_splat_z(v), glmm_splat_w(v)));
#endif
    /* TODO: measure speed of this compare to above */
    /* return vdupq_n_f32(vaddvq_f32(v)); */

    /*
    return vaddq_f32(vaddq_f32(glmm_splat_x(v), glmm_splat_y(v)),
                     vaddq_f32(glmm_splat_z(v), glmm_splat_w(v)));
     */
    /*
     this seems slower:
     v = vaddq_f32(v, vrev64q_f32(v));
     return vaddq_f32(v, vcombine_f32(vget_high_f32(v), vget_low_f32(v)));
     */
}

static inline
float
glmm_hadd(float32x4_t v)
{
#if PLAY_CGLM_ARM64
    return vaddvq_f32(v);
#else
    v = vaddq_f32(v, vrev64q_f32(v));
    v = vaddq_f32(v, vcombine_f32(vget_high_f32(v), vget_low_f32(v)));
    return vgetq_lane_f32(v, 0);
#endif
}

static inline
float
glmm_hmin(float32x4_t v)
{
    float32x2_t t;
    t = vpmin_f32(vget_low_f32(v), vget_high_f32(v));
    t = vpmin_f32(t, t);
    return vget_lane_f32(t, 0);
}

static inline
float
glmm_hmax(float32x4_t v)
{
    float32x2_t t;
    t = vpmax_f32(vget_low_f32(v), vget_high_f32(v));
    t = vpmax_f32(t, t);
    return vget_lane_f32(t, 0);
}

static inline
float
glmm_dot(float32x4_t a, float32x4_t b)
{
    return glmm_hadd(vmulq_f32(a, b));
}

static inline
float32x4_t
glmm_vdot(float32x4_t a, float32x4_t b)
{
    return glmm_vhadd(vmulq_f32(a, b));
}

static inline
float
glmm_norm(float32x4_t a)
{
    return sqrtf(glmm_dot(a, a));
}

static inline
float
glmm_norm2(float32x4_t a)
{
    return glmm_dot(a, a);
}

static inline
float
glmm_norm_one(float32x4_t a)
{
    return glmm_hadd(glmm_abs(a));
}

static inline
float
glmm_norm_inf(float32x4_t a)
{
    return glmm_hmax(glmm_abs(a));
}

static inline
float32x4_t
glmm_div(float32x4_t a, float32x4_t b)
{
#if PLAY_CGLM_ARM64
    return vdivq_f32(a, b);
#else
    /* 2 iterations of Newton-Raphson refinement of reciprocal */
    float32x4_t r0, r1;
    r0 = vrecpeq_f32(b);
    r1 = vrecpsq_f32(r0, b);
    r0 = vmulq_f32(r1, r0);
    r1 = vrecpsq_f32(r0, b);
    r0 = vmulq_f32(r1, r0);
    return vmulq_f32(a, r0);
#endif
}

static inline
float32x4_t
glmm_fmadd(float32x4_t a, float32x4_t b, float32x4_t c)
{
#if PLAY_CGLM_ARM64
    return vfmaq_f32(c, a, b); /* why vfmaq_f32 is slower than vmlaq_f32 ??? */
#else
    return vmlaq_f32(c, a, b);
#endif
}

static inline
float32x4_t
glmm_fnmadd(float32x4_t a, float32x4_t b, float32x4_t c)
{
#if PLAY_CGLM_ARM64
    return vfmsq_f32(c, a, b);
#else
    return vmlsq_f32(c, a, b);
#endif
}

static inline
float32x4_t
glmm_fmsub(float32x4_t a, float32x4_t b, float32x4_t c)
{
    return glmm_fmadd(a, b, vnegq_f32(c));
}

static inline
float32x4_t
glmm_fnmsub(float32x4_t a, float32x4_t b, float32x4_t c)
{
    return vsubq_f32(vdupq_n_f32(0.0f), glmm_fmadd(a, b, c));
}

#endif


/*** End of inlined file: arm.h ***/


#endif

#if defined(PLAY_CGLM_SIMD_WASM)

/*** Start of inlined file: wasm.h ***/



#ifdef PLAY_CGLM_SIMD_WASM

#define glmm_load(p)      wasm_v128_load(p)
#define glmm_store(p, a)  wasm_v128_store(p, (a))

#define glmm_set1(x)      wasm_f32x4_splat(x)
#define glmm_set1_ptr(x)  wasm_f32x4_splat(*x)
#define glmm_set1_rval(x) wasm_f32x4_splat(x)
#define glmm_128          v128_t

#define glmm_shuff1(xmm, z, y, x, w) wasm_i32x4_shuffle(xmm, xmm, w, x, y, z)

#define glmm_splat(x, lane) glmm_shuff1(x, lane, lane, lane, lane)

#define glmm_splat_x(x) glmm_splat(x, 0)
#define glmm_splat_y(x) glmm_splat(x, 1)
#define glmm_splat_z(x) glmm_splat(x, 2)
#define glmm_splat_w(x) glmm_splat(x, 3)

#define GLMM_NEGZEROf 0x80000000 /*  0x80000000 ---> -0.0f  */

/* _mm_set_ps(X, Y, Z, W); */
#define GLMM__SIGNMASKf(X, Y, Z, W) wasm_i32x4_const(X, Y, Z, W)

#define glmm_float32x4_SIGNMASK_PNPN GLMM__SIGNMASKf(0, GLMM_NEGZEROf, 0, GLMM_NEGZEROf)
#define glmm_float32x4_SIGNMASK_NPNP GLMM__SIGNMASKf(GLMM_NEGZEROf, 0, GLMM_NEGZEROf, 0)
#define glmm_float32x4_SIGNMASK_NPPN GLMM__SIGNMASKf(GLMM_NEGZEROf, 0, 0, GLMM_NEGZEROf)
#define glmm_float32x4_SIGNMASK_NEG  wasm_i32x4_const_splat(GLMM_NEGZEROf)

static inline glmm_128
glmm_abs(glmm_128 x)             { return wasm_f32x4_abs(x);     }
static inline glmm_128
glmm_min(glmm_128 a, glmm_128 b) { return wasm_f32x4_pmin(b, a); }
static inline glmm_128
glmm_max(glmm_128 a, glmm_128 b) { return wasm_f32x4_pmax(b, a); }

static inline
glmm_128
glmm_vhadd(glmm_128 v)
{
    glmm_128 x0;
    x0 = wasm_f32x4_add(v,  glmm_shuff1(v, 0, 1, 2, 3));
    x0 = wasm_f32x4_add(x0, glmm_shuff1(x0, 1, 0, 0, 1));
    return x0;
}

static inline
glmm_128
glmm_vhadds(glmm_128 v)
{
    glmm_128 shuf, sums;
    shuf = glmm_shuff1(v, 2, 3, 0, 1);
    sums = wasm_f32x4_add(v, shuf);
    /* shuf = _mm_movehl_ps(shuf, sums); */
    shuf = wasm_i32x4_shuffle(shuf, sums, 6, 7, 2, 3);
    sums = wasm_i32x4_shuffle(sums, wasm_f32x4_add(sums, shuf), 4, 1, 2, 3);
    return sums;
}

static inline
float
glmm_hadd(glmm_128 v)
{
    return wasm_f32x4_extract_lane(glmm_vhadds(v), 0);
}

static inline
glmm_128
glmm_vhmin(glmm_128 v)
{
    glmm_128 x0, x1, x2;
    x0 = glmm_shuff1(v, 2, 3, 2, 3);     /* [2, 3, 2, 3] */
    x1 = wasm_f32x4_pmin(x0, v);         /* [0|2, 1|3, 2|2, 3|3] */
    x2 = glmm_splat(x1, 1);              /* [1|3, 1|3, 1|3, 1|3] */
    return wasm_f32x4_pmin(x1, x2);
}

static inline
float
glmm_hmin(glmm_128 v)
{
    return wasm_f32x4_extract_lane(glmm_vhmin(v), 0);
}

static inline
glmm_128
glmm_vhmax(glmm_128 v)
{
    glmm_128 x0, x1, x2;
    x0 = glmm_shuff1(v, 2, 3, 2, 3);     /* [2, 3, 2, 3] */
    x1 = wasm_f32x4_pmax(x0, v);         /* [0|2, 1|3, 2|2, 3|3] */
    x2 = glmm_splat(x1, 1);              /* [1|3, 1|3, 1|3, 1|3] */
    /* _mm_max_ss */
    return wasm_i32x4_shuffle(x1, wasm_f32x4_pmax(x1, x2), 4, 1, 2, 3);
}

static inline
float
glmm_hmax(glmm_128 v)
{
    return wasm_f32x4_extract_lane(glmm_vhmax(v), 0);
}

static inline
glmm_128
glmm_vdots(glmm_128 a, glmm_128 b)
{
    return glmm_vhadds(wasm_f32x4_mul(a, b));
}

static inline
glmm_128
glmm_vdot(glmm_128 a, glmm_128 b)
{
    glmm_128 x0;
    x0 = wasm_f32x4_mul(a, b);
    x0 = wasm_f32x4_add(x0, glmm_shuff1(x0, 1, 0, 3, 2));
    return wasm_f32x4_add(x0, glmm_shuff1(x0, 0, 1, 0, 1));
}

static inline
float
glmm_dot(glmm_128 a, glmm_128 b)
{
    return wasm_f32x4_extract_lane(glmm_vdots(a, b), 0);
}

static inline
float
glmm_norm(glmm_128 a)
{
    glmm_128 x0;
    x0 = glmm_vhadds(wasm_f32x4_mul(a, a));
    return wasm_f32x4_extract_lane(
               wasm_i32x4_shuffle(x0, wasm_f32x4_sqrt(x0),4, 1, 2, 3), 0);
}

static inline
float
glmm_norm2(glmm_128 a)
{
    return wasm_f32x4_extract_lane(glmm_vhadds(wasm_f32x4_mul(a, a)), 0);
}

static inline
float
glmm_norm_one(glmm_128 a)
{
    return wasm_f32x4_extract_lane(glmm_vhadds(glmm_abs(a)), 0);
}

static inline
float
glmm_norm_inf(glmm_128 a)
{
    return wasm_f32x4_extract_lane(glmm_vhmax(glmm_abs(a)), 0);
}

static inline
glmm_128
glmm_load3(float v[3])
{
    glmm_128 xy = wasm_v128_load64_zero(v);
    return wasm_f32x4_replace_lane(xy, 2, v[2]);
}

static inline
void
glmm_store3(float v[3], glmm_128 vx)
{
    wasm_v128_store64_lane(v, vx, 0);
    wasm_v128_store32_lane(&v[2], vx, 2);
}

static inline
glmm_128
glmm_div(glmm_128 a, glmm_128 b)
{
    return wasm_f32x4_div(a, b);
}

static inline
glmm_128
glmm_fmadd(glmm_128 a, glmm_128 b, glmm_128 c)
{
    return wasm_f32x4_add(c, wasm_f32x4_mul(a, b));
}

static inline
glmm_128
glmm_fnmadd(glmm_128 a, glmm_128 b, glmm_128 c)
{
    return wasm_f32x4_sub(c, wasm_f32x4_mul(a, b));
}

static inline
glmm_128
glmm_fmsub(glmm_128 a, glmm_128 b, glmm_128 c)
{
    return wasm_f32x4_sub(wasm_f32x4_mul(a, b), c);
}

static inline
glmm_128
glmm_fnmsub(glmm_128 a, glmm_128 b, glmm_128 c)
{
    return wasm_f32x4_neg(wasm_f32x4_add(wasm_f32x4_mul(a, b), c));
}

#endif


/*** End of inlined file: wasm.h ***/


#endif



/*** End of inlined file: intrin.h ***/

#ifndef PLAY_CGLM_USE_DEFAULT_EPSILON
#  ifndef PLAY_CGLM_FLT_EPSILON
#    define PLAY_CGLM_FLT_EPSILON 1e-5f
#  endif
#else
#  define PLAY_CGLM_FLT_EPSILON FLT_EPSILON
#endif

/*
 * Clip control: define PLAY_CGLM_FORCE_DEPTH_ZERO_TO_ONE before including
 * CGLM to use a clip space between 0 to 1.
 * Coordinate system: define PLAY_CGLM_FORCE_LEFT_HANDED before including
 * CGLM to use the left handed coordinate system by default.
 */

#define PLAY_CGLM_CLIP_CONTROL_ZO_BIT (1 << 0) /* ZERO_TO_ONE */
#define PLAY_CGLM_CLIP_CONTROL_NO_BIT (1 << 1) /* NEGATIVE_ONE_TO_ONE */
#define PLAY_CGLM_CLIP_CONTROL_LH_BIT (1 << 2) /* LEFT_HANDED, For DirectX, Metal, Vulkan */
#define PLAY_CGLM_CLIP_CONTROL_RH_BIT (1 << 3) /* RIGHT_HANDED, For OpenGL, default in GLM */

#define PLAY_CGLM_CLIP_CONTROL_LH_ZO (PLAY_CGLM_CLIP_CONTROL_LH_BIT | PLAY_CGLM_CLIP_CONTROL_ZO_BIT)
#define PLAY_CGLM_CLIP_CONTROL_LH_NO (PLAY_CGLM_CLIP_CONTROL_LH_BIT | PLAY_CGLM_CLIP_CONTROL_NO_BIT)
#define PLAY_CGLM_CLIP_CONTROL_RH_ZO (PLAY_CGLM_CLIP_CONTROL_RH_BIT | PLAY_CGLM_CLIP_CONTROL_ZO_BIT)
#define PLAY_CGLM_CLIP_CONTROL_RH_NO (PLAY_CGLM_CLIP_CONTROL_RH_BIT | PLAY_CGLM_CLIP_CONTROL_NO_BIT)

#ifdef PLAY_CGLM_FORCE_DEPTH_ZERO_TO_ONE
#  ifdef PLAY_CGLM_FORCE_LEFT_HANDED
#    define PLAY_CGLM_CONFIG_CLIP_CONTROL PLAY_CGLM_CLIP_CONTROL_LH_ZO
#  else
#    define PLAY_CGLM_CONFIG_CLIP_CONTROL PLAY_CGLM_CLIP_CONTROL_RH_ZO
#  endif
#else
#  ifdef PLAY_CGLM_FORCE_LEFT_HANDED
#    define PLAY_CGLM_CONFIG_CLIP_CONTROL PLAY_CGLM_CLIP_CONTROL_LH_NO
#  else
#    define PLAY_CGLM_CONFIG_CLIP_CONTROL PLAY_CGLM_CLIP_CONTROL_RH_NO
#  endif
#endif

/* struct API configurator */
/* TODO: move struct/common.h? */
/* WARN: dont use concant helpers outside cglm headers, because they may be changed */

#define PLAY_CGLM_MACRO_CONCAT_HELPER(A, B, C, D, E, ...) A ## B ## C ## D ## E ## __VA_ARGS__
#define PLAY_CGLM_MACRO_CONCAT(A, B, C, D, E, ...) PLAY_CGLM_MACRO_CONCAT_HELPER(A, B, C, D, E,__VA_ARGS__)

#ifndef PLAY_CGLM_OMIT_NS_FROM_STRUCT_API
#  ifndef PLAY_CGLM_STRUCT_API_NS
#    define PLAY_CGLM_STRUCT_API_NS
#  endif
#  ifndef PLAY_CGLM_STRUCT_API_NS_SEPERATOR
#    define PLAY_CGLM_STRUCT_API_NS_SEPERATOR
#  endif
#else
#  define PLAY_CGLM_STRUCT_API_NS
#  define PLAY_CGLM_STRUCT_API_NS_SEPERATOR
#endif

#ifndef PLAY_CGLM_STRUCT_API_NAME_SUFFIX
#  define PLAY_CGLM_STRUCT_API_NAME_SUFFIX s
#endif

#define PLAY_CGLM_STRUCTAPI(A, ...) PLAY_CGLM_MACRO_CONCAT(PLAY_CGLM_STRUCT_API_NS,             \
                                                 PLAY_CGLM_STRUCT_API_NS_SEPERATOR,   \
                                                 A,                              \
                                                 PLAY_CGLM_STRUCT_API_NAME_SUFFIX,    \
                                                 _,                              \
                                                 __VA_ARGS__)



/*** End of inlined file: common.h ***/


/*** Start of inlined file: vec2.h ***/
/*
 Macros:
   PLAY_CGLM_VEC2_ONE_INIT
   PLAY_CGLM_VEC2_ZERO_INIT
   PLAY_CGLM_VEC2_ONE
   PLAY_CGLM_VEC2_ZERO

 Functions:
   PLAY_CGLM_INLINE void  vec2_new(float * __restrict v, vec2 dest)
   PLAY_CGLM_INLINE void  vec2_copy(vec2 a, vec2 dest)
   PLAY_CGLM_INLINE void  vec2_zero(vec2 v)
   PLAY_CGLM_INLINE void  vec2_one(vec2 v)
   PLAY_CGLM_INLINE float vec2_dot(vec2 a, vec2 b)
   PLAY_CGLM_INLINE float vec2_cross(vec2 a, vec2 b)
   PLAY_CGLM_INLINE float vec2_norm2(vec2 v)
   PLAY_CGLM_INLINE float vec2_norm(vec2 vec)
   PLAY_CGLM_INLINE void  vec2_add(vec2 a, vec2 b, vec2 dest)
   PLAY_CGLM_INLINE void  vec2_adds(vec2 v, float s, vec2 dest)
   PLAY_CGLM_INLINE void  vec2_sub(vec2 a, vec2 b, vec2 dest)
   PLAY_CGLM_INLINE void  vec2_subs(vec2 v, float s, vec2 dest)
   PLAY_CGLM_INLINE void  vec2_mul(vec2 a, vec2 b, vec2 d)
   PLAY_CGLM_INLINE void  vec2_scale(vec2 v, float s, vec2 dest)
   PLAY_CGLM_INLINE void  vec2_scale_as(vec2 v, float s, vec2 dest)
   PLAY_CGLM_INLINE void  vec2_div(vec2 a, vec2 b, vec2 dest)
   PLAY_CGLM_INLINE void  vec2_divs(vec2 v, float s, vec2 dest)
   PLAY_CGLM_INLINE void  vec2_addadd(vec2 a, vec2 b, vec2 dest)
   PLAY_CGLM_INLINE void  vec2_subadd(vec2 a, vec2 b, vec2 dest)
   PLAY_CGLM_INLINE void  vec2_muladd(vec2 a, vec2 b, vec2 dest)
   PLAY_CGLM_INLINE void  vec2_muladds(vec2 a, float s, vec2 dest)
   PLAY_CGLM_INLINE void  vec2_maxadd(vec2 a, vec2 b, vec2 dest)
   PLAY_CGLM_INLINE void  vec2_minadd(vec2 a, vec2 b, vec2 dest)
   PLAY_CGLM_INLINE void  vec2_subsub(vec2 a, vec2 b, vec2 dest)
   PLAY_CGLM_INLINE void  vec2_addsub(vec2 a, vec2 b, vec2 dest)
   PLAY_CGLM_INLINE void  vec2_mulsub(vec2 a, vec2 b, vec2 dest)
   PLAY_CGLM_INLINE void  vec2_mulsubs(vec2 a, float s, vec2 dest)
   PLAY_CGLM_INLINE void  vec2_maxsub(vec2 a, vec2 b, vec2 dest)
   PLAY_CGLM_INLINE void  vec2_minsub(vec2 a, vec2 b, vec2 dest)
   PLAY_CGLM_INLINE void  vec2_negate_to(vec2 v, vec2 dest)
   PLAY_CGLM_INLINE void  vec2_negate(vec2 v)
   PLAY_CGLM_INLINE void  vec2_normalize(vec2 v)
   PLAY_CGLM_INLINE void  vec2_normalize_to(vec2 vec, vec2 dest)
   PLAY_CGLM_INLINE void  vec2_rotate(vec2 v, float angle, vec2 dest)
   PLAY_CGLM_INLINE void  vec2_center(vec2 a, vec2 b, vec2 dest)
   PLAY_CGLM_INLINE float vec2_distance2(vec2 a, vec2 b)
   PLAY_CGLM_INLINE float vec2_distance(vec2 a, vec2 b)
   PLAY_CGLM_INLINE void  vec2_maxv(vec2 v1, vec2 v2, vec2 dest)
   PLAY_CGLM_INLINE void  vec2_minv(vec2 v1, vec2 v2, vec2 dest)
   PLAY_CGLM_INLINE void  vec2_clamp(vec2 v, float minVal, float maxVal)
   PLAY_CGLM_INLINE void  vec2_swizzle(vec2 v, int mask, vec2 dest)
   PLAY_CGLM_INLINE void  vec2_lerp(vec2 from, vec2 to, float t, vec2 dest)
   PLAY_CGLM_INLINE void  vec2_step(vec2 edge, vec2 x, vec2 dest)
   PLAY_CGLM_INLINE void  vec2_make(float * restrict src, vec2 dest)
   PLAY_CGLM_INLINE void  vec2_reflect(vec2 v, vec2 n, vec2 dest)
   PLAY_CGLM_INLINE void  vec2_refract(vec2 v, vec2 n, float eta, vec2 dest)
 */

#ifndef cvec2_h
#define cvec2_h


/*** Start of inlined file: util.h ***/
/*
 Functions:
   PLAY_CGLM_INLINE int   sign(int val);
   PLAY_CGLM_INLINE float signf(float val);
   PLAY_CGLM_INLINE float rad(float deg);
   PLAY_CGLM_INLINE float deg(float rad);
   PLAY_CGLM_INLINE void  make_rad(float *deg);
   PLAY_CGLM_INLINE void  make_deg(float *rad);
   PLAY_CGLM_INLINE float pow2(float x);
   PLAY_CGLM_INLINE float fmin(float a, float b);
   PLAY_CGLM_INLINE float fmax(float a, float b);
   PLAY_CGLM_INLINE float clamp(float val, float minVal, float maxVal);
   PLAY_CGLM_INLINE float clamp_zo(float val, float minVal, float maxVal);
   PLAY_CGLM_INLINE float lerp(float from, float to, float t);
   PLAY_CGLM_INLINE float lerpc(float from, float to, float t);
   PLAY_CGLM_INLINE float step(float edge, float x);
   PLAY_CGLM_INLINE float smooth(float t);
   PLAY_CGLM_INLINE float smoothstep(float edge0, float edge1, float x);
   PLAY_CGLM_INLINE float smoothinterp(float from, float to, float t);
   PLAY_CGLM_INLINE float smoothinterpc(float from, float to, float t);
   PLAY_CGLM_INLINE bool  eq(float a, float b);
   PLAY_CGLM_INLINE float percent(float from, float to, float current);
   PLAY_CGLM_INLINE float percentc(float from, float to, float current);
 */




#define PLAY_CGLM_MIN(X, Y) (((X) < (Y)) ? (X) : (Y))
#define PLAY_CGLM_MAX(X, Y) (((X) > (Y)) ? (X) : (Y))

/*!
 * @brief get sign of 32 bit integer as +1, -1, 0
 *
 * Important: It returns 0 for zero input
 *
 * @param val integer value
 */
PLAY_CGLM_INLINE
int
sign(int val)
{
    return ((val >> 31) - (-val >> 31));
}

/*!
 * @brief get sign of 32 bit float as +1, -1, 0
 *
 * Important: It returns 0 for zero/NaN input
 *
 * @param val float value
 */
PLAY_CGLM_INLINE
float
signf(float val)
{
    return (float)((val > 0.0f) - (val < 0.0f));
}

/*!
 * @brief convert degree to radians
 *
 * @param[in] deg angle in degrees
 */
PLAY_CGLM_INLINE
float
rad(float deg)
{
    return deg * PLAY_CGLM_PIf / 180.0f;
}

/*!
 * @brief convert radians to degree
 *
 * @param[in] rad angle in radians
 */
PLAY_CGLM_INLINE
float
deg(float rad)
{
    return rad * 180.0f / PLAY_CGLM_PIf;
}

/*!
 * @brief convert existing degree to radians. this will override degrees value
 *
 * @param[in, out] deg pointer to angle in degrees
 */
PLAY_CGLM_INLINE
void
make_rad(float *deg)
{
    *deg = *deg * PLAY_CGLM_PIf / 180.0f;
}

/*!
 * @brief convert existing radians to degree. this will override radians value
 *
 * @param[in, out] rad pointer to angle in radians
 */
PLAY_CGLM_INLINE
void
make_deg(float *rad)
{
    *rad = *rad * 180.0f / PLAY_CGLM_PIf;
}

/*!
 * @brief multiplies given parameter with itself = x * x or powf(x, 2)
 *
 * @param[in] x x
 */
PLAY_CGLM_INLINE
float
pow2(float x)
{
    return x * x;
}

/*!
 * @brief find minimum of given two values
 *
 * @param[in] a number 1
 * @param[in] b number 2
 */
PLAY_CGLM_INLINE
float
fmin(float a, float b)
{
    if (a < b)
        return a;
    return b;
}

/*!
 * @brief find maximum of given two values
 *
 * @param[in] a number 1
 * @param[in] b number 2
 */
PLAY_CGLM_INLINE
float
fmax(float a, float b)
{
    if (a > b)
        return a;
    return b;
}

/*!
 * @brief find minimum of given two values
 *
 * @param[in] a number 1
 * @param[in] b number 2
 *
 * @return smallest of the two values
 */
PLAY_CGLM_INLINE
int
imin(int a, int b)
{
    if (a < b)
        return a;
    return b;
}

/*!
 * @brief find maximum of given two values
 *
 * @param[in] a number 1
 * @param[in] b number 2
 *
 * @return largest of the two values
 */
PLAY_CGLM_INLINE
int
imax(int a, int b)
{
    if (a > b)
        return a;
    return b;
}

/*!
 * @brief clamp a number between min and max
 *
 * @param[in] val    value to clamp
 * @param[in] minVal minimum value
 * @param[in] maxVal maximum value
 */
PLAY_CGLM_INLINE
float
clamp(float val, float minVal, float maxVal)
{
    return fmin(fmax(val, minVal), maxVal);
}

/*!
 * @brief clamp a number to zero and one
 *
 * @param[in] val value to clamp
 */
PLAY_CGLM_INLINE
float
clamp_zo(float val)
{
    return clamp(val, 0.0f, 1.0f);
}

/*!
 * @brief linear interpolation between two numbers
 *
 * formula:  from + t * (to - from)
 *
 * @param[in]   from from value
 * @param[in]   to   to value
 * @param[in]   t    interpolant (amount)
 */
PLAY_CGLM_INLINE
float
lerp(float from, float to, float t)
{
    return from + t * (to - from);
}

/*!
 * @brief clamped linear interpolation between two numbers
 *
 * formula:  from + t * (to - from)
 *
 * @param[in]   from    from value
 * @param[in]   to      to value
 * @param[in]   t       interpolant (amount) clamped between 0 and 1
 */
PLAY_CGLM_INLINE
float
lerpc(float from, float to, float t)
{
    return lerp(from, to, clamp_zo(t));
}

/*!
 * @brief threshold function
 *
 * @param[in]   edge    threshold
 * @param[in]   x       value to test against threshold
 * @return      returns 0.0 if x < edge, else 1.0
 */
PLAY_CGLM_INLINE
float
step(float edge, float x)
{
    /* branching - no type conversion */
    return (x < edge) ? 0.0f : 1.0f;
    /*
     * An alternative implementation without branching
     * but with type conversion could be:
     * return !(x < edge);
     */
}

/*!
 * @brief smooth Hermite interpolation
 *
 * formula:  t^2 * (3-2t)
 *
 * @param[in]   t    interpolant (amount)
 */
PLAY_CGLM_INLINE
float
smooth(float t)
{
    return t * t * (3.0f - 2.0f * t);
}

/*!
 * @brief threshold function with a smooth transition (according to OpenCL specs)
 *
 * formula:  t^2 * (3-2t)
 *
 * @param[in]   edge0 low threshold
 * @param[in]   edge1 high threshold
 * @param[in]   x     interpolant (amount)
 */
PLAY_CGLM_INLINE
float
smoothstep(float edge0, float edge1, float x)
{
    float t;
    t = clamp_zo((x - edge0) / (edge1 - edge0));
    return smooth(t);
}

/*!
 * @brief smoothstep interpolation between two numbers
 *
 * formula:  from + smoothstep(t) * (to - from)
 *
 * @param[in]   from from value
 * @param[in]   to   to value
 * @param[in]   t    interpolant (amount)
 */
PLAY_CGLM_INLINE
float
smoothinterp(float from, float to, float t)
{
    return from + smooth(t) * (to - from);
}

/*!
 * @brief clamped smoothstep interpolation between two numbers
 *
 * formula:  from + smoothstep(t) * (to - from)
 *
 * @param[in]   from from value
 * @param[in]   to   to value
 * @param[in]   t    interpolant (amount) clamped between 0 and 1
 */
PLAY_CGLM_INLINE
float
smoothinterpc(float from, float to, float t)
{
    return smoothinterp(from, to, clamp_zo(t));
}

/*!
 * @brief check if two float equal with using EPSILON
 *
 * @param[in]   a   a
 * @param[in]   b   b
 */
PLAY_CGLM_INLINE
bool
eq(float a, float b)
{
    return fabsf(a - b) <= PLAY_CGLM_FLT_EPSILON;
}

/*!
 * @brief percentage of current value between start and end value
 *
 * maybe fraction could be alternative name.
 *
 * @param[in]   from    from value
 * @param[in]   to      to value
 * @param[in]   current current value
 */
PLAY_CGLM_INLINE
float
percent(float from, float to, float current)
{
    float t;

    if ((t = to - from) == 0.0f)
        return 1.0f;

    return (current - from) / t;
}

/*!
 * @brief clamped percentage of current value between start and end value
 *
 * @param[in]   from    from value
 * @param[in]   to      to value
 * @param[in]   current current value
 */
PLAY_CGLM_INLINE
float
percentc(float from, float to, float current)
{
    return clamp_zo(percent(from, to, current));
}

/*!
* @brief swap two float values
*
* @param[in]   a float value 1 (pointer)
* @param[in]   b float value 2 (pointer)
*/
PLAY_CGLM_INLINE
void
swapf(float * __restrict a, float * __restrict b)
{
    float t;
    t  = *a;
    *a = *b;
    *b = t;
}



/*** End of inlined file: util.h ***/


/*** Start of inlined file: vec2-ext.h ***/
/*
 Functions:
   PLAY_CGLM_INLINE void  vec2_fill(vec2 v, float val)
   PLAY_CGLM_INLINE bool  vec2_eq(vec2 v, float val);
   PLAY_CGLM_INLINE bool  vec2_eq_eps(vec2 v, float val);
   PLAY_CGLM_INLINE bool  vec2_eq_all(vec2 v);
   PLAY_CGLM_INLINE bool  vec2_eqv(vec2 a, vec2 b);
   PLAY_CGLM_INLINE bool  vec2_eqv_eps(vec2 a, vec2 b);
   PLAY_CGLM_INLINE float vec2_max(vec2 v);
   PLAY_CGLM_INLINE float vec2_min(vec2 v);
   PLAY_CGLM_INLINE bool  vec2_isnan(vec2 v);
   PLAY_CGLM_INLINE bool  vec2_isinf(vec2 v);
   PLAY_CGLM_INLINE bool  vec2_isvalid(vec2 v);
   PLAY_CGLM_INLINE void  vec2_sign(vec2 v, vec2 dest);
   PLAY_CGLM_INLINE void  vec2_abs(vec2 v, vec2 dest);
   PLAY_CGLM_INLINE void  vec2_fract(vec2 v, vec2 dest);
   PLAY_CGLM_INLINE void  vec2_floor(vec2 v, vec2 dest);
   PLAY_CGLM_INLINE float vec2_mods(vec2 v, float s, vec2 dest);
   PLAY_CGLM_INLINE float vec2_steps(float edge, vec2 v, vec2 dest);
   PLAY_CGLM_INLINE void  vec2_stepr(vec2 edge, float v, vec2 dest);
   PLAY_CGLM_INLINE void  vec2_sqrt(vec2 v, vec2 dest);
   PLAY_CGLM_INLINE void  vec2_complex_mul(vec2 a, vec2 b, vec2 dest)
   PLAY_CGLM_INLINE void  vec2_complex_div(vec2 a, vec2 b, vec2 dest)
   PLAY_CGLM_INLINE void  vec2_complex_conjugate(vec2 a, vec2 dest)
 */

#ifndef cvec2_ext_h
#define cvec2_ext_h

/*!
 * @brief fill a vector with specified value
 *
 * @param[out] v   dest
 * @param[in]  val value
 */
PLAY_CGLM_INLINE
void
vec2_fill(vec2 v, float val)
{
    v[0] = v[1] = val;
}

/*!
 * @brief check if vector is equal to value (without epsilon)
 *
 * @param[in] v   vector
 * @param[in] val value
 */
PLAY_CGLM_INLINE
bool
vec2_eq(vec2 v, float val)
{
    return v[0] == val && v[0] == v[1];
}

/*!
 * @brief check if vector is equal to value (with epsilon)
 *
 * @param[in] v   vector
 * @param[in] val value
 */
PLAY_CGLM_INLINE
bool
vec2_eq_eps(vec2 v, float val)
{
    return fabsf(v[0] - val) <= PLAY_CGLM_FLT_EPSILON
           && fabsf(v[1] - val) <= PLAY_CGLM_FLT_EPSILON;
}

/*!
 * @brief check if vector members are equal (without epsilon)
 *
 * @param[in] v   vector
 */
PLAY_CGLM_INLINE
bool
vec2_eq_all(vec2 v)
{
    return vec2_eq_eps(v, v[0]);
}

/*!
 * @brief check if vector is equal to another (without epsilon)
 *
 * @param[in] a vector
 * @param[in] b vector
 */
PLAY_CGLM_INLINE
bool
vec2_eqv(vec2 a, vec2 b)
{
    return a[0] == b[0] && a[1] == b[1];
}

/*!
 * @brief check if vector is equal to another (with epsilon)
 *
 * @param[in] a vector
 * @param[in] b vector
 */
PLAY_CGLM_INLINE
bool
vec2_eqv_eps(vec2 a, vec2 b)
{
    return fabsf(a[0] - b[0]) <= PLAY_CGLM_FLT_EPSILON
           && fabsf(a[1] - b[1]) <= PLAY_CGLM_FLT_EPSILON;
}

/*!
 * @brief max value of vector
 *
 * @param[in] v vector
 */
PLAY_CGLM_INLINE
float
vec2_max(vec2 v)
{
    return fmax(v[0], v[1]);
}

/*!
 * @brief min value of vector
 *
 * @param[in] v vector
 */
PLAY_CGLM_INLINE
float
vec2_min(vec2 v)
{
    return fmin(v[0], v[1]);
}

/*!
 * @brief check if one of items is NaN (not a number)
 *        you should only use this in DEBUG mode or very critical asserts
 *
 * @param[in] v vector
 */
PLAY_CGLM_INLINE
bool
vec2_isnan(vec2 v)
{
#ifndef PLAY_CGLM_FAST_MATH
    return isnan(v[0]) || isnan(v[1]);
#else
    return false;
#endif
}

/*!
 * @brief check if one of items is INFINITY
 *        you should only use this in DEBUG mode or very critical asserts
 *
 * @param[in] v vector
 */
PLAY_CGLM_INLINE
bool
vec2_isinf(vec2 v)
{
#ifndef PLAY_CGLM_FAST_MATH
    return isinf(v[0]) || isinf(v[1]);
#else
    return false;
#endif
}

/*!
 * @brief check if all items are valid number
 *        you should only use this in DEBUG mode or very critical asserts
 *
 * @param[in] v vector
 */
PLAY_CGLM_INLINE
bool
vec2_isvalid(vec2 v)
{
    return !vec2_isnan(v) && !vec2_isinf(v);
}

/*!
 * @brief get sign of 32 bit float as +1, -1, 0
 *
 * Important: It returns 0 for zero/NaN input
 *
 * @param v vector
 */
PLAY_CGLM_INLINE
void
vec2_sign(vec2 v, vec2 dest)
{
    dest[0] = signf(v[0]);
    dest[1] = signf(v[1]);
}

/*!
 * @brief absolute value of v
 *
 * @param[in]	v	vector
 * @param[out]	dest	destination
 */
PLAY_CGLM_INLINE
void
vec2_abs(vec2 v, vec2 dest)
{
    dest[0] = fabsf(v[0]);
    dest[1] = fabsf(v[1]);
}

/*!
 * @brief fractional part of each vector item
 *
 * @param[in]  v    vector
 * @param[out] dest destination vector
 */
PLAY_CGLM_INLINE
void
vec2_fract(vec2 v, vec2 dest)
{
    dest[0] = fminf(v[0] - floorf(v[0]), 0.999999940395355224609375f);
    dest[1] = fminf(v[1] - floorf(v[1]), 0.999999940395355224609375f);
}

/*!
 * @brief floor of each vector item
 *
 * @param[in]  v    vector
 * @param[out] dest destination vector
 */
PLAY_CGLM_INLINE
void
vec2_floor(vec2 v, vec2 dest)
{
    dest[0] = floorf(v[0]);
    dest[1] = floorf(v[1]);
}

/*!
 * @brief mod of each vector item, result is written to dest (dest = v % s)
 *
 * @param[in]  v    vector
 * @param[in]  s    scalar
 * @param[out] dest destination vector
 */
PLAY_CGLM_INLINE
void
vec2_mods(vec2 v, float s, vec2 dest)
{
    dest[0] = fmodf(v[0], s);
    dest[1] = fmodf(v[1], s);
}

/*!
 * @brief square root of each vector item
 *
 * @param[in]  v    vector
 * @param[out] dest destination vector
 */
PLAY_CGLM_INLINE
void
vec2_sqrt(vec2 v, vec2 dest)
{
    dest[0] = sqrtf(v[0]);
    dest[1] = sqrtf(v[1]);
}

/*!
 * @brief treat vectors as complex numbers and multiply them as such.
 *
 * @param[in]  a    left number
 * @param[in]  b    right number
 * @param[out] dest destination number
 */
PLAY_CGLM_INLINE
void
vec2_complex_mul(vec2 a, vec2 b, vec2 dest)
{
    float tr, ti;
    tr = a[0] * b[0] - a[1] * b[1];
    ti = a[0] * b[1] + a[1] * b[0];
    dest[0] = tr;
    dest[1] = ti;
}

/*!
 * @brief threshold each vector item with scalar
 *        condition is: (x[i] < edge) ? 0.0 : 1.0
 *
 * @param[in]   edge    threshold
 * @param[in]   x       vector to test against threshold
 * @param[out]  dest    destination
 */
PLAY_CGLM_INLINE
void
vec2_steps(float edge, vec2 x, vec2 dest)
{
    dest[0] = step(edge, x[0]);
    dest[1] = step(edge, x[1]);
}

/*!
 * @brief threshold a value with *vector* as the threshold
 *        condition is: (x < edge[i]) ? 0.0 : 1.0
 *
 * @param[in]   edge    threshold vector
 * @param[in]   x       value to test against threshold
 * @param[out]  dest    destination
 */
PLAY_CGLM_INLINE
void
vec2_stepr(vec2 edge, float x, vec2 dest)
{
    dest[0] = step(edge[0], x);
    dest[1] = step(edge[1], x);
}

/*!
 * @brief treat vectors as complex numbers and divide them as such.
 *
 * @param[in]  a    left number (numerator)
 * @param[in]  b    right number (denominator)
 * @param[out] dest destination number
 */
PLAY_CGLM_INLINE
void
vec2_complex_div(vec2 a, vec2 b, vec2 dest)
{
    float tr, ti;
    float const ibnorm2 = 1.0f / (b[0] * b[0] + b[1] * b[1]);
    tr = ibnorm2 * (a[0] * b[0] + a[1] * b[1]);
    ti = ibnorm2 * (a[1] * b[0] - a[0] * b[1]);
    dest[0] = tr;
    dest[1] = ti;
}

/*!
 * @brief treat the vector as a complex number and conjugate it as such.
 *
 * @param[in]  a    the number
 * @param[out] dest destination number
 */
PLAY_CGLM_INLINE
void
vec2_complex_conjugate(vec2 a, vec2 dest)
{
    dest[0] =  a[0];
    dest[1] = -a[1];
}

#endif /* cvec2_ext_h */

/*** End of inlined file: vec2-ext.h ***/

#define PLAY_CGLM_VEC2_ONE_INIT   {1.0f, 1.0f}
#define PLAY_CGLM_VEC2_ZERO_INIT  {0.0f, 0.0f}

#define PLAY_CGLM_VEC2_ONE  ((vec2)PLAY_CGLM_VEC2_ONE_INIT)
#define PLAY_CGLM_VEC2_ZERO ((vec2)PLAY_CGLM_VEC2_ZERO_INIT)

/*!
 * @brief init vec2 using another vector
 *
 * @param[in]  v    a vector
 * @param[out] dest destination
 */
PLAY_CGLM_INLINE
void
vec2_new(float * __restrict v, vec2 dest)
{
    dest[0] = v[0];
    dest[1] = v[1];
}

/*!
 * @brief copy all members of [a] to [dest]
 *
 * @param[in]  a    source
 * @param[out] dest destination
 */
PLAY_CGLM_INLINE
void
vec2_copy(vec2 a, vec2 dest)
{
    dest[0] = a[0];
    dest[1] = a[1];
}

/*!
 * @brief make vector zero
 *
 * @param[in, out]  v vector
 */
PLAY_CGLM_INLINE
void
vec2_zero(vec2 v)
{
    v[0] = v[1] = 0.0f;
}

/*!
 * @brief make vector one
 *
 * @param[in, out]  v vector
 */
PLAY_CGLM_INLINE
void
vec2_one(vec2 v)
{
    v[0] = v[1] = 1.0f;
}

/*!
 * @brief vec2 dot product
 *
 * @param[in] a vector1
 * @param[in] b vector2
 *
 * @return dot product
 */
PLAY_CGLM_INLINE
float
vec2_dot(vec2 a, vec2 b)
{
    return a[0] * b[0] + a[1] * b[1];
}

/*!
 * @brief vec2 cross product
 *
 * REF: http://allenchou.net/2013/07/cross-product-of-2d-vectors/
 *
 * @param[in]  a vector1
 * @param[in]  b vector2
 *
 * @return Z component of cross product
 */
PLAY_CGLM_INLINE
float
vec2_cross(vec2 a, vec2 b)
{
    /* just calculate the z-component */
    return a[0] * b[1] - a[1] * b[0];
}

/*!
 * @brief norm * norm (magnitude) of vec
 *
 * we can use this func instead of calling norm * norm, because it would call
 * sqrtf function twice but with this func we can avoid func call, maybe this is
 * not good name for this func
 *
 * @param[in] v vector
 *
 * @return norm * norm
 */
PLAY_CGLM_INLINE
float
vec2_norm2(vec2 v)
{
    return vec2_dot(v, v);
}

/*!
 * @brief norm (magnitude) of vec2
 *
 * @param[in] vec vector
 *
 * @return norm
 */
PLAY_CGLM_INLINE
float
vec2_norm(vec2 vec)
{
    return sqrtf(vec2_norm2(vec));
}

/*!
 * @brief add a vector to b vector store result in dest
 *
 * @param[in]  a    vector1
 * @param[in]  b    vector2
 * @param[out] dest destination vector
 */
PLAY_CGLM_INLINE
void
vec2_add(vec2 a, vec2 b, vec2 dest)
{
    dest[0] = a[0] + b[0];
    dest[1] = a[1] + b[1];
}

/*!
 * @brief add scalar to v vector store result in dest (d = v + s)
 *
 * @param[in]  v    vector
 * @param[in]  s    scalar
 * @param[out] dest destination vector
 */
PLAY_CGLM_INLINE
void
vec2_adds(vec2 v, float s, vec2 dest)
{
    dest[0] = v[0] + s;
    dest[1] = v[1] + s;
}

/*!
 * @brief subtract b vector from a vector store result in dest
 *
 * @param[in]  a    vector1
 * @param[in]  b    vector2
 * @param[out] dest destination vector
 */
PLAY_CGLM_INLINE
void
vec2_sub(vec2 a, vec2 b, vec2 dest)
{
    dest[0] = a[0] - b[0];
    dest[1] = a[1] - b[1];
}

/*!
 * @brief subtract scalar from v vector store result in dest (d = v - s)
 *
 * @param[in]  v    vector
 * @param[in]  s    scalar
 * @param[out] dest destination vector
 */
PLAY_CGLM_INLINE
void
vec2_subs(vec2 v, float s, vec2 dest)
{
    dest[0] = v[0] - s;
    dest[1] = v[1] - s;
}

/*!
 * @brief multiply two vectors (component-wise multiplication)
 *
 * @param a    v1
 * @param b    v2
 * @param dest v3 = (a[0] * b[0], a[1] * b[1])
 */
PLAY_CGLM_INLINE
void
vec2_mul(vec2 a, vec2 b, vec2 dest)
{
    dest[0] = a[0] * b[0];
    dest[1] = a[1] * b[1];
}

/*!
 * @brief multiply/scale vector with scalar: result = v * s
 *
 * @param[in]  v    vector
 * @param[in]  s    scalar
 * @param[out] dest destination vector
 */
PLAY_CGLM_INLINE
void
vec2_scale(vec2 v, float s, vec2 dest)
{
    dest[0] = v[0] * s;
    dest[1] = v[1] * s;
}

/*!
 * @brief scale as vector specified: result = unit(v) * s
 *
 * @param[in]  v    vector
 * @param[in]  s    scalar
 * @param[out] dest destination vector
 */
PLAY_CGLM_INLINE
void
vec2_scale_as(vec2 v, float s, vec2 dest)
{
    float norm;
    norm = vec2_norm(v);

    if (PLAY_CGLM_UNLIKELY(norm < FLT_EPSILON))
    {
        vec2_zero(dest);
        return;
    }

    vec2_scale(v, s / norm, dest);
}

/*!
 * @brief div vector with another component-wise division: d = a / b
 *
 * @param[in]  a    vector 1
 * @param[in]  b    vector 2
 * @param[out] dest result = (a[0]/b[0], a[1]/b[1])
 */
PLAY_CGLM_INLINE
void
vec2_div(vec2 a, vec2 b, vec2 dest)
{
    dest[0] = a[0] / b[0];
    dest[1] = a[1] / b[1];
}

/*!
 * @brief div vector with scalar: d = v / s
 *
 * @param[in]  v    vector
 * @param[in]  s    scalar
 * @param[out] dest result = (a[0]/s, a[1]/s)
 */
PLAY_CGLM_INLINE
void
vec2_divs(vec2 v, float s, vec2 dest)
{
    dest[0] = v[0] / s;
    dest[1] = v[1] / s;
}

/*!
 * @brief add two vectors and add result to sum
 *
 * it applies += operator so dest must be initialized
 *
 * @param[in]  a    vector 1
 * @param[in]  b    vector 2
 * @param[out] dest dest += (a + b)
 */
PLAY_CGLM_INLINE
void
vec2_addadd(vec2 a, vec2 b, vec2 dest)
{
    dest[0] += a[0] + b[0];
    dest[1] += a[1] + b[1];
}

/*!
 * @brief sub two vectors and add result to dest
 *
 * it applies += operator so dest must be initialized
 *
 * @param[in]  a    vector 1
 * @param[in]  b    vector 2
 * @param[out] dest dest += (a + b)
 */
PLAY_CGLM_INLINE
void
vec2_subadd(vec2 a, vec2 b, vec2 dest)
{
    dest[0] += a[0] - b[0];
    dest[1] += a[1] - b[1];
}

/*!
 * @brief mul two vectors and add result to dest
 *
 * it applies += operator so dest must be initialized
 *
 * @param[in]  a    vector 1
 * @param[in]  b    vector 2
 * @param[out] dest dest += (a * b)
 */
PLAY_CGLM_INLINE
void
vec2_muladd(vec2 a, vec2 b, vec2 dest)
{
    dest[0] += a[0] * b[0];
    dest[1] += a[1] * b[1];
}

/*!
 * @brief mul vector with scalar and add result to sum
 *
 * it applies += operator so dest must be initialized
 *
 * @param[in]  a    vector
 * @param[in]  s    scalar
 * @param[out] dest dest += (a * b)
 */
PLAY_CGLM_INLINE
void
vec2_muladds(vec2 a, float s, vec2 dest)
{
    dest[0] += a[0] * s;
    dest[1] += a[1] * s;
}

/*!
 * @brief add max of two vectors to result/dest
 *
 * it applies += operator so dest must be initialized
 *
 * @param[in]  a    vector 1
 * @param[in]  b    vector 2
 * @param[out] dest dest += fmax(a, b)
 */
PLAY_CGLM_INLINE
void
vec2_maxadd(vec2 a, vec2 b, vec2 dest)
{
    dest[0] += fmax(a[0], b[0]);
    dest[1] += fmax(a[1], b[1]);
}

/*!
 * @brief add min of two vectors to result/dest
 *
 * it applies += operator so dest must be initialized
 *
 * @param[in]  a    vector 1
 * @param[in]  b    vector 2
 * @param[out] dest dest += fmin(a, b)
 */
PLAY_CGLM_INLINE
void
vec2_minadd(vec2 a, vec2 b, vec2 dest)
{
    dest[0] += fmin(a[0], b[0]);
    dest[1] += fmin(a[1], b[1]);
}

/*!
 * @brief sub two vectors and sub result to dest
 *
 * it applies -= operator so dest must be initialized
 *
 * @param[in]  a    vector 1
 * @param[in]  b    vector 2
 * @param[out] dest dest -= (a - b)
 */
PLAY_CGLM_INLINE
void
vec2_subsub(vec2 a, vec2 b, vec2 dest)
{
    dest[0] -= a[0] - b[0];
    dest[1] -= a[1] - b[1];
}

/*!
 * @brief add two vectors and sub result to dest
 *
 * it applies -= operator so dest must be initialized
 *
 * @param[in]  a    vector 1
 * @param[in]  b    vector 2
 * @param[out] dest dest -= (a + b)
 */
PLAY_CGLM_INLINE
void
vec2_addsub(vec2 a, vec2 b, vec2 dest)
{
    dest[0] -= a[0] + b[0];
    dest[1] -= a[1] + b[1];
}

/*!
 * @brief mul two vectors and sub result to dest
 *
 * it applies -= operator so dest must be initialized
 *
 * @param[in]  a    vector 1
 * @param[in]  b    vector 2
 * @param[out] dest dest -= (a * b)
 */
PLAY_CGLM_INLINE
void
vec2_mulsub(vec2 a, vec2 b, vec2 dest)
{
    dest[0] -= a[0] * b[0];
    dest[1] -= a[1] * b[1];
}

/*!
 * @brief mul vector with scalar and sub result to sum
 *
 * it applies -= operator so dest must be initialized
 *
 * @param[in]  a    vector
 * @param[in]  s    scalar
 * @param[out] dest dest -= (a * b)
 */
PLAY_CGLM_INLINE
void
vec2_mulsubs(vec2 a, float s, vec2 dest)
{
    dest[0] -= a[0] * s;
    dest[1] -= a[1] * s;
}

/*!
 * @brief sub max of two vectors to result/dest
 *
 * it applies -= operator so dest must be initialized
 *
 * @param[in]  a    vector 1
 * @param[in]  b    vector 2
 * @param[out] dest dest -= fmax(a, b)
 */
PLAY_CGLM_INLINE
void
vec2_maxsub(vec2 a, vec2 b, vec2 dest)
{
    dest[0] -= fmax(a[0], b[0]);
    dest[1] -= fmax(a[1], b[1]);
}

/*!
 * @brief sub min of two vectors to result/dest
 *
 * it applies -= operator so dest must be initialized
 *
 * @param[in]  a    vector 1
 * @param[in]  b    vector 2
 * @param[out] dest dest -= fmin(a, b)
 */
PLAY_CGLM_INLINE
void
vec2_minsub(vec2 a, vec2 b, vec2 dest)
{
    dest[0] -= fmin(a[0], b[0]);
    dest[1] -= fmin(a[1], b[1]);
}

/*!
 * @brief negate vector components and store result in dest
 *
 * @param[in]   v     vector
 * @param[out]  dest  result vector
 */
PLAY_CGLM_INLINE
void
vec2_negate_to(vec2 v, vec2 dest)
{
    dest[0] = -v[0];
    dest[1] = -v[1];
}

/*!
 * @brief negate vector components
 *
 * @param[in, out]  v  vector
 */
PLAY_CGLM_INLINE
void
vec2_negate(vec2 v)
{
    vec2_negate_to(v, v);
}

/*!
 * @brief normalize vector and store result in same vec
 *
 * @param[in, out] v vector
 */
PLAY_CGLM_INLINE
void
vec2_normalize(vec2 v)
{
    float norm;

    norm = vec2_norm(v);

    if (PLAY_CGLM_UNLIKELY(norm < FLT_EPSILON))
    {
        v[0] = v[1] = 0.0f;
        return;
    }

    vec2_scale(v, 1.0f / norm, v);
}

/*!
 * @brief normalize vector to dest
 *
 * @param[in]  v    source
 * @param[out] dest destination
 */
PLAY_CGLM_INLINE
void
vec2_normalize_to(vec2 v, vec2 dest)
{
    float norm;

    norm = vec2_norm(v);

    if (PLAY_CGLM_UNLIKELY(norm < FLT_EPSILON))
    {
        vec2_zero(dest);
        return;
    }

    vec2_scale(v, 1.0f / norm, dest);
}

/*!
 * @brief rotate vec2 around origin by angle (CCW: counterclockwise)
 *
 * Formula:
 *   𝑥2 = cos(a)𝑥1 − sin(a)𝑦1
 *   𝑦2 = sin(a)𝑥1 + cos(a)𝑦1
 *
 * @param[in]  v     vector to rotate
 * @param[in]  angle angle by radians
 * @param[out] dest  destination vector
 */
PLAY_CGLM_INLINE
void
vec2_rotate(vec2 v, float angle, vec2 dest)
{
    float c, s, x1, y1;

    c  = cosf(angle);
    s  = sinf(angle);

    x1 = v[0];
    y1 = v[1];

    dest[0] = c * x1 - s * y1;
    dest[1] = s * x1 + c * y1;
}

/**
 * @brief find center point of two vector
 *
 * @param[in]  a    vector1
 * @param[in]  b    vector2
 * @param[out] dest center point
 */
PLAY_CGLM_INLINE
void
vec2_center(vec2 a, vec2 b, vec2 dest)
{
    vec2_add(a, b, dest);
    vec2_scale(dest, 0.5f, dest);
}

/**
 * @brief squared distance between two vectors
 *
 * @param[in] a vector1
 * @param[in] b vector2
 * @return returns squared distance (distance * distance)
 */
PLAY_CGLM_INLINE
float
vec2_distance2(vec2 a, vec2 b)
{
    return pow2(b[0] - a[0]) + pow2(b[1] - a[1]);
}

/**
 * @brief distance between two vectors
 *
 * @param[in] a vector1
 * @param[in] b vector2
 * @return returns distance
 */
PLAY_CGLM_INLINE
float
vec2_distance(vec2 a, vec2 b)
{
    return sqrtf(vec2_distance2(a, b));
}

/*!
 * @brief max values of vectors
 *
 * @param[in]  a    vector1
 * @param[in]  b    vector2
 * @param[out] dest destination
 */
PLAY_CGLM_INLINE
void
vec2_maxv(vec2 a, vec2 b, vec2 dest)
{
    dest[0] = fmax(a[0], b[0]);
    dest[1] = fmax(a[1], b[1]);
}

/*!
 * @brief min values of vectors
 *
 * @param[in]  a    vector1
 * @param[in]  b    vector2
 * @param[out] dest destination
 */
PLAY_CGLM_INLINE
void
vec2_minv(vec2 a, vec2 b, vec2 dest)
{
    dest[0] = fmin(a[0], b[0]);
    dest[1] = fmin(a[1], b[1]);
}

/*!
 * @brief clamp vector's individual members between min and max values
 *
 * @param[in, out]  v      vector
 * @param[in]       minval minimum value
 * @param[in]       maxval maximum value
 */
PLAY_CGLM_INLINE
void
vec2_clamp(vec2 v, float minval, float maxval)
{
    v[0] = clamp(v[0], minval, maxval);
    v[1] = clamp(v[1], minval, maxval);
}

/*!
 * @brief swizzle vector components
 *
 * @param[in]  v    source
 * @param[in]  mask mask
 * @param[out] dest destination
 */
PLAY_CGLM_INLINE
void
vec2_swizzle(vec2 v, int mask, vec2 dest)
{
    vec2 t;

    t[0] = v[(mask & (3 << 0))];
    t[1] = v[(mask & (3 << 2)) >> 2];

    vec2_copy(t, dest);
}

/*!
 * @brief linear interpolation between two vector
 *
 * formula:  from + s * (to - from)
 *
 * @param[in]   from from value
 * @param[in]   to   to value
 * @param[in]   t    interpolant (amount) clamped between 0 and 1
 * @param[out]  dest destination
 */
PLAY_CGLM_INLINE
void
vec2_lerp(vec2 from, vec2 to, float t, vec2 dest)
{
    vec2 s, v;

    /* from + s * (to - from) */
    vec2_fill(s, clamp_zo(t));
    vec2_sub(to, from, v);
    vec2_mul(s, v, v);
    vec2_add(from, v, dest);
}

/*!
 * @brief threshold function
 *
 * @param[in]   edge    threshold
 * @param[in]   x       value to test against threshold
 * @param[out]  dest    destination
 */
PLAY_CGLM_INLINE
void
vec2_step(vec2 edge, vec2 x, vec2 dest)
{
    dest[0] = step(edge[0], x[0]);
    dest[1] = step(edge[1], x[1]);
}

/*!
 * @brief Create two dimensional vector from pointer
 *
 * @param[in]  src  pointer to an array of floats
 * @param[out] dest destination vector
 */
PLAY_CGLM_INLINE
void
vec2_make(const float * __restrict src, vec2 dest)
{
    dest[0] = src[0];
    dest[1] = src[1];
}

/*!
 * @brief reflection vector using an incident ray and a surface normal
 *
 * @param[in]  v    incident vector
 * @param[in]  n    normalized normal vector
 * @param[out] dest destination vector for the reflection result
 */
PLAY_CGLM_INLINE
void
vec2_reflect(vec2 v, vec2 n, vec2 dest)
{
    vec2 temp;
    vec2_scale(n, 2.0f * vec2_dot(v, n), temp);
    vec2_sub(v, temp, dest);
}

/*!
 * @brief computes refraction vector for an incident vector and a surface normal.
 *
 * calculates the refraction vector based on Snell's law. If total internal reflection
 * occurs (angle too great given eta), dest is set to zero and returns false.
 * Otherwise, computes refraction vector, stores it in dest, and returns true.
 *
 * @param[in]  v    normalized incident vector
 * @param[in]  n    normalized normal vector
 * @param[in]  eta  ratio of indices of refraction (incident/transmitted)
 * @param[out] dest refraction vector if refraction occurs; zero vector otherwise
 *
 * @returns true if refraction occurs; false if total internal reflection occurs.
 */
PLAY_CGLM_INLINE
bool
vec2_refract(vec2 v, vec2 n, float eta, vec2 dest)
{
    float ndi, eni, k;

    ndi = vec2_dot(n, v);
    eni = eta * ndi;
    k   = 1.0f - eta * eta + eni * eni;

    if (k < 0.0f)
    {
        vec2_zero(dest);
        return false;
    }

    vec2_scale(v, eta, dest);
    vec2_mulsubs(n, eni + sqrtf(k), dest);
    return true;
}

#endif /* cvec2_h */

/*** End of inlined file: vec2.h ***/


/*** Start of inlined file: vec3.h ***/
/*
 Macros:
   PLAY_CGLM_VEC3_ONE_INIT
   PLAY_CGLM_VEC3_ZERO_INIT
   PLAY_CGLM_VEC3_ONE
   PLAY_CGLM_VEC3_ZERO
   PLAY_CGLM_YUP
   PLAY_CGLM_ZUP
   PLAY_CGLM_XUP

 Functions:
   PLAY_CGLM_INLINE void  vec3_new(vec4 v4, vec3 dest);
   PLAY_CGLM_INLINE void  vec3_copy(vec3 a, vec3 dest);
   PLAY_CGLM_INLINE void  vec3_zero(vec3 v);
   PLAY_CGLM_INLINE void  vec3_one(vec3 v);
   PLAY_CGLM_INLINE float vec3_dot(vec3 a, vec3 b);
   PLAY_CGLM_INLINE float vec3_norm2(vec3 v);
   PLAY_CGLM_INLINE float vec3_norm(vec3 v);
   PLAY_CGLM_INLINE float vec3_norm_one(vec3 v);
   PLAY_CGLM_INLINE float vec3_norm_inf(vec3 v);
   PLAY_CGLM_INLINE void  vec3_add(vec3 a, vec3 b, vec3 dest);
   PLAY_CGLM_INLINE void  vec3_adds(vec3 a, float s, vec3 dest);
   PLAY_CGLM_INLINE void  vec3_sub(vec3 a, vec3 b, vec3 dest);
   PLAY_CGLM_INLINE void  vec3_subs(vec3 a, float s, vec3 dest);
   PLAY_CGLM_INLINE void  vec3_mul(vec3 a, vec3 b, vec3 dest);
   PLAY_CGLM_INLINE void  vec3_scale(vec3 v, float s, vec3 dest);
   PLAY_CGLM_INLINE void  vec3_scale_as(vec3 v, float s, vec3 dest);
   PLAY_CGLM_INLINE void  vec3_div(vec3 a, vec3 b, vec3 dest);
   PLAY_CGLM_INLINE void  vec3_divs(vec3 a, float s, vec3 dest);
   PLAY_CGLM_INLINE void  vec3_addadd(vec3 a, vec3 b, vec3 dest);
   PLAY_CGLM_INLINE void  vec3_subadd(vec3 a, vec3 b, vec3 dest);
   PLAY_CGLM_INLINE void  vec3_muladd(vec3 a, vec3 b, vec3 dest);
   PLAY_CGLM_INLINE void  vec3_muladds(vec3 a, float s, vec3 dest);
   PLAY_CGLM_INLINE void  vec3_maxadd(vec3 a, vec3 b, vec3 dest);
   PLAY_CGLM_INLINE void  vec3_minadd(vec3 a, vec3 b, vec3 dest);
   PLAY_CGLM_INLINE void  vec3_subsub(vec3 a, vec3 b, vec3 dest);
   PLAY_CGLM_INLINE void  vec3_addsub(vec3 a, vec3 b, vec3 dest);
   PLAY_CGLM_INLINE void  vec3_mulsub(vec3 a, vec3 b, vec3 dest);
   PLAY_CGLM_INLINE void  vec3_mulsubs(vec3 a, float s, vec3 dest);
   PLAY_CGLM_INLINE void  vec3_maxsub(vec3 a, vec3 b, vec3 dest);
   PLAY_CGLM_INLINE void  vec3_minsub(vec3 a, vec3 b, vec3 dest);
   PLAY_CGLM_INLINE void  vec3_flipsign(vec3 v);
   PLAY_CGLM_INLINE void  vec3_flipsign_to(vec3 v, vec3 dest);
   PLAY_CGLM_INLINE void  vec3_negate_to(vec3 v, vec3 dest);
   PLAY_CGLM_INLINE void  vec3_negate(vec3 v);
   PLAY_CGLM_INLINE void  vec3_inv(vec3 v);
   PLAY_CGLM_INLINE void  vec3_inv_to(vec3 v, vec3 dest);
   PLAY_CGLM_INLINE void  vec3_normalize(vec3 v);
   PLAY_CGLM_INLINE void  vec3_normalize_to(vec3 v, vec3 dest);
   PLAY_CGLM_INLINE void  vec3_cross(vec3 a, vec3 b, vec3 d);
   PLAY_CGLM_INLINE void  vec3_crossn(vec3 a, vec3 b, vec3 dest);
   PLAY_CGLM_INLINE float vec3_angle(vec3 a, vec3 b);
   PLAY_CGLM_INLINE void  vec3_rotate(vec3 v, float angle, vec3 axis);
   PLAY_CGLM_INLINE void  vec3_rotate_m4(mat4 m, vec3 v, vec3 dest);
   PLAY_CGLM_INLINE void  vec3_rotate_m3(mat3 m, vec3 v, vec3 dest);
   PLAY_CGLM_INLINE void  vec3_proj(vec3 a, vec3 b, vec3 dest);
   PLAY_CGLM_INLINE void  vec3_center(vec3 a, vec3 b, vec3 dest);
   PLAY_CGLM_INLINE float vec3_distance(vec3 a, vec3 b);
   PLAY_CGLM_INLINE float vec3_distance2(vec3 a, vec3 b);
   PLAY_CGLM_INLINE void  vec3_maxv(vec3 a, vec3 b, vec3 dest);
   PLAY_CGLM_INLINE void  vec3_minv(vec3 a, vec3 b, vec3 dest);
   PLAY_CGLM_INLINE void  vec3_ortho(vec3 v, vec3 dest);
   PLAY_CGLM_INLINE void  vec3_clamp(vec3 v, float minVal, float maxVal);
   PLAY_CGLM_INLINE void  vec3_lerp(vec3 from, vec3 to, float t, vec3 dest);
   PLAY_CGLM_INLINE void  vec3_lerpc(vec3 from, vec3 to, float t, vec3 dest);
   PLAY_CGLM_INLINE void  vec3_mix(vec3 from, vec3 to, float t, vec3 dest);
   PLAY_CGLM_INLINE void  vec3_mixc(vec3 from, vec3 to, float t, vec3 dest);
   PLAY_CGLM_INLINE void  vec3_step(vec3 edge, vec3 x, vec3 dest);
   PLAY_CGLM_INLINE void  vec3_smoothstep_uni(float edge0, float edge1, vec3 x, vec3 dest);
   PLAY_CGLM_INLINE void  vec3_smoothstep(vec3 edge0, vec3 edge1, vec3 x, vec3 dest);
   PLAY_CGLM_INLINE void  vec3_smoothinterp(vec3 from, vec3 to, float t, vec3 dest);
   PLAY_CGLM_INLINE void  vec3_smoothinterpc(vec3 from, vec3 to, float t, vec3 dest);
   PLAY_CGLM_INLINE void  vec3_swizzle(vec3 v, int mask, vec3 dest);
   PLAY_CGLM_INLINE void  vec3_make(float * restrict src, vec3 dest);
   PLAY_CGLM_INLINE void  vec3_faceforward(vec3 n, vec3 v, vec3 nref, vec3 dest);
   PLAY_CGLM_INLINE void  vec3_reflect(vec3 v, vec3 n, vec3 dest);
   PLAY_CGLM_INLINE void  vec3_refract(vec3 v, vec3 n, float eta, vec3 dest);

 Convenient:
   PLAY_CGLM_INLINE void  cross(vec3 a, vec3 b, vec3 d);
   PLAY_CGLM_INLINE float dot(vec3 a, vec3 b);
   PLAY_CGLM_INLINE void  normalize(vec3 v);
   PLAY_CGLM_INLINE void  normalize_to(vec3 v, vec3 dest);

 DEPRECATED:
   vec3_dup
   vec3_flipsign
   vec3_flipsign_to
   vec3_inv
   vec3_inv_to
   vec3_mulv
   vec3_step_uni  -->  use vec3_steps
 */

#ifndef cvec3_h
#define cvec3_h


/*** Start of inlined file: vec4.h ***/
/*
 Macros:
   PLAY_CGLM_VEC4_ONE_INIT
   PLAY_CGLM_VEC4_BLACK_INIT
   PLAY_CGLM_VEC4_ZERO_INIT
   PLAY_CGLM_VEC4_ONE
   PLAY_CGLM_VEC4_BLACK
   PLAY_CGLM_VEC4_ZERO

 Functions:
   PLAY_CGLM_INLINE void  vec4_new(vec3 v3, float last, vec4 dest);
   PLAY_CGLM_INLINE void  vec4_copy3(vec4 a, vec3 dest);
   PLAY_CGLM_INLINE void  vec4_copy(vec4 v, vec4 dest);
   PLAY_CGLM_INLINE void  vec4_ucopy(vec4 v, vec4 dest);
   PLAY_CGLM_INLINE float vec4_dot(vec4 a, vec4 b);
   PLAY_CGLM_INLINE float vec4_norm2(vec4 v);
   PLAY_CGLM_INLINE float vec4_norm(vec4 v);
   PLAY_CGLM_INLINE float vec4_norm_one(vec4 v);
   PLAY_CGLM_INLINE float vec4_norm_inf(vec4 v);
   PLAY_CGLM_INLINE void  vec4_add(vec4 a, vec4 b, vec4 dest);
   PLAY_CGLM_INLINE void  vec4_adds(vec4 v, float s, vec4 dest);
   PLAY_CGLM_INLINE void  vec4_sub(vec4 a, vec4 b, vec4 dest);
   PLAY_CGLM_INLINE void  vec4_subs(vec4 v, float s, vec4 dest);
   PLAY_CGLM_INLINE void  vec4_mul(vec4 a, vec4 b, vec4 dest);
   PLAY_CGLM_INLINE void  vec4_scale(vec4 v, float s, vec4 dest);
   PLAY_CGLM_INLINE void  vec4_scale_as(vec4 v, float s, vec4 dest);
   PLAY_CGLM_INLINE void  vec4_div(vec4 a, vec4 b, vec4 dest);
   PLAY_CGLM_INLINE void  vec4_divs(vec4 v, float s, vec4 dest);
   PLAY_CGLM_INLINE void  vec4_addadd(vec4 a, vec4 b, vec4 dest);
   PLAY_CGLM_INLINE void  vec4_subadd(vec4 a, vec4 b, vec4 dest);
   PLAY_CGLM_INLINE void  vec4_muladd(vec4 a, vec4 b, vec4 dest);
   PLAY_CGLM_INLINE void  vec4_muladds(vec4 a, float s, vec4 dest);
   PLAY_CGLM_INLINE void  vec4_maxadd(vec4 a, vec4 b, vec4 dest);
   PLAY_CGLM_INLINE void  vec4_minadd(vec4 a, vec4 b, vec4 dest);
   PLAY_CGLM_INLINE void  vec4_subsub(vec4 a, vec4 b, vec4 dest);
   PLAY_CGLM_INLINE void  vec4_addsub(vec4 a, vec4 b, vec4 dest);
   PLAY_CGLM_INLINE void  vec4_mulsub(vec4 a, vec4 b, vec4 dest);
   PLAY_CGLM_INLINE void  vec4_mulsubs(vec4 a, float s, vec4 dest);
   PLAY_CGLM_INLINE void  vec4_maxsub(vec4 a, vec4 b, vec4 dest);
   PLAY_CGLM_INLINE void  vec4_minsub(vec4 a, vec4 b, vec4 dest);
   PLAY_CGLM_INLINE void  vec4_negate(vec4 v);
   PLAY_CGLM_INLINE void  vec4_inv(vec4 v);
   PLAY_CGLM_INLINE void  vec4_inv_to(vec4 v, vec4 dest);
   PLAY_CGLM_INLINE void  vec4_normalize(vec4 v);
   PLAY_CGLM_INLINE void  vec4_normalize_to(vec4 vec, vec4 dest);
   PLAY_CGLM_INLINE float vec4_distance(vec4 a, vec4 b);
   PLAY_CGLM_INLINE float vec4_distance2(vec4 a, vec4 b);
   PLAY_CGLM_INLINE void  vec4_maxv(vec4 a, vec4 b, vec4 dest);
   PLAY_CGLM_INLINE void  vec4_minv(vec4 a, vec4 b, vec4 dest);
   PLAY_CGLM_INLINE void  vec4_clamp(vec4 v, float minVal, float maxVal);
   PLAY_CGLM_INLINE void  vec4_lerp(vec4 from, vec4 to, float t, vec4 dest);
   PLAY_CGLM_INLINE void  vec4_lerpc(vec4 from, vec4 to, float t, vec4 dest);
   PLAY_CGLM_INLINE void  vec4_step(vec4 edge, vec4 x, vec4 dest);
   PLAY_CGLM_INLINE void  vec4_smoothstep_uni(float edge0, float edge1, vec4 x, vec4 dest);
   PLAY_CGLM_INLINE void  vec4_smoothstep(vec4 edge0, vec4 edge1, vec4 x, vec4 dest);
   PLAY_CGLM_INLINE void  vec4_smoothinterp(vec4 from, vec4 to, float t, vec4 dest);
   PLAY_CGLM_INLINE void  vec4_smoothinterpc(vec4 from, vec4 to, float t, vec4 dest);
   PLAY_CGLM_INLINE void  vec4_swizzle(vec4 v, int mask, vec4 dest);
   PLAY_CGLM_INLINE void  vec4_make(float * restrict src, vec4 dest);
   PLAY_CGLM_INLINE void  vec4_reflect(vec4 v, vec4 n, vec4 dest);
   PLAY_CGLM_INLINE void  vec4_refract(vec4 v, vec4 n, float eta, vec4 dest);

 DEPRECATED:
   vec4_dup
   vec4_flipsign
   vec4_flipsign_to
   vec4_inv
   vec4_inv_to
   vec4_mulv
   vec4_step_uni  --> use vec4_steps
 */

#ifndef cvec4_h
#define cvec4_h


/*** Start of inlined file: vec4-ext.h ***/
/*!
 * @brief SIMD like functions
 */

/*
 Functions:
   PLAY_CGLM_INLINE void  vec4_broadcast(float val, vec4 d);
   PLAY_CGLM_INLINE void  vec4_fill(vec4 v, float val);
   PLAY_CGLM_INLINE bool  vec4_eq(vec4 v, float val);
   PLAY_CGLM_INLINE bool  vec4_eq_eps(vec4 v, float val);
   PLAY_CGLM_INLINE bool  vec4_eq_all(vec4 v);
   PLAY_CGLM_INLINE bool  vec4_eqv(vec4 a, vec4 b);
   PLAY_CGLM_INLINE bool  vec4_eqv_eps(vec4 a, vec4 b);
   PLAY_CGLM_INLINE float vec4_max(vec4 v);
   PLAY_CGLM_INLINE float vec4_min(vec4 v);
   PLAY_CGLM_INLINE bool  vec4_isnan(vec4 v);
   PLAY_CGLM_INLINE bool  vec4_isinf(vec4 v);
   PLAY_CGLM_INLINE bool  vec4_isvalid(vec4 v);
   PLAY_CGLM_INLINE void  vec4_sign(vec4 v, vec4 dest);
   PLAY_CGLM_INLINE void  vec4_abs(vec4 v, vec4 dest);
   PLAY_CGLM_INLINE void  vec4_fract(vec4 v, vec4 dest);
   PLAY_CGLM_INLINE void  vec4_floor(vec4 v, vec4 dest);
   PLAY_CGLM_INLINE float vec4_mods(vec4 v, float s, vec4 dest);
   PLAY_CGLM_INLINE float vec4_steps(float edge, vec4 v, vec4 dest);
   PLAY_CGLM_INLINE void  vec4_stepr(vec4 edge, float v, vec4 dest);
   PLAY_CGLM_INLINE float vec4_hadd(vec4 v);
   PLAY_CGLM_INLINE void  vec4_sqrt(vec4 v, vec4 dest);
 */

#ifndef cvec4_ext_h
#define cvec4_ext_h


/*** Start of inlined file: vec3-ext.h ***/
/*!
 * @brief SIMD like functions
 */

/*
 Functions:
   PLAY_CGLM_INLINE void  vec3_broadcast(float val, vec3 d);
   PLAY_CGLM_INLINE void  vec3_fill(vec3 v, float val);
   PLAY_CGLM_INLINE bool  vec3_eq(vec3 v, float val);
   PLAY_CGLM_INLINE bool  vec3_eq_eps(vec3 v, float val);
   PLAY_CGLM_INLINE bool  vec3_eq_all(vec3 v);
   PLAY_CGLM_INLINE bool  vec3_eqv(vec3 a, vec3 b);
   PLAY_CGLM_INLINE bool  vec3_eqv_eps(vec3 a, vec3 b);
   PLAY_CGLM_INLINE float vec3_max(vec3 v);
   PLAY_CGLM_INLINE float vec3_min(vec3 v);
   PLAY_CGLM_INLINE bool  vec3_isnan(vec3 v);
   PLAY_CGLM_INLINE bool  vec3_isinf(vec3 v);
   PLAY_CGLM_INLINE bool  vec3_isvalid(vec3 v);
   PLAY_CGLM_INLINE void  vec3_sign(vec3 v, vec3 dest);
   PLAY_CGLM_INLINE void  vec3_abs(vec3 v, vec3 dest);
   PLAY_CGLM_INLINE void  vec3_fract(vec3 v, vec3 dest);
   PLAY_CGLM_INLINE void  vec3_floor(vec3 v, vec3 dest);
   PLAY_CGLM_INLINE float vec3_mods(vec3 v, float s, vec3 dest);
   PLAY_CGLM_INLINE float vec3_steps(float edge, vec3 v, vec3 dest);
   PLAY_CGLM_INLINE void  vec3_stepr(vec3 edge, float v, vec3 dest);
   PLAY_CGLM_INLINE float vec3_hadd(vec3 v);
   PLAY_CGLM_INLINE void  vec3_sqrt(vec3 v, vec3 dest);
 */

#ifndef cvec3_ext_h
#define cvec3_ext_h

/*!
 * @brief fill a vector with specified value
 *
 * @param[in]  val value
 * @param[out] d   dest
 */
PLAY_CGLM_INLINE
void
vec3_broadcast(float val, vec3 d)
{
    d[0] = d[1] = d[2] = val;
}

/*!
 * @brief fill a vector with specified value
 *
 * @param[out] v   dest
 * @param[in]  val value
 */
PLAY_CGLM_INLINE
void
vec3_fill(vec3 v, float val)
{
    v[0] = v[1] = v[2] = val;
}

/*!
 * @brief check if vector is equal to value (without epsilon)
 *
 * @param[in] v   vector
 * @param[in] val value
 */
PLAY_CGLM_INLINE
bool
vec3_eq(vec3 v, float val)
{
    return v[0] == val && v[0] == v[1] && v[0] == v[2];
}

/*!
 * @brief check if vector is equal to value (with epsilon)
 *
 * @param[in] v   vector
 * @param[in] val value
 */
PLAY_CGLM_INLINE
bool
vec3_eq_eps(vec3 v, float val)
{
    return fabsf(v[0] - val) <= PLAY_CGLM_FLT_EPSILON
           && fabsf(v[1] - val) <= PLAY_CGLM_FLT_EPSILON
           && fabsf(v[2] - val) <= PLAY_CGLM_FLT_EPSILON;
}

/*!
 * @brief check if vector members are equal (without epsilon)
 *
 * @param[in] v   vector
 */
PLAY_CGLM_INLINE
bool
vec3_eq_all(vec3 v)
{
    return vec3_eq_eps(v, v[0]);
}

/*!
 * @brief check if vector is equal to another (without epsilon)
 *
 * @param[in] a vector
 * @param[in] b vector
 */
PLAY_CGLM_INLINE
bool
vec3_eqv(vec3 a, vec3 b)
{
    return a[0] == b[0]
           && a[1] == b[1]
           && a[2] == b[2];
}

/*!
 * @brief check if vector is equal to another (with epsilon)
 *
 * @param[in] a vector
 * @param[in] b vector
 */
PLAY_CGLM_INLINE
bool
vec3_eqv_eps(vec3 a, vec3 b)
{
    return fabsf(a[0] - b[0]) <= PLAY_CGLM_FLT_EPSILON
           && fabsf(a[1] - b[1]) <= PLAY_CGLM_FLT_EPSILON
           && fabsf(a[2] - b[2]) <= PLAY_CGLM_FLT_EPSILON;
}

/*!
 * @brief max value of vector
 *
 * @param[in] v vector
 */
PLAY_CGLM_INLINE
float
vec3_max(vec3 v)
{
    float max;

    max = v[0];
    if (v[1] > max)
        max = v[1];
    if (v[2] > max)
        max = v[2];

    return max;
}

/*!
 * @brief min value of vector
 *
 * @param[in] v vector
 */
PLAY_CGLM_INLINE
float
vec3_min(vec3 v)
{
    float min;

    min = v[0];
    if (v[1] < min)
        min = v[1];
    if (v[2] < min)
        min = v[2];

    return min;
}

/*!
 * @brief check if one of items is NaN (not a number)
 *        you should only use this in DEBUG mode or very critical asserts
 *
 * @param[in] v vector
 */
PLAY_CGLM_INLINE
bool
vec3_isnan(vec3 v)
{
#ifndef PLAY_CGLM_FAST_MATH
    return isnan(v[0]) || isnan(v[1]) || isnan(v[2]);
#else
    return false;
#endif
}

/*!
 * @brief check if one of items is INFINITY
 *        you should only use this in DEBUG mode or very critical asserts
 *
 * @param[in] v vector
 */
PLAY_CGLM_INLINE
bool
vec3_isinf(vec3 v)
{
#ifndef PLAY_CGLM_FAST_MATH
    return isinf(v[0]) || isinf(v[1]) || isinf(v[2]);
#else
    return false;
#endif
}

/*!
 * @brief check if all items are valid number
 *        you should only use this in DEBUG mode or very critical asserts
 *
 * @param[in] v vector
 */
PLAY_CGLM_INLINE
bool
vec3_isvalid(vec3 v)
{
    return !vec3_isnan(v) && !vec3_isinf(v);
}

/*!
 * @brief get sign of 32 bit float as +1, -1, 0
 *
 * Important: It returns 0 for zero/NaN input
 *
 * @param v vector
 */
PLAY_CGLM_INLINE
void
vec3_sign(vec3 v, vec3 dest)
{
    dest[0] = signf(v[0]);
    dest[1] = signf(v[1]);
    dest[2] = signf(v[2]);
}

/*!
 * @brief absolute value of each vector item
 *
 * @param[in]  v    vector
 * @param[out] dest destination vector
 */
PLAY_CGLM_INLINE
void
vec3_abs(vec3 v, vec3 dest)
{
    dest[0] = fabsf(v[0]);
    dest[1] = fabsf(v[1]);
    dest[2] = fabsf(v[2]);
}

/*!
 * @brief fractional part of each vector item
 *
 * @param[in]  v    vector
 * @param[out] dest destination vector
 */
PLAY_CGLM_INLINE
void
vec3_fract(vec3 v, vec3 dest)
{
    dest[0] = fminf(v[0] - floorf(v[0]), 0.999999940395355224609375f);
    dest[1] = fminf(v[1] - floorf(v[1]), 0.999999940395355224609375f);
    dest[2] = fminf(v[2] - floorf(v[2]), 0.999999940395355224609375f);
}

/*!
 * @brief floor of each vector item
 *
 * @param[in]  v    vector
 * @param[out] dest destination vector
 */
PLAY_CGLM_INLINE
void
vec3_floor(vec3 v, vec3 dest)
{
    dest[0] = floorf(v[0]);
    dest[1] = floorf(v[1]);
    dest[2] = floorf(v[2]);
}

/*!
 * @brief mod of each vector item, result is written to dest (dest = v % s)
 *
 * @param[in]  v    vector
 * @param[in]  s    scalar
 * @param[out] dest destination vector
 */
PLAY_CGLM_INLINE
void
vec3_mods(vec3 v, float s, vec3 dest)
{
    dest[0] = fmodf(v[0], s);
    dest[1] = fmodf(v[1], s);
    dest[2] = fmodf(v[2], s);
}

/*!
 * @brief threshold each vector item with scalar
 *        condition is: (x[i] < edge) ? 0.0 : 1.0
 *
 * @param[in]   edge    threshold
 * @param[in]   x       vector to test against threshold
 * @param[out]  dest    destination
 */
PLAY_CGLM_INLINE
void
vec3_steps(float edge, vec3 x, vec3 dest)
{
    dest[0] = step(edge, x[0]);
    dest[1] = step(edge, x[1]);
    dest[2] = step(edge, x[2]);
}

/*!
 * @brief threshold a value with *vector* as the threshold
 *        condition is: (x < edge[i]) ? 0.0 : 1.0
 *
 * @param[in]   edge    threshold vector
 * @param[in]   x       value to test against threshold
 * @param[out]  dest    destination
 */
PLAY_CGLM_INLINE
void
vec3_stepr(vec3 edge, float x, vec3 dest)
{
    dest[0] = step(edge[0], x);
    dest[1] = step(edge[1], x);
    dest[2] = step(edge[2], x);
}

/*!
 * @brief vector reduction by summation
 * @warning could overflow
 *
 * @param[in]  v    vector
 * @return     sum of all vector's elements
 */
PLAY_CGLM_INLINE
float
vec3_hadd(vec3 v)
{
    return v[0] + v[1] + v[2];
}

/*!
 * @brief square root of each vector item
 *
 * @param[in]  v    vector
 * @param[out] dest destination vector
 */
PLAY_CGLM_INLINE
void
vec3_sqrt(vec3 v, vec3 dest)
{
    dest[0] = sqrtf(v[0]);
    dest[1] = sqrtf(v[1]);
    dest[2] = sqrtf(v[2]);
}

#endif /* cvec3_ext_h */

/*** End of inlined file: vec3-ext.h ***/

/*!
 * @brief fill a vector with specified value
 *
 * @param val value
 * @param d   dest
 */
PLAY_CGLM_INLINE
void
vec4_broadcast(float val, vec4 d)
{
#if defined(__wasm__) && defined(__wasm_simd128__)
    glmm_store(d, wasm_f32x4_splat(val));
#elif defined( __SSE__ ) || defined( __SSE2__ )
    glmm_store(d, glmm_set1(val));
#else
    d[0] = d[1] = d[2] = d[3] = val;
#endif
}

/*!
 * @brief fill a vector with specified value
 *
 * @param v   dest
 * @param val value
 */
PLAY_CGLM_INLINE
void
vec4_fill(vec4 v, float val)
{
#if defined(__wasm__) && defined(__wasm_simd128__)
    glmm_store(v, wasm_f32x4_splat(val));
#elif defined( __SSE__ ) || defined( __SSE2__ )
    glmm_store(v, glmm_set1(val));
#else
    v[0] = v[1] = v[2] = v[3] = val;
#endif
}

/*!
 * @brief check if vector is equal to value (without epsilon)
 *
 * @param v   vector
 * @param val value
 */
PLAY_CGLM_INLINE
bool
vec4_eq(vec4 v, float val)
{
    return v[0] == val
           && v[0] == v[1]
           && v[0] == v[2]
           && v[0] == v[3];
}

/*!
 * @brief check if vector is equal to value (with epsilon)
 *
 * @param v   vector
 * @param val value
 */
PLAY_CGLM_INLINE
bool
vec4_eq_eps(vec4 v, float val)
{
    return fabsf(v[0] - val) <= PLAY_CGLM_FLT_EPSILON
           && fabsf(v[1] - val) <= PLAY_CGLM_FLT_EPSILON
           && fabsf(v[2] - val) <= PLAY_CGLM_FLT_EPSILON
           && fabsf(v[3] - val) <= PLAY_CGLM_FLT_EPSILON;
}

/*!
 * @brief check if vector members are equal (without epsilon)
 *
 * @param v   vector
 */
PLAY_CGLM_INLINE
bool
vec4_eq_all(vec4 v)
{
    return vec4_eq_eps(v, v[0]);
}

/*!
 * @brief check if vector is equal to another (without epsilon)
 *
 * @param a vector
 * @param b vector
 */
PLAY_CGLM_INLINE
bool
vec4_eqv(vec4 a, vec4 b)
{
    return a[0] == b[0]
           && a[1] == b[1]
           && a[2] == b[2]
           && a[3] == b[3];
}

/*!
 * @brief check if vector is equal to another (with epsilon)
 *
 * @param a vector
 * @param b vector
 */
PLAY_CGLM_INLINE
bool
vec4_eqv_eps(vec4 a, vec4 b)
{
    return fabsf(a[0] - b[0]) <= PLAY_CGLM_FLT_EPSILON
           && fabsf(a[1] - b[1]) <= PLAY_CGLM_FLT_EPSILON
           && fabsf(a[2] - b[2]) <= PLAY_CGLM_FLT_EPSILON
           && fabsf(a[3] - b[3]) <= PLAY_CGLM_FLT_EPSILON;
}

/*!
 * @brief max value of vector
 *
 * @param v vector
 */
PLAY_CGLM_INLINE
float
vec4_max(vec4 v)
{
    float max;

    max = vec3_max(v);
    if (v[3] > max)
        max = v[3];

    return max;
}

/*!
 * @brief min value of vector
 *
 * @param v vector
 */
PLAY_CGLM_INLINE
float
vec4_min(vec4 v)
{
    float min;

    min = vec3_min(v);
    if (v[3] < min)
        min = v[3];

    return min;
}

/*!
 * @brief check if one of items is NaN (not a number)
 *        you should only use this in DEBUG mode or very critical asserts
 *
 * @param[in] v vector
 */
PLAY_CGLM_INLINE
bool
vec4_isnan(vec4 v)
{
#ifndef PLAY_CGLM_FAST_MATH
    return isnan(v[0]) || isnan(v[1]) || isnan(v[2]) || isnan(v[3]);
#else
    return false;
#endif
}

/*!
 * @brief check if one of items is INFINITY
 *        you should only use this in DEBUG mode or very critical asserts
 *
 * @param[in] v vector
 */
PLAY_CGLM_INLINE
bool
vec4_isinf(vec4 v)
{
#ifndef PLAY_CGLM_FAST_MATH
    return isinf(v[0]) || isinf(v[1]) || isinf(v[2]) || isinf(v[3]);
#else
    return false;
#endif
}

/*!
 * @brief check if all items are valid number
 *        you should only use this in DEBUG mode or very critical asserts
 *
 * @param[in] v vector
 */
PLAY_CGLM_INLINE
bool
vec4_isvalid(vec4 v)
{
    return !vec4_isnan(v) && !vec4_isinf(v);
}

/*!
 * @brief get sign of 32 bit float as +1, -1, 0
 *
 * Important: It returns 0 for zero/NaN input
 *
 * @param v vector
 */
PLAY_CGLM_INLINE
void
vec4_sign(vec4 v, vec4 dest)
{
#if defined( __SSE__ ) || defined( __SSE2__ )
    __m128 x0, x1, x2, x3, x4;

    x0 = glmm_load(v);
    x1 = _mm_set_ps(0.0f, 0.0f, 1.0f, -1.0f);
    x2 = glmm_splat(x1, 2);

    x3 = _mm_and_ps(_mm_cmpgt_ps(x0, x2), glmm_splat(x1, 1));
    x4 = _mm_and_ps(_mm_cmplt_ps(x0, x2), glmm_splat(x1, 0));

    glmm_store(dest, _mm_or_ps(x3, x4));
#else
    dest[0] = signf(v[0]);
    dest[1] = signf(v[1]);
    dest[2] = signf(v[2]);
    dest[3] = signf(v[3]);
#endif
}

/*!
 * @brief absolute value of each vector item
 *
 * @param[in]  v    vector
 * @param[out] dest destination vector
 */
PLAY_CGLM_INLINE
void
vec4_abs(vec4 v, vec4 dest)
{
#if defined(__wasm__) && defined(__wasm_simd128__)
    glmm_store(dest, glmm_abs(glmm_load(v)));
#elif defined( __SSE__ ) || defined( __SSE2__ )
    glmm_store(dest, glmm_abs(glmm_load(v)));
#elif defined(PLAY_CGLM_NEON_FP)
    vst1q_f32(dest, vabsq_f32(vld1q_f32(v)));
#else
    dest[0] = fabsf(v[0]);
    dest[1] = fabsf(v[1]);
    dest[2] = fabsf(v[2]);
    dest[3] = fabsf(v[3]);
#endif
}

/*!
 * @brief fractional part of each vector item
 *
 * @param[in]  v    vector
 * @param[out] dest destination vector
 */
PLAY_CGLM_INLINE
void
vec4_fract(vec4 v, vec4 dest)
{
    dest[0] = fminf(v[0] - floorf(v[0]), 0.999999940395355224609375f);
    dest[1] = fminf(v[1] - floorf(v[1]), 0.999999940395355224609375f);
    dest[2] = fminf(v[2] - floorf(v[2]), 0.999999940395355224609375f);
    dest[3] = fminf(v[3] - floorf(v[3]), 0.999999940395355224609375f);
}

/*!
 * @brief floor of each vector item
 *
 * @param[in]  v    vector
 * @param[out] dest destination vector
 */
PLAY_CGLM_INLINE
void
vec4_floor(vec4 v, vec4 dest)
{
    dest[0] = floorf(v[0]);
    dest[1] = floorf(v[1]);
    dest[2] = floorf(v[2]);
    dest[3] = floorf(v[3]);
}

/*!
 * @brief mod of each vector item, result is written to dest (dest = v % s)
 *
 * @param[in]  v    vector
 * @param[in]  s    scalar
 * @param[out] dest destination vector
 */
PLAY_CGLM_INLINE
void
vec4_mods(vec4 v, float s, vec4 dest)
{
    dest[0] = fmodf(v[0], s);
    dest[1] = fmodf(v[1], s);
    dest[2] = fmodf(v[2], s);
    dest[3] = fmodf(v[3], s);
}

/*!
 * @brief threshold each vector item with scalar
 *        condition is: (x[i] < edge) ? 0.0 : 1.0
 *
 * @param[in]   edge    threshold
 * @param[in]   x       vector to test against threshold
 * @param[out]  dest    destination
 */
PLAY_CGLM_INLINE
void
vec4_steps(float edge, vec4 x, vec4 dest)
{
    dest[0] = step(edge, x[0]);
    dest[1] = step(edge, x[1]);
    dest[2] = step(edge, x[2]);
    dest[3] = step(edge, x[3]);
}

/*!
 * @brief threshold a value with *vector* as the threshold
 *        condition is: (x < edge[i]) ? 0.0 : 1.0
 *
 * @param[in]   edge    threshold vector
 * @param[in]   x       value to test against threshold
 * @param[out]  dest    destination
 */
PLAY_CGLM_INLINE
void
vec4_stepr(vec4 edge, float x, vec4 dest)
{
    dest[0] = step(edge[0], x);
    dest[1] = step(edge[1], x);
    dest[2] = step(edge[2], x);
    dest[3] = step(edge[3], x);
}

/*!
 * @brief vector reduction by summation
 * @warning could overflow
 *
 * @param[in]   v    vector
 * @return      sum of all vector's elements
 */
PLAY_CGLM_INLINE
float
vec4_hadd(vec4 v)
{
#if defined(__wasm__) && defined(__wasm_simd128__)
    return glmm_hadd(glmm_load(v));
#elif defined( __SSE__ ) || defined( __SSE2__ )
    return glmm_hadd(glmm_load(v));
#else
    return v[0] + v[1] + v[2] + v[3];
#endif
}

/*!
 * @brief square root of each vector item
 *
 * @param[in]  v    vector
 * @param[out] dest destination vector
 */
PLAY_CGLM_INLINE
void
vec4_sqrt(vec4 v, vec4 dest)
{
#if defined(__wasm__) && defined(__wasm_simd128__)
    glmm_store(dest, wasm_f32x4_sqrt(glmm_load(v)));
#elif defined( __SSE__ ) || defined( __SSE2__ )
    glmm_store(dest, _mm_sqrt_ps(glmm_load(v)));
#else
    dest[0] = sqrtf(v[0]);
    dest[1] = sqrtf(v[1]);
    dest[2] = sqrtf(v[2]);
    dest[3] = sqrtf(v[3]);
#endif
}

#endif /* cvec4_ext_h */

/*** End of inlined file: vec4-ext.h ***/

/* DEPRECATED! functions */
#define vec4_dup3(v, dest)         vec4_copy3(v, dest)
#define vec4_dup(v, dest)          vec4_copy(v, dest)
#define vec4_flipsign(v)           vec4_negate(v)
#define vec4_flipsign_to(v, dest)  vec4_negate_to(v, dest)
#define vec4_inv(v)                vec4_negate(v)
#define vec4_inv_to(v, dest)       vec4_negate_to(v, dest)
#define vec4_mulv(a, b, d)         vec4_mul(a, b, d)
#define vec4_step_uni(edge, x, dest) vec4_steps(edge, x, dest)

#define PLAY_CGLM_VEC4_ONE_INIT   {1.0f, 1.0f, 1.0f, 1.0f}
#define PLAY_CGLM_VEC4_BLACK_INIT {0.0f, 0.0f, 0.0f, 1.0f}
#define PLAY_CGLM_VEC4_ZERO_INIT  {0.0f, 0.0f, 0.0f, 0.0f}

#define PLAY_CGLM_VEC4_ONE        ((vec4)PLAY_CGLM_VEC4_ONE_INIT)
#define PLAY_CGLM_VEC4_BLACK      ((vec4)PLAY_CGLM_VEC4_BLACK_INIT)
#define PLAY_CGLM_VEC4_ZERO       ((vec4)PLAY_CGLM_VEC4_ZERO_INIT)

#define PLAY_CGLM_XXXX PLAY_CGLM_SHUFFLE4(0, 0, 0, 0)
#define PLAY_CGLM_YYYY PLAY_CGLM_SHUFFLE4(1, 1, 1, 1)
#define PLAY_CGLM_ZZZZ PLAY_CGLM_SHUFFLE4(2, 2, 2, 2)
#define PLAY_CGLM_WWWW PLAY_CGLM_SHUFFLE4(3, 3, 3, 3)
#define PLAY_CGLM_WZYX PLAY_CGLM_SHUFFLE4(0, 1, 2, 3)

/*!
 * @brief init vec4 using vec3
 *
 * @param[in]  v3   vector3
 * @param[in]  last last item
 * @param[out] dest destination
 */
PLAY_CGLM_INLINE
void
vec4_new(vec3 v3, float last, vec4 dest)
{
    dest[0] = v3[0];
    dest[1] = v3[1];
    dest[2] = v3[2];
    dest[3] = last;
}

/*!
 * @brief copy first 3 members of [a] to [dest]
 *
 * @param[in]  a    source
 * @param[out] dest destination
 */
PLAY_CGLM_INLINE
void
vec4_copy3(vec4 a, vec3 dest)
{
    dest[0] = a[0];
    dest[1] = a[1];
    dest[2] = a[2];
}

/*!
 * @brief copy all members of [a] to [dest]
 *
 * @param[in]  v    source
 * @param[out] dest destination
 */
PLAY_CGLM_INLINE
void
vec4_copy(vec4 v, vec4 dest)
{
#if defined(__wasm__) && defined(__wasm_simd128__)
    glmm_store(dest, glmm_load(v));
#elif defined( __SSE__ ) || defined( __SSE2__ )
    glmm_store(dest, glmm_load(v));
#elif defined(PLAY_CGLM_NEON_FP)
    vst1q_f32(dest, vld1q_f32(v));
#else
    dest[0] = v[0];
    dest[1] = v[1];
    dest[2] = v[2];
    dest[3] = v[3];
#endif
}

/*!
 * @brief copy all members of [a] to [dest]
 *
 * alignment is not required
 *
 * @param[in]  v    source
 * @param[out] dest destination
 */
PLAY_CGLM_INLINE
void
vec4_ucopy(vec4 v, vec4 dest)
{
#if defined(__wasm__) && defined(__wasm_simd128__)
    /* note here wasm v128.load/v128.store support unaligned loads and stores */
    wasm_v128_store(dest, wasm_v128_load(v));
#else
    dest[0] = v[0];
    dest[1] = v[1];
    dest[2] = v[2];
    dest[3] = v[3];
#endif
}

/*!
 * @brief make vector zero
 *
 * @param[in, out]  v vector
 */
PLAY_CGLM_INLINE
void
vec4_zero(vec4 v)
{
#if defined(__wasm__) && defined(__wasm_simd128__)
    glmm_store(v, wasm_f32x4_const_splat(0.f));
#elif defined( __SSE__ ) || defined( __SSE2__ )
    glmm_store(v, _mm_setzero_ps());
#elif defined(PLAY_CGLM_NEON_FP)
    vst1q_f32(v, vdupq_n_f32(0.0f));
#else
    v[0] = 0.0f;
    v[1] = 0.0f;
    v[2] = 0.0f;
    v[3] = 0.0f;
#endif
}

/*!
 * @brief make vector one
 *
 * @param[in, out]  v vector
 */
PLAY_CGLM_INLINE
void
vec4_one(vec4 v)
{
#if defined(__wasm__) && defined(__wasm_simd128__)
    glmm_store(v, wasm_f32x4_const_splat(1.0f));
#elif defined( __SSE__ ) || defined( __SSE2__ )
    glmm_store(v, glmm_set1_rval(1.0f));
#elif defined(PLAY_CGLM_NEON_FP)
    vst1q_f32(v, vdupq_n_f32(1.0f));
#else
    v[0] = 1.0f;
    v[1] = 1.0f;
    v[2] = 1.0f;
    v[3] = 1.0f;
#endif
}

/*!
 * @brief vec4 dot product
 *
 * @param[in] a vector1
 * @param[in] b vector2
 *
 * @return dot product
 */
PLAY_CGLM_INLINE
float
vec4_dot(vec4 a, vec4 b)
{
#if defined(PLAY_CGLM_SIMD)
    return glmm_dot(glmm_load(a), glmm_load(b));
#else
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2] + a[3] * b[3];
#endif
}

/*!
 * @brief norm * norm (magnitude) of vec
 *
 * we can use this func instead of calling norm * norm, because it would call
 * sqrtf function twice but with this func we can avoid func call, maybe this is
 * not good name for this func
 *
 * @param[in] v vec4
 *
 * @return norm * norm
 */
PLAY_CGLM_INLINE
float
vec4_norm2(vec4 v)
{
    return vec4_dot(v, v);
}

/*!
 * @brief euclidean norm (magnitude), also called L2 norm
 *        this will give magnitude of vector in euclidean space
 *
 * @param[in] v vector
 *
 * @return norm
 */
PLAY_CGLM_INLINE
float
vec4_norm(vec4 v)
{
#if defined(PLAY_CGLM_SIMD)
    return glmm_norm(glmm_load(v));
#else
    return sqrtf(vec4_dot(v, v));
#endif
}

/*!
 * @brief L1 norm of vec4
 * Also known as Manhattan Distance or Taxicab norm.
 * L1 Norm is the sum of the magnitudes of the vectors in a space.
 * It is calculated as the sum of the absolute values of the vector components.
 * In this norm, all the components of the vector are weighted equally.
 *
 * This computes:
 * L1 norm = |v[0]| + |v[1]| + |v[2]| + |v[3]|
 *
 * @param[in] v vector
 *
 * @return L1 norm
 */
PLAY_CGLM_INLINE
float
vec4_norm_one(vec4 v)
{
#if defined(PLAY_CGLM_SIMD)
    return glmm_norm_one(glmm_load(v));
#else
    vec4 t;
    vec4_abs(v, t);
    return vec4_hadd(t);
#endif
}

/*!
 * @brief infinity norm of vec4
 * Also known as Maximum norm.
 * Infinity Norm is the largest magnitude among each element of a vector.
 * It is calculated as the maximum of the absolute values of the vector components.
 *
 * This computes:
 * inf norm = fmax(|v[0]|, |v[1]|, |v[2]|, |v[3]|)
 *
 * @param[in] v vector
 *
 * @return infinity norm
 */
PLAY_CGLM_INLINE
float
vec4_norm_inf(vec4 v)
{
#if defined(PLAY_CGLM_SIMD)
    return glmm_norm_inf(glmm_load(v));
#else
    vec4 t;
    vec4_abs(v, t);
    return vec4_max(t);
#endif
}

/*!
 * @brief add b vector to a vector store result in dest
 *
 * @param[in]  a    vector1
 * @param[in]  b    vector2
 * @param[out] dest destination vector
 */
PLAY_CGLM_INLINE
void
vec4_add(vec4 a, vec4 b, vec4 dest)
{
#if defined(__wasm__) && defined(__wasm_simd128__)
    glmm_store(dest, wasm_f32x4_add(glmm_load(a), glmm_load(b)));
#elif defined( __SSE__ ) || defined( __SSE2__ )
    glmm_store(dest, _mm_add_ps(glmm_load(a), glmm_load(b)));
#elif defined(PLAY_CGLM_NEON_FP)
    vst1q_f32(dest, vaddq_f32(vld1q_f32(a), vld1q_f32(b)));
#else
    dest[0] = a[0] + b[0];
    dest[1] = a[1] + b[1];
    dest[2] = a[2] + b[2];
    dest[3] = a[3] + b[3];
#endif
}

/*!
 * @brief add scalar to v vector store result in dest (d = v + vec(s))
 *
 * @param[in]  v    vector
 * @param[in]  s    scalar
 * @param[out] dest destination vector
 */
PLAY_CGLM_INLINE
void
vec4_adds(vec4 v, float s, vec4 dest)
{
#if defined(__wasm__) && defined(__wasm_simd128__)
    glmm_store(dest, wasm_f32x4_add(glmm_load(v), wasm_f32x4_splat(s)));
#elif defined( __SSE__ ) || defined( __SSE2__ )
    glmm_store(dest, _mm_add_ps(glmm_load(v), glmm_set1(s)));
#elif defined(PLAY_CGLM_NEON_FP)
    vst1q_f32(dest, vaddq_f32(vld1q_f32(v), vdupq_n_f32(s)));
#else
    dest[0] = v[0] + s;
    dest[1] = v[1] + s;
    dest[2] = v[2] + s;
    dest[3] = v[3] + s;
#endif
}

/*!
 * @brief subtract b vector from a vector store result in dest (d = a - b)
 *
 * @param[in]  a    vector1
 * @param[in]  b    vector2
 * @param[out] dest destination vector
 */
PLAY_CGLM_INLINE
void
vec4_sub(vec4 a, vec4 b, vec4 dest)
{
#if defined(__wasm__) && defined(__wasm_simd128__)
    glmm_store(dest, wasm_f32x4_sub(glmm_load(a), glmm_load(b)));
#elif defined( __SSE__ ) || defined( __SSE2__ )
    glmm_store(dest, _mm_sub_ps(glmm_load(a), glmm_load(b)));
#elif defined(PLAY_CGLM_NEON_FP)
    vst1q_f32(dest, vsubq_f32(vld1q_f32(a), vld1q_f32(b)));
#else
    dest[0] = a[0] - b[0];
    dest[1] = a[1] - b[1];
    dest[2] = a[2] - b[2];
    dest[3] = a[3] - b[3];
#endif
}

/*!
 * @brief subtract scalar from v vector store result in dest (d = v - vec(s))
 *
 * @param[in]  v    vector
 * @param[in]  s    scalar
 * @param[out] dest destination vector
 */
PLAY_CGLM_INLINE
void
vec4_subs(vec4 v, float s, vec4 dest)
{
#if defined(__wasm__) && defined(__wasm_simd128__)
    glmm_store(dest, wasm_f32x4_sub(glmm_load(v), wasm_f32x4_splat(s)));
#elif defined( __SSE__ ) || defined( __SSE2__ )
    glmm_store(dest, _mm_sub_ps(glmm_load(v), glmm_set1(s)));
#elif defined(PLAY_CGLM_NEON_FP)
    vst1q_f32(dest, vsubq_f32(vld1q_f32(v), vdupq_n_f32(s)));
#else
    dest[0] = v[0] - s;
    dest[1] = v[1] - s;
    dest[2] = v[2] - s;
    dest[3] = v[3] - s;
#endif
}

/*!
 * @brief multiply two vectors (component-wise multiplication)
 *
 * @param a    vector1
 * @param b    vector2
 * @param dest dest = (a[0] * b[0], a[1] * b[1], a[2] * b[2], a[3] * b[3])
 */
PLAY_CGLM_INLINE
void
vec4_mul(vec4 a, vec4 b, vec4 dest)
{
#if defined(__wasm__) && defined(__wasm_simd128__)
    glmm_store(dest, wasm_f32x4_mul(glmm_load(a), glmm_load(b)));
#elif defined( __SSE__ ) || defined( __SSE2__ )
    glmm_store(dest, _mm_mul_ps(glmm_load(a), glmm_load(b)));
#elif defined(PLAY_CGLM_NEON_FP)
    vst1q_f32(dest, vmulq_f32(vld1q_f32(a), vld1q_f32(b)));
#else
    dest[0] = a[0] * b[0];
    dest[1] = a[1] * b[1];
    dest[2] = a[2] * b[2];
    dest[3] = a[3] * b[3];
#endif
}

/*!
 * @brief multiply/scale vec4 vector with scalar: result = v * s
 *
 * @param[in]  v    vector
 * @param[in]  s    scalar
 * @param[out] dest destination vector
 */
PLAY_CGLM_INLINE
void
vec4_scale(vec4 v, float s, vec4 dest)
{
#if defined(__wasm__) && defined(__wasm_simd128__)
    glmm_store(dest, wasm_f32x4_mul(glmm_load(v), wasm_f32x4_splat(s)));
#elif defined( __SSE__ ) || defined( __SSE2__ )
    glmm_store(dest, _mm_mul_ps(glmm_load(v), glmm_set1(s)));
#elif defined(PLAY_CGLM_NEON_FP)
    vst1q_f32(dest, vmulq_f32(vld1q_f32(v), vdupq_n_f32(s)));
#else
    dest[0] = v[0] * s;
    dest[1] = v[1] * s;
    dest[2] = v[2] * s;
    dest[3] = v[3] * s;
#endif
}

/*!
 * @brief make vec4 vector scale as specified: result = unit(v) * s
 *
 * @param[in]  v    vector
 * @param[in]  s    scalar
 * @param[out] dest destination vector
 */
PLAY_CGLM_INLINE
void
vec4_scale_as(vec4 v, float s, vec4 dest)
{
    float norm;
    norm = vec4_norm(v);

    if (PLAY_CGLM_UNLIKELY(norm < FLT_EPSILON))
    {
        vec4_zero(dest);
        return;
    }

    vec4_scale(v, s / norm, dest);
}

/*!
 * @brief div vector with another component-wise division: d = a / b
 *
 * @param[in]  a    vector 1
 * @param[in]  b    vector 2
 * @param[out] dest result = (a[0]/b[0], a[1]/b[1], a[2]/b[2], a[3]/b[3])
 */
PLAY_CGLM_INLINE
void
vec4_div(vec4 a, vec4 b, vec4 dest)
{
#if defined(PLAY_CGLM_SIMD)
    glmm_store(dest, glmm_div(glmm_load(a), glmm_load(b)));
#else
    dest[0] = a[0] / b[0];
    dest[1] = a[1] / b[1];
    dest[2] = a[2] / b[2];
    dest[3] = a[3] / b[3];
#endif
}

/*!
 * @brief div vec4 vector with scalar: d = v / s
 *
 * @param[in]  v    vector
 * @param[in]  s    scalar
 * @param[out] dest destination vector
 */
PLAY_CGLM_INLINE
void
vec4_divs(vec4 v, float s, vec4 dest)
{
#if defined(PLAY_CGLM_SIMD)
    glmm_store(dest, glmm_div(glmm_load(v), glmm_set1(s)));
#else
    vec4_scale(v, 1.0f / s, dest);
#endif
}

/*!
 * @brief add two vectors and add result to sum
 *
 * it applies += operator so dest must be initialized
 *
 * @param[in]  a    vector 1
 * @param[in]  b    vector 2
 * @param[out] dest dest += (a + b)
 */
PLAY_CGLM_INLINE
void
vec4_addadd(vec4 a, vec4 b, vec4 dest)
{
#if defined(__wasm__) && defined(__wasm_simd128__)
    glmm_store(dest, wasm_f32x4_add(
                   glmm_load(dest),
                   wasm_f32x4_add(glmm_load(a), glmm_load(b))));
#elif defined( __SSE__ ) || defined( __SSE2__ )
    glmm_store(dest, _mm_add_ps(glmm_load(dest),
                                _mm_add_ps(glmm_load(a),
                                           glmm_load(b))));
#elif defined(PLAY_CGLM_NEON_FP)
    vst1q_f32(dest, vaddq_f32(vld1q_f32(dest),
                              vaddq_f32(vld1q_f32(a),
                                        vld1q_f32(b))));
#else
    dest[0] += a[0] + b[0];
    dest[1] += a[1] + b[1];
    dest[2] += a[2] + b[2];
    dest[3] += a[3] + b[3];
#endif
}

/*!
 * @brief sub two vectors and add result to dest
 *
 * it applies += operator so dest must be initialized
 *
 * @param[in]  a    vector 1
 * @param[in]  b    vector 2
 * @param[out] dest dest += (a - b)
 */
PLAY_CGLM_INLINE
void
vec4_subadd(vec4 a, vec4 b, vec4 dest)
{
#if defined(__wasm__) && defined(__wasm_simd128__)
    glmm_store(dest, wasm_f32x4_add(
                   glmm_load(dest),
                   wasm_f32x4_sub(glmm_load(a), glmm_load(b))));
#elif defined( __SSE__ ) || defined( __SSE2__ )
    glmm_store(dest, _mm_add_ps(glmm_load(dest),
                                _mm_sub_ps(glmm_load(a),
                                           glmm_load(b))));
#elif defined(PLAY_CGLM_NEON_FP)
    vst1q_f32(dest, vaddq_f32(vld1q_f32(dest),
                              vsubq_f32(vld1q_f32(a),
                                        vld1q_f32(b))));
#else
    dest[0] += a[0] - b[0];
    dest[1] += a[1] - b[1];
    dest[2] += a[2] - b[2];
    dest[3] += a[3] - b[3];
#endif
}

/*!
 * @brief mul two vectors and add result to dest
 *
 * it applies += operator so dest must be initialized
 *
 * @param[in]  a    vector 1
 * @param[in]  b    vector 2
 * @param[out] dest dest += (a * b)
 */
PLAY_CGLM_INLINE
void
vec4_muladd(vec4 a, vec4 b, vec4 dest)
{
#if defined(PLAY_CGLM_SIMD)
    glmm_store(dest, glmm_fmadd(glmm_load(a), glmm_load(b), glmm_load(dest)));
#else
    dest[0] += a[0] * b[0];
    dest[1] += a[1] * b[1];
    dest[2] += a[2] * b[2];
    dest[3] += a[3] * b[3];
#endif
}

/*!
 * @brief mul vector with scalar and add result to sum
 *
 * it applies += operator so dest must be initialized
 *
 * @param[in]  a    vector
 * @param[in]  s    scalar
 * @param[out] dest dest += (a * b)
 */
PLAY_CGLM_INLINE
void
vec4_muladds(vec4 a, float s, vec4 dest)
{
#if defined(PLAY_CGLM_SIMD)
    glmm_store(dest, glmm_fmadd(glmm_load(a), glmm_set1(s), glmm_load(dest)));
#else
    dest[0] += a[0] * s;
    dest[1] += a[1] * s;
    dest[2] += a[2] * s;
    dest[3] += a[3] * s;
#endif
}

/*!
 * @brief add max of two vectors to result/dest
 *
 * it applies += operator so dest must be initialized
 *
 * @param[in]  a    vector 1
 * @param[in]  b    vector 2
 * @param[out] dest dest += fmax(a, b)
 */
PLAY_CGLM_INLINE
void
vec4_maxadd(vec4 a, vec4 b, vec4 dest)
{
#if defined(__wasm__) && defined(__wasm_simd128__)
    glmm_store(dest, wasm_f32x4_add(glmm_load(dest),
                                    glmm_max(glmm_load(a), glmm_load(b))));
#elif defined( __SSE__ ) || defined( __SSE2__ )
    glmm_store(dest, _mm_add_ps(glmm_load(dest),
                                glmm_max(glmm_load(a), glmm_load(b))));
#elif defined(PLAY_CGLM_NEON_FP)
    glmm_store(dest, vaddq_f32(glmm_load(dest),
                               glmm_max(glmm_load(a), glmm_load(b))));
#else
    dest[0] += fmax(a[0], b[0]);
    dest[1] += fmax(a[1], b[1]);
    dest[2] += fmax(a[2], b[2]);
    dest[3] += fmax(a[3], b[3]);
#endif
}

/*!
 * @brief add min of two vectors to result/dest
 *
 * it applies += operator so dest must be initialized
 *
 * @param[in]  a    vector 1
 * @param[in]  b    vector 2
 * @param[out] dest dest += fmin(a, b)
 */
PLAY_CGLM_INLINE
void
vec4_minadd(vec4 a, vec4 b, vec4 dest)
{
#if defined(__wasm__) && defined(__wasm_simd128__)
    glmm_store(dest, wasm_f32x4_add(glmm_load(dest),
                                    glmm_min(glmm_load(a), glmm_load(b))));
#elif defined( __SSE__ ) || defined( __SSE2__ )
    glmm_store(dest, _mm_add_ps(glmm_load(dest),
                                glmm_min(glmm_load(a), glmm_load(b))));
#elif defined(PLAY_CGLM_NEON_FP)
    glmm_store(dest, vaddq_f32(glmm_load(dest),
                               glmm_min(glmm_load(a), glmm_load(b))));
#else
    dest[0] += fmin(a[0], b[0]);
    dest[1] += fmin(a[1], b[1]);
    dest[2] += fmin(a[2], b[2]);
    dest[3] += fmin(a[3], b[3]);
#endif
}

/*!
 * @brief sub two vectors and sub result to dest
 *
 * it applies -= operator so dest must be initialized
 *
 * @param[in]  a    vector 1
 * @param[in]  b    vector 2
 * @param[out] dest dest -= (a - b)
 */
PLAY_CGLM_INLINE
void
vec4_subsub(vec4 a, vec4 b, vec4 dest)
{
#if defined(__wasm__) && defined(__wasm_simd128__)
    glmm_store(dest, wasm_f32x4_sub(
                   glmm_load(dest),
                   wasm_f32x4_sub(glmm_load(a), glmm_load(b))));
#elif defined( __SSE__ ) || defined( __SSE2__ )
    glmm_store(dest, _mm_sub_ps(glmm_load(dest),
                                _mm_sub_ps(glmm_load(a),
                                           glmm_load(b))));
#elif defined(PLAY_CGLM_NEON_FP)
    vst1q_f32(dest, vsubq_f32(vld1q_f32(dest),
                              vsubq_f32(vld1q_f32(a),
                                        vld1q_f32(b))));
#else
    dest[0] -= a[0] - b[0];
    dest[1] -= a[1] - b[1];
    dest[2] -= a[2] - b[2];
    dest[3] -= a[3] - b[3];
#endif
}

/*!
 * @brief add two vectors and sub result to dest
 *
 * it applies -= operator so dest must be initialized
 *
 * @param[in]  a    vector 1
 * @param[in]  b    vector 2
 * @param[out] dest dest -= (a + b)
 */
PLAY_CGLM_INLINE
void
vec4_addsub(vec4 a, vec4 b, vec4 dest)
{
#if defined(__wasm__) && defined(__wasm_simd128__)
    glmm_store(dest, wasm_f32x4_sub(
                   glmm_load(dest),
                   wasm_f32x4_add(glmm_load(a), glmm_load(b))));
#elif defined( __SSE__ ) || defined( __SSE2__ )
    glmm_store(dest, _mm_sub_ps(glmm_load(dest),
                                _mm_add_ps(glmm_load(a),
                                           glmm_load(b))));
#elif defined(PLAY_CGLM_NEON_FP)
    vst1q_f32(dest, vsubq_f32(vld1q_f32(dest),
                              vaddq_f32(vld1q_f32(a),
                                        vld1q_f32(b))));
#else
    dest[0] -= a[0] + b[0];
    dest[1] -= a[1] + b[1];
    dest[2] -= a[2] + b[2];
    dest[3] -= a[3] + b[3];
#endif
}

/*!
 * @brief mul two vectors and sub result to dest
 *
 * it applies -= operator so dest must be initialized
 *
 * @param[in]  a    vector 1
 * @param[in]  b    vector 2
 * @param[out] dest dest -= (a * b)
 */
PLAY_CGLM_INLINE
void
vec4_mulsub(vec4 a, vec4 b, vec4 dest)
{
#if defined(PLAY_CGLM_SIMD)
    glmm_store(dest, glmm_fnmadd(glmm_load(a), glmm_load(b), glmm_load(dest)));
#else
    dest[0] -= a[0] * b[0];
    dest[1] -= a[1] * b[1];
    dest[2] -= a[2] * b[2];
    dest[3] -= a[3] * b[3];
#endif
}

/*!
 * @brief mul vector with scalar and sub result to dest
 *
 * it applies -= operator so dest must be initialized
 *
 * @param[in]  a    vector
 * @param[in]  s    scalar
 * @param[out] dest dest -= (a * b)
 */
PLAY_CGLM_INLINE
void
vec4_mulsubs(vec4 a, float s, vec4 dest)
{
#if defined(PLAY_CGLM_SIMD)
    glmm_store(dest, glmm_fnmadd(glmm_load(a), glmm_set1(s), glmm_load(dest)));
#else
    dest[0] -= a[0] * s;
    dest[1] -= a[1] * s;
    dest[2] -= a[2] * s;
    dest[3] -= a[3] * s;
#endif
}

/*!
 * @brief sub max of two vectors to dest
 *
 * it applies -= operator so dest must be initialized
 *
 * @param[in]  a    vector 1
 * @param[in]  b    vector 2
 * @param[out] dest dest -= fmax(a, b)
 */
PLAY_CGLM_INLINE
void
vec4_maxsub(vec4 a, vec4 b, vec4 dest)
{
#if defined(__wasm__) && defined(__wasm_simd128__)
    glmm_store(dest, wasm_f32x4_sub(glmm_load(dest),
                                    glmm_max(glmm_load(a), glmm_load(b))));
#elif defined( __SSE__ ) || defined( __SSE2__ )
    glmm_store(dest, _mm_sub_ps(glmm_load(dest),
                                glmm_max(glmm_load(a), glmm_load(b))));
#elif defined(PLAY_CGLM_NEON_FP)
    glmm_store(dest, vsubq_f32(glmm_load(dest),
                               glmm_max(glmm_load(a), glmm_load(b))));
#else
    dest[0] -= fmax(a[0], b[0]);
    dest[1] -= fmax(a[1], b[1]);
    dest[2] -= fmax(a[2], b[2]);
    dest[3] -= fmax(a[3], b[3]);
#endif
}

/*!
 * @brief sub min of two vectors to dest
 *
 * it applies -= operator so dest must be initialized
 *
 * @param[in]  a    vector 1
 * @param[in]  b    vector 2
 * @param[out] dest dest -= fmin(a, b)
 */
PLAY_CGLM_INLINE
void
vec4_minsub(vec4 a, vec4 b, vec4 dest)
{
#if defined(__wasm__) && defined(__wasm_simd128__)
    glmm_store(dest, wasm_f32x4_sub(glmm_load(dest),
                                    glmm_min(glmm_load(a), glmm_load(b))));
#elif defined( __SSE__ ) || defined( __SSE2__ )
    glmm_store(dest, _mm_sub_ps(glmm_load(dest),
                                glmm_min(glmm_load(a), glmm_load(b))));
#elif defined(PLAY_CGLM_NEON_FP)
    glmm_store(dest, vsubq_f32(vld1q_f32(dest),
                               glmm_min(glmm_load(a), glmm_load(b))));
#else
    dest[0] -= fmin(a[0], b[0]);
    dest[1] -= fmin(a[1], b[1]);
    dest[2] -= fmin(a[2], b[2]);
    dest[3] -= fmin(a[3], b[3]);
#endif
}

/*!
 * @brief negate vector components and store result in dest
 *
 * @param[in]  v     vector
 * @param[out] dest  result vector
 */
PLAY_CGLM_INLINE
void
vec4_negate_to(vec4 v, vec4 dest)
{
#if defined(__wasm__) && defined(__wasm_simd128__)
    glmm_store(dest, wasm_f32x4_neg(glmm_load(v)));
#elif defined( __SSE__ ) || defined( __SSE2__ )
    glmm_store(dest, _mm_xor_ps(glmm_load(v), glmm_float32x4_SIGNMASK_NEG));
#elif defined(PLAY_CGLM_NEON_FP)
    vst1q_f32(dest, vnegq_f32(vld1q_f32(v)));
#else
    dest[0] = -v[0];
    dest[1] = -v[1];
    dest[2] = -v[2];
    dest[3] = -v[3];
#endif
}

/*!
 * @brief flip sign of all vec4 members
 *
 * @param[in, out]  v  vector
 */
PLAY_CGLM_INLINE
void
vec4_negate(vec4 v)
{
    vec4_negate_to(v, v);
}

/*!
 * @brief normalize vec4 to dest
 *
 * @param[in]  v    source
 * @param[out] dest destination
 */
PLAY_CGLM_INLINE
void
vec4_normalize_to(vec4 v, vec4 dest)
{
#if defined(__wasm__) && defined(__wasm_simd128__)
    glmm_128 xdot, x0;
    float  dot;

    x0   = glmm_load(v);
    xdot = glmm_vdot(x0, x0);
    /* dot  = _mm_cvtss_f32(xdot); */
    dot  = wasm_f32x4_extract_lane(xdot, 0);

    if (PLAY_CGLM_UNLIKELY(dot < FLT_EPSILON))
    {
        glmm_store(dest, wasm_f32x4_const_splat(0.f));
        return;
    }

    glmm_store(dest, glmm_div(x0, wasm_f32x4_sqrt(xdot)));
#elif defined( __SSE__ ) || defined( __SSE2__ )
    __m128 xdot, x0;
    float  dot;

    x0   = glmm_load(v);
    xdot = glmm_vdot(x0, x0);
    dot  = _mm_cvtss_f32(xdot);

    if (PLAY_CGLM_UNLIKELY(dot < FLT_EPSILON))
    {
        glmm_store(dest, _mm_setzero_ps());
        return;
    }

    glmm_store(dest, glmm_div(x0, _mm_sqrt_ps(xdot)));
#else
    float norm;

    norm = vec4_norm(v);

    if (PLAY_CGLM_UNLIKELY(norm < FLT_EPSILON))
    {
        vec4_zero(dest);
        return;
    }

    vec4_scale(v, 1.0f / norm, dest);
#endif
}

/*!
 * @brief normalize vec4 and store result in same vec
 *
 * @param[in, out] v vector
 */
PLAY_CGLM_INLINE
void
vec4_normalize(vec4 v)
{
    vec4_normalize_to(v, v);
}

/**
 * @brief distance between two vectors
 *
 * @param[in] a vector1
 * @param[in] b vector2
 * @return returns distance
 */
PLAY_CGLM_INLINE
float
vec4_distance(vec4 a, vec4 b)
{
#if defined(__wasm__) && defined(__wasm_simd128__)
    return glmm_norm(wasm_f32x4_sub(glmm_load(a), glmm_load(b)));
#elif defined( __SSE__ ) || defined( __SSE2__ )
    return glmm_norm(_mm_sub_ps(glmm_load(a), glmm_load(b)));
#elif defined(PLAY_CGLM_NEON_FP)
    return glmm_norm(vsubq_f32(glmm_load(a), glmm_load(b)));
#else
    return sqrtf(pow2(a[0] - b[0])
                 + pow2(a[1] - b[1])
                 + pow2(a[2] - b[2])
                 + pow2(a[3] - b[3]));
#endif
}

/**
 * @brief squared distance between two vectors
 *
 * @param[in] a vector1
 * @param[in] b vector2
 * @return returns squared distance
 */
PLAY_CGLM_INLINE
float
vec4_distance2(vec4 a, vec4 b)
{
#if defined(__wasm__) && defined(__wasm_simd128__)
    return glmm_norm2(wasm_f32x4_sub(glmm_load(a), glmm_load(b)));
#elif defined( __SSE__ ) || defined( __SSE2__ )
    return glmm_norm2(_mm_sub_ps(glmm_load(a), glmm_load(b)));
#elif defined(PLAY_CGLM_NEON_FP)
    return glmm_norm2(vsubq_f32(glmm_load(a), glmm_load(b)));
#else
    return pow2(a[0] - b[0])
           + pow2(a[1] - b[1])
           + pow2(a[2] - b[2])
           + pow2(a[3] - b[3]);
#endif
}

/*!
 * @brief max values of vectors
 *
 * @param[in]  a    vector1
 * @param[in]  b    vector2
 * @param[out] dest destination
 */
PLAY_CGLM_INLINE
void
vec4_maxv(vec4 a, vec4 b, vec4 dest)
{
#if defined(PLAY_CGLM_SIMD)
    glmm_store(dest, glmm_max(glmm_load(a), glmm_load(b)));
#else
    dest[0] = fmax(a[0], b[0]);
    dest[1] = fmax(a[1], b[1]);
    dest[2] = fmax(a[2], b[2]);
    dest[3] = fmax(a[3], b[3]);
#endif
}

/*!
 * @brief min values of vectors
 *
 * @param[in]  a    vector1
 * @param[in]  b    vector2
 * @param[out] dest destination
 */
PLAY_CGLM_INLINE
void
vec4_minv(vec4 a, vec4 b, vec4 dest)
{
#if defined(PLAY_CGLM_SIMD)
    glmm_store(dest, glmm_min(glmm_load(a), glmm_load(b)));
#else
    dest[0] = fmin(a[0], b[0]);
    dest[1] = fmin(a[1], b[1]);
    dest[2] = fmin(a[2], b[2]);
    dest[3] = fmin(a[3], b[3]);
#endif
}

/*!
 * @brief clamp vector's individual members between min and max values
 *
 * @param[in, out]  v      vector
 * @param[in]       minVal minimum value
 * @param[in]       maxVal maximum value
 */
PLAY_CGLM_INLINE
void
vec4_clamp(vec4 v, float minVal, float maxVal)
{
#if defined(__wasm__) && defined(__wasm_simd128__)
    glmm_store(v, glmm_min(glmm_max(glmm_load(v), wasm_f32x4_splat(minVal)),
                           wasm_f32x4_splat(maxVal)));
#elif defined( __SSE__ ) || defined( __SSE2__ )
    glmm_store(v, glmm_min(glmm_max(glmm_load(v), glmm_set1(minVal)),
                           glmm_set1(maxVal)));
#elif defined(PLAY_CGLM_NEON_FP)
    glmm_store(v, glmm_min(glmm_max(vld1q_f32(v), vdupq_n_f32(minVal)),
                           vdupq_n_f32(maxVal)));
#else
    v[0] = clamp(v[0], minVal, maxVal);
    v[1] = clamp(v[1], minVal, maxVal);
    v[2] = clamp(v[2], minVal, maxVal);
    v[3] = clamp(v[3], minVal, maxVal);
#endif
}

/*!
 * @brief linear interpolation between two vectors
 *
 * formula:  from + t * (to - from)
 *
 * @param[in]   from from value
 * @param[in]   to   to value
 * @param[in]   t    interpolant (amount)
 * @param[out]  dest destination
 */
PLAY_CGLM_INLINE
void
vec4_lerp(vec4 from, vec4 to, float t, vec4 dest)
{
    vec4 s, v;

    /* from + s * (to - from) */
    vec4_broadcast(t, s);
    vec4_sub(to, from, v);
    vec4_mul(s, v, v);
    vec4_add(from, v, dest);
}

/*!
 * @brief linear interpolation between two vectors (clamped)
 *
 * formula:  from + t * (to - from)
 *
 * @param[in]   from from value
 * @param[in]   to   to value
 * @param[in]   t    interpolant (amount) clamped between 0 and 1
 * @param[out]  dest destination
 */
PLAY_CGLM_INLINE
void
vec4_lerpc(vec4 from, vec4 to, float t, vec4 dest)
{
    vec4_lerp(from, to, clamp_zo(t), dest);
}

/*!
 * @brief linear interpolation between two vectors
 *
 * formula:  from + t * (to - from)
 *
 * @param[in]   from from value
 * @param[in]   to   to value
 * @param[in]   t    interpolant (amount)
 * @param[out]  dest destination
 */
PLAY_CGLM_INLINE
void
vec4_mix(vec4 from, vec4 to, float t, vec4 dest)
{
    vec4_lerp(from, to, t, dest);
}

/*!
 * @brief linear interpolation between two vectors (clamped)
 *
 * formula:  from + t * (to - from)
 *
 * @param[in]   from from value
 * @param[in]   to   to value
 * @param[in]   t    interpolant (amount) clamped between 0 and 1
 * @param[out]  dest destination
 */
PLAY_CGLM_INLINE
void
vec4_mixc(vec4 from, vec4 to, float t, vec4 dest)
{
    vec4_lerpc(from, to, t, dest);
}

/*!
 * @brief threshold function
 *
 * @param[in]   edge    threshold
 * @param[in]   x       value to test against threshold
 * @param[out]  dest    destination
 */
PLAY_CGLM_INLINE
void
vec4_step(vec4 edge, vec4 x, vec4 dest)
{
    dest[0] = step(edge[0], x[0]);
    dest[1] = step(edge[1], x[1]);
    dest[2] = step(edge[2], x[2]);
    dest[3] = step(edge[3], x[3]);
}

/*!
 * @brief threshold function with a smooth transition (unidimensional)
 *
 * @param[in]   edge0   low threshold
 * @param[in]   edge1   high threshold
 * @param[in]   x       value to test against threshold
 * @param[out]  dest    destination
 */
PLAY_CGLM_INLINE
void
vec4_smoothstep_uni(float edge0, float edge1, vec4 x, vec4 dest)
{
    dest[0] = smoothstep(edge0, edge1, x[0]);
    dest[1] = smoothstep(edge0, edge1, x[1]);
    dest[2] = smoothstep(edge0, edge1, x[2]);
    dest[3] = smoothstep(edge0, edge1, x[3]);
}

/*!
 * @brief threshold function with a smooth transition
 *
 * @param[in]   edge0   low threshold
 * @param[in]   edge1   high threshold
 * @param[in]   x       value to test against threshold
 * @param[out]  dest    destination
 */
PLAY_CGLM_INLINE
void
vec4_smoothstep(vec4 edge0, vec4 edge1, vec4 x, vec4 dest)
{
    dest[0] = smoothstep(edge0[0], edge1[0], x[0]);
    dest[1] = smoothstep(edge0[1], edge1[1], x[1]);
    dest[2] = smoothstep(edge0[2], edge1[2], x[2]);
    dest[3] = smoothstep(edge0[3], edge1[3], x[3]);
}

/*!
 * @brief smooth Hermite interpolation between two vectors
 *
 * formula:  t^2 * (3 - 2*t)
 *
 * @param[in]   from    from value
 * @param[in]   to      to value
 * @param[in]   t       interpolant (amount)
 * @param[out]  dest    destination
 */
PLAY_CGLM_INLINE
void
vec4_smoothinterp(vec4 from, vec4 to, float t, vec4 dest)
{
    vec4 s, v;

    /* from + smoothstep * (to - from) */
    vec4_broadcast(smooth(t), s);
    vec4_sub(to, from, v);
    vec4_mul(s, v, v);
    vec4_add(from, v, dest);
}

/*!
 * @brief smooth Hermite interpolation between two vectors (clamped)
 *
 * formula:  t^2 * (3 - 2*t)
 *
 * @param[in]   from    from value
 * @param[in]   to      to value
 * @param[in]   t       interpolant (amount) clamped between 0 and 1
 * @param[out]  dest    destination
 */
PLAY_CGLM_INLINE
void
vec4_smoothinterpc(vec4 from, vec4 to, float t, vec4 dest)
{
    vec4_smoothinterp(from, to, clamp_zo(t), dest);
}

/*!
 * @brief helper to fill vec4 as [S^3, S^2, S, 1]
 *
 * @param[in]   s    parameter
 * @param[out]  dest destination
 */
PLAY_CGLM_INLINE
void
vec4_cubic(float s, vec4 dest)
{
    float ss;

    ss = s * s;

    dest[0] = ss * s;
    dest[1] = ss;
    dest[2] = s;
    dest[3] = 1.0f;
}

/*!
 * @brief swizzle vector components
 *
 * you can use existing masks e.g. PLAY_CGLM_XXXX, PLAY_CGLM_WZYX
 *
 * @param[in]  v    source
 * @param[in]  mask mask
 * @param[out] dest destination
 */
PLAY_CGLM_INLINE
void
vec4_swizzle(vec4 v, int mask, vec4 dest)
{
    vec4 t;

    t[0] = v[(mask & (3 << 0))];
    t[1] = v[(mask & (3 << 2)) >> 2];
    t[2] = v[(mask & (3 << 4)) >> 4];
    t[3] = v[(mask & (3 << 6)) >> 6];

    vec4_copy(t, dest);
}

/*!
 * @brief Create four dimensional vector from pointer
 *
 * @param[in]  src  pointer to an array of floats
 * @param[out] dest destination vector
 */
PLAY_CGLM_INLINE
void
vec4_make(const float * __restrict src, vec4 dest)
{
    dest[0] = src[0];
    dest[1] = src[1];
    dest[2] = src[2];
    dest[3] = src[3];
}

/*!
 * @brief reflection vector using an incident ray and a surface normal
 *
 * @param[in]  v    incident vector
 * @param[in]  n    normalized normal vector
 * @param[out] dest destination vector for the reflection result
 */
PLAY_CGLM_INLINE
void
vec4_reflect(vec4 v, vec4 n, vec4 dest)
{
    vec4 temp;

    /* TODO: direct simd touch */
    vec4_scale(n, 2.0f * vec4_dot(v, n), temp);
    vec4_sub(v, temp, dest);

    dest[3] = v[3];
}

/*!
 * @brief computes refraction vector for an incident vector and a surface normal.
 *
 * calculates the refraction vector based on Snell's law. If total internal reflection
 * occurs (angle too great given eta), dest is set to zero and returns false.
 * Otherwise, computes refraction vector, stores it in dest, and returns true.
 *
 * this implementation does not explicitly preserve the 'w' component of the
 * incident vector 'I' in the output 'dest', users requiring the preservation of
 * the 'w' component should manually adjust 'dest' after calling this function.
 *
 * @param[in]  v    normalized incident vector
 * @param[in]  n    normalized normal vector
 * @param[in]  eta  ratio of indices of refraction (incident/transmitted)
 * @param[out] dest refraction vector if refraction occurs; zero vector otherwise
 *
 * @returns true if refraction occurs; false if total internal reflection occurs.
 */
PLAY_CGLM_INLINE
bool
vec4_refract(vec4 v, vec4 n, float eta, vec4 dest)
{
    float ndi, eni, k;

    ndi = vec4_dot(n, v);
    eni = eta * ndi;
    k   = 1.0f - eta * eta + eni * eni;

    if (k < 0.0f)
    {
        vec4_zero(dest);
        return false;
    }

    vec4_scale(v, eta, dest);
    vec4_mulsubs(n, eni + sqrtf(k), dest);
    return true;
}

#endif /* cvec4_h */

/*** End of inlined file: vec4.h ***/

/* DEPRECATED! use _copy, _ucopy versions */
#define vec3_dup(v, dest)         vec3_copy(v, dest)
#define vec3_flipsign(v)          vec3_negate(v)
#define vec3_flipsign_to(v, dest) vec3_negate_to(v, dest)
#define vec3_inv(v)               vec3_negate(v)
#define vec3_inv_to(v, dest)      vec3_negate_to(v, dest)
#define vec3_mulv(a, b, d)        vec3_mul(a, b, d)
#define vec3_step_uni(edge, x, dest) vec3_steps(edge, x, dest)

#define PLAY_CGLM_VEC3_ONE_INIT   {1.0f, 1.0f, 1.0f}
#define PLAY_CGLM_VEC3_ZERO_INIT  {0.0f, 0.0f, 0.0f}

#define PLAY_CGLM_VEC3_ONE  ((vec3)PLAY_CGLM_VEC3_ONE_INIT)
#define PLAY_CGLM_VEC3_ZERO ((vec3)PLAY_CGLM_VEC3_ZERO_INIT)

#define PLAY_CGLM_YUP       ((vec3){0.0f,  1.0f,  0.0f})
#define PLAY_CGLM_ZUP       ((vec3){0.0f,  0.0f,  1.0f})
#define PLAY_CGLM_XUP       ((vec3){1.0f,  0.0f,  0.0f})
#define PLAY_CGLM_FORWARD   ((vec3){0.0f,  0.0f, -1.0f})

#define PLAY_CGLM_XXX PLAY_CGLM_SHUFFLE3(0, 0, 0)
#define PLAY_CGLM_YYY PLAY_CGLM_SHUFFLE3(1, 1, 1)
#define PLAY_CGLM_ZZZ PLAY_CGLM_SHUFFLE3(2, 2, 2)
#define PLAY_CGLM_ZYX PLAY_CGLM_SHUFFLE3(0, 1, 2)

/*!
 * @brief init vec3 using vec4
 *
 * @param[in]  v4   vector4
 * @param[out] dest destination
 */
PLAY_CGLM_INLINE
void
vec3_new(vec4 v4, vec3 dest)
{
    dest[0] = v4[0];
    dest[1] = v4[1];
    dest[2] = v4[2];
}

/*!
 * @brief copy all members of [a] to [dest]
 *
 * @param[in]  a    source
 * @param[out] dest destination
 */
PLAY_CGLM_INLINE
void
vec3_copy(vec3 a, vec3 dest)
{
    dest[0] = a[0];
    dest[1] = a[1];
    dest[2] = a[2];
}

/*!
 * @brief make vector zero
 *
 * @param[in, out]  v vector
 */
PLAY_CGLM_INLINE
void
vec3_zero(vec3 v)
{
    v[0] = v[1] = v[2] = 0.0f;
}

/*!
 * @brief make vector one
 *
 * @param[in, out]  v vector
 */
PLAY_CGLM_INLINE
void
vec3_one(vec3 v)
{
    v[0] = v[1] = v[2] = 1.0f;
}

/*!
 * @brief vec3 dot product
 *
 * @param[in] a vector1
 * @param[in] b vector2
 *
 * @return dot product
 */
PLAY_CGLM_INLINE
float
vec3_dot(vec3 a, vec3 b)
{
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
}

/*!
 * @brief norm * norm (magnitude) of vec
 *
 * we can use this func instead of calling norm * norm, because it would call
 * sqrtf function twice but with this func we can avoid func call, maybe this is
 * not good name for this func
 *
 * @param[in] v vector
 *
 * @return norm * norm
 */
PLAY_CGLM_INLINE
float
vec3_norm2(vec3 v)
{
    return vec3_dot(v, v);
}

/*!
 * @brief euclidean norm (magnitude), also called L2 norm
 *        this will give magnitude of vector in euclidean space
 *
 * @param[in] v vector
 *
 * @return norm
 */
PLAY_CGLM_INLINE
float
vec3_norm(vec3 v)
{
    return sqrtf(vec3_norm2(v));
}

/*!
 * @brief L1 norm of vec3
 * Also known as Manhattan Distance or Taxicab norm.
 * L1 Norm is the sum of the magnitudes of the vectors in a space.
 * It is calculated as the sum of the absolute values of the vector components.
 * In this norm, all the components of the vector are weighted equally.
 *
 * This computes:
 * R = |v[0]| + |v[1]| + |v[2]|
 *
 * @param[in] v vector
 *
 * @return L1 norm
 */
PLAY_CGLM_INLINE
float
vec3_norm_one(vec3 v)
{
    vec3 t;
    vec3_abs(v, t);
    return vec3_hadd(t);
}

/*!
 * @brief infinity norm of vec3
 * Also known as Maximum norm.
 * Infinity Norm is the largest magnitude among each element of a vector.
 * It is calculated as the maximum of the absolute values of the vector components.
 *
 * This computes:
 * inf norm = fmax(|v[0]|, |v[1]|, |v[2]|)
 *
 * @param[in] v vector
 *
 * @return infinity norm
 */
PLAY_CGLM_INLINE
float
vec3_norm_inf(vec3 v)
{
    vec3 t;
    vec3_abs(v, t);
    return vec3_max(t);
}

/*!
 * @brief add a vector to b vector store result in dest
 *
 * @param[in]  a    vector1
 * @param[in]  b    vector2
 * @param[out] dest destination vector
 */
PLAY_CGLM_INLINE
void
vec3_add(vec3 a, vec3 b, vec3 dest)
{
    dest[0] = a[0] + b[0];
    dest[1] = a[1] + b[1];
    dest[2] = a[2] + b[2];
}

/*!
 * @brief add scalar to v vector store result in dest (d = v + s)
 *
 * @param[in]  v    vector
 * @param[in]  s    scalar
 * @param[out] dest destination vector
 */
PLAY_CGLM_INLINE
void
vec3_adds(vec3 v, float s, vec3 dest)
{
    dest[0] = v[0] + s;
    dest[1] = v[1] + s;
    dest[2] = v[2] + s;
}

/*!
 * @brief subtract b vector from a vector store result in dest
 *
 * @param[in]  a    vector1
 * @param[in]  b    vector2
 * @param[out] dest destination vector
 */
PLAY_CGLM_INLINE
void
vec3_sub(vec3 a, vec3 b, vec3 dest)
{
    dest[0] = a[0] - b[0];
    dest[1] = a[1] - b[1];
    dest[2] = a[2] - b[2];
}

/*!
 * @brief subtract scalar from v vector store result in dest (d = v - s)
 *
 * @param[in]  v    vector
 * @param[in]  s    scalar
 * @param[out] dest destination vector
 */
PLAY_CGLM_INLINE
void
vec3_subs(vec3 v, float s, vec3 dest)
{
    dest[0] = v[0] - s;
    dest[1] = v[1] - s;
    dest[2] = v[2] - s;
}

/*!
 * @brief multiply two vectors (component-wise multiplication)
 *
 * @param a    vector1
 * @param b    vector2
 * @param dest v3 = (a[0] * b[0], a[1] * b[1], a[2] * b[2])
 */
PLAY_CGLM_INLINE
void
vec3_mul(vec3 a, vec3 b, vec3 dest)
{
    dest[0] = a[0] * b[0];
    dest[1] = a[1] * b[1];
    dest[2] = a[2] * b[2];
}

/*!
 * @brief multiply/scale vec3 vector with scalar: result = v * s
 *
 * @param[in]  v    vector
 * @param[in]  s    scalar
 * @param[out] dest destination vector
 */
PLAY_CGLM_INLINE
void
vec3_scale(vec3 v, float s, vec3 dest)
{
    dest[0] = v[0] * s;
    dest[1] = v[1] * s;
    dest[2] = v[2] * s;
}

/*!
 * @brief make vec3 vector scale as specified: result = unit(v) * s
 *
 * @param[in]  v    vector
 * @param[in]  s    scalar
 * @param[out] dest destination vector
 */
PLAY_CGLM_INLINE
void
vec3_scale_as(vec3 v, float s, vec3 dest)
{
    float norm;
    norm = vec3_norm(v);

    if (PLAY_CGLM_UNLIKELY(norm < FLT_EPSILON))
    {
        vec3_zero(dest);
        return;
    }

    vec3_scale(v, s / norm, dest);
}

/*!
 * @brief div vector with another component-wise division: d = a / b
 *
 * @param[in]  a    vector 1
 * @param[in]  b    vector 2
 * @param[out] dest result = (a[0]/b[0], a[1]/b[1], a[2]/b[2])
 */
PLAY_CGLM_INLINE
void
vec3_div(vec3 a, vec3 b, vec3 dest)
{
    dest[0] = a[0] / b[0];
    dest[1] = a[1] / b[1];
    dest[2] = a[2] / b[2];
}

/*!
 * @brief div vector with scalar: d = v / s
 *
 * @param[in]  v    vector
 * @param[in]  s    scalar
 * @param[out] dest result = (a[0]/s, a[1]/s, a[2]/s)
 */
PLAY_CGLM_INLINE
void
vec3_divs(vec3 v, float s, vec3 dest)
{
    dest[0] = v[0] / s;
    dest[1] = v[1] / s;
    dest[2] = v[2] / s;
}

/*!
 * @brief add two vectors and add result to sum
 *
 * it applies += operator so dest must be initialized
 *
 * @param[in]  a    vector 1
 * @param[in]  b    vector 2
 * @param[out] dest dest += (a + b)
 */
PLAY_CGLM_INLINE
void
vec3_addadd(vec3 a, vec3 b, vec3 dest)
{
    dest[0] += a[0] + b[0];
    dest[1] += a[1] + b[1];
    dest[2] += a[2] + b[2];
}

/*!
 * @brief sub two vectors and add result to dest
 *
 * it applies += operator so dest must be initialized
 *
 * @param[in]  a    vector 1
 * @param[in]  b    vector 2
 * @param[out] dest dest += (a + b)
 */
PLAY_CGLM_INLINE
void
vec3_subadd(vec3 a, vec3 b, vec3 dest)
{
    dest[0] += a[0] - b[0];
    dest[1] += a[1] - b[1];
    dest[2] += a[2] - b[2];
}

/*!
 * @brief mul two vectors and add result to dest
 *
 * it applies += operator so dest must be initialized
 *
 * @param[in]  a    vector 1
 * @param[in]  b    vector 2
 * @param[out] dest dest += (a * b)
 */
PLAY_CGLM_INLINE
void
vec3_muladd(vec3 a, vec3 b, vec3 dest)
{
    dest[0] += a[0] * b[0];
    dest[1] += a[1] * b[1];
    dest[2] += a[2] * b[2];
}

/*!
 * @brief mul vector with scalar and add result to sum
 *
 * it applies += operator so dest must be initialized
 *
 * @param[in]  a    vector
 * @param[in]  s    scalar
 * @param[out] dest dest += (a * b)
 */
PLAY_CGLM_INLINE
void
vec3_muladds(vec3 a, float s, vec3 dest)
{
    dest[0] += a[0] * s;
    dest[1] += a[1] * s;
    dest[2] += a[2] * s;
}

/*!
 * @brief add max of two vectors to result/dest
 *
 * it applies += operator so dest must be initialized
 *
 * @param[in]  a    vector 1
 * @param[in]  b    vector 2
 * @param[out] dest dest += fmax(a, b)
 */
PLAY_CGLM_INLINE
void
vec3_maxadd(vec3 a, vec3 b, vec3 dest)
{
    dest[0] += fmax(a[0], b[0]);
    dest[1] += fmax(a[1], b[1]);
    dest[2] += fmax(a[2], b[2]);
}

/*!
 * @brief add min of two vectors to result/dest
 *
 * it applies += operator so dest must be initialized
 *
 * @param[in]  a    vector 1
 * @param[in]  b    vector 2
 * @param[out] dest dest += fmin(a, b)
 */
PLAY_CGLM_INLINE
void
vec3_minadd(vec3 a, vec3 b, vec3 dest)
{
    dest[0] += fmin(a[0], b[0]);
    dest[1] += fmin(a[1], b[1]);
    dest[2] += fmin(a[2], b[2]);
}

/*!
 * @brief sub two vectors and sub result to dest
 *
 * it applies += operator so dest must be initialized
 *
 * @param[in]  a    vector 1
 * @param[in]  b    vector 2
 * @param[out] dest dest -= (a - b)
 */
PLAY_CGLM_INLINE
void
vec3_subsub(vec3 a, vec3 b, vec3 dest)
{
    dest[0] -= a[0] - b[0];
    dest[1] -= a[1] - b[1];
    dest[2] -= a[2] - b[2];
}

/*!
 * @brief add two vectors and sub result to dest
 *
 * it applies += operator so dest must be initialized
 *
 * @param[in]  a    vector 1
 * @param[in]  b    vector 2
 * @param[out] dest dest -= (a + b)
 */
PLAY_CGLM_INLINE
void
vec3_addsub(vec3 a, vec3 b, vec3 dest)
{
    dest[0] -= a[0] + b[0];
    dest[1] -= a[1] + b[1];
    dest[2] -= a[2] + b[2];
}

/*!
 * @brief mul two vectors and sub result to dest
 *
 * it applies -= operator so dest must be initialized
 *
 * @param[in]  a    vector 1
 * @param[in]  b    vector 2
 * @param[out] dest dest -= (a * b)
 */
PLAY_CGLM_INLINE
void
vec3_mulsub(vec3 a, vec3 b, vec3 dest)
{
    dest[0] -= a[0] * b[0];
    dest[1] -= a[1] * b[1];
    dest[2] -= a[2] * b[2];
}

/*!
 * @brief mul vector with scalar and sub result to dest
 *
 * it applies -= operator so dest must be initialized
 *
 * @param[in]  a    vector
 * @param[in]  s    scalar
 * @param[out] dest dest -= (a * b)
 */
PLAY_CGLM_INLINE
void
vec3_mulsubs(vec3 a, float s, vec3 dest)
{
    dest[0] -= a[0] * s;
    dest[1] -= a[1] * s;
    dest[2] -= a[2] * s;
}

/*!
 * @brief sub max of two vectors to result/dest
 *
 * it applies -= operator so dest must be initialized
 *
 * @param[in]  a    vector 1
 * @param[in]  b    vector 2
 * @param[out] dest dest -= fmax(a, b)
 */
PLAY_CGLM_INLINE
void
vec3_maxsub(vec3 a, vec3 b, vec3 dest)
{
    dest[0] -= fmax(a[0], b[0]);
    dest[1] -= fmax(a[1], b[1]);
    dest[2] -= fmax(a[2], b[2]);
}

/*!
 * @brief sub min of two vectors to result/dest
 *
 * it applies -= operator so dest must be initialized
 *
 * @param[in]  a    vector 1
 * @param[in]  b    vector 2
 * @param[out] dest dest -= fmin(a, b)
 */
PLAY_CGLM_INLINE
void
vec3_minsub(vec3 a, vec3 b, vec3 dest)
{
    dest[0] -= fmin(a[0], b[0]);
    dest[1] -= fmin(a[1], b[1]);
    dest[2] -= fmin(a[2], b[2]);
}

/*!
 * @brief negate vector components and store result in dest
 *
 * @param[in]   v     vector
 * @param[out]  dest  result vector
 */
PLAY_CGLM_INLINE
void
vec3_negate_to(vec3 v, vec3 dest)
{
    dest[0] = -v[0];
    dest[1] = -v[1];
    dest[2] = -v[2];
}

/*!
 * @brief negate vector components
 *
 * @param[in, out]  v  vector
 */
PLAY_CGLM_INLINE
void
vec3_negate(vec3 v)
{
    vec3_negate_to(v, v);
}

/*!
 * @brief normalize vec3 and store result in same vec
 *
 * @param[in, out] v vector
 */
PLAY_CGLM_INLINE
void
vec3_normalize(vec3 v)
{
    float norm;

    norm = vec3_norm(v);

    if (PLAY_CGLM_UNLIKELY(norm < FLT_EPSILON))
    {
        v[0] = v[1] = v[2] = 0.0f;
        return;
    }

    vec3_scale(v, 1.0f / norm, v);
}

/*!
 * @brief normalize vec3 to dest
 *
 * @param[in]  v    source
 * @param[out] dest destination
 */
PLAY_CGLM_INLINE
void
vec3_normalize_to(vec3 v, vec3 dest)
{
    float norm;

    norm = vec3_norm(v);

    if (PLAY_CGLM_UNLIKELY(norm < FLT_EPSILON))
    {
        vec3_zero(dest);
        return;
    }

    vec3_scale(v, 1.0f / norm, dest);
}

/*!
 * @brief cross product of two vector (RH)
 *
 * @param[in]  a    vector 1
 * @param[in]  b    vector 2
 * @param[out] dest destination
 */
PLAY_CGLM_INLINE
void
vec3_cross(vec3 a, vec3 b, vec3 dest)
{
    vec3 c;
    /* (u2.v3 - u3.v2, u3.v1 - u1.v3, u1.v2 - u2.v1) */
    c[0] = a[1] * b[2] - a[2] * b[1];
    c[1] = a[2] * b[0] - a[0] * b[2];
    c[2] = a[0] * b[1] - a[1] * b[0];
    vec3_copy(c, dest);
}

/*!
 * @brief cross product of two vector (RH) and normalize the result
 *
 * @param[in]  a    vector 1
 * @param[in]  b    vector 2
 * @param[out] dest destination
 */
PLAY_CGLM_INLINE
void
vec3_crossn(vec3 a, vec3 b, vec3 dest)
{
    vec3_cross(a, b, dest);
    vec3_normalize(dest);
}

/*!
 * @brief angle between two vector
 *
 * @param[in] a  vector1
 * @param[in] b  vector2
 *
 * @return angle as radians
 */
PLAY_CGLM_INLINE
float
vec3_angle(vec3 a, vec3 b)
{
    float norm, dot;

    /* maybe compiler generate approximation instruction (rcp) */
    norm = 1.0f / (vec3_norm(a) * vec3_norm(b));
    dot  = vec3_dot(a, b) * norm;

    if (dot > 1.0f)
        return 0.0f;
    else if (dot < -1.0f)
        return PLAY_CGLM_PI;

    return acosf(dot);
}

/*!
 * @brief rotate vec3 around axis by angle using Rodrigues' rotation formula
 *
 * @param[in, out] v     vector
 * @param[in]      axis  axis vector (must be unit vector)
 * @param[in]      angle angle by radians
 */
PLAY_CGLM_INLINE
void
vec3_rotate(vec3 v, float angle, vec3 axis)
{
    vec3   v1, v2, k;
    float  c, s;

    c = cosf(angle);
    s = sinf(angle);

    vec3_normalize_to(axis, k);

    /* Right Hand, Rodrigues' rotation formula:
          v = v*cos(t) + (kxv)sin(t) + k*(k.v)(1 - cos(t))
     */
    vec3_scale(v, c, v1);

    vec3_cross(k, v, v2);
    vec3_scale(v2, s, v2);

    vec3_add(v1, v2, v1);

    vec3_scale(k, vec3_dot(k, v) * (1.0f - c), v2);
    vec3_add(v1, v2, v);
}

/*!
 * @brief apply rotation matrix to vector
 *
 *  matrix format should be (no perspective):
 *   a  b  c  x
 *   e  f  g  y
 *   i  j  k  z
 *   0  0  0  w
 *
 * @param[in]  m    affine matrix or rot matrix
 * @param[in]  v    vector
 * @param[out] dest rotated vector
 */
PLAY_CGLM_INLINE
void
vec3_rotate_m4(mat4 m, vec3 v, vec3 dest)
{
    vec4 x, y, z, res;

    vec4_normalize_to(m[0], x);
    vec4_normalize_to(m[1], y);
    vec4_normalize_to(m[2], z);

    vec4_scale(x,   v[0], res);
    vec4_muladds(y, v[1], res);
    vec4_muladds(z, v[2], res);

    vec3_new(res, dest);
}

/*!
 * @brief apply rotation matrix to vector
 *
 * @param[in]  m    affine matrix or rot matrix
 * @param[in]  v    vector
 * @param[out] dest rotated vector
 */
PLAY_CGLM_INLINE
void
vec3_rotate_m3(mat3 m, vec3 v, vec3 dest)
{
    vec4 res, x, y, z;

    vec4_new(m[0], 0.0f, x);
    vec4_new(m[1], 0.0f, y);
    vec4_new(m[2], 0.0f, z);

    vec4_normalize(x);
    vec4_normalize(y);
    vec4_normalize(z);

    vec4_scale(x,   v[0], res);
    vec4_muladds(y, v[1], res);
    vec4_muladds(z, v[2], res);

    vec3_new(res, dest);
}

/*!
 * @brief project a vector onto b vector
 *
 * @param[in]  a    vector1
 * @param[in]  b    vector2
 * @param[out] dest projected vector
 */
PLAY_CGLM_INLINE
void
vec3_proj(vec3 a, vec3 b, vec3 dest)
{
    vec3_scale(b,
               vec3_dot(a, b) / vec3_norm2(b),
               dest);
}

/**
 * @brief find center point of two vector
 *
 * @param[in]  a    vector1
 * @param[in]  b    vector2
 * @param[out] dest center point
 */
PLAY_CGLM_INLINE
void
vec3_center(vec3 a, vec3 b, vec3 dest)
{
    vec3_add(a, b, dest);
    vec3_scale(dest, 0.5f, dest);
}

/**
 * @brief squared distance between two vectors
 *
 * @param[in] a vector1
 * @param[in] b vector2
 * @return returns squared distance (distance * distance)
 */
PLAY_CGLM_INLINE
float
vec3_distance2(vec3 a, vec3 b)
{
    return pow2(a[0] - b[0])
           + pow2(a[1] - b[1])
           + pow2(a[2] - b[2]);
}

/**
 * @brief distance between two vectors
 *
 * @param[in] a vector1
 * @param[in] b vector2
 * @return returns distance
 */
PLAY_CGLM_INLINE
float
vec3_distance(vec3 a, vec3 b)
{
    return sqrtf(vec3_distance2(a, b));
}

/*!
 * @brief max values of vectors
 *
 * @param[in]  a    vector1
 * @param[in]  b    vector2
 * @param[out] dest destination
 */
PLAY_CGLM_INLINE
void
vec3_maxv(vec3 a, vec3 b, vec3 dest)
{
    dest[0] = fmax(a[0], b[0]);
    dest[1] = fmax(a[1], b[1]);
    dest[2] = fmax(a[2], b[2]);
}

/*!
 * @brief min values of vectors
 *
 * @param[in]  a    vector1
 * @param[in]  b    vector2
 * @param[out] dest destination
 */
PLAY_CGLM_INLINE
void
vec3_minv(vec3 a, vec3 b, vec3 dest)
{
    dest[0] = fmin(a[0], b[0]);
    dest[1] = fmin(a[1], b[1]);
    dest[2] = fmin(a[2], b[2]);
}

/*!
 * @brief possible orthogonal/perpendicular vector
 *
 * @param[in]  v    vector
 * @param[out] dest orthogonal/perpendicular vector
 */
PLAY_CGLM_INLINE
void
vec3_ortho(vec3 v, vec3 dest)
{
    float ignore;
    float f      = modff(fabsf(v[0]) + 0.5f, &ignore);
    vec3  result = {-v[1], v[0] - f * v[2], f * v[1]};
    vec3_copy(result, dest);
}

/*!
 * @brief clamp vector's individual members between min and max values
 *
 * @param[in, out]  v      vector
 * @param[in]       minVal minimum value
 * @param[in]       maxVal maximum value
 */
PLAY_CGLM_INLINE
void
vec3_clamp(vec3 v, float minVal, float maxVal)
{
    v[0] = clamp(v[0], minVal, maxVal);
    v[1] = clamp(v[1], minVal, maxVal);
    v[2] = clamp(v[2], minVal, maxVal);
}

/*!
 * @brief linear interpolation between two vectors
 *
 * formula:  from + s * (to - from)
 *
 * @param[in]   from from value
 * @param[in]   to   to value
 * @param[in]   t    interpolant (amount)
 * @param[out]  dest destination
 */
PLAY_CGLM_INLINE
void
vec3_lerp(vec3 from, vec3 to, float t, vec3 dest)
{
    vec3 s, v;

    /* from + s * (to - from) */
    vec3_broadcast(t, s);
    vec3_sub(to, from, v);
    vec3_mul(s, v, v);
    vec3_add(from, v, dest);
}

/*!
 * @brief linear interpolation between two vectors (clamped)
 *
 * formula:  from + s * (to - from)
 *
 * @param[in]   from from value
 * @param[in]   to   to value
 * @param[in]   t    interpolant (amount) clamped between 0 and 1
 * @param[out]  dest destination
 */
PLAY_CGLM_INLINE
void
vec3_lerpc(vec3 from, vec3 to, float t, vec3 dest)
{
    vec3_lerp(from, to, clamp_zo(t), dest);
}

/*!
 * @brief linear interpolation between two vectors
 *
 * formula:  from + s * (to - from)
 *
 * @param[in]   from from value
 * @param[in]   to   to value
 * @param[in]   t    interpolant (amount)
 * @param[out]  dest destination
 */
PLAY_CGLM_INLINE
void
vec3_mix(vec3 from, vec3 to, float t, vec3 dest)
{
    vec3_lerp(from, to, t, dest);
}

/*!
 * @brief linear interpolation between two vectors (clamped)
 *
 * formula:  from + s * (to - from)
 *
 * @param[in]   from from value
 * @param[in]   to   to value
 * @param[in]   t    interpolant (amount) clamped between 0 and 1
 * @param[out]  dest destination
 */
PLAY_CGLM_INLINE
void
vec3_mixc(vec3 from, vec3 to, float t, vec3 dest)
{
    vec3_lerpc(from, to, t, dest);
}

/*!
 * @brief threshold function
 *
 * @param[in]   edge    threshold
 * @param[in]   x       value to test against threshold
 * @param[out]  dest    destination
 */
PLAY_CGLM_INLINE
void
vec3_step(vec3 edge, vec3 x, vec3 dest)
{
    dest[0] = step(edge[0], x[0]);
    dest[1] = step(edge[1], x[1]);
    dest[2] = step(edge[2], x[2]);
}

/*!
 * @brief threshold function with a smooth transition (unidimensional)
 *
 * @param[in]   edge0   low threshold
 * @param[in]   edge1   high threshold
 * @param[in]   x       value to test against threshold
 * @param[out]  dest    destination
 */
PLAY_CGLM_INLINE
void
vec3_smoothstep_uni(float edge0, float edge1, vec3 x, vec3 dest)
{
    dest[0] = smoothstep(edge0, edge1, x[0]);
    dest[1] = smoothstep(edge0, edge1, x[1]);
    dest[2] = smoothstep(edge0, edge1, x[2]);
}

/*!
 * @brief threshold function with a smooth transition
 *
 * @param[in]   edge0   low threshold
 * @param[in]   edge1   high threshold
 * @param[in]   x       value to test against threshold
 * @param[out]  dest    destination
 */
PLAY_CGLM_INLINE
void
vec3_smoothstep(vec3 edge0, vec3 edge1, vec3 x, vec3 dest)
{
    dest[0] = smoothstep(edge0[0], edge1[0], x[0]);
    dest[1] = smoothstep(edge0[1], edge1[1], x[1]);
    dest[2] = smoothstep(edge0[2], edge1[2], x[2]);
}

/*!
 * @brief smooth Hermite interpolation between two vectors
 *
 * formula:  from + s * (to - from)
 *
 * @param[in]   from from value
 * @param[in]   to   to value
 * @param[in]   t    interpolant (amount)
 * @param[out]  dest destination
 */
PLAY_CGLM_INLINE
void
vec3_smoothinterp(vec3 from, vec3 to, float t, vec3 dest)
{
    vec3 s, v;

    /* from + s * (to - from) */
    vec3_broadcast(smooth(t), s);
    vec3_sub(to, from, v);
    vec3_mul(s, v, v);
    vec3_add(from, v, dest);
}

/*!
 * @brief smooth Hermite interpolation between two vectors (clamped)
 *
 * formula:  from + s * (to - from)
 *
 * @param[in]   from from value
 * @param[in]   to   to value
 * @param[in]   t    interpolant (amount) clamped between 0 and 1
 * @param[out]  dest destination
 */
PLAY_CGLM_INLINE
void
vec3_smoothinterpc(vec3 from, vec3 to, float t, vec3 dest)
{
    vec3_smoothinterp(from, to, clamp_zo(t), dest);
}

/*!
 * @brief swizzle vector components
 *
 * you can use existing masks e.g. PLAY_CGLM_XXX, PLAY_CGLM_ZYX
 *
 * @param[in]  v    source
 * @param[in]  mask mask
 * @param[out] dest destination
 */
PLAY_CGLM_INLINE
void
vec3_swizzle(vec3 v, int mask, vec3 dest)
{
    vec3 t;

    t[0] = v[(mask & (3 << 0))];
    t[1] = v[(mask & (3 << 2)) >> 2];
    t[2] = v[(mask & (3 << 4)) >> 4];

    vec3_copy(t, dest);
}

/*!
 * @brief vec3 cross product
 *
 * this is just convenient wrapper
 *
 * @param[in]  a source 1
 * @param[in]  b source 2
 * @param[out] d destination
 */
PLAY_CGLM_INLINE
void
cross(vec3 a, vec3 b, vec3 d)
{
    vec3_cross(a, b, d);
}

/*!
 * @brief vec3 dot product
 *
 * this is just convenient wrapper
 *
 * @param[in] a vector1
 * @param[in] b vector2
 *
 * @return dot product
 */
PLAY_CGLM_INLINE
float
dot(vec3 a, vec3 b)
{
    return vec3_dot(a, b);
}

/*!
 * @brief normalize vec3 and store result in same vec
 *
 * this is just convenient wrapper
 *
 * @param[in, out] v vector
 */
PLAY_CGLM_INLINE
void
normalize(vec3 v)
{
    vec3_normalize(v);
}

/*!
 * @brief normalize vec3 to dest
 *
 * this is just convenient wrapper
 *
 * @param[in]  v    source
 * @param[out] dest destination
 */
PLAY_CGLM_INLINE
void
normalize_to(vec3 v, vec3 dest)
{
    vec3_normalize_to(v, dest);
}

/*!
 * @brief Create three dimensional vector from pointer
 *
 * @param[in]  src  pointer to an array of floats
 * @param[out] dest destination vector
 */
PLAY_CGLM_INLINE
void
vec3_make(const float * __restrict src, vec3 dest)
{
    dest[0] = src[0];
    dest[1] = src[1];
    dest[2] = src[2];
}

/*!
 * @brief a vector pointing in the same direction as another
 *
 * orients a vector to point away from a surface as defined by its normal
 *
 * @param[in] n      vector to orient
 * @param[in] v      incident vector
 * @param[in] nref   reference vector
 * @param[out] dest  oriented vector, pointing away from the surface
 */
PLAY_CGLM_INLINE
void
vec3_faceforward(vec3 n, vec3 v, vec3 nref, vec3 dest)
{
    if (vec3_dot(v, nref) < 0.0f)
    {
        /* N is facing away from I */
        vec3_copy(n, dest);
    }
    else
    {
        /* N is facing towards I, negate it */
        vec3_negate_to(n, dest);
    }
}

/*!
 * @brief reflection vector using an incident ray and a surface normal
 *
 * @param[in]  v    incident vector
 * @param[in]  n    normalized normal vector
 * @param[out] dest reflection result
 */
PLAY_CGLM_INLINE
void
vec3_reflect(vec3 v, vec3 n, vec3 dest)
{
    vec3 temp;
    vec3_scale(n, 2.0f * vec3_dot(v, n), temp);
    vec3_sub(v, temp, dest);
}

/*!
 * @brief computes refraction vector for an incident vector and a surface normal.
 *
 * calculates the refraction vector based on Snell's law. If total internal reflection
 * occurs (angle too great given eta), dest is set to zero and returns false.
 * Otherwise, computes refraction vector, stores it in dest, and returns true.
 *
 * @param[in]  v    normalized incident vector
 * @param[in]  n    normalized normal vector
 * @param[in]  eta  ratio of indices of refraction (incident/transmitted)
 * @param[out] dest refraction vector if refraction occurs; zero vector otherwise
 *
 * @returns true if refraction occurs; false if total internal reflection occurs.
 */
PLAY_CGLM_INLINE
bool
vec3_refract(vec3 v, vec3 n, float eta, vec3 dest)
{
    float ndi, eni, k;

    ndi = vec3_dot(n, v);
    eni = eta * ndi;
    k   = 1.0f - eta * eta + eni * eni;

    if (k < 0.0f)
    {
        vec3_zero(dest);
        return false;
    }

    vec3_scale(v, eta, dest);
    vec3_mulsubs(n, eni + sqrtf(k), dest);
    return true;
}

#endif /* cvec3_h */

/*** End of inlined file: vec3.h ***/


/*** Start of inlined file: ivec2.h ***/
/*
 Macros:
   PLAY_CGLM_IVEC2_ONE_INIT
   PLAY_CGLM_IVEC2_ZERO_INIT
   PLAY_CGLM_IVEC2_ONE
   PLAY_CGLM_IVEC2_ZERO

 Functions:
  PLAY_CGLM_INLINE void ivec2_new(int * __restrict v, ivec2 dest)
  PLAY_CGLM_INLINE void ivec2_copy(ivec2 a, ivec2 dest)
  PLAY_CGLM_INLINE void ivec2_zero(ivec2 v)
  PLAY_CGLM_INLINE void ivec2_one(ivec2 v)
  PLAY_CGLM_INLINE int ivec2_dot(ivec2 a, ivec2 b)
  PLAY_CGLM_INLINE int ivec2_cross(ivec2 a, ivec2 b)
  PLAY_CGLM_INLINE void ivec2_add(ivec2 a, ivec2 b, ivec2 dest)
  PLAY_CGLM_INLINE void ivec2_adds(ivec2 v, int s, ivec2 dest)
  PLAY_CGLM_INLINE void ivec2_sub(ivec2 a, ivec2 b, ivec2 dest)
  PLAY_CGLM_INLINE void ivec2_subs(ivec2 v, int s, ivec2 dest)
  PLAY_CGLM_INLINE void ivec2_mul(ivec2 a, ivec2 b, ivec2 dest)
  PLAY_CGLM_INLINE void ivec2_scale(ivec2 v, int s, ivec2 dest)
  PLAY_CGLM_INLINE void ivec2_div(ivec2 a, ivec2 b, ivec2 dest)
  PLAY_CGLM_INLINE void ivec2_divs(ivec2 v, int s, ivec2 dest)
  PLAY_CGLM_INLINE void ivec2_mod(ivec2 a, ivec2 b, ivec2 dest)
  PLAY_CGLM_INLINE void ivec2_addadd(ivec2 a, ivec2 b, ivec2 dest)
  PLAY_CGLM_INLINE void ivec2_addadds(ivec2 a, int s, ivec2 dest)
  PLAY_CGLM_INLINE void ivec2_subadd(ivec2 a, ivec2 b, ivec2 dest)
  PLAY_CGLM_INLINE void ivec2_subadds(ivec2 a, int s, ivec2 dest)
  PLAY_CGLM_INLINE void ivec2_muladd(ivec2 a, ivec2 b, ivec2 dest)
  PLAY_CGLM_INLINE void ivec2_muladds(ivec2 a, int s, ivec2 dest)
  PLAY_CGLM_INLINE void ivec2_maxadd(ivec2 a, ivec2 b, ivec2 dest)
  PLAY_CGLM_INLINE void ivec2_minadd(ivec2 a, ivec2 b, ivec2 dest)
  PLAY_CGLM_INLINE void ivec2_subsub(ivec2 a, ivec2 b, ivec2 dest)
  PLAY_CGLM_INLINE void ivec2_subsubs(ivec2 a, int s, ivec2 dest)
  PLAY_CGLM_INLINE void ivec2_addsub(ivec2 a, ivec2 b, ivec2 dest)
  PLAY_CGLM_INLINE void ivec2_addsubs(ivec2 a, int s, ivec2 dest)
  PLAY_CGLM_INLINE void ivec2_mulsub(ivec2 a, ivec2 b, ivec2 dest)
  PLAY_CGLM_INLINE void ivec2_mulsubs(ivec2 a, int s, ivec2 dest)
  PLAY_CGLM_INLINE void ivec2_maxsub(ivec2 a, ivec2 b, ivec2 dest)
  PLAY_CGLM_INLINE void ivec2_minsub(ivec2 a, ivec2 b, ivec2 dest)
  PLAY_CGLM_INLINE int ivec2_distance2(ivec2 a, ivec2 b)
  PLAY_CGLM_INLINE float ivec2_distance(ivec2 a, ivec2 b)
  PLAY_CGLM_INLINE void ivec2_fill(ivec2 v, int val);
  PLAY_CGLM_INLINE bool ivec2_eq(ivec2 v, int val);
  PLAY_CGLM_INLINE bool ivec2_eqv(ivec2 a, ivec2 b);
  PLAY_CGLM_INLINE void ivec2_maxv(ivec2 a, ivec2 b, ivec2 dest)
  PLAY_CGLM_INLINE void ivec2_minv(ivec2 a, ivec2 b, ivec2 dest)
  PLAY_CGLM_INLINE void ivec2_clamp(ivec2 v, int minVal, int maxVal)
  PLAY_CGLM_INLINE void ivec2_abs(ivec2 v, ivec2 dest)
 */

#ifndef civec2_h
#define civec2_h

#define PLAY_CGLM_IVEC2_ONE_INIT   {1, 1}
#define PLAY_CGLM_IVEC2_ZERO_INIT  {0, 0}

#define PLAY_CGLM_IVEC2_ONE  ((ivec2)PLAY_CGLM_IVEC2_ONE_INIT)
#define PLAY_CGLM_IVEC2_ZERO ((ivec2)PLAY_CGLM_IVEC2_ZERO_INIT)

/*!
 * @brief init ivec2 using vec3 or vec4
 *
 * @param[in]  v    vector
 * @param[out] dest destination
 */
PLAY_CGLM_INLINE
void
ivec2_new(int * __restrict v, ivec2 dest)
{
    dest[0] = v[0];
    dest[1] = v[1];
}

/*!
 * @brief copy all members of [a] to [dest]
 *
 * @param[in]  a    source vector
 * @param[out] dest destination
 */
PLAY_CGLM_INLINE
void
ivec2_copy(ivec2 a, ivec2 dest)
{
    dest[0] = a[0];
    dest[1] = a[1];
}

/*!
 * @brief set all members of [v] to zero
 *
 * @param[out] v vector
 */
PLAY_CGLM_INLINE
void
ivec2_zero(ivec2 v)
{
    v[0] = v[1] = 0;
}

/*!
 * @brief set all members of [v] to one
 *
 * @param[out] v vector
 */
PLAY_CGLM_INLINE
void
ivec2_one(ivec2 v)
{
    v[0] = v[1] = 1;
}

/*!
 * @brief ivec2 dot product
 *
 * @param[in] a vector1
 * @param[in] b vector2
 *
 * @return dot product
 */
PLAY_CGLM_INLINE
int
ivec2_dot(ivec2 a, ivec2 b)
{
    return a[0] * b[0] + a[1] * b[1];
}

/*!
 * @brief ivec2 cross product
 *
 * REF: http://allenchou.net/2013/07/cross-product-of-2d-vectors/
 *
 * @param[in]  a vector1
 * @param[in]  b vector2
 *
 * @return Z component of cross product
 */
PLAY_CGLM_INLINE
int
ivec2_cross(ivec2 a, ivec2 b)
{
    return a[0] * b[1] - a[1] * b[0];
}

/*!
 * @brief add vector [a] to vector [b] and store result in [dest]
 *
 * @param[in]  a    first vector
 * @param[in]  b    second vector
 * @param[out] dest destination
 */
PLAY_CGLM_INLINE
void
ivec2_add(ivec2 a, ivec2 b, ivec2 dest)
{
    dest[0] = a[0] + b[0];
    dest[1] = a[1] + b[1];
}

/*!
 * @brief add scalar s to vector [v] and store result in [dest]
 *
 * @param[in]  v    vector
 * @param[in]  s    scalar
 * @param[out] dest destination
 */
PLAY_CGLM_INLINE
void
ivec2_adds(ivec2 v, int s, ivec2 dest)
{
    dest[0] = v[0] + s;
    dest[1] = v[1] + s;
}

/*!
 * @brief subtract vector [b] from vector [a] and store result in [dest]
 *
 * @param[in]  a    first vector
 * @param[in]  b    second vector
 * @param[out] dest destination
 */
PLAY_CGLM_INLINE
void
ivec2_sub(ivec2 a, ivec2 b, ivec2 dest)
{
    dest[0] = a[0] - b[0];
    dest[1] = a[1] - b[1];
}

/*!
 * @brief subtract scalar s from vector [v] and store result in [dest]
 *
 * @param[in]  v    vector
 * @param[in]  s    scalar
 * @param[out] dest destination
 */
PLAY_CGLM_INLINE
void
ivec2_subs(ivec2 v, int s, ivec2 dest)
{
    dest[0] = v[0] - s;
    dest[1] = v[1] - s;
}

/*!
 * @brief multiply vector [a] with vector [b] and store result in [dest]
 *
 * @param[in]  a    first vector
 * @param[in]  b    second vector
 * @param[out] dest destination
 */
PLAY_CGLM_INLINE
void
ivec2_mul(ivec2 a, ivec2 b, ivec2 dest)
{
    dest[0] = a[0] * b[0];
    dest[1] = a[1] * b[1];
}

/*!
 * @brief multiply vector [a] with scalar s and store result in [dest]
 *
 * @param[in]  v    vector
 * @param[in]  s    scalar
 * @param[out] dest destination
 */
PLAY_CGLM_INLINE
void
ivec2_scale(ivec2 v, int s, ivec2 dest)
{
    dest[0] = v[0] * s;
    dest[1] = v[1] * s;
}

/*!
 * @brief div vector with another component-wise division: d = a / b
 *
 * @param[in]  a    vector 1
 * @param[in]  b    vector 2
 * @param[out] dest result = (a[0]/b[0], a[1]/b[1])
 */
PLAY_CGLM_INLINE
void
ivec2_div(ivec2 a, ivec2 b, ivec2 dest)
{
    dest[0] = a[0] / b[0];
    dest[1] = a[1] / b[1];
}

/*!
 * @brief div vector with scalar: d = v / s
 *
 * @param[in]  v    vector
 * @param[in]  s    scalar
 * @param[out] dest result = (a[0]/s, a[1]/s)
 */
PLAY_CGLM_INLINE
void
ivec2_divs(ivec2 v, int s, ivec2 dest)
{
    dest[0] = v[0] / s;
    dest[1] = v[1] / s;
}

/*!
 * @brief mod vector with another component-wise modulo: d = a % b
 *
 * @param[in]  a    vector 1
 * @param[in]  b    vector 2
 * @param[out] dest result = (a[0]%b[0], a[1]%b[1])
 */
PLAY_CGLM_INLINE
void
ivec2_mod(ivec2 a, ivec2 b, ivec2 dest)
{
    dest[0] = a[0] % b[0];
    dest[1] = a[1] % b[1];
}

/*!
 * @brief add vector [a] with vector [b] and add result to vector [dest]
 *
 * applies += operator so dest must be initialized
 *
 * @param[in]  a    first vector
 * @param[in]  b    second vector
 * @param[out] dest dest += (a + b)
 */
PLAY_CGLM_INLINE
void
ivec2_addadd(ivec2 a, ivec2 b, ivec2 dest)
{
    dest[0] += a[0] + b[0];
    dest[1] += a[1] + b[1];
}

/*!
 * @brief add scalar [s] onto vector [a] and add result to vector [dest]
 *
 * applies += operator so dest must be initialized
 *
 * @param[in]  a    vector
 * @param[in]  s    scalar
 * @param[out] dest dest += (a + s)
 */
PLAY_CGLM_INLINE
void
ivec2_addadds(ivec2 a, int s, ivec2 dest)
{
    dest[0] += a[0] + s;
    dest[1] += a[1] + s;
}

/*!
 * @brief subtract vector [a] from vector [b] and add result to [dest]
 *
 * applies += operator so dest must be initialized
 *
 * @param[in]  a    first vector
 * @param[in]  b    second vector
 * @param[out] dest dest += (a - b)
 */
PLAY_CGLM_INLINE
void
ivec2_subadd(ivec2 a, ivec2 b, ivec2 dest)
{
    dest[0] += a[0] - b[0];
    dest[1] += a[1] - b[1];
}

/*!
 * @brief subtract scalar [s] from vector [a] and add result to [dest]
 *
 * applies += operator so dest must be initialized
 *
 * @param[in]  a    first
 * @param[in]  s    scalar
 * @param[out] dest dest += (a - s)
 */
PLAY_CGLM_INLINE
void
ivec2_subadds(ivec2 a, int s, ivec2 dest)
{
    dest[0] += a[0] - s;
    dest[1] += a[1] - s;
}

/*!
 * @brief multiply vector [a] with vector [b] and add result to [dest]
 *
 * applies += operator so dest must be initialized
 *
 * @param[in]  a    first vector
 * @param[in]  b    second vector
 * @param[out] dest dest += (a * b)
 */
PLAY_CGLM_INLINE
void
ivec2_muladd(ivec2 a, ivec2 b, ivec2 dest)
{
    dest[0] += a[0] * b[0];
    dest[1] += a[1] * b[1];
}

/*!
 * @brief multiply vector [a] with scalar [s] and add result to [dest]
 *
 * applies += operator so dest must be initialized
 *
 * @param[in]  a    vector
 * @param[in]  s    scalar
 * @param[out] dest dest += (a * s)
 */
PLAY_CGLM_INLINE
void
ivec2_muladds(ivec2 a, int s, ivec2 dest)
{
    dest[0] += a[0] * s;
    dest[1] += a[1] * s;
}

/*!
 * @brief add maximum of vector [a] and vector [b] to vector [dest]
 *
 * applies += operator so dest must be initialized
 *
 * @param[in]  a    first vector
 * @param[in]  b    second vector
 * @param[out] dest dest += fmax(a, b)
 */
PLAY_CGLM_INLINE
void
ivec2_maxadd(ivec2 a, ivec2 b, ivec2 dest)
{
    dest[0] += imax(a[0], b[0]);
    dest[1] += imax(a[1], b[1]);
}

/*!
 * @brief add minimum of vector [a] and vector [b] to vector [dest]
 *
 * applies += operator so dest must be initialized
 *
 * @param[in]  a    first vector
 * @param[in]  b    second vector
 * @param[out] dest dest += fmin(a, b)
 */
PLAY_CGLM_INLINE
void
ivec2_minadd(ivec2 a, ivec2 b, ivec2 dest)
{
    dest[0] += imin(a[0], b[0]);
    dest[1] += imin(a[1], b[1]);
}

/*!
 * @brief subtract vector [a] from vector [b] and subtract result from [dest]
 *
 * applies -= operator so dest must be initialized
 *
 * @param[in]  a    first vector
 * @param[in]  b    second vector
 * @param[out] dest dest -= (a - b)
 */
PLAY_CGLM_INLINE
void
ivec2_subsub(ivec2 a, ivec2 b, ivec2 dest)
{
    dest[0] -= a[0] - b[0];
    dest[1] -= a[1] - b[1];
}

/*!
 * @brief subtract scalar [s] from vector [a] and subtract result from [dest]
 *
 * applies -= operator so dest must be initialized
 *
 * @param[in]  a    vector
 * @param[in]  s    scalar
 * @param[out] dest dest -= (a - s)
 */
PLAY_CGLM_INLINE
void
ivec2_subsubs(ivec2 a, int s, ivec2 dest)
{
    dest[0] -= a[0] - s;
    dest[1] -= a[1] - s;
}

/*!
 * @brief add vector [a] to vector [b] and subtract the result from [dest]
 *
 * applies -= operator so dest must be initialized
 *
 * @param[in]  a    vector
 * @param[in]  b    scalar
 * @param[out] dest dest -= (a + b)
 */
PLAY_CGLM_INLINE
void
ivec2_addsub(ivec2 a, ivec2 b, ivec2 dest)
{
    dest[0] -= a[0] + b[0];
    dest[1] -= a[1] + b[1];
}

/*!
 * @brief add scalar [s] to vector [a] and subtract the result from [dest]
 *
 * applies -= operator so dest must be initialized
 *
 * @param[in]  a    vector
 * @param[in]  s    scalar
 * @param[out] dest dest -= (a + b)
 */
PLAY_CGLM_INLINE
void
ivec2_addsubs(ivec2 a, int s, ivec2 dest)
{
    dest[0] -= a[0] + s;
    dest[1] -= a[1] + s;
}

/*!
 * @brief multiply vector [a] and vector [b] and subtract the result from [dest]
 *
 * applies -= operator so dest must be initialized
 *
 * @param[in]  a    vector
 * @param[in]  b    scalar
 * @param[out] dest dest -= (a * b)
 */
PLAY_CGLM_INLINE
void
ivec2_mulsub(ivec2 a, ivec2 b, ivec2 dest)
{
    dest[0] -= a[0] * b[0];
    dest[1] -= a[1] * b[1];
}

/*!
 * @brief multiply vector [a] with scalar [s] and subtract the result from [dest]
 *
 * applies -= operator so dest must be initialized
 *
 * @param[in]  a    vector
 * @param[in]  s    scalar
 * @param[out] dest dest -= (a * s)
 */
PLAY_CGLM_INLINE
void
ivec2_mulsubs(ivec2 a, int s, ivec2 dest)
{
    dest[0] -= a[0] * s;
    dest[1] -= a[1] * s;
}

/*!
 * @brief subtract maximum of vector [a] and vector [b] from vector [dest]
 *
 * applies += operator so dest must be initialized
 *
 * @param[in]  a    first vector
 * @param[in]  b    second vector
 * @param[out] dest dest -= fmax(a, b)
 */
PLAY_CGLM_INLINE
void
ivec2_maxsub(ivec2 a, ivec2 b, ivec2 dest)
{
    dest[0] -= imax(a[0], b[0]);
    dest[1] -= imax(a[1], b[1]);
}

/*!
 * @brief subtract minimum of vector [a] and vector [b] from vector [dest]
 *
 * applies -= operator so dest must be initialized
 *
 * @param[in]  a    first vector
 * @param[in]  b    second vector
 * @param[out] dest dest -= fmin(a, b)
 */
PLAY_CGLM_INLINE
void
ivec2_minsub(ivec2 a, ivec2 b, ivec2 dest)
{
    dest[0] -= imin(a[0], b[0]);
    dest[1] -= imin(a[1], b[1]);
}

/*!
 * @brief squared distance between two vectors
 *
 * @param[in] a first vector
 * @param[in] b second vector
 * @return returns squared distance (distance * distance)
 */
PLAY_CGLM_INLINE
int
ivec2_distance2(ivec2 a, ivec2 b)
{
    int xd, yd;
    xd = a[0] - b[0];
    yd = a[1] - b[1];
    return xd * xd + yd * yd;
}

/*!
 * @brief distance between two vectors
 *
 * @param[in] a first vector
 * @param[in] b second vector
 * @return returns distance
 */
PLAY_CGLM_INLINE
float
ivec2_distance(ivec2 a, ivec2 b)
{
    return sqrtf((float)ivec2_distance2(a, b));
}

/*!
 * @brief fill a vector with specified value
 *
 * @param[out] v   dest
 * @param[in]  val value
 */
PLAY_CGLM_INLINE
void
ivec2_fill(ivec2 v, int val)
{
    v[0] = v[1] = val;
}

/*!
 * @brief check if vector is equal to value
 *
 * @param[in] v   vector
 * @param[in] val value
 */
PLAY_CGLM_INLINE
bool
ivec2_eq(ivec2 v, int val)
{
    return v[0] == val && v[0] == v[1];
}

/*!
 * @brief check if vector is equal to another
 *
 * @param[in] a vector
 * @param[in] b vector
 */
PLAY_CGLM_INLINE
bool
ivec2_eqv(ivec2 a, ivec2 b)
{
    return a[0] == b[0]
           && a[1] == b[1];
}

/*!
 * @brief set each member of dest to greater of vector a and b
 *
 * @param[in]  a    first vector
 * @param[in]  b    second vector
 * @param[out] dest destination
 */
PLAY_CGLM_INLINE
void
ivec2_maxv(ivec2 a, ivec2 b, ivec2 dest)
{
    dest[0] = a[0] > b[0] ? a[0] : b[0];
    dest[1] = a[1] > b[1] ? a[1] : b[1];
}

/*!
 * @brief set each member of dest to lesser of vector a and b
 *
 * @param[in]  a    first vector
 * @param[in]  b    second vector
 * @param[out] dest destination
 */
PLAY_CGLM_INLINE
void
ivec2_minv(ivec2 a, ivec2 b, ivec2 dest)
{
    dest[0] = a[0] < b[0] ? a[0] : b[0];
    dest[1] = a[1] < b[1] ? a[1] : b[1];
}

/*!
 * @brief clamp each member of [v] between minVal and maxVal (inclusive)
 *
 * @param[in, out] v      vector
 * @param[in]      minVal minimum value
 * @param[in]      maxVal maximum value
 */
PLAY_CGLM_INLINE
void
ivec2_clamp(ivec2 v, int minVal, int maxVal)
{
    if (v[0] < minVal)
        v[0] = minVal;
    else if(v[0] > maxVal)
        v[0] = maxVal;

    if (v[1] < minVal)
        v[1] = minVal;
    else if(v[1] > maxVal)
        v[1] = maxVal;
}

/*!
 * @brief absolute value of v
 *
 * @param[in]	v	vector
 * @param[out]	dest	destination
 */
PLAY_CGLM_INLINE
void
ivec2_abs(ivec2 v, ivec2 dest)
{
    dest[0] = abs(v[0]);
    dest[1] = abs(v[1]);
}

#endif /* civec2_h */

/*** End of inlined file: ivec2.h ***/


/*** Start of inlined file: ivec3.h ***/
/*
 Macros:
   PLAY_CGLM_IVEC3_ONE_INIT
   PLAY_CGLM_IVEC3_ZERO_INIT
   PLAY_CGLM_IVEC3_ONE
   PLAY_CGLM_IVEC3_ZERO

 Functions:
  PLAY_CGLM_INLINE void ivec3_new(ivec4 v4, ivec3 dest)
  PLAY_CGLM_INLINE void ivec3_copy(ivec3 a, ivec3 dest)
  PLAY_CGLM_INLINE void ivec3_zero(ivec3 v)
  PLAY_CGLM_INLINE void ivec3_one(ivec3 v)
  PLAY_CGLM_INLINE int ivec3_dot(ivec3 a, ivec3 b)
  PLAY_CGLM_INLINE int ivec3_norm2(ivec3 v)
  PLAY_CGLM_INLINE int ivec3_norm(ivec3 v)
  PLAY_CGLM_INLINE void ivec3_add(ivec3 a, ivec3 b, ivec3 dest)
  PLAY_CGLM_INLINE void ivec3_adds(ivec3 v, int s, ivec3 dest)
  PLAY_CGLM_INLINE void ivec3_sub(ivec3 a, ivec3 b, ivec3 dest)
  PLAY_CGLM_INLINE void ivec3_subs(ivec3 v, int s, ivec3 dest)
  PLAY_CGLM_INLINE void ivec3_mul(ivec3 a, ivec3 b, ivec3 dest)
  PLAY_CGLM_INLINE void ivec3_scale(ivec3 v, int s, ivec3 dest)
  PLAY_CGLM_INLINE void ivec3_div(ivec3 a, ivec3 b, ivec3 dest)
  PLAY_CGLM_INLINE void ivec3_divs(ivec3 v, int s, ivec3 dest)
  PLAY_CGLM_INLINE void ivec3_mod(ivec3 a, ivec3 b, ivec3 dest)
  PLAY_CGLM_INLINE void ivec3_addadd(ivec3 a, ivec3 b, ivec3 dest)
  PLAY_CGLM_INLINE void ivec3_addadds(ivec3 a, int s, ivec3 dest)
  PLAY_CGLM_INLINE void ivec3_subadd(ivec3 a, ivec3 b, ivec3 dest)
  PLAY_CGLM_INLINE void ivec3_subadds(ivec3 a, int s, ivec3 dest)
  PLAY_CGLM_INLINE void ivec3_muladd(ivec3 a, ivec3 b, ivec3 dest)
  PLAY_CGLM_INLINE void ivec3_muladds(ivec3 a, int s, ivec3 dest)
  PLAY_CGLM_INLINE void ivec3_maxadd(ivec3 a, ivec3 b, ivec3 dest)
  PLAY_CGLM_INLINE void ivec3_minadd(ivec3 a, ivec3 b, ivec3 dest)
  PLAY_CGLM_INLINE void ivec3_subsub(ivec3 a, ivec3 b, ivec3 dest)
  PLAY_CGLM_INLINE void ivec3_subsubs(ivec3 a, int s, ivec3 dest)
  PLAY_CGLM_INLINE void ivec3_addsub(ivec3 a, ivec3 b, ivec3 dest)
  PLAY_CGLM_INLINE void ivec3_addsubs(ivec3 a, int s, ivec3 dest)
  PLAY_CGLM_INLINE void ivec3_mulsub(ivec3 a, ivec3 b, ivec3 dest)
  PLAY_CGLM_INLINE void ivec3_mulsubs(ivec3 a, int s, ivec3 dest)
  PLAY_CGLM_INLINE void ivec3_maxsub(ivec3 a, ivec3 b, ivec3 dest)
  PLAY_CGLM_INLINE void ivec3_minsub(ivec3 a, ivec3 b, ivec3 dest)
  PLAY_CGLM_INLINE int ivec3_distance2(ivec3 a, ivec3 b)
  PLAY_CGLM_INLINE float ivec3_distance(ivec3 a, ivec3 b)
  PLAY_CGLM_INLINE void ivec3_fill(ivec3 v, int val);
  PLAY_CGLM_INLINE bool ivec3_eq(ivec3 v, int val);
  PLAY_CGLM_INLINE bool ivec3_eqv(ivec3 a, ivec3 b);
  PLAY_CGLM_INLINE void ivec3_maxv(ivec3 a, ivec3 b, ivec3 dest)
  PLAY_CGLM_INLINE void ivec3_minv(ivec3 a, ivec3 b, ivec3 dest)
  PLAY_CGLM_INLINE void ivec3_clamp(ivec3 v, int minVal, int maxVal)
  PLAY_CGLM_INLINE void ivec3_abs(ivec3 v, ivec3 dest)
 */

#ifndef civec3_h
#define civec3_h

#define PLAY_CGLM_IVEC3_ONE_INIT   {1, 1, 1}
#define PLAY_CGLM_IVEC3_ZERO_INIT  {0, 0, 0}

#define PLAY_CGLM_IVEC3_ONE  ((ivec3)PLAY_CGLM_IVEC3_ONE_INIT)
#define PLAY_CGLM_IVEC3_ZERO ((ivec3)PLAY_CGLM_IVEC3_ZERO_INIT)

/*!
 * @brief init ivec3 using ivec4
 *
 * @param[in]  v4   vector4
 * @param[out] dest destination
 */
PLAY_CGLM_INLINE
void
ivec3_new(ivec4 v4, ivec3 dest)
{
    dest[0] = v4[0];
    dest[1] = v4[1];
    dest[2] = v4[2];
}

/*!
 * @brief copy all members of [a] to [dest]
 *
 * @param[in]  a    source vector
 * @param[out] dest destination
 */
PLAY_CGLM_INLINE
void
ivec3_copy(ivec3 a, ivec3 dest)
{
    dest[0] = a[0];
    dest[1] = a[1];
    dest[2] = a[2];
}

/*!
 * @brief set all members of [v] to zero
 *
 * @param[out] v vector
 */
PLAY_CGLM_INLINE
void
ivec3_zero(ivec3 v)
{
    v[0] = v[1] = v[2] = 0;
}

/*!
 * @brief set all members of [v] to one
 *
 * @param[out] v vector
 */
PLAY_CGLM_INLINE
void
ivec3_one(ivec3 v)
{
    v[0] = v[1] = v[2] = 1;
}

/*!
 * @brief ivec3 dot product
 *
 * @param[in] a vector1
 * @param[in] b vector2
 *
 * @return dot product
 */
PLAY_CGLM_INLINE
int
ivec3_dot(ivec3 a, ivec3 b)
{
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
}

/*!
 * @brief norm * norm (magnitude) of vec
 *
 * we can use this func instead of calling norm * norm, because it would call
 * sqrtf function twice but with this func we can avoid func call, maybe this is
 * not good name for this func
 *
 * @param[in] v vector
 *
 * @return norm * norm
 */
PLAY_CGLM_INLINE
int
ivec3_norm2(ivec3 v)
{
    return ivec3_dot(v, v);
}

/*!
 * @brief euclidean norm (magnitude), also called L2 norm
 *        this will give magnitude of vector in euclidean space
 *
 * @param[in] v vector
 *
 * @return norm
 */
PLAY_CGLM_INLINE
int
ivec3_norm(ivec3 v)
{
    return (int)sqrtf((float)ivec3_norm2(v));
}

/*!
 * @brief add vector [a] to vector [b] and store result in [dest]
 *
 * @param[in]  a    first vector
 * @param[in]  b    second vector
 * @param[out] dest destination
 */
PLAY_CGLM_INLINE
void
ivec3_add(ivec3 a, ivec3 b, ivec3 dest)
{
    dest[0] = a[0] + b[0];
    dest[1] = a[1] + b[1];
    dest[2] = a[2] + b[2];
}

/*!
 * @brief add scalar s to vector [v] and store result in [dest]
 *
 * @param[in]  v    vector
 * @param[in]  s    scalar
 * @param[out] dest destination
 */
PLAY_CGLM_INLINE
void
ivec3_adds(ivec3 v, int s, ivec3 dest)
{
    dest[0] = v[0] + s;
    dest[1] = v[1] + s;
    dest[2] = v[2] + s;
}

/*!
 * @brief subtract vector [b] from vector [a] and store result in [dest]
 *
 * @param[in]  a    first vector
 * @param[in]  b    second vector
 * @param[out] dest destination
 */
PLAY_CGLM_INLINE
void
ivec3_sub(ivec3 a, ivec3 b, ivec3 dest)
{
    dest[0] = a[0] - b[0];
    dest[1] = a[1] - b[1];
    dest[2] = a[2] - b[2];
}

/*!
 * @brief subtract scalar s from vector [v] and store result in [dest]
 *
 * @param[in]  v    vector
 * @param[in]  s    scalar
 * @param[out] dest destination
 */
PLAY_CGLM_INLINE
void
ivec3_subs(ivec3 v, int s, ivec3 dest)
{
    dest[0] = v[0] - s;
    dest[1] = v[1] - s;
    dest[2] = v[2] - s;
}

/*!
 * @brief multiply vector [a] with vector [b] and store result in [dest]
 *
 * @param[in]  a    first vector
 * @param[in]  b    second vector
 * @param[out] dest destination
 */
PLAY_CGLM_INLINE
void
ivec3_mul(ivec3 a, ivec3 b, ivec3 dest)
{
    dest[0] = a[0] * b[0];
    dest[1] = a[1] * b[1];
    dest[2] = a[2] * b[2];
}

/*!
 * @brief multiply vector [a] with scalar s and store result in [dest]
 *
 * @param[in]  v    vector
 * @param[in]  s    scalar
 * @param[out] dest destination
 */
PLAY_CGLM_INLINE
void
ivec3_scale(ivec3 v, int s, ivec3 dest)
{
    dest[0] = v[0] * s;
    dest[1] = v[1] * s;
    dest[2] = v[2] * s;
}

/*!
 * @brief div vector with another component-wise division: d = a / b
 *
 * @param[in]  a    vector 1
 * @param[in]  b    vector 2
 * @param[out] dest result = (a[0]/b[0], a[1]/b[1], a[2]/b[2])
 */
PLAY_CGLM_INLINE
void
ivec3_div(ivec3 a, ivec3 b, ivec3 dest)
{
    dest[0] = a[0] / b[0];
    dest[1] = a[1] / b[1];
    dest[2] = a[2] / b[2];
}

/*!
 * @brief div vector with scalar: d = v / s
 *
 * @param[in]  v    vector
 * @param[in]  s    scalar
 * @param[out] dest result = (a[0]/s, a[1]/s, a[2]/s)
 */
PLAY_CGLM_INLINE
void
ivec3_divs(ivec3 v, int s, ivec3 dest)
{
    dest[0] = v[0] / s;
    dest[1] = v[1] / s;
    dest[2] = v[2] / s;
}

/*!
 * @brief Element-wise modulo operation on ivec3 vectors: dest = a % b
 *
 * Performs element-wise modulo on each component of vectors `a` and `b`.
 *
 * @param[in]  a    vector 1
 * @param[in]  b    vector 2
 * @param[out] dest result = (a[0]%b[0], a[1]%b[1], a[2]%b[2])
 */
PLAY_CGLM_INLINE
void
ivec3_mod(ivec3 a, ivec3 b, ivec3 dest)
{
    dest[0] = a[0] % b[0];
    dest[1] = a[1] % b[1];
    dest[2] = a[2] % b[2];
}

/*!
 * @brief add vector [a] with vector [b] and add result to vector [dest]
 *
 * applies += operator so dest must be initialized
 *
 * @param[in]  a    first vector
 * @param[in]  b    second vector
 * @param[out] dest dest += (a + b)
 */
PLAY_CGLM_INLINE
void
ivec3_addadd(ivec3 a, ivec3 b, ivec3 dest)
{
    dest[0] += a[0] + b[0];
    dest[1] += a[1] + b[1];
    dest[2] += a[2] + b[2];
}

/*!
 * @brief add scalar [s] onto vector [a] and add result to vector [dest]
 *
 * applies += operator so dest must be initialized
 *
 * @param[in]  a    vector
 * @param[in]  s    scalar
 * @param[out] dest dest += (a + s)
 */
PLAY_CGLM_INLINE
void
ivec3_addadds(ivec3 a, int s, ivec3 dest)
{
    dest[0] += a[0] + s;
    dest[1] += a[1] + s;
    dest[2] += a[2] + s;
}

/*!
 * @brief subtract vector [a] from vector [b] and add result to [dest]
 *
 * applies += operator so dest must be initialized
 *
 * @param[in]  a    first vector
 * @param[in]  b    second vector
 * @param[out] dest dest += (a - b)
 */
PLAY_CGLM_INLINE
void
ivec3_subadd(ivec3 a, ivec3 b, ivec3 dest)
{
    dest[0] += a[0] - b[0];
    dest[1] += a[1] - b[1];
    dest[2] += a[2] - b[2];
}

/*!
 * @brief subtract scalar [s] from vector [a] and add result to [dest]
 *
 * applies += operator so dest must be initialized
 *
 * @param[in]  a    first
 * @param[in]  s    scalar
 * @param[out] dest dest += (a - s)
 */
PLAY_CGLM_INLINE
void
ivec3_subadds(ivec3 a, int s, ivec3 dest)
{
    dest[0] += a[0] - s;
    dest[1] += a[1] - s;
    dest[2] += a[2] - s;
}

/*!
 * @brief multiply vector [a] with vector [b] and add result to [dest]
 *
 * applies += operator so dest must be initialized
 *
 * @param[in]  a    first vector
 * @param[in]  b    second vector
 * @param[out] dest dest += (a * b)
 */
PLAY_CGLM_INLINE
void
ivec3_muladd(ivec3 a, ivec3 b, ivec3 dest)
{
    dest[0] += a[0] * b[0];
    dest[1] += a[1] * b[1];
    dest[2] += a[2] * b[2];
}

/*!
 * @brief multiply vector [a] with scalar [s] and add result to [dest]
 *
 * applies += operator so dest must be initialized
 *
 * @param[in]  a    vector
 * @param[in]  s    scalar
 * @param[out] dest dest += (a * s)
 */
PLAY_CGLM_INLINE
void
ivec3_muladds(ivec3 a, int s, ivec3 dest)
{
    dest[0] += a[0] * s;
    dest[1] += a[1] * s;
    dest[2] += a[2] * s;
}

/*!
 * @brief add maximum of vector [a] and vector [b] to vector [dest]
 *
 * applies += operator so dest must be initialized
 *
 * @param[in]  a    first vector
 * @param[in]  b    second vector
 * @param[out] dest dest += fmax(a, b)
 */
PLAY_CGLM_INLINE
void
ivec3_maxadd(ivec3 a, ivec3 b, ivec3 dest)
{
    dest[0] += imax(a[0], b[0]);
    dest[1] += imax(a[1], b[1]);
    dest[2] += imax(a[2], b[2]);
}

/*!
 * @brief add minimum of vector [a] and vector [b] to vector [dest]
 *
 * applies += operator so dest must be initialized
 *
 * @param[in]  a    first vector
 * @param[in]  b    second vector
 * @param[out] dest dest += fmin(a, b)
 */
PLAY_CGLM_INLINE
void
ivec3_minadd(ivec3 a, ivec3 b, ivec3 dest)
{
    dest[0] += imin(a[0], b[0]);
    dest[1] += imin(a[1], b[1]);
    dest[2] += imin(a[2], b[2]);
}

/*!
 * @brief subtract vector [a] from vector [b] and subtract result from [dest]
 *
 * applies -= operator so dest must be initialized
 *
 * @param[in]  a    first vector
 * @param[in]  b    second vector
 * @param[out] dest dest -= (a - b)
 */
PLAY_CGLM_INLINE
void
ivec3_subsub(ivec3 a, ivec3 b, ivec3 dest)
{
    dest[0] -= a[0] - b[0];
    dest[1] -= a[1] - b[1];
    dest[2] -= a[2] - b[2];
}

/*!
 * @brief subtract scalar [s] from vector [a] and subtract result from [dest]
 *
 * applies -= operator so dest must be initialized
 *
 * @param[in]  a    vector
 * @param[in]  s    scalar
 * @param[out] dest dest -= (a - s)
 */
PLAY_CGLM_INLINE
void
ivec3_subsubs(ivec3 a, int s, ivec3 dest)
{
    dest[0] -= a[0] - s;
    dest[1] -= a[1] - s;
    dest[2] -= a[2] - s;
}

/*!
 * @brief add vector [a] to vector [b] and subtract the result from [dest]
 *
 * applies -= operator so dest must be initialized
 *
 * @param[in]  a    vector
 * @param[in]  b    scalar
 * @param[out] dest dest -= (a + b)
 */
PLAY_CGLM_INLINE
void
ivec3_addsub(ivec3 a, ivec3 b, ivec3 dest)
{
    dest[0] -= a[0] + b[0];
    dest[1] -= a[1] + b[1];
    dest[2] -= a[2] + b[2];
}

/*!
 * @brief add scalar [s] to vector [a] and subtract the result from [dest]
 *
 * applies -= operator so dest must be initialized
 *
 * @param[in]  a    vector
 * @param[in]  s    scalar
 * @param[out] dest dest -= (a + b)
 */
PLAY_CGLM_INLINE
void
ivec3_addsubs(ivec3 a, int s, ivec3 dest)
{
    dest[0] -= a[0] + s;
    dest[1] -= a[1] + s;
    dest[2] -= a[2] + s;
}

/*!
 * @brief multiply vector [a] and vector [b] and subtract the result from [dest]
 *
 * applies -= operator so dest must be initialized
 *
 * @param[in]  a    vector
 * @param[in]  b    scalar
 * @param[out] dest dest -= (a * b)
 */
PLAY_CGLM_INLINE
void
ivec3_mulsub(ivec3 a, ivec3 b, ivec3 dest)
{
    dest[0] -= a[0] * b[0];
    dest[1] -= a[1] * b[1];
    dest[2] -= a[2] * b[2];
}

/*!
 * @brief multiply vector [a] with scalar [s] and subtract the result from [dest]
 *
 * applies -= operator so dest must be initialized
 *
 * @param[in]  a    vector
 * @param[in]  s    scalar
 * @param[out] dest dest -= (a * s)
 */
PLAY_CGLM_INLINE
void
ivec3_mulsubs(ivec3 a, int s, ivec3 dest)
{
    dest[0] -= a[0] * s;
    dest[1] -= a[1] * s;
    dest[2] -= a[2] * s;
}

/*!
 * @brief subtract maximum of vector [a] and vector [b] from vector [dest]
 *
 * applies += operator so dest must be initialized
 *
 * @param[in]  a    first vector
 * @param[in]  b    second vector
 * @param[out] dest dest -= fmax(a, b)
 */
PLAY_CGLM_INLINE
void
ivec3_maxsub(ivec3 a, ivec3 b, ivec3 dest)
{
    dest[0] -= imax(a[0], b[0]);
    dest[1] -= imax(a[1], b[1]);
    dest[2] -= imax(a[2], b[2]);
}

/*!
 * @brief subtract minimum of vector [a] and vector [b] from vector [dest]
 *
 * applies -= operator so dest must be initialized
 *
 * @param[in]  a    first vector
 * @param[in]  b    second vector
 * @param[out] dest dest -= fmin(a, b)
 */
PLAY_CGLM_INLINE
void
ivec3_minsub(ivec3 a, ivec3 b, ivec3 dest)
{
    dest[0] -= imin(a[0], b[0]);
    dest[1] -= imin(a[1], b[1]);
    dest[2] -= imin(a[2], b[2]);
}

/*!
 * @brief squared distance between two vectors
 *
 * @param[in] a first vector
 * @param[in] b second vector
 * @return returns squared distance (distance * distance)
 */
PLAY_CGLM_INLINE
int
ivec3_distance2(ivec3 a, ivec3 b)
{
    int xd, yd, zd;
    xd = a[0] - b[0];
    yd = a[1] - b[1];
    zd = a[2] - b[2];
    return xd * xd + yd * yd + zd * zd;
}

/*!
 * @brief distance between two vectors
 *
 * @param[in] a first vector
 * @param[in] b second vector
 * @return returns distance
 */
PLAY_CGLM_INLINE
float
ivec3_distance(ivec3 a, ivec3 b)
{
    return sqrtf((float)ivec3_distance2(a, b));
}

/*!
 * @brief fill a vector with specified value
 *
 * @param[out] v   dest
 * @param[in]  val value
 */
PLAY_CGLM_INLINE
void
ivec3_fill(ivec3 v, int val)
{
    v[0] = v[1] = v[2] = val;
}

/*!
 * @brief check if vector is equal to value
 *
 * @param[in] v   vector
 * @param[in] val value
 */
PLAY_CGLM_INLINE
bool
ivec3_eq(ivec3 v, int val)
{
    return v[0] == val && v[0] == v[1] && v[0] == v[2];
}

/*!
 * @brief check if vector is equal to another
 *
 * @param[in] a vector
 * @param[in] b vector
 */
PLAY_CGLM_INLINE
bool
ivec3_eqv(ivec3 a, ivec3 b)
{
    return a[0] == b[0]
           && a[1] == b[1]
           && a[2] == b[2];
}

/*!
 * @brief set each member of dest to greater of vector a and b
 *
 * @param[in]  a    first vector
 * @param[in]  b    second vector
 * @param[out] dest destination
 */
PLAY_CGLM_INLINE
void
ivec3_maxv(ivec3 a, ivec3 b, ivec3 dest)
{
    dest[0] = a[0] > b[0] ? a[0] : b[0];
    dest[1] = a[1] > b[1] ? a[1] : b[1];
    dest[2] = a[2] > b[2] ? a[2] : b[2];
}

/*!
 * @brief set each member of dest to lesser of vector a and b
 *
 * @param[in]  a    first vector
 * @param[in]  b    second vector
 * @param[out] dest destination
 */
PLAY_CGLM_INLINE
void
ivec3_minv(ivec3 a, ivec3 b, ivec3 dest)
{
    dest[0] = a[0] < b[0] ? a[0] : b[0];
    dest[1] = a[1] < b[1] ? a[1] : b[1];
    dest[2] = a[2] < b[2] ? a[2] : b[2];
}

/*!
 * @brief clamp each member of [v] between minVal and maxVal (inclusive)
 *
 * @param[in, out] v      vector
 * @param[in]      minVal minimum value
 * @param[in]      maxVal maximum value
 */
PLAY_CGLM_INLINE
void
ivec3_clamp(ivec3 v, int minVal, int maxVal)
{
    if (v[0] < minVal)
        v[0] = minVal;
    else if(v[0] > maxVal)
        v[0] = maxVal;

    if (v[1] < minVal)
        v[1] = minVal;
    else if(v[1] > maxVal)
        v[1] = maxVal;

    if (v[2] < minVal)
        v[2] = minVal;
    else if(v[2] > maxVal)
        v[2] = maxVal;
}

/*!
 * @brief absolute value of v
 *
 * @param[in]	v	vector
 * @param[out]	dest	destination
 */
PLAY_CGLM_INLINE
void
ivec3_abs(ivec3 v, ivec3 dest)
{
    dest[0] = abs(v[0]);
    dest[1] = abs(v[1]);
    dest[2] = abs(v[2]);
}

#endif /* civec3_h */

/*** End of inlined file: ivec3.h ***/


/*** Start of inlined file: ivec4.h ***/
/*
 Macros:
   PLAY_CGLM_IVEC4_ONE_INIT
   PLAY_CGLM_IVEC4_ZERO_INIT
   PLAY_CGLM_IVEC4_ONE
   PLAY_CGLM_IVEC4_ZERO

 Functions:
  PLAY_CGLM_INLINE void ivec4_new(ivec3 v3, int last, ivec4 dest)
  PLAY_CGLM_INLINE void ivec4_copy(ivec4 a, ivec4 dest)
  PLAY_CGLM_INLINE void ivec4_zero(ivec4 v)
  PLAY_CGLM_INLINE void ivec4_one(ivec4 v)
  PLAY_CGLM_INLINE void ivec4_add(ivec4 a, ivec4 b, ivec4 dest)
  PLAY_CGLM_INLINE void ivec4_adds(ivec4 v, int s, ivec4 dest)
  PLAY_CGLM_INLINE void ivec4_sub(ivec4 a, ivec4 b, ivec4 dest)
  PLAY_CGLM_INLINE void ivec4_subs(ivec4 v, int s, ivec4 dest)
  PLAY_CGLM_INLINE void ivec4_mul(ivec4 a, ivec4 b, ivec4 dest)
  PLAY_CGLM_INLINE void ivec4_scale(ivec4 v, int s, ivec4 dest)
  PLAY_CGLM_INLINE void ivec4_addadd(ivec4 a, ivec4 b, ivec4 dest)
  PLAY_CGLM_INLINE void ivec4_addadds(ivec4 a, int s, ivec4 dest)
  PLAY_CGLM_INLINE void ivec4_subadd(ivec4 a, ivec4 b, ivec4 dest)
  PLAY_CGLM_INLINE void ivec4_subadds(ivec4 a, int s, ivec4 dest)
  PLAY_CGLM_INLINE void ivec4_muladd(ivec4 a, ivec4 b, ivec4 dest)
  PLAY_CGLM_INLINE void ivec4_muladds(ivec4 a, int s, ivec4 dest)
  PLAY_CGLM_INLINE void ivec4_maxadd(ivec4 a, ivec4 b, ivec4 dest)
  PLAY_CGLM_INLINE void ivec4_minadd(ivec4 a, ivec4 b, ivec4 dest)
  PLAY_CGLM_INLINE void ivec4_subsub(ivec4 a, ivec4 b, ivec4 dest)
  PLAY_CGLM_INLINE void ivec4_subsubs(ivec4 a, int s, ivec4 dest)
  PLAY_CGLM_INLINE void ivec4_addsub(ivec4 a, ivec4 b, ivec4 dest)
  PLAY_CGLM_INLINE void ivec4_addsubs(ivec4 a, int s, ivec4 dest)
  PLAY_CGLM_INLINE void ivec4_mulsub(ivec4 a, ivec4 b, ivec4 dest)
  PLAY_CGLM_INLINE void ivec4_mulsubs(ivec4 a, int s, ivec4 dest)
  PLAY_CGLM_INLINE void ivec4_maxsub(ivec4 a, ivec4 b, ivec4 dest)
  PLAY_CGLM_INLINE void ivec4_minsub(ivec4 a, ivec4 b, ivec4 dest)
  PLAY_CGLM_INLINE int ivec4_distance2(ivec4 a, ivec4 b)
  PLAY_CGLM_INLINE float ivec4_distance(ivec4 a, ivec4 b)
  PLAY_CGLM_INLINE void ivec4_maxv(ivec4 a, ivec4 b, ivec4 dest)
  PLAY_CGLM_INLINE void ivec4_minv(ivec4 a, ivec4 b, ivec4 dest)
  PLAY_CGLM_INLINE void ivec4_clamp(ivec4 v, int minVal, int maxVal)
  PLAY_CGLM_INLINE void ivec4_abs(ivec4 v, ivec4 dest)
 */

#ifndef civec4_h
#define civec4_h

#define PLAY_CGLM_IVEC4_ONE_INIT   {1, 1, 1, 1}
#define PLAY_CGLM_IVEC4_ZERO_INIT  {0, 0, 0, 0}

#define PLAY_CGLM_IVEC4_ONE  ((ivec4)PLAY_CGLM_IVEC4_ONE_INIT)
#define PLAY_CGLM_IVEC4_ZERO ((ivec4)PLAY_CGLM_IVEC4_ZERO_INIT)

/*!
 * @brief init ivec4 using ivec3
 *
 * @param[in]  v3   vector3
 * @param[in]  last last item
 * @param[out] dest destination
 */
PLAY_CGLM_INLINE
void
ivec4_new(ivec3 v3, int last, ivec4 dest)
{
    dest[0] = v3[0];
    dest[1] = v3[1];
    dest[2] = v3[2];
    dest[3] = last;
}

/*!
 * @brief copy all members of [a] to [dest]
 *
 * @param[in]  a    source vector
 * @param[out] dest destination
 */
PLAY_CGLM_INLINE
void
ivec4_copy(ivec4 a, ivec4 dest)
{
    dest[0] = a[0];
    dest[1] = a[1];
    dest[2] = a[2];
    dest[3] = a[3];
}

/*!
 * @brief set all members of [v] to zero
 *
 * @param[out] v vector
 */
PLAY_CGLM_INLINE
void
ivec4_zero(ivec4 v)
{
    v[0] = v[1] = v[2] = v[3] = 0;
}

/*!
 * @brief set all members of [v] to one
 *
 * @param[out] v vector
 */
PLAY_CGLM_INLINE
void
ivec4_one(ivec4 v)
{
    v[0] = v[1] = v[2] = v[3] = 1;
}

/*!
 * @brief add vector [a] to vector [b] and store result in [dest]
 *
 * @param[in]  a    first vector
 * @param[in]  b    second vector
 * @param[out] dest destination
 */
PLAY_CGLM_INLINE
void
ivec4_add(ivec4 a, ivec4 b, ivec4 dest)
{
    dest[0] = a[0] + b[0];
    dest[1] = a[1] + b[1];
    dest[2] = a[2] + b[2];
    dest[3] = a[3] + b[3];
}

/*!
 * @brief add scalar s to vector [v] and store result in [dest]
 *
 * @param[in]  v    vector
 * @param[in]  s    scalar
 * @param[out] dest destination
 */
PLAY_CGLM_INLINE
void
ivec4_adds(ivec4 v, int s, ivec4 dest)
{
    dest[0] = v[0] + s;
    dest[1] = v[1] + s;
    dest[2] = v[2] + s;
    dest[3] = v[3] + s;
}

/*!
 * @brief subtract vector [b] from vector [a] and store result in [dest]
 *
 * @param[in]  a    first vector
 * @param[in]  b    second vector
 * @param[out] dest destination
 */
PLAY_CGLM_INLINE
void
ivec4_sub(ivec4 a, ivec4 b, ivec4 dest)
{
    dest[0] = a[0] - b[0];
    dest[1] = a[1] - b[1];
    dest[2] = a[2] - b[2];
    dest[3] = a[3] - b[3];
}

/*!
 * @brief subtract scalar s from vector [v] and store result in [dest]
 *
 * @param[in]  v    vector
 * @param[in]  s    scalar
 * @param[out] dest destination
 */
PLAY_CGLM_INLINE
void
ivec4_subs(ivec4 v, int s, ivec4 dest)
{
    dest[0] = v[0] - s;
    dest[1] = v[1] - s;
    dest[2] = v[2] - s;
    dest[3] = v[3] - s;
}

/*!
 * @brief multiply vector [a] with vector [b] and store result in [dest]
 *
 * @param[in]  a    first vector
 * @param[in]  b    second vector
 * @param[out] dest destination
 */
PLAY_CGLM_INLINE
void
ivec4_mul(ivec4 a, ivec4 b, ivec4 dest)
{
    dest[0] = a[0] * b[0];
    dest[1] = a[1] * b[1];
    dest[2] = a[2] * b[2];
    dest[3] = a[3] * b[3];
}

/*!
 * @brief multiply vector [a] with scalar s and store result in [dest]
 *
 * @param[in]  v    vector
 * @param[in]  s    scalar
 * @param[out] dest destination
 */
PLAY_CGLM_INLINE
void
ivec4_scale(ivec4 v, int s, ivec4 dest)
{
    dest[0] = v[0] * s;
    dest[1] = v[1] * s;
    dest[2] = v[2] * s;
    dest[3] = v[3] * s;
}

/*!
 * @brief add vector [a] with vector [b] and add result to vector [dest]
 *
 * applies += operator so dest must be initialized
 *
 * @param[in]  a    first vector
 * @param[in]  b    second vector
 * @param[out] dest dest += (a + b)
 */
PLAY_CGLM_INLINE
void
ivec4_addadd(ivec4 a, ivec4 b, ivec4 dest)
{
    dest[0] += a[0] + b[0];
    dest[1] += a[1] + b[1];
    dest[2] += a[2] + b[2];
    dest[3] += a[3] + b[3];
}

/*!
 * @brief add scalar [s] onto vector [a] and add result to vector [dest]
 *
 * applies += operator so dest must be initialized
 *
 * @param[in]  a    vector
 * @param[in]  s    scalar
 * @param[out] dest dest += (a + s)
 */
PLAY_CGLM_INLINE
void
ivec4_addadds(ivec4 a, int s, ivec4 dest)
{
    dest[0] += a[0] + s;
    dest[1] += a[1] + s;
    dest[2] += a[2] + s;
    dest[3] += a[3] + s;
}

/*!
 * @brief subtract vector [a] from vector [b] and add result to [dest]
 *
 * applies += operator so dest must be initialized
 *
 * @param[in]  a    first vector
 * @param[in]  b    second vector
 * @param[out] dest dest += (a - b)
 */
PLAY_CGLM_INLINE
void
ivec4_subadd(ivec4 a, ivec4 b, ivec4 dest)
{
    dest[0] += a[0] - b[0];
    dest[1] += a[1] - b[1];
    dest[2] += a[2] - b[2];
    dest[3] += a[3] - b[3];
}

/*!
 * @brief subtract scalar [s] from vector [a] and add result to [dest]
 *
 * applies += operator so dest must be initialized
 *
 * @param[in]  a    first
 * @param[in]  s    scalar
 * @param[out] dest dest += (a - s)
 */
PLAY_CGLM_INLINE
void
ivec4_subadds(ivec4 a, int s, ivec4 dest)
{
    dest[0] += a[0] - s;
    dest[1] += a[1] - s;
    dest[2] += a[2] - s;
    dest[3] += a[3] - s;
}

/*!
 * @brief multiply vector [a] with vector [b] and add result to [dest]
 *
 * applies += operator so dest must be initialized
 *
 * @param[in]  a    first vector
 * @param[in]  b    second vector
 * @param[out] dest dest += (a * b)
 */
PLAY_CGLM_INLINE
void
ivec4_muladd(ivec4 a, ivec4 b, ivec4 dest)
{
    dest[0] += a[0] * b[0];
    dest[1] += a[1] * b[1];
    dest[2] += a[2] * b[2];
    dest[3] += a[3] * b[3];
}

/*!
 * @brief multiply vector [a] with scalar [s] and add result to [dest]
 *
 * applies += operator so dest must be initialized
 *
 * @param[in]  a    vector
 * @param[in]  s    scalar
 * @param[out] dest dest += (a * s)
 */
PLAY_CGLM_INLINE
void
ivec4_muladds(ivec4 a, int s, ivec4 dest)
{
    dest[0] += a[0] * s;
    dest[1] += a[1] * s;
    dest[2] += a[2] * s;
    dest[3] += a[3] * s;
}

/*!
 * @brief add maximum of vector [a] and vector [b] to vector [dest]
 *
 * applies += operator so dest must be initialized
 *
 * @param[in]  a    first vector
 * @param[in]  b    second vector
 * @param[out] dest dest += fmax(a, b)
 */
PLAY_CGLM_INLINE
void
ivec4_maxadd(ivec4 a, ivec4 b, ivec4 dest)
{
    dest[0] += imax(a[0], b[0]);
    dest[1] += imax(a[1], b[1]);
    dest[2] += imax(a[2], b[2]);
    dest[3] += imax(a[3], b[3]);
}

/*!
 * @brief add minimum of vector [a] and vector [b] to vector [dest]
 *
 * applies += operator so dest must be initialized
 *
 * @param[in]  a    first vector
 * @param[in]  b    second vector
 * @param[out] dest dest += fmin(a, b)
 */
PLAY_CGLM_INLINE
void
ivec4_minadd(ivec4 a, ivec4 b, ivec4 dest)
{
    dest[0] += imin(a[0], b[0]);
    dest[1] += imin(a[1], b[1]);
    dest[2] += imin(a[2], b[2]);
    dest[3] += imin(a[3], b[3]);
}

/*!
 * @brief subtract vector [a] from vector [b] and subtract result from [dest]
 *
 * applies -= operator so dest must be initialized
 *
 * @param[in]  a    first vector
 * @param[in]  b    second vector
 * @param[out] dest dest -= (a - b)
 */
PLAY_CGLM_INLINE
void
ivec4_subsub(ivec4 a, ivec4 b, ivec4 dest)
{
    dest[0] -= a[0] - b[0];
    dest[1] -= a[1] - b[1];
    dest[2] -= a[2] - b[2];
    dest[3] -= a[3] - b[3];
}

/*!
 * @brief subtract scalar [s] from vector [a] and subtract result from [dest]
 *
 * applies -= operator so dest must be initialized
 *
 * @param[in]  a    vector
 * @param[in]  s    scalar
 * @param[out] dest dest -= (a - s)
 */
PLAY_CGLM_INLINE
void
ivec4_subsubs(ivec4 a, int s, ivec4 dest)
{
    dest[0] -= a[0] - s;
    dest[1] -= a[1] - s;
    dest[2] -= a[2] - s;
    dest[3] -= a[3] - s;
}

/*!
 * @brief add vector [a] to vector [b] and subtract the result from [dest]
 *
 * applies -= operator so dest must be initialized
 *
 * @param[in]  a    vector
 * @param[in]  b    scalar
 * @param[out] dest dest -= (a + b)
 */
PLAY_CGLM_INLINE
void
ivec4_addsub(ivec4 a, ivec4 b, ivec4 dest)
{
    dest[0] -= a[0] + b[0];
    dest[1] -= a[1] + b[1];
    dest[2] -= a[2] + b[2];
    dest[3] -= a[3] + b[3];
}

/*!
 * @brief add scalar [s] to vector [a] and subtract the result from [dest]
 *
 * applies -= operator so dest must be initialized
 *
 * @param[in]  a    vector
 * @param[in]  s    scalar
 * @param[out] dest dest -= (a + b)
 */
PLAY_CGLM_INLINE
void
ivec4_addsubs(ivec4 a, int s, ivec4 dest)
{
    dest[0] -= a[0] + s;
    dest[1] -= a[1] + s;
    dest[2] -= a[2] + s;
    dest[3] -= a[3] + s;
}

/*!
 * @brief multiply vector [a] and vector [b] and subtract the result from [dest]
 *
 * applies -= operator so dest must be initialized
 *
 * @param[in]  a    vector
 * @param[in]  b    scalar
 * @param[out] dest dest -= (a * b)
 */
PLAY_CGLM_INLINE
void
ivec4_mulsub(ivec4 a, ivec4 b, ivec4 dest)
{
    dest[0] -= a[0] * b[0];
    dest[1] -= a[1] * b[1];
    dest[2] -= a[2] * b[2];
    dest[3] -= a[3] * b[3];
}

/*!
 * @brief multiply vector [a] with scalar [s] and subtract the result from [dest]
 *
 * applies -= operator so dest must be initialized
 *
 * @param[in]  a    vector
 * @param[in]  s    scalar
 * @param[out] dest dest -= (a * s)
 */
PLAY_CGLM_INLINE
void
ivec4_mulsubs(ivec4 a, int s, ivec4 dest)
{
    dest[0] -= a[0] * s;
    dest[1] -= a[1] * s;
    dest[2] -= a[2] * s;
    dest[3] -= a[3] * s;
}

/*!
 * @brief subtract maximum of vector [a] and vector [b] from vector [dest]
 *
 * applies += operator so dest must be initialized
 *
 * @param[in]  a    first vector
 * @param[in]  b    second vector
 * @param[out] dest dest -= fmax(a, b)
 */
PLAY_CGLM_INLINE
void
ivec4_maxsub(ivec4 a, ivec4 b, ivec4 dest)
{
    dest[0] -= imax(a[0], b[0]);
    dest[1] -= imax(a[1], b[1]);
    dest[2] -= imax(a[2], b[2]);
    dest[3] -= imax(a[3], b[3]);
}

/*!
 * @brief subtract minimum of vector [a] and vector [b] from vector [dest]
 *
 * applies -= operator so dest must be initialized
 *
 * @param[in]  a    first vector
 * @param[in]  b    second vector
 * @param[out] dest dest -= fmin(a, b)
 */
PLAY_CGLM_INLINE
void
ivec4_minsub(ivec4 a, ivec4 b, ivec4 dest)
{
    dest[0] -= imin(a[0], b[0]);
    dest[1] -= imin(a[1], b[1]);
    dest[2] -= imin(a[2], b[2]);
    dest[3] -= imin(a[3], b[3]);
}

/*!
 * @brief squared distance between two vectors
 *
 * @param[in] a first vector
 * @param[in] b second vector
 * @return returns squared distance (distance * distance)
 */
PLAY_CGLM_INLINE
int
ivec4_distance2(ivec4 a, ivec4 b)
{
    int xd, yd, zd, wd;
    xd = a[0] - b[0];
    yd = a[1] - b[1];
    zd = a[2] - b[2];
    wd = a[3] - b[3];
    return xd * xd + yd * yd + zd * zd + wd * wd;
}

/*!
 * @brief distance between two vectors
 *
 * @param[in] a first vector
 * @param[in] b second vector
 * @return returns distance
 */
PLAY_CGLM_INLINE
float
ivec4_distance(ivec4 a, ivec4 b)
{
    return sqrtf((float)ivec4_distance2(a, b));
}

/*!
 * @brief set each member of dest to greater of vector a and b
 *
 * @param[in]  a    first vector
 * @param[in]  b    second vector
 * @param[out] dest destination
 */
PLAY_CGLM_INLINE
void
ivec4_maxv(ivec4 a, ivec4 b, ivec4 dest)
{
    dest[0] = a[0] > b[0] ? a[0] : b[0];
    dest[1] = a[1] > b[1] ? a[1] : b[1];
    dest[2] = a[2] > b[2] ? a[2] : b[2];
    dest[3] = a[3] > b[3] ? a[3] : b[3];
}

/*!
 * @brief set each member of dest to lesser of vector a and b
 *
 * @param[in]  a    first vector
 * @param[in]  b    second vector
 * @param[out] dest destination
 */
PLAY_CGLM_INLINE
void
ivec4_minv(ivec4 a, ivec4 b, ivec4 dest)
{
    dest[0] = a[0] < b[0] ? a[0] : b[0];
    dest[1] = a[1] < b[1] ? a[1] : b[1];
    dest[2] = a[2] < b[2] ? a[2] : b[2];
    dest[3] = a[3] < b[3] ? a[3] : b[3];
}

/*!
 * @brief clamp each member of [v] between minVal and maxVal (inclusive)
 *
 * @param[in, out] v      vector
 * @param[in]      minVal minimum value
 * @param[in]      maxVal maximum value
 */
PLAY_CGLM_INLINE
void
ivec4_clamp(ivec4 v, int minVal, int maxVal)
{
    if (v[0] < minVal)
        v[0] = minVal;
    else if(v[0] > maxVal)
        v[0] = maxVal;

    if (v[1] < minVal)
        v[1] = minVal;
    else if(v[1] > maxVal)
        v[1] = maxVal;

    if (v[2] < minVal)
        v[2] = minVal;
    else if(v[2] > maxVal)
        v[2] = maxVal;

    if (v[3] < minVal)
        v[3] = minVal;
    else if(v[3] > maxVal)
        v[3] = maxVal;
}

/*!
 * @brief absolute value of v
 *
 * @param[in]	v	vector
 * @param[out]	dest	destination
 */
PLAY_CGLM_INLINE
void
ivec4_abs(ivec4 v, ivec4 dest)
{
    dest[0] = abs(v[0]);
    dest[1] = abs(v[1]);
    dest[2] = abs(v[2]);
    dest[3] = abs(v[3]);
}

#endif /* civec4_h */

/*** End of inlined file: ivec4.h ***/


/*** Start of inlined file: mat4.h ***/
/*!
 * Most of functions in this header are optimized manually with SIMD
 * if available. You dont need to call/incude SIMD headers manually
 */

/*
 Macros:
   PLAY_CGLM_MAT4_IDENTITY_INIT
   PLAY_CGLM_MAT4_ZERO_INIT
   PLAY_CGLM_MAT4_IDENTITY
   PLAY_CGLM_MAT4_ZERO

 Functions:
   PLAY_CGLM_INLINE void  mat4_ucopy(mat4 mat, mat4 dest);
   PLAY_CGLM_INLINE void  mat4_copy(mat4 mat, mat4 dest);
   PLAY_CGLM_INLINE void  mat4_identity(mat4 mat);
   PLAY_CGLM_INLINE void  mat4_identity_array(mat4 * restrict mat, size_t count);
   PLAY_CGLM_INLINE void  mat4_zero(mat4 mat);
   PLAY_CGLM_INLINE void  mat4_pick3(mat4 mat, mat3 dest);
   PLAY_CGLM_INLINE void  mat4_pick3t(mat4 mat, mat3 dest);
   PLAY_CGLM_INLINE void  mat4_ins3(mat3 mat, mat4 dest);
   PLAY_CGLM_INLINE void  mat4_mul(mat4 m1, mat4 m2, mat4 dest);
   PLAY_CGLM_INLINE void  mat4_mulN(mat4 *matrices[], int len, mat4 dest);
   PLAY_CGLM_INLINE void  mat4_mulv(mat4 m, vec4 v, vec4 dest);
   PLAY_CGLM_INLINE void  mat4_mulv3(mat4 m, vec3 v, float last, vec3 dest);
   PLAY_CGLM_INLINE float mat4_trace(mat4 m);
   PLAY_CGLM_INLINE float mat4_trace3(mat4 m);
   PLAY_CGLM_INLINE void  mat4_quat(mat4 m, versor dest) ;
   PLAY_CGLM_INLINE void  mat4_transpose_to(mat4 m, mat4 dest);
   PLAY_CGLM_INLINE void  mat4_transpose(mat4 m);
   PLAY_CGLM_INLINE void  mat4_scale_p(mat4 m, float s);
   PLAY_CGLM_INLINE void  mat4_scale(mat4 m, float s);
   PLAY_CGLM_INLINE float mat4_det(mat4 mat);
   PLAY_CGLM_INLINE void  mat4_inv(mat4 mat, mat4 dest);
   PLAY_CGLM_INLINE void  mat4_inv_fast(mat4 mat, mat4 dest);
   PLAY_CGLM_INLINE void  mat4_swap_col(mat4 mat, int col1, int col2);
   PLAY_CGLM_INLINE void  mat4_swap_row(mat4 mat, int row1, int row2);
   PLAY_CGLM_INLINE float mat4_rmc(vec4 r, mat4 m, vec4 c);
   PLAY_CGLM_INLINE void  mat4_make(float * restrict src, mat4 dest);
   PLAY_CGLM_INLINE void  mat4_textrans(float sx, float sy, float rot, float tx, float ty, mat4 dest);
 */




#ifdef PLAY_CGLM_SSE_FP

/*** Start of inlined file: mat4.h ***/


#if defined( __SSE__ ) || defined( __SSE2__ )

#define mat4_inv_precise_sse2(mat, dest) mat4_inv_sse2(mat, dest)

PLAY_CGLM_INLINE
void
mat4_scale_sse2(mat4 m, float s)
{
    __m128 x0;
    x0 = glmm_set1(s);

    glmm_store(m[0], _mm_mul_ps(glmm_load(m[0]), x0));
    glmm_store(m[1], _mm_mul_ps(glmm_load(m[1]), x0));
    glmm_store(m[2], _mm_mul_ps(glmm_load(m[2]), x0));
    glmm_store(m[3], _mm_mul_ps(glmm_load(m[3]), x0));
}

PLAY_CGLM_INLINE
void
mat4_transp_sse2(mat4 m, mat4 dest)
{
    __m128 r0, r1, r2, r3;

    r0 = glmm_load(m[0]);
    r1 = glmm_load(m[1]);
    r2 = glmm_load(m[2]);
    r3 = glmm_load(m[3]);

    _MM_TRANSPOSE4_PS(r0, r1, r2, r3);

    glmm_store(dest[0], r0);
    glmm_store(dest[1], r1);
    glmm_store(dest[2], r2);
    glmm_store(dest[3], r3);
}

PLAY_CGLM_INLINE
void
mat4_mul_sse2(mat4 m1, mat4 m2, mat4 dest)
{
    /* D = R * L (Column-Major) */

    glmm_128 l, r0, r1, r2, r3, v0, v1, v2, v3;

    l  = glmm_load(m1[0]);
    r0 = glmm_load(m2[0]);
    r1 = glmm_load(m2[1]);
    r2 = glmm_load(m2[2]);
    r3 = glmm_load(m2[3]);

    v0 = _mm_mul_ps(glmm_splat_x(r0), l);
    v1 = _mm_mul_ps(glmm_splat_x(r1), l);
    v2 = _mm_mul_ps(glmm_splat_x(r2), l);
    v3 = _mm_mul_ps(glmm_splat_x(r3), l);

    l  = glmm_load(m1[1]);
    v0 = glmm_fmadd(glmm_splat_y(r0), l, v0);
    v1 = glmm_fmadd(glmm_splat_y(r1), l, v1);
    v2 = glmm_fmadd(glmm_splat_y(r2), l, v2);
    v3 = glmm_fmadd(glmm_splat_y(r3), l, v3);

    l  = glmm_load(m1[2]);
    v0 = glmm_fmadd(glmm_splat_z(r0), l, v0);
    v1 = glmm_fmadd(glmm_splat_z(r1), l, v1);
    v2 = glmm_fmadd(glmm_splat_z(r2), l, v2);
    v3 = glmm_fmadd(glmm_splat_z(r3), l, v3);

    l  = glmm_load(m1[3]);
    v0 = glmm_fmadd(glmm_splat_w(r0), l, v0);
    v1 = glmm_fmadd(glmm_splat_w(r1), l, v1);
    v2 = glmm_fmadd(glmm_splat_w(r2), l, v2);
    v3 = glmm_fmadd(glmm_splat_w(r3), l, v3);

    glmm_store(dest[0], v0);
    glmm_store(dest[1], v1);
    glmm_store(dest[2], v2);
    glmm_store(dest[3], v3);
}

PLAY_CGLM_INLINE
void
mat4_mulv_sse2(mat4 m, vec4 v, vec4 dest)
{
    __m128 x0, x1, m0, m1, m2, m3, v0, v1, v2, v3;

    m0 = glmm_load(m[0]);
    m1 = glmm_load(m[1]);
    m2 = glmm_load(m[2]);
    m3 = glmm_load(m[3]);

    x0 = glmm_load(v);
    v0 = glmm_splat_x(x0);
    v1 = glmm_splat_y(x0);
    v2 = glmm_splat_z(x0);
    v3 = glmm_splat_w(x0);

    x1 = _mm_mul_ps(m3, v3);
    x1 = glmm_fmadd(m2, v2, x1);
    x1 = glmm_fmadd(m1, v1, x1);
    x1 = glmm_fmadd(m0, v0, x1);

    glmm_store(dest, x1);
}

PLAY_CGLM_INLINE
float
mat4_det_sse2(mat4 mat)
{
    __m128 r0, r1, r2, r3, x0, x1, x2;

    /* 127 <- 0, [square] det(A) = det(At) */
    r0 = glmm_load(mat[0]); /* d c b a */
    r1 = glmm_load(mat[1]); /* h g f e */
    r2 = glmm_load(mat[2]); /* l k j i */
    r3 = glmm_load(mat[3]); /* p o n m */

    /*
     t[1] = j * p - n * l;
     t[2] = j * o - n * k;
     t[3] = i * p - m * l;
     t[4] = i * o - m * k;
     */
    x0 = glmm_fnmadd(glmm_shuff1(r3, 0, 0, 1, 1), glmm_shuff1(r2, 2, 3, 2, 3),
                     _mm_mul_ps(glmm_shuff1(r2, 0, 0, 1, 1),
                                glmm_shuff1(r3, 2, 3, 2, 3)));
    /*
     t[0] = k * p - o * l;
     t[0] = k * p - o * l;
     t[5] = i * n - m * j;
     t[5] = i * n - m * j;
     */
    x1 = glmm_fnmadd(glmm_shuff1(r3, 0, 0, 2, 2), glmm_shuff1(r2, 1, 1, 3, 3),
                     _mm_mul_ps(glmm_shuff1(r2, 0, 0, 2, 2),
                                glmm_shuff1(r3, 1, 1, 3, 3)));

    /*
       a * (f * t[0] - g * t[1] + h * t[2])
     - b * (e * t[0] - g * t[3] + h * t[4])
     + c * (e * t[1] - f * t[3] + h * t[5])
     - d * (e * t[2] - f * t[4] + g * t[5])
     */
    x2 = glmm_fnmadd(glmm_shuff1(r1, 1, 1, 2, 2), glmm_shuff1(x0, 3, 2, 2, 0),
                     _mm_mul_ps(glmm_shuff1(r1, 0, 0, 0, 1),
                                _mm_shuffle_ps(x1, x0, _MM_SHUFFLE(1, 0, 0, 0))));
    x2 = glmm_fmadd(glmm_shuff1(r1, 2, 3, 3, 3),
                    _mm_shuffle_ps(x0, x1, _MM_SHUFFLE(2, 2, 3, 1)),
                    x2);

    x2 = _mm_xor_ps(x2, glmm_float32x4_SIGNMASK_NPNP);

    return glmm_hadd(_mm_mul_ps(x2, r0));
}

PLAY_CGLM_INLINE
void
mat4_inv_fast_sse2(mat4 mat, mat4 dest)
{
    __m128 r0, r1, r2, r3,
           v0, v1, v2, v3,
           t0, t1, t2, t3, t4, t5,
           x0, x1, x2, x3, x4, x5, x6, x7, x8, x9;

    /* x8 = _mm_set_ps(-0.f, 0.f, -0.f, 0.f); */
    x8 = glmm_float32x4_SIGNMASK_NPNP;
    x9 = glmm_shuff1(x8, 2, 1, 2, 1);

    /* 127 <- 0 */
    r0 = glmm_load(mat[0]); /* d c b a */
    r1 = glmm_load(mat[1]); /* h g f e */
    r2 = glmm_load(mat[2]); /* l k j i */
    r3 = glmm_load(mat[3]); /* p o n m */

    x0 = _mm_movehl_ps(r3, r2);                            /* p o l k */
    x3 = _mm_movelh_ps(r2, r3);                            /* n m j i */
    x1 = glmm_shuff1(x0, 1, 3, 3,3);                       /* l p p p */
    x2 = glmm_shuff1(x0, 0, 2, 2, 2);                      /* k o o o */
    x4 = glmm_shuff1(x3, 1, 3, 3, 3);                      /* j n n n */
    x7 = glmm_shuff1(x3, 0, 2, 2, 2);                      /* i m m m */

    x6 = _mm_shuffle_ps(r2, r1, _MM_SHUFFLE(0, 0, 0, 0));  /* e e i i */
    x5 = _mm_shuffle_ps(r2, r1, _MM_SHUFFLE(1, 1, 1, 1));  /* f f j j */
    x3 = _mm_shuffle_ps(r2, r1, _MM_SHUFFLE(2, 2, 2, 2));  /* g g k k */
    x0 = _mm_shuffle_ps(r2, r1, _MM_SHUFFLE(3, 3, 3, 3));  /* h h l l */

    t0 = _mm_mul_ps(x3, x1);
    t1 = _mm_mul_ps(x5, x1);
    t2 = _mm_mul_ps(x5, x2);
    t3 = _mm_mul_ps(x6, x1);
    t4 = _mm_mul_ps(x6, x2);
    t5 = _mm_mul_ps(x6, x4);

    /* t1[0] = k * p - o * l;
       t1[0] = k * p - o * l;
       t2[0] = g * p - o * h;
       t3[0] = g * l - k * h; */
    t0 = glmm_fnmadd(x2, x0, t0);

    /* t1[1] = j * p - n * l;
       t1[1] = j * p - n * l;
       t2[1] = f * p - n * h;
       t3[1] = f * l - j * h; */
    t1 = glmm_fnmadd(x4, x0, t1);

    /* t1[2] = j * o - n * k
       t1[2] = j * o - n * k;
       t2[2] = f * o - n * g;
       t3[2] = f * k - j * g; */
    t2 = glmm_fnmadd(x4, x3, t2);

    /* t1[3] = i * p - m * l;
       t1[3] = i * p - m * l;
       t2[3] = e * p - m * h;
       t3[3] = e * l - i * h; */
    t3 = glmm_fnmadd(x7, x0, t3);

    /* t1[4] = i * o - m * k;
       t1[4] = i * o - m * k;
       t2[4] = e * o - m * g;
       t3[4] = e * k - i * g; */
    t4 = glmm_fnmadd(x7, x3, t4);

    /* t1[5] = i * n - m * j;
       t1[5] = i * n - m * j;
       t2[5] = e * n - m * f;
       t3[5] = e * j - i * f; */
    t5 = glmm_fnmadd(x7, x5, t5);

    x4 = _mm_movelh_ps(r0, r1);        /* f e b a */
    x5 = _mm_movehl_ps(r1, r0);        /* h g d c */

    x0 = glmm_shuff1(x4, 0, 0, 0, 2);  /* a a a e */
    x1 = glmm_shuff1(x4, 1, 1, 1, 3);  /* b b b f */
    x2 = glmm_shuff1(x5, 0, 0, 0, 2);  /* c c c g */
    x3 = glmm_shuff1(x5, 1, 1, 1, 3);  /* d d d h */

    v2 = _mm_mul_ps(x0, t1);
    v1 = _mm_mul_ps(x0, t0);
    v3 = _mm_mul_ps(x0, t2);
    v0 = _mm_mul_ps(x1, t0);

    v2 = glmm_fnmadd(x1, t3, v2);
    v3 = glmm_fnmadd(x1, t4, v3);
    v0 = glmm_fnmadd(x2, t1, v0);
    v1 = glmm_fnmadd(x2, t3, v1);

    v3 = glmm_fmadd(x2, t5, v3);
    v0 = glmm_fmadd(x3, t2, v0);
    v2 = glmm_fmadd(x3, t5, v2);
    v1 = glmm_fmadd(x3, t4, v1);

    /*
     dest[0][0] =  f * t1[0] - g * t1[1] + h * t1[2];
     dest[0][1] =-(b * t1[0] - c * t1[1] + d * t1[2]);
     dest[0][2] =  b * t2[0] - c * t2[1] + d * t2[2];
     dest[0][3] =-(b * t3[0] - c * t3[1] + d * t3[2]); */
    v0 = _mm_xor_ps(v0, x8);

    /*
     dest[2][0] =  e * t1[1] - f * t1[3] + h * t1[5];
     dest[2][1] =-(a * t1[1] - b * t1[3] + d * t1[5]);
     dest[2][2] =  a * t2[1] - b * t2[3] + d * t2[5];
     dest[2][3] =-(a * t3[1] - b * t3[3] + d * t3[5]);*/
    v2 = _mm_xor_ps(v2, x8);

    /*
     dest[1][0] =-(e * t1[0] - g * t1[3] + h * t1[4]);
     dest[1][1] =  a * t1[0] - c * t1[3] + d * t1[4];
     dest[1][2] =-(a * t2[0] - c * t2[3] + d * t2[4]);
     dest[1][3] =  a * t3[0] - c * t3[3] + d * t3[4]; */
    v1 = _mm_xor_ps(v1, x9);

    /*
     dest[3][0] =-(e * t1[2] - f * t1[4] + g * t1[5]);
     dest[3][1] =  a * t1[2] - b * t1[4] + c * t1[5];
     dest[3][2] =-(a * t2[2] - b * t2[4] + c * t2[5]);
     dest[3][3] =  a * t3[2] - b * t3[4] + c * t3[5]; */
    v3 = _mm_xor_ps(v3, x9);

    /* determinant */
    x0 = _mm_shuffle_ps(v0, v1, _MM_SHUFFLE(0, 0, 0, 0));
    x1 = _mm_shuffle_ps(v2, v3, _MM_SHUFFLE(0, 0, 0, 0));
    x0 = _mm_shuffle_ps(x0, x1, _MM_SHUFFLE(2, 0, 2, 0));

    x0 = _mm_rcp_ps(glmm_vhadd(_mm_mul_ps(x0, r0)));

    glmm_store(dest[0], _mm_mul_ps(v0, x0));
    glmm_store(dest[1], _mm_mul_ps(v1, x0));
    glmm_store(dest[2], _mm_mul_ps(v2, x0));
    glmm_store(dest[3], _mm_mul_ps(v3, x0));
}

/* old one */
#if 0
PLAY_CGLM_INLINE
void
mat4_inv_sse2(mat4 mat, mat4 dest)
{
    __m128 r0, r1, r2, r3,
           v0, v1, v2, v3,
           t0, t1, t2, t3, t4, t5,
           x0, x1, x2, x3, x4, x5, x6, x7, x8, x9;

    /* x8 = _mm_set_ps(-0.f, 0.f, -0.f, 0.f); */
    x8 = glmm_float32x4_SIGNMASK_NPNP;
    x9 = glmm_shuff1(x8, 2, 1, 2, 1);

    /* 127 <- 0 */
    r0 = glmm_load(mat[0]); /* d c b a */
    r1 = glmm_load(mat[1]); /* h g f e */
    r2 = glmm_load(mat[2]); /* l k j i */
    r3 = glmm_load(mat[3]); /* p o n m */

    x0 = _mm_movehl_ps(r3, r2);                            /* p o l k */
    x3 = _mm_movelh_ps(r2, r3);                            /* n m j i */
    x1 = glmm_shuff1(x0, 1, 3, 3,3);                       /* l p p p */
    x2 = glmm_shuff1(x0, 0, 2, 2, 2);                      /* k o o o */
    x4 = glmm_shuff1(x3, 1, 3, 3, 3);                      /* j n n n */
    x7 = glmm_shuff1(x3, 0, 2, 2, 2);                      /* i m m m */

    x6 = _mm_shuffle_ps(r2, r1, _MM_SHUFFLE(0, 0, 0, 0));  /* e e i i */
    x5 = _mm_shuffle_ps(r2, r1, _MM_SHUFFLE(1, 1, 1, 1));  /* f f j j */
    x3 = _mm_shuffle_ps(r2, r1, _MM_SHUFFLE(2, 2, 2, 2));  /* g g k k */
    x0 = _mm_shuffle_ps(r2, r1, _MM_SHUFFLE(3, 3, 3, 3));  /* h h l l */

    t0 = _mm_mul_ps(x3, x1);
    t1 = _mm_mul_ps(x5, x1);
    t2 = _mm_mul_ps(x5, x2);
    t3 = _mm_mul_ps(x6, x1);
    t4 = _mm_mul_ps(x6, x2);
    t5 = _mm_mul_ps(x6, x4);

    /* t1[0] = k * p - o * l;
       t1[0] = k * p - o * l;
       t2[0] = g * p - o * h;
       t3[0] = g * l - k * h; */
    t0 = glmm_fnmadd(x2, x0, t0);

    /* t1[1] = j * p - n * l;
       t1[1] = j * p - n * l;
       t2[1] = f * p - n * h;
       t3[1] = f * l - j * h; */
    t1 = glmm_fnmadd(x4, x0, t1);

    /* t1[2] = j * o - n * k
       t1[2] = j * o - n * k;
       t2[2] = f * o - n * g;
       t3[2] = f * k - j * g; */
    t2 = glmm_fnmadd(x4, x3, t2);

    /* t1[3] = i * p - m * l;
       t1[3] = i * p - m * l;
       t2[3] = e * p - m * h;
       t3[3] = e * l - i * h; */
    t3 = glmm_fnmadd(x7, x0, t3);

    /* t1[4] = i * o - m * k;
       t1[4] = i * o - m * k;
       t2[4] = e * o - m * g;
       t3[4] = e * k - i * g; */
    t4 = glmm_fnmadd(x7, x3, t4);

    /* t1[5] = i * n - m * j;
       t1[5] = i * n - m * j;
       t2[5] = e * n - m * f;
       t3[5] = e * j - i * f; */
    t5 = glmm_fnmadd(x7, x5, t5);

    x4 = _mm_movelh_ps(r0, r1);        /* f e b a */
    x5 = _mm_movehl_ps(r1, r0);        /* h g d c */

    x0 = glmm_shuff1(x4, 0, 0, 0, 2);  /* a a a e */
    x1 = glmm_shuff1(x4, 1, 1, 1, 3);  /* b b b f */
    x2 = glmm_shuff1(x5, 0, 0, 0, 2);  /* c c c g */
    x3 = glmm_shuff1(x5, 1, 1, 1, 3);  /* d d d h */

    v2 = _mm_mul_ps(x0, t1);
    v1 = _mm_mul_ps(x0, t0);
    v3 = _mm_mul_ps(x0, t2);
    v0 = _mm_mul_ps(x1, t0);

    v2 = glmm_fnmadd(x1, t3, v2);
    v3 = glmm_fnmadd(x1, t4, v3);
    v0 = glmm_fnmadd(x2, t1, v0);
    v1 = glmm_fnmadd(x2, t3, v1);

    v3 = glmm_fmadd(x2, t5, v3);
    v0 = glmm_fmadd(x3, t2, v0);
    v2 = glmm_fmadd(x3, t5, v2);
    v1 = glmm_fmadd(x3, t4, v1);

    /*
     dest[0][0] =  f * t1[0] - g * t1[1] + h * t1[2];
     dest[0][1] =-(b * t1[0] - c * t1[1] + d * t1[2]);
     dest[0][2] =  b * t2[0] - c * t2[1] + d * t2[2];
     dest[0][3] =-(b * t3[0] - c * t3[1] + d * t3[2]); */
    v0 = _mm_xor_ps(v0, x8);

    /*
     dest[2][0] =  e * t1[1] - f * t1[3] + h * t1[5];
     dest[2][1] =-(a * t1[1] - b * t1[3] + d * t1[5]);
     dest[2][2] =  a * t2[1] - b * t2[3] + d * t2[5];
     dest[2][3] =-(a * t3[1] - b * t3[3] + d * t3[5]);*/
    v2 = _mm_xor_ps(v2, x8);

    /*
     dest[1][0] =-(e * t1[0] - g * t1[3] + h * t1[4]);
     dest[1][1] =  a * t1[0] - c * t1[3] + d * t1[4];
     dest[1][2] =-(a * t2[0] - c * t2[3] + d * t2[4]);
     dest[1][3] =  a * t3[0] - c * t3[3] + d * t3[4]; */
    v1 = _mm_xor_ps(v1, x9);

    /*
     dest[3][0] =-(e * t1[2] - f * t1[4] + g * t1[5]);
     dest[3][1] =  a * t1[2] - b * t1[4] + c * t1[5];
     dest[3][2] =-(a * t2[2] - b * t2[4] + c * t2[5]);
     dest[3][3] =  a * t3[2] - b * t3[4] + c * t3[5]; */
    v3 = _mm_xor_ps(v3, x9);

    /* determinant */
    x0 = _mm_shuffle_ps(v0, v1, _MM_SHUFFLE(0, 0, 0, 0));
    x1 = _mm_shuffle_ps(v2, v3, _MM_SHUFFLE(0, 0, 0, 0));
    x0 = _mm_shuffle_ps(x0, x1, _MM_SHUFFLE(2, 0, 2, 0));

    x0 = _mm_div_ps(glmm_set1(1.0f), glmm_vhadd(_mm_mul_ps(x0, r0)));

    glmm_store(dest[0], _mm_mul_ps(v0, x0));
    glmm_store(dest[1], _mm_mul_ps(v1, x0));
    glmm_store(dest[2], _mm_mul_ps(v2, x0));
    glmm_store(dest[3], _mm_mul_ps(v3, x0));
}
#endif

PLAY_CGLM_INLINE
void
mat4_inv_sse2(mat4 mat, mat4 dest)
{
    __m128 r0, r1, r2, r3, s1, s2,
           v0, v1, v2, v3, v4, v5,
           t0, t1, t2,
           x0, x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12, x13;

    /* s1 = _mm_set_ps(-0.f, 0.f, -0.f, 0.f); */
    s1 = glmm_float32x4_SIGNMASK_NPNP;
    s2 = glmm_shuff1(s1, 2, 1, 2, 1);

    /* 127 <- 0 */
    r1 = glmm_load(mat[1]); /* h g f e */
    r0 = glmm_load(mat[0]); /* d c b a */
    r3 = glmm_load(mat[3]); /* p o n m */
    r2 = glmm_load(mat[2]); /* l k j i */

    x4  = _mm_unpackhi_ps(r0, r2); /* l d k c */
    x5  = _mm_unpacklo_ps(r0, r2); /* j b i a */
    x6  = _mm_unpackhi_ps(r1, r3); /* p h o g */
    x7  = _mm_unpacklo_ps(r1, r3); /* n f m e */

    x0  = _mm_unpackhi_ps(x7, x5); /* j n b f */
    x1  = _mm_unpacklo_ps(x7, x5); /* i m a e */
    x2  = _mm_unpackhi_ps(x6, x4); /* l p d h */
    x3  = _mm_unpacklo_ps(x6, x4); /* k o c g */

    /* c2 = c * h - d * g   c12 = a * g - c * e    c8  = a * f - b * e
       c1 = k * p - l * o   c11 = i * o - k * m    c7  = i * n - j * m
       c4 = a * h - d * e   c6  = b * h - d * f    c10 = b * g - c * f
       c3 = i * p - l * m   c5  = j * p - l * n    c9  = j * o - k * n */

    x8  = _mm_shuffle_ps(x0, x3, _MM_SHUFFLE(3, 1, 3, 1)); /* k c j b */
    x9  = _mm_shuffle_ps(x0, x3, _MM_SHUFFLE(2, 0, 2, 0)); /* o g n f */

    x10 = glmm_shuff1(x2, 2, 0, 2, 0);                     /* p h p h */
    x11 = glmm_shuff1(x2, 3, 1, 3, 1);                     /* l d l d */

#if 0 /* TODO measure both */
    x12 = _mm_shuffle_ps(x4, x5, _MM_SHUFFLE(1, 0, 1, 0)); /* i a k c */
    x13 = _mm_shuffle_ps(x6, x7, _MM_SHUFFLE(1, 0, 1, 0)); /* m e o g */
#else
    x12 = _mm_movelh_ps(x4, x5);                           /* i a k c */
    x13 = _mm_movelh_ps(x6, x7);                           /* m e o g */
#endif

    t0 = _mm_mul_ps(x12, x10);
    t1 = _mm_mul_ps(x5, x6);
    t2 = _mm_mul_ps(x5, x9);

    t0 = glmm_fnmadd(x11, x13, t0);
    t1 = glmm_fnmadd(x4, x7, t1);
    t2 = glmm_fnmadd(x8, x7, t2);

    /* det */
    /* v0: c3 * c10 + c4 * c9 + c1 * c8 + c2 * c7 */
    /* v1: c5 * c12 + c6 * c11 */

    v5 = glmm_set1_rval(1.0f);
    v0 = glmm_shuff1(t2, 2, 3, 0, 1);
    v1 = glmm_shuff1(t1, 0, 1, 2, 3);
    v0 = _mm_mul_ps(t0, v0);
    v1 = _mm_mul_ps(t1, v1);
    v2 = glmm_shuff1(v1, 1, 0, 0, 1);
    v3 = glmm_shuff1(v0, 0, 1, 2, 3);
    v1 = _mm_add_ps(v1, v2);
    v0 = _mm_add_ps(v0, v3);
    v2 = glmm_shuff1(v0, 1, 0, 0, 1);
    v0 = _mm_add_ps(v0, v2);

    v0 = _mm_sub_ps(v0, v1); /* det */
    v0 = _mm_div_ps(v5, v0); /* idt */

    /* multiply t0,t1,t2 by idt to reduce 1mul below: 2eor+4mul vs 3mul+4eor */
    t0 = _mm_mul_ps(t0, v0);
    t1 = _mm_mul_ps(t1, v0);
    t2 = _mm_mul_ps(t2, v0);

    v0 = glmm_shuff1(t0, 0, 0, 1, 1); /* c2  c2  c1  c1  */
    v1 = glmm_shuff1(t0, 2, 2, 3, 3); /* c4  c4  c3 c3   */
    v2 = glmm_shuff1(t1, 0, 0, 1, 1); /* c12 c12 c11 c11 */
    v3 = glmm_shuff1(t1, 2, 2, 3, 3); /* c6  c6  c5 c5   */
    v4 = glmm_shuff1(t2, 0, 0, 1, 1); /* c8  c8  c7  c7  */
    v5 = glmm_shuff1(t2, 2, 2, 3, 3); /* c10 c10 c9 c9   */

    /* result */

    /* dest[0][0] = (f * c1  - g * c5  + h * c9)  * idt;
       dest[0][1] = (b * c1  - c * c5  + d * c9)  * ndt;
       dest[0][2] = (n * c2  - o * c6  + p * c10) * idt;
       dest[0][3] = (j * c2  - k * c6  + l * c10) * ndt;

       dest[1][0] = (e * c1  - g * c3  + h * c11) * ndt;
       dest[1][1] = (a * c1  - c * c3  + d * c11) * idt;
       dest[1][2] = (m * c2  - o * c4  + p * c12) * ndt;
       dest[1][3] = (i * c2  - k * c4  + l * c12) * idt;

       dest[2][0] = (e * c5  - f * c3  + h * c7)  * idt;
       dest[2][1] = (a * c5  - b * c3  + d * c7)  * ndt;
       dest[2][2] = (m * c6  - n * c4  + p * c8)  * idt;
       dest[2][3] = (i * c6  - j * c4  + l * c8)  * ndt;

       dest[3][0] = (e * c9  - f * c11 + g * c7)  * ndt;
       dest[3][1] = (a * c9  - b * c11 + c * c7)  * idt;
       dest[3][2] = (m * c10 - n * c12 + o * c8)  * ndt;
       dest[3][3] = (i * c10 - j * c12 + k * c8)  * idt; */

    r0 = _mm_mul_ps(x0, v0);
    r1 = _mm_mul_ps(x1, v0);
    r2 = _mm_mul_ps(x1, v3);
    r3 = _mm_mul_ps(x1, v5);

    r0 = glmm_fnmadd(x3, v3, r0);
    r1 = glmm_fnmadd(x3, v1, r1);
    r2 = glmm_fnmadd(x0, v1, r2);
    r3 = glmm_fnmadd(x0, v2, r3);

    r0 = glmm_fmadd(x2, v5, r0);
    r1 = glmm_fmadd(x2, v2, r1);
    r2 = glmm_fmadd(x2, v4, r2);
    r3 = glmm_fmadd(x3, v4, r3);

    /* 4xor may be fastart then 4mul, see above  */
    r0 = _mm_xor_ps(r0, s1);
    r1 = _mm_xor_ps(r1, s2);
    r2 = _mm_xor_ps(r2, s1);
    r3 = _mm_xor_ps(r3, s2);

    glmm_store(dest[0], r0);
    glmm_store(dest[1], r1);
    glmm_store(dest[2], r2);
    glmm_store(dest[3], r3);
}
#endif


/*** End of inlined file: mat4.h ***/


#endif

#ifdef PLAY_CGLM_AVX_FP

/*** Start of inlined file: mat4.h ***/


#ifdef __AVX__

PLAY_CGLM_INLINE
void
mat4_scale_avx(mat4 m, float s)
{
    __m256 y0, y1, y2, y3, y4;

    y0 = glmm_load256(m[0]);            /* h g f e d c b a */
    y1 = glmm_load256(m[2]);            /* p o n m l k j i */

    y2 = _mm256_broadcast_ss(&s);

    y3 = _mm256_mul_ps(y0, y2);
    y4 = _mm256_mul_ps(y1, y2);

    glmm_store256(m[0], y3);
    glmm_store256(m[2], y4);
}

/* TODO: this must be tested and compared to SSE version, may be slower!!! */
PLAY_CGLM_INLINE
void
mat4_transp_avx(mat4 m, mat4 dest)
{
    __m256 y0, y1, y2, y3;

    y0 = glmm_load256(m[0]);                   /* h g f e d c b a */
    y1 = glmm_load256(m[2]);                   /* p o n m l k j i */

    y2 = _mm256_unpacklo_ps(y0, y1);           /* n f m e j b i a */
    y3 = _mm256_unpackhi_ps(y0, y1);           /* p h o g l d k c */

    y0 = _mm256_permute2f128_ps(y2, y3, 0x20); /* l d k c j b i a */
    y1 = _mm256_permute2f128_ps(y2, y3, 0x31); /* p h o g n f m e */

    y2 = _mm256_unpacklo_ps(y0, y1);           /* o k g c m i e a */
    y3 = _mm256_unpackhi_ps(y0, y1);           /* p l h d n j f b */

    y0 = _mm256_permute2f128_ps(y2, y3, 0x20); /* n j f b m i e a */
    y1 = _mm256_permute2f128_ps(y2, y3, 0x31); /* p l h d o k g c */

    glmm_store256(dest[0], y0);
    glmm_store256(dest[2], y1);
}

PLAY_CGLM_INLINE
void
mat4_mul_avx(mat4 m1, mat4 m2, mat4 dest)
{
    /* D = R * L (Column-Major) */

    __m256  y0, y1, y2, y3, y4, y5, y6, y7, y8, y9, y10, y11, y12, y13;
    __m256i yi0, yi1, yi2, yi3;

    y0 = glmm_load256(m2[0]); /* h g f e d c b a */
    y1 = glmm_load256(m2[2]); /* p o n m l k j i */

    y2 = glmm_load256(m1[0]); /* h g f e d c b a */
    y3 = glmm_load256(m1[2]); /* p o n m l k j i */

    /* 0x03: 0b00000011 */
    y4 = _mm256_permute2f128_ps(y2, y2, 0x03); /* d c b a h g f e */
    y5 = _mm256_permute2f128_ps(y3, y3, 0x03); /* l k j i p o n m */

    yi0 = _mm256_set_epi32(1, 1, 1, 1, 0, 0, 0, 0);
    yi1 = _mm256_set_epi32(3, 3, 3, 3, 2, 2, 2, 2);
    yi2 = _mm256_set_epi32(0, 0, 0, 0, 1, 1, 1, 1);
    yi3 = _mm256_set_epi32(2, 2, 2, 2, 3, 3, 3, 3);

    /* f f f f a a a a */
    /* h h h h c c c c */
    /* e e e e b b b b */
    /* g g g g d d d d */
    y6 = _mm256_permutevar_ps(y0, yi0);
    y7 = _mm256_permutevar_ps(y0, yi1);
    y8 = _mm256_permutevar_ps(y0, yi2);
    y9 = _mm256_permutevar_ps(y0, yi3);

    /* n n n n i i i i */
    /* p p p p k k k k */
    /* m m m m j j j j */
    /* o o o o l l l l */
    y10 = _mm256_permutevar_ps(y1, yi0);
    y11 = _mm256_permutevar_ps(y1, yi1);
    y12 = _mm256_permutevar_ps(y1, yi2);
    y13 = _mm256_permutevar_ps(y1, yi3);

    y0 = _mm256_mul_ps(y2, y6);
    y1 = _mm256_mul_ps(y2, y10);

    y0 = glmm256_fmadd(y3, y7, y0);
    y1 = glmm256_fmadd(y3, y11, y1);

    y0 = glmm256_fmadd(y4, y8, y0);
    y1 = glmm256_fmadd(y4, y12, y1);

    y0 = glmm256_fmadd(y5, y9, y0);
    y1 = glmm256_fmadd(y5, y13, y1);

    glmm_store256(dest[0], y0);
    glmm_store256(dest[2], y1);
}

#endif


/*** End of inlined file: mat4.h ***/


#endif

#ifdef PLAY_CGLM_NEON_FP

/*** Start of inlined file: mat4.h ***/
#ifndef cmat4_neon_h
#define cmat4_neon_h
#if defined(PLAY_CGLM_NEON_FP)

PLAY_CGLM_INLINE
void
mat4_scale_neon(mat4 m, float s)
{
    float32x4_t v0;

    v0 = vdupq_n_f32(s);

    vst1q_f32(m[0], vmulq_f32(vld1q_f32(m[0]), v0));
    vst1q_f32(m[1], vmulq_f32(vld1q_f32(m[1]), v0));
    vst1q_f32(m[2], vmulq_f32(vld1q_f32(m[2]), v0));
    vst1q_f32(m[3], vmulq_f32(vld1q_f32(m[3]), v0));
}

PLAY_CGLM_INLINE
void
mat4_transp_neon(mat4 m, mat4 dest)
{
    float32x4x4_t vmat;

    vmat = vld4q_f32(m[0]);

    vst1q_f32(dest[0], vmat.val[0]);
    vst1q_f32(dest[1], vmat.val[1]);
    vst1q_f32(dest[2], vmat.val[2]);
    vst1q_f32(dest[3], vmat.val[3]);
}

PLAY_CGLM_INLINE
void
mat4_mul_neon(mat4 m1, mat4 m2, mat4 dest)
{
    /* D = R * L (Column-Major) */

    glmm_128 l, r0, r1, r2, r3, v0, v1, v2, v3;

    l  = glmm_load(m1[0]);
    r0 = glmm_load(m2[0]);
    r1 = glmm_load(m2[1]);
    r2 = glmm_load(m2[2]);
    r3 = glmm_load(m2[3]);

    v0 = vmulq_f32(glmm_splat_x(r0), l);
    v1 = vmulq_f32(glmm_splat_x(r1), l);
    v2 = vmulq_f32(glmm_splat_x(r2), l);
    v3 = vmulq_f32(glmm_splat_x(r3), l);

    l  = glmm_load(m1[1]);
    v0 = glmm_fmadd(glmm_splat_y(r0), l, v0);
    v1 = glmm_fmadd(glmm_splat_y(r1), l, v1);
    v2 = glmm_fmadd(glmm_splat_y(r2), l, v2);
    v3 = glmm_fmadd(glmm_splat_y(r3), l, v3);

    l  = glmm_load(m1[2]);
    v0 = glmm_fmadd(glmm_splat_z(r0), l, v0);
    v1 = glmm_fmadd(glmm_splat_z(r1), l, v1);
    v2 = glmm_fmadd(glmm_splat_z(r2), l, v2);
    v3 = glmm_fmadd(glmm_splat_z(r3), l, v3);

    l  = glmm_load(m1[3]);
    v0 = glmm_fmadd(glmm_splat_w(r0), l, v0);
    v1 = glmm_fmadd(glmm_splat_w(r1), l, v1);
    v2 = glmm_fmadd(glmm_splat_w(r2), l, v2);
    v3 = glmm_fmadd(glmm_splat_w(r3), l, v3);

    glmm_store(dest[0], v0);
    glmm_store(dest[1], v1);
    glmm_store(dest[2], v2);
    glmm_store(dest[3], v3);
}

PLAY_CGLM_INLINE
void
mat4_mulv_neon(mat4 m, vec4 v, vec4 dest)
{
    float32x4_t l0, l1, l2, l3;
    float32x2_t vlo, vhi;

    l0  = vld1q_f32(m[0]);
    l1  = vld1q_f32(m[1]);
    l2  = vld1q_f32(m[2]);
    l3  = vld1q_f32(m[3]);

    vlo = vld1_f32(&v[0]);
    vhi = vld1_f32(&v[2]);

    l0  = vmulq_lane_f32(l0, vlo, 0);
    l0  = vmlaq_lane_f32(l0, l1, vlo, 1);
    l0  = vmlaq_lane_f32(l0, l2, vhi, 0);
    l0  = vmlaq_lane_f32(l0, l3, vhi, 1);

    vst1q_f32(dest, l0);
}

PLAY_CGLM_INLINE
float
mat4_det_neon(mat4 mat)
{
    float32x4_t   r0, r1, r2, r3, x0, x1, x2;
    float32x2_t   ij, op, mn, kl, nn, mm, jj, ii, gh, ef, t12, t34;
    float32x4x2_t a1;
    float32x4_t   x3 = glmm_float32x4_SIGNMASK_PNPN;

    /* 127 <- 0, [square] det(A) = det(At) */
    r0 = glmm_load(mat[0]);              /* d c b a */
    r1 = vrev64q_f32(glmm_load(mat[1])); /* g h e f */
    r2 = vrev64q_f32(glmm_load(mat[2])); /* l k i j */
    r3 = vrev64q_f32(glmm_load(mat[3])); /* o p m n */

    gh = vget_high_f32(r1);
    ef = vget_low_f32(r1);
    kl = vget_high_f32(r2);
    ij = vget_low_f32(r2);
    op = vget_high_f32(r3);
    mn = vget_low_f32(r3);
    mm = vdup_lane_f32(mn, 1);
    nn = vdup_lane_f32(mn, 0);
    ii = vdup_lane_f32(ij, 1);
    jj = vdup_lane_f32(ij, 0);

    /*
     t[1] = j * p - n * l;
     t[2] = j * o - n * k;
     t[3] = i * p - m * l;
     t[4] = i * o - m * k;
     */
    x0 = glmm_fnmadd(vcombine_f32(kl, kl), vcombine_f32(nn, mm),
                     vmulq_f32(vcombine_f32(op, op), vcombine_f32(jj, ii)));

    t12 = vget_low_f32(x0);
    t34 = vget_high_f32(x0);

    /* 1 3 1 3 2 4 2 4 */
    a1 = vuzpq_f32(x0, x0);

    /*
     t[0] = k * p - o * l;
     t[0] = k * p - o * l;
     t[5] = i * n - m * j;
     t[5] = i * n - m * j;
     */
    x1 = glmm_fnmadd(vcombine_f32(vdup_lane_f32(kl, 0), jj),
                     vcombine_f32(vdup_lane_f32(op, 1), mm),
                     vmulq_f32(vcombine_f32(vdup_lane_f32(op, 0), nn),
                               vcombine_f32(vdup_lane_f32(kl, 1), ii)));

    /*
       a * (f * t[0] - g * t[1] + h * t[2])
     - b * (e * t[0] - g * t[3] + h * t[4])
     + c * (e * t[1] - f * t[3] + h * t[5])
     - d * (e * t[2] - f * t[4] + g * t[5])
     */
    x2 = glmm_fnmadd(vcombine_f32(vdup_lane_f32(gh, 1), vdup_lane_f32(ef, 0)),
                     vcombine_f32(vget_low_f32(a1.val[0]), t34),
                     vmulq_f32(vcombine_f32(ef, vdup_lane_f32(ef, 1)),
                               vcombine_f32(vget_low_f32(x1), t12)));

    x2 = glmm_fmadd(vcombine_f32(vdup_lane_f32(gh, 0), gh),
                    vcombine_f32(vget_low_f32(a1.val[1]), vget_high_f32(x1)), x2);

    x2 = glmm_xor(x2, x3);

    return glmm_hadd(vmulq_f32(x2, r0));
}

/* old one */
#if 0
PLAY_CGLM_INLINE
void
mat4_inv_neon(mat4 mat, mat4 dest)
{
    float32x4_t   r0, r1, r2, r3,
                  v0, v1, v2, v3,
                  t0, t1, t2, t3, t4, t5,
                  x0, x1, x2, x3, x4, x5, x6, x7, x8;
    float32x4x2_t a1;
    float32x2_t   lp, ko, hg, jn, im, fe, ae, bf, cg, dh;
    float32x4_t   x9 = glmm_float32x4_SIGNMASK_NPNP;

    x8 = vrev64q_f32(x9);

    /* 127 <- 0 */
    r0 = glmm_load(mat[0]); /* d c b a */
    r1 = glmm_load(mat[1]); /* h g f e */
    r2 = glmm_load(mat[2]); /* l k j i */
    r3 = glmm_load(mat[3]); /* p o n m */

    /* l p k o, j n i m */
    a1  = vzipq_f32(r3, r2);

    jn  = vget_high_f32(a1.val[0]);
    im  = vget_low_f32(a1.val[0]);
    lp  = vget_high_f32(a1.val[1]);
    ko  = vget_low_f32(a1.val[1]);
    hg  = vget_high_f32(r1);

    x1  = vcombine_f32(vdup_lane_f32(lp, 0), lp);                   /* l p p p */
    x2  = vcombine_f32(vdup_lane_f32(ko, 0), ko);                   /* k o o o */
    x0  = vcombine_f32(vdup_lane_f32(lp, 1), vdup_lane_f32(hg, 1)); /* h h l l */
    x3  = vcombine_f32(vdup_lane_f32(ko, 1), vdup_lane_f32(hg, 0)); /* g g k k */

    /* t1[0] = k * p - o * l;
       t1[0] = k * p - o * l;
       t2[0] = g * p - o * h;
       t3[0] = g * l - k * h; */
    t0 = glmm_fnmadd(x2, x0, vmulq_f32(x3, x1));

    fe = vget_low_f32(r1);
    x4 = vcombine_f32(vdup_lane_f32(jn, 0), jn);                   /* j n n n */
    x5 = vcombine_f32(vdup_lane_f32(jn, 1), vdup_lane_f32(fe, 1)); /* f f j j */

    /* t1[1] = j * p - n * l;
       t1[1] = j * p - n * l;
       t2[1] = f * p - n * h;
       t3[1] = f * l - j * h; */
    t1 = glmm_fnmadd(x4, x0, vmulq_f32(x5, x1));

    /* t1[2] = j * o - n * k
       t1[2] = j * o - n * k;
       t2[2] = f * o - n * g;
       t3[2] = f * k - j * g; */
    t2 = glmm_fnmadd(x4, x3, vmulq_f32(x5, x2));

    x6 = vcombine_f32(vdup_lane_f32(im, 1), vdup_lane_f32(fe, 0)); /* e e i i */
    x7 = vcombine_f32(vdup_lane_f32(im, 0), im);                   /* i m m m */

    /* t1[3] = i * p - m * l;
       t1[3] = i * p - m * l;
       t2[3] = e * p - m * h;
       t3[3] = e * l - i * h; */
    t3 = glmm_fnmadd(x7, x0, vmulq_f32(x6, x1));

    /* t1[4] = i * o - m * k;
       t1[4] = i * o - m * k;
       t2[4] = e * o - m * g;
       t3[4] = e * k - i * g; */
    t4 = glmm_fnmadd(x7, x3, vmulq_f32(x6, x2));

    /* t1[5] = i * n - m * j;
       t1[5] = i * n - m * j;
       t2[5] = e * n - m * f;
       t3[5] = e * j - i * f; */
    t5 = glmm_fnmadd(x7, x5, vmulq_f32(x6, x4));

    /* h d f b, g c e a */
    a1 = vtrnq_f32(r0, r1);

    x4 = vrev64q_f32(a1.val[0]); /* c g a e */
    x5 = vrev64q_f32(a1.val[1]); /* d h b f */

    ae = vget_low_f32(x4);
    cg = vget_high_f32(x4);
    bf = vget_low_f32(x5);
    dh = vget_high_f32(x5);

    x0 = vcombine_f32(ae, vdup_lane_f32(ae, 1)); /* a a a e */
    x1 = vcombine_f32(bf, vdup_lane_f32(bf, 1)); /* b b b f */
    x2 = vcombine_f32(cg, vdup_lane_f32(cg, 1)); /* c c c g */
    x3 = vcombine_f32(dh, vdup_lane_f32(dh, 1)); /* d d d h */

    /*
     dest[0][0] =  f * t1[0] - g * t1[1] + h * t1[2];
     dest[0][1] =-(b * t1[0] - c * t1[1] + d * t1[2]);
     dest[0][2] =  b * t2[0] - c * t2[1] + d * t2[2];
     dest[0][3] =-(b * t3[0] - c * t3[1] + d * t3[2]); */
    v0 = glmm_xor(glmm_fmadd(x3, t2, glmm_fnmadd(x2, t1, vmulq_f32(x1, t0))), x8);

    /*
     dest[2][0] =  e * t1[1] - f * t1[3] + h * t1[5];
     dest[2][1] =-(a * t1[1] - b * t1[3] + d * t1[5]);
     dest[2][2] =  a * t2[1] - b * t2[3] + d * t2[5];
     dest[2][3] =-(a * t3[1] - b * t3[3] + d * t3[5]);*/
    v2 = glmm_xor(glmm_fmadd(x3, t5, glmm_fnmadd(x1, t3, vmulq_f32(x0, t1))), x8);

    /*
     dest[1][0] =-(e * t1[0] - g * t1[3] + h * t1[4]);
     dest[1][1] =  a * t1[0] - c * t1[3] + d * t1[4];
     dest[1][2] =-(a * t2[0] - c * t2[3] + d * t2[4]);
     dest[1][3] =  a * t3[0] - c * t3[3] + d * t3[4]; */
    v1 = glmm_xor(glmm_fmadd(x3, t4, glmm_fnmadd(x2, t3, vmulq_f32(x0, t0))), x9);

    /*
     dest[3][0] =-(e * t1[2] - f * t1[4] + g * t1[5]);
     dest[3][1] =  a * t1[2] - b * t1[4] + c * t1[5];
     dest[3][2] =-(a * t2[2] - b * t2[4] + c * t2[5]);
     dest[3][3] =  a * t3[2] - b * t3[4] + c * t3[5]; */
    v3 = glmm_xor(glmm_fmadd(x2, t5, glmm_fnmadd(x1, t4, vmulq_f32(x0, t2))), x9);

    /* determinant */
    x0 = vcombine_f32(vget_low_f32(vzipq_f32(v0, v1).val[0]),
                      vget_low_f32(vzipq_f32(v2, v3).val[0]));

    /*
    x0 = glmm_div(glmm_set1_rval(1.0f), glmm_vhadd(vmulq_f32(x0, r0)));

    glmm_store(dest[0], vmulq_f32(v0, x0));
    glmm_store(dest[1], vmulq_f32(v1, x0));
    glmm_store(dest[2], vmulq_f32(v2, x0));
    glmm_store(dest[3], vmulq_f32(v3, x0));
    */

    x0 = glmm_vhadd(vmulq_f32(x0, r0));

    glmm_store(dest[0], glmm_div(v0, x0));
    glmm_store(dest[1], glmm_div(v1, x0));
    glmm_store(dest[2], glmm_div(v2, x0));
    glmm_store(dest[3], glmm_div(v3, x0));
}
#endif

PLAY_CGLM_INLINE
void
mat4_inv_neon(mat4 mat, mat4 dest)
{
    float32x4_t   r0, r1, r2, r3,
                  v0, v1, v2, v3, v4, v5,
                  t0, t1, t2;
    float32x4x2_t a0, a1, a2, a3, a4;
    float32x4_t   s1 = glmm_float32x4_SIGNMASK_PNPN, s2;

#if !PLAY_CGLM_ARM64
    float32x2_t   l0, l1;
#endif

    s2 = vrev64q_f32(s1);

    /* 127 <- 0 */
    r0 = glmm_load(mat[0]);                  /* d c b a */
    r1 = glmm_load(mat[1]);                  /* h g f e */
    r2 = glmm_load(mat[2]);                  /* l k j i */
    r3 = glmm_load(mat[3]);                  /* p o n m */

    a1 = vzipq_f32(r0, r2);                  /* l d k c, j b i a */
    a2 = vzipq_f32(r1, r3);                  /* p h o g, n f m e */
    a3 = vzipq_f32(a2.val[0], a1.val[0]);    /* j n b f, i m a e */
    a4 = vzipq_f32(a2.val[1], a1.val[1]);    /* l p d h, k o c g */

    v0 = vextq_f32(a1.val[0], a1.val[1], 2); /* k c j b */
    v1 = vextq_f32(a2.val[0], a2.val[1], 2); /* o g n f */
    v2 = vextq_f32(a1.val[1], a2.val[0], 2); /* m e l d */
    v3 = vextq_f32(a2.val[1], a1.val[0], 2); /* i a p h */
    v4 = vextq_f32(v1, v2, 2);               /* l d o g */
    v5 = vextq_f32(v0, v3, 2);               /* p h k c */

    /* c2 = c * h - g * d   c12 = a * g - c * e   c8  = a * f - b * e
       c1 = k * p - o * l   c11 = i * o - k * m   c7  = i * n - j * m
       c4 = h * a - d * e   c6  = b * h - d * f   c10 = b * g - c * f
       c3 = p * i - l * m   c5  = j * p - l * n   c9  = j * o - k * n */
    t0 = vmulq_f32(v5, v3);
    t1 = vmulq_f32(a1.val[0], a2.val[1]);
    t2 = vmulq_f32(a1.val[0], v1);

    t0 = glmm_fnmadd(v4, v2, t0);
    t1 = glmm_fnmadd(a1.val[1], a2.val[0], t1);
    t2 = glmm_fnmadd(v0, a2.val[0], t2);

    t0 = vrev64q_f32(t0);
    t1 = vrev64q_f32(t1);
    t2 = vrev64q_f32(t2);

    /* det */
    v0 = vrev64q_f32(t2);
    v1 = vextq_f32(t1, t1, 2);
    v0 = vmulq_f32(t0, v0);
    v1 = vrev64q_f32(v1);
    v1 = vmulq_f32(v1, t1);

    /* c3 * c10 + c4 * c9 + c1 * c8 + c2 * c7 */
#if PLAY_CGLM_ARM64
    v0 = vpaddq_f32(v0, v0);
    v0 = vpaddq_f32(v0, v0);
#else
    l0 = vget_low_f32(v0);
    l1 = vget_high_f32(v0);

    l0 = vpadd_f32(l0, l0); /* [a+b, a+b] */
    l1 = vpadd_f32(l1, l1); /* [c+d, c+d] */
    l0 = vadd_f32(l0, l1);  /* [sum, sum] */

    v0 = vcombine_f32(l0, l0);
#endif

    /* c5 * c12 + c6 * c11 */
#if PLAY_CGLM_ARM64
    v1 = vpaddq_f32(v1, v1);
#else
    l0 = vget_low_f32(v1);
    l1 = vget_high_f32(v1);

    l0 = vpadd_f32(l0, l0); /* [a+b, a+b] */
    l1 = vpadd_f32(l1, l1); /* [c+d, c+d] */

    v1 = vcombine_f32(l0, l1);
#endif

    v0 = vsubq_f32(v0, v1);    /* det */

    /* inv div */
    v1 = vdupq_n_f32(1.0f);
    v0 = glmm_div(v1, v0);     /* inv div */

    /* multiply t0,t1,t2 by idt to reduce 1mul below: 2eor+4mul vs 3mul+4eor */
    t0 = vmulq_f32(t0, v0);
    t1 = vmulq_f32(t1, v0);
    t2 = vmulq_f32(t2, v0);

    a0 = vzipq_f32(t0, t0);    /* c4  c4  c3 c3, c2  c2  c1  c1  */
    a1 = vzipq_f32(t1, t1);    /* c6  c6  c5 c5, c12 c12 c11 c11 */
    a2 = vzipq_f32(t2, t2);    /* c10 c10 c9 c9, c8  c8  c7  c7  */

    /* result */

    /* dest[0][0] = (f * c1  - g * c5  + h * c9)  * idt;
       dest[0][1] = (b * c1  - c * c5  + d * c9)  * ndt;
       dest[0][2] = (n * c2  - o * c6  + p * c10) * idt;
       dest[0][3] = (j * c2  - k * c6  + l * c10) * ndt;

       dest[1][0] = (e * c1  - g * c3  + h * c11) * ndt;
       dest[1][1] = (a * c1  - c * c3  + d * c11) * idt;
       dest[1][2] = (m * c2  - o * c4  + p * c12) * ndt;
       dest[1][3] = (i * c2  - k * c4  + l * c12) * idt;

       dest[2][0] = (e * c5  - f * c3  + h * c7)  * idt;
       dest[2][1] = (a * c5  - b * c3  + d * c7)  * ndt;
       dest[2][2] = (m * c6  - n * c4  + p * c8)  * idt;
       dest[2][3] = (i * c6  - j * c4  + l * c8)  * ndt;

       dest[3][0] = (e * c9  - f * c11 + g * c7)  * ndt;
       dest[3][1] = (a * c9  - b * c11 + c * c7)  * idt;
       dest[3][2] = (m * c10 - n * c12 + o * c8)  * ndt;
       dest[3][3] = (i * c10 - j * c12 + k * c8)  * idt; */

    r0 = vmulq_f32(a3.val[1], a0.val[0]);
    r1 = vmulq_f32(a3.val[0], a0.val[0]);
    r2 = vmulq_f32(a3.val[0], a1.val[1]);
    r3 = vmulq_f32(a3.val[0], a2.val[1]);

    r0 = glmm_fnmadd(a4.val[0], a1.val[1], r0);
    r1 = glmm_fnmadd(a4.val[0], a0.val[1], r1);
    r2 = glmm_fnmadd(a3.val[1], a0.val[1], r2);
    r3 = glmm_fnmadd(a3.val[1], a1.val[0], r3);

    r0 = glmm_fmadd(a4.val[1], a2.val[1], r0);
    r1 = glmm_fmadd(a4.val[1], a1.val[0], r1);
    r2 = glmm_fmadd(a4.val[1], a2.val[0], r2);
    r3 = glmm_fmadd(a4.val[0], a2.val[0], r3);

    /* 4xor may be fastart then 4mul, see above  */
    r0 = glmm_xor(r0, s1);
    r1 = glmm_xor(r1, s2);
    r2 = glmm_xor(r2, s1);
    r3 = glmm_xor(r3, s2);

    glmm_store(dest[0], r0);
    glmm_store(dest[1], r1);
    glmm_store(dest[2], r2);
    glmm_store(dest[3], r3);
}

#endif
#endif /* cmat4_neon_h */

/*** End of inlined file: mat4.h ***/


#endif

#ifdef PLAY_CGLM_SIMD_WASM

/*** Start of inlined file: mat4.h ***/


#if defined(__wasm__) && defined(__wasm_simd128__)

#define mat4_inv_precise_wasm(mat, dest) mat4_inv_wasm(mat, dest)

PLAY_CGLM_INLINE
void
mat4_scale_wasm(mat4 m, float s)
{
    glmm_128 x0;
    x0 = wasm_f32x4_splat(s);

    glmm_store(m[0], wasm_f32x4_mul(glmm_load(m[0]), x0));
    glmm_store(m[1], wasm_f32x4_mul(glmm_load(m[1]), x0));
    glmm_store(m[2], wasm_f32x4_mul(glmm_load(m[2]), x0));
    glmm_store(m[3], wasm_f32x4_mul(glmm_load(m[3]), x0));
}

PLAY_CGLM_INLINE
void
mat4_transp_wasm(mat4 m, mat4 dest)
{
    glmm_128 r0, r1, r2, r3, tmp0, tmp1, tmp2, tmp3;

    r0 = glmm_load(m[0]);
    r1 = glmm_load(m[1]);
    r2 = glmm_load(m[2]);
    r3 = glmm_load(m[3]);

    /* _MM_TRANSPOSE4_PS(r0, r1, r2, r3); */
    tmp0 = wasm_i32x4_shuffle(r0, r1, 0, 4, 1, 5);
    tmp1 = wasm_i32x4_shuffle(r0, r1, 2, 6, 3, 7);
    tmp2 = wasm_i32x4_shuffle(r2, r3, 0, 4, 1, 5);
    tmp3 = wasm_i32x4_shuffle(r2, r3, 2, 6, 3, 7);
    /* r0 = _mm_movelh_ps(tmp0, tmp2); */
    r0 = wasm_i32x4_shuffle(tmp0, tmp2, 0, 1, 4, 5);
    /* r1 = _mm_movehl_ps(tmp2, tmp0); */
    r1 = wasm_i32x4_shuffle(tmp2, tmp0, 6, 7, 2, 3);
    /* r2 = _mm_movelh_ps(tmp1, tmp3); */
    r2 = wasm_i32x4_shuffle(tmp1, tmp3, 0, 1, 4, 5);
    /* r3 = _mm_movehl_ps(tmp3, tmp1); */
    r3 = wasm_i32x4_shuffle(tmp3, tmp1, 6, 7, 2, 3);

    glmm_store(dest[0], r0);
    glmm_store(dest[1], r1);
    glmm_store(dest[2], r2);
    glmm_store(dest[3], r3);
}

PLAY_CGLM_INLINE
void
mat4_mul_wasm(mat4 m1, mat4 m2, mat4 dest)
{
    /* D = R * L (Column-Major) */

    glmm_128 l, r0, r1, r2, r3, v0, v1, v2, v3;

    l  = glmm_load(m1[0]);
    r0 = glmm_load(m2[0]);
    r1 = glmm_load(m2[1]);
    r2 = glmm_load(m2[2]);
    r3 = glmm_load(m2[3]);

    v0 = wasm_f32x4_mul(glmm_splat_x(r0), l);
    v1 = wasm_f32x4_mul(glmm_splat_x(r1), l);
    v2 = wasm_f32x4_mul(glmm_splat_x(r2), l);
    v3 = wasm_f32x4_mul(glmm_splat_x(r3), l);

    l  = glmm_load(m1[1]);
    v0 = glmm_fmadd(glmm_splat_y(r0), l, v0);
    v1 = glmm_fmadd(glmm_splat_y(r1), l, v1);
    v2 = glmm_fmadd(glmm_splat_y(r2), l, v2);
    v3 = glmm_fmadd(glmm_splat_y(r3), l, v3);

    l  = glmm_load(m1[2]);
    v0 = glmm_fmadd(glmm_splat_z(r0), l, v0);
    v1 = glmm_fmadd(glmm_splat_z(r1), l, v1);
    v2 = glmm_fmadd(glmm_splat_z(r2), l, v2);
    v3 = glmm_fmadd(glmm_splat_z(r3), l, v3);

    l  = glmm_load(m1[3]);
    v0 = glmm_fmadd(glmm_splat_w(r0), l, v0);
    v1 = glmm_fmadd(glmm_splat_w(r1), l, v1);
    v2 = glmm_fmadd(glmm_splat_w(r2), l, v2);
    v3 = glmm_fmadd(glmm_splat_w(r3), l, v3);

    glmm_store(dest[0], v0);
    glmm_store(dest[1], v1);
    glmm_store(dest[2], v2);
    glmm_store(dest[3], v3);
}

PLAY_CGLM_INLINE
void
mat4_mulv_wasm(mat4 m, vec4 v, vec4 dest)
{
    glmm_128 x0, x1, m0, m1, m2, m3, v0, v1, v2, v3;

    m0 = glmm_load(m[0]);
    m1 = glmm_load(m[1]);
    m2 = glmm_load(m[2]);
    m3 = glmm_load(m[3]);

    x0 = glmm_load(v);
    v0 = glmm_splat_x(x0);
    v1 = glmm_splat_y(x0);
    v2 = glmm_splat_z(x0);
    v3 = glmm_splat_w(x0);

    x1 = wasm_f32x4_mul(m3, v3);
    x1 = glmm_fmadd(m2, v2, x1);
    x1 = glmm_fmadd(m1, v1, x1);
    x1 = glmm_fmadd(m0, v0, x1);

    glmm_store(dest, x1);
}

PLAY_CGLM_INLINE
float
mat4_det_wasm(mat4 mat)
{
    glmm_128 r0, r1, r2, r3, x0, x1, x2;

    /* 127 <- 0, [square] det(A) = det(At) */
    r0 = glmm_load(mat[0]); /* d c b a */
    r1 = glmm_load(mat[1]); /* h g f e */
    r2 = glmm_load(mat[2]); /* l k j i */
    r3 = glmm_load(mat[3]); /* p o n m */

    /*
     t[1] = j * p - n * l;
     t[2] = j * o - n * k;
     t[3] = i * p - m * l;
     t[4] = i * o - m * k;
     */
    x0 = glmm_fnmadd(glmm_shuff1(r3, 0, 0, 1, 1), glmm_shuff1(r2, 2, 3, 2, 3),
                     wasm_f32x4_mul(glmm_shuff1(r2, 0, 0, 1, 1),
                                    glmm_shuff1(r3, 2, 3, 2, 3)));
    /*
     t[0] = k * p - o * l;
     t[0] = k * p - o * l;
     t[5] = i * n - m * j;
     t[5] = i * n - m * j;
     */
    x1 = glmm_fnmadd(glmm_shuff1(r3, 0, 0, 2, 2), glmm_shuff1(r2, 1, 1, 3, 3),
                     wasm_f32x4_mul(glmm_shuff1(r2, 0, 0, 2, 2),
                                    glmm_shuff1(r3, 1, 1, 3, 3)));

    /*
       a * (f * t[0] - g * t[1] + h * t[2])
     - b * (e * t[0] - g * t[3] + h * t[4])
     + c * (e * t[1] - f * t[3] + h * t[5])
     - d * (e * t[2] - f * t[4] + g * t[5])
     */
    x2 = glmm_fnmadd(glmm_shuff1(r1, 1, 1, 2, 2), glmm_shuff1(x0, 3, 2, 2, 0),
                     wasm_f32x4_mul(glmm_shuff1(r1, 0, 0, 0, 1),
                                    wasm_i32x4_shuffle(x1, x0, 0, 0, 4, 5)));
    x2 = glmm_fmadd(glmm_shuff1(r1, 2, 3, 3, 3),
                    wasm_i32x4_shuffle(x0, x1, 1, 3, 6, 6),
                    x2);
    /* x2 = wasm_v128_xor(x2, wasm_f32x4_const(0.f, -0.f, 0.f, -0.f)); */
    x2 = wasm_v128_xor(x2, glmm_float32x4_SIGNMASK_PNPN);

    return glmm_hadd(wasm_f32x4_mul(x2, r0));
}

PLAY_CGLM_INLINE
void
mat4_inv_fast_wasm(mat4 mat, mat4 dest)
{
    glmm_128 r0, r1, r2, r3,
             v0, v1, v2, v3,
             t0, t1, t2, t3, t4, t5,
             x0, x1, x2, x3, x4, x5, x6, x7, x8, x9;

    /* x8 = wasm_f32x4_const(0.f, -0.f, 0.f, -0.f); */
    x8 = glmm_float32x4_SIGNMASK_PNPN;
    x9 = glmm_shuff1(x8, 2, 1, 2, 1);

    /* 127 <- 0 */
    r0 = glmm_load(mat[0]); /* d c b a */
    r1 = glmm_load(mat[1]); /* h g f e */
    r2 = glmm_load(mat[2]); /* l k j i */
    r3 = glmm_load(mat[3]); /* p o n m */
    /* x0 = _mm_movehl_ps(r3, r2); */
    x0 = wasm_i32x4_shuffle(r3, r2, 6, 7, 2, 3);           /* p o l k */
    /* x3 = _mm_movelh_ps(r2, r3); */
    x3 = wasm_i32x4_shuffle(r2, r3, 0, 1, 4, 5);           /* n m j i */
    x1 = glmm_shuff1(x0, 1, 3, 3,3);                       /* l p p p */
    x2 = glmm_shuff1(x0, 0, 2, 2, 2);                      /* k o o o */
    x4 = glmm_shuff1(x3, 1, 3, 3, 3);                      /* j n n n */
    x7 = glmm_shuff1(x3, 0, 2, 2, 2);                      /* i m m m */

    x6 = wasm_i32x4_shuffle(r2, r1, 0, 0, 4, 4);           /* e e i i */
    x5 = wasm_i32x4_shuffle(r2, r1, 1, 1, 5, 5);           /* f f j j */
    x3 = wasm_i32x4_shuffle(r2, r1, 2, 2, 6, 6);           /* g g k k */
    x0 = wasm_i32x4_shuffle(r2, r1, 3, 3, 7, 7);           /* h h l l */

    t0 = wasm_f32x4_mul(x3, x1);
    t1 = wasm_f32x4_mul(x5, x1);
    t2 = wasm_f32x4_mul(x5, x2);
    t3 = wasm_f32x4_mul(x6, x1);
    t4 = wasm_f32x4_mul(x6, x2);
    t5 = wasm_f32x4_mul(x6, x4);

    /* t1[0] = k * p - o * l;
       t1[0] = k * p - o * l;
       t2[0] = g * p - o * h;
       t3[0] = g * l - k * h; */
    t0 = glmm_fnmadd(x2, x0, t0);

    /* t1[1] = j * p - n * l;
       t1[1] = j * p - n * l;
       t2[1] = f * p - n * h;
       t3[1] = f * l - j * h; */
    t1 = glmm_fnmadd(x4, x0, t1);

    /* t1[2] = j * o - n * k
       t1[2] = j * o - n * k;
       t2[2] = f * o - n * g;
       t3[2] = f * k - j * g; */
    t2 = glmm_fnmadd(x4, x3, t2);

    /* t1[3] = i * p - m * l;
       t1[3] = i * p - m * l;
       t2[3] = e * p - m * h;
       t3[3] = e * l - i * h; */
    t3 = glmm_fnmadd(x7, x0, t3);

    /* t1[4] = i * o - m * k;
       t1[4] = i * o - m * k;
       t2[4] = e * o - m * g;
       t3[4] = e * k - i * g; */
    t4 = glmm_fnmadd(x7, x3, t4);

    /* t1[5] = i * n - m * j;
       t1[5] = i * n - m * j;
       t2[5] = e * n - m * f;
       t3[5] = e * j - i * f; */
    t5 = glmm_fnmadd(x7, x5, t5);
    /* x4 = _mm_movelh_ps(r0, r1); */
    x4 = wasm_i32x4_shuffle(r0, r1, 0, 1, 4, 5);           /* f e b a */
    /* x5 = _mm_movehl_ps(r1, r0); */
    x5 = wasm_i32x4_shuffle(r1, r0, 6, 7, 2, 3);           /* h g d c */

    x0 = glmm_shuff1(x4, 0, 0, 0, 2);                      /* a a a e */
    x1 = glmm_shuff1(x4, 1, 1, 1, 3);                      /* b b b f */
    x2 = glmm_shuff1(x5, 0, 0, 0, 2);                      /* c c c g */
    x3 = glmm_shuff1(x5, 1, 1, 1, 3);                      /* d d d h */

    v2 = wasm_f32x4_mul(x0, t1);
    v1 = wasm_f32x4_mul(x0, t0);
    v3 = wasm_f32x4_mul(x0, t2);
    v0 = wasm_f32x4_mul(x1, t0);

    v2 = glmm_fnmadd(x1, t3, v2);
    v3 = glmm_fnmadd(x1, t4, v3);
    v0 = glmm_fnmadd(x2, t1, v0);
    v1 = glmm_fnmadd(x2, t3, v1);

    v3 = glmm_fmadd(x2, t5, v3);
    v0 = glmm_fmadd(x3, t2, v0);
    v2 = glmm_fmadd(x3, t5, v2);
    v1 = glmm_fmadd(x3, t4, v1);

    /*
     dest[0][0] =  f * t1[0] - g * t1[1] + h * t1[2];
     dest[0][1] =-(b * t1[0] - c * t1[1] + d * t1[2]);
     dest[0][2] =  b * t2[0] - c * t2[1] + d * t2[2];
     dest[0][3] =-(b * t3[0] - c * t3[1] + d * t3[2]); */
    v0 = wasm_v128_xor(v0, x8);

    /*
     dest[2][0] =  e * t1[1] - f * t1[3] + h * t1[5];
     dest[2][1] =-(a * t1[1] - b * t1[3] + d * t1[5]);
     dest[2][2] =  a * t2[1] - b * t2[3] + d * t2[5];
     dest[2][3] =-(a * t3[1] - b * t3[3] + d * t3[5]);*/
    v2 = wasm_v128_xor(v2, x8);

    /*
     dest[1][0] =-(e * t1[0] - g * t1[3] + h * t1[4]);
     dest[1][1] =  a * t1[0] - c * t1[3] + d * t1[4];
     dest[1][2] =-(a * t2[0] - c * t2[3] + d * t2[4]);
     dest[1][3] =  a * t3[0] - c * t3[3] + d * t3[4]; */
    v1 = wasm_v128_xor(v1, x9);

    /*
     dest[3][0] =-(e * t1[2] - f * t1[4] + g * t1[5]);
     dest[3][1] =  a * t1[2] - b * t1[4] + c * t1[5];
     dest[3][2] =-(a * t2[2] - b * t2[4] + c * t2[5]);
     dest[3][3] =  a * t3[2] - b * t3[4] + c * t3[5]; */
    v3 = wasm_v128_xor(v3, x9);

    /* determinant */
    x0 = wasm_i32x4_shuffle(v0, v1, 0, 0, 4, 4);
    x1 = wasm_i32x4_shuffle(v2, v3, 0, 0, 4, 4);
    x0 = wasm_i32x4_shuffle(x0, x1, 0, 2, 4, 6);

    /* x0 = _mm_rcp_ps(glmm_vhadd(wasm_f32x4_mul(x0, r0))); */
    x0 = wasm_f32x4_div(wasm_f32x4_const_splat(1.0f),
                        glmm_vhadd(wasm_f32x4_mul(x0, r0)));

    glmm_store(dest[0], wasm_f32x4_mul(v0, x0));
    glmm_store(dest[1], wasm_f32x4_mul(v1, x0));
    glmm_store(dest[2], wasm_f32x4_mul(v2, x0));
    glmm_store(dest[3], wasm_f32x4_mul(v3, x0));
}

PLAY_CGLM_INLINE
void
mat4_inv_wasm(mat4 mat, mat4 dest)
{
    glmm_128 r0, r1, r2, r3,
             v0, v1, v2, v3,
             t0, t1, t2, t3, t4, t5,
             x0, x1, x2, x3, x4, x5, x6, x7, x8, x9;

    /* x8 = wasm_f32x4_const(0.f, -0.f, 0.f, -0.f); */
    x8 = glmm_float32x4_SIGNMASK_PNPN;
    x9 = glmm_shuff1(x8, 2, 1, 2, 1);

    /* 127 <- 0 */
    r0 = glmm_load(mat[0]); /* d c b a */
    r1 = glmm_load(mat[1]); /* h g f e */
    r2 = glmm_load(mat[2]); /* l k j i */
    r3 = glmm_load(mat[3]); /* p o n m */
    /* x0 = _mm_movehl_ps(r3, r2); */
    x0 = wasm_i32x4_shuffle(r3, r2, 6, 7, 2, 3);           /* p o l k */
    /* x3 = _mm_movelh_ps(r2, r3); */
    x3 = wasm_i32x4_shuffle(r2, r3, 0, 1, 4, 5);           /* n m j i */
    x1 = glmm_shuff1(x0, 1, 3, 3,3);                       /* l p p p */
    x2 = glmm_shuff1(x0, 0, 2, 2, 2);                      /* k o o o */
    x4 = glmm_shuff1(x3, 1, 3, 3, 3);                      /* j n n n */
    x7 = glmm_shuff1(x3, 0, 2, 2, 2);                      /* i m m m */

    x6 = wasm_i32x4_shuffle(r2, r1, 0, 0, 4, 4);           /* e e i i */
    x5 = wasm_i32x4_shuffle(r2, r1, 1, 1, 5, 5);           /* f f j j */
    x3 = wasm_i32x4_shuffle(r2, r1, 2, 2, 6, 6);           /* g g k k */
    x0 = wasm_i32x4_shuffle(r2, r1, 3, 3, 7, 7);           /* h h l l */

    t0 = wasm_f32x4_mul(x3, x1);
    t1 = wasm_f32x4_mul(x5, x1);
    t2 = wasm_f32x4_mul(x5, x2);
    t3 = wasm_f32x4_mul(x6, x1);
    t4 = wasm_f32x4_mul(x6, x2);
    t5 = wasm_f32x4_mul(x6, x4);

    /* t1[0] = k * p - o * l;
       t1[0] = k * p - o * l;
       t2[0] = g * p - o * h;
       t3[0] = g * l - k * h; */
    t0 = glmm_fnmadd(x2, x0, t0);

    /* t1[1] = j * p - n * l;
       t1[1] = j * p - n * l;
       t2[1] = f * p - n * h;
       t3[1] = f * l - j * h; */
    t1 = glmm_fnmadd(x4, x0, t1);

    /* t1[2] = j * o - n * k
       t1[2] = j * o - n * k;
       t2[2] = f * o - n * g;
       t3[2] = f * k - j * g; */
    t2 = glmm_fnmadd(x4, x3, t2);

    /* t1[3] = i * p - m * l;
       t1[3] = i * p - m * l;
       t2[3] = e * p - m * h;
       t3[3] = e * l - i * h; */
    t3 = glmm_fnmadd(x7, x0, t3);

    /* t1[4] = i * o - m * k;
       t1[4] = i * o - m * k;
       t2[4] = e * o - m * g;
       t3[4] = e * k - i * g; */
    t4 = glmm_fnmadd(x7, x3, t4);

    /* t1[5] = i * n - m * j;
       t1[5] = i * n - m * j;
       t2[5] = e * n - m * f;
       t3[5] = e * j - i * f; */
    t5 = glmm_fnmadd(x7, x5, t5);
    /* x4 = _mm_movelh_ps(r0, r1); */
    x4 = wasm_i32x4_shuffle(r0, r1, 0, 1, 4, 5);           /* f e b a */
    /* x5 = _mm_movehl_ps(r1, r0); */
    x5 = wasm_i32x4_shuffle(r1, r0, 6, 7, 2, 3);           /* h g d c */

    x0 = glmm_shuff1(x4, 0, 0, 0, 2);                      /* a a a e */
    x1 = glmm_shuff1(x4, 1, 1, 1, 3);                      /* b b b f */
    x2 = glmm_shuff1(x5, 0, 0, 0, 2);                      /* c c c g */
    x3 = glmm_shuff1(x5, 1, 1, 1, 3);                      /* d d d h */

    v2 = wasm_f32x4_mul(x0, t1);
    v1 = wasm_f32x4_mul(x0, t0);
    v3 = wasm_f32x4_mul(x0, t2);
    v0 = wasm_f32x4_mul(x1, t0);

    v2 = glmm_fnmadd(x1, t3, v2);
    v3 = glmm_fnmadd(x1, t4, v3);
    v0 = glmm_fnmadd(x2, t1, v0);
    v1 = glmm_fnmadd(x2, t3, v1);

    v3 = glmm_fmadd(x2, t5, v3);
    v0 = glmm_fmadd(x3, t2, v0);
    v2 = glmm_fmadd(x3, t5, v2);
    v1 = glmm_fmadd(x3, t4, v1);

    /*
     dest[0][0] =  f * t1[0] - g * t1[1] + h * t1[2];
     dest[0][1] =-(b * t1[0] - c * t1[1] + d * t1[2]);
     dest[0][2] =  b * t2[0] - c * t2[1] + d * t2[2];
     dest[0][3] =-(b * t3[0] - c * t3[1] + d * t3[2]); */
    v0 = wasm_v128_xor(v0, x8);

    /*
     dest[2][0] =  e * t1[1] - f * t1[3] + h * t1[5];
     dest[2][1] =-(a * t1[1] - b * t1[3] + d * t1[5]);
     dest[2][2] =  a * t2[1] - b * t2[3] + d * t2[5];
     dest[2][3] =-(a * t3[1] - b * t3[3] + d * t3[5]);*/
    v2 = wasm_v128_xor(v2, x8);

    /*
     dest[1][0] =-(e * t1[0] - g * t1[3] + h * t1[4]);
     dest[1][1] =  a * t1[0] - c * t1[3] + d * t1[4];
     dest[1][2] =-(a * t2[0] - c * t2[3] + d * t2[4]);
     dest[1][3] =  a * t3[0] - c * t3[3] + d * t3[4]; */
    v1 = wasm_v128_xor(v1, x9);

    /*
     dest[3][0] =-(e * t1[2] - f * t1[4] + g * t1[5]);
     dest[3][1] =  a * t1[2] - b * t1[4] + c * t1[5];
     dest[3][2] =-(a * t2[2] - b * t2[4] + c * t2[5]);
     dest[3][3] =  a * t3[2] - b * t3[4] + c * t3[5]; */
    v3 = wasm_v128_xor(v3, x9);

    /* determinant */
    x0 = wasm_i32x4_shuffle(v0, v1, 0, 0, 4, 4);
    x1 = wasm_i32x4_shuffle(v2, v3, 0, 0, 4, 4);
    x0 = wasm_i32x4_shuffle(x0, x1, 0, 2, 4, 6);

    x0 = wasm_f32x4_div(wasm_f32x4_splat(1.0f), glmm_vhadd(wasm_f32x4_mul(x0, r0)));

    glmm_store(dest[0], wasm_f32x4_mul(v0, x0));
    glmm_store(dest[1], wasm_f32x4_mul(v1, x0));
    glmm_store(dest[2], wasm_f32x4_mul(v2, x0));
    glmm_store(dest[3], wasm_f32x4_mul(v3, x0));
}

#endif


/*** End of inlined file: mat4.h ***/


#endif


#define PLAY_CGLM_MAT4_IDENTITY_INIT  {{1.0f, 0.0f, 0.0f, 0.0f},                    \
                                 {0.0f, 1.0f, 0.0f, 0.0f},                    \
                                 {0.0f, 0.0f, 1.0f, 0.0f},                    \
                                 {0.0f, 0.0f, 0.0f, 1.0f}}

#define PLAY_CGLM_MAT4_ZERO_INIT      {{0.0f, 0.0f, 0.0f, 0.0f},                    \
                                 {0.0f, 0.0f, 0.0f, 0.0f},                    \
                                 {0.0f, 0.0f, 0.0f, 0.0f},                    \
                                 {0.0f, 0.0f, 0.0f, 0.0f}}

/* for C only */
#define PLAY_CGLM_MAT4_IDENTITY ((mat4)PLAY_CGLM_MAT4_IDENTITY_INIT)
#define PLAY_CGLM_MAT4_ZERO     ((mat4)PLAY_CGLM_MAT4_ZERO_INIT)

/* DEPRECATED! use _copy, _ucopy versions */
#define mat4_udup(mat, dest) mat4_ucopy(mat, dest)
#define mat4_dup(mat, dest)  mat4_copy(mat, dest)

/* DEPRECATED! default is precise now. */
#define mat4_inv_precise(mat, dest) mat4_inv(mat, dest)

/*!
 * @brief copy all members of [mat] to [dest]
 *
 * matrix may not be aligned, u stands for unaligned, this may be useful when
 * copying a matrix from external source e.g. asset importer...
 *
 * @param[in]  mat  source
 * @param[out] dest destination
 */
PLAY_CGLM_INLINE
void
mat4_ucopy(mat4 mat, mat4 dest)
{
    dest[0][0] = mat[0][0];
    dest[1][0] = mat[1][0];
    dest[0][1] = mat[0][1];
    dest[1][1] = mat[1][1];
    dest[0][2] = mat[0][2];
    dest[1][2] = mat[1][2];
    dest[0][3] = mat[0][3];
    dest[1][3] = mat[1][3];

    dest[2][0] = mat[2][0];
    dest[3][0] = mat[3][0];
    dest[2][1] = mat[2][1];
    dest[3][1] = mat[3][1];
    dest[2][2] = mat[2][2];
    dest[3][2] = mat[3][2];
    dest[2][3] = mat[2][3];
    dest[3][3] = mat[3][3];
}

/*!
 * @brief copy all members of [mat] to [dest]
 *
 * @param[in]  mat  source
 * @param[out] dest destination
 */
PLAY_CGLM_INLINE
void
mat4_copy(mat4 mat, mat4 dest)
{
#if defined(__wasm__) && defined(__wasm_simd128__)
    glmm_store(dest[0], glmm_load(mat[0]));
    glmm_store(dest[1], glmm_load(mat[1]));
    glmm_store(dest[2], glmm_load(mat[2]));
    glmm_store(dest[3], glmm_load(mat[3]));
#elif defined(__AVX__)
    glmm_store256(dest[0], glmm_load256(mat[0]));
    glmm_store256(dest[2], glmm_load256(mat[2]));
#elif defined( __SSE__ ) || defined( __SSE2__ )
    glmm_store(dest[0], glmm_load(mat[0]));
    glmm_store(dest[1], glmm_load(mat[1]));
    glmm_store(dest[2], glmm_load(mat[2]));
    glmm_store(dest[3], glmm_load(mat[3]));
#elif defined(PLAY_CGLM_NEON_FP)
    vst1q_f32(dest[0], vld1q_f32(mat[0]));
    vst1q_f32(dest[1], vld1q_f32(mat[1]));
    vst1q_f32(dest[2], vld1q_f32(mat[2]));
    vst1q_f32(dest[3], vld1q_f32(mat[3]));
#else
    mat4_ucopy(mat, dest);
#endif
}

/*!
 * @brief make given matrix identity. It is identical with below,
 *        but it is more easy to do that with this func especially for members
 *        e.g. mat4_identity(aStruct->aMatrix);
 *
 * @code
 * mat4_copy(PLAY_CGLM_MAT4_IDENTITY, mat); // C only
 *
 * // or
 * mat4 mat = PLAY_CGLM_MAT4_IDENTITY_INIT;
 * @endcode
 *
 * @param[in, out]  mat  destination
 */
PLAY_CGLM_INLINE
void
mat4_identity(mat4 mat)
{
    PLAY_CGLM_ALIGN_MAT mat4 t = PLAY_CGLM_MAT4_IDENTITY_INIT;
    mat4_copy(t, mat);
}

/*!
 * @brief make given matrix array's each element identity matrix
 *
 * @param[in, out]  mat   matrix array (must be aligned (16/32)
 *                        if alignment is not disabled)
 *
 * @param[in]       count count of matrices
 */
PLAY_CGLM_INLINE
void
mat4_identity_array(mat4 * __restrict mat, size_t count)
{
    PLAY_CGLM_ALIGN_MAT mat4 t = PLAY_CGLM_MAT4_IDENTITY_INIT;
    size_t i;

    for (i = 0; i < count; i++)
    {
        mat4_copy(t, mat[i]);
    }
}

/*!
 * @brief make given matrix zero.
 *
 * @param[in, out]  mat  matrix
 */
PLAY_CGLM_INLINE
void
mat4_zero(mat4 mat)
{
#if defined(__wasm__) && defined(__wasm_simd128__)
    glmm_128 x0;
    x0 = wasm_f32x4_const_splat(0.f);
    glmm_store(mat[0], x0);
    glmm_store(mat[1], x0);
    glmm_store(mat[2], x0);
    glmm_store(mat[3], x0);
#elif defined(__AVX__)
    __m256 y0;
    y0 = _mm256_setzero_ps();
    glmm_store256(mat[0], y0);
    glmm_store256(mat[2], y0);
#elif defined( __SSE__ ) || defined( __SSE2__ )
    glmm_128 x0;
    x0 = _mm_setzero_ps();
    glmm_store(mat[0], x0);
    glmm_store(mat[1], x0);
    glmm_store(mat[2], x0);
    glmm_store(mat[3], x0);
#elif defined(PLAY_CGLM_NEON_FP)
    glmm_128 x0;
    x0 = vdupq_n_f32(0.0f);
    vst1q_f32(mat[0], x0);
    vst1q_f32(mat[1], x0);
    vst1q_f32(mat[2], x0);
    vst1q_f32(mat[3], x0);
#else
    PLAY_CGLM_ALIGN_MAT mat4 t = PLAY_CGLM_MAT4_ZERO_INIT;
    mat4_copy(t, mat);
#endif
}

/*!
 * @brief copy upper-left of mat4 to mat3
 *
 * @param[in]  mat  source
 * @param[out] dest destination
 */
PLAY_CGLM_INLINE
void
mat4_pick3(mat4 mat, mat3 dest)
{
    dest[0][0] = mat[0][0];
    dest[0][1] = mat[0][1];
    dest[0][2] = mat[0][2];

    dest[1][0] = mat[1][0];
    dest[1][1] = mat[1][1];
    dest[1][2] = mat[1][2];

    dest[2][0] = mat[2][0];
    dest[2][1] = mat[2][1];
    dest[2][2] = mat[2][2];
}

/*!
 * @brief copy upper-left of mat4 to mat3 (transposed)
 *
 * the postfix t stands for transpose
 *
 * @param[in]  mat  source
 * @param[out] dest destination
 */
PLAY_CGLM_INLINE
void
mat4_pick3t(mat4 mat, mat3 dest)
{
    dest[0][0] = mat[0][0];
    dest[0][1] = mat[1][0];
    dest[0][2] = mat[2][0];

    dest[1][0] = mat[0][1];
    dest[1][1] = mat[1][1];
    dest[1][2] = mat[2][1];

    dest[2][0] = mat[0][2];
    dest[2][1] = mat[1][2];
    dest[2][2] = mat[2][2];
}

/*!
 * @brief copy mat3 to mat4's upper-left
 *
 * @param[in]  mat  source
 * @param[out] dest destination
 */
PLAY_CGLM_INLINE
void
mat4_ins3(mat3 mat, mat4 dest)
{
    dest[0][0] = mat[0][0];
    dest[0][1] = mat[0][1];
    dest[0][2] = mat[0][2];

    dest[1][0] = mat[1][0];
    dest[1][1] = mat[1][1];
    dest[1][2] = mat[1][2];

    dest[2][0] = mat[2][0];
    dest[2][1] = mat[2][1];
    dest[2][2] = mat[2][2];
}

/*!
 * @brief multiply m1 and m2 to dest
 *
 * m1, m2 and dest matrices can be same matrix, it is possible to write this:
 *
 * @code
 * mat4 m = PLAY_CGLM_MAT4_IDENTITY_INIT;
 * mat4_mul(m, m, m);
 * @endcode
 *
 * @param[in]  m1   left matrix
 * @param[in]  m2   right matrix
 * @param[out] dest destination matrix
 */
PLAY_CGLM_INLINE
void
mat4_mul(mat4 m1, mat4 m2, mat4 dest)
{
#if defined(__wasm__) && defined(__wasm_simd128__)
    mat4_mul_wasm(m1, m2, dest);
#elif defined(__AVX__)
    mat4_mul_avx(m1, m2, dest);
#elif defined( __SSE__ ) || defined( __SSE2__ )
    mat4_mul_sse2(m1, m2, dest);
#elif defined(PLAY_CGLM_NEON_FP)
    mat4_mul_neon(m1, m2, dest);
#else
    float a00 = m1[0][0], a01 = m1[0][1], a02 = m1[0][2], a03 = m1[0][3],
                                                          a10 = m1[1][0], a11 = m1[1][1], a12 = m1[1][2], a13 = m1[1][3],
                                                                                                          a20 = m1[2][0], a21 = m1[2][1], a22 = m1[2][2], a23 = m1[2][3],
                                                                                                                                                          a30 = m1[3][0], a31 = m1[3][1], a32 = m1[3][2], a33 = m1[3][3],

                                                                                                                                                                                                          b00 = m2[0][0], b01 = m2[0][1], b02 = m2[0][2], b03 = m2[0][3],
                                                                                                                                                                                                                                                          b10 = m2[1][0], b11 = m2[1][1], b12 = m2[1][2], b13 = m2[1][3],
                                                                                                                                                                                                                                                                                                          b20 = m2[2][0], b21 = m2[2][1], b22 = m2[2][2], b23 = m2[2][3],
                                                                                                                                                                                                                                                                                                                                                          b30 = m2[3][0], b31 = m2[3][1], b32 = m2[3][2], b33 = m2[3][3];

    dest[0][0] = a00 * b00 + a10 * b01 + a20 * b02 + a30 * b03;
    dest[0][1] = a01 * b00 + a11 * b01 + a21 * b02 + a31 * b03;
    dest[0][2] = a02 * b00 + a12 * b01 + a22 * b02 + a32 * b03;
    dest[0][3] = a03 * b00 + a13 * b01 + a23 * b02 + a33 * b03;
    dest[1][0] = a00 * b10 + a10 * b11 + a20 * b12 + a30 * b13;
    dest[1][1] = a01 * b10 + a11 * b11 + a21 * b12 + a31 * b13;
    dest[1][2] = a02 * b10 + a12 * b11 + a22 * b12 + a32 * b13;
    dest[1][3] = a03 * b10 + a13 * b11 + a23 * b12 + a33 * b13;
    dest[2][0] = a00 * b20 + a10 * b21 + a20 * b22 + a30 * b23;
    dest[2][1] = a01 * b20 + a11 * b21 + a21 * b22 + a31 * b23;
    dest[2][2] = a02 * b20 + a12 * b21 + a22 * b22 + a32 * b23;
    dest[2][3] = a03 * b20 + a13 * b21 + a23 * b22 + a33 * b23;
    dest[3][0] = a00 * b30 + a10 * b31 + a20 * b32 + a30 * b33;
    dest[3][1] = a01 * b30 + a11 * b31 + a21 * b32 + a31 * b33;
    dest[3][2] = a02 * b30 + a12 * b31 + a22 * b32 + a32 * b33;
    dest[3][3] = a03 * b30 + a13 * b31 + a23 * b32 + a33 * b33;
#endif
}

/*!
 * @brief mupliply N mat4 matrices and store result in dest
 *
 * this function lets you multiply multiple (more than two or more...) matrices
 * <br><br>multiplication will be done in loop, this may reduce instructions
 * size but if <b>len</b> is too small then compiler may unroll whole loop,
 * usage:
 * @code
 * mat4 m1, m2, m3, m4, res;
 *
 * mat4_mulN((mat4 *[]){&m1, &m2, &m3, &m4}, 4, res);
 * @endcode
 *
 * @warning matrices parameter is pointer array not mat4 array!
 *
 * @param[in]  matrices mat4 * array
 * @param[in]  len      matrices count
 * @param[out] dest     result
 */
PLAY_CGLM_INLINE
void
mat4_mulN(mat4 * __restrict matrices[], uint32_t len, mat4 dest)
{
    uint32_t i;

#ifndef NDEBUG
    assert(len > 1 && "there must be least 2 matrices to go!");
#endif

    mat4_mul(*matrices[0], *matrices[1], dest);

    for (i = 2; i < len; i++)
        mat4_mul(dest, *matrices[i], dest);
}

/*!
 * @brief multiply mat4 with vec4 (column vector) and store in dest vector
 *
 * @param[in]  m    mat4 (left)
 * @param[in]  v    vec4 (right, column vector)
 * @param[out] dest vec4 (result, column vector)
 */
PLAY_CGLM_INLINE
void
mat4_mulv(mat4 m, vec4 v, vec4 dest)
{
#if defined(__wasm__) && defined(__wasm_simd128__)
    mat4_mulv_wasm(m, v, dest);
#elif defined( __SSE__ ) || defined( __SSE2__ )
    mat4_mulv_sse2(m, v, dest);
#elif defined(PLAY_CGLM_NEON_FP)
    mat4_mulv_neon(m, v, dest);
#else
    vec4 res;
    res[0] = m[0][0] * v[0] + m[1][0] * v[1] + m[2][0] * v[2] + m[3][0] * v[3];
    res[1] = m[0][1] * v[0] + m[1][1] * v[1] + m[2][1] * v[2] + m[3][1] * v[3];
    res[2] = m[0][2] * v[0] + m[1][2] * v[1] + m[2][2] * v[2] + m[3][2] * v[3];
    res[3] = m[0][3] * v[0] + m[1][3] * v[1] + m[2][3] * v[2] + m[3][3] * v[3];
    vec4_copy(res, dest);
#endif
}

/*!
 * @brief trace of matrix
 *
 * sum of the elements on the main diagonal from upper left to the lower right
 *
 * @param[in]  m matrix
 */
PLAY_CGLM_INLINE
float
mat4_trace(mat4 m)
{
    return m[0][0] + m[1][1] + m[2][2] + m[3][3];
}

/*!
 * @brief trace of matrix (rotation part)
 *
 * sum of the elements on the main diagonal from upper left to the lower right
 *
 * @param[in]  m matrix
 */
PLAY_CGLM_INLINE
float
mat4_trace3(mat4 m)
{
    return m[0][0] + m[1][1] + m[2][2];
}

/*!
 * @brief convert mat4's rotation part to quaternion
 *
 * @param[in]  m    affine matrix
 * @param[out] dest destination quaternion
 */
PLAY_CGLM_INLINE
void
mat4_quat(mat4 m, versor dest)
{
    float trace, r, rinv;

    /* it seems using like m12 instead of m[1][2] causes extra instructions */

    trace = m[0][0] + m[1][1] + m[2][2];
    if (trace >= 0.0f)
    {
        r       = sqrtf(1.0f + trace);
        rinv    = 0.5f / r;

        dest[0] = rinv * (m[1][2] - m[2][1]);
        dest[1] = rinv * (m[2][0] - m[0][2]);
        dest[2] = rinv * (m[0][1] - m[1][0]);
        dest[3] = r    * 0.5f;
    }
    else if (m[0][0] >= m[1][1] && m[0][0] >= m[2][2])
    {
        r       = sqrtf(1.0f - m[1][1] - m[2][2] + m[0][0]);
        rinv    = 0.5f / r;

        dest[0] = r    * 0.5f;
        dest[1] = rinv * (m[0][1] + m[1][0]);
        dest[2] = rinv * (m[0][2] + m[2][0]);
        dest[3] = rinv * (m[1][2] - m[2][1]);
    }
    else if (m[1][1] >= m[2][2])
    {
        r       = sqrtf(1.0f - m[0][0] - m[2][2] + m[1][1]);
        rinv    = 0.5f / r;

        dest[0] = rinv * (m[0][1] + m[1][0]);
        dest[1] = r    * 0.5f;
        dest[2] = rinv * (m[1][2] + m[2][1]);
        dest[3] = rinv * (m[2][0] - m[0][2]);
    }
    else
    {
        r       = sqrtf(1.0f - m[0][0] - m[1][1] + m[2][2]);
        rinv    = 0.5f / r;

        dest[0] = rinv * (m[0][2] + m[2][0]);
        dest[1] = rinv * (m[1][2] + m[2][1]);
        dest[2] = r    * 0.5f;
        dest[3] = rinv * (m[0][1] - m[1][0]);
    }
}

/*!
 * @brief multiply vector with mat4
 *
 * actually the result is vec4, after multiplication the last component
 * is trimmed. if you need it don't use this func.
 *
 * @param[in]  m    mat4_new(affine transform)
 * @param[in]  v    vec3
 * @param[in]  last 4th item to make it vec4
 * @param[out] dest result vector (vec3)
 */
PLAY_CGLM_INLINE
void
mat4_mulv3(mat4 m, vec3 v, float last, vec3 dest)
{
    vec4 res;
    vec4_new(v, last, res);
    mat4_mulv(m, res, res);
    vec3_new(res, dest);
}

/*!
 * @brief transpose mat4 and store in dest
 *
 * source matrix will not be transposed unless dest is m
 *
 * @param[in]  m    matrix
 * @param[out] dest result
 */
PLAY_CGLM_INLINE
void
mat4_transpose_to(mat4 m, mat4 dest)
{
#if defined(__wasm__) && defined(__wasm_simd128__)
    mat4_transp_wasm(m, dest);
#elif defined(__AVX__)
    mat4_transp_avx(m, dest);
#elif defined( __SSE__ ) || defined( __SSE2__ )
    mat4_transp_sse2(m, dest);
#elif defined(PLAY_CGLM_NEON_FP)
    mat4_transp_neon(m, dest);
#else
    dest[0][0] = m[0][0];
    dest[1][0] = m[0][1];
    dest[0][1] = m[1][0];
    dest[1][1] = m[1][1];
    dest[0][2] = m[2][0];
    dest[1][2] = m[2][1];
    dest[0][3] = m[3][0];
    dest[1][3] = m[3][1];
    dest[2][0] = m[0][2];
    dest[3][0] = m[0][3];
    dest[2][1] = m[1][2];
    dest[3][1] = m[1][3];
    dest[2][2] = m[2][2];
    dest[3][2] = m[2][3];
    dest[2][3] = m[3][2];
    dest[3][3] = m[3][3];
#endif
}

/*!
 * @brief transpose mat4 and store result in same matrix
 *
 * @param[in, out] m source and dest
 */
PLAY_CGLM_INLINE
void
mat4_transpose(mat4 m)
{
#if defined(__wasm__) && defined(__wasm_simd128__)
    mat4_transp_wasm(m, m);
#elif defined(__AVX__)
    mat4_transp_avx(m, m);
#elif defined( __SSE__ ) || defined( __SSE2__ )
    mat4_transp_sse2(m, m);
#elif defined(PLAY_CGLM_NEON_FP)
    mat4_transp_neon(m, m);
#else
    mat4 d;
    mat4_transpose_to(m, d);
    mat4_ucopy(d, m);
#endif
}

/*!
 * @brief scale (multiply with scalar) matrix without simd optimization
 *
 * multiply matrix with scalar
 *
 * @param[in, out] m matrix
 * @param[in]      s scalar
 */
PLAY_CGLM_INLINE
void
mat4_scale_p(mat4 m, float s)
{
    m[0][0] *= s;
    m[0][1] *= s;
    m[0][2] *= s;
    m[0][3] *= s;
    m[1][0] *= s;
    m[1][1] *= s;
    m[1][2] *= s;
    m[1][3] *= s;
    m[2][0] *= s;
    m[2][1] *= s;
    m[2][2] *= s;
    m[2][3] *= s;
    m[3][0] *= s;
    m[3][1] *= s;
    m[3][2] *= s;
    m[3][3] *= s;
}

/*!
 * @brief scale (multiply with scalar) matrix
 *
 * multiply matrix with scalar
 *
 * @param[in, out] m matrix
 * @param[in]      s scalar
 */
PLAY_CGLM_INLINE
void
mat4_scale(mat4 m, float s)
{
#if defined(__wasm__) && defined(__wasm_simd128__)
    mat4_scale_wasm(m, s);
#elif defined(__AVX__)
    mat4_scale_avx(m, s);
#elif defined( __SSE__ ) || defined( __SSE2__ )
    mat4_scale_sse2(m, s);
#elif defined(PLAY_CGLM_NEON_FP)
    mat4_scale_neon(m, s);
#else
    mat4_scale_p(m, s);
#endif
}

/*!
 * @brief mat4 determinant
 *
 * @param[in] mat matrix
 *
 * @return determinant
 */
PLAY_CGLM_INLINE
float
mat4_det(mat4 mat)
{
#if defined(__wasm__) && defined(__wasm_simd128__)
    return mat4_det_wasm(mat);
#elif defined( __SSE__ ) || defined( __SSE2__ )
    return mat4_det_sse2(mat);
#elif defined(PLAY_CGLM_NEON_FP)
    return mat4_det_neon(mat);
#else
    /* [square] det(A) = det(At) */
    float t[6];
    float a = mat[0][0], b = mat[0][1], c = mat[0][2], d = mat[0][3],
                                                       e = mat[1][0], f = mat[1][1], g = mat[1][2], h = mat[1][3],
                                                                                                    i = mat[2][0], j = mat[2][1], k = mat[2][2], l = mat[2][3],
                                                                                                                                                 m = mat[3][0], n = mat[3][1], o = mat[3][2], p = mat[3][3];

    t[0] = k * p - o * l;
    t[1] = j * p - n * l;
    t[2] = j * o - n * k;
    t[3] = i * p - m * l;
    t[4] = i * o - m * k;
    t[5] = i * n - m * j;

    return a * (f * t[0] - g * t[1] + h * t[2])
           - b * (e * t[0] - g * t[3] + h * t[4])
           + c * (e * t[1] - f * t[3] + h * t[5])
           - d * (e * t[2] - f * t[4] + g * t[5]);
#endif
}

/*!
 * @brief inverse mat4 and store in dest
 *
 * @param[in]  mat  matrix
 * @param[out] dest inverse matrix
 */
PLAY_CGLM_INLINE
void
mat4_inv(mat4 mat, mat4 dest)
{
#if defined(__wasm__) && defined(__wasm_simd128__)
    mat4_inv_wasm(mat, dest);
#elif defined( __SSE__ ) || defined( __SSE2__ )
    mat4_inv_sse2(mat, dest);
#elif defined(PLAY_CGLM_NEON_FP)
    mat4_inv_neon(mat, dest);
#else
    float a = mat[0][0], b = mat[0][1], c = mat[0][2], d = mat[0][3],
                                                       e = mat[1][0], f = mat[1][1], g = mat[1][2], h = mat[1][3],
                                                                                                    i = mat[2][0], j = mat[2][1], k = mat[2][2], l = mat[2][3],
                                                                                                                                                 m = mat[3][0], n = mat[3][1], o = mat[3][2], p = mat[3][3],

                                                                                                                                                                                              c1  = k * p - l * o,  c2  = c * h - d * g,  c3  = i * p - l * m,
                                                                                                                                                                                                                                          c4  = a * h - d * e,  c5  = j * p - l * n,  c6  = b * h - d * f,
                                                                                                                                                                                                                                                                                      c7  = i * n - j * m,  c8  = a * f - b * e,  c9  = j * o - k * n,
                                                                                                                                                                                                                                                                                                                                  c10 = b * g - c * f,  c11 = i * o - k * m,  c12 = a * g - c * e,

                                                                                                                                                                                                                                                                                                                                                                              idt = 1.0f/(c8*c1+c4*c9+c10*c3+c2*c7-c12*c5-c6*c11), ndt = -idt;

    dest[0][0] = (f * c1  - g * c5  + h * c9)  * idt;
    dest[0][1] = (b * c1  - c * c5  + d * c9)  * ndt;
    dest[0][2] = (n * c2  - o * c6  + p * c10) * idt;
    dest[0][3] = (j * c2  - k * c6  + l * c10) * ndt;

    dest[1][0] = (e * c1  - g * c3  + h * c11) * ndt;
    dest[1][1] = (a * c1  - c * c3  + d * c11) * idt;
    dest[1][2] = (m * c2  - o * c4  + p * c12) * ndt;
    dest[1][3] = (i * c2  - k * c4  + l * c12) * idt;

    dest[2][0] = (e * c5  - f * c3  + h * c7)  * idt;
    dest[2][1] = (a * c5  - b * c3  + d * c7)  * ndt;
    dest[2][2] = (m * c6  - n * c4  + p * c8)  * idt;
    dest[2][3] = (i * c6  - j * c4  + l * c8)  * ndt;

    dest[3][0] = (e * c9  - f * c11 + g * c7)  * ndt;
    dest[3][1] = (a * c9  - b * c11 + c * c7)  * idt;
    dest[3][2] = (m * c10 - n * c12 + o * c8)  * ndt;
    dest[3][3] = (i * c10 - j * c12 + k * c8)  * idt;
#endif
}

/*!
 * @brief inverse mat4 and store in dest
 *
 * this func uses reciprocal approximation without extra corrections
 * e.g Newton-Raphson. this should work faster than normal,
 * to get more precise use mat4_inv version.
 *
 * NOTE: You will lose precision, mat4_inv is more accurate
 *
 * @param[in]  mat  matrix
 * @param[out] dest inverse matrix
 */
PLAY_CGLM_INLINE
void
mat4_inv_fast(mat4 mat, mat4 dest)
{
#if defined(__wasm__) && defined(__wasm_simd128__)
    mat4_inv_fast_wasm(mat, dest);
#elif defined( __SSE__ ) || defined( __SSE2__ )
    mat4_inv_fast_sse2(mat, dest);
#else
    mat4_inv(mat, dest);
#endif
}

/*!
 * @brief swap two matrix columns
 *
 * @param[in,out] mat  matrix
 * @param[in]     col1 col1
 * @param[in]     col2 col2
 */
PLAY_CGLM_INLINE
void
mat4_swap_col(mat4 mat, int col1, int col2)
{
    PLAY_CGLM_ALIGN(16) vec4 tmp;
    vec4_copy(mat[col1], tmp);
    vec4_copy(mat[col2], mat[col1]);
    vec4_copy(tmp, mat[col2]);
}

/*!
 * @brief swap two matrix rows
 *
 * @param[in,out] mat  matrix
 * @param[in]     row1 row1
 * @param[in]     row2 row2
 */
PLAY_CGLM_INLINE
void
mat4_swap_row(mat4 mat, int row1, int row2)
{
    PLAY_CGLM_ALIGN(16) vec4 tmp;
    tmp[0] = mat[0][row1];
    tmp[1] = mat[1][row1];
    tmp[2] = mat[2][row1];
    tmp[3] = mat[3][row1];

    mat[0][row1] = mat[0][row2];
    mat[1][row1] = mat[1][row2];
    mat[2][row1] = mat[2][row2];
    mat[3][row1] = mat[3][row2];

    mat[0][row2] = tmp[0];
    mat[1][row2] = tmp[1];
    mat[2][row2] = tmp[2];
    mat[3][row2] = tmp[3];
}

/*!
 * @brief helper for  R (row vector) * M (matrix) * C (column vector)
 *
 * rmc stands for Row * Matrix * Column
 *
 * the result is scalar because R * M = Matrix1x4 (row vector),
 * then Matrix1x4 * Vec4 (column vector) = Matrix1x1 (Scalar)
 *
 * @param[in]  r   row vector or matrix1x4
 * @param[in]  m   matrix4x4
 * @param[in]  c   column vector or matrix4x1
 *
 * @return scalar value e.g. B(s)
 */
PLAY_CGLM_INLINE
float
mat4_rmc(vec4 r, mat4 m, vec4 c)
{
    vec4 tmp;
    mat4_mulv(m, c, tmp);
    return vec4_dot(r, tmp);
}

/*!
 * @brief Create mat4 matrix from pointer
 *
 * @param[in]  src  pointer to an array of floats
 * @param[out] dest matrix
 */
PLAY_CGLM_INLINE
void
mat4_make(const float * __restrict src, mat4 dest)
{
    dest[0][0] = src[0];
    dest[1][0] = src[4];
    dest[0][1] = src[1];
    dest[1][1] = src[5];
    dest[0][2] = src[2];
    dest[1][2] = src[6];
    dest[0][3] = src[3];
    dest[1][3] = src[7];

    dest[2][0] = src[8];
    dest[3][0] = src[12];
    dest[2][1] = src[9];
    dest[3][1] = src[13];
    dest[2][2] = src[10];
    dest[3][2] = src[14];
    dest[2][3] = src[11];
    dest[3][3] = src[15];
}

/*!
 * @brief Create mat4 matrix from texture transform parameters
 *
 * @param[in]  sx   scale x
 * @param[in]  sy   scale y
 * @param[in]  rot  rotation in radians CCW/RH
 * @param[in]  tx   translate x
 * @param[in]  ty   translate y
 * @param[out] dest texture transform matrix
 */
PLAY_CGLM_INLINE
void
mat4_textrans(float sx, float sy, float rot, float tx, float ty, mat4 dest)
{
    float c, s;

    c = cosf(rot);
    s = sinf(rot);

    mat4_identity(dest);

    dest[0][0] =  c * sx;
    dest[0][1] = -s * sy;
    dest[1][0] =  s * sx;
    dest[1][1] =  c * sy;
    dest[3][0] =  tx;
    dest[3][1] =  ty;
}



/*** End of inlined file: mat4.h ***/


/*** Start of inlined file: mat4x2.h ***/
/*
 Macros:
   PLAY_CGLM_MAT4X2_ZERO_INIT
   PLAY_CGLM_MAT4X2_ZERO

 Functions:
   PLAY_CGLM_INLINE void mat4x2_copy(mat4x2 src, mat4x2 dest);
   PLAY_CGLM_INLINE void mat4x2_zero(mat4x2 m);
   PLAY_CGLM_INLINE void mat4x2_make(const float * __restrict src, mat4x2 dest);
   PLAY_CGLM_INLINE void mat4x2_mul(mat4x2 m1, mat2x4 m2, mat2 dest);
   PLAY_CGLM_INLINE void mat4x2_mulv(mat4x2 m, vec4 v, vec2 dest);
   PLAY_CGLM_INLINE void mat4x2_transpose(mat4x2 src, mat2x4 dest);
   PLAY_CGLM_INLINE void mat4x2_scale(mat4x2 m, float s);
 */

#ifndef cmat4x2_h
#define cmat4x2_h

#define PLAY_CGLM_MAT4X2_ZERO_INIT {{0.0f, 0.0f}, {0.0f, 0.0f}, {0.0f, 0.0f}, {0.0f, 0.0f}}

/* for C only */
#define PLAY_CGLM_MAT4X2_ZERO PLAY_CGLM_MAT4X2_ZERO_INIT

/*!
 * @brief Copy mat4x2 (src) to mat4x2 (dest).
 *
 * @param[in]  src  mat4x2 (left)
 * @param[out] dest destination (result, mat4x2)
 */
PLAY_CGLM_INLINE
void
mat4x2_copy(mat4x2 src, mat4x2 dest)
{
    vec2_copy(src[0], dest[0]);
    vec2_copy(src[1], dest[1]);
    vec2_copy(src[2], dest[2]);
    vec2_copy(src[3], dest[3]);
}

/*!
 * @brief Zero out the mat4x2 (m).
 *
 * @param[in, out] mat4x2 (src, dest)
 */
PLAY_CGLM_INLINE
void
mat4x2_zero(mat4x2 m)
{
    PLAY_CGLM_ALIGN_MAT mat4x2 t = PLAY_CGLM_MAT4X2_ZERO_INIT;
    mat4x2_copy(t, m);
}

/*!
 * @brief Create mat4x2 (dest) from pointer (src).
 *
 * @param[in]  src  pointer to an array of floats (left)
 * @param[out] dest destination (result, mat4x2)
 */
PLAY_CGLM_INLINE
void
mat4x2_make(const float * __restrict src, mat4x2 dest)
{
    dest[0][0] = src[0];
    dest[0][1] = src[1];

    dest[1][0] = src[2];
    dest[1][1] = src[3];

    dest[2][0] = src[4];
    dest[2][1] = src[5];

    dest[3][0] = src[6];
    dest[3][1] = src[7];
}

/*!
 * @brief Multiply mat4x2 (m1) by mat2x4 (m2) and store in mat2 (dest).
 *
 * @code
 * mat4x2_mul(mat4x2, mat2x4, mat2);
 * @endcode
 *
 * @param[in]  m1   mat4x2 (left)
 * @param[in]  m2   mat2x4 (right)
 * @param[out] dest destination (result, mat2)
 */
PLAY_CGLM_INLINE
void
mat4x2_mul(mat4x2 m1, mat2x4 m2, mat2 dest)
{
    float a00 = m1[0][0], a01 = m1[0][1],
                          a10 = m1[1][0], a11 = m1[1][1],
                                          a20 = m1[2][0], a21 = m1[2][1],
                                                          a30 = m1[3][0], a31 = m1[3][1],

                                                                          b00 = m2[0][0], b01 = m2[0][1], b02 = m2[0][2], b03 = m2[0][3],
                                                                                                                          b10 = m2[1][0], b11 = m2[1][1], b12 = m2[1][2], b13 = m2[1][3];

    dest[0][0] = a00 * b00 + a10 * b01 + a20 * b02 + a30 * b03;
    dest[0][1] = a01 * b00 + a11 * b01 + a21 * b02 + a31 * b03;

    dest[1][0] = a00 * b10 + a10 * b11 + a20 * b12 + a30 * b13;
    dest[1][1] = a01 * b10 + a11 * b11 + a21 * b12 + a31 * b13;
}

/*!
 * @brief Multiply mat4x2 (m) by vec4 (v) and store in vec2 (dest).
 *
 * @param[in]  m    mat4x2 (left)
 * @param[in]  v    vec4 (right, column vector)
 * @param[out] dest destination (result, column vector)
 */
PLAY_CGLM_INLINE
void
mat4x2_mulv(mat4x2 m, vec4 v, vec2 dest)
{
    float v0 = v[0], v1 = v[1], v2 = v[2], v3 = v[3];

    dest[0] = m[0][0] * v0 + m[1][0] * v1 + m[2][0] * v2 + m[3][0] * v3;
    dest[1] = m[0][1] * v0 + m[1][1] * v1 + m[2][1] * v2 + m[3][1] * v3;
}

/*!
 * @brief Transpose mat4x2 (src) and store in mat2x4 (dest).
 *
 * @param[in]  src  mat4x2 (left)
 * @param[out] dest destination (result, mat2x4)
 */
PLAY_CGLM_INLINE
void
mat4x2_transpose(mat4x2 m, mat2x4 dest)
{
    dest[0][0] = m[0][0];
    dest[0][1] = m[1][0];
    dest[0][2] = m[2][0];
    dest[0][3] = m[3][0];
    dest[1][0] = m[0][1];
    dest[1][1] = m[1][1];
    dest[1][2] = m[2][1];
    dest[1][3] = m[3][1];
}

/*!
 * @brief Multiply mat4x2 (m) by scalar constant (s).
 *
 * @param[in, out] m (src, dest)
 * @param[in]      s float (scalar)
 */
PLAY_CGLM_INLINE
void
mat4x2_scale(mat4x2 m, float s)
{
    m[0][0] *= s;
    m[0][1] *= s;
    m[1][0] *= s;
    m[1][1] *= s;
    m[2][0] *= s;
    m[2][1] *= s;
    m[3][0] *= s;
    m[3][1] *= s;
}

#endif

/*** End of inlined file: mat4x2.h ***/


/*** Start of inlined file: mat4x3.h ***/
/*
 Macros:
   PLAY_CGLM_MAT4X3_ZERO_INIT
   PLAY_CGLM_MAT4X3_ZERO

 Functions:
   PLAY_CGLM_INLINE void mat4x3_copy(mat4x3 src, mat4x3 dest);
   PLAY_CGLM_INLINE void mat4x3_zero(mat4x3 m);
   PLAY_CGLM_INLINE void mat4x3_make(const float * __restrict src, mat4x3 dest);
   PLAY_CGLM_INLINE void mat4x3_mul(mat4x3 m1, mat3x4 m2, mat3 dest);
   PLAY_CGLM_INLINE void mat4x3_mulv(mat4x3 m, vec4 v, vec3 dest);
   PLAY_CGLM_INLINE void mat4x3_transpose(mat4x3 src, mat3x4 dest);
   PLAY_CGLM_INLINE void mat4x3_scale(mat4x3 m, float s);
 */

#ifndef cmat4x3_h
#define cmat4x3_h

#define PLAY_CGLM_MAT4X3_ZERO_INIT {{0.0f, 0.0f, 0.0f}, {0.0f, 0.0f, 0.0f}, \
                              {0.0f, 0.0f, 0.0f}, {0.0f, 0.0f, 0.0f}}

/* for C only */
#define PLAY_CGLM_MAT4X3_ZERO PLAY_CGLM_MAT4X3_ZERO_INIT

/*!
 * @brief Copy mat4x3 (src) to mat4x3 (dest).
 *
 * @param[in]  src  mat4x3 (left)
 * @param[out] dest destination (result, mat4x3)
 */
PLAY_CGLM_INLINE
void
mat4x3_copy(mat4x3 src, mat4x3 dest)
{
    vec3_copy(src[0], dest[0]);
    vec3_copy(src[1], dest[1]);
    vec3_copy(src[2], dest[2]);
    vec3_copy(src[3], dest[3]);
}

/*!
 * @brief Zero out the mat4x3 (m).
 *
 * @param[in, out] mat4x3 (src, dest)
 */
PLAY_CGLM_INLINE
void
mat4x3_zero(mat4x3 m)
{
    PLAY_CGLM_ALIGN_MAT mat4x3 t = PLAY_CGLM_MAT4X3_ZERO_INIT;
    mat4x3_copy(t, m);
}

/*!
 * @brief Create mat4x3 (dest) from pointer (src).
 *
 * @param[in]  src  pointer to an array of floats (left)
 * @param[out] dest destination (result, mat4x3)
 */
PLAY_CGLM_INLINE
void
mat4x3_make(const float * __restrict src, mat4x3 dest)
{
    dest[0][0] = src[0];
    dest[0][1] = src[1];
    dest[0][2] = src[2];

    dest[1][0] = src[3];
    dest[1][1] = src[4];
    dest[1][2] = src[5];

    dest[2][0] = src[6];
    dest[2][1] = src[7];
    dest[2][2] = src[8];

    dest[3][0] = src[9];
    dest[3][1] = src[10];
    dest[3][2] = src[11];
}

/*!
 * @brief Multiply mat4x3 (m1) by mat3x4 (m2) and store in mat3 (dest).
 *
 * @code
 * mat4x3_mul(mat4x3, mat3x4, mat3);
 * @endcode
 *
 * @param[in]  m1   mat4x3 (left)
 * @param[in]  m2   mat3x4 (right)
 * @param[out] dest destination (result, mat3)
 */
PLAY_CGLM_INLINE
void
mat4x3_mul(mat4x3 m1, mat3x4 m2, mat3 dest)
{
    float a00 = m1[0][0], a01 = m1[0][1], a02 = m1[0][2],
                                          a10 = m1[1][0], a11 = m1[1][1], a12 = m1[1][2],
                                                                          a20 = m1[2][0], a21 = m1[2][1], a22 = m1[2][2],
                                                                                                          a30 = m1[3][0], a31 = m1[3][1], a32 = m1[3][2],

                                                                                                                                          b00 = m2[0][0], b01 = m2[0][1], b02 = m2[0][2], b03 = m2[0][3],
                                                                                                                                                                                          b10 = m2[1][0], b11 = m2[1][1], b12 = m2[1][2], b13 = m2[1][3],
                                                                                                                                                                                                                                          b20 = m2[2][0], b21 = m2[2][1], b22 = m2[2][2], b23 = m2[2][3];

    dest[0][0] = a00 * b00 + a10 * b01 + a20 * b02 + a30 * b03;
    dest[0][1] = a01 * b00 + a11 * b01 + a21 * b02 + a31 * b03;
    dest[0][2] = a02 * b00 + a12 * b01 + a22 * b02 + a32 * b03;

    dest[1][0] = a00 * b10 + a10 * b11 + a20 * b12 + a30 * b13;
    dest[1][1] = a01 * b10 + a11 * b11 + a21 * b12 + a31 * b13;
    dest[1][2] = a02 * b10 + a12 * b11 + a22 * b12 + a32 * b13;

    dest[2][0] = a00 * b20 + a10 * b21 + a20 * b22 + a30 * b23;
    dest[2][1] = a01 * b20 + a11 * b21 + a21 * b22 + a31 * b23;
    dest[2][2] = a02 * b20 + a12 * b21 + a22 * b22 + a32 * b23;
}

/*!
 * @brief Multiply mat4x3 (m) by vec4 (v) and store in vec3 (dest).
 *
 * @param[in]  m    mat4x3 (left)
 * @param[in]  v    vec3 (right, column vector)
 * @param[out] dest destination (result, column vector)
 */
PLAY_CGLM_INLINE
void
mat4x3_mulv(mat4x3 m, vec4 v, vec3 dest)
{
    float v0 = v[0], v1 = v[1], v2 = v[2], v3 = v[3];

    dest[0] = m[0][0] * v0 + m[1][0] * v1 + m[2][0] * v2 + m[3][0] * v3;
    dest[1] = m[0][1] * v0 + m[1][1] * v1 + m[2][1] * v2 + m[3][1] * v3;
    dest[2] = m[0][2] * v0 + m[1][2] * v1 + m[2][2] * v2 + m[3][2] * v3;
}

/*!
 * @brief Transpose mat4x3 (src) and store in mat3x4 (dest).
 *
 * @param[in]  src  mat4x3 (left)
 * @param[out] dest destination (result, mat3x4)
 */
PLAY_CGLM_INLINE
void
mat4x3_transpose(mat4x3 src, mat3x4 dest)
{
    dest[0][0] = src[0][0];
    dest[0][1] = src[1][0];
    dest[0][2] = src[2][0];
    dest[0][3] = src[3][0];
    dest[1][0] = src[0][1];
    dest[1][1] = src[1][1];
    dest[1][2] = src[2][1];
    dest[1][3] = src[3][1];
    dest[2][0] = src[0][2];
    dest[2][1] = src[1][2];
    dest[2][2] = src[2][2];
    dest[2][3] = src[3][2];
}

/*!
 * @brief Multiply mat4x3 (m) by scalar constant (s).
 *
 * @param[in, out] m (src, dest)
 * @param[in]      s float (scalar)
 */
PLAY_CGLM_INLINE
void
mat4x3_scale(mat4x3 m, float s)
{
    m[0][0] *= s;
    m[0][1] *= s;
    m[0][2] *= s;
    m[1][0] *= s;
    m[1][1] *= s;
    m[1][2] *= s;
    m[2][0] *= s;
    m[2][1] *= s;
    m[2][2] *= s;
    m[3][0] *= s;
    m[3][1] *= s;
    m[3][2] *= s;
}

#endif /* cmat4x3_h */

/*** End of inlined file: mat4x3.h ***/


/*** Start of inlined file: mat3.h ***/
/*
 Macros:
   PLAY_CGLM_MAT3_IDENTITY_INIT
   PLAY_CGLM_MAT3_ZERO_INIT
   PLAY_CGLM_MAT3_IDENTITY
   PLAY_CGLM_MAT3_ZERO
   mat3_dup(mat, dest)

 Functions:
   PLAY_CGLM_INLINE void  mat3_copy(mat3 mat, mat3 dest);
   PLAY_CGLM_INLINE void  mat3_identity(mat3 mat);
   PLAY_CGLM_INLINE void  mat3_identity_array(mat3 * restrict mat, size_t count);
   PLAY_CGLM_INLINE void  mat3_zero(mat3 mat);
   PLAY_CGLM_INLINE void  mat3_mul(mat3 m1, mat3 m2, mat3 dest);
   PLAY_CGLM_INLINE void  mat3_transpose_to(mat3 m, mat3 dest);
   PLAY_CGLM_INLINE void  mat3_transpose(mat3 m);
   PLAY_CGLM_INLINE void  mat3_mulv(mat3 m, vec3 v, vec3 dest);
   PLAY_CGLM_INLINE float mat3_trace(mat3 m);
   PLAY_CGLM_INLINE void  mat3_quat(mat3 m, versor dest);
   PLAY_CGLM_INLINE void  mat3_scale(mat3 m, float s);
   PLAY_CGLM_INLINE float mat3_det(mat3 mat);
   PLAY_CGLM_INLINE void  mat3_inv(mat3 mat, mat3 dest);
   PLAY_CGLM_INLINE void  mat3_swap_col(mat3 mat, int col1, int col2);
   PLAY_CGLM_INLINE void  mat3_swap_row(mat3 mat, int row1, int row2);
   PLAY_CGLM_INLINE float mat3_rmc(vec3 r, mat3 m, vec3 c);
   PLAY_CGLM_INLINE void  mat3_make(float * restrict src, mat3 dest);
   PLAY_CGLM_INLINE void  mat3_textrans(float sx, float sy, float rot, float tx, float ty, mat3 dest);
 */

#ifndef cmat3_h
#define cmat3_h

#ifdef PLAY_CGLM_SSE_FP

/*** Start of inlined file: mat3.h ***/
#ifndef cmat3_sse_h
#define cmat3_sse_h
#if defined( __SSE__ ) || defined( __SSE2__ )

PLAY_CGLM_INLINE
void
mat3_mul_sse2(mat3 m1, mat3 m2, mat3 dest)
{
    __m128 l0, l1, l2, r0, r1, r2, x0, x1, x2, x3, x4, x5, x6, x7, x8, x9;

    l0 = _mm_loadu_ps(m1[0]);
    l1 = _mm_loadu_ps(&m1[1][1]);

    r0 = _mm_loadu_ps(m2[0]);
    r1 = _mm_loadu_ps(&m2[1][1]);

    x8 = glmm_shuff1(l0, 0, 2, 1, 0);                     /* a00 a02 a01 a00 */
    x1 = glmm_shuff1(r0, 3, 0, 0, 0);                     /* b10 b00 b00 b00 */
    x2 = _mm_shuffle_ps(l0, l1, _MM_SHUFFLE(1, 0, 3, 3)); /* a12 a11 a10 a10 */
    x3 = _mm_shuffle_ps(r0, r1, _MM_SHUFFLE(2, 0, 3, 1)); /* b20 b11 b10 b01 */
    x0 = _mm_mul_ps(x8, x1);

    x6 = glmm_shuff1(l0, 1, 0, 2, 1);                     /* a01 a00 a02 a01 */
    x7 = glmm_shuff1(x3, 3, 3, 1, 1);                     /* b20 b20 b10 b10 */
    l2 = _mm_load_ss(&m1[2][2]);
    r2 = _mm_load_ss(&m2[2][2]);
    x1 = _mm_mul_ps(x6, x7);
    l2 = glmm_shuff1(l2, 0, 0, 1, 0);                     /* a22 a22 0.f a22 */
    r2 = glmm_shuff1(r2, 0, 0, 1, 0);                     /* b22 b22 0.f b22 */

    x4 = glmm_shuff1(x2, 0, 3, 2, 0);                     /* a10 a12 a11 a10 */
    x5 = glmm_shuff1(x2, 2, 0, 3, 2);                     /* a11 a10 a12 a11 */
    x6 = glmm_shuff1(x3, 2, 0, 0, 0);                     /* b11 b01 b01 b01 */
    x2 = glmm_shuff1(r1, 3, 3, 0, 0);                     /* b21 b21 b11 b11 */

    x8 = _mm_unpackhi_ps(x8, x4);                         /* a10 a00 a12 a02 */
    x9 = _mm_unpackhi_ps(x7, x2);                         /* b21 b20 b21 b20 */

    x0 = glmm_fmadd(x4, x6, x0);
    x1 = glmm_fmadd(x5, x2, x1);

    x2 = _mm_movehl_ps(l2, l1);                           /* a22 a22 a21 a20 */
    x3 = glmm_shuff1(x2, 0, 2, 1, 0);                     /* a20 a22 a21 a20 */
    x2 = glmm_shuff1(x2, 1, 0, 2, 1);                     /* a21 a20 a22 a21 */
    x4 = _mm_shuffle_ps(r0, r1, _MM_SHUFFLE(1, 1, 2, 2)); /* b12 b12 b02 b02 */

    x5 = glmm_shuff1(x4, 3, 0, 0, 0);                     /* b12 b02 b02 b02 */
    x4 = _mm_movehl_ps(r2, x4);                           /* b22 b22 b12 b12 */
    x0 = glmm_fmadd(x3, x5, x0);
    x1 = glmm_fmadd(x2, x4, x1);

    /*
     Dot Product : dest[2][2] =  a02 * b20 +
                                 a12 * b21 +
                                 a22 * b22 +
                                 0   * 00                                    */
    x2 = _mm_movelh_ps(x8, l2);                           /* 0.f a22 a12 a02 */
    x3 = _mm_movelh_ps(x9, r2);                           /* 0.f b22 b21 b20 */
    x2 = glmm_vdots(x2, x3);

    _mm_storeu_ps(&dest[0][0], x0);
    _mm_storeu_ps(&dest[1][1], x1);
    _mm_store_ss (&dest[2][2], x2);
}

#endif
#endif /* cmat3_sse_h */

/*** End of inlined file: mat3.h ***/


#endif

#ifdef PLAY_CGLM_SIMD_WASM

/*** Start of inlined file: mat3.h ***/
#ifndef cmat3_wasm_h
#define cmat3_wasm_h
#if defined(__wasm__) && defined(__wasm_simd128__)

PLAY_CGLM_INLINE
void
mat3_mul_wasm(mat3 m1, mat3 m2, mat3 dest)
{
    glmm_128 l0, l1, l2, r0, r1, r2, x0, x1, x2, x3, x4, x5, x6, x7, x8, x9;

    l0 = wasm_v128_load(m1[0]);
    l1 = wasm_v128_load(&m1[1][1]);

    r0 = wasm_v128_load(m2[0]);
    r1 = wasm_v128_load(&m2[1][1]);

    x8 = glmm_shuff1(l0, 0, 2, 1, 0);                     /* a00 a02 a01 a00 */
    x1 = glmm_shuff1(r0, 3, 0, 0, 0);                     /* b10 b00 b00 b00 */
    x2 = wasm_i32x4_shuffle(l0, l1, 3, 3, 4, 5);          /* a12 a11 a10 a10 */
    x3 = wasm_i32x4_shuffle(r0, r1, 1, 3, 4, 6);          /* b20 b11 b10 b01 */
    x0 = wasm_f32x4_mul(x8, x1);

    x6 = glmm_shuff1(l0, 1, 0, 2, 1);                     /* a01 a00 a02 a01 */
    x7 = glmm_shuff1(x3, 3, 3, 1, 1);                     /* b20 b20 b10 b10 */
    l2 = wasm_v128_load32_zero(&m1[2][2]);
    r2 = wasm_v128_load32_zero(&m2[2][2]);
    x1 = wasm_f32x4_mul(x6, x7);
    l2 = glmm_shuff1(l2, 0, 0, 1, 0);                     /* a22 a22 0.f a22 */
    r2 = glmm_shuff1(r2, 0, 0, 1, 0);                     /* b22 b22 0.f b22 */

    x4 = glmm_shuff1(x2, 0, 3, 2, 0);                     /* a10 a12 a11 a10 */
    x5 = glmm_shuff1(x2, 2, 0, 3, 2);                     /* a11 a10 a12 a11 */
    x6 = glmm_shuff1(x3, 2, 0, 0, 0);                     /* b11 b01 b01 b01 */
    x2 = glmm_shuff1(r1, 3, 3, 0, 0);                     /* b21 b21 b11 b11 */

    /* x8 = _mm_unpackhi_ps(x8, x4); */
    /* x9 = _mm_unpackhi_ps(x7, x2); */
    x8 = wasm_i32x4_shuffle(x8, x4, 2, 6, 3, 7);          /* a10 a00 a12 a02 */
    x9 = wasm_i32x4_shuffle(x7, x2, 2, 6, 3, 7);          /* b21 b20 b21 b20 */

    x0 = glmm_fmadd(x4, x6, x0);
    x1 = glmm_fmadd(x5, x2, x1);

    /* x2 = _mm_movehl_ps(l2, l1); */
    x2 = wasm_i32x4_shuffle(l2, l1, 6, 7, 2, 3);          /* a22 a22 a21 a20 */
    x3 = glmm_shuff1(x2, 0, 2, 1, 0);                     /* a20 a22 a21 a20 */
    x2 = glmm_shuff1(x2, 1, 0, 2, 1);                     /* a21 a20 a22 a21 */
    x4 = wasm_i32x4_shuffle(r0, r1, 2, 2, 5, 5);          /* b12 b12 b02 b02 */

    x5 = glmm_shuff1(x4, 3, 0, 0, 0);                     /* b12 b02 b02 b02 */
    /* x4 = _mm_movehl_ps(r2, x4); */
    x4 = wasm_i32x4_shuffle(r2, x4, 6, 7, 2, 3);          /* b22 b22 b12 b12 */
    x0 = glmm_fmadd(x3, x5, x0);
    x1 = glmm_fmadd(x2, x4, x1);

    /*
     Dot Product : dest[2][2] =  a02 * b20 +
                                 a12 * b21 +
                                 a22 * b22 +
                                 0   * 00                                    */
    /* x2 = _mm_movelh_ps(x8, l2); */
    /* x3 = _mm_movelh_ps(x9, r2); */
    x2 = wasm_i32x4_shuffle(x8, l2, 0, 1, 4, 5);           /* 0.f a22 a12 a02 */
    x3 = wasm_i32x4_shuffle(x9, r2, 0, 1, 4, 5);           /* 0.f b22 b21 b20 */
    x2 = glmm_vdots(x2, x3);

    /* _mm_storeu_ps(&dest[0][0], x0); */
    wasm_v128_store(&dest[0][0], x0);
    /* _mm_storeu_ps(&dest[1][1], x1); */
    wasm_v128_store(&dest[1][1], x1);
    /* _mm_store_ss (&dest[2][2], x2); */
    wasm_v128_store32_lane(&dest[2][2], x2, 0);
}

#endif
#endif /* cmat3_wasm_h */

/*** End of inlined file: mat3.h ***/


#endif

#define PLAY_CGLM_MAT3_IDENTITY_INIT  {{1.0f, 0.0f, 0.0f},                          \
                                 {0.0f, 1.0f, 0.0f},                          \
                                 {0.0f, 0.0f, 1.0f}}
#define PLAY_CGLM_MAT3_ZERO_INIT      {{0.0f, 0.0f, 0.0f},                          \
                                 {0.0f, 0.0f, 0.0f},                          \
                                 {0.0f, 0.0f, 0.0f}}

/* for C only */
#define PLAY_CGLM_MAT3_IDENTITY ((mat3)PLAY_CGLM_MAT3_IDENTITY_INIT)
#define PLAY_CGLM_MAT3_ZERO     ((mat3)PLAY_CGLM_MAT3_ZERO_INIT)

/* DEPRECATED! use _copy, _ucopy versions */
#define mat3_dup(mat, dest) mat3_copy(mat, dest)

/*!
 * @brief copy all members of [mat] to [dest]
 *
 * @param[in]  mat  source
 * @param[out] dest destination
 */
PLAY_CGLM_INLINE
void
mat3_copy(mat3 mat, mat3 dest)
{
    dest[0][0] = mat[0][0];
    dest[0][1] = mat[0][1];
    dest[0][2] = mat[0][2];

    dest[1][0] = mat[1][0];
    dest[1][1] = mat[1][1];
    dest[1][2] = mat[1][2];

    dest[2][0] = mat[2][0];
    dest[2][1] = mat[2][1];
    dest[2][2] = mat[2][2];
}

/*!
 * @brief make given matrix identity. It is identical with below,
 *        but it is more easy to do that with this func especially for members
 *        e.g. mat3_identity(aStruct->aMatrix);
 *
 * @code
 * mat3_copy(PLAY_CGLM_MAT3_IDENTITY, mat); // C only
 *
 * // or
 * mat3 mat = PLAY_CGLM_MAT3_IDENTITY_INIT;
 * @endcode
 *
 * @param[in, out]  mat  destination
 */
PLAY_CGLM_INLINE
void
mat3_identity(mat3 mat)
{
    PLAY_CGLM_ALIGN_MAT mat3 t = PLAY_CGLM_MAT3_IDENTITY_INIT;
    mat3_copy(t, mat);
}

/*!
 * @brief make given matrix array's each element identity matrix
 *
 * @param[in, out]  mat   matrix array (must be aligned (16/32)
 *                        if alignment is not disabled)
 *
 * @param[in]       count count of matrices
 */
PLAY_CGLM_INLINE
void
mat3_identity_array(mat3 * __restrict mat, size_t count)
{
    PLAY_CGLM_ALIGN_MAT mat3 t = PLAY_CGLM_MAT3_IDENTITY_INIT;
    size_t i;

    for (i = 0; i < count; i++)
    {
        mat3_copy(t, mat[i]);
    }
}

/*!
 * @brief make given matrix zero.
 *
 * @param[in, out]  mat  matrix
 */
PLAY_CGLM_INLINE
void
mat3_zero(mat3 mat)
{
    PLAY_CGLM_ALIGN_MAT mat3 t = PLAY_CGLM_MAT3_ZERO_INIT;
    mat3_copy(t, mat);
}

/*!
 * @brief multiply m1 and m2 to dest
 *
 * m1, m2 and dest matrices can be same matrix, it is possible to write this:
 *
 * @code
 * mat3 m = PLAY_CGLM_MAT3_IDENTITY_INIT;
 * mat3_mul(m, m, m);
 * @endcode
 *
 * @param[in]  m1   left matrix
 * @param[in]  m2   right matrix
 * @param[out] dest destination matrix
 */
PLAY_CGLM_INLINE
void
mat3_mul(mat3 m1, mat3 m2, mat3 dest)
{
#if defined(__wasm__) && defined(__wasm_simd128__)
    mat3_mul_wasm(m1, m2, dest);
#elif defined( __SSE__ ) || defined( __SSE2__ )
    mat3_mul_sse2(m1, m2, dest);
#else
    float a00 = m1[0][0], a01 = m1[0][1], a02 = m1[0][2],
                                          a10 = m1[1][0], a11 = m1[1][1], a12 = m1[1][2],
                                                                          a20 = m1[2][0], a21 = m1[2][1], a22 = m1[2][2],

                                                                                                          b00 = m2[0][0], b01 = m2[0][1], b02 = m2[0][2],
                                                                                                                                          b10 = m2[1][0], b11 = m2[1][1], b12 = m2[1][2],
                                                                                                                                                                          b20 = m2[2][0], b21 = m2[2][1], b22 = m2[2][2];

    dest[0][0] = a00 * b00 + a10 * b01 + a20 * b02;
    dest[0][1] = a01 * b00 + a11 * b01 + a21 * b02;
    dest[0][2] = a02 * b00 + a12 * b01 + a22 * b02;
    dest[1][0] = a00 * b10 + a10 * b11 + a20 * b12;
    dest[1][1] = a01 * b10 + a11 * b11 + a21 * b12;
    dest[1][2] = a02 * b10 + a12 * b11 + a22 * b12;
    dest[2][0] = a00 * b20 + a10 * b21 + a20 * b22;
    dest[2][1] = a01 * b20 + a11 * b21 + a21 * b22;
    dest[2][2] = a02 * b20 + a12 * b21 + a22 * b22;
#endif
}

/*!
 * @brief transpose mat3 and store in dest
 *
 * source matrix will not be transposed unless dest is m
 *
 * @param[in]  m     matrix
 * @param[out] dest  result
 */
PLAY_CGLM_INLINE
void
mat3_transpose_to(mat3 m, mat3 dest)
{
    dest[0][0] = m[0][0];
    dest[0][1] = m[1][0];
    dest[0][2] = m[2][0];
    dest[1][0] = m[0][1];
    dest[1][1] = m[1][1];
    dest[1][2] = m[2][1];
    dest[2][0] = m[0][2];
    dest[2][1] = m[1][2];
    dest[2][2] = m[2][2];
}

/*!
 * @brief transpose mat3 and store result in same matrix
 *
 * @param[in, out] m source and dest
 */
PLAY_CGLM_INLINE
void
mat3_transpose(mat3 m)
{
    PLAY_CGLM_ALIGN_MAT mat3 tmp;

    tmp[0][1] = m[1][0];
    tmp[0][2] = m[2][0];
    tmp[1][0] = m[0][1];
    tmp[1][2] = m[2][1];
    tmp[2][0] = m[0][2];
    tmp[2][1] = m[1][2];

    m[0][1] = tmp[0][1];
    m[0][2] = tmp[0][2];
    m[1][0] = tmp[1][0];
    m[1][2] = tmp[1][2];
    m[2][0] = tmp[2][0];
    m[2][1] = tmp[2][1];
}

/*!
 * @brief multiply mat3 with vec3 (column vector) and store in dest vector
 *
 * @param[in]  m    mat3 (left)
 * @param[in]  v    vec3 (right, column vector)
 * @param[out] dest vec3 (result, column vector)
 */
PLAY_CGLM_INLINE
void
mat3_mulv(mat3 m, vec3 v, vec3 dest)
{
    vec3 res;
    res[0] = m[0][0] * v[0] + m[1][0] * v[1] + m[2][0] * v[2];
    res[1] = m[0][1] * v[0] + m[1][1] * v[1] + m[2][1] * v[2];
    res[2] = m[0][2] * v[0] + m[1][2] * v[1] + m[2][2] * v[2];
    vec3_copy(res, dest);
}

/*!
 * @brief trace of matrix
 *
 * sum of the elements on the main diagonal from upper left to the lower right
 *
 * @param[in]  m matrix
 */
PLAY_CGLM_INLINE
float
mat3_trace(mat3 m)
{
    return m[0][0] + m[1][1] + m[2][2];
}

/*!
 * @brief convert mat3 to quaternion
 *
 * @param[in]  m    rotation matrix
 * @param[out] dest destination quaternion
 */
PLAY_CGLM_INLINE
void
mat3_quat(mat3 m, versor dest)
{
    float trace, r, rinv;

    /* it seems using like m12 instead of m[1][2] causes extra instructions */

    trace = m[0][0] + m[1][1] + m[2][2];
    if (trace >= 0.0f)
    {
        r       = sqrtf(1.0f + trace);
        rinv    = 0.5f / r;

        dest[0] = rinv * (m[1][2] - m[2][1]);
        dest[1] = rinv * (m[2][0] - m[0][2]);
        dest[2] = rinv * (m[0][1] - m[1][0]);
        dest[3] = r    * 0.5f;
    }
    else if (m[0][0] >= m[1][1] && m[0][0] >= m[2][2])
    {
        r       = sqrtf(1.0f - m[1][1] - m[2][2] + m[0][0]);
        rinv    = 0.5f / r;

        dest[0] = r    * 0.5f;
        dest[1] = rinv * (m[0][1] + m[1][0]);
        dest[2] = rinv * (m[0][2] + m[2][0]);
        dest[3] = rinv * (m[1][2] - m[2][1]);
    }
    else if (m[1][1] >= m[2][2])
    {
        r       = sqrtf(1.0f - m[0][0] - m[2][2] + m[1][1]);
        rinv    = 0.5f / r;

        dest[0] = rinv * (m[0][1] + m[1][0]);
        dest[1] = r    * 0.5f;
        dest[2] = rinv * (m[1][2] + m[2][1]);
        dest[3] = rinv * (m[2][0] - m[0][2]);
    }
    else
    {
        r       = sqrtf(1.0f - m[0][0] - m[1][1] + m[2][2]);
        rinv    = 0.5f / r;

        dest[0] = rinv * (m[0][2] + m[2][0]);
        dest[1] = rinv * (m[1][2] + m[2][1]);
        dest[2] = r    * 0.5f;
        dest[3] = rinv * (m[0][1] - m[1][0]);
    }
}

/*!
 * @brief scale (multiply with scalar) matrix
 *
 * multiply matrix with scalar
 *
 * @param[in, out] m matrix
 * @param[in]      s scalar
 */
PLAY_CGLM_INLINE
void
mat3_scale(mat3 m, float s)
{
    m[0][0] *= s;
    m[0][1] *= s;
    m[0][2] *= s;
    m[1][0] *= s;
    m[1][1] *= s;
    m[1][2] *= s;
    m[2][0] *= s;
    m[2][1] *= s;
    m[2][2] *= s;
}

/*!
 * @brief mat3 determinant
 *
 * @param[in] mat matrix
 *
 * @return determinant
 */
PLAY_CGLM_INLINE
float
mat3_det(mat3 mat)
{
    float a = mat[0][0], b = mat[0][1], c = mat[0][2],
                                        d = mat[1][0], e = mat[1][1], f = mat[1][2],
                                                                      g = mat[2][0], h = mat[2][1], i = mat[2][2];

    return a * (e * i - h * f) - d * (b * i - h * c) + g * (b * f - e * c);
}

/*!
 * @brief inverse mat3 and store in dest
 *
 * @param[in]  mat  matrix
 * @param[out] dest inverse matrix
 */
PLAY_CGLM_INLINE
void
mat3_inv(mat3 mat, mat3 dest)
{
    float a = mat[0][0], b = mat[0][1], c = mat[0][2],
                                        d = mat[1][0], e = mat[1][1], f = mat[1][2],
                                                                      g = mat[2][0], h = mat[2][1], i = mat[2][2],

                                                                                                    c1  = e * i - f * h, c2 = d * i - g * f, c3 = d * h - g * e,
                                                                                                                                             idt = 1.0f / (a * c1 - b * c2 + c * c3), ndt = -idt;

    dest[0][0] = idt * c1;
    dest[0][1] = ndt * (b * i - h * c);
    dest[0][2] = idt * (b * f - e * c);
    dest[1][0] = ndt * c2;
    dest[1][1] = idt * (a * i - g * c);
    dest[1][2] = ndt * (a * f - d * c);
    dest[2][0] = idt * c3;
    dest[2][1] = ndt * (a * h - g * b);
    dest[2][2] = idt * (a * e - d * b);
}

/*!
 * @brief swap two matrix columns
 *
 * @param[in,out] mat  matrix
 * @param[in]     col1 col1
 * @param[in]     col2 col2
 */
PLAY_CGLM_INLINE
void
mat3_swap_col(mat3 mat, int col1, int col2)
{
    vec3 tmp;
    vec3_copy(mat[col1], tmp);
    vec3_copy(mat[col2], mat[col1]);
    vec3_copy(tmp, mat[col2]);
}

/*!
 * @brief swap two matrix rows
 *
 * @param[in,out] mat  matrix
 * @param[in]     row1 row1
 * @param[in]     row2 row2
 */
PLAY_CGLM_INLINE
void
mat3_swap_row(mat3 mat, int row1, int row2)
{
    vec3 tmp;
    tmp[0] = mat[0][row1];
    tmp[1] = mat[1][row1];
    tmp[2] = mat[2][row1];

    mat[0][row1] = mat[0][row2];
    mat[1][row1] = mat[1][row2];
    mat[2][row1] = mat[2][row2];

    mat[0][row2] = tmp[0];
    mat[1][row2] = tmp[1];
    mat[2][row2] = tmp[2];
}

/*!
 * @brief helper for  R (row vector) * M (matrix) * C (column vector)
 *
 * rmc stands for Row * Matrix * Column
 *
 * the result is scalar because R * M = Matrix1x3 (row vector),
 * then Matrix1x3 * Vec3 (column vector) = Matrix1x1 (Scalar)
 *
 * @param[in]  r   row vector or matrix1x3
 * @param[in]  m   matrix3x3
 * @param[in]  c   column vector or matrix3x1
 *
 * @return scalar value e.g. Matrix1x1
 */
PLAY_CGLM_INLINE
float
mat3_rmc(vec3 r, mat3 m, vec3 c)
{
    vec3 tmp;
    mat3_mulv(m, c, tmp);
    return vec3_dot(r, tmp);
}

/*!
 * @brief Create mat3 matrix from pointer
 *
 * @param[in]  src  pointer to an array of floats
 * @param[out] dest matrix
 */
PLAY_CGLM_INLINE
void
mat3_make(const float * __restrict src, mat3 dest)
{
    dest[0][0] = src[0];
    dest[0][1] = src[1];
    dest[0][2] = src[2];

    dest[1][0] = src[3];
    dest[1][1] = src[4];
    dest[1][2] = src[5];

    dest[2][0] = src[6];
    dest[2][1] = src[7];
    dest[2][2] = src[8];
}

/*!
 * @brief Create mat3 matrix from texture transform parameters
 *
 * @param[in]  sx   scale x
 * @param[in]  sy   scale y
 * @param[in]  rot  rotation in radians CCW/RH
 * @param[in]  tx   translate x
 * @param[in]  ty   translate y
 * @param[out] dest texture transform matrix
 */
PLAY_CGLM_INLINE
void
mat3_textrans(float sx, float sy, float rot, float tx, float ty, mat3 dest)
{
    float c, s;

    c = cosf(rot);
    s = sinf(rot);

    mat3_identity(dest);

    dest[0][0] =  c * sx;
    dest[0][1] = -s * sy;
    dest[1][0] =  s * sx;
    dest[1][1] =  c * sy;
    dest[2][0] =  tx;
    dest[2][1] =  ty;
}

#endif /* cmat3_h */

/*** End of inlined file: mat3.h ***/


/*** Start of inlined file: mat3x2.h ***/
/*
 Macros:
   PLAY_CGLM_MAT3X2_ZERO_INIT
   PLAY_CGLM_MAT3X2_ZERO

 Functions:
   PLAY_CGLM_INLINE void mat3x2_copy(mat3x2 src, mat3x2 dest);
   PLAY_CGLM_INLINE void mat3x2_zero(mat3x2 m);
   PLAY_CGLM_INLINE void mat3x2_make(const float * __restrict src, mat3x2 dest);
   PLAY_CGLM_INLINE void mat3x2_mul(mat3x2 m1, mat2x3 m2, mat2 dest);
   PLAY_CGLM_INLINE void mat3x2_mulv(mat3x2 m, vec3 v, vec2 dest);
   PLAY_CGLM_INLINE void mat3x2_transpose(mat3x2 src, mat2x3 dest);
   PLAY_CGLM_INLINE void mat3x2_scale(mat3x2 m, float s);
 */

#ifndef cmat3x2_h
#define cmat3x2_h

#define PLAY_CGLM_MAT3X2_ZERO_INIT {{0.0f, 0.0f}, {0.0f, 0.0f}, {0.0f, 0.0f}}

/* for C only */
#define PLAY_CGLM_MAT3X2_ZERO PLAY_CGLM_MAT3X2_ZERO_INIT

/*!
 * @brief Copy mat3x2 (src) to mat3x2 (dest).
 *
 * @param[in]  src  mat3x2 (left)
 * @param[out] dest destination (result, mat3x2)
 */
PLAY_CGLM_INLINE
void
mat3x2_copy(mat3x2 src, mat3x2 dest)
{
    vec2_copy(src[0], dest[0]);
    vec2_copy(src[1], dest[1]);
    vec2_copy(src[2], dest[2]);
}

/*!
 * @brief Zero out the mat3x2 (m).
 *
 * @param[in, out] mat3x2 (src, dest)
 */
PLAY_CGLM_INLINE
void
mat3x2_zero(mat3x2 m)
{
    PLAY_CGLM_ALIGN_MAT mat3x2 t = PLAY_CGLM_MAT3X2_ZERO_INIT;
    mat3x2_copy(t, m);
}

/*!
 * @brief Create mat3x2 (dest) from pointer (src).
 *
 * @param[in]  src  pointer to an array of floats (left)
 * @param[out] dest destination (result, mat3x2)
 */
PLAY_CGLM_INLINE
void
mat3x2_make(const float * __restrict src, mat3x2 dest)
{
    dest[0][0] = src[0];
    dest[0][1] = src[1];

    dest[1][0] = src[2];
    dest[1][1] = src[3];

    dest[2][0] = src[4];
    dest[2][1] = src[5];
}

/*!
 * @brief Multiply mat3x2 (m1) by mat2x3 (m2) and store in mat2 (dest).
 *
 * @code
 * mat3x2_mul(mat3x2, mat2x3, mat2);
 * @endcode
 *
 * @param[in]  m1   mat3x2 (left)
 * @param[in]  m2   mat2x3 (right)
 * @param[out] dest destination (result, mat2)
 */
PLAY_CGLM_INLINE
void
mat3x2_mul(mat3x2 m1, mat2x3 m2, mat2 dest)
{
    float a00 = m1[0][0], a01 = m1[0][1],
                          a10 = m1[1][0], a11 = m1[1][1],
                                          a20 = m1[2][0], a21 = m1[2][1],

                                                          b00 = m2[0][0], b01 = m2[0][1], b02 = m2[0][2],
                                                                                          b10 = m2[1][0], b11 = m2[1][1], b12 = m2[1][2];

    dest[0][0] = a00 * b00 + a10 * b01 + a20 * b02;
    dest[0][1] = a01 * b00 + a11 * b01 + a21 * b02;

    dest[1][0] = a00 * b10 + a10 * b11 + a20 * b12;
    dest[1][1] = a01 * b10 + a11 * b11 + a21 * b12;
}

/*!
 * @brief Multiply mat3x2 (m) by vec3 (v) and store in vec2 (dest).
 *
 * @param[in]  m    mat3x2 (left)
 * @param[in]  v    vec3 (right, column vector)
 * @param[out] dest destination (result, column vector)
 */
PLAY_CGLM_INLINE
void
mat3x2_mulv(mat3x2 m, vec3 v, vec2 dest)
{
    float v0 = v[0], v1 = v[1], v2 = v[2];

    dest[0] = m[0][0] * v0 + m[1][0] * v1 + m[2][0] * v2;
    dest[1] = m[0][1] * v0 + m[1][1] * v1 + m[2][1] * v2;
}

/*!
 * @brief Transpose mat3x2 (src) and store in mat2x3 (dest).
 *
 * @param[in]  src  mat3x2 (left)
 * @param[out] dest destination (result, mat2x3)
 */
PLAY_CGLM_INLINE
void
mat3x2_transpose(mat3x2 src, mat2x3 dest)
{
    dest[0][0] = src[0][0];
    dest[0][1] = src[1][0];
    dest[0][2] = src[2][0];
    dest[1][0] = src[0][1];
    dest[1][1] = src[1][1];
    dest[1][2] = src[2][1];
}

/*!
 * @brief Multiply mat3x2 (m) by scalar constant (s).
 *
 * @param[in, out] m (src, dest)
 * @param[in]      s float (scalar)
 */
PLAY_CGLM_INLINE
void
mat3x2_scale(mat3x2 m, float s)
{
    m[0][0] *= s;
    m[0][1] *= s;
    m[1][0] *= s;
    m[1][1] *= s;
    m[2][0] *= s;
    m[2][1] *= s;
}

#endif

/*** End of inlined file: mat3x2.h ***/


/*** Start of inlined file: mat3x4.h ***/
/*
 Macros:
   PLAY_CGLM_MAT3X4_ZERO_INIT
   PLAY_CGLM_MAT3X4_ZERO

 Functions:
   PLAY_CGLM_INLINE void mat3x4_copy(mat3x4 src, mat3x4 dest);
   PLAY_CGLM_INLINE void mat3x4_zero(mat3x4 m);
   PLAY_CGLM_INLINE void mat3x4_make(const float * __restrict src, mat3x4 dest);
   PLAY_CGLM_INLINE void mat3x4_mul(mat3x4 m1, mat4x3 m2, mat4 dest);
   PLAY_CGLM_INLINE void mat3x4_mulv(mat3x4 m, vec3 v, vec4 dest);
   PLAY_CGLM_INLINE void mat3x4_transpose(mat3x4 src, mat4x3 dest);
   PLAY_CGLM_INLINE void mat3x4_scale(mat3x4 m, float s);
 */

#ifndef cmat3x4_h
#define cmat3x4_h

#define PLAY_CGLM_MAT3X4_ZERO_INIT {{0.0f, 0.0f, 0.0f, 0.0f}, \
                              {0.0f, 0.0f, 0.0f, 0.0f}, \
                              {0.0f, 0.0f, 0.0f, 0.0f}}

/* for C only */
#define PLAY_CGLM_MAT3X4_ZERO PLAY_CGLM_MAT3X4_ZERO_INIT

/*!
 * @brief Copy mat3x4 (src) to mat3x4 (dest).
 *
 * @param[in]  src  mat3x4 (left)
 * @param[out] dest destination (result, mat3x4)
 */
PLAY_CGLM_INLINE
void
mat3x4_copy(mat3x4 src, mat3x4 dest)
{
    vec4_ucopy(src[0], dest[0]);
    vec4_ucopy(src[1], dest[1]);
    vec4_ucopy(src[2], dest[2]);
}

/*!
 * @brief Zero out the mat3x4 (m).
 *
 * @param[in, out] mat3x4 (src, dest)
 */
PLAY_CGLM_INLINE
void
mat3x4_zero(mat3x4 m)
{
    PLAY_CGLM_ALIGN_MAT mat3x4 t = PLAY_CGLM_MAT3X4_ZERO_INIT;
    mat3x4_copy(t, m);
}

/*!
 * @brief Create mat3x4 (dest) from pointer (src).
 *
 * @param[in]  src  pointer to an array of floats (left)
 * @param[out] dest destination (result, mat3x4)
 */
PLAY_CGLM_INLINE
void
mat3x4_make(const float * __restrict src, mat3x4 dest)
{
    dest[0][0] = src[0];
    dest[0][1] = src[1];
    dest[0][2] = src[2];
    dest[0][3] = src[3];

    dest[1][0] = src[4];
    dest[1][1] = src[5];
    dest[1][2] = src[6];
    dest[1][3] = src[7];

    dest[2][0] = src[8];
    dest[2][1] = src[9];
    dest[2][2] = src[10];
    dest[2][3] = src[11];
}

/*!
 * @brief Multiply mat3x4 (m1) by mat4x3 (m2) and store in mat4 (dest).
 *
 * @code
 * mat3x4_mul(mat3x4, mat4x3, mat4);
 * @endcode
 *
 * @param[in]  m1   mat3x4 (left)
 * @param[in]  m2   mat4x3 (right)
 * @param[out] dest destination (result, mat4)
 */
PLAY_CGLM_INLINE
void
mat3x4_mul(mat3x4 m1, mat4x3 m2, mat4 dest)
{
    float a00 = m1[0][0], a01 = m1[0][1], a02 = m1[0][2], a03 = m1[0][3],
                                                          a10 = m1[1][0], a11 = m1[1][1], a12 = m1[1][2], a13 = m1[1][3],
                                                                                                          a20 = m1[2][0], a21 = m1[2][1], a22 = m1[2][2], a23 = m1[2][3],

                                                                                                                                                          b00 = m2[0][0], b01 = m2[0][1], b02 = m2[0][2],
                                                                                                                                                                                          b10 = m2[1][0], b11 = m2[1][1], b12 = m2[1][2],
                                                                                                                                                                                                                          b20 = m2[2][0], b21 = m2[2][1], b22 = m2[2][2],
                                                                                                                                                                                                                                                          b30 = m2[3][0], b31 = m2[3][1], b32 = m2[3][2];

    dest[0][0] = a00 * b00 + a10 * b01 + a20 * b02;
    dest[0][1] = a01 * b00 + a11 * b01 + a21 * b02;
    dest[0][2] = a02 * b00 + a12 * b01 + a22 * b02;
    dest[0][3] = a03 * b00 + a13 * b01 + a23 * b02;

    dest[1][0] = a00 * b10 + a10 * b11 + a20 * b12;
    dest[1][1] = a01 * b10 + a11 * b11 + a21 * b12;
    dest[1][2] = a02 * b10 + a12 * b11 + a22 * b12;
    dest[1][3] = a03 * b10 + a13 * b11 + a23 * b12;

    dest[2][0] = a00 * b20 + a10 * b21 + a20 * b22;
    dest[2][1] = a01 * b20 + a11 * b21 + a21 * b22;
    dest[2][2] = a02 * b20 + a12 * b21 + a22 * b22;
    dest[2][3] = a03 * b20 + a13 * b21 + a23 * b22;

    dest[3][0] = a00 * b30 + a10 * b31 + a20 * b32;
    dest[3][1] = a01 * b30 + a11 * b31 + a21 * b32;
    dest[3][2] = a02 * b30 + a12 * b31 + a22 * b32;
    dest[3][3] = a03 * b30 + a13 * b31 + a23 * b32;
}

/*!
 * @brief Multiply mat3x4 (m) by vec3 (v) and store in vec4 (dest).
 *
 * @param[in]  m    mat3x4 (left)
 * @param[in]  v    vec3 (right, column vector)
 * @param[out] dest destination (result, column vector)
 */
PLAY_CGLM_INLINE
void
mat3x4_mulv(mat3x4 m, vec3 v, vec4 dest)
{
    float v0 = v[0], v1 = v[1], v2 = v[2];

    dest[0] = m[0][0] * v0 + m[1][0] * v1 + m[2][0] * v2;
    dest[1] = m[0][1] * v0 + m[1][1] * v1 + m[2][1] * v2;
    dest[2] = m[0][2] * v0 + m[1][2] * v1 + m[2][2] * v2;
    dest[3] = m[0][3] * v0 + m[1][3] * v1 + m[2][3] * v2;
}

/*!
 * @brief Transpose mat3x4 (src) and store in mat4x3 (dest).
 *
 * @param[in]  src  mat3x4 (left)
 * @param[out] dest destination (result, mat4x3)
 */
PLAY_CGLM_INLINE
void
mat3x4_transpose(mat3x4 src, mat4x3 dest)
{
    dest[0][0] = src[0][0];
    dest[0][1] = src[1][0];
    dest[0][2] = src[2][0];
    dest[1][0] = src[0][1];
    dest[1][1] = src[1][1];
    dest[1][2] = src[2][1];
    dest[2][0] = src[0][2];
    dest[2][1] = src[1][2];
    dest[2][2] = src[2][2];
    dest[3][0] = src[0][3];
    dest[3][1] = src[1][3];
    dest[3][2] = src[2][3];
}

/*!
 * @brief Multiply mat3x4 (m) by scalar constant (s).
 *
 * @param[in, out] m (src, dest)
 * @param[in]      s float (scalar)
 */
PLAY_CGLM_INLINE
void
mat3x4_scale(mat3x4 m, float s)
{
    m[0][0] *= s;
    m[0][1] *= s;
    m[0][2] *= s;
    m[0][3] *= s;
    m[1][0] *= s;
    m[1][1] *= s;
    m[1][2] *= s;
    m[1][3] *= s;
    m[2][0] *= s;
    m[2][1] *= s;
    m[2][2] *= s;
    m[2][3] *= s;
}

#endif

/*** End of inlined file: mat3x4.h ***/


/*** Start of inlined file: mat2.h ***/
/*
 Macros:
   PLAY_CGLM_MAT2_IDENTITY_INIT
   PLAY_CGLM_MAT2_ZERO_INIT
   PLAY_CGLM_MAT2_IDENTITY
   PLAY_CGLM_MAT2_ZERO

 Functions:
   PLAY_CGLM_INLINE void  mat2_make(float * restrict src, mat2 dest)
   PLAY_CGLM_INLINE void  mat2_copy(mat2 mat, mat2 dest)
   PLAY_CGLM_INLINE void  mat2_identity(mat2 m)
   PLAY_CGLM_INLINE void  mat2_identity_array(mat2 * restrict mats, size_t count)
   PLAY_CGLM_INLINE void  mat2_zero(mat2 m)
   PLAY_CGLM_INLINE void  mat2_mul(mat2 m1, mat2 m2, mat2 dest)
   PLAY_CGLM_INLINE void  mat2_mulv(mat2 m, vec2 v, vec2 dest)
   PLAY_CGLM_INLINE void  mat2_transpose_to(mat2 mat, mat2 dest)
   PLAY_CGLM_INLINE void  mat2_transpose(mat2 m)
   PLAY_CGLM_INLINE void  mat2_scale(mat2 m, float s)
   PLAY_CGLM_INLINE void  mat2_inv(mat2 mat, mat2 dest)
   PLAY_CGLM_INLINE void  mat2_swap_col(mat2 mat, int col1, int col2)
   PLAY_CGLM_INLINE void  mat2_swap_row(mat2 mat, int row1, int row2)
   PLAY_CGLM_INLINE float mat2_det(mat2 m)
   PLAY_CGLM_INLINE float mat2_trace(mat2 m)
   PLAY_CGLM_INLINE float mat2_rmc(vec2 r, mat2 m, vec2 c)
 */

#ifndef cmat2_h
#define cmat2_h

#ifdef PLAY_CGLM_SSE_FP

/*** Start of inlined file: mat2.h ***/
#ifndef cmat2_sse_h
#define cmat2_sse_h
#if defined( __SSE__ ) || defined( __SSE2__ )

PLAY_CGLM_INLINE
void
mat2_mul_sse2(mat2 m1, mat2 m2, mat2 dest)
{
    __m128 x0, x1, x2, x3, x4;

    x1 = glmm_load(m1[0]); /* d c b a */
    x2 = glmm_load(m2[0]); /* h g f e */

    x3 = glmm_shuff1(x2, 2, 2, 0, 0);
    x4 = glmm_shuff1(x2, 3, 3, 1, 1);
    x0 = _mm_movelh_ps(x1, x1);
    x2 = _mm_movehl_ps(x1, x1);

    /*
     dest[0][0] = a * e + c * f;
     dest[0][1] = b * e + d * f;
     dest[1][0] = a * g + c * h;
     dest[1][1] = b * g + d * h;
     */
    x0 = glmm_fmadd(x0, x3, _mm_mul_ps(x2, x4));

    glmm_store(dest[0], x0);
}

PLAY_CGLM_INLINE
void
mat2_transp_sse2(mat2 m, mat2 dest)
{
    /* d c b a */
    /* d b c a */
    glmm_store(dest[0], glmm_shuff1(glmm_load(m[0]), 3, 1, 2, 0));
}

#endif
#endif /* cmat2_sse_h */

/*** End of inlined file: mat2.h ***/


#endif

#ifdef PLAY_CGLM_NEON_FP

/*** Start of inlined file: mat2.h ***/
#ifndef cmat2_neon_h
#define cmat2_neon_h
#if defined(PLAY_CGLM_NEON_FP)

PLAY_CGLM_INLINE
void
mat2_mul_neon(mat2 m1, mat2 m2, mat2 dest)
{
    float32x4x2_t a1;
    glmm_128 x0,  x1, x2;
    float32x2_t   dc, ba;

    x1 = glmm_load(m1[0]); /* d c b a */
    x2 = glmm_load(m2[0]); /* h g f e */

    dc = vget_high_f32(x1);
    ba = vget_low_f32(x1);

    /* g g e e, h h f f */
    a1 = vtrnq_f32(x2, x2);

    /*
     dest[0][0] = a * e + c * f;
     dest[0][1] = b * e + d * f;
     dest[1][0] = a * g + c * h;
     dest[1][1] = b * g + d * h;
     */
    x0 = glmm_fmadd(vcombine_f32(ba, ba), a1.val[0],
                    vmulq_f32(vcombine_f32(dc, dc), a1.val[1]));

    glmm_store(dest[0], x0);
}

#endif
#endif /* cmat2_neon_h */

/*** End of inlined file: mat2.h ***/


#endif

#ifdef PLAY_CGLM_SIMD_WASM

/*** Start of inlined file: mat2.h ***/
#ifndef cmat2_wasm_h
#define cmat2_wasm_h
#if defined(__wasm__) && defined(__wasm_simd128__)

PLAY_CGLM_INLINE
void
mat2_mul_wasm(mat2 m1, mat2 m2, mat2 dest)
{
    glmm_128 x0, x1, x2, x3, x4;

    x1 = glmm_load(m1[0]); /* d c b a */
    x2 = glmm_load(m2[0]); /* h g f e */

    x3 = glmm_shuff1(x2, 2, 2, 0, 0);
    x4 = glmm_shuff1(x2, 3, 3, 1, 1);
    /* x0 = _mm_movelh_ps(x1, x1); */
    x0 = wasm_i32x4_shuffle(x1, x1, 0, 1, 4, 5);
    /* x2 = _mm_movehl_ps(x1, x1); */
    x2 = wasm_i32x4_shuffle(x1, x1, 6, 7, 2, 3);

    /*
     dest[0][0] = a * e + c * f;
     dest[0][1] = b * e + d * f;
     dest[1][0] = a * g + c * h;
     dest[1][1] = b * g + d * h;
     */
    x0 = glmm_fmadd(x0, x3, wasm_f32x4_mul(x2, x4));

    glmm_store(dest[0], x0);
}

PLAY_CGLM_INLINE
void
mat2_transp_wasm(mat2 m, mat2 dest)
{
    /* d c b a */
    /* d b c a */
    glmm_store(dest[0], glmm_shuff1(glmm_load(m[0]), 3, 1, 2, 0));
}

#endif
#endif /* cmat2_wasm_h */

/*** End of inlined file: mat2.h ***/


#endif

#define PLAY_CGLM_MAT2_IDENTITY_INIT  {{1.0f, 0.0f}, {0.0f, 1.0f}}
#define PLAY_CGLM_MAT2_ZERO_INIT      {{0.0f, 0.0f}, {0.0f, 0.0f}}

/* for C only */
#define PLAY_CGLM_MAT2_IDENTITY ((mat2)PLAY_CGLM_MAT2_IDENTITY_INIT)
#define PLAY_CGLM_MAT2_ZERO     ((mat2)PLAY_CGLM_MAT2_ZERO_INIT)

/*!
 * @brief Create mat2 (dest) from pointer (src).
 *
 * @param[in]  src  pointer to an array of floats (left)
 * @param[out] dest destination (result, mat2)
 */
PLAY_CGLM_INLINE
void
mat2_make(const float * __restrict src, mat2 dest)
{
    dest[0][0] = src[0];
    dest[0][1] = src[1];
    dest[1][0] = src[2];
    dest[1][1] = src[3];
}

/*!
 * @brief Copy mat2 (mat) to mat2 (dest).
 *
 * @param[in]  mat  mat2 (left, src)
 * @param[out] dest destination (result, mat2)
 */
PLAY_CGLM_INLINE
void
mat2_copy(mat2 mat, mat2 dest)
{
    vec4_ucopy(mat[0], dest[0]);
}

/*!
 * @brief Copy a mat2 identity to mat2 (m), or makes mat2 (m) an identity.
 *
 *        The same thing may be achieved with either of bellow methods,
 *        but it is more easy to do that with this func especially for members
 *        e.g. mat2_identity(aStruct->aMatrix);
 *
 * @code
 * mat2_copy(PLAY_CGLM_MAT2_IDENTITY, mat); // C only
 *
 * // or
 * mat2 mat = PLAY_CGLM_MAT2_IDENTITY_INIT;
 * @endcode
 *
 * @param[in, out] m mat2 (src, dest)
 */
PLAY_CGLM_INLINE
void
mat2_identity(mat2 m)
{
    PLAY_CGLM_ALIGN_MAT mat2 t = PLAY_CGLM_MAT2_IDENTITY_INIT;
    mat2_copy(t, m);
}

/*!
 * @brief Given an array of mat2’s (mats) make each matrix an identity matrix.
 *
 * @param[in, out] mats Array of mat2’s (must be aligned (16/32) if alignment is not disabled)
 * @param[in]      count Array size of mats or number of matrices
 */
PLAY_CGLM_INLINE
void
mat2_identity_array(mat2 * __restrict mats, size_t count)
{
    PLAY_CGLM_ALIGN_MAT mat2 t = PLAY_CGLM_MAT2_IDENTITY_INIT;
    size_t i;

    for (i = 0; i < count; i++)
    {
        mat2_copy(t, mats[i]);
    }
}

/*!
 * @brief Zero out the mat2 (m).
 *
 * @param[in, out] m mat2 (src, dest)
 */
PLAY_CGLM_INLINE
void
mat2_zero(mat2 m)
{
    PLAY_CGLM_ALIGN_MAT mat2 t = PLAY_CGLM_MAT2_ZERO_INIT;
    mat2_copy(t, m);
}

/*!
 * @brief Multiply mat2 (m1) by mat2 (m2) and store in mat2 (dest).
 *
 *        m1, m2 and dest matrices can be same matrix, it is possible to write this:
 *
 * @code
 * mat2 m = PLAY_CGLM_MAT2_IDENTITY_INIT;
 * mat2_mul(m, m, m);
 * @endcode
 *
 * @param[in]  m1   mat2 (left)
 * @param[in]  m2   mat2 (right)
 * @param[out] dest destination (result, mat2)
 */
PLAY_CGLM_INLINE
void
mat2_mul(mat2 m1, mat2 m2, mat2 dest)
{
#if defined(__wasm__) && defined(__wasm_simd128__)
    mat2_mul_wasm(m1, m2, dest);
#elif defined( __SSE__ ) || defined( __SSE2__ )
    mat2_mul_sse2(m1, m2, dest);
#elif defined(PLAY_CGLM_NEON_FP)
    mat2_mul_neon(m1, m2, dest);
#else
    float a00 = m1[0][0], a01 = m1[0][1],
                          a10 = m1[1][0], a11 = m1[1][1],
                                          b00 = m2[0][0], b01 = m2[0][1],
                                                          b10 = m2[1][0], b11 = m2[1][1];

    dest[0][0] = a00 * b00 + a10 * b01;
    dest[0][1] = a01 * b00 + a11 * b01;
    dest[1][0] = a00 * b10 + a10 * b11;
    dest[1][1] = a01 * b10 + a11 * b11;
#endif
}

/*!
 * @brief Multiply mat2 (m) by vec2 (v) and store in vec2 (dest).
 *
 * @param[in]  m    mat2 (left)
 * @param[in]  v    vec2 (right, column vector)
 * @param[out] dest destination (result, column vector)
 */
PLAY_CGLM_INLINE
void
mat2_mulv(mat2 m, vec2 v, vec2 dest)
{
    dest[0] = m[0][0] * v[0] + m[1][0] * v[1];
    dest[1] = m[0][1] * v[0] + m[1][1] * v[1];
}

/*!
 * @brief Transpose mat2 (mat) and store in mat2 (dest).
 *
 * @param[in]  mat  mat2 (left, src)
 * @param[out] dest destination (result, mat2)
 */
PLAY_CGLM_INLINE
void
mat2_transpose_to(mat2 mat, mat2 dest)
{
#if defined(__wasm__) && defined(__wasm_simd128__)
    mat2_transp_wasm(mat, dest);
#elif defined( __SSE__ ) || defined( __SSE2__ )
    mat2_transp_sse2(mat, dest);
#else
    dest[0][0] = mat[0][0];
    dest[0][1] = mat[1][0];
    dest[1][0] = mat[0][1];
    dest[1][1] = mat[1][1];
#endif
}

/*!
 * @brief Transpose mat2 (m) and store result in the same matrix.
 *
 * @param[in, out] m mat2 (src, dest)
 */
PLAY_CGLM_INLINE
void
mat2_transpose(mat2 m)
{
    float tmp;
    tmp     = m[0][1];
    m[0][1] = m[1][0];
    m[1][0] = tmp;
}

/*!
 * @brief Multiply mat2 (m) by scalar constant (s).
 *
 * @param[in, out] m mat2 (src, dest)
 * @param[in]      s float (scalar)
 */
PLAY_CGLM_INLINE
void
mat2_scale(mat2 m, float s)
{
#if defined(__wasm__) && defined(__wasm_simd128__)
    glmm_store(m[0], wasm_f32x4_mul(wasm_v128_load(m[0]),
                                    wasm_f32x4_splat(s)));
#elif defined( __SSE__ ) || defined( __SSE2__ )
    glmm_store(m[0], _mm_mul_ps(_mm_loadu_ps(m[0]), glmm_set1(s)));
#elif defined(PLAY_CGLM_NEON_FP)
    vst1q_f32(m[0], vmulq_f32(vld1q_f32(m[0]), vdupq_n_f32(s)));
#else
    m[0][0] = m[0][0] * s;
    m[0][1] = m[0][1] * s;
    m[1][0] = m[1][0] * s;
    m[1][1] = m[1][1] * s;
#endif
}

/*!
 * @brief Inverse mat2 (mat) and store in mat2 (dest).
 *
 * @param[in]  mat  mat2 (left, src)
 * @param[out] dest destination (result, inverse mat2)
 */
PLAY_CGLM_INLINE
void
mat2_inv(mat2 mat, mat2 dest)
{
    float det;
    float a = mat[0][0], b = mat[0][1],
                         c = mat[1][0], d = mat[1][1];

    det = 1.0f / (a * d - b * c);

    dest[0][0] =  d * det;
    dest[0][1] = -b * det;
    dest[1][0] = -c * det;
    dest[1][1] =  a * det;
}

/*!
 * @brief Swap two columns in mat2 (mat) and store in same matrix.
 *
 * @param[in, out] mat  mat2 (src, dest)
 * @param[in]      col1 Column 1 array index
 * @param[in]      col2 Column 2 array index
 */
PLAY_CGLM_INLINE
void
mat2_swap_col(mat2 mat, int col1, int col2)
{
    float a, b;

    a = mat[col1][0];
    b = mat[col1][1];

    mat[col1][0] = mat[col2][0];
    mat[col1][1] = mat[col2][1];

    mat[col2][0] = a;
    mat[col2][1] = b;
}

/*!
 * @brief Swap two rows in mat2 (mat) and store in same matrix.
 *
 * @param[in, out] mat  mat2 (src, dest)
 * @param[in]      row1 Row 1 array index
 * @param[in]      row2 Row 2 array index
 */
PLAY_CGLM_INLINE
void
mat2_swap_row(mat2 mat, int row1, int row2)
{
    float a, b;

    a = mat[0][row1];
    b = mat[1][row1];

    mat[0][row1] = mat[0][row2];
    mat[1][row1] = mat[1][row2];

    mat[0][row2] = a;
    mat[1][row2] = b;
}

/*!
 * @brief Returns mat2 determinant.
 *
 * @param[in] m mat2 (src)
 *
 * @return[out] mat2 determinant (float)
 */
PLAY_CGLM_INLINE
float
mat2_det(mat2 m)
{
    return m[0][0] * m[1][1] - m[1][0] * m[0][1];
}

/*!
 * @brief Returns trace of matrix. Which is:
 *
 *        The sum of the elements on the main diagonal from
 *        upper left corner to the bottom right corner.
 *
 * @param[in] m mat2 (src)
 *
 * @return[out] mat2 trace (float)
 */
PLAY_CGLM_INLINE
float
mat2_trace(mat2 m)
{
    return m[0][0] + m[1][1];
}

/*!
 * @brief Helper for  R (row vector) * M (matrix) * C (column vector)
 *
 *        rmc stands for Row * Matrix * Column
 *
 *        the result is scalar because M * C = ResC (1x2, column vector),
 *        then if you take the dot_product(R (2x1), ResC (1x2)) = scalar value.
 *
 * @param[in] r vec2 (2x1, row vector)
 * @param[in] m mat2 (2x2, matrix)
 * @param[in] c vec2 (1x2, column vector)
 *
 * @return[out] Scalar value (float, 1x1)
 */
PLAY_CGLM_INLINE
float
mat2_rmc(vec2 r, mat2 m, vec2 c)
{
    vec2 tmp;
    mat2_mulv(m, c, tmp);
    return vec2_dot(r, tmp);
}

#endif /* cmat2_h */

/*** End of inlined file: mat2.h ***/


/*** Start of inlined file: mat2x3.h ***/
/*
 Macros:
   PLAY_CGLM_MAT2X3_ZERO_INIT
   PLAY_CGLM_MAT2X3_ZERO

 Functions:
   PLAY_CGLM_INLINE void mat2x3_copy(mat2x3 src, mat2x3 dest);
   PLAY_CGLM_INLINE void mat2x3_zero(mat2x3 m);
   PLAY_CGLM_INLINE void mat2x3_make(const float * __restrict src, mat2x3 dest);
   PLAY_CGLM_INLINE void mat2x3_mul(mat2x3 m1, mat3x2 m2, mat3 dest);
   PLAY_CGLM_INLINE void mat2x3_mulv(mat2x3 m, vec2 v, vec3 dest);
   PLAY_CGLM_INLINE void mat2x3_transpose(mat2x3 src, mat3x2 dest);
   PLAY_CGLM_INLINE void mat2x3_scale(mat2x3 m, float s);
 */

#ifndef cmat2x3_h
#define cmat2x3_h

#define PLAY_CGLM_MAT2X3_ZERO_INIT {{0.0f, 0.0f, 0.0f}, {0.0f, 0.0f, 0.0f}}

/* for C only */
#define PLAY_CGLM_MAT2X3_ZERO PLAY_CGLM_MAT2X3_ZERO_INIT

/*!
 * @brief Copy mat2x3 (src) to mat2x3 (dest).
 *
 * @param[in]  src  mat2x3 (left)
 * @param[out] dest destination (result, mat2x3)
 */
PLAY_CGLM_INLINE
void
mat2x3_copy(mat2x3 src, mat2x3 dest)
{
    vec3_copy(src[0], dest[0]);
    vec3_copy(src[1], dest[1]);
}

/*!
 * @brief Zero out the mat2x3 (m).
 *
 * @param[in, out] mat2x3 (src, dest)
 */
PLAY_CGLM_INLINE
void
mat2x3_zero(mat2x3 m)
{
    PLAY_CGLM_ALIGN_MAT mat2x3 t = PLAY_CGLM_MAT2X3_ZERO_INIT;
    mat2x3_copy(t, m);
}

/*!
 * @brief Create mat2x3 (dest) from pointer (src).
 *
 * @param[in]  src  pointer to an array of floats (left)
 * @param[out] dest destination (result, mat2x3)
 */
PLAY_CGLM_INLINE
void
mat2x3_make(const float * __restrict src, mat2x3 dest)
{
    dest[0][0] = src[0];
    dest[0][1] = src[1];
    dest[0][2] = src[2];

    dest[1][0] = src[3];
    dest[1][1] = src[4];
    dest[1][2] = src[5];
}

/*!
 * @brief Multiply mat2x3 (m1) by mat3x2 (m2) and store in mat3 (dest).
 *
 * @code
 * mat2x3_mul(mat2x3, mat3x2, mat3);
 * @endcode
 *
 * @param[in]  m1   mat2x3 (left)
 * @param[in]  m2   mat3x2 (right)
 * @param[out] dest destination (result, mat3)
 */
PLAY_CGLM_INLINE
void
mat2x3_mul(mat2x3 m1, mat3x2 m2, mat3 dest)
{
    float a00 = m1[0][0], a01 = m1[0][1], a02 = m1[0][2],
                                          a10 = m1[1][0], a11 = m1[1][1], a12 = m1[1][2],

                                                                          b00 = m2[0][0], b01 = m2[0][1],
                                                                                          b10 = m2[1][0], b11 = m2[1][1],
                                                                                                          b20 = m2[2][0], b21 = m2[2][1];

    dest[0][0] = a00 * b00 + a10 * b01;
    dest[0][1] = a01 * b00 + a11 * b01;
    dest[0][2] = a02 * b00 + a12 * b01;

    dest[1][0] = a00 * b10 + a10 * b11;
    dest[1][1] = a01 * b10 + a11 * b11;
    dest[1][2] = a02 * b10 + a12 * b11;

    dest[2][0] = a00 * b20 + a10 * b21;
    dest[2][1] = a01 * b20 + a11 * b21;
    dest[2][2] = a02 * b20 + a12 * b21;
}

/*!
 * @brief Multiply mat2x3 (m) by vec2 (v) and store in vec3 (dest).
 *
 * @param[in]  m    mat2x3 (left)
 * @param[in]  v    vec2 (right, column vector)
 * @param[out] dest destination (result, column vector)
 */
PLAY_CGLM_INLINE
void
mat2x3_mulv(mat2x3 m, vec2 v, vec3 dest)
{
    float v0 = v[0], v1 = v[1];

    dest[0] = m[0][0] * v0 + m[1][0] * v1;
    dest[1] = m[0][1] * v0 + m[1][1] * v1;
    dest[2] = m[0][2] * v0 + m[1][2] * v1;
}

/*!
 * @brief Transpose mat2x3 (src) and store in mat3x2 (dest).
 *
 * @param[in]  src  mat2x3 (left)
 * @param[out] dest destination (result, mat3x2)
 */
PLAY_CGLM_INLINE
void
mat2x3_transpose(mat2x3 src, mat3x2 dest)
{
    dest[0][0] = src[0][0];
    dest[0][1] = src[1][0];
    dest[1][0] = src[0][1];
    dest[1][1] = src[1][1];
    dest[2][0] = src[0][2];
    dest[2][1] = src[1][2];
}

/*!
 * @brief Multiply mat2x3 (m) by scalar constant (s).
 *
 * @param[in, out] m (src, dest)
 * @param[in]      float (scalar)
 */
PLAY_CGLM_INLINE
void
mat2x3_scale(mat2x3 m, float s)
{
    m[0][0] *= s;
    m[0][1] *= s;
    m[0][2] *= s;
    m[1][0] *= s;
    m[1][1] *= s;
    m[1][2] *= s;
}

#endif

/*** End of inlined file: mat2x3.h ***/


/*** Start of inlined file: mat2x4.h ***/
/*
 Macros:
   PLAY_CGLM_MAT2X4_ZERO_INIT
   PLAY_CGLM_MAT2X4_ZERO

 Functions:
   PLAY_CGLM_INLINE void mat2x4_copy(mat2x4 src, mat2x4 dest);
   PLAY_CGLM_INLINE void mat2x4_zero(mat2x4 m);
   PLAY_CGLM_INLINE void mat2x4_make(const float * __restrict src, mat2x4 dest);
   PLAY_CGLM_INLINE void mat2x4_mul(mat2x4 m1, mat4x2 m2, mat4 dest);
   PLAY_CGLM_INLINE void mat2x4_mulv(mat2x4 m, vec2 v, vec4 dest);
   PLAY_CGLM_INLINE void mat2x4_transpose(mat2x4 src, mat4x2 dest);
   PLAY_CGLM_INLINE void mat2x4_scale(mat2x4 m, float s);
 */

#ifndef cmat2x4_h
#define cmat2x4_h

#define PLAY_CGLM_MAT2X4_ZERO_INIT {{0.0f, 0.0f, 0.0f, 0.0f}, {0.0f, 0.0f, 0.0f, 0.0f}}

/* for C only */
#define PLAY_CGLM_MAT2X4_ZERO PLAY_CGLM_MAT2X4_ZERO_INIT

/*!
 * @brief Copy mat2x4 (src) to mat2x4 (dest).
 *
 * @param[in]  src  mat2x4 (left)
 * @param[out] dest destination (result, mat2x4)
 */
PLAY_CGLM_INLINE
void
mat2x4_copy(mat2x4 src, mat2x4 dest)
{
    vec4_ucopy(src[0], dest[0]);
    vec4_ucopy(src[1], dest[1]);
}

/*!
 * @brief Zero out the mat2x4 (m).
 *
 * @param[in, out] mat2x4 (src, dest)
 */
PLAY_CGLM_INLINE
void
mat2x4_zero(mat2x4 m)
{
    PLAY_CGLM_ALIGN_MAT mat2x4 t = PLAY_CGLM_MAT2X4_ZERO_INIT;
    mat2x4_copy(t, m);
}

/*!
 * @brief Create mat2x4 (dest) from pointer (src).
 *
 * @param[in]  src  pointer to an array of floats (left)
 * @param[out] dest destination (result, mat2x4)
 */
PLAY_CGLM_INLINE
void
mat2x4_make(const float * __restrict src, mat2x4 dest)
{
    dest[0][0] = src[0];
    dest[0][1] = src[1];
    dest[0][2] = src[2];
    dest[0][3] = src[3];

    dest[1][0] = src[4];
    dest[1][1] = src[5];
    dest[1][2] = src[6];
    dest[1][3] = src[7];
}

/*!
 * @brief Multiply mat2x4 (m1) by mat4x2 (m2) and store in mat4 (dest).
 *
 * @code
 * mat2x4_mul(mat2x4, mat4x2, mat4);
 * @endcode
 *
 * @param[in]  m1   mat2x4 (left)
 * @param[in]  m2   mat4x2 (right)
 * @param[out] dest destination (result, mat4)
 */
PLAY_CGLM_INLINE
void
mat2x4_mul(mat2x4 m1, mat4x2 m2, mat4 dest)
{
    float a00 = m1[0][0], a01 = m1[0][1], a02 = m1[0][2], a03 = m1[0][3],
                                                          a10 = m1[1][0], a11 = m1[1][1], a12 = m1[1][2], a13 = m1[1][3],

                                                                                                          b00 = m2[0][0], b01 = m2[0][1],
                                                                                                                          b10 = m2[1][0], b11 = m2[1][1],
                                                                                                                                          b20 = m2[2][0], b21 = m2[2][1],
                                                                                                                                                          b30 = m2[3][0], b31 = m2[3][1];

    dest[0][0] = a00 * b00 + a10 * b01;
    dest[0][1] = a01 * b00 + a11 * b01;
    dest[0][2] = a02 * b00 + a12 * b01;
    dest[0][3] = a03 * b00 + a13 * b01;

    dest[1][0] = a00 * b10 + a10 * b11;
    dest[1][1] = a01 * b10 + a11 * b11;
    dest[1][2] = a02 * b10 + a12 * b11;
    dest[1][3] = a03 * b10 + a13 * b11;

    dest[2][0] = a00 * b20 + a10 * b21;
    dest[2][1] = a01 * b20 + a11 * b21;
    dest[2][2] = a02 * b20 + a12 * b21;
    dest[2][3] = a03 * b20 + a13 * b21;

    dest[3][0] = a00 * b30 + a10 * b31;
    dest[3][1] = a01 * b30 + a11 * b31;
    dest[3][2] = a02 * b30 + a12 * b31;
    dest[3][3] = a03 * b30 + a13 * b31;
}

/*!
 * @brief Multiply mat2x4 (m) by vec2 (v) and store in vec4 (dest).
 *
 * @param[in]  m    mat2x4 (left)
 * @param[in]  v    vec2 (right, column vector)
 * @param[out] dest destination (result, column vector)
 */
PLAY_CGLM_INLINE
void
mat2x4_mulv(mat2x4 m, vec2 v, vec4 dest)
{
    float v0 = v[0], v1 = v[1];

    dest[0] = m[0][0] * v0 + m[1][0] * v1;
    dest[1] = m[0][1] * v0 + m[1][1] * v1;
    dest[2] = m[0][2] * v0 + m[1][2] * v1;
    dest[3] = m[0][3] * v0 + m[1][3] * v1;
}

/*!
 * @brief Transpose mat2x4 (src) and store in mat4x2 (dest).
 *
 * @param[in]  src  mat2x4 (left)
 * @param[out] dest destination (result, mat4x2)
 */
PLAY_CGLM_INLINE
void
mat2x4_transpose(mat2x4 src, mat4x2 dest)
{
    dest[0][0] = src[0][0];
    dest[0][1] = src[1][0];
    dest[1][0] = src[0][1];
    dest[1][1] = src[1][1];
    dest[2][0] = src[0][2];
    dest[2][1] = src[1][2];
    dest[3][0] = src[0][3];
    dest[3][1] = src[1][3];
}

/*!
 * @brief Multiply mat2x4 (m) by scalar constant (s).
 *
 * @param[in, out] m (src, dest)
 * @param[in]      s float (scalar)
 */
PLAY_CGLM_INLINE
void
mat2x4_scale(mat2x4 m, float s)
{
    m[0][0] *= s;
    m[0][1] *= s;
    m[0][2] *= s;
    m[0][3] *= s;
    m[1][0] *= s;
    m[1][1] *= s;
    m[1][2] *= s;
    m[1][3] *= s;
}

#endif

/*** End of inlined file: mat2x4.h ***/


/*** Start of inlined file: affine.h ***/
/*
 Functions:
   PLAY_CGLM_INLINE void translate_to(mat4 m, vec3 v, mat4 dest);
   PLAY_CGLM_INLINE void translate(mat4 m, vec3 v);
   PLAY_CGLM_INLINE void translate_x(mat4 m, float to);
   PLAY_CGLM_INLINE void translate_y(mat4 m, float to);
   PLAY_CGLM_INLINE void translate_z(mat4 m, float to);
   PLAY_CGLM_INLINE void translate_make(mat4 m, vec3 v);
   PLAY_CGLM_INLINE void scale_to(mat4 m, vec3 v, mat4 dest);
   PLAY_CGLM_INLINE void scale_make(mat4 m, vec3 v);
   PLAY_CGLM_INLINE void scale(mat4 m, vec3 v);
   PLAY_CGLM_INLINE void scale_uni(mat4 m, float s);
   PLAY_CGLM_INLINE void rotate_x(mat4 m, float angle, mat4 dest);
   PLAY_CGLM_INLINE void rotate_y(mat4 m, float angle, mat4 dest);
   PLAY_CGLM_INLINE void rotate_z(mat4 m, float angle, mat4 dest);
   PLAY_CGLM_INLINE void rotate_make(mat4 m, float angle, vec3 axis);
   PLAY_CGLM_INLINE void rotate(mat4 m, float angle, vec3 axis);
   PLAY_CGLM_INLINE void rotate_at(mat4 m, vec3 pivot, float angle, vec3 axis);
   PLAY_CGLM_INLINE void rotate_atm(mat4 m, vec3 pivot, float angle, vec3 axis);
   PLAY_CGLM_INLINE void spin(mat4 m, float angle, vec3 axis);
   PLAY_CGLM_INLINE void decompose_scalev(mat4 m, vec3 s);
   PLAY_CGLM_INLINE bool uniscaled(mat4 m);
   PLAY_CGLM_INLINE void decompose_rs(mat4 m, mat4 r, vec3 s);
   PLAY_CGLM_INLINE void decompose(mat4 m, vec4 t, mat4 r, vec3 s);
 */





/*** Start of inlined file: affine-mat.h ***/
/*
 Functions:
   PLAY_CGLM_INLINE void mul(mat4 m1, mat4 m2, mat4 dest);
   PLAY_CGLM_INLINE void mul_rot(mat4 m1, mat4 m2, mat4 dest);
   PLAY_CGLM_INLINE void inv_tr(mat4 mat);
 */




#ifdef PLAY_CGLM_SSE_FP

/*** Start of inlined file: affine.h ***/


#if defined( __SSE__ ) || defined( __SSE2__ )

PLAY_CGLM_INLINE
void
mul_sse2(mat4 m1, mat4 m2, mat4 dest)
{
    /* D = R * L (Column-Major) */
    glmm_128 l, r0, r1, r2, r3, v0, v1, v2, v3;

    l  = glmm_load(m1[0]);
    r0 = glmm_load(m2[0]);
    r1 = glmm_load(m2[1]);
    r2 = glmm_load(m2[2]);
    r3 = glmm_load(m2[3]);

    v0 = _mm_mul_ps(glmm_splat_x(r0), l);
    v1 = _mm_mul_ps(glmm_splat_x(r1), l);
    v2 = _mm_mul_ps(glmm_splat_x(r2), l);
    v3 = _mm_mul_ps(glmm_splat_x(r3), l);

    l  = glmm_load(m1[1]);
    v0 = glmm_fmadd(glmm_splat_y(r0), l, v0);
    v1 = glmm_fmadd(glmm_splat_y(r1), l, v1);
    v2 = glmm_fmadd(glmm_splat_y(r2), l, v2);
    v3 = glmm_fmadd(glmm_splat_y(r3), l, v3);

    l  = glmm_load(m1[2]);
    v0 = glmm_fmadd(glmm_splat_z(r0), l, v0);
    v1 = glmm_fmadd(glmm_splat_z(r1), l, v1);
    v2 = glmm_fmadd(glmm_splat_z(r2), l, v2);
    v3 = glmm_fmadd(glmm_splat_z(r3), l, v3);

    l  = glmm_load(m1[3]);
    v3 = glmm_fmadd(glmm_splat_w(r3), l, v3);

    glmm_store(dest[0], v0);
    glmm_store(dest[1], v1);
    glmm_store(dest[2], v2);
    glmm_store(dest[3], v3);
}

PLAY_CGLM_INLINE
void
mul_rot_sse2(mat4 m1, mat4 m2, mat4 dest)
{
    /* D = R * L (Column-Major) */

    glmm_128 l, r0, r1, r2, v0, v1, v2;

    l  = glmm_load(m1[0]);
    r0 = glmm_load(m2[0]);
    r1 = glmm_load(m2[1]);
    r2 = glmm_load(m2[2]);

    v0 = _mm_mul_ps(glmm_splat_x(r0), l);
    v1 = _mm_mul_ps(glmm_splat_x(r1), l);
    v2 = _mm_mul_ps(glmm_splat_x(r2), l);

    l  = glmm_load(m1[1]);
    v0 = glmm_fmadd(glmm_splat_y(r0), l, v0);
    v1 = glmm_fmadd(glmm_splat_y(r1), l, v1);
    v2 = glmm_fmadd(glmm_splat_y(r2), l, v2);

    l  = glmm_load(m1[2]);
    v0 = glmm_fmadd(glmm_splat_z(r0), l, v0);
    v1 = glmm_fmadd(glmm_splat_z(r1), l, v1);
    v2 = glmm_fmadd(glmm_splat_z(r2), l, v2);

    glmm_store(dest[0], v0);
    glmm_store(dest[1], v1);
    glmm_store(dest[2], v2);
    glmm_store(dest[3], glmm_load(m1[3]));
}

PLAY_CGLM_INLINE
void
inv_tr_sse2(mat4 mat)
{
    __m128 r0, r1, r2, r3, x0, x1, x2, x3, x4, x5;

    r0 = glmm_load(mat[0]);
    r1 = glmm_load(mat[1]);
    r2 = glmm_load(mat[2]);
    r3 = glmm_load(mat[3]);
    x1 = _mm_set_ps(1.0f, 0.0f, 0.0f, 0.0f);

    _MM_TRANSPOSE4_PS(r0, r1, r2, x1);

    x2 = glmm_shuff1(r3, 0, 0, 0, 0);
    x3 = glmm_shuff1(r3, 1, 1, 1, 1);
    x4 = glmm_shuff1(r3, 2, 2, 2, 2);
    x5 = glmm_float32x4_SIGNMASK_NEG;

    x0 = glmm_fmadd(r0, x2, glmm_fmadd(r1, x3, _mm_mul_ps(r2, x4)));
    x0 = _mm_xor_ps(x0, x5);

    x0 = _mm_add_ps(x0, x1);

    glmm_store(mat[0], r0);
    glmm_store(mat[1], r1);
    glmm_store(mat[2], r2);
    glmm_store(mat[3], x0);
}

#endif


/*** End of inlined file: affine.h ***/


#endif

#ifdef PLAY_CGLM_AVX_FP

/*** Start of inlined file: affine.h ***/


#ifdef __AVX__

PLAY_CGLM_INLINE
void
mul_avx(mat4 m1, mat4 m2, mat4 dest)
{
    /* D = R * L (Column-Major) */

    __m256 y0, y1, y2, y3, y4, y5, y6, y7, y8, y9;

    y0 = glmm_load256(m2[0]); /* h g f e d c b a */
    y1 = glmm_load256(m2[2]); /* p o n m l k j i */

    y2 = glmm_load256(m1[0]); /* h g f e d c b a */
    y3 = glmm_load256(m1[2]); /* p o n m l k j i */

    /* 0x03: 0b00000011 */
    y4 = _mm256_permute2f128_ps(y2, y2, 0x03); /* d c b a h g f e */
    y5 = _mm256_permute2f128_ps(y3, y3, 0x03); /* l k j i p o n m */

    /* f f f f a a a a */
    /* h h h h c c c c */
    /* e e e e b b b b */
    /* g g g g d d d d */
    y6 = _mm256_permutevar_ps(y0, _mm256_set_epi32(1, 1, 1, 1, 0, 0, 0, 0));
    y7 = _mm256_permutevar_ps(y0, _mm256_set_epi32(3, 3, 3, 3, 2, 2, 2, 2));
    y8 = _mm256_permutevar_ps(y0, _mm256_set_epi32(0, 0, 0, 0, 1, 1, 1, 1));
    y9 = _mm256_permutevar_ps(y0, _mm256_set_epi32(2, 2, 2, 2, 3, 3, 3, 3));

    glmm_store256(dest[0],
                  _mm256_add_ps(_mm256_add_ps(_mm256_mul_ps(y2, y6),
                                _mm256_mul_ps(y3, y7)),
                                _mm256_add_ps(_mm256_mul_ps(y4, y8),
                                        _mm256_mul_ps(y5, y9))));

    /* n n n n i i i i */
    /* p p p p k k k k */
    /* m m m m j j j j */
    /* o o o o l l l l */
    y6 = _mm256_permutevar_ps(y1, _mm256_set_epi32(1, 1, 1, 1, 0, 0, 0, 0));
    y7 = _mm256_permutevar_ps(y1, _mm256_set_epi32(3, 3, 3, 3, 2, 2, 2, 2));
    y8 = _mm256_permutevar_ps(y1, _mm256_set_epi32(0, 0, 0, 0, 1, 1, 1, 1));
    y9 = _mm256_permutevar_ps(y1, _mm256_set_epi32(2, 2, 2, 2, 3, 3, 3, 3));

    glmm_store256(dest[2],
                  _mm256_add_ps(_mm256_add_ps(_mm256_mul_ps(y2, y6),
                                _mm256_mul_ps(y3, y7)),
                                _mm256_add_ps(_mm256_mul_ps(y4, y8),
                                        _mm256_mul_ps(y5, y9))));
}

#endif


/*** End of inlined file: affine.h ***/


#endif

#ifdef PLAY_CGLM_NEON_FP

/*** Start of inlined file: affine.h ***/


#if defined(PLAY_CGLM_NEON_FP)

PLAY_CGLM_INLINE
void
mul_neon(mat4 m1, mat4 m2, mat4 dest)
{
    /* D = R * L (Column-Major) */

    glmm_128 l, r0, r1, r2, r3, v0, v1, v2, v3;

    l  = glmm_load(m1[0]);
    r0 = glmm_load(m2[0]);
    r1 = glmm_load(m2[1]);
    r2 = glmm_load(m2[2]);
    r3 = glmm_load(m2[3]);

    v0 = vmulq_f32(glmm_splat_x(r0), l);
    v1 = vmulq_f32(glmm_splat_x(r1), l);
    v2 = vmulq_f32(glmm_splat_x(r2), l);
    v3 = vmulq_f32(glmm_splat_x(r3), l);

    l  = glmm_load(m1[1]);
    v0 = glmm_fmadd(glmm_splat_y(r0), l, v0);
    v1 = glmm_fmadd(glmm_splat_y(r1), l, v1);
    v2 = glmm_fmadd(glmm_splat_y(r2), l, v2);
    v3 = glmm_fmadd(glmm_splat_y(r3), l, v3);

    l  = glmm_load(m1[2]);
    v0 = glmm_fmadd(glmm_splat_z(r0), l, v0);
    v1 = glmm_fmadd(glmm_splat_z(r1), l, v1);
    v2 = glmm_fmadd(glmm_splat_z(r2), l, v2);
    v3 = glmm_fmadd(glmm_splat_z(r3), l, v3);

    v3 = glmm_fmadd(glmm_splat_w(r3), glmm_load(m1[3]), v3);

    glmm_store(dest[0], v0);
    glmm_store(dest[1], v1);
    glmm_store(dest[2], v2);
    glmm_store(dest[3], v3);
}

PLAY_CGLM_INLINE
void
mul_rot_neon(mat4 m1, mat4 m2, mat4 dest)
{
    /* D = R * L (Column-Major) */

    glmm_128 l, r0, r1, r2, v0, v1, v2;

    l  = glmm_load(m1[0]);
    r0 = glmm_load(m2[0]);
    r1 = glmm_load(m2[1]);
    r2 = glmm_load(m2[2]);

    v0 = vmulq_f32(glmm_splat_x(r0), l);
    v1 = vmulq_f32(glmm_splat_x(r1), l);
    v2 = vmulq_f32(glmm_splat_x(r2), l);

    l  = glmm_load(m1[1]);
    v0 = glmm_fmadd(glmm_splat_y(r0), l, v0);
    v1 = glmm_fmadd(glmm_splat_y(r1), l, v1);
    v2 = glmm_fmadd(glmm_splat_y(r2), l, v2);

    l  = glmm_load(m1[2]);
    v0 = glmm_fmadd(glmm_splat_z(r0), l, v0);
    v1 = glmm_fmadd(glmm_splat_z(r1), l, v1);
    v2 = glmm_fmadd(glmm_splat_z(r2), l, v2);

    glmm_store(dest[0], v0);
    glmm_store(dest[1], v1);
    glmm_store(dest[2], v2);
    glmm_store(dest[3], glmm_load(m1[3]));
}

PLAY_CGLM_INLINE
void
inv_tr_neon(mat4 mat)
{
    float32x4x4_t vmat;
    glmm_128      r0, r1, r2, x0;

    vmat = vld4q_f32(mat[0]);
    r0   = vmat.val[0];
    r1   = vmat.val[1];
    r2   = vmat.val[2];

    x0 = glmm_fmadd(r0, glmm_splat_w(r0),
                    glmm_fmadd(r1, glmm_splat_w(r1),
                               vmulq_f32(r2, glmm_splat_w(r2))));
    x0 = vnegq_f32(x0);

    glmm_store(mat[0], r0);
    glmm_store(mat[1], r1);
    glmm_store(mat[2], r2);
    glmm_store(mat[3], x0);

    mat[0][3] = 0.0f;
    mat[1][3] = 0.0f;
    mat[2][3] = 0.0f;
    mat[3][3] = 1.0f;

    /* TODO: ?
    zo   = vget_high_f32(r3);
    vst1_lane_f32(&mat[0][3], zo, 0);
    vst1_lane_f32(&mat[1][3], zo, 0);
    vst1_lane_f32(&mat[2][3], zo, 0);
    vst1_lane_f32(&mat[3][3], zo, 1);
    */
}

#endif


/*** End of inlined file: affine.h ***/


#endif

#ifdef PLAY_CGLM_SIMD_WASM

/*** Start of inlined file: affine.h ***/


#if defined(__wasm__) && defined(__wasm_simd128__)

PLAY_CGLM_INLINE
void
mul_wasm(mat4 m1, mat4 m2, mat4 dest)
{
    /* D = R * L (Column-Major) */
    glmm_128 l, r0, r1, r2, r3, v0, v1, v2, v3;

    l  = glmm_load(m1[0]);
    r0 = glmm_load(m2[0]);
    r1 = glmm_load(m2[1]);
    r2 = glmm_load(m2[2]);
    r3 = glmm_load(m2[3]);

    v0 = wasm_f32x4_mul(glmm_splat_x(r0), l);
    v1 = wasm_f32x4_mul(glmm_splat_x(r1), l);
    v2 = wasm_f32x4_mul(glmm_splat_x(r2), l);
    v3 = wasm_f32x4_mul(glmm_splat_x(r3), l);

    l  = glmm_load(m1[1]);
    v0 = glmm_fmadd(glmm_splat_y(r0), l, v0);
    v1 = glmm_fmadd(glmm_splat_y(r1), l, v1);
    v2 = glmm_fmadd(glmm_splat_y(r2), l, v2);
    v3 = glmm_fmadd(glmm_splat_y(r3), l, v3);

    l  = glmm_load(m1[2]);
    v0 = glmm_fmadd(glmm_splat_z(r0), l, v0);
    v1 = glmm_fmadd(glmm_splat_z(r1), l, v1);
    v2 = glmm_fmadd(glmm_splat_z(r2), l, v2);
    v3 = glmm_fmadd(glmm_splat_z(r3), l, v3);

    l  = glmm_load(m1[3]);
    v3 = glmm_fmadd(glmm_splat_w(r3), l, v3);

    glmm_store(dest[0], v0);
    glmm_store(dest[1], v1);
    glmm_store(dest[2], v2);
    glmm_store(dest[3], v3);
}

PLAY_CGLM_INLINE
void
mul_rot_wasm(mat4 m1, mat4 m2, mat4 dest)
{
    /* D = R * L (Column-Major) */

    glmm_128 l, r0, r1, r2, v0, v1, v2;

    l  = glmm_load(m1[0]);
    r0 = glmm_load(m2[0]);
    r1 = glmm_load(m2[1]);
    r2 = glmm_load(m2[2]);

    v0 = wasm_f32x4_mul(glmm_splat_x(r0), l);
    v1 = wasm_f32x4_mul(glmm_splat_x(r1), l);
    v2 = wasm_f32x4_mul(glmm_splat_x(r2), l);

    l  = glmm_load(m1[1]);
    v0 = glmm_fmadd(glmm_splat_y(r0), l, v0);
    v1 = glmm_fmadd(glmm_splat_y(r1), l, v1);
    v2 = glmm_fmadd(glmm_splat_y(r2), l, v2);

    l  = glmm_load(m1[2]);
    v0 = glmm_fmadd(glmm_splat_z(r0), l, v0);
    v1 = glmm_fmadd(glmm_splat_z(r1), l, v1);
    v2 = glmm_fmadd(glmm_splat_z(r2), l, v2);

    glmm_store(dest[0], v0);
    glmm_store(dest[1], v1);
    glmm_store(dest[2], v2);
    glmm_store(dest[3], glmm_load(m1[3]));
}

PLAY_CGLM_INLINE
void
inv_tr_wasm(mat4 mat)
{
    glmm_128 r0, r1, r2, r3, x0, x1, x2, x3, x4, x5;

    r0 = glmm_load(mat[0]);
    r1 = glmm_load(mat[1]);
    r2 = glmm_load(mat[2]);
    r3 = glmm_load(mat[3]);
    x1 = wasm_f32x4_const(0.0f, 0.0f, 0.0f, 1.0f);

    /* _MM_TRANSPOSE4_PS(r0, r1, r2, x1); */
    x2 = wasm_i32x4_shuffle(r0, r1, 0, 4, 1, 5);
    x3 = wasm_i32x4_shuffle(r0, r1, 2, 6, 3, 7);
    x4 = wasm_i32x4_shuffle(r2, x1, 0, 4, 1, 5);
    x5 = wasm_i32x4_shuffle(r2, x1, 2, 6, 3, 7);
    /* r0 = _mm_movelh_ps(x2, x4); */
    r0 = wasm_i32x4_shuffle(x2, x4, 0, 1, 4, 5);
    /* r1 = _mm_movehl_ps(x4, x2); */
    r1 = wasm_i32x4_shuffle(x4, x2, 6, 7, 2, 3);
    /* r2 = _mm_movelh_ps(x3, x5); */
    r2 = wasm_i32x4_shuffle(x3, x5, 0, 1, 4, 5);
    /* x1 = _mm_movehl_ps(x5, x3); */
    x1 = wasm_i32x4_shuffle(x5, x3, 6, 7, 2, 3);

    x2 = glmm_shuff1(r3, 0, 0, 0, 0);
    x3 = glmm_shuff1(r3, 1, 1, 1, 1);
    x4 = glmm_shuff1(r3, 2, 2, 2, 2);

    x0 = glmm_fmadd(r0, x2,
                    glmm_fmadd(r1, x3, wasm_f32x4_mul(r2, x4)));
    x0 = wasm_f32x4_neg(x0);

    x0 = wasm_f32x4_add(x0, x1);

    glmm_store(mat[0], r0);
    glmm_store(mat[1], r1);
    glmm_store(mat[2], r2);
    glmm_store(mat[3], x0);
}

#endif


/*** End of inlined file: affine.h ***/


#endif

/*!
 * @brief this is similar to mat4_mul but specialized to affine transform
 *
 * Matrix format should be:
 *   R  R  R  X
 *   R  R  R  Y
 *   R  R  R  Z
 *   0  0  0  W
 *
 * this reduces some multiplications. It should be faster than mat4_mul.
 * if you are not sure about matrix format then DON'T use this! use mat4_mul
 *
 * @param[in]   m1    affine matrix 1
 * @param[in]   m2    affine matrix 2
 * @param[out]  dest  result matrix
 */
PLAY_CGLM_INLINE
void
mul(mat4 m1, mat4 m2, mat4 dest)
{
#if defined(__wasm__) && defined(__wasm_simd128__)
    mul_wasm(m1, m2, dest);
#elif defined(__AVX__)
    mul_avx(m1, m2, dest);
#elif defined( __SSE__ ) || defined( __SSE2__ )
    mul_sse2(m1, m2, dest);
#elif defined(PLAY_CGLM_NEON_FP)
    mul_neon(m1, m2, dest);
#else
    float a00 = m1[0][0], a01 = m1[0][1], a02 = m1[0][2], a03 = m1[0][3],
                                                          a10 = m1[1][0], a11 = m1[1][1], a12 = m1[1][2], a13 = m1[1][3],
                                                                                                          a20 = m1[2][0], a21 = m1[2][1], a22 = m1[2][2], a23 = m1[2][3],
                                                                                                                                                          a30 = m1[3][0], a31 = m1[3][1], a32 = m1[3][2], a33 = m1[3][3],

                                                                                                                                                                                                          b00 = m2[0][0], b01 = m2[0][1], b02 = m2[0][2],
                                                                                                                                                                                                                                          b10 = m2[1][0], b11 = m2[1][1], b12 = m2[1][2],
                                                                                                                                                                                                                                                                          b20 = m2[2][0], b21 = m2[2][1], b22 = m2[2][2],
                                                                                                                                                                                                                                                                                                          b30 = m2[3][0], b31 = m2[3][1], b32 = m2[3][2], b33 = m2[3][3];

    dest[0][0] = a00 * b00 + a10 * b01 + a20 * b02;
    dest[0][1] = a01 * b00 + a11 * b01 + a21 * b02;
    dest[0][2] = a02 * b00 + a12 * b01 + a22 * b02;
    dest[0][3] = a03 * b00 + a13 * b01 + a23 * b02;

    dest[1][0] = a00 * b10 + a10 * b11 + a20 * b12;
    dest[1][1] = a01 * b10 + a11 * b11 + a21 * b12;
    dest[1][2] = a02 * b10 + a12 * b11 + a22 * b12;
    dest[1][3] = a03 * b10 + a13 * b11 + a23 * b12;

    dest[2][0] = a00 * b20 + a10 * b21 + a20 * b22;
    dest[2][1] = a01 * b20 + a11 * b21 + a21 * b22;
    dest[2][2] = a02 * b20 + a12 * b21 + a22 * b22;
    dest[2][3] = a03 * b20 + a13 * b21 + a23 * b22;

    dest[3][0] = a00 * b30 + a10 * b31 + a20 * b32 + a30 * b33;
    dest[3][1] = a01 * b30 + a11 * b31 + a21 * b32 + a31 * b33;
    dest[3][2] = a02 * b30 + a12 * b31 + a22 * b32 + a32 * b33;
    dest[3][3] = a03 * b30 + a13 * b31 + a23 * b32 + a33 * b33;
#endif
}

/*!
 * @brief this is similar to mat4_mul but specialized to affine transform
 *
 * Right Matrix format should be:
 *   R  R  R  0
 *   R  R  R  0
 *   R  R  R  0
 *   0  0  0  1
 *
 * this reduces some multiplications. It should be faster than mat4_mul.
 * if you are not sure about matrix format then DON'T use this! use mat4_mul
 *
 * @param[in]   m1    affine matrix 1
 * @param[in]   m2    affine matrix 2
 * @param[out]  dest  result matrix
 */
PLAY_CGLM_INLINE
void
mul_rot(mat4 m1, mat4 m2, mat4 dest)
{
#if defined(__wasm__) && defined(__wasm_simd128__)
    mul_rot_wasm(m1, m2, dest);
#elif defined( __SSE__ ) || defined( __SSE2__ )
    mul_rot_sse2(m1, m2, dest);
#elif defined(PLAY_CGLM_NEON_FP)
    mul_rot_neon(m1, m2, dest);
#else
    float a00 = m1[0][0], a01 = m1[0][1], a02 = m1[0][2], a03 = m1[0][3],
                                                          a10 = m1[1][0], a11 = m1[1][1], a12 = m1[1][2], a13 = m1[1][3],
                                                                                                          a20 = m1[2][0], a21 = m1[2][1], a22 = m1[2][2], a23 = m1[2][3],
                                                                                                                                                          a30 = m1[3][0], a31 = m1[3][1], a32 = m1[3][2], a33 = m1[3][3],

                                                                                                                                                                                                          b00 = m2[0][0], b01 = m2[0][1], b02 = m2[0][2],
                                                                                                                                                                                                                                          b10 = m2[1][0], b11 = m2[1][1], b12 = m2[1][2],
                                                                                                                                                                                                                                                                          b20 = m2[2][0], b21 = m2[2][1], b22 = m2[2][2];

    dest[0][0] = a00 * b00 + a10 * b01 + a20 * b02;
    dest[0][1] = a01 * b00 + a11 * b01 + a21 * b02;
    dest[0][2] = a02 * b00 + a12 * b01 + a22 * b02;
    dest[0][3] = a03 * b00 + a13 * b01 + a23 * b02;

    dest[1][0] = a00 * b10 + a10 * b11 + a20 * b12;
    dest[1][1] = a01 * b10 + a11 * b11 + a21 * b12;
    dest[1][2] = a02 * b10 + a12 * b11 + a22 * b12;
    dest[1][3] = a03 * b10 + a13 * b11 + a23 * b12;

    dest[2][0] = a00 * b20 + a10 * b21 + a20 * b22;
    dest[2][1] = a01 * b20 + a11 * b21 + a21 * b22;
    dest[2][2] = a02 * b20 + a12 * b21 + a22 * b22;
    dest[2][3] = a03 * b20 + a13 * b21 + a23 * b22;

    dest[3][0] = a30;
    dest[3][1] = a31;
    dest[3][2] = a32;
    dest[3][3] = a33;
#endif
}

/*!
 * @brief inverse orthonormal rotation + translation matrix (ridig-body)
 *
 * @code
 * X = | R  T |   X' = | R' -R'T |
 *     | 0  1 |        | 0     1 |
 * @endcode
 *
 * @param[in,out]  mat  matrix
 */
PLAY_CGLM_INLINE
void
inv_tr(mat4 mat)
{
#if defined(__wasm__) && defined(__wasm_simd128__)
    inv_tr_wasm(mat);
#elif defined( __SSE__ ) || defined( __SSE2__ )
    inv_tr_sse2(mat);
#elif defined(PLAY_CGLM_NEON_FP)
    inv_tr_neon(mat);
#else
    PLAY_CGLM_ALIGN_MAT mat3 r;
    PLAY_CGLM_ALIGN(8)  vec3 t;

    /* rotate */
    mat4_pick3t(mat, r);
    mat4_ins3(r, mat);

    /* translate */
    mat3_mulv(r, mat[3], t);
    vec3_negate(t);
    vec3_copy(t, mat[3]);
#endif
}



/*** End of inlined file: affine-mat.h ***/

/*!
 * @brief creates NEW translate transform matrix by v vector
 *
 * @param[out]  m  affine transform
 * @param[in]   v  translate vector [x, y, z]
 */
PLAY_CGLM_INLINE
void
translate_make(mat4 m, vec3 v)
{
    mat4_identity(m);
    vec3_copy(v, m[3]);
}

/*!
 * @brief scale existing transform matrix by v vector
 *        and store result in dest
 *
 * @param[in]  m    affine transform
 * @param[in]  v    scale vector [x, y, z]
 * @param[out] dest scaled matrix
 */
PLAY_CGLM_INLINE
void
scale_to(mat4 m, vec3 v, mat4 dest)
{
    vec4_scale(m[0], v[0], dest[0]);
    vec4_scale(m[1], v[1], dest[1]);
    vec4_scale(m[2], v[2], dest[2]);

    vec4_copy(m[3], dest[3]);
}

/*!
 * @brief creates NEW scale matrix by v vector
 *
 * @param[out]  m  affine transform
 * @param[in]   v  scale vector [x, y, z]
 */
PLAY_CGLM_INLINE
void
scale_make(mat4 m, vec3 v)
{
    mat4_identity(m);
    m[0][0] = v[0];
    m[1][1] = v[1];
    m[2][2] = v[2];
}

/*!
 * @brief scales existing transform matrix by v vector
 *        and stores result in same matrix
 *
 * @param[in, out]  m  affine transform
 * @param[in]       v  scale vector [x, y, z]
 */
PLAY_CGLM_INLINE
void
scale(mat4 m, vec3 v)
{
    scale_to(m, v, m);
}

/*!
 * @brief applies uniform scale to existing transform matrix v = [s, s, s]
 *        and stores result in same matrix
 *
 * @param[in, out]  m  affine transform
 * @param[in]       s  scale factor
 */
PLAY_CGLM_INLINE
void
scale_uni(mat4 m, float s)
{
    PLAY_CGLM_ALIGN(8) vec3 v = { s, s, s };
    scale_to(m, v, m);
}

/*!
 * @brief creates NEW rotation matrix by angle and axis
 *
 * axis will be normalized so you don't need to normalize it
 *
 * @param[out] m     affine transform
 * @param[in]  angle angle (radians)
 * @param[in]  axis  axis
 */
PLAY_CGLM_INLINE
void
rotate_make(mat4 m, float angle, vec3 axis)
{
    PLAY_CGLM_ALIGN(8) vec3 axisn, v, vs;
    float c;

    c = cosf(angle);

    vec3_normalize_to(axis, axisn);
    vec3_scale(axisn, 1.0f - c, v);
    vec3_scale(axisn, sinf(angle), vs);

    vec3_scale(axisn, v[0], m[0]);
    vec3_scale(axisn, v[1], m[1]);
    vec3_scale(axisn, v[2], m[2]);

    m[0][0] += c;
    m[1][0] -= vs[2];
    m[2][0] += vs[1];
    m[0][1] += vs[2];
    m[1][1] += c;
    m[2][1] -= vs[0];
    m[0][2] -= vs[1];
    m[1][2] += vs[0];
    m[2][2] += c;

    m[0][3] = m[1][3] = m[2][3] = m[3][0] = m[3][1] = m[3][2] = 0.0f;
    m[3][3] = 1.0f;
}

/*!
 * @brief decompose scale vector
 *
 * @param[in]  m  affine transform
 * @param[out] s  scale vector (Sx, Sy, Sz)
 */
PLAY_CGLM_INLINE
void
decompose_scalev(mat4 m, vec3 s)
{
    s[0] = vec3_norm(m[0]);
    s[1] = vec3_norm(m[1]);
    s[2] = vec3_norm(m[2]);
}

/*!
 * @brief returns true if matrix is uniform scaled. This is helpful for
 *        creating normal matrix.
 *
 * @param[in] m m
 *
 * @return boolean
 */
PLAY_CGLM_INLINE
bool
uniscaled(mat4 m)
{
    PLAY_CGLM_ALIGN(8) vec3 s;
    decompose_scalev(m, s);
    return vec3_eq_all(s);
}

/*!
 * @brief decompose rotation matrix (mat4) and scale vector [Sx, Sy, Sz]
 *        DON'T pass projected matrix here
 *
 * @param[in]  m affine transform
 * @param[out] r rotation matrix
 * @param[out] s scale matrix
 */
PLAY_CGLM_INLINE
void
decompose_rs(mat4 m, mat4 r, vec3 s)
{
    PLAY_CGLM_ALIGN(16) vec4 t = {0.0f, 0.0f, 0.0f, 1.0f};
    PLAY_CGLM_ALIGN(8)  vec3 v;

    vec4_copy(m[0], r[0]);
    vec4_copy(m[1], r[1]);
    vec4_copy(m[2], r[2]);
    vec4_copy(t,    r[3]);

    s[0] = vec3_norm(m[0]);
    s[1] = vec3_norm(m[1]);
    s[2] = vec3_norm(m[2]);

    vec4_scale(r[0], 1.0f/s[0], r[0]);
    vec4_scale(r[1], 1.0f/s[1], r[1]);
    vec4_scale(r[2], 1.0f/s[2], r[2]);

    /* Note from Apple Open Source (assume that the matrix is orthonormal):
       check for a coordinate system flip.  If the determinant
       is -1, then negate the matrix and the scaling factors. */
    vec3_cross(m[0], m[1], v);
    if (vec3_dot(v, m[2]) < 0.0f)
    {
        vec4_negate(r[0]);
        vec4_negate(r[1]);
        vec4_negate(r[2]);
        vec3_negate(s);
    }
}

/*!
 * @brief decompose affine transform, TODO: extract shear factors.
 *        DON'T pass projected matrix here
 *
 * @param[in]  m affine transform
 * @param[out] t translation vector
 * @param[out] r rotation matrix (mat4)
 * @param[out] s scaling vector [X, Y, Z]
 */
PLAY_CGLM_INLINE
void
decompose(mat4 m, vec4 t, mat4 r, vec3 s)
{
    vec4_copy(m[3], t);
    decompose_rs(m, r, s);
}


/*** Start of inlined file: affine-pre.h ***/



/*
 Functions:
   PLAY_CGLM_INLINE void translate_to(mat4 m, vec3 v, mat4 dest);
   PLAY_CGLM_INLINE void translate(mat4 m, vec3 v);
   PLAY_CGLM_INLINE void translate_x(mat4 m, float to);
   PLAY_CGLM_INLINE void translate_y(mat4 m, float to);
   PLAY_CGLM_INLINE void translate_z(mat4 m, float to);
   PLAY_CGLM_INLINE void rotate_x(mat4 m, float angle, mat4 dest);
   PLAY_CGLM_INLINE void rotate_y(mat4 m, float angle, mat4 dest);
   PLAY_CGLM_INLINE void rotate_z(mat4 m, float angle, mat4 dest);
   PLAY_CGLM_INLINE void rotate(mat4 m, float angle, vec3 axis);
   PLAY_CGLM_INLINE void rotate_at(mat4 m, vec3 pivot, float angle, vec3 axis);
   PLAY_CGLM_INLINE void rotate_atm(mat4 m, vec3 pivot, float angle, vec3 axis);
   PLAY_CGLM_INLINE void spin(mat4 m, float angle, vec3 axis);
 */

/*!
 * @brief translate existing transform matrix by v vector
 *        and stores result in same matrix
 *
 * @param[in, out]  m  affine transform
 * @param[in]       v  translate vector [x, y, z]
 */
PLAY_CGLM_INLINE
void
translate(mat4 m, vec3 v)
{
#if defined(PLAY_CGLM_SIMD)
    glmm_128 m0, m1, m2, m3;

    m0 = glmm_load(m[0]);
    m1 = glmm_load(m[1]);
    m2 = glmm_load(m[2]);
    m3 = glmm_load(m[3]);

    glmm_store(m[3],
               glmm_fmadd(m0, glmm_set1(v[0]),
                          glmm_fmadd(m1, glmm_set1(v[1]),
                                     glmm_fmadd(m2, glmm_set1(v[2]), m3))));
#else
    vec4_muladds(m[0], v[0], m[3]);
    vec4_muladds(m[1], v[1], m[3]);
    vec4_muladds(m[2], v[2], m[3]);
#endif
}

/*!
 * @brief translate existing transform matrix by v vector
 *        and store result in dest
 *
 * source matrix will remain same
 *
 * @param[in]  m    affine transform
 * @param[in]  v    translate vector [x, y, z]
 * @param[out] dest translated matrix
 */
PLAY_CGLM_INLINE
void
translate_to(mat4 m, vec3 v, mat4 dest)
{
    mat4_copy(m, dest);
    translate(dest, v);
}

/*!
 * @brief translate existing transform matrix by x factor
 *
 * @param[in, out]  m  affine transform
 * @param[in]       x  x factor
 */
PLAY_CGLM_INLINE
void
translate_x(mat4 m, float x)
{
#if defined(PLAY_CGLM_SIMD)
    glmm_store(m[3], glmm_fmadd(glmm_load(m[0]), glmm_set1(x), glmm_load(m[3])));
#else
    vec4 v1;
    vec4_scale(m[0], x, v1);
    vec4_add(v1, m[3], m[3]);
#endif
}

/*!
 * @brief translate existing transform matrix by y factor
 *
 * @param[in, out]  m  affine transform
 * @param[in]       y  y factor
 */
PLAY_CGLM_INLINE
void
translate_y(mat4 m, float y)
{
#if defined(PLAY_CGLM_SIMD)
    glmm_store(m[3], glmm_fmadd(glmm_load(m[1]), glmm_set1(y), glmm_load(m[3])));
#else
    vec4 v1;
    vec4_scale(m[1], y, v1);
    vec4_add(v1, m[3], m[3]);
#endif
}

/*!
 * @brief translate existing transform matrix by z factor
 *
 * @param[in, out]  m  affine transform
 * @param[in]       z  z factor
 */
PLAY_CGLM_INLINE
void
translate_z(mat4 m, float z)
{
#if defined(PLAY_CGLM_SIMD)
    glmm_store(m[3], glmm_fmadd(glmm_load(m[2]), glmm_set1(z), glmm_load(m[3])));
#else
    vec4 v1;
    vec4_scale(m[2], z, v1);
    vec4_add(v1, m[3], m[3]);
#endif
}

/*!
 * @brief rotate existing transform matrix around X axis by angle
 *        and store result in dest
 *
 * @param[in]   m      affine transform
 * @param[in]   angle  angle (radians)
 * @param[out]  dest   rotated matrix
 */
PLAY_CGLM_INLINE
void
rotate_x(mat4 m, float angle, mat4 dest)
{
    PLAY_CGLM_ALIGN_MAT mat4 t = PLAY_CGLM_MAT4_IDENTITY_INIT;
    float c, s;

    c = cosf(angle);
    s = sinf(angle);

    t[1][1] =  c;
    t[1][2] =  s;
    t[2][1] = -s;
    t[2][2] =  c;

    mul_rot(m, t, dest);
}

/*!
 * @brief rotate existing transform matrix around Y axis by angle
 *        and store result in dest
 *
 * @param[in]   m      affine transform
 * @param[in]   angle  angle (radians)
 * @param[out]  dest   rotated matrix
 */
PLAY_CGLM_INLINE
void
rotate_y(mat4 m, float angle, mat4 dest)
{
    PLAY_CGLM_ALIGN_MAT mat4 t = PLAY_CGLM_MAT4_IDENTITY_INIT;
    float c, s;

    c = cosf(angle);
    s = sinf(angle);

    t[0][0] =  c;
    t[0][2] = -s;
    t[2][0] =  s;
    t[2][2] =  c;

    mul_rot(m, t, dest);
}

/*!
 * @brief rotate existing transform matrix around Z axis by angle
 *        and store result in dest
 *
 * @param[in]   m      affine transform
 * @param[in]   angle  angle (radians)
 * @param[out]  dest   rotated matrix
 */
PLAY_CGLM_INLINE
void
rotate_z(mat4 m, float angle, mat4 dest)
{
    PLAY_CGLM_ALIGN_MAT mat4 t = PLAY_CGLM_MAT4_IDENTITY_INIT;
    float c, s;

    c = cosf(angle);
    s = sinf(angle);

    t[0][0] =  c;
    t[0][1] =  s;
    t[1][0] = -s;
    t[1][1] =  c;

    mul_rot(m, t, dest);
}

/*!
 * @brief rotate existing transform matrix
 *        around given axis by angle at ORIGIN (0,0,0)
 *
 *   **❗️IMPORTANT ❗️**
 *
 *   If you need to rotate object around itself e.g. center of object or at
 *   some point [of object] then `rotate_at()` would be better choice to do so.
 *
 *   Even if object's model transform is identity, rotation may not be around
 *   center of object if object does not lay out at ORIGIN perfectly.
 *
 *   Using `rotate_at()` with center of bounding shape ( AABB, Sphere ... )
 *   would be an easy option to rotate around object if object is not at origin.
 *
 *   One another option to rotate around itself at any point is `spin()`
 *   which is perfect if only rotating around model position is desired e.g. not
 *   specific point on model for instance center of geometry or center of mass,
 *   again if geometry is not perfectly centered at origin at identity transform,
 *   rotation may not be around geometry.
 *
 * @param[in, out]  m      affine transform
 * @param[in]       angle  angle (radians)
 * @param[in]       axis   axis
 */
PLAY_CGLM_INLINE
void
rotate(mat4 m, float angle, vec3 axis)
{
    PLAY_CGLM_ALIGN_MAT mat4 rot;
    rotate_make(rot, angle, axis);
    mul_rot(m, rot, m);
}

/*!
 * @brief rotate existing transform
 *        around given axis by angle at given pivot point (rotation center)
 *
 * @param[in, out]  m      affine transform
 * @param[in]       pivot  rotation center
 * @param[in]       angle  angle (radians)
 * @param[in]       axis   axis
 */
PLAY_CGLM_INLINE
void
rotate_at(mat4 m, vec3 pivot, float angle, vec3 axis)
{
    PLAY_CGLM_ALIGN(8) vec3 pivotInv;

    vec3_negate_to(pivot, pivotInv);

    translate(m, pivot);
    rotate(m, angle, axis);
    translate(m, pivotInv);
}

/*!
 * @brief creates NEW rotation matrix by angle and axis at given point
 *
 * this creates rotation matrix, it assumes you don't have a matrix
 *
 * this should work faster than rotate_at because it reduces
 * one translate.
 *
 * @param[out] m      affine transform
 * @param[in]  pivot  rotation center
 * @param[in]  angle  angle (radians)
 * @param[in]  axis   axis
 */
PLAY_CGLM_INLINE
void
rotate_atm(mat4 m, vec3 pivot, float angle, vec3 axis)
{
    PLAY_CGLM_ALIGN(8) vec3 pivotInv;

    vec3_negate_to(pivot, pivotInv);

    translate_make(m, pivot);
    rotate(m, angle, axis);
    translate(m, pivotInv);
}

/*!
 * @brief rotate existing transform matrix
 *        around given axis by angle around self (doesn't affected by position)
 *
 * @param[in, out]  m      affine transform
 * @param[in]       angle  angle (radians)
 * @param[in]       axis   axis
 */
PLAY_CGLM_INLINE
void
spin(mat4 m, float angle, vec3 axis)
{
    PLAY_CGLM_ALIGN_MAT mat4 rot;
    rotate_atm(rot, m[3], angle, axis);
    mat4_mul(m, rot, m);
}



/*** End of inlined file: affine-pre.h ***/


/*** Start of inlined file: affine-post.h ***/



/*
 Functions:
   PLAY_CGLM_INLINE void translated_to(mat4 m, vec3 v, mat4 dest);
   PLAY_CGLM_INLINE void translated(mat4 m, vec3 v);
   PLAY_CGLM_INLINE void translated_x(mat4 m, float to);
   PLAY_CGLM_INLINE void translated_y(mat4 m, float to);
   PLAY_CGLM_INLINE void translated_z(mat4 m, float to);
   PLAY_CGLM_INLINE void rotated_x(mat4 m, float angle, mat4 dest);
   PLAY_CGLM_INLINE void rotated_y(mat4 m, float angle, mat4 dest);
   PLAY_CGLM_INLINE void rotated_z(mat4 m, float angle, mat4 dest);
   PLAY_CGLM_INLINE void rotated(mat4 m, float angle, vec3 axis);
   PLAY_CGLM_INLINE void rotated_at(mat4 m, vec3 pivot, float angle, vec3 axis);
   PLAY_CGLM_INLINE void spinned(mat4 m, float angle, vec3 axis);
 */

/*!
 * @brief translate existing transform matrix by v vector
 *        and stores result in same matrix
 *
 *  this is POST transform, applies to existing transform as last transform
 *
 * @param[in, out]  m  affine transform
 * @param[in]       v  translate vector [x, y, z]
 */
PLAY_CGLM_INLINE
void
translated(mat4 m, vec3 v)
{
    vec3_add(m[3], v, m[3]);
}

/*!
 * @brief translate existing transform matrix by v vector
 *        and store result in dest
 *
 * source matrix will remain same
 *
 *  this is POST transform, applies to existing transform as last transform
 *
 * @param[in]  m    affine transform
 * @param[in]  v    translate vector [x, y, z]
 * @param[out] dest translated matrix
 */
PLAY_CGLM_INLINE
void
translated_to(mat4 m, vec3 v, mat4 dest)
{
    mat4_copy(m, dest);
    translated(dest, v);
}

/*!
 * @brief translate existing transform matrix by x factor
 *
 *  this is POST transform, applies to existing transform as last transform
 *
 * @param[in, out]  m  affine transform
 * @param[in]       x  x factor
 */
PLAY_CGLM_INLINE
void
translated_x(mat4 m, float x)
{
    m[3][0] += x;
}

/*!
 * @brief translate existing transform matrix by y factor
 *
 *  this is POST transform, applies to existing transform as last transform
 *
 * @param[in, out]  m  affine transform
 * @param[in]       y  y factor
 */
PLAY_CGLM_INLINE
void
translated_y(mat4 m, float y)
{
    m[3][1] += y;
}

/*!
 * @brief translate existing transform matrix by z factor
 *
 *  this is POST transform, applies to existing transform as last transform
 *
 * @param[in, out]  m  affine transform
 * @param[in]       z  z factor
 */
PLAY_CGLM_INLINE
void
translated_z(mat4 m, float z)
{
    m[3][2] += z;
}

/*!
 * @brief rotate existing transform matrix around X axis by angle
 *        and store result in dest
 *
 *  this is POST transform, applies to existing transform as last transform
 *
 * @param[in]   m      affine transform
 * @param[in]   angle  angle (radians)
 * @param[out]  dest   rotated matrix
 */
PLAY_CGLM_INLINE
void
rotated_x(mat4 m, float angle, mat4 dest)
{
    PLAY_CGLM_ALIGN_MAT mat4 t = PLAY_CGLM_MAT4_IDENTITY_INIT;
    float c, s;

    c = cosf(angle);
    s = sinf(angle);

    t[1][1] =  c;
    t[1][2] =  s;
    t[2][1] = -s;
    t[2][2] =  c;

    mul_rot(t, m, dest);
}

/*!
 * @brief rotate existing transform matrix around Y axis by angle
 *        and store result in dest
 *
 *  this is POST transform, applies to existing transform as last transform
 *
 * @param[in]   m      affine transform
 * @param[in]   angle  angle (radians)
 * @param[out]  dest   rotated matrix
 */
PLAY_CGLM_INLINE
void
rotated_y(mat4 m, float angle, mat4 dest)
{
    PLAY_CGLM_ALIGN_MAT mat4 t = PLAY_CGLM_MAT4_IDENTITY_INIT;
    float c, s;

    c = cosf(angle);
    s = sinf(angle);

    t[0][0] =  c;
    t[0][2] = -s;
    t[2][0] =  s;
    t[2][2] =  c;

    mul_rot(t, m, dest);
}

/*!
 * @brief rotate existing transform matrix around Z axis by angle
 *        and store result in dest
 *
 *  this is POST transform, applies to existing transform as last transform
 *
 * @param[in]   m      affine transform
 * @param[in]   angle  angle (radians)
 * @param[out]  dest   rotated matrix
 */
PLAY_CGLM_INLINE
void
rotated_z(mat4 m, float angle, mat4 dest)
{
    PLAY_CGLM_ALIGN_MAT mat4 t = PLAY_CGLM_MAT4_IDENTITY_INIT;
    float c, s;

    c = cosf(angle);
    s = sinf(angle);

    t[0][0] =  c;
    t[0][1] =  s;
    t[1][0] = -s;
    t[1][1] =  c;

    mul_rot(t, m, dest);
}

/*!
 * @brief rotate existing transform matrix around given axis by angle
 *
 *  this is POST transform, applies to existing transform as last transform
 *
 * @param[in, out]  m      affine transform
 * @param[in]       angle  angle (radians)
 * @param[in]       axis   axis
 */
PLAY_CGLM_INLINE
void
rotated(mat4 m, float angle, vec3 axis)
{
    PLAY_CGLM_ALIGN_MAT mat4 rot;
    rotate_make(rot, angle, axis);
    mul_rot(rot, m, m);
}

/*!
 * @brief rotate existing transform
 *        around given axis by angle at given pivot point (rotation center)
 *
 *  this is POST transform, applies to existing transform as last transform
 *
 * @param[in, out]  m      affine transform
 * @param[in]       pivot  rotation center
 * @param[in]       angle  angle (radians)
 * @param[in]       axis   axis
 */
PLAY_CGLM_INLINE
void
rotated_at(mat4 m, vec3 pivot, float angle, vec3 axis)
{
    PLAY_CGLM_ALIGN(8) vec3 pivotInv;

    vec3_negate_to(pivot, pivotInv);

    translated(m, pivot);
    rotated(m, angle, axis);
    translated(m, pivotInv);
}

/*!
 * @brief rotate existing transform matrix around given axis by angle around self (doesn't affected by position)
 *
 *  this is POST transform, applies to existing transform as last transform
 *
 * @param[in, out]  m      affine transform
 * @param[in]       angle  angle (radians)
 * @param[in]       axis   axis
 */
PLAY_CGLM_INLINE
void
spinned(mat4 m, float angle, vec3 axis)
{
    PLAY_CGLM_ALIGN_MAT mat4 rot;
    rotate_atm(rot, m[3], angle, axis);
    mat4_mul(rot, m, m);
}



/*** End of inlined file: affine-post.h ***/



/*** End of inlined file: affine.h ***/


/*** Start of inlined file: cam.h ***/
/*
 Functions:
   PLAY_CGLM_INLINE void  frustum(float left,   float right,
                                 float bottom, float top,
                                 float nearZ,  float farZ,
                                 mat4  dest)
   PLAY_CGLM_INLINE void  ortho(float left,   float right,
                               float bottom, float top,
                               float nearZ,  float farZ,
                               mat4  dest)
   PLAY_CGLM_INLINE void  ortho_aabb(vec3 box[2], mat4 dest)
   PLAY_CGLM_INLINE void  ortho_aabb_p(vec3 box[2],  float padding, mat4 dest)
   PLAY_CGLM_INLINE void  ortho_aabb_pz(vec3 box[2], float padding, mat4 dest)
   PLAY_CGLM_INLINE void  ortho_default(float aspect, mat4  dest)
   PLAY_CGLM_INLINE void  ortho_default_s(float aspect, float size, mat4 dest)
   PLAY_CGLM_INLINE void  perspective(float fovy,
                                     float aspect,
                                     float nearZ,
                                     float farZ,
                                     mat4  dest)
   PLAY_CGLM_INLINE void  perspective_default(float aspect, mat4 dest)
   PLAY_CGLM_INLINE void  perspective_resize(float aspect, mat4 proj)
   PLAY_CGLM_INLINE void  lookat(vec3 eye, vec3 center, vec3 up, mat4 dest)
   PLAY_CGLM_INLINE void  look(vec3 eye, vec3 dir, vec3 up, mat4 dest)
   PLAY_CGLM_INLINE void  look_anyup(vec3 eye, vec3 dir, mat4 dest)
   PLAY_CGLM_INLINE void  persp_decomp(mat4   proj,
                                      float *nearZ, float *farZ,
                                      float *top,   float *bottom,
                                      float *left,  float *right)
   PLAY_CGLM_INLINE void  persp_decompv(mat4 proj, float dest[6])
   PLAY_CGLM_INLINE void  persp_decomp_x(mat4 proj, float *left, float *right)
   PLAY_CGLM_INLINE void  persp_decomp_y(mat4 proj, float *top,  float *bottom)
   PLAY_CGLM_INLINE void  persp_decomp_z(mat4 proj, float *nearv, float *farv)
   PLAY_CGLM_INLINE void  persp_decomp_far(mat4 proj, float *farZ)
   PLAY_CGLM_INLINE void  persp_decomp_near(mat4 proj, float *nearZ)
   PLAY_CGLM_INLINE float persp_fovy(mat4 proj)
   PLAY_CGLM_INLINE float persp_aspect(mat4 proj)
   PLAY_CGLM_INLINE void  persp_sizes(mat4 proj, float fovy, vec4 dest)
 */





/*** Start of inlined file: plane.h ***/



/*
 Plane equation:  Ax + By + Cz + D = 0;

 It stored in vec4 as [A, B, C, D]. (A, B, C) is normal and D is distance
*/

/*
 Functions:
   PLAY_CGLM_INLINE void  plane_normalize(vec4 plane);
 */

/*!
 * @brief normalizes a plane
 *
 * @param[in, out] plane plane to normalize
 */
PLAY_CGLM_INLINE
void
plane_normalize(vec4 plane)
{
    float norm;

    if (PLAY_CGLM_UNLIKELY((norm = vec3_norm(plane)) < FLT_EPSILON))
    {
        vec4_zero(plane);
        return;
    }

    vec4_scale(plane, 1.0f / norm, plane);
}



/*** End of inlined file: plane.h ***/


/*** Start of inlined file: persp.h ***/
/*
 Functions:
   PLAY_CGLM_INLINE void  persp_decomp_far(mat4 proj, float *farZ)
   PLAY_CGLM_INLINE float persp_fovy(mat4 proj)
   PLAY_CGLM_INLINE float persp_aspect(mat4 proj)
   PLAY_CGLM_INLINE void  persp_sizes(mat4 proj, float fovy, vec4 dest)
 */




/*!
 * @brief returns field of view angle along the Y-axis (in radians)
 *
 * if you need to degrees, use deg to convert it or use this:
 * fovy_deg = deg(persp_fovy(projMatrix))
 *
 * @param[in] proj perspective projection matrix
 */
PLAY_CGLM_INLINE
float
persp_fovy(mat4 proj)
{
    return 2.0f * atanf(1.0f / proj[1][1]);
}

/*!
 * @brief returns aspect ratio of perspective projection
 *
 * @param[in] proj perspective projection matrix
 */
PLAY_CGLM_INLINE
float
persp_aspect(mat4 proj)
{
    return proj[1][1] / proj[0][0];
}



/*** End of inlined file: persp.h ***/


/*** Start of inlined file: ortho_lh_zo.h ***/
/*
 Functions:
   PLAY_CGLM_INLINE void ortho_lh_zo(float left,    float right,
                                    float bottom,  float top,
                                    float nearZ, float farZ,
                                    mat4  dest)
   PLAY_CGLM_INLINE void ortho_aabb_lh_zo(vec3 box[2], mat4 dest)
   PLAY_CGLM_INLINE void ortho_aabb_p_lh_zo(vec3 box[2],
                                           float padding,
                                           mat4 dest)
   PLAY_CGLM_INLINE void ortho_aabb_pz_lh_zo(vec3 box[2],
                                            float padding,
                                            mat4 dest)
   PLAY_CGLM_INLINE void ortho_default_lh_zo(float aspect,
                                            mat4 dest)
   PLAY_CGLM_INLINE void ortho_default_s_lh_zo(float aspect,
                                              float size,
                                              mat4 dest)
 */




/*!
 * @brief set up orthographic projection matrix with a left-hand coordinate
 *        system and a clip-space of [0, 1].
 *
 * @param[in]  left    viewport.left
 * @param[in]  right   viewport.right
 * @param[in]  bottom  viewport.bottom
 * @param[in]  top     viewport.top
 * @param[in]  nearZ   near clipping plane
 * @param[in]  farZ    far clipping plane
 * @param[out] dest    result matrix
 */
PLAY_CGLM_INLINE
void
ortho_lh_zo(float left,    float right,
            float bottom,  float top,
            float nearZ, float farZ,
            mat4  dest)
{
    float rl, tb, fn;

    mat4_zero(dest);

    rl = 1.0f / (right  - left);
    tb = 1.0f / (top    - bottom);
    fn =-1.0f / (farZ - nearZ);

    dest[0][0] = 2.0f * rl;
    dest[1][1] = 2.0f * tb;
    dest[2][2] =-fn;
    dest[3][0] =-(right  + left)    * rl;
    dest[3][1] =-(top    + bottom)  * tb;
    dest[3][2] = nearZ * fn;
    dest[3][3] = 1.0f;
}

/*!
 * @brief set up orthographic projection matrix using bounding box
 *        with a left-hand coordinate system and a clip-space of [0, 1].
 *
 * bounding box (AABB) must be in view space
 *
 * @param[in]  box   AABB
 * @param[out] dest  result matrix
 */
PLAY_CGLM_INLINE
void
ortho_aabb_lh_zo(vec3 box[2], mat4 dest)
{
    ortho_lh_zo(box[0][0],  box[1][0],
                box[0][1],  box[1][1],
                -box[1][2], -box[0][2],
                dest);
}

/*!
 * @brief set up orthographic projection matrix using bounding box
 *        with a left-hand coordinate system and a clip-space of [0, 1].
 *
 * bounding box (AABB) must be in view space
 *
 * @param[in]  box     AABB
 * @param[in]  padding padding
 * @param[out] dest    result matrix
 */
PLAY_CGLM_INLINE
void
ortho_aabb_p_lh_zo(vec3 box[2], float padding, mat4 dest)
{
    ortho_lh_zo(box[0][0] - padding,    box[1][0] + padding,
                box[0][1] - padding,    box[1][1] + padding,
                -(box[1][2] + padding), -(box[0][2] - padding),
                dest);
}

/*!
 * @brief set up orthographic projection matrix using bounding box
 *        with a left-hand coordinate system and a clip-space of [0, 1].
 *
 * bounding box (AABB) must be in view space
 *
 * @param[in]  box     AABB
 * @param[in]  padding padding for near and far
 * @param[out] dest    result matrix
 */
PLAY_CGLM_INLINE
void
ortho_aabb_pz_lh_zo(vec3 box[2], float padding, mat4 dest)
{
    ortho_lh_zo(box[0][0],              box[1][0],
                box[0][1],              box[1][1],
                -(box[1][2] + padding), -(box[0][2] - padding),
                dest);
}

/*!
 * @brief set up unit orthographic projection matrix
 *        with a left-hand coordinate system and a clip-space of [0, 1].
 *
 * @param[in]  aspect aspect ration ( width / height )
 * @param[out] dest   result matrix
 */
PLAY_CGLM_INLINE
void
ortho_default_lh_zo(float aspect, mat4 dest)
{
    if (aspect >= 1.0f)
    {
        ortho_lh_zo(-aspect, aspect, -1.0f, 1.0f, -100.0f, 100.0f, dest);
        return;
    }

    aspect = 1.0f / aspect;

    ortho_lh_zo(-1.0f, 1.0f, -aspect, aspect, -100.0f, 100.0f, dest);
}

/*!
 * @brief set up orthographic projection matrix with given CUBE size
 *        with a left-hand coordinate system and a clip-space of [0, 1].
 *
 * @param[in]  aspect aspect ratio ( width / height )
 * @param[in]  size   cube size
 * @param[out] dest   result matrix
 */
PLAY_CGLM_INLINE
void
ortho_default_s_lh_zo(float aspect, float size, mat4 dest)
{
    if (aspect >= 1.0f)
    {
        ortho_lh_zo(-size * aspect,
                    size * aspect,
                    -size,
                    size,
                    -size - 100.0f,
                    size + 100.0f,
                    dest);
        return;
    }

    ortho_lh_zo(-size,
                size,
                -size / aspect,
                size / aspect,
                -size - 100.0f,
                size + 100.0f,
                dest);
}



/*** End of inlined file: ortho_lh_zo.h ***/


/*** Start of inlined file: persp_lh_zo.h ***/
/*
 Functions:
   PLAY_CGLM_INLINE void frustum_lh_zo(float left,    float right,
                                      float bottom,  float top,
                                      float nearZ, float farZ,
                                      mat4  dest)
   PLAY_CGLM_INLINE void perspective_lh_zo(float fovy,
                                          float aspect,
                                          float nearZ,
                                          float farZ,
                                          mat4  dest)
   PLAY_CGLM_INLINE void perspective_default_lh_zo(float aspect, mat4 dest)
   PLAY_CGLM_INLINE void perspective_resize_lh_zo(float aspect, mat4 proj)
   PLAY_CGLM_INLINE void persp_move_far_lh_zo(mat4 proj,
                                             float deltaFar)
   PLAY_CGLM_INLINE void persp_decomp_lh_zo(mat4 proj,
                                           float * __restrict nearZ,
                                           float * __restrict farZ,
                                           float * __restrict top,
                                           float * __restrict bottom,
                                           float * __restrict left,
                                           float * __restrict right)
  PLAY_CGLM_INLINE void persp_decompv_lh_zo(mat4 proj,
                                           float dest[6])
  PLAY_CGLM_INLINE void persp_decomp_x_lh_zo(mat4 proj,
                                            float * __restrict left,
                                            float * __restrict right)
  PLAY_CGLM_INLINE void persp_decomp_y_lh_zo(mat4 proj,
                                            float * __restrict top,
                                            float * __restrict bottom)
  PLAY_CGLM_INLINE void persp_decomp_z_lh_zo(mat4 proj,
                                            float * __restrict nearZ,
                                            float * __restrict farZ)
  PLAY_CGLM_INLINE void persp_decomp_far_lh_zo(mat4 proj, float * __restrict farZ)
  PLAY_CGLM_INLINE void persp_decomp_near_lh_zo(mat4 proj, float * __restrict nearZ)
  PLAY_CGLM_INLINE void persp_sizes_lh_zo(mat4 proj, float fovy, vec4 dest)
 */




/*!
 * @brief set up perspective peprojection matrix with a left-hand coordinate
 *        system and a clip-space of [0, 1].
 *
 * @param[in]  left    viewport.left
 * @param[in]  right   viewport.right
 * @param[in]  bottom  viewport.bottom
 * @param[in]  top     viewport.top
 * @param[in]  nearZ   near clipping plane
 * @param[in]  farZ    far clipping plane
 * @param[out] dest    result matrix
 */
PLAY_CGLM_INLINE
void
frustum_lh_zo(float left,    float right,
              float bottom,  float top,
              float nearZ, float farZ,
              mat4  dest)
{
    float rl, tb, fn, nv;

    mat4_zero(dest);

    rl = 1.0f / (right  - left);
    tb = 1.0f / (top    - bottom);
    fn =-1.0f / (farZ - nearZ);
    nv = 2.0f * nearZ;

    dest[0][0] = nv * rl;
    dest[1][1] = nv * tb;
    dest[2][0] = (right  + left)    * rl;
    dest[2][1] = (top    + bottom)  * tb;
    dest[2][2] =-farZ * fn;
    dest[2][3] = 1.0f;
    dest[3][2] = farZ * nearZ * fn;
}

/*!
 * @brief set up perspective projection matrix with a left-hand coordinate
 * system and a clip-space of [0, 1].
 *
 * @param[in]  fovy    field of view angle
 * @param[in]  aspect  aspect ratio ( width / height )
 * @param[in]  nearZ   near clipping plane
 * @param[in]  farZ    far clipping planes
 * @param[out] dest    result matrix
 */
PLAY_CGLM_INLINE
void
perspective_lh_zo(float fovy,
                  float aspect,
                  float nearZ,
                  float farZ,
                  mat4  dest)
{
    float f, fn;

    mat4_zero(dest);

    f  = 1.0f / tanf(fovy * 0.5f);
    fn = 1.0f / (nearZ - farZ);

    dest[0][0] = f / aspect;
    dest[1][1] = f;
    dest[2][2] =-farZ * fn;
    dest[2][3] = 1.0f;
    dest[3][2] = nearZ * farZ * fn;
}

/*!
 * @brief extend perspective projection matrix's far distance with a
 *        left-hand coordinate system and a clip-space with depth values
 *        from zero to one.
 *
 * this function does not guarantee far >= near, be aware of that!
 *
 * @param[in, out] proj      projection matrix to extend
 * @param[in]      deltaFar  distance from existing far (negative to shink)
 */
PLAY_CGLM_INLINE
void
persp_move_far_lh_zo(mat4 proj, float deltaFar)
{
    float fn, farZ, nearZ, p22, p32;

    p22        = -proj[2][2];
    p32        = proj[3][2];

    nearZ    = p32 / p22;
    farZ     = p32 / (p22 + 1.0f) + deltaFar;
    fn         = 1.0f / (nearZ - farZ);

    proj[2][2] = -farZ * fn;
    proj[3][2] = nearZ * farZ * fn;
}

/*!
 * @brief set up perspective projection matrix with default near/far
 *        and angle values with a left-hand coordinate system and a
 *        clip-space of [0, 1].
 *
 * @param[in]  aspect aspect ratio ( width / height )
 * @param[out] dest   result matrix
 */
PLAY_CGLM_INLINE
void
perspective_default_lh_zo(float aspect, mat4 dest)
{
    perspective_lh_zo(PLAY_CGLM_PI_4f, aspect, 0.01f, 100.0f, dest);
}

/*!
 * @brief resize perspective matrix by aspect ratio ( width / height )
 *        this makes very easy to resize proj matrix when window /viewport
 *        reized
 *
 * @param[in]      aspect aspect ratio ( width / height )
 * @param[in, out] proj   perspective projection matrix
 */
PLAY_CGLM_INLINE
void
perspective_resize_lh_zo(float aspect, mat4 proj)
{
    if (proj[0][0] == 0.0f)
        return;

    proj[0][0] = proj[1][1] / aspect;
}

/*!
 * @brief decomposes frustum values of perspective projection
 *        with angle values with a left-hand coordinate system and a
 *        clip-space of [0, 1].
 *
 * @param[in]  proj    perspective projection matrix
 * @param[out] nearZ   near
 * @param[out] farZ    far
 * @param[out] top     top
 * @param[out] bottom  bottom
 * @param[out] left    left
 * @param[out] right   right
 */
PLAY_CGLM_INLINE
void
persp_decomp_lh_zo(mat4 proj,
                   float * __restrict nearZ, float * __restrict farZ,
                   float * __restrict top,     float * __restrict bottom,
                   float * __restrict left,    float * __restrict right)
{
    float m00, m11, m20, m21, m22, m32, n, f;
    float n_m11, n_m00;

    m00 = proj[0][0];
    m11 = proj[1][1];
    m20 = proj[2][0];
    m21 = proj[2][1];
    m22 =-proj[2][2];
    m32 = proj[3][2];

    n = m32 / m22;
    f = m32 / (m22 + 1.0f);

    n_m11 = n / m11;
    n_m00 = n / m00;

    *nearZ = n;
    *farZ  = f;
    *bottom  = n_m11 * (m21 - 1.0f);
    *top     = n_m11 * (m21 + 1.0f);
    *left    = n_m00 * (m20 - 1.0f);
    *right   = n_m00 * (m20 + 1.0f);
}

/*!
 * @brief decomposes frustum values of perspective projection
 *        with angle values with a left-hand coordinate system and a
 *        clip-space of [0, 1].
 *        this makes easy to get all values at once
 *
 * @param[in]  proj   perspective projection matrix
 * @param[out] dest   array
 */
PLAY_CGLM_INLINE
void
persp_decompv_lh_zo(mat4 proj, float dest[6])
{
    persp_decomp_lh_zo(proj, &dest[0], &dest[1], &dest[2],
                       &dest[3], &dest[4], &dest[5]);
}

/*!
 * @brief decomposes left and right values of perspective projection (ZO).
 *        x stands for x axis (left / right axis)
 *
 * @param[in]  proj  perspective projection matrix
 * @param[out] left  left
 * @param[out] right right
 */
PLAY_CGLM_INLINE
void
persp_decomp_x_lh_zo(mat4 proj,
                     float * __restrict left,
                     float * __restrict right)
{
    float nearZ, m20, m00;

    m00 = proj[0][0];
    m20 = proj[2][0];

    nearZ = proj[3][2] / (proj[3][3]);
    *left   = nearZ * (m20 - 1.0f) / m00;
    *right  = nearZ * (m20 + 1.0f) / m00;
}

/*!
 * @brief decomposes top and bottom values of perspective projection
 *        with angle values with a left-hand coordinate system and a
 *        clip-space of [0, 1].
 *        y stands for y axis (top / bottom axis)
 *
 * @param[in]  proj   perspective projection matrix
 * @param[out] top    top
 * @param[out] bottom bottom
 */
PLAY_CGLM_INLINE
void
persp_decomp_y_lh_zo(mat4 proj,
                     float * __restrict top,
                     float * __restrict bottom)
{
    float nearZ, m21, m11;

    m21 = proj[2][1];
    m11 = proj[1][1];

    nearZ = proj[3][2] / (proj[3][3]);
    *bottom = nearZ * (m21 - 1) / m11;
    *top    = nearZ * (m21 + 1) / m11;
}

/*!
 * @brief decomposes near and far values of perspective projection
 *        with angle values with a left-hand coordinate system and a
 *        clip-space of [0, 1].
 *        z stands for z axis (near / far axis)
 *
 * @param[in]  proj    perspective projection matrix
 * @param[out] nearZ   near
 * @param[out] farZ    far
 */
PLAY_CGLM_INLINE
void
persp_decomp_z_lh_zo(mat4 proj,
                     float * __restrict nearZ,
                     float * __restrict farZ)
{
    float m32, m22;

    m32 = proj[3][2];
    m22 = -proj[2][2];

    *nearZ = m32 / m22;
    *farZ  = m32 / (m22 + 1.0f);
}

/*!
 * @brief decomposes far value of perspective projection
 *        with angle values with a left-hand coordinate system and a
 *        clip-space of [0, 1].
 *
 * @param[in]  proj   perspective projection matrix
 * @param[out] farZ   far
 */
PLAY_CGLM_INLINE
void
persp_decomp_far_lh_zo(mat4 proj, float * __restrict farZ)
{
    *farZ = proj[3][2] / (-proj[2][2] + 1.0f);
}

/*!
 * @brief decomposes near value of perspective projection
 *        with angle values with a left-hand coordinate system and a
 *        clip-space of [0, 1].
 *
 * @param[in]  proj    perspective projection matrix
 * @param[out] nearZ   near
 */
PLAY_CGLM_INLINE
void
persp_decomp_near_lh_zo(mat4 proj, float * __restrict nearZ)
{
    *nearZ = proj[3][2] / -proj[2][2];
}

/*!
 * @brief returns sizes of near and far planes of perspective projection
 *        with a left-hand coordinate system and a
 *        clip-space of [0, 1].
 *
 * @param[in]  proj perspective projection matrix
 * @param[in]  fovy fovy (see brief)
 * @param[out] dest sizes order: [Wnear, Hnear, Wfar, Hfar]
 */
PLAY_CGLM_INLINE
void
persp_sizes_lh_zo(mat4 proj, float fovy, vec4 dest)
{
    float t, a, nearZ, farZ;

    t = 2.0f * tanf(fovy * 0.5f);
    a = persp_aspect(proj);

    persp_decomp_z_lh_zo(proj, &nearZ, &farZ);

    dest[1]  = t * nearZ;
    dest[3]  = t * farZ;
    dest[0]  = a * dest[1];
    dest[2]  = a * dest[3];
}

/*!
 * @brief returns field of view angle along the Y-axis (in radians)
 *        with a left-hand coordinate system and a clip-space of [0, 1].
 *
 * if you need to degrees, use deg to convert it or use this:
 * fovy_deg = deg(persp_fovy(projMatrix))
 *
 * @param[in] proj perspective projection matrix
 */
PLAY_CGLM_INLINE
float
persp_fovy_lh_zo(mat4 proj)
{
    return persp_fovy(proj);
}

/*!
 * @brief returns aspect ratio of perspective projection
 *        with a left-hand coordinate system and a clip-space of [0, 1].
 *
 * @param[in] proj perspective projection matrix
 */
PLAY_CGLM_INLINE
float
persp_aspect_lh_zo(mat4 proj)
{
    return persp_aspect(proj);
}



/*** End of inlined file: persp_lh_zo.h ***/


/*** Start of inlined file: ortho_lh_no.h ***/
/*
 Functions:
   PLAY_CGLM_INLINE void ortho_lh_no(float left,    float right,
                                    float bottom,  float top,
                                    float nearZ, float farZ,
                                    mat4  dest)
   PLAY_CGLM_INLINE void ortho_aabb_lh_no(vec3 box[2], mat4 dest)
   PLAY_CGLM_INLINE void ortho_aabb_p_lh_no(vec3 box[2],
                                           float padding,
                                           mat4 dest)
   PLAY_CGLM_INLINE void ortho_aabb_pz_lh_no(vec3 box[2],
                                            float padding,
                                            mat4 dest)
   PLAY_CGLM_INLINE void ortho_default_lh_no(float aspect,
                                            mat4 dest)
   PLAY_CGLM_INLINE void ortho_default_s_lh_no(float aspect,
                                              float size,
                                              mat4 dest)
 */




/*!
 * @brief set up orthographic projection matrix
 *        with a left-hand coordinate system and a
 *        clip-space of [-1, 1].
 *
 * @param[in]  left    viewport.left
 * @param[in]  right   viewport.right
 * @param[in]  bottom  viewport.bottom
 * @param[in]  top     viewport.top
 * @param[in]  nearZ   near clipping plane
 * @param[in]  farZ    far clipping plane
 * @param[out] dest    result matrix
 */
PLAY_CGLM_INLINE
void
ortho_lh_no(float left,    float right,
            float bottom,  float top,
            float nearZ, float farZ,
            mat4  dest)
{
    float rl, tb, fn;

    mat4_zero(dest);

    rl = 1.0f / (right  - left);
    tb = 1.0f / (top    - bottom);
    fn =-1.0f / (farZ - nearZ);

    dest[0][0] = 2.0f * rl;
    dest[1][1] = 2.0f * tb;
    dest[2][2] =-2.0f * fn;
    dest[3][0] =-(right  + left)    * rl;
    dest[3][1] =-(top    + bottom)  * tb;
    dest[3][2] = (farZ + nearZ) * fn;
    dest[3][3] = 1.0f;
}

/*!
 * @brief set up orthographic projection matrix using bounding box
 *        with a left-hand coordinate system and a
 *        clip-space of [-1, 1].
 *
 * bounding box (AABB) must be in view space
 *
 * @param[in]  box   AABB
 * @param[out] dest  result matrix
 */
PLAY_CGLM_INLINE
void
ortho_aabb_lh_no(vec3 box[2], mat4 dest)
{
    ortho_lh_no(box[0][0],  box[1][0],
                box[0][1],  box[1][1],
                -box[1][2], -box[0][2],
                dest);
}

/*!
 * @brief set up orthographic projection matrix using bounding box
 *        with a left-hand coordinate system and a
 *        clip-space of [-1, 1].
 *
 * bounding box (AABB) must be in view space
 *
 * @param[in]  box     AABB
 * @param[in]  padding padding
 * @param[out] dest    result matrix
 */
PLAY_CGLM_INLINE
void
ortho_aabb_p_lh_no(vec3 box[2], float padding, mat4 dest)
{
    ortho_lh_no(box[0][0] - padding,    box[1][0] + padding,
                box[0][1] - padding,    box[1][1] + padding,
                -(box[1][2] + padding), -(box[0][2] - padding),
                dest);
}

/*!
 * @brief set up orthographic projection matrix using bounding box
 *        with a left-hand coordinate system and a
 *        clip-space of [-1, 1].
 *
 * bounding box (AABB) must be in view space
 *
 * @param[in]  box     AABB
 * @param[in]  padding padding for near and far
 * @param[out] dest    result matrix
 */
PLAY_CGLM_INLINE
void
ortho_aabb_pz_lh_no(vec3 box[2], float padding, mat4 dest)
{
    ortho_lh_no(box[0][0],              box[1][0],
                box[0][1],              box[1][1],
                -(box[1][2] + padding), -(box[0][2] - padding),
                dest);
}

/*!
 * @brief set up unit orthographic projection matrix
 *        with a left-hand coordinate system and a
 *        clip-space of [-1, 1].
 *
 * @param[in]  aspect aspect ration ( width / height )
 * @param[out] dest   result matrix
 */
PLAY_CGLM_INLINE
void
ortho_default_lh_no(float aspect, mat4 dest)
{
    if (aspect >= 1.0f)
    {
        ortho_lh_no(-aspect, aspect, -1.0f, 1.0f, -100.0f, 100.0f, dest);
        return;
    }

    aspect = 1.0f / aspect;

    ortho_lh_no(-1.0f, 1.0f, -aspect, aspect, -100.0f, 100.0f, dest);
}

/*!
 * @brief set up orthographic projection matrix with given CUBE size
 *        with a left-hand coordinate system and a
 *        clip-space of [-1, 1].
 *
 * @param[in]  aspect aspect ratio ( width / height )
 * @param[in]  size   cube size
 * @param[out] dest   result matrix
 */
PLAY_CGLM_INLINE
void
ortho_default_s_lh_no(float aspect, float size, mat4 dest)
{
    if (aspect >= 1.0f)
    {
        ortho_lh_no(-size * aspect,
                    size * aspect,
                    -size,
                    size,
                    -size - 100.0f,
                    size + 100.0f,
                    dest);
        return;
    }

    ortho_lh_no(-size,
                size,
                -size / aspect,
                size / aspect,
                -size - 100.0f,
                size + 100.0f,
                dest);
}



/*** End of inlined file: ortho_lh_no.h ***/


/*** Start of inlined file: persp_lh_no.h ***/
/*
 Functions:
   PLAY_CGLM_INLINE void frustum_lh_no(float left,    float right,
                                       float bottom,  float top,
                                       float nearZ, float farZ,
                                       mat4  dest)
   PLAY_CGLM_INLINE void perspective_lh_no(float fovy,
                                          float aspect,
                                          float nearZ,
                                          float farZ,
                                          mat4  dest)
   PLAY_CGLM_INLINE void perspective_default_lh_no(float aspect, mat4 dest)
   PLAY_CGLM_INLINE void perspective_resize_lh_no(float aspect, mat4 proj)
   PLAY_CGLM_INLINE void persp_move_far_lh_no(mat4 proj,
                                             float deltaFar)
   PLAY_CGLM_INLINE void persp_decomp_lh_no(mat4 proj,
                                           float * __restrict nearZ,
                                           float * __restrict farZ,
                                           float * __restrict top,
                                           float * __restrict bottom,
                                           float * __restrict left,
                                           float * __restrict right)
  PLAY_CGLM_INLINE void persp_decompv_lh_no(mat4 proj,
                                           float dest[6])
  PLAY_CGLM_INLINE void persp_decomp_x_lh_no(mat4 proj,
                                            float * __restrict left,
                                            float * __restrict right)
  PLAY_CGLM_INLINE void persp_decomp_y_lh_no(mat4 proj,
                                            float * __restrict top,
                                            float * __restrict bottom)
  PLAY_CGLM_INLINE void persp_decomp_z_lh_no(mat4 proj,
                                            float * __restrict nearZ,
                                            float * __restrict farZ)
  PLAY_CGLM_INLINE void persp_decomp_far_lh_no(mat4 proj, float * __restrict farZ)
  PLAY_CGLM_INLINE void persp_decomp_near_lh_no(mat4 proj, float * __restrict nearZ)
  PLAY_CGLM_INLINE void persp_sizes_lh_no(mat4 proj, float fovy, vec4 dest)
 */




/*!
 * @brief set up perspective peprojection matrix
 *        with a left-hand coordinate system and a
 *        clip-space of [-1, 1].
 *
 * @param[in]  left    viewport.left
 * @param[in]  right   viewport.right
 * @param[in]  bottom  viewport.bottom
 * @param[in]  top     viewport.top
 * @param[in]  nearZ   near clipping plane
 * @param[in]  farZ    far clipping plane
 * @param[out] dest    result matrix
 */
PLAY_CGLM_INLINE
void
frustum_lh_no(float left,    float right,
              float bottom,  float top,
              float nearZ, float farZ,
              mat4  dest)
{
    float rl, tb, fn, nv;

    mat4_zero(dest);

    rl = 1.0f / (right  - left);
    tb = 1.0f / (top    - bottom);
    fn =-1.0f / (farZ - nearZ);
    nv = 2.0f * nearZ;

    dest[0][0] = nv * rl;
    dest[1][1] = nv * tb;
    dest[2][0] = (right  + left)    * rl;
    dest[2][1] = (top    + bottom)  * tb;
    dest[2][2] =-(farZ + nearZ) * fn;
    dest[2][3] = 1.0f;
    dest[3][2] = farZ * nv * fn;
}

/*!
 * @brief set up perspective projection matrix
 *        with a left-hand coordinate system and a
 *        clip-space of [-1, 1].
 *
 * @param[in]  fovy    field of view angle
 * @param[in]  aspect  aspect ratio ( width / height )
 * @param[in]  nearZ   near clipping plane
 * @param[in]  farZ    far clipping planes
 * @param[out] dest    result matrix
 */
PLAY_CGLM_INLINE
void
perspective_lh_no(float fovy,
                  float aspect,
                  float nearZ,
                  float farZ,
                  mat4  dest)
{
    float f, fn;

    mat4_zero(dest);

    f  = 1.0f / tanf(fovy * 0.5f);
    fn = 1.0f / (nearZ - farZ);

    dest[0][0] = f / aspect;
    dest[1][1] = f;
    dest[2][2] =-(nearZ + farZ) * fn;
    dest[2][3] = 1.0f;
    dest[3][2] = 2.0f * nearZ * farZ * fn;

}

/*!
 * @brief set up perspective projection matrix with default near/far
 *        and angle values with a left-hand coordinate system and a
 *        clip-space of [-1, 1].
 *
 * @param[in]  aspect aspect ratio ( width / height )
 * @param[out] dest   result matrix
 */
PLAY_CGLM_INLINE
void
perspective_default_lh_no(float aspect, mat4 dest)
{
    perspective_lh_no(PLAY_CGLM_PI_4f, aspect, 0.01f, 100.0f, dest);
}

/*!
 * @brief resize perspective matrix by aspect ratio ( width / height )
 *        this makes very easy to resize proj matrix when window /viewport
 *        resized with a left-hand coordinate system and a
 *        clip-space of [-1, 1].
 *
 * @param[in]      aspect aspect ratio ( width / height )
 * @param[in, out] proj   perspective projection matrix
 */
PLAY_CGLM_INLINE
void
perspective_resize_lh_no(float aspect, mat4 proj)
{
    if (proj[0][0] == 0.0f)
        return;

    proj[0][0] = proj[1][1] / aspect;
}

/*!
 * @brief extend perspective projection matrix's far distance
 *        with a left-hand coordinate system and a
 *        clip-space of [-1, 1].
 *
 * this function does not guarantee far >= near, be aware of that!
 *
 * @param[in, out] proj      projection matrix to extend
 * @param[in]      deltaFar  distance from existing far (negative to shink)
 */
PLAY_CGLM_INLINE
void
persp_move_far_lh_no(mat4 proj, float deltaFar)
{
    float fn, farZ, nearZ, p22, p32;

    p22        = -proj[2][2];
    p32        = proj[3][2];

    nearZ    = p32 / (p22 - 1.0f);
    farZ     = p32 / (p22 + 1.0f) + deltaFar;
    fn         = 1.0f / (nearZ - farZ);

    proj[2][2] = -(farZ + nearZ) * fn;
    proj[3][2] = 2.0f * nearZ * farZ * fn;
}

/*!
 * @brief decomposes frustum values of perspective projection
 *        with a left-hand coordinate system and a
 *        clip-space of [-1, 1].
 *
 * @param[in]  proj    perspective projection matrix
 * @param[out] nearZ   near
 * @param[out] farZ    far
 * @param[out] top     top
 * @param[out] bottom  bottom
 * @param[out] left    left
 * @param[out] right   right
 */
PLAY_CGLM_INLINE
void
persp_decomp_lh_no(mat4 proj,
                   float * __restrict nearZ, float * __restrict farZ,
                   float * __restrict top,   float * __restrict bottom,
                   float * __restrict left,  float * __restrict right)
{
    float m00, m11, m20, m21, m22, m32, n, f;
    float n_m11, n_m00;

    m00 = proj[0][0];
    m11 = proj[1][1];
    m20 = proj[2][0];
    m21 = proj[2][1];
    m22 =-proj[2][2];
    m32 = proj[3][2];

    n = m32 / (m22 - 1.0f);
    f = m32 / (m22 + 1.0f);

    n_m11 = n / m11;
    n_m00 = n / m00;

    *nearZ = n;
    *farZ  = f;
    *bottom  = n_m11 * (m21 - 1.0f);
    *top     = n_m11 * (m21 + 1.0f);
    *left    = n_m00 * (m20 - 1.0f);
    *right   = n_m00 * (m20 + 1.0f);
}

/*!
 * @brief decomposes frustum values of perspective projection
 *        with a left-hand coordinate system and a
 *        clip-space of [-1, 1].
 *        this makes easy to get all values at once
 *
 * @param[in]  proj   perspective projection matrix
 * @param[out] dest   array
 */
PLAY_CGLM_INLINE
void
persp_decompv_lh_no(mat4 proj, float dest[6])
{
    persp_decomp_lh_no(proj, &dest[0], &dest[1], &dest[2],
                       &dest[3], &dest[4], &dest[5]);
}

/*!
 * @brief decomposes left and right values of perspective projection
 *        with a left-hand coordinate system and a
 *        clip-space of [-1, 1].
 *        x stands for x axis (left / right axis)
 *
 * @param[in]  proj  perspective projection matrix
 * @param[out] left  left
 * @param[out] right right
 */
PLAY_CGLM_INLINE
void
persp_decomp_x_lh_no(mat4 proj,
                     float * __restrict left,
                     float * __restrict right)
{
    float nearZ, m20, m00, m22;

    m00 = proj[0][0];
    m20 = proj[2][0];
    m22 =-proj[2][2];

    nearZ = proj[3][2] / (m22 - 1.0f);
    *left   = nearZ * (m20 - 1.0f) / m00;
    *right  = nearZ * (m20 + 1.0f) / m00;
}

/*!
 * @brief decomposes top and bottom values of perspective projection
 *        with a left-hand coordinate system and a
 *        clip-space of [-1, 1].
 *        y stands for y axis (top / bottom axis)
 *
 * @param[in]  proj   perspective projection matrix
 * @param[out] top    top
 * @param[out] bottom bottom
 */
PLAY_CGLM_INLINE
void
persp_decomp_y_lh_no(mat4 proj,
                     float * __restrict top,
                     float * __restrict bottom)
{
    float nearZ, m21, m11, m22;

    m21 = proj[2][1];
    m11 = proj[1][1];
    m22 =-proj[2][2];

    nearZ = proj[3][2] / (m22 - 1.0f);
    *bottom = nearZ * (m21 - 1.0f) / m11;
    *top    = nearZ * (m21 + 1.0f) / m11;
}

/*!
 * @brief decomposes near and far values of perspective projection
 *        with a left-hand coordinate system and a
 *        clip-space of [-1, 1].
 *        z stands for z axis (near / far axis)
 *
 * @param[in]  proj    perspective projection matrix
 * @param[out] nearZ   near
 * @param[out] farZ    far
 */
PLAY_CGLM_INLINE
void
persp_decomp_z_lh_no(mat4 proj,
                     float * __restrict nearZ,
                     float * __restrict farZ)
{
    float m32, m22;

    m32 = proj[3][2];
    m22 =-proj[2][2];

    *nearZ = m32 / (m22 - 1.0f);
    *farZ  = m32 / (m22 + 1.0f);
}

/*!
 * @brief decomposes far value of perspective projection
 *        with a left-hand coordinate system and a
 *        clip-space of [-1, 1].
 *
 * @param[in]  proj   perspective projection matrix
 * @param[out] farZ   far
 */
PLAY_CGLM_INLINE
void
persp_decomp_far_lh_no(mat4 proj, float * __restrict farZ)
{
    *farZ = proj[3][2] / (-proj[2][2] + 1.0f);
}

/*!
 * @brief decomposes near value of perspective projection
 *        with a left-hand coordinate system and a
 *        clip-space of [-1, 1].
 *
 * @param[in]  proj    perspective projection matrix
 * @param[out] nearZ   near
 */
PLAY_CGLM_INLINE
void
persp_decomp_near_lh_no(mat4 proj, float * __restrict nearZ)
{
    *nearZ = proj[3][2] / (-proj[2][2] - 1.0f);
}

/*!
 * @brief returns sizes of near and far planes of perspective projection
 *        with a left-hand coordinate system and a
 *        clip-space of [-1, 1].
 *
 * @param[in]  proj perspective projection matrix
 * @param[in]  fovy fovy (see brief)
 * @param[out] dest sizes order: [Wnear, Hnear, Wfar, Hfar]
 */
PLAY_CGLM_INLINE
void
persp_sizes_lh_no(mat4 proj, float fovy, vec4 dest)
{
    float t, a, nearZ, farZ;

    t = 2.0f * tanf(fovy * 0.5f);
    a = persp_aspect(proj);

    persp_decomp_z_lh_no(proj, &nearZ, &farZ);

    dest[1]  = t * nearZ;
    dest[3]  = t * farZ;
    dest[0]  = a * dest[1];
    dest[2]  = a * dest[3];
}

/*!
 * @brief returns field of view angle along the Y-axis (in radians)
 *        with a left-hand coordinate system and a clip-space of [-1, 1].
 *
 * if you need to degrees, use deg to convert it or use this:
 * fovy_deg = deg(persp_fovy(projMatrix))
 *
 * @param[in] proj perspective projection matrix
 */
PLAY_CGLM_INLINE
float
persp_fovy_lh_no(mat4 proj)
{
    return persp_fovy(proj);
}

/*!
 * @brief returns aspect ratio of perspective projection
 *        with a left-hand coordinate system and a clip-space of [-1, 1].
 *
 * @param[in] proj perspective projection matrix
 */
PLAY_CGLM_INLINE
float
persp_aspect_lh_no(mat4 proj)
{
    return persp_aspect(proj);
}



/*** End of inlined file: persp_lh_no.h ***/


/*** Start of inlined file: ortho_rh_zo.h ***/
/*
 Functions:
   PLAY_CGLM_INLINE void ortho_rh_zo(float left,    float right,
                                    float bottom,  float top,
                                    float nearZ, float farZ,
                                    mat4  dest)
   PLAY_CGLM_INLINE void ortho_aabb_rh_zo(vec3 box[2], mat4 dest)
   PLAY_CGLM_INLINE void ortho_aabb_p_rh_zo(vec3 box[2],
                                           float padding,
                                           mat4 dest)
   PLAY_CGLM_INLINE void ortho_aabb_pz_rh_zo(vec3 box[2],
                                            float padding,
                                            mat4 dest)
   PLAY_CGLM_INLINE void ortho_default_rh_zo(float aspect,
                                            mat4 dest)
   PLAY_CGLM_INLINE void ortho_default_s_rh_zo(float aspect,
                                              float size,
                                              mat4 dest)
 */




/*!
 * @brief set up orthographic projection matrix with a right-hand coordinate
 *        system and a clip-space of [0, 1].
 *
 * @param[in]  left    viewport.left
 * @param[in]  right   viewport.right
 * @param[in]  bottom  viewport.bottom
 * @param[in]  top     viewport.top
 * @param[in]  nearZ   near clipping plane
 * @param[in]  farZ    far clipping plane
 * @param[out] dest    result matrix
 */
PLAY_CGLM_INLINE
void
ortho_rh_zo(float left,    float right,
            float bottom,  float top,
            float nearZ, float farZ,
            mat4  dest)
{
    float rl, tb, fn;

    mat4_zero(dest);

    rl = 1.0f / (right  - left);
    tb = 1.0f / (top    - bottom);
    fn =-1.0f / (farZ - nearZ);

    dest[0][0] = 2.0f * rl;
    dest[1][1] = 2.0f * tb;
    dest[2][2] = fn;
    dest[3][0] =-(right  + left)    * rl;
    dest[3][1] =-(top    + bottom)  * tb;
    dest[3][2] = nearZ * fn;
    dest[3][3] = 1.0f;
}

/*!
 * @brief set up orthographic projection matrix using bounding box
 *        with a right-hand coordinate system and a clip-space with depth
 *        values from zero to one.
 *
 * bounding box (AABB) must be in view space
 *
 * @param[in]  box   AABB
 * @param[out] dest  result matrix
 */
PLAY_CGLM_INLINE
void
ortho_aabb_rh_zo(vec3 box[2], mat4 dest)
{
    ortho_rh_zo(box[0][0],  box[1][0],
                box[0][1],  box[1][1],
                -box[1][2], -box[0][2],
                dest);
}

/*!
 * @brief set up orthographic projection matrix using bounding box
 *        with a right-hand coordinate system and a clip-space with depth
 *        values from zero to one.
 *
 * bounding box (AABB) must be in view space
 *
 * @param[in]  box     AABB
 * @param[in]  padding padding
 * @param[out] dest    result matrix
 */
PLAY_CGLM_INLINE
void
ortho_aabb_p_rh_zo(vec3 box[2], float padding, mat4 dest)
{
    ortho_rh_zo(box[0][0] - padding,    box[1][0] + padding,
                box[0][1] - padding,    box[1][1] + padding,
                -(box[1][2] + padding), -(box[0][2] - padding),
                dest);
}

/*!
 * @brief set up orthographic projection matrix using bounding box
 *        with a right-hand coordinate system and a clip-space with depth
 *        values from zero to one.
 *
 * bounding box (AABB) must be in view space
 *
 * @param[in]  box     AABB
 * @param[in]  padding padding for near and far
 * @param[out] dest    result matrix
 */
PLAY_CGLM_INLINE
void
ortho_aabb_pz_rh_zo(vec3 box[2], float padding, mat4 dest)
{
    ortho_rh_zo(box[0][0],              box[1][0],
                box[0][1],              box[1][1],
                -(box[1][2] + padding), -(box[0][2] - padding),
                dest);
}

/*!
 * @brief set up unit orthographic projection matrix with a right-hand
 *        coordinate system and a clip-space of [0, 1].
 *
 * @param[in]  aspect aspect ration ( width / height )
 * @param[out] dest   result matrix
 */
PLAY_CGLM_INLINE
void
ortho_default_rh_zo(float aspect, mat4 dest)
{
    if (aspect >= 1.0f)
    {
        ortho_rh_zo(-aspect, aspect, -1.0f, 1.0f, -100.0f, 100.0f, dest);
        return;
    }

    aspect = 1.0f / aspect;

    ortho_rh_zo(-1.0f, 1.0f, -aspect, aspect, -100.0f, 100.0f, dest);
}

/*!
 * @brief set up orthographic projection matrix with given CUBE size
 *        with a right-hand coordinate system and a clip-space with depth
 *        values from zero to one.
 *
 * @param[in]  aspect aspect ratio ( width / height )
 * @param[in]  size   cube size
 * @param[out] dest   result matrix
 */
PLAY_CGLM_INLINE
void
ortho_default_s_rh_zo(float aspect, float size, mat4 dest)
{
    if (aspect >= 1.0f)
    {
        ortho_rh_zo(-size * aspect,
                    size * aspect,
                    -size,
                    size,
                    -size - 100.0f,
                    size + 100.0f,
                    dest);
        return;
    }

    ortho_rh_zo(-size,
                size,
                -size / aspect,
                size / aspect,
                -size - 100.0f,
                size + 100.0f,
                dest);
}



/*** End of inlined file: ortho_rh_zo.h ***/


/*** Start of inlined file: persp_rh_zo.h ***/
/*
 Functions:
   PLAY_CGLM_INLINE void frustum_rh_zo(float left,    float right,
                                      float bottom,  float top,
                                      float nearZ, float farZ,
                                      mat4  dest)
   PLAY_CGLM_INLINE void perspective_rh_zo(float fovy,
                                          float aspect,
                                          float nearZ,
                                          float farZ,
                                          mat4  dest)
   PLAY_CGLM_INLINE void perspective_default_rh_zo(float aspect, mat4 dest)
   PLAY_CGLM_INLINE void perspective_resize_rh_zo(float aspect, mat4 proj)
   PLAY_CGLM_INLINE void persp_move_far_rh_zo(mat4 proj,
                                             float deltaFar)
   PLAY_CGLM_INLINE void persp_decomp_rh_zo(mat4 proj,
                                           float * __restrict nearZ,
                                           float * __restrict farZ,
                                           float * __restrict top,
                                           float * __restrict bottom,
                                           float * __restrict left,
                                           float * __restrict right)
  PLAY_CGLM_INLINE void persp_decompv_rh_zo(mat4 proj,
                                           float dest[6])
  PLAY_CGLM_INLINE void persp_decomp_x_rh_zo(mat4 proj,
                                            float * __restrict left,
                                            float * __restrict right)
  PLAY_CGLM_INLINE void persp_decomp_y_rh_zo(mat4 proj,
                                            float * __restrict top,
                                            float * __restrict bottom)
  PLAY_CGLM_INLINE void persp_decomp_z_rh_zo(mat4 proj,
                                            float * __restrict nearZ,
                                            float * __restrict farZ)
  PLAY_CGLM_INLINE void persp_decomp_far_rh_zo(mat4 proj, float * __restrict farZ)
  PLAY_CGLM_INLINE void persp_decomp_near_rh_zo(mat4 proj, float * __restrict nearZ)
  PLAY_CGLM_INLINE void persp_sizes_rh_zo(mat4 proj, float fovy, vec4 dest)
 */




/*!
 * @brief set up perspective peprojection matrix with a right-hand coordinate
 *        system and a clip-space of [0, 1].
 *
 * @param[in]  left    viewport.left
 * @param[in]  right   viewport.right
 * @param[in]  bottom  viewport.bottom
 * @param[in]  top     viewport.top
 * @param[in]  nearZ   near clipping plane
 * @param[in]  farZ    far clipping plane
 * @param[out] dest    result matrix
 */
PLAY_CGLM_INLINE
void
frustum_rh_zo(float left,    float right,
              float bottom,  float top,
              float nearZ, float farZ,
              mat4  dest)
{
    float rl, tb, fn, nv;

    mat4_zero(dest);

    rl = 1.0f / (right  - left);
    tb = 1.0f / (top    - bottom);
    fn =-1.0f / (farZ - nearZ);
    nv = 2.0f * nearZ;

    dest[0][0] = nv * rl;
    dest[1][1] = nv * tb;
    dest[2][0] = (right  + left)    * rl;
    dest[2][1] = (top    + bottom)  * tb;
    dest[2][2] = farZ * fn;
    dest[2][3] =-1.0f;
    dest[3][2] = farZ * nearZ * fn;
}

/*!
 * @brief set up perspective projection matrix with a right-hand coordinate
 *        system and a clip-space of [0, 1].
 *
 * @param[in]  fovy    field of view angle
 * @param[in]  aspect  aspect ratio ( width / height )
 * @param[in]  nearZ near clipping plane
 * @param[in]  farZ  far clipping planes
 * @param[out] dest    result matrix
 */
PLAY_CGLM_INLINE
void
perspective_rh_zo(float fovy,
                  float aspect,
                  float nearZ,
                  float farZ,
                  mat4  dest)
{
    float f, fn;

    mat4_zero(dest);

    f  = 1.0f / tanf(fovy * 0.5f);
    fn = 1.0f / (nearZ - farZ);

    dest[0][0] = f / aspect;
    dest[1][1] = f;
    dest[2][2] = farZ * fn;
    dest[2][3] =-1.0f;
    dest[3][2] = nearZ * farZ * fn;
}

/*!
 * @brief set up perspective projection matrix with default near/far
 *        and angle values with a right-hand coordinate system and a
 *        clip-space of [0, 1].
 *
 * @param[in]  aspect aspect ratio ( width / height )
 * @param[out] dest   result matrix
 */
PLAY_CGLM_INLINE
void
perspective_default_rh_zo(float aspect, mat4 dest)
{
    perspective_rh_zo(PLAY_CGLM_PI_4f, aspect, 0.01f, 100.0f, dest);
}

/*!
 * @brief resize perspective matrix by aspect ratio ( width / height )
 *        this makes very easy to resize proj matrix when window /viewport
 *        resized with a right-hand coordinate system and a clip-space of
 *        [0, 1].
 *
 * @param[in]      aspect aspect ratio ( width / height )
 * @param[in, out] proj   perspective projection matrix
 */
PLAY_CGLM_INLINE
void
perspective_resize_rh_zo(float aspect, mat4 proj)
{
    if (proj[0][0] == 0.0f)
        return;

    proj[0][0] = proj[1][1] / aspect;
}

/*!
 * @brief extend perspective projection matrix's far distance with a
 *        right-hand coordinate system and a clip-space of [0, 1].
 *
 * this function does not guarantee far >= near, be aware of that!
 *
 * @param[in, out] proj      projection matrix to extend
 * @param[in]      deltaFar  distance from existing far (negative to shink)
 */
PLAY_CGLM_INLINE
void
persp_move_far_rh_zo(mat4 proj, float deltaFar)
{
    float fn, farZ, nearZ, p22, p32;

    p22        = proj[2][2];
    p32        = proj[3][2];

    nearZ    = p32 / p22;
    farZ     = p32 / (p22 + 1.0f) + deltaFar;
    fn         = 1.0f / (nearZ - farZ);

    proj[2][2] = farZ * fn;
    proj[3][2] = nearZ * farZ * fn;
}

/*!
 * @brief decomposes frustum values of perspective projection
 *        with angle values with a right-hand coordinate system and a
 *        clip-space of [0, 1].
 *
 * @param[in]  proj    perspective projection matrix
 * @param[out] nearZ   near
 * @param[out] farZ    far
 * @param[out] top     top
 * @param[out] bottom  bottom
 * @param[out] left    left
 * @param[out] right   right
 */
PLAY_CGLM_INLINE
void
persp_decomp_rh_zo(mat4 proj,
                   float * __restrict nearZ, float * __restrict farZ,
                   float * __restrict top,     float * __restrict bottom,
                   float * __restrict left,    float * __restrict right)
{
    float m00, m11, m20, m21, m22, m32, n, f;
    float n_m11, n_m00;

    m00 = proj[0][0];
    m11 = proj[1][1];
    m20 = proj[2][0];
    m21 = proj[2][1];
    m22 = proj[2][2];
    m32 = proj[3][2];

    n = m32 / m22;
    f = m32 / (m22 + 1.0f);

    n_m11 = n / m11;
    n_m00 = n / m00;

    *nearZ = n;
    *farZ  = f;
    *bottom  = n_m11 * (m21 - 1.0f);
    *top     = n_m11 * (m21 + 1.0f);
    *left    = n_m00 * (m20 - 1.0f);
    *right   = n_m00 * (m20 + 1.0f);
}

/*!
 * @brief decomposes frustum values of perspective projection
 *        with angle values with a right-hand coordinate system and a
 *        clip-space of [0, 1].
 *        this makes easy to get all values at once
 *
 * @param[in]  proj   perspective projection matrix
 * @param[out] dest   array
 */
PLAY_CGLM_INLINE
void
persp_decompv_rh_zo(mat4 proj, float dest[6])
{
    persp_decomp_rh_zo(proj, &dest[0], &dest[1], &dest[2],
                       &dest[3], &dest[4], &dest[5]);
}

/*!
 * @brief decomposes left and right values of perspective projection (ZO).
 *        x stands for x axis (left / right axis)
 *
 * @param[in]  proj  perspective projection matrix
 * @param[out] left  left
 * @param[out] right right
 */
PLAY_CGLM_INLINE
void
persp_decomp_x_rh_zo(mat4 proj,
                     float * __restrict left,
                     float * __restrict right)
{
    float nearZ, m20, m00, m22;

    m00 = proj[0][0];
    m20 = proj[2][0];
    m22 = proj[2][2];

    nearZ = proj[3][2] / m22;
    *left   = nearZ * (m20 - 1.0f) / m00;
    *right  = nearZ * (m20 + 1.0f) / m00;
}

/*!
 * @brief decomposes top and bottom values of perspective projection
 *        with angle values with a right-hand coordinate system and a
 *        clip-space of [0, 1].
 *        y stands for y axis (top / bottom axis)
 *
 * @param[in]  proj   perspective projection matrix
 * @param[out] top    top
 * @param[out] bottom bottom
 */
PLAY_CGLM_INLINE
void
persp_decomp_y_rh_zo(mat4 proj,
                     float * __restrict top,
                     float * __restrict bottom)
{
    float nearZ, m21, m11, m22;

    m21 = proj[2][1];
    m11 = proj[1][1];
    m22 = proj[2][2];

    nearZ = proj[3][2] / m22;
    *bottom = nearZ * (m21 - 1) / m11;
    *top    = nearZ * (m21 + 1) / m11;
}

/*!
 * @brief decomposes near and far values of perspective projection
 *        with angle values with a right-hand coordinate system and a
 *        clip-space of [0, 1].
 *        z stands for z axis (near / far axis)
 *
 * @param[in]  proj    perspective projection matrix
 * @param[out] nearZ   near
 * @param[out] farZ    far
 */
PLAY_CGLM_INLINE
void
persp_decomp_z_rh_zo(mat4 proj,
                     float * __restrict nearZ,
                     float * __restrict farZ)
{
    float m32, m22;

    m32 = proj[3][2];
    m22 = proj[2][2];

    *nearZ = m32 / m22;
    *farZ  = m32 / (m22 + 1.0f);
}

/*!
 * @brief decomposes far value of perspective projection
 *        with angle values with a right-hand coordinate system and a
 *        clip-space of [0, 1].
 *
 * @param[in]  proj   perspective projection matrix
 * @param[out] farZ   far
 */
PLAY_CGLM_INLINE
void
persp_decomp_far_rh_zo(mat4 proj, float * __restrict farZ)
{
    *farZ = proj[3][2] / (proj[2][2] + 1.0f);
}

/*!
 * @brief decomposes near value of perspective projection
 *        with angle values with a right-hand coordinate system and a
 *        clip-space of [0, 1].
 *
 * @param[in]  proj    perspective projection matrix
 * @param[out] nearZ   near
 */
PLAY_CGLM_INLINE
void
persp_decomp_near_rh_zo(mat4 proj, float * __restrict nearZ)
{
    *nearZ = proj[3][2] / proj[2][2];
}

/*!
 * @brief returns sizes of near and far planes of perspective projection
 *        with a right-hand coordinate system and a
 *        clip-space of [0, 1].
 *
 * @param[in]  proj perspective projection matrix
 * @param[in]  fovy fovy (see brief)
 * @param[out] dest sizes order: [Wnear, Hnear, Wfar, Hfar]
 */
PLAY_CGLM_INLINE
void
persp_sizes_rh_zo(mat4 proj, float fovy, vec4 dest)
{
    float t, a, nearZ, farZ;

    t = 2.0f * tanf(fovy * 0.5f);
    a = persp_aspect(proj);

    persp_decomp_z_rh_zo(proj, &nearZ, &farZ);

    dest[1]  = t * nearZ;
    dest[3]  = t * farZ;
    dest[0]  = a * dest[1];
    dest[2]  = a * dest[3];
}

/*!
 * @brief returns field of view angle along the Y-axis (in radians)
 *        with a right-hand coordinate system and a clip-space of [0, 1].
 *
 * if you need to degrees, use deg to convert it or use this:
 * fovy_deg = deg(persp_fovy(projMatrix))
 *
 * @param[in] proj perspective projection matrix
 */
PLAY_CGLM_INLINE
float
persp_fovy_rh_zo(mat4 proj)
{
    return persp_fovy(proj);
}

/*!
 * @brief returns aspect ratio of perspective projection
 *        with a right-hand coordinate system and a clip-space of [0, 1].
 *
 * @param[in] proj perspective projection matrix
 */
PLAY_CGLM_INLINE
float
persp_aspect_rh_zo(mat4 proj)
{
    return persp_aspect(proj);
}



/*** End of inlined file: persp_rh_zo.h ***/


/*** Start of inlined file: ortho_rh_no.h ***/
/*
 Functions:
   PLAY_CGLM_INLINE void ortho_rh_no(float left,    float right,
                                    float bottom,  float top,
                                    float nearZ, float farZ,
                                    mat4  dest)
   PLAY_CGLM_INLINE void ortho_aabb_rh_no(vec3 box[2], mat4 dest)
   PLAY_CGLM_INLINE void ortho_aabb_p_rh_no(vec3 box[2],
                                           float padding,
                                           mat4 dest)
   PLAY_CGLM_INLINE void ortho_aabb_pz_rh_no(vec3 box[2],
                                            float padding,
                                            mat4 dest)
   PLAY_CGLM_INLINE void ortho_default_rh_no(float aspect,
                                            mat4 dest)
   PLAY_CGLM_INLINE void ortho_default_s_rh_no(float aspect,
                                              float size,
                                              mat4 dest)
 */




/*!
 * @brief set up orthographic projection matrix
 *        with a right-hand coordinate system and a
 *        clip-space of [-1, 1].
 *
 * @param[in]  left    viewport.left
 * @param[in]  right   viewport.right
 * @param[in]  bottom  viewport.bottom
 * @param[in]  top     viewport.top
 * @param[in]  nearZ   near clipping plane
 * @param[in]  farZ    far clipping plane
 * @param[out] dest    result matrix
 */
PLAY_CGLM_INLINE
void
ortho_rh_no(float left,    float right,
            float bottom,  float top,
            float nearZ, float farZ,
            mat4  dest)
{
    float rl, tb, fn;

    mat4_zero(dest);

    rl = 1.0f / (right  - left);
    tb = 1.0f / (top    - bottom);
    fn =-1.0f / (farZ - nearZ);

    dest[0][0] = 2.0f * rl;
    dest[1][1] = 2.0f * tb;
    dest[2][2] = 2.0f * fn;
    dest[3][0] =-(right  + left)    * rl;
    dest[3][1] =-(top    + bottom)  * tb;
    dest[3][2] = (farZ + nearZ) * fn;
    dest[3][3] = 1.0f;
}

/*!
 * @brief set up orthographic projection matrix using bounding box
 *        with a right-hand coordinate system and a
 *        clip-space of [-1, 1].
 *
 * bounding box (AABB) must be in view space
 *
 * @param[in]  box   AABB
 * @param[out] dest  result matrix
 */
PLAY_CGLM_INLINE
void
ortho_aabb_rh_no(vec3 box[2], mat4 dest)
{
    ortho_rh_no(box[0][0],  box[1][0],
                box[0][1],  box[1][1],
                -box[1][2], -box[0][2],
                dest);
}

/*!
 * @brief set up orthographic projection matrix using bounding box
 *        with a right-hand coordinate system and a
 *        clip-space of [-1, 1].
 *
 * bounding box (AABB) must be in view space
 *
 * @param[in]  box     AABB
 * @param[in]  padding padding
 * @param[out] dest    result matrix
 */
PLAY_CGLM_INLINE
void
ortho_aabb_p_rh_no(vec3 box[2], float padding, mat4 dest)
{
    ortho_rh_no(box[0][0] - padding,    box[1][0] + padding,
                box[0][1] - padding,    box[1][1] + padding,
                -(box[1][2] + padding), -(box[0][2] - padding),
                dest);
}

/*!
 * @brief set up orthographic projection matrix using bounding box
 *        with a right-hand coordinate system and a
 *        clip-space of [-1, 1].
 *
 * bounding box (AABB) must be in view space
 *
 * @param[in]  box     AABB
 * @param[in]  padding padding for near and far
 * @param[out] dest    result matrix
 */
PLAY_CGLM_INLINE
void
ortho_aabb_pz_rh_no(vec3 box[2], float padding, mat4 dest)
{
    ortho_rh_no(box[0][0],              box[1][0],
                box[0][1],              box[1][1],
                -(box[1][2] + padding), -(box[0][2] - padding),
                dest);
}

/*!
 * @brief set up unit orthographic projection matrix
 *        with a right-hand coordinate system and a
 *        clip-space of [-1, 1].
 *
 * @param[in]  aspect aspect ration ( width / height )
 * @param[out] dest   result matrix
 */
PLAY_CGLM_INLINE
void
ortho_default_rh_no(float aspect, mat4 dest)
{
    if (aspect >= 1.0f)
    {
        ortho_rh_no(-aspect, aspect, -1.0f, 1.0f, -100.0f, 100.0f, dest);
        return;
    }

    aspect = 1.0f / aspect;

    ortho_rh_no(-1.0f, 1.0f, -aspect, aspect, -100.0f, 100.0f, dest);
}

/*!
 * @brief set up orthographic projection matrix with given CUBE size
 *        with a right-hand coordinate system and a
 *        clip-space of [-1, 1].
 *
 * @param[in]  aspect aspect ratio ( width / height )
 * @param[in]  size   cube size
 * @param[out] dest   result matrix
 */
PLAY_CGLM_INLINE
void
ortho_default_s_rh_no(float aspect, float size, mat4 dest)
{
    if (aspect >= 1.0f)
    {
        ortho_rh_no(-size * aspect,
                    size * aspect,
                    -size,
                    size,
                    -size - 100.0f,
                    size + 100.0f,
                    dest);
        return;
    }

    ortho_rh_no(-size,
                size,
                -size / aspect,
                size / aspect,
                -size - 100.0f,
                size + 100.0f,
                dest);
}



/*** End of inlined file: ortho_rh_no.h ***/


/*** Start of inlined file: persp_rh_no.h ***/
/*
 Functions:
   PLAY_CGLM_INLINE void frustum_rh_no(float left,    float right,
                                      float bottom,  float top,
                                      float nearZ, float farZ,
                                      mat4  dest)
   PLAY_CGLM_INLINE void perspective_rh_no(float fovy,
                                          float aspect,
                                          float nearZ,
                                          float farZ,
                                          mat4  dest)
   PLAY_CGLM_INLINE void perspective_default_rh_no(float aspect, mat4 dest)
   PLAY_CGLM_INLINE void perspective_resize_rh_no(float aspect, mat4 proj)
   PLAY_CGLM_INLINE void persp_move_far_rh_no(mat4 proj,
                                             float deltaFar)
   PLAY_CGLM_INLINE void persp_decomp_rh_no(mat4 proj,
                                           float * __restrict nearZ,
                                           float * __restrict farZ,
                                           float * __restrict top,
                                           float * __restrict bottom,
                                           float * __restrict left,
                                           float * __restrict right)
  PLAY_CGLM_INLINE void persp_decompv_rh_no(mat4 proj,
                                           float dest[6])
  PLAY_CGLM_INLINE void persp_decomp_x_rh_no(mat4 proj,
                                            float * __restrict left,
                                            float * __restrict right)
  PLAY_CGLM_INLINE void persp_decomp_y_rh_no(mat4 proj,
                                            float * __restrict top,
                                            float * __restrict bottom)
  PLAY_CGLM_INLINE void persp_decomp_z_rh_no(mat4 proj,
                                            float * __restrict nearZ,
                                            float * __restrict farZ)
  PLAY_CGLM_INLINE void persp_decomp_far_rh_no(mat4 proj, float * __restrict farZ)
  PLAY_CGLM_INLINE void persp_decomp_near_rh_no(mat4 proj, float * __restrict nearZ)
  PLAY_CGLM_INLINE void persp_sizes_rh_no(mat4 proj, float fovy, vec4 dest)
 */




/*!
 * @brief set up perspective peprojection matrix
 *        with a right-hand coordinate system and a
 *        clip-space of [-1, 1].
 *
 * @param[in]  left    viewport.left
 * @param[in]  right   viewport.right
 * @param[in]  bottom  viewport.bottom
 * @param[in]  top     viewport.top
 * @param[in]  nearZ   near clipping plane
 * @param[in]  farZ    far clipping plane
 * @param[out] dest    result matrix
 */
PLAY_CGLM_INLINE
void
frustum_rh_no(float left,    float right,
              float bottom,  float top,
              float nearZ, float farZ,
              mat4  dest)
{
    float rl, tb, fn, nv;

    mat4_zero(dest);

    rl = 1.0f / (right  - left);
    tb = 1.0f / (top    - bottom);
    fn =-1.0f / (farZ - nearZ);
    nv = 2.0f * nearZ;

    dest[0][0] = nv * rl;
    dest[1][1] = nv * tb;
    dest[2][0] = (right  + left)    * rl;
    dest[2][1] = (top    + bottom)  * tb;
    dest[2][2] = (farZ + nearZ) * fn;
    dest[2][3] =-1.0f;
    dest[3][2] = farZ * nv * fn;
}

/*!
 * @brief set up perspective projection matrix
 *        with a right-hand coordinate system and a
 *        clip-space of [-1, 1].
 *
 * @param[in]  fovy    field of view angle
 * @param[in]  aspect  aspect ratio ( width / height )
 * @param[in]  nearZ   near clipping plane
 * @param[in]  farZ    far clipping planes
 * @param[out] dest    result matrix
 */
PLAY_CGLM_INLINE
void
perspective_rh_no(float fovy,
                  float aspect,
                  float nearZ,
                  float farZ,
                  mat4  dest)
{
    float f, fn;

    mat4_zero(dest);

    f  = 1.0f / tanf(fovy * 0.5f);
    fn = 1.0f / (nearZ - farZ);

    dest[0][0] = f / aspect;
    dest[1][1] = f;
    dest[2][2] = (nearZ + farZ) * fn;
    dest[2][3] =-1.0f;
    dest[3][2] = 2.0f * nearZ * farZ * fn;

}

/*!
 * @brief set up perspective projection matrix with default near/far
 *        and angle values with a right-hand coordinate system and a
 *        clip-space of [-1, 1].
 *
 * @param[in]  aspect aspect ratio ( width / height )
 * @param[out] dest   result matrix
 */
PLAY_CGLM_INLINE
void
perspective_default_rh_no(float aspect, mat4 dest)
{
    perspective_rh_no(PLAY_CGLM_PI_4f, aspect, 0.01f, 100.0f, dest);
}

/*!
 * @brief resize perspective matrix by aspect ratio ( width / height )
 *        this makes very easy to resize proj matrix when window /viewport
 *        resized with a right-hand coordinate system and a
 *        clip-space of [-1, 1].
 *
 * @param[in]      aspect aspect ratio ( width / height )
 * @param[in, out] proj   perspective projection matrix
 */
PLAY_CGLM_INLINE
void
perspective_resize_rh_no(float aspect, mat4 proj)
{
    if (proj[0][0] == 0.0f)
        return;

    proj[0][0] = proj[1][1] / aspect;
}

/*!
 * @brief extend perspective projection matrix's far distance
 *        with a right-hand coordinate system and a
 *        clip-space of [-1, 1].
 *
 * this function does not guarantee far >= near, be aware of that!
 *
 * @param[in, out] proj      projection matrix to extend
 * @param[in]      deltaFar  distance from existing far (negative to shink)
 */
PLAY_CGLM_INLINE
void
persp_move_far_rh_no(mat4 proj, float deltaFar)
{
    float fn, farZ, nearZ, p22, p32;

    p22        = proj[2][2];
    p32        = proj[3][2];

    nearZ    = p32 / (p22 - 1.0f);
    farZ     = p32 / (p22 + 1.0f) + deltaFar;
    fn         = 1.0f / (nearZ - farZ);

    proj[2][2] = (farZ + nearZ) * fn;
    proj[3][2] = 2.0f * nearZ * farZ * fn;
}

/*!
 * @brief decomposes frustum values of perspective projection
 *        with a right-hand coordinate system and a
 *        clip-space of [-1, 1].
 *
 * @param[in]  proj    perspective projection matrix
 * @param[out] nearZ   near
 * @param[out] farZ    far
 * @param[out] top     top
 * @param[out] bottom  bottom
 * @param[out] left    left
 * @param[out] right   right
 */
PLAY_CGLM_INLINE
void
persp_decomp_rh_no(mat4 proj,
                   float * __restrict nearZ, float * __restrict farZ,
                   float * __restrict top,     float * __restrict bottom,
                   float * __restrict left,    float * __restrict right)
{
    float m00, m11, m20, m21, m22, m32, n, f;
    float n_m11, n_m00;

    m00 = proj[0][0];
    m11 = proj[1][1];
    m20 = proj[2][0];
    m21 = proj[2][1];
    m22 = proj[2][2];
    m32 = proj[3][2];

    n = m32 / (m22 - 1.0f);
    f = m32 / (m22 + 1.0f);

    n_m11 = n / m11;
    n_m00 = n / m00;

    *nearZ = n;
    *farZ  = f;
    *bottom  = n_m11 * (m21 - 1.0f);
    *top     = n_m11 * (m21 + 1.0f);
    *left    = n_m00 * (m20 - 1.0f);
    *right   = n_m00 * (m20 + 1.0f);
}

/*!
 * @brief decomposes frustum values of perspective projection
 *        with a right-hand coordinate system and a
 *        clip-space of [-1, 1].
 *        this makes easy to get all values at once
 *
 * @param[in]  proj   perspective projection matrix
 * @param[out] dest   array
 */
PLAY_CGLM_INLINE
void
persp_decompv_rh_no(mat4 proj, float dest[6])
{
    persp_decomp_rh_no(proj, &dest[0], &dest[1], &dest[2],
                       &dest[3], &dest[4], &dest[5]);
}

/*!
 * @brief decomposes left and right values of perspective projection
 *        with a right-hand coordinate system and a
 *        clip-space of [-1, 1].
 *        x stands for x axis (left / right axis)
 *
 * @param[in]  proj  perspective projection matrix
 * @param[out] left  left
 * @param[out] right right
 */
PLAY_CGLM_INLINE
void
persp_decomp_x_rh_no(mat4 proj,
                     float * __restrict left,
                     float * __restrict right)
{
    float nearZ, m20, m00, m22;

    m00 = proj[0][0];
    m20 = proj[2][0];
    m22 = proj[2][2];

    nearZ = proj[3][2] / (m22 - 1.0f);
    *left   = nearZ * (m20 - 1.0f) / m00;
    *right  = nearZ * (m20 + 1.0f) / m00;
}

/*!
 * @brief decomposes top and bottom values of perspective projection
 *        with a right-hand coordinate system and a
 *        clip-space of [-1, 1].
 *        y stands for y axis (top / bottom axis)
 *
 * @param[in]  proj   perspective projection matrix
 * @param[out] top    top
 * @param[out] bottom bottom
 */
PLAY_CGLM_INLINE
void
persp_decomp_y_rh_no(mat4 proj,
                     float * __restrict top,
                     float * __restrict bottom)
{
    float nearZ, m21, m11, m22;

    m21 = proj[2][1];
    m11 = proj[1][1];
    m22 = proj[2][2];

    nearZ = proj[3][2] / (m22 - 1.0f);
    *bottom = nearZ * (m21 - 1.0f) / m11;
    *top    = nearZ * (m21 + 1.0f) / m11;
}

/*!
 * @brief decomposes near and far values of perspective projection
 *        with a right-hand coordinate system and a
 *        clip-space of [-1, 1].
 *        z stands for z axis (near / far axis)
 *
 * @param[in]  proj    perspective projection matrix
 * @param[out] nearZ   near
 * @param[out] farZ    far
 */
PLAY_CGLM_INLINE
void
persp_decomp_z_rh_no(mat4 proj,
                     float * __restrict nearZ,
                     float * __restrict farZ)
{
    float m32, m22;

    m32 = proj[3][2];
    m22 = proj[2][2];

    *nearZ = m32 / (m22 - 1.0f);
    *farZ  = m32 / (m22 + 1.0f);
}

/*!
 * @brief decomposes far value of perspective projection
 *        with a right-hand coordinate system and a
 *        clip-space of [-1, 1].
 *
 * @param[in]  proj   perspective projection matrix
 * @param[out] farZ   far
 */
PLAY_CGLM_INLINE
void
persp_decomp_far_rh_no(mat4 proj, float * __restrict farZ)
{
    *farZ = proj[3][2] / (proj[2][2] + 1.0f);
}

/*!
 * @brief decomposes near value of perspective projection
 *        with a right-hand coordinate system and a
 *        clip-space of [-1, 1].
 *
 * @param[in]  proj    perspective projection matrix
 * @param[out] nearZ near
 */
PLAY_CGLM_INLINE
void
persp_decomp_near_rh_no(mat4 proj, float * __restrict nearZ)
{
    *nearZ = proj[3][2] / (proj[2][2] - 1.0f);
}

/*!
 * @brief returns sizes of near and far planes of perspective projection
 *        with a right-hand coordinate system and a
 *        clip-space of [-1, 1].
 *
 * @param[in]  proj perspective projection matrix
 * @param[in]  fovy fovy (see brief)
 * @param[out] dest sizes order: [Wnear, Hnear, Wfar, Hfar]
 */
PLAY_CGLM_INLINE
void
persp_sizes_rh_no(mat4 proj, float fovy, vec4 dest)
{
    float t, a, nearZ, farZ;

    t = 2.0f * tanf(fovy * 0.5f);
    a = persp_aspect(proj);

    persp_decomp_z_rh_no(proj, &nearZ, &farZ);

    dest[1]  = t * nearZ;
    dest[3]  = t * farZ;
    dest[0]  = a * dest[1];
    dest[2]  = a * dest[3];
}

/*!
 * @brief returns field of view angle along the Y-axis (in radians)
 *        with a right-hand coordinate system and a clip-space of [-1, 1].
 *
 * if you need to degrees, use deg to convert it or use this:
 * fovy_deg = deg(persp_fovy(projMatrix))
 *
 * @param[in] proj perspective projection matrix
 */
PLAY_CGLM_INLINE
float
persp_fovy_rh_no(mat4 proj)
{
    return persp_fovy(proj);
}

/*!
 * @brief returns aspect ratio of perspective projection
 *        with a right-hand coordinate system and a clip-space of [-1, 1].
 *
 * @param[in] proj perspective projection matrix
 */
PLAY_CGLM_INLINE
float
persp_aspect_rh_no(mat4 proj)
{
    return persp_aspect(proj);
}



/*** End of inlined file: persp_rh_no.h ***/


/*** Start of inlined file: view_lh_zo.h ***/
/*
 Functions:
   PLAY_CGLM_INLINE void lookat_lh_zo(vec3 eye, vec3 center, vec3 up, mat4 dest)
   PLAY_CGLM_INLINE void look_lh_zo(vec3 eye, vec3 dir, vec3 up, mat4 dest)
   PLAY_CGLM_INLINE void look_anyup_lh_zo(vec3 eye, vec3 dir, mat4 dest)
 */





/*** Start of inlined file: view_lh.h ***/
/*
 Functions:
   PLAY_CGLM_INLINE void lookat_lh(vec3 eye, vec3 center, vec3 up, mat4 dest)
   PLAY_CGLM_INLINE void look_lh(vec3 eye, vec3 dir, vec3 up, mat4 dest)
   PLAY_CGLM_INLINE void look_anyup_lh(vec3 eye, vec3 dir, mat4 dest)
 */




/*!
 * @brief set up view matrix (LH)
 *
 * NOTE: The UP vector must not be parallel to the line of sight from
 *       the eye point to the reference point
 *
 * @param[in]  eye    eye vector
 * @param[in]  center center vector
 * @param[in]  up     up vector
 * @param[out] dest   result matrix
 */
PLAY_CGLM_INLINE
void
lookat_lh(vec3 eye, vec3 center, vec3 up, mat4 dest)
{
    PLAY_CGLM_ALIGN(8) vec3 f, u, s;

    vec3_sub(center, eye, f);
    vec3_normalize(f);

    vec3_crossn(up, f, s);
    vec3_cross(f, s, u);

    dest[0][0] = s[0];
    dest[0][1] = u[0];
    dest[0][2] = f[0];
    dest[1][0] = s[1];
    dest[1][1] = u[1];
    dest[1][2] = f[1];
    dest[2][0] = s[2];
    dest[2][1] = u[2];
    dest[2][2] = f[2];
    dest[3][0] =-vec3_dot(s, eye);
    dest[3][1] =-vec3_dot(u, eye);
    dest[3][2] =-vec3_dot(f, eye);
    dest[0][3] = dest[1][3] = dest[2][3] = 0.0f;
    dest[3][3] = 1.0f;
}

/*!
 * @brief set up view matrix with left handed coordinate system
 *
 * convenient wrapper for lookat: if you only have direction not target self
 * then this might be useful. Because you need to get target from direction.
 *
 * NOTE: The UP vector must not be parallel to the line of sight from
 *       the eye point to the reference point
 *
 * @param[in]  eye    eye vector
 * @param[in]  dir    direction vector
 * @param[in]  up     up vector
 * @param[out] dest   result matrix
 */
PLAY_CGLM_INLINE
void
look_lh(vec3 eye, vec3 dir, vec3 up, mat4 dest)
{
    PLAY_CGLM_ALIGN(8) vec3 target;
    vec3_add(eye, dir, target);
    lookat_lh(eye, target, up, dest);
}

/*!
 * @brief set up view matrix with left handed coordinate system
 *
 * convenient wrapper for look: if you only have direction and if you don't
 * care what UP vector is then this might be useful to create view matrix
 *
 * @param[in]  eye    eye vector
 * @param[in]  dir    direction vector
 * @param[out] dest   result matrix
 */
PLAY_CGLM_INLINE
void
look_anyup_lh(vec3 eye, vec3 dir, mat4 dest)
{
    PLAY_CGLM_ALIGN(8) vec3 up;
    vec3_ortho(dir, up);
    look_lh(eye, dir, up, dest);
}



/*** End of inlined file: view_lh.h ***/

/*!
 * @brief set up view matrix with left handed coordinate system.
 *
 * NOTE: The UP vector must not be parallel to the line of sight from
 *       the eye point to the reference point
 *
 * @param[in]  eye    eye vector
 * @param[in]  center center vector
 * @param[in]  up     up vector
 * @param[out] dest   result matrix
 */
PLAY_CGLM_INLINE
void
lookat_lh_zo(vec3 eye, vec3 center, vec3 up, mat4 dest)
{
    lookat_lh(eye, center, up, dest);
}

/*!
 * @brief set up view matrix with left handed coordinate system.
 *
 * convenient wrapper for lookat: if you only have direction not target self
 * then this might be useful. Because you need to get target from direction.
 *
 * NOTE: The UP vector must not be parallel to the line of sight from
 *       the eye point to the reference point
 *
 * @param[in]  eye    eye vector
 * @param[in]  dir    direction vector
 * @param[in]  up     up vector
 * @param[out] dest   result matrix
 */
PLAY_CGLM_INLINE
void
look_lh_zo(vec3 eye, vec3 dir, vec3 up, mat4 dest)
{
    look_lh(eye, dir, up, dest);
}

/*!
 * @brief set up view matrix with left handed coordinate system.
 *
 * convenient wrapper for look: if you only have direction and if you don't
 * care what UP vector is then this might be useful to create view matrix
 *
 * @param[in]  eye    eye vector
 * @param[in]  dir    direction vector
 * @param[out] dest   result matrix
 */
PLAY_CGLM_INLINE
void
look_anyup_lh_zo(vec3 eye, vec3 dir, mat4 dest)
{
    look_anyup_lh(eye, dir, dest);
}



/*** End of inlined file: view_lh_zo.h ***/


/*** Start of inlined file: view_lh_no.h ***/
/*
 Functions:
   PLAY_CGLM_INLINE void lookat_lh_no(vec3 eye, vec3 center, vec3 up, mat4 dest)
   PLAY_CGLM_INLINE void look_lh_no(vec3 eye, vec3 dir, vec3 up, mat4 dest)
   PLAY_CGLM_INLINE void look_anyup_lh_no(vec3 eye, vec3 dir, mat4 dest)
 */




/*!
 * @brief set up view matrix with left handed coordinate system.
 *
 * NOTE: The UP vector must not be parallel to the line of sight from
 *       the eye point to the reference point
 *
 * @param[in]  eye    eye vector
 * @param[in]  center center vector
 * @param[in]  up     up vector
 * @param[out] dest   result matrix
 */
PLAY_CGLM_INLINE
void
lookat_lh_no(vec3 eye, vec3 center, vec3 up, mat4 dest)
{
    lookat_lh(eye, center, up, dest);
}

/*!
 * @brief set up view matrix with left handed coordinate system.
 *
 * convenient wrapper for lookat: if you only have direction not target self
 * then this might be useful. Because you need to get target from direction.
 *
 * NOTE: The UP vector must not be parallel to the line of sight from
 *       the eye point to the reference point
 *
 * @param[in]  eye    eye vector
 * @param[in]  dir    direction vector
 * @param[in]  up     up vector
 * @param[out] dest   result matrix
 */
PLAY_CGLM_INLINE
void
look_lh_no(vec3 eye, vec3 dir, vec3 up, mat4 dest)
{
    look_lh(eye, dir, up, dest);
}

/*!
 * @brief set up view matrix with left handed coordinate system.
 *
 * convenient wrapper for look: if you only have direction and if you don't
 * care what UP vector is then this might be useful to create view matrix
 *
 * @param[in]  eye    eye vector
 * @param[in]  dir    direction vector
 * @param[out] dest   result matrix
 */
PLAY_CGLM_INLINE
void
look_anyup_lh_no(vec3 eye, vec3 dir, mat4 dest)
{
    look_anyup_lh(eye, dir, dest);
}



/*** End of inlined file: view_lh_no.h ***/


/*** Start of inlined file: view_rh_zo.h ***/
/*
 Functions:
   PLAY_CGLM_INLINE void lookat_rh_zo(vec3 eye, vec3 center, vec3 up, mat4 dest)
   PLAY_CGLM_INLINE void look_rh_zo(vec3 eye, vec3 dir, vec3 up, mat4 dest)
   PLAY_CGLM_INLINE void look_anyup_rh_zo(vec3 eye, vec3 dir, mat4 dest)
 */





/*** Start of inlined file: view_rh.h ***/
/*
 Functions:
   PLAY_CGLM_INLINE void lookat_rh(vec3 eye, vec3 center, vec3 up, mat4 dest)
   PLAY_CGLM_INLINE void look_rh(vec3 eye, vec3 dir, vec3 up, mat4 dest)
   PLAY_CGLM_INLINE void look_anyup_rh(vec3 eye, vec3 dir, mat4 dest)
 */




/*!
 * @brief set up view matrix with right handed coordinate system.
 *
 * NOTE: The UP vector must not be parallel to the line of sight from
 *       the eye point to the reference point
 *
 * @param[in]  eye    eye vector
 * @param[in]  center center vector
 * @param[in]  up     up vector
 * @param[out] dest   result matrix
 */
PLAY_CGLM_INLINE
void
lookat_rh(vec3 eye, vec3 center, vec3 up, mat4 dest)
{
    PLAY_CGLM_ALIGN(8) vec3 f, u, s;

    vec3_sub(center, eye, f);
    vec3_normalize(f);

    vec3_crossn(f, up, s);
    vec3_cross(s, f, u);

    dest[0][0] = s[0];
    dest[0][1] = u[0];
    dest[0][2] =-f[0];
    dest[1][0] = s[1];
    dest[1][1] = u[1];
    dest[1][2] =-f[1];
    dest[2][0] = s[2];
    dest[2][1] = u[2];
    dest[2][2] =-f[2];
    dest[3][0] =-vec3_dot(s, eye);
    dest[3][1] =-vec3_dot(u, eye);
    dest[3][2] = vec3_dot(f, eye);
    dest[0][3] = dest[1][3] = dest[2][3] = 0.0f;
    dest[3][3] = 1.0f;
}

/*!
 * @brief set up view matrix with right handed coordinate system.
 *
 * convenient wrapper for lookat: if you only have direction not target self
 * then this might be useful. Because you need to get target from direction.
 *
 * NOTE: The UP vector must not be parallel to the line of sight from
 *       the eye point to the reference point
 *
 * @param[in]  eye    eye vector
 * @param[in]  dir    direction vector
 * @param[in]  up     up vector
 * @param[out] dest   result matrix
 */
PLAY_CGLM_INLINE
void
look_rh(vec3 eye, vec3 dir, vec3 up, mat4 dest)
{
    PLAY_CGLM_ALIGN(8) vec3 target;
    vec3_add(eye, dir, target);
    lookat_rh(eye, target, up, dest);
}

/*!
 * @brief set up view matrix with right handed coordinate system.
 *
 * convenient wrapper for look: if you only have direction and if you don't
 * care what UP vector is then this might be useful to create view matrix
 *
 * @param[in]  eye    eye vector
 * @param[in]  dir    direction vector
 * @param[out] dest   result matrix
 */
PLAY_CGLM_INLINE
void
look_anyup_rh(vec3 eye, vec3 dir, mat4 dest)
{
    PLAY_CGLM_ALIGN(8) vec3 up;
    vec3_ortho(dir, up);
    look_rh(eye, dir, up, dest);
}



/*** End of inlined file: view_rh.h ***/

/*!
 * @brief set up view matrix with right handed coordinate system.
 *
 * NOTE: The UP vector must not be parallel to the line of sight from
 *       the eye point to the reference point
 *
 * @param[in]  eye    eye vector
 * @param[in]  center center vector
 * @param[in]  up     up vector
 * @param[out] dest   result matrix
 */
PLAY_CGLM_INLINE
void
lookat_rh_zo(vec3 eye, vec3 center, vec3 up, mat4 dest)
{
    lookat_rh(eye, center, up, dest);
}

/*!
 * @brief set up view matrix with right handed coordinate system.
 *
 * convenient wrapper for lookat: if you only have direction not target self
 * then this might be useful. Because you need to get target from direction.
 *
 * NOTE: The UP vector must not be parallel to the line of sight from
 *       the eye point to the reference point
 *
 * @param[in]  eye    eye vector
 * @param[in]  dir    direction vector
 * @param[in]  up     up vector
 * @param[out] dest   result matrix
 */
PLAY_CGLM_INLINE
void
look_rh_zo(vec3 eye, vec3 dir, vec3 up, mat4 dest)
{
    look_rh(eye, dir, up, dest);
}

/*!
 * @brief set up view matrix with right handed coordinate system.
 *
 * convenient wrapper for look: if you only have direction and if you don't
 * care what UP vector is then this might be useful to create view matrix
 *
 * @param[in]  eye    eye vector
 * @param[in]  dir    direction vector
 * @param[out] dest   result matrix
 */
PLAY_CGLM_INLINE
void
look_anyup_rh_zo(vec3 eye, vec3 dir, mat4 dest)
{
    look_anyup_rh(eye, dir, dest);
}



/*** End of inlined file: view_rh_zo.h ***/


/*** Start of inlined file: view_rh_no.h ***/
/*
 Functions:
   PLAY_CGLM_INLINE void lookat_rh_no(vec3 eye, vec3 center, vec3 up, mat4 dest)
   PLAY_CGLM_INLINE void look_rh_no(vec3 eye, vec3 dir, vec3 up, mat4 dest)
   PLAY_CGLM_INLINE void look_anyup_rh_no(vec3 eye, vec3 dir, mat4 dest)
 */




/*!
 * @brief set up view matrix with right handed coordinate system.
 *
 * NOTE: The UP vector must not be parallel to the line of sight from
 *       the eye point to the reference point
 *
 * @param[in]  eye    eye vector
 * @param[in]  center center vector
 * @param[in]  up     up vector
 * @param[out] dest   result matrix
 */
PLAY_CGLM_INLINE
void
lookat_rh_no(vec3 eye, vec3 center, vec3 up, mat4 dest)
{
    lookat_rh(eye, center, up, dest);
}

/*!
 * @brief set up view matrix with right handed coordinate system.
 *
 * convenient wrapper for lookat: if you only have direction not target self
 * then this might be useful. Because you need to get target from direction.
 *
 * NOTE: The UP vector must not be parallel to the line of sight from
 *       the eye point to the reference point
 *
 * @param[in]  eye    eye vector
 * @param[in]  dir    direction vector
 * @param[in]  up     up vector
 * @param[out] dest   result matrix
 */
PLAY_CGLM_INLINE
void
look_rh_no(vec3 eye, vec3 dir, vec3 up, mat4 dest)
{
    look_rh(eye, dir, up, dest);
}

/*!
 * @brief set up view matrix with right handed coordinate system.
 *
 * convenient wrapper for look: if you only have direction and if you don't
 * care what UP vector is then this might be useful to create view matrix
 *
 * @param[in]  eye    eye vector
 * @param[in]  dir    direction vector
 * @param[out] dest   result matrix
 */
PLAY_CGLM_INLINE
void
look_anyup_rh_no(vec3 eye, vec3 dir, mat4 dest)
{
    look_anyup_rh(eye, dir, dest);
}



/*** End of inlined file: view_rh_no.h ***/

/*!
 * @brief set up perspective peprojection matrix
 *
 * @param[in]  left    viewport.left
 * @param[in]  right   viewport.right
 * @param[in]  bottom  viewport.bottom
 * @param[in]  top     viewport.top
 * @param[in]  nearZ   near clipping plane
 * @param[in]  farZ    far clipping plane
 * @param[out] dest    result matrix
 */
PLAY_CGLM_INLINE
void
frustum(float left,    float right,
        float bottom,  float top,
        float nearZ,   float farZ,
        mat4  dest)
{
#if PLAY_CGLM_CONFIG_CLIP_CONTROL == PLAY_CGLM_CLIP_CONTROL_LH_ZO
    frustum_lh_zo(left, right, bottom, top, nearZ, farZ, dest);
#elif PLAY_CGLM_CONFIG_CLIP_CONTROL == PLAY_CGLM_CLIP_CONTROL_LH_NO
    frustum_lh_no(left, right, bottom, top, nearZ, farZ, dest);
#elif PLAY_CGLM_CONFIG_CLIP_CONTROL == PLAY_CGLM_CLIP_CONTROL_RH_ZO
    frustum_rh_zo(left, right, bottom, top, nearZ, farZ, dest);
#elif PLAY_CGLM_CONFIG_CLIP_CONTROL == PLAY_CGLM_CLIP_CONTROL_RH_NO
    frustum_rh_no(left, right, bottom, top, nearZ, farZ, dest);
#endif
}

/*!
 * @brief set up orthographic projection matrix
 *
 * @param[in]  left    viewport.left
 * @param[in]  right   viewport.right
 * @param[in]  bottom  viewport.bottom
 * @param[in]  top     viewport.top
 * @param[in]  nearZ   near clipping plane
 * @param[in]  farZ    far clipping plane
 * @param[out] dest    result matrix
 */
PLAY_CGLM_INLINE
void
ortho(float left,    float right,
      float bottom,  float top,
      float nearZ,   float farZ,
      mat4  dest)
{
#if PLAY_CGLM_CONFIG_CLIP_CONTROL == PLAY_CGLM_CLIP_CONTROL_LH_ZO
    ortho_lh_zo(left, right, bottom, top, nearZ, farZ, dest);
#elif PLAY_CGLM_CONFIG_CLIP_CONTROL == PLAY_CGLM_CLIP_CONTROL_LH_NO
    ortho_lh_no(left, right, bottom, top, nearZ, farZ, dest);
#elif PLAY_CGLM_CONFIG_CLIP_CONTROL == PLAY_CGLM_CLIP_CONTROL_RH_ZO
    ortho_rh_zo(left, right, bottom, top, nearZ, farZ, dest);
#elif PLAY_CGLM_CONFIG_CLIP_CONTROL == PLAY_CGLM_CLIP_CONTROL_RH_NO
    ortho_rh_no(left, right, bottom, top, nearZ, farZ, dest);
#endif
}

/*!
 * @brief set up orthographic projection matrix using bounding box
 *
 * bounding box (AABB) must be in view space
 *
 * @param[in]  box   AABB
 * @param[out] dest  result matrix
 */
PLAY_CGLM_INLINE
void
ortho_aabb(vec3 box[2], mat4 dest)
{
#if PLAY_CGLM_CONFIG_CLIP_CONTROL == PLAY_CGLM_CLIP_CONTROL_LH_ZO
    ortho_aabb_lh_zo(box, dest);
#elif PLAY_CGLM_CONFIG_CLIP_CONTROL == PLAY_CGLM_CLIP_CONTROL_LH_NO
    ortho_aabb_lh_no(box, dest);
#elif PLAY_CGLM_CONFIG_CLIP_CONTROL == PLAY_CGLM_CLIP_CONTROL_RH_ZO
    ortho_aabb_rh_zo(box, dest);
#elif PLAY_CGLM_CONFIG_CLIP_CONTROL == PLAY_CGLM_CLIP_CONTROL_RH_NO
    ortho_aabb_rh_no(box, dest);
#endif
}

/*!
 * @brief set up orthographic projection matrix using bounding box
 *
 * bounding box (AABB) must be in view space
 *
 * @param[in]  box     AABB
 * @param[in]  padding padding
 * @param[out] dest    result matrix
 */
PLAY_CGLM_INLINE
void
ortho_aabb_p(vec3 box[2], float padding, mat4 dest)
{
#if PLAY_CGLM_CONFIG_CLIP_CONTROL == PLAY_CGLM_CLIP_CONTROL_LH_ZO
    ortho_aabb_p_lh_zo(box, padding, dest);
#elif PLAY_CGLM_CONFIG_CLIP_CONTROL == PLAY_CGLM_CLIP_CONTROL_LH_NO
    ortho_aabb_p_lh_no(box, padding, dest);
#elif PLAY_CGLM_CONFIG_CLIP_CONTROL == PLAY_CGLM_CLIP_CONTROL_RH_ZO
    ortho_aabb_p_rh_zo(box, padding, dest);
#elif PLAY_CGLM_CONFIG_CLIP_CONTROL == PLAY_CGLM_CLIP_CONTROL_RH_NO
    ortho_aabb_p_rh_no(box, padding, dest);
#endif
}

/*!
 * @brief set up orthographic projection matrix using bounding box
 *
 * bounding box (AABB) must be in view space
 *
 * @param[in]  box     AABB
 * @param[in]  padding padding for near and far
 * @param[out] dest    result matrix
 */
PLAY_CGLM_INLINE
void
ortho_aabb_pz(vec3 box[2], float padding, mat4 dest)
{
#if PLAY_CGLM_CONFIG_CLIP_CONTROL == PLAY_CGLM_CLIP_CONTROL_LH_ZO
    ortho_aabb_pz_lh_zo(box, padding, dest);
#elif PLAY_CGLM_CONFIG_CLIP_CONTROL == PLAY_CGLM_CLIP_CONTROL_LH_NO
    ortho_aabb_pz_lh_no(box, padding, dest);
#elif PLAY_CGLM_CONFIG_CLIP_CONTROL == PLAY_CGLM_CLIP_CONTROL_RH_ZO
    ortho_aabb_pz_rh_zo(box, padding, dest);
#elif PLAY_CGLM_CONFIG_CLIP_CONTROL == PLAY_CGLM_CLIP_CONTROL_RH_NO
    ortho_aabb_pz_rh_no(box, padding, dest);
#endif
}

/*!
 * @brief set up unit orthographic projection matrix
 *
 * @param[in]  aspect aspect ration ( width / height )
 * @param[out] dest   result matrix
 */
PLAY_CGLM_INLINE
void
ortho_default(float aspect, mat4 dest)
{
#if PLAY_CGLM_CONFIG_CLIP_CONTROL == PLAY_CGLM_CLIP_CONTROL_LH_ZO
    ortho_default_lh_zo(aspect, dest);
#elif PLAY_CGLM_CONFIG_CLIP_CONTROL == PLAY_CGLM_CLIP_CONTROL_LH_NO
    ortho_default_lh_no(aspect, dest);
#elif PLAY_CGLM_CONFIG_CLIP_CONTROL == PLAY_CGLM_CLIP_CONTROL_RH_ZO
    ortho_default_rh_zo(aspect, dest);
#elif PLAY_CGLM_CONFIG_CLIP_CONTROL == PLAY_CGLM_CLIP_CONTROL_RH_NO
    ortho_default_rh_no(aspect, dest);
#endif
}

/*!
 * @brief set up orthographic projection matrix with given CUBE size
 *
 * @param[in]  aspect aspect ratio ( width / height )
 * @param[in]  size   cube size
 * @param[out] dest   result matrix
 */
PLAY_CGLM_INLINE
void
ortho_default_s(float aspect, float size, mat4 dest)
{
#if PLAY_CGLM_CONFIG_CLIP_CONTROL == PLAY_CGLM_CLIP_CONTROL_LH_ZO
    ortho_default_s_lh_zo(aspect, size, dest);
#elif PLAY_CGLM_CONFIG_CLIP_CONTROL == PLAY_CGLM_CLIP_CONTROL_LH_NO
    ortho_default_s_lh_no(aspect, size, dest);
#elif PLAY_CGLM_CONFIG_CLIP_CONTROL == PLAY_CGLM_CLIP_CONTROL_RH_ZO
    ortho_default_s_rh_zo(aspect, size, dest);
#elif PLAY_CGLM_CONFIG_CLIP_CONTROL == PLAY_CGLM_CLIP_CONTROL_RH_NO
    ortho_default_s_rh_no(aspect, size, dest);
#endif
}

/*!
 * @brief set up perspective projection matrix
 *
 * @param[in]  fovy    field of view angle
 * @param[in]  aspect  aspect ratio ( width / height )
 * @param[in]  nearZ   near clipping plane
 * @param[in]  farZ    far clipping planes
 * @param[out] dest    result matrix
 */
PLAY_CGLM_INLINE
void
perspective(float fovy, float aspect, float nearZ, float farZ, mat4 dest)
{
#if PLAY_CGLM_CONFIG_CLIP_CONTROL == PLAY_CGLM_CLIP_CONTROL_LH_ZO
    perspective_lh_zo(fovy, aspect, nearZ, farZ, dest);
#elif PLAY_CGLM_CONFIG_CLIP_CONTROL == PLAY_CGLM_CLIP_CONTROL_LH_NO
    perspective_lh_no(fovy, aspect, nearZ, farZ, dest);
#elif PLAY_CGLM_CONFIG_CLIP_CONTROL == PLAY_CGLM_CLIP_CONTROL_RH_ZO
    perspective_rh_zo(fovy, aspect, nearZ, farZ, dest);
#elif PLAY_CGLM_CONFIG_CLIP_CONTROL == PLAY_CGLM_CLIP_CONTROL_RH_NO
    perspective_rh_no(fovy, aspect, nearZ, farZ, dest);
#endif
}

/*!
 * @brief extend perspective projection matrix's far distance
 *
 * this function does not guarantee far >= near, be aware of that!
 *
 * @param[in, out] proj      projection matrix to extend
 * @param[in]      deltaFar  distance from existing far (negative to shink)
 */
PLAY_CGLM_INLINE
void
persp_move_far(mat4 proj, float deltaFar)
{
#if PLAY_CGLM_CONFIG_CLIP_CONTROL == PLAY_CGLM_CLIP_CONTROL_LH_ZO
    persp_move_far_lh_zo(proj, deltaFar);
#elif PLAY_CGLM_CONFIG_CLIP_CONTROL == PLAY_CGLM_CLIP_CONTROL_LH_NO
    persp_move_far_lh_no(proj, deltaFar);
#elif PLAY_CGLM_CONFIG_CLIP_CONTROL == PLAY_CGLM_CLIP_CONTROL_RH_ZO
    persp_move_far_rh_zo(proj, deltaFar);
#elif PLAY_CGLM_CONFIG_CLIP_CONTROL == PLAY_CGLM_CLIP_CONTROL_RH_NO
    persp_move_far_rh_no(proj, deltaFar);
#endif
}

/*!
 * @brief set up perspective projection matrix with default near/far
 *        and angle values
 *
 * @param[in]  aspect aspect ratio ( width / height )
 * @param[out] dest   result matrix
 */
PLAY_CGLM_INLINE
void
perspective_default(float aspect, mat4 dest)
{
#if PLAY_CGLM_CONFIG_CLIP_CONTROL == PLAY_CGLM_CLIP_CONTROL_LH_ZO
    perspective_default_lh_zo(aspect, dest);
#elif PLAY_CGLM_CONFIG_CLIP_CONTROL == PLAY_CGLM_CLIP_CONTROL_LH_NO
    perspective_default_lh_no(aspect, dest);
#elif PLAY_CGLM_CONFIG_CLIP_CONTROL == PLAY_CGLM_CLIP_CONTROL_RH_ZO
    perspective_default_rh_zo(aspect, dest);
#elif PLAY_CGLM_CONFIG_CLIP_CONTROL == PLAY_CGLM_CLIP_CONTROL_RH_NO
    perspective_default_rh_no(aspect, dest);
#endif
}

/*!
 * @brief resize perspective matrix by aspect ratio ( width / height )
 *        this makes very easy to resize proj matrix when window /viewport
 *        reized
 *
 * @param[in]      aspect aspect ratio ( width / height )
 * @param[in, out] proj   perspective projection matrix
 */
PLAY_CGLM_INLINE
void
perspective_resize(float aspect, mat4 proj)
{
    if (proj[0][0] == 0.0f)
        return;

    proj[0][0] = proj[1][1] / aspect;
}

/*!
 * @brief set up view matrix
 *
 * NOTE: The UP vector must not be parallel to the line of sight from
 *       the eye point to the reference point
 *
 * @param[in]  eye    eye vector
 * @param[in]  center center vector
 * @param[in]  up     up vector
 * @param[out] dest   result matrix
 */
PLAY_CGLM_INLINE
void
lookat(vec3 eye, vec3 center, vec3 up, mat4 dest)
{
#if PLAY_CGLM_CONFIG_CLIP_CONTROL & PLAY_CGLM_CLIP_CONTROL_LH_BIT
    lookat_lh(eye, center, up, dest);
#elif PLAY_CGLM_CONFIG_CLIP_CONTROL & PLAY_CGLM_CLIP_CONTROL_RH_BIT
    lookat_rh(eye, center, up, dest);
#endif
}

/*!
 * @brief set up view matrix
 *
 * convenient wrapper for lookat: if you only have direction not target self
 * then this might be useful. Because you need to get target from direction.
 *
 * NOTE: The UP vector must not be parallel to the line of sight from
 *       the eye point to the reference point
 *
 * @param[in]  eye    eye vector
 * @param[in]  dir    direction vector
 * @param[in]  up     up vector
 * @param[out] dest   result matrix
 */
PLAY_CGLM_INLINE
void
look(vec3 eye, vec3 dir, vec3 up, mat4 dest)
{
#if PLAY_CGLM_CONFIG_CLIP_CONTROL & PLAY_CGLM_CLIP_CONTROL_LH_BIT
    look_lh(eye, dir, up, dest);
#elif PLAY_CGLM_CONFIG_CLIP_CONTROL & PLAY_CGLM_CLIP_CONTROL_RH_BIT
    look_rh(eye, dir, up, dest);
#endif
}

/*!
 * @brief set up view matrix
 *
 * convenient wrapper for look: if you only have direction and if you don't
 * care what UP vector is then this might be useful to create view matrix
 *
 * @param[in]  eye    eye vector
 * @param[in]  dir    direction vector
 * @param[out] dest   result matrix
 */
PLAY_CGLM_INLINE
void
look_anyup(vec3 eye, vec3 dir, mat4 dest)
{
#if PLAY_CGLM_CONFIG_CLIP_CONTROL & PLAY_CGLM_CLIP_CONTROL_LH_BIT
    look_anyup_lh(eye, dir, dest);
#elif PLAY_CGLM_CONFIG_CLIP_CONTROL & PLAY_CGLM_CLIP_CONTROL_RH_BIT
    look_anyup_rh(eye, dir, dest);
#endif
}

/*!
 * @brief decomposes frustum values of perspective projection.
 *
 * @param[in]  proj    perspective projection matrix
 * @param[out] nearZ   near
 * @param[out] farZ    far
 * @param[out] top     top
 * @param[out] bottom  bottom
 * @param[out] left    left
 * @param[out] right   right
 */
PLAY_CGLM_INLINE
void
persp_decomp(mat4 proj,
             float * __restrict nearZ, float * __restrict farZ,
             float * __restrict top,   float * __restrict bottom,
             float * __restrict left,  float * __restrict right)
{
#if PLAY_CGLM_CONFIG_CLIP_CONTROL == PLAY_CGLM_CLIP_CONTROL_LH_ZO
    persp_decomp_lh_zo(proj, nearZ, farZ, top, bottom, left, right);
#elif PLAY_CGLM_CONFIG_CLIP_CONTROL == PLAY_CGLM_CLIP_CONTROL_LH_NO
    persp_decomp_lh_no(proj, nearZ, farZ, top, bottom, left, right);
#elif PLAY_CGLM_CONFIG_CLIP_CONTROL == PLAY_CGLM_CLIP_CONTROL_RH_ZO
    persp_decomp_rh_zo(proj, nearZ, farZ, top, bottom, left, right);
#elif PLAY_CGLM_CONFIG_CLIP_CONTROL == PLAY_CGLM_CLIP_CONTROL_RH_NO
    persp_decomp_rh_no(proj, nearZ, farZ, top, bottom, left, right);
#endif
}

/*!
 * @brief decomposes frustum values of perspective projection.
 *        this makes easy to get all values at once
 *
 * @param[in]  proj   perspective projection matrix
 * @param[out] dest   array
 */
PLAY_CGLM_INLINE
void
persp_decompv(mat4 proj, float dest[6])
{
#if PLAY_CGLM_CONFIG_CLIP_CONTROL == PLAY_CGLM_CLIP_CONTROL_LH_ZO
    persp_decompv_lh_zo(proj, dest);
#elif PLAY_CGLM_CONFIG_CLIP_CONTROL == PLAY_CGLM_CLIP_CONTROL_LH_NO
    persp_decompv_lh_no(proj, dest);
#elif PLAY_CGLM_CONFIG_CLIP_CONTROL == PLAY_CGLM_CLIP_CONTROL_RH_ZO
    persp_decompv_rh_zo(proj, dest);
#elif PLAY_CGLM_CONFIG_CLIP_CONTROL == PLAY_CGLM_CLIP_CONTROL_RH_NO
    persp_decompv_rh_no(proj, dest);
#endif
}

/*!
 * @brief decomposes left and right values of perspective projection.
 *        x stands for x axis (left / right axis)
 *
 * @param[in]  proj  perspective projection matrix
 * @param[out] left  left
 * @param[out] right right
 */
PLAY_CGLM_INLINE
void
persp_decomp_x(mat4 proj,
               float * __restrict left,
               float * __restrict right)
{
#if PLAY_CGLM_CONFIG_CLIP_CONTROL == PLAY_CGLM_CLIP_CONTROL_LH_ZO
    persp_decomp_x_lh_zo(proj, left, right);
#elif PLAY_CGLM_CONFIG_CLIP_CONTROL == PLAY_CGLM_CLIP_CONTROL_LH_NO
    persp_decomp_x_lh_no(proj, left, right);
#elif PLAY_CGLM_CONFIG_CLIP_CONTROL == PLAY_CGLM_CLIP_CONTROL_RH_ZO
    persp_decomp_x_rh_zo(proj, left, right);
#elif PLAY_CGLM_CONFIG_CLIP_CONTROL == PLAY_CGLM_CLIP_CONTROL_RH_NO
    persp_decomp_x_rh_no(proj, left, right);
#endif
}

/*!
 * @brief decomposes top and bottom values of perspective projection.
 *        y stands for y axis (top / bottom axis)
 *
 * @param[in]  proj   perspective projection matrix
 * @param[out] top    top
 * @param[out] bottom bottom
 */
PLAY_CGLM_INLINE
void
persp_decomp_y(mat4 proj,
               float * __restrict top,
               float * __restrict bottom)
{
#if PLAY_CGLM_CONFIG_CLIP_CONTROL == PLAY_CGLM_CLIP_CONTROL_LH_ZO
    persp_decomp_y_lh_zo(proj, top, bottom);
#elif PLAY_CGLM_CONFIG_CLIP_CONTROL == PLAY_CGLM_CLIP_CONTROL_LH_NO
    persp_decomp_y_lh_no(proj, top, bottom);
#elif PLAY_CGLM_CONFIG_CLIP_CONTROL == PLAY_CGLM_CLIP_CONTROL_RH_ZO
    persp_decomp_y_rh_zo(proj, top, bottom);
#elif PLAY_CGLM_CONFIG_CLIP_CONTROL == PLAY_CGLM_CLIP_CONTROL_RH_NO
    persp_decomp_y_rh_no(proj, top, bottom);
#endif
}

/*!
 * @brief decomposes near and far values of perspective projection.
 *        z stands for z axis (near / far axis)
 *
 * @param[in]  proj    perspective projection matrix
 * @param[out] nearZ   near
 * @param[out] farZ    far
 */
PLAY_CGLM_INLINE
void
persp_decomp_z(mat4 proj, float * __restrict nearZ, float * __restrict farZ)
{
#if PLAY_CGLM_CONFIG_CLIP_CONTROL == PLAY_CGLM_CLIP_CONTROL_LH_ZO
    persp_decomp_z_lh_zo(proj, nearZ, farZ);
#elif PLAY_CGLM_CONFIG_CLIP_CONTROL == PLAY_CGLM_CLIP_CONTROL_LH_NO
    persp_decomp_z_lh_no(proj, nearZ, farZ);
#elif PLAY_CGLM_CONFIG_CLIP_CONTROL == PLAY_CGLM_CLIP_CONTROL_RH_ZO
    persp_decomp_z_rh_zo(proj, nearZ, farZ);
#elif PLAY_CGLM_CONFIG_CLIP_CONTROL == PLAY_CGLM_CLIP_CONTROL_RH_NO
    persp_decomp_z_rh_no(proj, nearZ, farZ);
#endif
}

/*!
 * @brief decomposes far value of perspective projection.
 *
 * @param[in]  proj   perspective projection matrix
 * @param[out] farZ   far
 */
PLAY_CGLM_INLINE
void
persp_decomp_far(mat4 proj, float * __restrict farZ)
{
#if PLAY_CGLM_CONFIG_CLIP_CONTROL == PLAY_CGLM_CLIP_CONTROL_LH_ZO
    persp_decomp_far_lh_zo(proj, farZ);
#elif PLAY_CGLM_CONFIG_CLIP_CONTROL == PLAY_CGLM_CLIP_CONTROL_LH_NO
    persp_decomp_far_lh_no(proj, farZ);
#elif PLAY_CGLM_CONFIG_CLIP_CONTROL == PLAY_CGLM_CLIP_CONTROL_RH_ZO
    persp_decomp_far_rh_zo(proj, farZ);
#elif PLAY_CGLM_CONFIG_CLIP_CONTROL == PLAY_CGLM_CLIP_CONTROL_RH_NO
    persp_decomp_far_rh_no(proj, farZ);
#endif
}

/*!
 * @brief decomposes near value of perspective projection.
 *
 * @param[in]  proj   perspective projection matrix
 * @param[out] nearZ  near
 */
PLAY_CGLM_INLINE
void
persp_decomp_near(mat4 proj, float * __restrict nearZ)
{
#if PLAY_CGLM_CONFIG_CLIP_CONTROL == PLAY_CGLM_CLIP_CONTROL_LH_ZO
    persp_decomp_near_lh_zo(proj, nearZ);
#elif PLAY_CGLM_CONFIG_CLIP_CONTROL == PLAY_CGLM_CLIP_CONTROL_LH_NO
    persp_decomp_near_lh_no(proj, nearZ);
#elif PLAY_CGLM_CONFIG_CLIP_CONTROL == PLAY_CGLM_CLIP_CONTROL_RH_ZO
    persp_decomp_near_rh_zo(proj, nearZ);
#elif PLAY_CGLM_CONFIG_CLIP_CONTROL == PLAY_CGLM_CLIP_CONTROL_RH_NO
    persp_decomp_near_rh_no(proj, nearZ);
#endif
}

/*!
 * @brief returns sizes of near and far planes of perspective projection
 *
 * @param[in]  proj perspective projection matrix
 * @param[in]  fovy fovy (see brief)
 * @param[out] dest sizes order: [Wnear, Hnear, Wfar, Hfar]
 */
PLAY_CGLM_INLINE
void
persp_sizes(mat4 proj, float fovy, vec4 dest)
{
#if PLAY_CGLM_CONFIG_CLIP_CONTROL == PLAY_CGLM_CLIP_CONTROL_LH_ZO
    persp_sizes_lh_zo(proj, fovy, dest);
#elif PLAY_CGLM_CONFIG_CLIP_CONTROL == PLAY_CGLM_CLIP_CONTROL_LH_NO
    persp_sizes_lh_no(proj, fovy, dest);
#elif PLAY_CGLM_CONFIG_CLIP_CONTROL == PLAY_CGLM_CLIP_CONTROL_RH_ZO
    persp_sizes_rh_zo(proj, fovy, dest);
#elif PLAY_CGLM_CONFIG_CLIP_CONTROL == PLAY_CGLM_CLIP_CONTROL_RH_NO
    persp_sizes_rh_no(proj, fovy, dest);
#endif
}



/*** End of inlined file: cam.h ***/


/*** Start of inlined file: frustum.h ***/



#define PLAY_CGLM_LBN 0 /* left  bottom near */
#define PLAY_CGLM_LTN 1 /* left  top    near */
#define PLAY_CGLM_RTN 2 /* right top    near */
#define PLAY_CGLM_RBN 3 /* right bottom near */

#define PLAY_CGLM_LBF 4 /* left  bottom far  */
#define PLAY_CGLM_LTF 5 /* left  top    far  */
#define PLAY_CGLM_RTF 6 /* right top    far  */
#define PLAY_CGLM_RBF 7 /* right bottom far  */

#define PLAY_CGLM_LEFT   0
#define PLAY_CGLM_RIGHT  1
#define PLAY_CGLM_BOTTOM 2
#define PLAY_CGLM_TOP    3
#define PLAY_CGLM_NEAR   4
#define PLAY_CGLM_FAR    5

/* you can override clip space coords
   but you have to provide all with same name
   e.g.: define PLAY_CGLM_CSCOORD_LBN {0.0f, 0.0f, 1.0f, 1.0f} */
#ifndef PLAY_CGLM_CUSTOM_CLIPSPACE

/* near */
#define PLAY_CGLM_CSCOORD_LBN {-1.0f, -1.0f, -1.0f, 1.0f}
#define PLAY_CGLM_CSCOORD_LTN {-1.0f,  1.0f, -1.0f, 1.0f}
#define PLAY_CGLM_CSCOORD_RTN { 1.0f,  1.0f, -1.0f, 1.0f}
#define PLAY_CGLM_CSCOORD_RBN { 1.0f, -1.0f, -1.0f, 1.0f}

/* far */
#define PLAY_CGLM_CSCOORD_LBF {-1.0f, -1.0f,  1.0f, 1.0f}
#define PLAY_CGLM_CSCOORD_LTF {-1.0f,  1.0f,  1.0f, 1.0f}
#define PLAY_CGLM_CSCOORD_RTF { 1.0f,  1.0f,  1.0f, 1.0f}
#define PLAY_CGLM_CSCOORD_RBF { 1.0f, -1.0f,  1.0f, 1.0f}

#endif

/*!
 * @brief extracts view frustum planes
 *
 * planes' space:
 *  1- if m = proj:     View Space
 *  2- if m = viewProj: World Space
 *  3- if m = MVP:      Object Space
 *
 * You probably want to extract planes in world space so use viewProj as m
 * Computing viewProj:
 *   mat4_mul(proj, view, viewProj);
 *
 * Exracted planes order: [left, right, bottom, top, near, far]
 *
 * @param[in]  m    matrix (see brief)
 * @param[out] dest extracted view frustum planes (see brief)
 */
PLAY_CGLM_INLINE
void
frustum_planes(mat4 m, vec4 dest[6])
{
    mat4 t;

    mat4_transpose_to(m, t);

    vec4_add(t[3], t[0], dest[0]); /* left   */
    vec4_sub(t[3], t[0], dest[1]); /* right  */
    vec4_add(t[3], t[1], dest[2]); /* bottom */
    vec4_sub(t[3], t[1], dest[3]); /* top    */
    vec4_add(t[3], t[2], dest[4]); /* near   */
    vec4_sub(t[3], t[2], dest[5]); /* far    */

    plane_normalize(dest[0]);
    plane_normalize(dest[1]);
    plane_normalize(dest[2]);
    plane_normalize(dest[3]);
    plane_normalize(dest[4]);
    plane_normalize(dest[5]);
}

/*!
 * @brief extracts view frustum corners using clip-space coordinates
 *
 * corners' space:
 *  1- if m = invViewProj: World Space
 *  2- if m = invMVP:      Object Space
 *
 * You probably want to extract corners in world space so use invViewProj
 * Computing invViewProj:
 *   mat4_mul(proj, view, viewProj);
 *   ...
 *   mat4_inv(viewProj, invViewProj);
 *
 * if you have a near coord at i index, you can get it's far coord by i + 4
 *
 * Find center coordinates:
 *   for (j = 0; j < 4; j++) {
 *     vec3_center(corners[i], corners[i + 4], centerCorners[i]);
 *   }
 *
 * @param[in]  invMat matrix (see brief)
 * @param[out] dest   exracted view frustum corners (see brief)
 */
PLAY_CGLM_INLINE
void
frustum_corners(mat4 invMat, vec4 dest[8])
{
    vec4 c[8];

    /* indexOf(nearCoord) = indexOf(farCoord) + 4 */
    vec4 csCoords[8] =
    {
        PLAY_CGLM_CSCOORD_LBN,
        PLAY_CGLM_CSCOORD_LTN,
        PLAY_CGLM_CSCOORD_RTN,
        PLAY_CGLM_CSCOORD_RBN,

        PLAY_CGLM_CSCOORD_LBF,
        PLAY_CGLM_CSCOORD_LTF,
        PLAY_CGLM_CSCOORD_RTF,
        PLAY_CGLM_CSCOORD_RBF
    };

    mat4_mulv(invMat, csCoords[0], c[0]);
    mat4_mulv(invMat, csCoords[1], c[1]);
    mat4_mulv(invMat, csCoords[2], c[2]);
    mat4_mulv(invMat, csCoords[3], c[3]);
    mat4_mulv(invMat, csCoords[4], c[4]);
    mat4_mulv(invMat, csCoords[5], c[5]);
    mat4_mulv(invMat, csCoords[6], c[6]);
    mat4_mulv(invMat, csCoords[7], c[7]);

    vec4_scale(c[0], 1.0f / c[0][3], dest[0]);
    vec4_scale(c[1], 1.0f / c[1][3], dest[1]);
    vec4_scale(c[2], 1.0f / c[2][3], dest[2]);
    vec4_scale(c[3], 1.0f / c[3][3], dest[3]);
    vec4_scale(c[4], 1.0f / c[4][3], dest[4]);
    vec4_scale(c[5], 1.0f / c[5][3], dest[5]);
    vec4_scale(c[6], 1.0f / c[6][3], dest[6]);
    vec4_scale(c[7], 1.0f / c[7][3], dest[7]);
}

/*!
 * @brief finds center of view frustum
 *
 * @param[in]  corners view frustum corners
 * @param[out] dest    view frustum center
 */
PLAY_CGLM_INLINE
void
frustum_center(vec4 corners[8], vec4 dest)
{
    vec4 center;

    vec4_copy(corners[0], center);

    vec4_add(corners[1], center, center);
    vec4_add(corners[2], center, center);
    vec4_add(corners[3], center, center);
    vec4_add(corners[4], center, center);
    vec4_add(corners[5], center, center);
    vec4_add(corners[6], center, center);
    vec4_add(corners[7], center, center);

    vec4_scale(center, 0.125f, dest);
}

/*!
 * @brief finds bounding box of frustum relative to given matrix e.g. view mat
 *
 * @param[in]  corners view frustum corners
 * @param[in]  m       matrix to convert existing conners
 * @param[out] box     bounding box as array [min, max]
 */
PLAY_CGLM_INLINE
void
frustum_box(vec4 corners[8], mat4 m, vec3 box[2])
{
    vec4 v;
    vec3 min, max;
    int  i;

    vec3_broadcast(FLT_MAX, min);
    vec3_broadcast(-FLT_MAX, max);

    for (i = 0; i < 8; i++)
    {
        mat4_mulv(m, corners[i], v);

        min[0] = fmin(min[0], v[0]);
        min[1] = fmin(min[1], v[1]);
        min[2] = fmin(min[2], v[2]);

        max[0] = fmax(max[0], v[0]);
        max[1] = fmax(max[1], v[1]);
        max[2] = fmax(max[2], v[2]);
    }

    vec3_copy(min, box[0]);
    vec3_copy(max, box[1]);
}

/*!
 * @brief finds planes corners which is between near and far planes (parallel)
 *
 * this will be helpful if you want to split a frustum e.g. CSM/PSSM. This will
 * find planes' corners but you will need to one more plane.
 * Actually you have it, it is near, far or created previously with this func ;)
 *
 * @param[in]  corners view  frustum corners
 * @param[in]  splitDist     split distance
 * @param[in]  farDist       far distance (zFar)
 * @param[out] planeCorners  plane corners [LB, LT, RT, RB]
 */
PLAY_CGLM_INLINE
void
frustum_corners_at(vec4  corners[8],
                   float splitDist,
                   float farDist,
                   vec4  planeCorners[4])
{
    vec4  corner;
    float dist, sc;

    /* because distance and scale is same for all */
    dist = vec3_distance(corners[PLAY_CGLM_RTF], corners[PLAY_CGLM_RTN]);
    sc   = dist * (splitDist / farDist);

    /* left bottom */
    vec4_sub(corners[PLAY_CGLM_LBF], corners[PLAY_CGLM_LBN], corner);
    vec4_scale_as(corner, sc, corner);
    vec4_add(corners[PLAY_CGLM_LBN], corner, planeCorners[0]);

    /* left top */
    vec4_sub(corners[PLAY_CGLM_LTF], corners[PLAY_CGLM_LTN], corner);
    vec4_scale_as(corner, sc, corner);
    vec4_add(corners[PLAY_CGLM_LTN], corner, planeCorners[1]);

    /* right top */
    vec4_sub(corners[PLAY_CGLM_RTF], corners[PLAY_CGLM_RTN], corner);
    vec4_scale_as(corner, sc, corner);
    vec4_add(corners[PLAY_CGLM_RTN], corner, planeCorners[2]);

    /* right bottom */
    vec4_sub(corners[PLAY_CGLM_RBF], corners[PLAY_CGLM_RBN], corner);
    vec4_scale_as(corner, sc, corner);
    vec4_add(corners[PLAY_CGLM_RBN], corner, planeCorners[3]);
}



/*** End of inlined file: frustum.h ***/


/*** Start of inlined file: quat.h ***/
/*
 Macros:
   PLAY_CGLM_QUAT_IDENTITY_INIT
   PLAY_CGLM_QUAT_IDENTITY

 Functions:
   PLAY_CGLM_INLINE void quat_identity(versor q);
   PLAY_CGLM_INLINE void quat_init(versor q, float x, float y, float z, float w);
   PLAY_CGLM_INLINE void quat(versor q, float angle, float x, float y, float z);
   PLAY_CGLM_INLINE void quatv(versor q, float angle, vec3 axis);
   PLAY_CGLM_INLINE void quat_copy(versor q, versor dest);
   PLAY_CGLM_INLINE void quat_from_vecs(vec3 a, vec3 b, versor dest);
   PLAY_CGLM_INLINE float quat_norm(versor q);
   PLAY_CGLM_INLINE void quat_normalize(versor q);
   PLAY_CGLM_INLINE void quat_normalize_to(versor q, versor dest);
   PLAY_CGLM_INLINE float quat_dot(versor p, versor q);
   PLAY_CGLM_INLINE void quat_conjugate(versor q, versor dest);
   PLAY_CGLM_INLINE void quat_inv(versor q, versor dest);
   PLAY_CGLM_INLINE void quat_add(versor p, versor q, versor dest);
   PLAY_CGLM_INLINE void quat_sub(versor p, versor q, versor dest);
   PLAY_CGLM_INLINE float quat_real(versor q);
   PLAY_CGLM_INLINE void quat_imag(versor q, vec3 dest);
   PLAY_CGLM_INLINE void quat_imagn(versor q, vec3 dest);
   PLAY_CGLM_INLINE float quat_imaglen(versor q);
   PLAY_CGLM_INLINE float quat_angle(versor q);
   PLAY_CGLM_INLINE void quat_axis(versor q, vec3 dest);
   PLAY_CGLM_INLINE void quat_mul(versor p, versor q, versor dest);
   PLAY_CGLM_INLINE void quat_mat4(versor q, mat4 dest);
   PLAY_CGLM_INLINE void quat_mat4t(versor q, mat4 dest);
   PLAY_CGLM_INLINE void quat_mat3(versor q, mat3 dest);
   PLAY_CGLM_INLINE void quat_mat3t(versor q, mat3 dest);
   PLAY_CGLM_INLINE void quat_lerp(versor from, versor to, float t, versor dest);
   PLAY_CGLM_INLINE void quat_lerpc(versor from, versor to, float t, versor dest);
   PLAY_CGLM_INLINE void quat_slerp(versor q, versor r, float t, versor dest);
   PLAY_CGLM_INLINE void quat_slerp_longest(versor q, versor r, float t, versor dest);
   PLAY_CGLM_INLINE void quat_nlerp(versor q, versor r, float t, versor dest);
   PLAY_CGLM_INLINE void quat_look(vec3 eye, versor ori, mat4 dest);
   PLAY_CGLM_INLINE void quat_for(vec3 dir, vec3 fwd, vec3 up, versor dest);
   PLAY_CGLM_INLINE void quat_forp(vec3 from,
                                  vec3 to,
                                  vec3 fwd,
                                  vec3 up,
                                  versor dest);
   PLAY_CGLM_INLINE void quat_rotatev(versor q, vec3 v, vec3 dest);
   PLAY_CGLM_INLINE void quat_rotate(mat4 m, versor q, mat4 dest);
   PLAY_CGLM_INLINE void quat_make(float * restrict src, versor dest);
 */




#ifdef PLAY_CGLM_SSE_FP

/*** Start of inlined file: quat.h ***/


#if defined( __SSE__ ) || defined( __SSE2__ )

PLAY_CGLM_INLINE
void
quat_mul_sse2(versor p, versor q, versor dest)
{
    /*
     + (a1 b2 + b1 a2 + c1 d2 − d1 c2)i
     + (a1 c2 − b1 d2 + c1 a2 + d1 b2)j
     + (a1 d2 + b1 c2 − c1 b2 + d1 a2)k
       a1 a2 − b1 b2 − c1 c2 − d1 d2
     */

    __m128 xp, xq, x1, x2, x3, r, x, y, z;

    xp = glmm_load(p); /* 3 2 1 0 */
    xq = glmm_load(q);
    x1 = glmm_float32x4_SIGNMASK_NPNP; /* TODO: _mm_set1_ss() + shuff ? */
    r  = _mm_mul_ps(glmm_splat_w(xp), xq);

    x2 = _mm_unpackhi_ps(x1, x1);
    x3 = glmm_shuff1(x1, 3, 2, 0, 1);
    x  = glmm_splat_x(xp);
    y  = glmm_splat_y(xp);
    z  = glmm_splat_z(xp);

    x  = _mm_xor_ps(x, x1);
    y  = _mm_xor_ps(y, x2);
    z  = _mm_xor_ps(z, x3);

    x1 = glmm_shuff1(xq, 0, 1, 2, 3);
    x2 = glmm_shuff1(xq, 1, 0, 3, 2);
    x3 = glmm_shuff1(xq, 2, 3, 0, 1);

    r  = glmm_fmadd(x, x1, r);
    r  = glmm_fmadd(y, x2, r);
    r  = glmm_fmadd(z, x3, r);

    glmm_store(dest, r);
}

#endif


/*** End of inlined file: quat.h ***/


#endif

#ifdef PLAY_CGLM_NEON_FP

/*** Start of inlined file: quat.h ***/


#if defined(PLAY_CGLM_NEON_FP)

PLAY_CGLM_INLINE
void
quat_mul_neon(versor p, versor q, versor dest)
{
    /*
     + (a1 b2 + b1 a2 + c1 d2 − d1 c2)i
     + (a1 c2 − b1 d2 + c1 a2 + d1 b2)j
     + (a1 d2 + b1 c2 − c1 b2 + d1 a2)k
       a1 a2 − b1 b2 − c1 c2 − d1 d2
     */

    glmm_128 xp, xq, xqr, r, x, y, z, s2, s3;
    glmm_128 s1 = glmm_float32x4_SIGNMASK_NPPN;

    float32x2_t   qh, ql;

    xp  = glmm_load(p); /* 3 2 1 0 */
    xq  = glmm_load(q);

    r   = vmulq_f32(glmm_splat_w(xp), xq);
    x   = glmm_splat_x(xp);
    y   = glmm_splat_y(xp);
    z   = glmm_splat_z(xp);

    ql  = vget_high_f32(s1);
    s3  = vcombine_f32(ql, ql);
    s2  = vzipq_f32(s3, s3).val[0];

    xqr = vrev64q_f32(xq);
    qh  = vget_high_f32(xqr);
    ql  = vget_low_f32(xqr);

    r = glmm_fmadd(glmm_xor(x, s3), vcombine_f32(qh, ql), r);

    r = glmm_fmadd(glmm_xor(y, s2), vcombine_f32(vget_high_f32(xq),
                   vget_low_f32(xq)), r);

    r = glmm_fmadd(glmm_xor(z, s1), vcombine_f32(ql, qh), r);

    glmm_store(dest, r);
}

#endif


/*** End of inlined file: quat.h ***/


#endif

#ifdef PLAY_CGLM_SIMD_WASM

/*** Start of inlined file: quat.h ***/


#if defined(__wasm__) && defined(__wasm_simd128__)

PLAY_CGLM_INLINE
void
quat_mul_wasm(versor p, versor q, versor dest)
{
    /*
     + (a1 b2 + b1 a2 + c1 d2 − d1 c2)i
     + (a1 c2 − b1 d2 + c1 a2 + d1 b2)j
     + (a1 d2 + b1 c2 − c1 b2 + d1 a2)k
       a1 a2 − b1 b2 − c1 c2 − d1 d2
     */

    glmm_128 xp, xq, x1, x2, x3, r, x, y, z;

    xp = glmm_load(p); /* 3 2 1 0 */
    xq = glmm_load(q);
    /* x1 = wasm_f32x4_const(0.f, -0.f, 0.f, -0.f); */
    x1 = glmm_float32x4_SIGNMASK_PNPN; /* TODO: _mm_set1_ss() + shuff ? */
    r  = wasm_f32x4_mul(glmm_splat_w(xp), xq);
    /* x2 = _mm_unpackhi_ps(x1, x1); */
    x2 = wasm_i32x4_shuffle(x1, x1, 2, 6, 3, 7);
    x3 = glmm_shuff1(x1, 3, 2, 0, 1);
    x  = glmm_splat_x(xp);
    y  = glmm_splat_y(xp);
    z  = glmm_splat_z(xp);

    x  = wasm_v128_xor(x, x1);
    y  = wasm_v128_xor(y, x2);
    z  = wasm_v128_xor(z, x3);

    x1 = glmm_shuff1(xq, 0, 1, 2, 3);
    x2 = glmm_shuff1(xq, 1, 0, 3, 2);
    x3 = glmm_shuff1(xq, 2, 3, 0, 1);

    r  = glmm_fmadd(x, x1, r);
    r  = glmm_fmadd(y, x2, r);
    r  = glmm_fmadd(z, x3, r);

    glmm_store(dest, r);
}

#endif


/*** End of inlined file: quat.h ***/


#endif

PLAY_CGLM_INLINE void quat_normalize(versor q);

/*
 * IMPORTANT:
 * ----------------------------------------------------------------------------
 * cglm stores quat as [x, y, z, w] since v0.3.6
 *
 * it was [w, x, y, z] before v0.3.6 it has been changed to [x, y, z, w]
 * with v0.3.6 version.
 * ----------------------------------------------------------------------------
 */

#define PLAY_CGLM_QUAT_IDENTITY_INIT  {0.0f, 0.0f, 0.0f, 1.0f}
#define PLAY_CGLM_QUAT_IDENTITY       ((versor)PLAY_CGLM_QUAT_IDENTITY_INIT)

/*!
 * @brief makes given quat to identity
 *
 * @param[in, out]  q  quaternion
 */
PLAY_CGLM_INLINE
void
quat_identity(versor q)
{
    PLAY_CGLM_ALIGN(16) versor v = PLAY_CGLM_QUAT_IDENTITY_INIT;
    vec4_copy(v, q);
}

/*!
 * @brief make given quaternion array's each element identity quaternion
 *
 * @param[in, out]  q     quat array (must be aligned (16)
 *                        if alignment is not disabled)
 *
 * @param[in]       count count of quaternions
 */
PLAY_CGLM_INLINE
void
quat_identity_array(versor * __restrict q, size_t count)
{
    PLAY_CGLM_ALIGN(16) versor v = PLAY_CGLM_QUAT_IDENTITY_INIT;
    size_t i;

    for (i = 0; i < count; i++)
    {
        vec4_copy(v, q[i]);
    }
}

/*!
 * @brief inits quaternion with raw values
 *
 * @param[out]  q     quaternion
 * @param[in]   x     x
 * @param[in]   y     y
 * @param[in]   z     z
 * @param[in]   w     w (real part)
 */
PLAY_CGLM_INLINE
void
quat_init(versor q, float x, float y, float z, float w)
{
    q[0] = x;
    q[1] = y;
    q[2] = z;
    q[3] = w;
}

/*!
 * @brief creates NEW quaternion with axis vector
 *
 * @param[out]  q     quaternion
 * @param[in]   angle angle (radians)
 * @param[in]   axis  axis
 */
PLAY_CGLM_INLINE
void
quatv(versor q, float angle, vec3 axis)
{
    PLAY_CGLM_ALIGN(8) vec3 k;
    float a, c, s;

    a = angle * 0.5f;
    c = cosf(a);
    s = sinf(a);

    normalize_to(axis, k);

    q[0] = s * k[0];
    q[1] = s * k[1];
    q[2] = s * k[2];
    q[3] = c;
}

/*!
 * @brief creates NEW quaternion with individual axis components
 *
 * @param[out]  q     quaternion
 * @param[in]   angle angle (radians)
 * @param[in]   x     axis.x
 * @param[in]   y     axis.y
 * @param[in]   z     axis.z
 */
PLAY_CGLM_INLINE
void
quat(versor q, float angle, float x, float y, float z)
{
    PLAY_CGLM_ALIGN(8) vec3 axis = {x, y, z};
    quatv(q, angle, axis);
}

/*!
 * @brief copy quaternion to another one
 *
 * @param[in]  q     quaternion
 * @param[out] dest  destination
 */
PLAY_CGLM_INLINE
void
quat_copy(versor q, versor dest)
{
    vec4_copy(q, dest);
}

/*!
 * @brief compute quaternion rotating vector A to vector B
 *
 * @param[in]   a     vec3 (must have unit length)
 * @param[in]   b     vec3 (must have unit length)
 * @param[out]  dest  quaternion (of unit length)
 */
PLAY_CGLM_INLINE
void
quat_from_vecs(vec3 a, vec3 b, versor dest)
{
    PLAY_CGLM_ALIGN(8) vec3 axis;
    float cos_theta;
    float cos_half_theta;

    cos_theta = vec3_dot(a, b);
    if (cos_theta >= 1.f - PLAY_CGLM_FLT_EPSILON)    /*  a ∥ b  */
    {
        quat_identity(dest);
        return;
    }
    if (cos_theta < -1.f + PLAY_CGLM_FLT_EPSILON)    /*  angle(a, b) = π  */
    {
        vec3_ortho(a, axis);
        cos_half_theta = 0.f;                    /*  cos π/2 */
    }
    else
    {
        vec3_cross(a, b, axis);
        cos_half_theta = 1.0f + cos_theta;       /*  cos 0 + cos θ  */
    }

    quat_init(dest, axis[0], axis[1], axis[2], cos_half_theta);
    quat_normalize(dest);
}

/*!
 * @brief returns norm (magnitude) of quaternion
 *
 * @param[in]  q  quaternion
 */
PLAY_CGLM_INLINE
float
quat_norm(versor q)
{
    return vec4_norm(q);
}

/*!
 * @brief normalize quaternion and store result in dest
 *
 * @param[in]   q     quaternion to normalze
 * @param[out]  dest  destination quaternion
 */
PLAY_CGLM_INLINE
void
quat_normalize_to(versor q, versor dest)
{
#if defined(__wasm__) && defined(__wasm_simd128__)
    glmm_128 xdot, x0;
    float  dot;

    x0   = glmm_load(q);
    xdot = glmm_vdot(x0, x0);
    /* dot  = _mm_cvtss_f32(xdot); */
    dot  = wasm_f32x4_extract_lane(xdot, 0);

    if (dot <= 0.0f)
    {
        quat_identity(dest);
        return;
    }

    glmm_store(dest, wasm_f32x4_div(x0, wasm_f32x4_sqrt(xdot)));
#elif defined( __SSE__ ) || defined( __SSE2__ )
    __m128 xdot, x0;
    float  dot;

    x0   = glmm_load(q);
    xdot = glmm_vdot(x0, x0);
    dot  = _mm_cvtss_f32(xdot);

    if (dot <= 0.0f)
    {
        quat_identity(dest);
        return;
    }

    glmm_store(dest, _mm_div_ps(x0, _mm_sqrt_ps(xdot)));
#else
    float dot;

    dot = vec4_norm2(q);

    if (dot <= 0.0f)
    {
        quat_identity(dest);
        return;
    }

    vec4_scale(q, 1.0f / sqrtf(dot), dest);
#endif
}

/*!
 * @brief normalize quaternion
 *
 * @param[in, out]  q  quaternion
 */
PLAY_CGLM_INLINE
void
quat_normalize(versor q)
{
    quat_normalize_to(q, q);
}

/*!
 * @brief dot product of two quaternion
 *
 * @param[in]  p  quaternion 1
 * @param[in]  q  quaternion 2
 */
PLAY_CGLM_INLINE
float
quat_dot(versor p, versor q)
{
    return vec4_dot(p, q);
}

/*!
 * @brief conjugate of quaternion
 *
 * @param[in]   q     quaternion
 * @param[out]  dest  conjugate
 */
PLAY_CGLM_INLINE
void
quat_conjugate(versor q, versor dest)
{
    vec4_negate_to(q, dest);
    dest[3] = -dest[3];
}

/*!
 * @brief inverse of non-zero quaternion
 *
 * @param[in]   q    quaternion
 * @param[out]  dest inverse quaternion
 */
PLAY_CGLM_INLINE
void
quat_inv(versor q, versor dest)
{
    PLAY_CGLM_ALIGN(16) versor conj;
    quat_conjugate(q, conj);
    vec4_scale(conj, 1.0f / vec4_norm2(q), dest);
}

/*!
 * @brief add (componentwise) two quaternions and store result in dest
 *
 * @param[in]   p    quaternion 1
 * @param[in]   q    quaternion 2
 * @param[out]  dest result quaternion
 */
PLAY_CGLM_INLINE
void
quat_add(versor p, versor q, versor dest)
{
    vec4_add(p, q, dest);
}

/*!
 * @brief subtract (componentwise) two quaternions and store result in dest
 *
 * @param[in]   p    quaternion 1
 * @param[in]   q    quaternion 2
 * @param[out]  dest result quaternion
 */
PLAY_CGLM_INLINE
void
quat_sub(versor p, versor q, versor dest)
{
    vec4_sub(p, q, dest);
}

/*!
 * @brief returns real part of quaternion
 *
 * @param[in]   q    quaternion
 */
PLAY_CGLM_INLINE
float
quat_real(versor q)
{
    return q[3];
}

/*!
 * @brief returns imaginary part of quaternion
 *
 * @param[in]   q    quaternion
 * @param[out]  dest imag
 */
PLAY_CGLM_INLINE
void
quat_imag(versor q, vec3 dest)
{
    dest[0] = q[0];
    dest[1] = q[1];
    dest[2] = q[2];
}

/*!
 * @brief returns normalized imaginary part of quaternion
 *
 * @param[in]   q    quaternion
 */
PLAY_CGLM_INLINE
void
quat_imagn(versor q, vec3 dest)
{
    normalize_to(q, dest);
}

/*!
 * @brief returns length of imaginary part of quaternion
 *
 * @param[in]   q    quaternion
 */
PLAY_CGLM_INLINE
float
quat_imaglen(versor q)
{
    return vec3_norm(q);
}

/*!
 * @brief returns angle of quaternion
 *
 * @param[in]   q    quaternion
 */
PLAY_CGLM_INLINE
float
quat_angle(versor q)
{
    /*
     sin(theta / 2) = length(x*x + y*y + z*z)
     cos(theta / 2) = w
     theta          = 2 * atan(sin(theta / 2) / cos(theta / 2))
     */
    return 2.0f * atan2f(quat_imaglen(q), quat_real(q));
}

/*!
 * @brief axis of quaternion
 *
 * @param[in]   q    quaternion
 * @param[out]  dest axis of quaternion
 */
PLAY_CGLM_INLINE
void
quat_axis(versor q, vec3 dest)
{
    quat_imagn(q, dest);
}

/*!
 * @brief multiplies two quaternion and stores result in dest
 *        this is also called Hamilton Product
 *
 * According to WikiPedia:
 * The product of two rotation quaternions [clarification needed] will be
 * equivalent to the rotation q followed by the rotation p
 *
 * @param[in]   p     quaternion 1
 * @param[in]   q     quaternion 2
 * @param[out]  dest  result quaternion
 */
PLAY_CGLM_INLINE
void
quat_mul(versor p, versor q, versor dest)
{
    /*
      + (a1 b2 + b1 a2 + c1 d2 − d1 c2)i
      + (a1 c2 − b1 d2 + c1 a2 + d1 b2)j
      + (a1 d2 + b1 c2 − c1 b2 + d1 a2)k
         a1 a2 − b1 b2 − c1 c2 − d1 d2
     */
#if defined(__wasm__) && defined(__wasm_simd128__)
    quat_mul_wasm(p, q, dest);
#elif defined( __SSE__ ) || defined( __SSE2__ )
    quat_mul_sse2(p, q, dest);
#elif defined(PLAY_CGLM_NEON_FP)
    quat_mul_neon(p, q, dest);
#else
    dest[0] = p[3] * q[0] + p[0] * q[3] + p[1] * q[2] - p[2] * q[1];
    dest[1] = p[3] * q[1] - p[0] * q[2] + p[1] * q[3] + p[2] * q[0];
    dest[2] = p[3] * q[2] + p[0] * q[1] - p[1] * q[0] + p[2] * q[3];
    dest[3] = p[3] * q[3] - p[0] * q[0] - p[1] * q[1] - p[2] * q[2];
#endif
}

/*!
 * @brief convert quaternion to mat4
 *
 * @param[in]   q     quaternion
 * @param[out]  dest  result matrix
 */
PLAY_CGLM_INLINE
void
quat_mat4(versor q, mat4 dest)
{
    float w, x, y, z,
          xx, yy, zz,
          xy, yz, xz,
          wx, wy, wz, norm, s;

    norm = quat_norm(q);
    s    = norm > 0.0f ? 2.0f / norm : 0.0f;

    x = q[0];
    y = q[1];
    z = q[2];
    w = q[3];

    xx = s * x * x;
    xy = s * x * y;
    wx = s * w * x;
    yy = s * y * y;
    yz = s * y * z;
    wy = s * w * y;
    zz = s * z * z;
    xz = s * x * z;
    wz = s * w * z;

    dest[0][0] = 1.0f - yy - zz;
    dest[1][1] = 1.0f - xx - zz;
    dest[2][2] = 1.0f - xx - yy;

    dest[0][1] = xy + wz;
    dest[1][2] = yz + wx;
    dest[2][0] = xz + wy;

    dest[1][0] = xy - wz;
    dest[2][1] = yz - wx;
    dest[0][2] = xz - wy;

    dest[0][3] = 0.0f;
    dest[1][3] = 0.0f;
    dest[2][3] = 0.0f;
    dest[3][0] = 0.0f;
    dest[3][1] = 0.0f;
    dest[3][2] = 0.0f;
    dest[3][3] = 1.0f;
}

/*!
 * @brief convert quaternion to mat4 (transposed)
 *
 * @param[in]   q     quaternion
 * @param[out]  dest  result matrix as transposed
 */
PLAY_CGLM_INLINE
void
quat_mat4t(versor q, mat4 dest)
{
    float w, x, y, z,
          xx, yy, zz,
          xy, yz, xz,
          wx, wy, wz, norm, s;

    norm = quat_norm(q);
    s    = norm > 0.0f ? 2.0f / norm : 0.0f;

    x = q[0];
    y = q[1];
    z = q[2];
    w = q[3];

    xx = s * x * x;
    xy = s * x * y;
    wx = s * w * x;
    yy = s * y * y;
    yz = s * y * z;
    wy = s * w * y;
    zz = s * z * z;
    xz = s * x * z;
    wz = s * w * z;

    dest[0][0] = 1.0f - yy - zz;
    dest[1][1] = 1.0f - xx - zz;
    dest[2][2] = 1.0f - xx - yy;

    dest[1][0] = xy + wz;
    dest[2][1] = yz + wx;
    dest[0][2] = xz + wy;

    dest[0][1] = xy - wz;
    dest[1][2] = yz - wx;
    dest[2][0] = xz - wy;

    dest[0][3] = 0.0f;
    dest[1][3] = 0.0f;
    dest[2][3] = 0.0f;
    dest[3][0] = 0.0f;
    dest[3][1] = 0.0f;
    dest[3][2] = 0.0f;
    dest[3][3] = 1.0f;
}

/*!
 * @brief convert quaternion to mat3
 *
 * @param[in]   q     quaternion
 * @param[out]  dest  result matrix
 */
PLAY_CGLM_INLINE
void
quat_mat3(versor q, mat3 dest)
{
    float w, x, y, z,
          xx, yy, zz,
          xy, yz, xz,
          wx, wy, wz, norm, s;

    norm = quat_norm(q);
    s    = norm > 0.0f ? 2.0f / norm : 0.0f;

    x = q[0];
    y = q[1];
    z = q[2];
    w = q[3];

    xx = s * x * x;
    xy = s * x * y;
    wx = s * w * x;
    yy = s * y * y;
    yz = s * y * z;
    wy = s * w * y;
    zz = s * z * z;
    xz = s * x * z;
    wz = s * w * z;

    dest[0][0] = 1.0f - yy - zz;
    dest[1][1] = 1.0f - xx - zz;
    dest[2][2] = 1.0f - xx - yy;

    dest[0][1] = xy + wz;
    dest[1][2] = yz + wx;
    dest[2][0] = xz + wy;

    dest[1][0] = xy - wz;
    dest[2][1] = yz - wx;
    dest[0][2] = xz - wy;
}

/*!
 * @brief convert quaternion to mat3 (transposed)
 *
 * @param[in]   q     quaternion
 * @param[out]  dest  result matrix
 */
PLAY_CGLM_INLINE
void
quat_mat3t(versor q, mat3 dest)
{
    float w, x, y, z,
          xx, yy, zz,
          xy, yz, xz,
          wx, wy, wz, norm, s;

    norm = quat_norm(q);
    s    = norm > 0.0f ? 2.0f / norm : 0.0f;

    x = q[0];
    y = q[1];
    z = q[2];
    w = q[3];

    xx = s * x * x;
    xy = s * x * y;
    wx = s * w * x;
    yy = s * y * y;
    yz = s * y * z;
    wy = s * w * y;
    zz = s * z * z;
    xz = s * x * z;
    wz = s * w * z;

    dest[0][0] = 1.0f - yy - zz;
    dest[1][1] = 1.0f - xx - zz;
    dest[2][2] = 1.0f - xx - yy;

    dest[1][0] = xy + wz;
    dest[2][1] = yz + wx;
    dest[0][2] = xz + wy;

    dest[0][1] = xy - wz;
    dest[1][2] = yz - wx;
    dest[2][0] = xz - wy;
}

/*!
 * @brief interpolates between two quaternions
 *        using linear interpolation (LERP)
 *
 * @param[in]   from  from
 * @param[in]   to    to
 * @param[in]   t     interpolant (amount)
 * @param[out]  dest  result quaternion
 */
PLAY_CGLM_INLINE
void
quat_lerp(versor from, versor to, float t, versor dest)
{
    vec4_lerp(from, to, t, dest);
}

/*!
 * @brief interpolates between two quaternions
 *        using linear interpolation (LERP)
 *
 * @param[in]   from  from
 * @param[in]   to    to
 * @param[in]   t     interpolant (amount) clamped between 0 and 1
 * @param[out]  dest  result quaternion
 */
PLAY_CGLM_INLINE
void
quat_lerpc(versor from, versor to, float t, versor dest)
{
    vec4_lerpc(from, to, t, dest);
}

/*!
 * @brief interpolates between two quaternions
 *        taking the shortest rotation path using
 *        normalized linear interpolation (NLERP)
 *
 * @param[in]   from  from
 * @param[in]   to    to
 * @param[in]   t     interpolant (amount)
 * @param[out]  dest  result quaternion
 */
PLAY_CGLM_INLINE
void
quat_nlerp(versor from, versor to, float t, versor dest)
{
    versor target;
    float  dot;

    dot = vec4_dot(from, to);

    vec4_scale(to, (dot >= 0) ? 1.0f : -1.0f, target);
    quat_lerp(from, target, t, dest);
    quat_normalize(dest);
}

/*!
 * @brief interpolates between two quaternions
 *        using spherical linear interpolation (SLERP)
 *
 * @param[in]   from  from
 * @param[in]   to    to
 * @param[in]   t     amount
 * @param[out]  dest  result quaternion
 */
PLAY_CGLM_INLINE
void
quat_slerp(versor from, versor to, float t, versor dest)
{
    PLAY_CGLM_ALIGN(16) vec4 q1, q2;
    float cosTheta, sinTheta, angle;

    cosTheta = quat_dot(from, to);
    quat_copy(from, q1);

    if (fabsf(cosTheta) >= 1.0f)
    {
        quat_copy(q1, dest);
        return;
    }

    if (cosTheta < 0.0f)
    {
        vec4_negate(q1);
        cosTheta = -cosTheta;
    }

    sinTheta = sqrtf(1.0f - cosTheta * cosTheta);

    /* LERP to avoid zero division */
    if (fabsf(sinTheta) < 0.001f)
    {
        quat_lerp(from, to, t, dest);
        return;
    }

    /* SLERP */
    angle = acosf(cosTheta);
    vec4_scale(q1, sinf((1.0f - t) * angle), q1);
    vec4_scale(to, sinf(t * angle), q2);

    vec4_add(q1, q2, q1);
    vec4_scale(q1, 1.0f / sinTheta, dest);
}

/*!
 * @brief interpolates between two quaternions
 *        using spherical linear interpolation (SLERP) and always takes the long path
 *
 * @param[in]   from  from
 * @param[in]   to    to
 * @param[in]   t     amount
 * @param[out]  dest  result quaternion
 */
PLAY_CGLM_INLINE
void
quat_slerp_longest(versor from, versor to, float t, versor dest)
{
    PLAY_CGLM_ALIGN(16) vec4 q1, q2;
    float cosTheta, sinTheta, angle;

    cosTheta = quat_dot(from, to);
    quat_copy(from, q1);

    if (fabsf(cosTheta) >= 1.0f)
    {
        quat_copy(q1, dest);
        return;
    }

    /* longest path */
    if (!(cosTheta < 0.0f))
    {
        vec4_negate(q1);
        cosTheta = -cosTheta;
    }

    sinTheta = sqrtf(1.0f - cosTheta * cosTheta);

    /* LERP to avoid zero division */
    if (fabsf(sinTheta) < 0.001f)
    {
        quat_lerp(from, to, t, dest);
        return;
    }

    /* SLERP */
    angle = acosf(cosTheta);
    vec4_scale(q1, sinf((1.0f - t) * angle), q1);
    vec4_scale(to, sinf(t * angle), q2);

    vec4_add(q1, q2, q1);
    vec4_scale(q1, 1.0f / sinTheta, dest);
}

/*!
 * @brief creates view matrix using quaternion as camera orientation
 *
 * @param[in]   eye   eye
 * @param[in]   ori   orientation in world space as quaternion
 * @param[out]  dest  view matrix
 */
PLAY_CGLM_INLINE
void
quat_look(vec3 eye, versor ori, mat4 dest)
{
    /* orientation */
    quat_mat4t(ori, dest);

    /* translate */
    mat4_mulv3(dest, eye, 1.0f, dest[3]);
    vec3_negate(dest[3]);
}

/*!
 * @brief creates look rotation quaternion
 *
 * @param[in]   dir   direction to look
 * @param[in]   up    up vector
 * @param[out]  dest  destination quaternion
 */
PLAY_CGLM_INLINE
void
quat_for(vec3 dir, vec3 up, versor dest)
{
    PLAY_CGLM_ALIGN_MAT mat3 m;

    vec3_normalize_to(dir, m[2]);

    /* No need to negate in LH, but we use RH here */
    vec3_negate(m[2]);

    vec3_crossn(up, m[2], m[0]);
    vec3_cross(m[2], m[0], m[1]);

    mat3_quat(m, dest);
}

/*!
 * @brief creates look rotation quaternion using source and
 *        destination positions p suffix stands for position
 *
 * @param[in]   from  source point
 * @param[in]   to    destination point
 * @param[in]   up    up vector
 * @param[out]  dest  destination quaternion
 */
PLAY_CGLM_INLINE
void
quat_forp(vec3 from, vec3 to, vec3 up, versor dest)
{
    PLAY_CGLM_ALIGN(8) vec3 dir;
    vec3_sub(to, from, dir);
    quat_for(dir, up, dest);
}

/*!
 * @brief rotate vector using using quaternion
 *
 * @param[in]   q     quaternion
 * @param[in]   v     vector to rotate
 * @param[out]  dest  rotated vector
 */
PLAY_CGLM_INLINE
void
quat_rotatev(versor q, vec3 v, vec3 dest)
{
    PLAY_CGLM_ALIGN(16) versor p;
    PLAY_CGLM_ALIGN(8)  vec3   u, v1, v2;
    float s;

    quat_normalize_to(q, p);
    quat_imag(p, u);
    s = quat_real(p);

    vec3_scale(u, 2.0f * vec3_dot(u, v), v1);
    vec3_scale(v, s * s - vec3_dot(u, u), v2);
    vec3_add(v1, v2, v1);

    vec3_cross(u, v, v2);
    vec3_scale(v2, 2.0f * s, v2);

    vec3_add(v1, v2, dest);
}

/*!
 * @brief rotate existing transform matrix using quaternion
 *
 * @param[in]   m     existing transform matrix
 * @param[in]   q     quaternion
 * @param[out]  dest  rotated matrix/transform
 */
PLAY_CGLM_INLINE
void
quat_rotate(mat4 m, versor q, mat4 dest)
{
    PLAY_CGLM_ALIGN_MAT mat4 rot;
    quat_mat4(q, rot);
    mul_rot(m, rot, dest);
}

/*!
 * @brief rotate existing transform matrix using quaternion at pivot point
 *
 * @param[in, out]   m     existing transform matrix
 * @param[in]        q     quaternion
 * @param[out]       pivot pivot
 */
PLAY_CGLM_INLINE
void
quat_rotate_at(mat4 m, versor q, vec3 pivot)
{
    PLAY_CGLM_ALIGN(8) vec3 pivotInv;

    vec3_negate_to(pivot, pivotInv);

    translate(m, pivot);
    quat_rotate(m, q, m);
    translate(m, pivotInv);
}

/*!
 * @brief rotate NEW transform matrix using quaternion at pivot point
 *
 * this creates rotation matrix, it assumes you don't have a matrix
 *
 * this should work faster than quat_rotate_at because it reduces
 * one translate.
 *
 * @param[out]  m     existing transform matrix
 * @param[in]   q     quaternion
 * @param[in]   pivot pivot
 */
PLAY_CGLM_INLINE
void
quat_rotate_atm(mat4 m, versor q, vec3 pivot)
{
    PLAY_CGLM_ALIGN(8) vec3 pivotInv;

    vec3_negate_to(pivot, pivotInv);

    translate_make(m, pivot);
    quat_rotate(m, q, m);
    translate(m, pivotInv);
}

/*!
 * @brief Create quaternion from pointer
 *
 * @param[in]  src  pointer to an array of floats
 * @param[out] dest quaternion
 */
PLAY_CGLM_INLINE
void
quat_make(const float * __restrict src, versor dest)
{
    dest[0] = src[0];
    dest[1] = src[1];
    dest[2] = src[2];
    dest[3] = src[3];
}



/*** End of inlined file: quat.h ***/


/*** Start of inlined file: euler.h ***/
/*
 NOTE:
  angles must be passed as [X-Angle, Y-Angle, Z-angle] order
  For instance you don't pass angles as [Z-Angle, X-Angle, Y-angle] to
  euler_zxy function, All RELATED functions accept angles same order
  which is [X, Y, Z].
 */

/*
 Types:
   enum euler_seq

 Functions:
   PLAY_CGLM_INLINE euler_seq euler_order(int newOrder[3]);
   PLAY_CGLM_INLINE void euler_angles(mat4 m, vec3 dest);
   PLAY_CGLM_INLINE void euler(vec3 angles, mat4 dest);
   PLAY_CGLM_INLINE void euler_xyz(vec3 angles, mat4 dest);
   PLAY_CGLM_INLINE void euler_zyx(vec3 angles, mat4 dest);
   PLAY_CGLM_INLINE void euler_zxy(vec3 angles, mat4 dest);
   PLAY_CGLM_INLINE void euler_xzy(vec3 angles, mat4 dest);
   PLAY_CGLM_INLINE void euler_yzx(vec3 angles, mat4 dest);
   PLAY_CGLM_INLINE void euler_yxz(vec3 angles, mat4 dest);
   PLAY_CGLM_INLINE void euler_by_order(vec3         angles,
                                       euler_seq ord,
                                       mat4         dest);
   PLAY_CGLM_INLINE void euler_xyz_quat(vec3 angles, versor dest);
   PLAY_CGLM_INLINE void euler_xzy_quat(vec3 angles, versor dest);
   PLAY_CGLM_INLINE void euler_yxz_quat(vec3 angles, versor dest);
   PLAY_CGLM_INLINE void euler_yzx_quat(vec3 angles, versor dest);
   PLAY_CGLM_INLINE void euler_zxy_quat(vec3 angles, versor dest);
   PLAY_CGLM_INLINE void euler_zyx_quat(vec3 angles, versor dest);
 */




#ifdef PLAY_CGLM_FORCE_LEFT_HANDED

/*** Start of inlined file: euler_to_quat_lh.h ***/
/*
 Functions:
   PLAY_CGLM_INLINE void euler_xyz_quat_lh(vec3 angles, versor dest);
   PLAY_CGLM_INLINE void euler_xzy_quat_lh(vec3 angles, versor dest);
   PLAY_CGLM_INLINE void euler_yxz_quat_lh(vec3 angles, versor dest);
   PLAY_CGLM_INLINE void euler_yzx_quat_lh(vec3 angles, versor dest);
   PLAY_CGLM_INLINE void euler_zxy_quat_lh(vec3 angles, versor dest);
   PLAY_CGLM_INLINE void euler_zyx_quat_lh(vec3 angles, versor dest);
 */

/*
 Things to note:
 The only difference between euler to quat rh vs lh is that the zsin part is negative
 */




/*!
 * @brief creates NEW quaternion using rotation angles and does
 *        rotations in x y z order in left hand (roll pitch yaw)
 *
 * @param[in]   angles angles x y z (radians)
 * @param[out]  dest   quaternion
 */
PLAY_CGLM_INLINE
void
euler_xyz_quat_lh(vec3 angles, versor dest)
{
    float xc, yc, zc,
          xs, ys, zs;

    xs =  sinf(angles[0] * 0.5f);
    xc = cosf(angles[0] * 0.5f);
    ys =  sinf(angles[1] * 0.5f);
    yc = cosf(angles[1] * 0.5f);
    zs = -sinf(angles[2] * 0.5f);
    zc = cosf(angles[2] * 0.5f);

    dest[0] = xc * ys * zs + xs * yc * zc;
    dest[1] = xc * ys * zc - xs * yc * zs;
    dest[2] = xc * yc * zs + xs * ys * zc;
    dest[3] = xc * yc * zc - xs * ys * zs;
}

/*!
 * @brief creates NEW quaternion using rotation angles and does
 *        rotations in x z y order in left hand (roll yaw pitch)
 *
 * @param[in]   angles angles x y z (radians)
 * @param[out]  dest   quaternion
 */
PLAY_CGLM_INLINE
void
euler_xzy_quat_lh(vec3 angles, versor dest)
{
    float xc, yc, zc,
          xs, ys, zs;

    xs =  sinf(angles[0] * 0.5f);
    xc = cosf(angles[0] * 0.5f);
    ys =  sinf(angles[1] * 0.5f);
    yc = cosf(angles[1] * 0.5f);
    zs = -sinf(angles[2] * 0.5f);
    zc = cosf(angles[2] * 0.5f);

    dest[0] = -xc * zs * ys + xs * zc * yc;
    dest[1] =  xc * zc * ys - xs * zs * yc;
    dest[2] =  xc * zs * yc + xs * zc * ys;
    dest[3] =  xc * zc * yc + xs * zs * ys;
}

/*!
 * @brief creates NEW quaternion using rotation angles and does
 *        rotations in y x z order in left hand (pitch roll yaw)
 *
 * @param[in]   angles angles x y z (radians)
 * @param[out]  dest   quaternion
 */
PLAY_CGLM_INLINE
void
euler_yxz_quat_lh(vec3 angles, versor dest)
{
    float xc, yc, zc,
          xs, ys, zs;

    xs =  sinf(angles[0] * 0.5f);
    xc = cosf(angles[0] * 0.5f);
    ys =  sinf(angles[1] * 0.5f);
    yc = cosf(angles[1] * 0.5f);
    zs = -sinf(angles[2] * 0.5f);
    zc = cosf(angles[2] * 0.5f);

    dest[0] =  yc * xs * zc + ys * xc * zs;
    dest[1] = -yc * xs * zs + ys * xc * zc;
    dest[2] =  yc * xc * zs - ys * xs * zc;
    dest[3] =  yc * xc * zc + ys * xs * zs;
}

/*!
 * @brief creates NEW quaternion using rotation angles and does
 *        rotations in y z x order in left hand (pitch yaw roll)
 *
 * @param[in]   angles angles x y z (radians)
 * @param[out]  dest   quaternion
 */
PLAY_CGLM_INLINE
void
euler_yzx_quat_lh(vec3 angles, versor dest)
{
    float xc, yc, zc,
          xs, ys, zs;

    xs =  sinf(angles[0] * 0.5f);
    xc = cosf(angles[0] * 0.5f);
    ys =  sinf(angles[1] * 0.5f);
    yc = cosf(angles[1] * 0.5f);
    zs = -sinf(angles[2] * 0.5f);
    zc = cosf(angles[2] * 0.5f);

    dest[0] = yc * zc * xs + ys * zs * xc;
    dest[1] = yc * zs * xs + ys * zc * xc;
    dest[2] = yc * zs * xc - ys * zc * xs;
    dest[3] = yc * zc * xc - ys * zs * xs;
}

/*!
 * @brief creates NEW quaternion using rotation angles and does
 *        rotations in z x y order in left hand (yaw roll pitch)
 *
 * @param[in]   angles angles x y z (radians)
 * @param[out]  dest   quaternion
 */
PLAY_CGLM_INLINE
void
euler_zxy_quat_lh(vec3 angles, versor dest)
{
    float xc, yc, zc,
          xs, ys, zs;

    xs =  sinf(angles[0] * 0.5f);
    xc = cosf(angles[0] * 0.5f);
    ys =  sinf(angles[1] * 0.5f);
    yc = cosf(angles[1] * 0.5f);
    zs = -sinf(angles[2] * 0.5f);
    zc = cosf(angles[2] * 0.5f);

    dest[0] = zc * xs * yc - zs * xc * ys;
    dest[1] = zc * xc * ys + zs * xs * yc;
    dest[2] = zc * xs * ys + zs * xc * yc;
    dest[3] = zc * xc * yc - zs * xs * ys;
}

/*!
 * @brief creates NEW quaternion using rotation angles and does
 *        rotations in z y x order in left hand (yaw pitch roll)
 *
 * @param[in]   angles angles x y z (radians)
 * @param[out]  dest   quaternion
 */
PLAY_CGLM_INLINE
void
euler_zyx_quat_lh(vec3 angles, versor dest)
{
    float xc, yc, zc,
          xs, ys, zs;

    xs =  sinf(angles[0] * 0.5f);
    xc = cosf(angles[0] * 0.5f);
    ys =  sinf(angles[1] * 0.5f);
    yc = cosf(angles[1] * 0.5f);
    zs = -sinf(angles[2] * 0.5f);
    zc = cosf(angles[2] * 0.5f);

    dest[0] =  zc * yc * xs - zs * ys * xc;
    dest[1] =  zc * ys * xc + zs * yc * xs;
    dest[2] = -zc * ys * xs + zs * yc * xc;
    dest[3] =  zc * yc * xc + zs * ys * xs;
}



/*** End of inlined file: euler_to_quat_lh.h ***/


#else

/*** Start of inlined file: euler_to_quat_rh.h ***/
/*
 Functions:
   PLAY_CGLM_INLINE void euler_xyz_quat_rh(vec3 angles, versor dest);
   PLAY_CGLM_INLINE void euler_xzy_quat_rh(vec3 angles, versor dest);
   PLAY_CGLM_INLINE void euler_yxz_quat_rh(vec3 angles, versor dest);
   PLAY_CGLM_INLINE void euler_yzx_quat_rh(vec3 angles, versor dest);
   PLAY_CGLM_INLINE void euler_zxy_quat_rh(vec3 angles, versor dest);
   PLAY_CGLM_INLINE void euler_zyx_quat_rh(vec3 angles, versor dest);
 */

/*
 Things to note:
 The only difference between euler to quat rh vs lh is that the zsin part is negative
 */




/*!
 * @brief creates NEW quaternion using rotation angles and does
 *        rotations in x y z order in right hand (roll pitch yaw)
 *
 * @param[in]   angles angles x y z (radians)
 * @param[out]  dest   quaternion
 */
PLAY_CGLM_INLINE
void
euler_xyz_quat_rh(vec3 angles, versor dest)
{
    float xc, yc, zc,
          xs, ys, zs;

    xs = sinf(angles[0] * 0.5f);
    xc = cosf(angles[0] * 0.5f);
    ys = sinf(angles[1] * 0.5f);
    yc = cosf(angles[1] * 0.5f);
    zs = sinf(angles[2] * 0.5f);
    zc = cosf(angles[2] * 0.5f);

    dest[0] = xc * ys * zs + xs * yc * zc;
    dest[1] = xc * ys * zc - xs * yc * zs;
    dest[2] = xc * yc * zs + xs * ys * zc;
    dest[3] = xc * yc * zc - xs * ys * zs;

}

/*!
 * @brief creates NEW quaternion using rotation angles and does
 *        rotations in x z y order in right hand (roll yaw pitch)
 *
 * @param[in]   angles angles x y z (radians)
 * @param[out]  dest   quaternion
 */
PLAY_CGLM_INLINE
void
euler_xzy_quat_rh(vec3 angles, versor dest)
{
    float xc, yc, zc,
          xs, ys, zs;

    xs = sinf(angles[0] * 0.5f);
    xc = cosf(angles[0] * 0.5f);
    ys = sinf(angles[1] * 0.5f);
    yc = cosf(angles[1] * 0.5f);
    zs = sinf(angles[2] * 0.5f);
    zc = cosf(angles[2] * 0.5f);

    dest[0] = -xc * zs * ys + xs * zc * yc;
    dest[1] =  xc * zc * ys - xs * zs * yc;
    dest[2] =  xc * zs * yc + xs * zc * ys;
    dest[3] =  xc * zc * yc + xs * zs * ys;

}

/*!
 * @brief creates NEW quaternion using rotation angles and does
 *        rotations in y x z order in right hand (pitch roll yaw)
 *
 * @param[in]   angles angles x y z (radians)
 * @param[out]  dest   quaternion
 */
PLAY_CGLM_INLINE
void
euler_yxz_quat_rh(vec3 angles, versor dest)
{
    float xc, yc, zc,
          xs, ys, zs;

    xs = sinf(angles[0] * 0.5f);
    xc = cosf(angles[0] * 0.5f);
    ys = sinf(angles[1] * 0.5f);
    yc = cosf(angles[1] * 0.5f);
    zs = sinf(angles[2] * 0.5f);
    zc = cosf(angles[2] * 0.5f);

    dest[0] =  yc * xs * zc + ys * xc * zs;
    dest[1] = -yc * xs * zs + ys * xc * zc;
    dest[2] =  yc * xc * zs - ys * xs * zc;
    dest[3] =  yc * xc * zc + ys * xs * zs;
}

/*!
 * @brief creates NEW quaternion using rotation angles and does
 *        rotations in y z x order in right hand (pitch yaw roll)
 *
 * @param[in]   angles angles x y z (radians)
 * @param[out]  dest   quaternion
 */
PLAY_CGLM_INLINE
void
euler_yzx_quat_rh(vec3 angles, versor dest)
{
    float xc, yc, zc,
          xs, ys, zs;

    xs = sinf(angles[0] * 0.5f);
    xc = cosf(angles[0] * 0.5f);
    ys = sinf(angles[1] * 0.5f);
    yc = cosf(angles[1] * 0.5f);
    zs = sinf(angles[2] * 0.5f);
    zc = cosf(angles[2] * 0.5f);

    dest[0] = yc * zc * xs + ys * zs * xc;
    dest[1] = yc * zs * xs + ys * zc * xc;
    dest[2] = yc * zs * xc - ys * zc * xs;
    dest[3] = yc * zc * xc - ys * zs * xs;

}

/*!
 * @brief creates NEW quaternion using rotation angles and does
 *        rotations in z x y order in right hand (yaw roll pitch)
 *
 * @param[in]   angles angles x y z (radians)
 * @param[out]  dest   quaternion
 */
PLAY_CGLM_INLINE
void
euler_zxy_quat_rh(vec3 angles, versor dest)
{
    float xc, yc, zc,
          xs, ys, zs;

    xs = sinf(angles[0] * 0.5f);
    xc = cosf(angles[0] * 0.5f);
    ys = sinf(angles[1] * 0.5f);
    yc = cosf(angles[1] * 0.5f);
    zs = sinf(angles[2] * 0.5f);
    zc = cosf(angles[2] * 0.5f);

    dest[0] = zc * xs * yc - zs * xc * ys;
    dest[1] = zc * xc * ys + zs * xs * yc;
    dest[2] = zc * xs * ys + zs * xc * yc;
    dest[3] = zc * xc * yc - zs * xs * ys;
}

/*!
 * @brief creates NEW quaternion using rotation angles and does
 *        rotations in z y x order in right hand (yaw pitch roll)
 *
 * @param[in]   angles angles x y z (radians)
 * @param[out]  dest   quaternion
 */
PLAY_CGLM_INLINE
void
euler_zyx_quat_rh(vec3 angles, versor dest)
{
    float xc, yc, zc,
          xs, ys, zs;

    xs = sinf(angles[0] * 0.5f);
    xc = cosf(angles[0] * 0.5f);
    ys = sinf(angles[1] * 0.5f);
    yc = cosf(angles[1] * 0.5f);
    zs = sinf(angles[2] * 0.5f);
    zc = cosf(angles[2] * 0.5f);

    dest[0] =  zc * yc * xs - zs * ys * xc;
    dest[1] =  zc * ys * xc + zs * yc * xs;
    dest[2] = -zc * ys * xs + zs * yc * xc;
    dest[3] =  zc * yc * xc + zs * ys * xs;
}



/*** End of inlined file: euler_to_quat_rh.h ***/


#endif

/*!
 * if you have axis order like vec3 orderVec = [0, 1, 2] or [0, 2, 1]...
 * vector then you can convert it to this enum by doing this:
 * @code
 * euler_seq order;
 * order = orderVec[0] | orderVec[1] << 2 | orderVec[2] << 4;
 * @endcode
 * you may need to explicit cast if required
 */
typedef enum euler_seq
{
    PLAY_CGLM_EULER_XYZ = 0 << 0 | 1 << 2 | 2 << 4,
    PLAY_CGLM_EULER_XZY = 0 << 0 | 2 << 2 | 1 << 4,
    PLAY_CGLM_EULER_YZX = 1 << 0 | 2 << 2 | 0 << 4,
    PLAY_CGLM_EULER_YXZ = 1 << 0 | 0 << 2 | 2 << 4,
    PLAY_CGLM_EULER_ZXY = 2 << 0 | 0 << 2 | 1 << 4,
    PLAY_CGLM_EULER_ZYX = 2 << 0 | 1 << 2 | 0 << 4
} euler_seq;

PLAY_CGLM_INLINE
euler_seq
euler_order(int ord[3])
{
    return (euler_seq)(ord[0] << 0 | ord[1] << 2 | ord[2] << 4);
}

/*!
 * @brief extract euler angles (in radians) using xyz order
 *
 * @param[in]  m    affine transform
 * @param[out] dest angles vector [x, y, z]
 */
PLAY_CGLM_INLINE
void
euler_angles(mat4 m, vec3 dest)
{
    float m00, m01, m10, m11, m20, m21, m22;
    float thetaX, thetaY, thetaZ;

    m00 = m[0][0];
    m10 = m[1][0];
    m20 = m[2][0];
    m01 = m[0][1];
    m11 = m[1][1];
    m21 = m[2][1];
    m22 = m[2][2];

    if (m20 < 1.0f)
    {
        if (m20 > -1.0f)
        {
            thetaY = asinf(m20);
            thetaX = atan2f(-m21, m22);
            thetaZ = atan2f(-m10, m00);
        }
        else     /* m20 == -1 */
        {
            /* Not a unique solution */
            thetaY = -PLAY_CGLM_PI_2f;
            thetaX = -atan2f(m01, m11);
            thetaZ =  0.0f;
        }
    }
    else     /* m20 == +1 */
    {
        thetaY = PLAY_CGLM_PI_2f;
        thetaX = atan2f(m01, m11);
        thetaZ = 0.0f;
    }

    dest[0] = thetaX;
    dest[1] = thetaY;
    dest[2] = thetaZ;
}

/*!
 * @brief build rotation matrix from euler angles
 *
 * @param[in]  angles angles as vector [Xangle, Yangle, Zangle]
 * @param[out] dest   rotation matrix
 */
PLAY_CGLM_INLINE
void
euler_xyz(vec3 angles, mat4 dest)
{
    float cx, cy, cz,
          sx, sy, sz, czsx, cxcz, sysz;

    sx   = sinf(angles[0]);
    cx = cosf(angles[0]);
    sy   = sinf(angles[1]);
    cy = cosf(angles[1]);
    sz   = sinf(angles[2]);
    cz = cosf(angles[2]);

    czsx = cz * sx;
    cxcz = cx * cz;
    sysz = sy * sz;

    dest[0][0] =  cy * cz;
    dest[0][1] =  czsx * sy + cx * sz;
    dest[0][2] = -cxcz * sy + sx * sz;
    dest[1][0] = -cy * sz;
    dest[1][1] =  cxcz - sx * sysz;
    dest[1][2] =  czsx + cx * sysz;
    dest[2][0] =  sy;
    dest[2][1] = -cy * sx;
    dest[2][2] =  cx * cy;
    dest[0][3] =  0.0f;
    dest[1][3] =  0.0f;
    dest[2][3] =  0.0f;
    dest[3][0] =  0.0f;
    dest[3][1] =  0.0f;
    dest[3][2] =  0.0f;
    dest[3][3] =  1.0f;
}

/*!
 * @brief build rotation matrix from euler angles
 *
 * @param[in]  angles angles as vector [Xangle, Yangle, Zangle]
 * @param[out] dest   rotation matrix
 */
PLAY_CGLM_INLINE
void
euler(vec3 angles, mat4 dest)
{
    euler_xyz(angles, dest);
}

/*!
 * @brief build rotation matrix from euler angles
 *
 * @param[in]  angles angles as vector [Xangle, Yangle, Zangle]
 * @param[out] dest   rotation matrix
 */
PLAY_CGLM_INLINE
void
euler_xzy(vec3 angles, mat4 dest)
{
    float cx, cy, cz,
          sx, sy, sz, sxsy, cysx, cxsy, cxcy;

    sx   = sinf(angles[0]);
    cx = cosf(angles[0]);
    sy   = sinf(angles[1]);
    cy = cosf(angles[1]);
    sz   = sinf(angles[2]);
    cz = cosf(angles[2]);

    sxsy = sx * sy;
    cysx = cy * sx;
    cxsy = cx * sy;
    cxcy = cx * cy;

    dest[0][0] =  cy * cz;
    dest[0][1] =  sxsy + cxcy * sz;
    dest[0][2] = -cxsy + cysx * sz;
    dest[1][0] = -sz;
    dest[1][1] =  cx * cz;
    dest[1][2] =  cz * sx;
    dest[2][0] =  cz * sy;
    dest[2][1] = -cysx + cxsy * sz;
    dest[2][2] =  cxcy + sxsy * sz;
    dest[0][3] =  0.0f;
    dest[1][3] =  0.0f;
    dest[2][3] =  0.0f;
    dest[3][0] =  0.0f;
    dest[3][1] =  0.0f;
    dest[3][2] =  0.0f;
    dest[3][3] =  1.0f;
}

/*!
 * @brief build rotation matrix from euler angles
 *
 * @param[in]  angles angles as vector [Xangle, Yangle, Zangle]
 * @param[out] dest   rotation matrix
 */
PLAY_CGLM_INLINE
void
euler_yxz(vec3 angles, mat4 dest)
{
    float cx, cy, cz,
          sx, sy, sz, cycz, sysz, czsy, cysz;

    sx   = sinf(angles[0]);
    cx = cosf(angles[0]);
    sy   = sinf(angles[1]);
    cy = cosf(angles[1]);
    sz   = sinf(angles[2]);
    cz = cosf(angles[2]);

    cycz = cy * cz;
    sysz = sy * sz;
    czsy = cz * sy;
    cysz = cy * sz;

    dest[0][0] =  cycz + sx * sysz;
    dest[0][1] =  cx * sz;
    dest[0][2] = -czsy + cysz * sx;
    dest[1][0] = -cysz + czsy * sx;
    dest[1][1] =  cx * cz;
    dest[1][2] =  cycz * sx + sysz;
    dest[2][0] =  cx * sy;
    dest[2][1] = -sx;
    dest[2][2] =  cx * cy;
    dest[0][3] =  0.0f;
    dest[1][3] =  0.0f;
    dest[2][3] =  0.0f;
    dest[3][0] =  0.0f;
    dest[3][1] =  0.0f;
    dest[3][2] =  0.0f;
    dest[3][3] =  1.0f;
}

/*!
 * @brief build rotation matrix from euler angles
 *
 * @param[in]  angles angles as vector [Xangle, Yangle, Zangle]
 * @param[out] dest   rotation matrix
 */
PLAY_CGLM_INLINE
void
euler_yzx(vec3 angles, mat4 dest)
{
    float cx, cy, cz,
          sx, sy, sz, sxsy, cxcy, cysx, cxsy;

    sx   = sinf(angles[0]);
    cx = cosf(angles[0]);
    sy   = sinf(angles[1]);
    cy = cosf(angles[1]);
    sz   = sinf(angles[2]);
    cz = cosf(angles[2]);

    sxsy = sx * sy;
    cxcy = cx * cy;
    cysx = cy * sx;
    cxsy = cx * sy;

    dest[0][0] =  cy * cz;
    dest[0][1] =  sz;
    dest[0][2] = -cz * sy;
    dest[1][0] =  sxsy - cxcy * sz;
    dest[1][1] =  cx * cz;
    dest[1][2] =  cysx + cxsy * sz;
    dest[2][0] =  cxsy + cysx * sz;
    dest[2][1] = -cz * sx;
    dest[2][2] =  cxcy - sxsy * sz;
    dest[0][3] =  0.0f;
    dest[1][3] =  0.0f;
    dest[2][3] =  0.0f;
    dest[3][0] =  0.0f;
    dest[3][1] =  0.0f;
    dest[3][2] =  0.0f;
    dest[3][3] =  1.0f;
}

/*!
 * @brief build rotation matrix from euler angles
 *
 * @param[in]  angles angles as vector [Xangle, Yangle, Zangle]
 * @param[out] dest   rotation matrix
 */
PLAY_CGLM_INLINE
void
euler_zxy(vec3 angles, mat4 dest)
{
    float cx, cy, cz,
          sx, sy, sz, cycz, sxsy, cysz;

    sx   = sinf(angles[0]);
    cx = cosf(angles[0]);
    sy   = sinf(angles[1]);
    cy = cosf(angles[1]);
    sz   = sinf(angles[2]);
    cz = cosf(angles[2]);

    cycz = cy * cz;
    sxsy = sx * sy;
    cysz = cy * sz;

    dest[0][0] =  cycz - sxsy * sz;
    dest[0][1] =  cz * sxsy + cysz;
    dest[0][2] = -cx * sy;
    dest[1][0] = -cx * sz;
    dest[1][1] =  cx * cz;
    dest[1][2] =  sx;
    dest[2][0] =  cz * sy + cysz * sx;
    dest[2][1] = -cycz * sx + sy * sz;
    dest[2][2] =  cx * cy;
    dest[0][3] =  0.0f;
    dest[1][3] =  0.0f;
    dest[2][3] =  0.0f;
    dest[3][0] =  0.0f;
    dest[3][1] =  0.0f;
    dest[3][2] =  0.0f;
    dest[3][3] =  1.0f;
}

/*!
 * @brief build rotation matrix from euler angles
 *
 * @param[in]  angles angles as vector [Xangle, Yangle, Zangle]
 * @param[out] dest   rotation matrix
 */
PLAY_CGLM_INLINE
void
euler_zyx(vec3 angles, mat4 dest)
{
    float cx, cy, cz,
          sx, sy, sz, czsx, cxcz, sysz;

    sx   = sinf(angles[0]);
    cx = cosf(angles[0]);
    sy   = sinf(angles[1]);
    cy = cosf(angles[1]);
    sz   = sinf(angles[2]);
    cz = cosf(angles[2]);

    czsx = cz * sx;
    cxcz = cx * cz;
    sysz = sy * sz;

    dest[0][0] =  cy * cz;
    dest[0][1] =  cy * sz;
    dest[0][2] = -sy;
    dest[1][0] =  czsx * sy - cx * sz;
    dest[1][1] =  cxcz + sx * sysz;
    dest[1][2] =  cy * sx;
    dest[2][0] =  cxcz * sy + sx * sz;
    dest[2][1] = -czsx + cx * sysz;
    dest[2][2] =  cx * cy;
    dest[0][3] =  0.0f;
    dest[1][3] =  0.0f;
    dest[2][3] =  0.0f;
    dest[3][0] =  0.0f;
    dest[3][1] =  0.0f;
    dest[3][2] =  0.0f;
    dest[3][3] =  1.0f;
}

/*!
 * @brief build rotation matrix from euler angles
 *
 * @param[in]  angles angles as vector [Xangle, Yangle, Zangle]
 * @param[in]  ord    euler order
 * @param[out] dest   rotation matrix
 */
PLAY_CGLM_INLINE
void
euler_by_order(vec3 angles, euler_seq ord, mat4 dest)
{
    float cx, cy, cz,
          sx, sy, sz;

    float cycz, cysz, cysx, cxcy,
          czsy, cxcz, czsx, cxsz,
          sysz;

    sx = sinf(angles[0]);
    cx = cosf(angles[0]);
    sy = sinf(angles[1]);
    cy = cosf(angles[1]);
    sz = sinf(angles[2]);
    cz = cosf(angles[2]);

    cycz = cy * cz;
    cysz = cy * sz;
    cysx = cy * sx;
    cxcy = cx * cy;
    czsy = cz * sy;
    cxcz = cx * cz;
    czsx = cz * sx;
    cxsz = cx * sz;
    sysz = sy * sz;

    switch (ord)
    {
    case PLAY_CGLM_EULER_XZY:
        dest[0][0] =  cycz;
        dest[0][1] =  sx * sy + cx * cysz;
        dest[0][2] = -cx * sy + cysx * sz;
        dest[1][0] = -sz;
        dest[1][1] =  cxcz;
        dest[1][2] =  czsx;
        dest[2][0] =  czsy;
        dest[2][1] = -cysx + cx * sysz;
        dest[2][2] =  cxcy + sx * sysz;
        break;
    case PLAY_CGLM_EULER_XYZ:
        dest[0][0] =  cycz;
        dest[0][1] =  czsx * sy + cxsz;
        dest[0][2] = -cx * czsy + sx * sz;
        dest[1][0] = -cysz;
        dest[1][1] =  cxcz - sx * sysz;
        dest[1][2] =  czsx + cx * sysz;
        dest[2][0] =  sy;
        dest[2][1] = -cysx;
        dest[2][2] =  cxcy;
        break;
    case PLAY_CGLM_EULER_YXZ:
        dest[0][0] =  cycz + sx * sysz;
        dest[0][1] =  cxsz;
        dest[0][2] = -czsy + cysx * sz;
        dest[1][0] =  czsx * sy - cysz;
        dest[1][1] =  cxcz;
        dest[1][2] =  cycz * sx + sysz;
        dest[2][0] =  cx * sy;
        dest[2][1] = -sx;
        dest[2][2] =  cxcy;
        break;
    case PLAY_CGLM_EULER_YZX:
        dest[0][0] =  cycz;
        dest[0][1] =  sz;
        dest[0][2] = -czsy;
        dest[1][0] =  sx * sy - cx * cysz;
        dest[1][1] =  cxcz;
        dest[1][2] =  cysx + cx * sysz;
        dest[2][0] =  cx * sy + cysx * sz;
        dest[2][1] = -czsx;
        dest[2][2] =  cxcy - sx * sysz;
        break;
    case PLAY_CGLM_EULER_ZXY:
        dest[0][0] =  cycz - sx * sysz;
        dest[0][1] =  czsx * sy + cysz;
        dest[0][2] = -cx * sy;
        dest[1][0] = -cxsz;
        dest[1][1] =  cxcz;
        dest[1][2] =  sx;
        dest[2][0] =  czsy + cysx * sz;
        dest[2][1] = -cycz * sx + sysz;
        dest[2][2] =  cxcy;
        break;
    case PLAY_CGLM_EULER_ZYX:
        dest[0][0] =  cycz;
        dest[0][1] =  cysz;
        dest[0][2] = -sy;
        dest[1][0] =  czsx * sy - cxsz;
        dest[1][1] =  cxcz + sx * sysz;
        dest[1][2] =  cysx;
        dest[2][0] =  cx * czsy + sx * sz;
        dest[2][1] = -czsx + cx * sysz;
        dest[2][2] =  cxcy;
        break;
    }

    dest[0][3] = 0.0f;
    dest[1][3] = 0.0f;
    dest[2][3] = 0.0f;
    dest[3][0] = 0.0f;
    dest[3][1] = 0.0f;
    dest[3][2] = 0.0f;
    dest[3][3] = 1.0f;
}

/*!
 * @brief creates NEW quaternion using rotation angles and does
 *        rotations in x y z order (roll pitch yaw)
 *
 * @param[in]   angles angles x y z (radians)
 * @param[out]  dest   quaternion
 */
PLAY_CGLM_INLINE
void
euler_xyz_quat(vec3 angles, versor dest)
{
#ifdef PLAY_CGLM_FORCE_LEFT_HANDED
    euler_xyz_quat_lh(angles, dest);
#else
    euler_xyz_quat_rh(angles, dest);
#endif
}

/*!
 * @brief creates NEW quaternion using rotation angles and does
 *        rotations in x z y order (roll yaw pitch)
 *
 * @param[in]   angles angles x y z (radians)
 * @param[out]  dest   quaternion
 */
PLAY_CGLM_INLINE
void
euler_xzy_quat(vec3 angles, versor dest)
{
#ifdef PLAY_CGLM_FORCE_LEFT_HANDED
    euler_xzy_quat_lh(angles, dest);
#else
    euler_xzy_quat_rh(angles, dest);
#endif
}

/*!
 * @brief creates NEW quaternion using rotation angles and does
 *        rotations in y x z order (pitch roll yaw)
 *
 * @param[in]   angles angles x y z (radians)
 * @param[out]  dest   quaternion
 */
PLAY_CGLM_INLINE
void
euler_yxz_quat(vec3 angles, versor dest)
{
#ifdef PLAY_CGLM_FORCE_LEFT_HANDED
    euler_yxz_quat_lh(angles, dest);
#else
    euler_yxz_quat_rh(angles, dest);
#endif
}

/*!
 * @brief creates NEW quaternion using rotation angles and does
 *        rotations in y z x order (pitch yaw roll)
 *
 * @param[in]   angles angles x y z (radians)
 * @param[out]  dest   quaternion
 */
PLAY_CGLM_INLINE
void
euler_yzx_quat(vec3 angles, versor dest)
{
#ifdef PLAY_CGLM_FORCE_LEFT_HANDED
    euler_yzx_quat_lh(angles, dest);
#else
    euler_yzx_quat_rh(angles, dest);
#endif
}

/*!
 * @brief creates NEW quaternion using rotation angles and does
 *        rotations in z x y order (yaw roll pitch)
 *
 * @param[in]   angles angles x y z (radians)
 * @param[out]  dest   quaternion
 */
PLAY_CGLM_INLINE
void
euler_zxy_quat(vec3 angles, versor dest)
{
#ifdef PLAY_CGLM_FORCE_LEFT_HANDED
    euler_zxy_quat_lh(angles, dest);
#else
    euler_zxy_quat_rh(angles, dest);
#endif
}

/*!
 * @brief creates NEW quaternion using rotation angles and does
 *        rotations in z y x order (yaw pitch roll)
 *
 * @param[in]   angles angles x y z (radians)
 * @param[out]  dest   quaternion
 */
PLAY_CGLM_INLINE
void
euler_zyx_quat(vec3 angles, versor dest)
{
#ifdef PLAY_CGLM_FORCE_LEFT_HANDED
    euler_zyx_quat_lh(angles, dest);
#else
    euler_zyx_quat_rh(angles, dest);
#endif
}



/*** End of inlined file: euler.h ***/


/*** Start of inlined file: noise.h ***/



#define _noiseDetail_mod289(x) (x - floorf(x * (1.0f / 289.0f)) * 289.0f)

/* _noiseDetail_permute(vec4 x, vec4 dest) */
#define _noiseDetail_permute(x, dest) { \
  dest[0] = _noiseDetail_mod289((x[0] * 34.0f + 1.0f) * x[0]); \
  dest[1] = _noiseDetail_mod289((x[1] * 34.0f + 1.0f) * x[1]); \
  dest[2] = _noiseDetail_mod289((x[2] * 34.0f + 1.0f) * x[2]); \
  dest[3] = _noiseDetail_mod289((x[3] * 34.0f + 1.0f) * x[3]); \
}

/* _noiseDetail_fade_vec4(vec4 t, vec4 dest) */
#define _noiseDetail_fade_vec4(t, dest) { \
  /* dest = (t * t * t) * (t * (t * 6.0f - 15.0f) + 10.0f) */ \
  vec4 temp; \
  vec4_mul(t, t, temp); \
  vec4_mul(temp, t, temp); \
  /* dest = (t * (t * 6.0f - 15.0f) + 10.0f) */ \
  vec4_scale(t, 6.0f, dest); \
  vec4_subs(dest, 15.0f, dest); \
  vec4_mul(t, dest, dest); \
  vec4_adds(dest, 10.0f, dest); \
  /* dest = temp * dest */ \
  vec4_mul(temp, dest, dest); \
}

/* _noiseDetail_fade_vec3(vec3 t, vec3 dest) */
#define _noiseDetail_fade_vec3(t, dest) { \
  /* dest = (t * t * t) * (t * (t * 6.0f - 15.0f) + 10.0f) */ \
  /* temp = t * t * t */ \
  vec3 temp; \
  vec3_mul(t, t, temp); \
  vec3_mul(temp, t, temp); \
  /* dest = (t * (t * 6.0f - 15.0f) + 10.0f) */ \
  vec3_scale(t, 6.0f, dest); \
  vec3_subs(dest, 15.0f, dest); \
  vec3_mul(t, dest, dest); \
  vec3_adds(dest, 10.0f, dest); \
  /* dest = temp * dest */ \
  vec3_mul(temp, dest, dest); \
}

/* _noiseDetail_fade_vec2(vec2 t, vec2 dest) */
#define _noiseDetail_fade_vec2(t, dest) { \
    /* dest = (t * t * t) * (t * (t * 6.0f - 15.0f) + 10.0f) */ \
    /* temp = t * t * t */ \
    vec2 temp; \
    vec2_mul(t, t, temp); \
    vec2_mul(temp, t, temp); \
    /* dest = (t * (t * 6.0f - 15.0f) + 10.0f) */ \
    vec2_scale(t, 6.0f, dest); \
    vec2_subs(dest, 15.0f, dest); \
    vec2_mul(t, dest, dest); \
    vec2_adds(dest, 10.0f, dest); \
    /* dest = temp * dest */ \
    vec2_mul(temp, dest, dest); \
}

/* _noiseDetail_taylorInvSqrt(vec4 x, vec4 dest) */
#define _noiseDetail_taylorInvSqrt(x, dest) {                        \
  /* dest = 1.79284291400159f - 0.85373472095314f * x */                 \
  vec4 temp;                                                             \
  vec4_scale(x, 0.85373472095314f, temp); /* temp = 0.853...f * x */ \
  vec4_fill(dest, 1.79284291400159f); /* dest = 1.792...f */         \
  vec4_sub(dest, temp, dest); /* dest = 1.79284291400159f - temp */  \
}

/* norm = taylorInvSqrt(vec4_new(
 *     dot(g00__, g00__),
 *     dot(g01__, g01__),
 *     dot(g10__, g10__),
 *     dot(g11__, g11__)
 * ));
*/

/* _noiseDetail_gradNorm_vec4(vec4 g00__, vec4 g01__, vec4 g10__, vec4 g11__) */
#define _noiseDetail_gradNorm_vec4(g00__, g01__, g10__, g11__) {           \
  vec4 norm;                                                                   \
  norm[0] = vec4_dot(g00__, g00__); /* norm.x = dot(g00__, g00__) */       \
  norm[1] = vec4_dot(g01__, g01__); /* norm.y = dot(g01__, g01__) */       \
  norm[2] = vec4_dot(g10__, g10__); /* norm.z = dot(g10__, g10__) */       \
  norm[3] = vec4_dot(g11__, g11__); /* norm.w = dot(g11__, g11__) */       \
  _noiseDetail_taylorInvSqrt(norm, norm); /* norm = taylorInvSqrt(norm) */ \
                                                                               \
  vec4_scale(g00__, norm[0], g00__); /* g00__ *= norm.x */                 \
  vec4_scale(g01__, norm[1], g01__); /* g01__ *= norm.y */                 \
  vec4_scale(g10__, norm[2], g10__); /* g10__ *= norm.z */                 \
  vec4_scale(g11__, norm[3], g11__); /* g11__ *= norm.w */                 \
}

/* _noiseDetail_gradNorm_vec3(vec3 g00_, vec3 g01_, vec3 g10_, vec3 g11_) */
#define _noiseDetail_gradNorm_vec3(g00_, g01_, g10_, g11_) {               \
  vec4 norm;                                                                   \
  norm[0] = vec3_dot(g00_, g00_); /* norm.x = dot(g00_, g00_) */           \
  norm[1] = vec3_dot(g01_, g01_); /* norm.y = dot(g01_, g01_) */           \
  norm[2] = vec3_dot(g10_, g10_); /* norm.z = dot(g10_, g10_) */           \
  norm[3] = vec3_dot(g11_, g11_); /* norm.w = dot(g11_, g11_) */           \
  _noiseDetail_taylorInvSqrt(norm, norm); /* norm = taylorInvSqrt(norm) */ \
                                                                               \
  vec3_scale(g00_, norm[0], g00_); /* g00_ *= norm.x */                    \
  vec3_scale(g01_, norm[1], g01_); /* g01_ *= norm.y */                    \
  vec3_scale(g10_, norm[2], g10_); /* g10_ *= norm.z */                    \
  vec3_scale(g11_, norm[3], g11_); /* g11_ *= norm.w */                    \
}

/* _noiseDetail_gradNorm_vec2(vec2 g00, vec2 g01, vec2 g10, vec2 g11) */
#define _noiseDetail_gradNorm_vec2(g00, g01, g10, g11) {                   \
  vec4 norm;                                                                   \
  norm[0] = vec2_dot(g00, g00); /* norm.x = dot(g00, g00) */               \
  norm[1] = vec2_dot(g01, g01); /* norm.y = dot(g01, g01) */               \
  norm[2] = vec2_dot(g10, g10); /* norm.z = dot(g10, g10) */               \
  norm[3] = vec2_dot(g11, g11); /* norm.w = dot(g11, g11) */               \
  _noiseDetail_taylorInvSqrt(norm, norm); /* norm = taylorInvSqrt(norm) */ \
                                                                               \
  vec2_scale(g00, norm[0], g00); /* g00 *= norm.x */                       \
  vec2_scale(g01, norm[1], g01); /* g01 *= norm.y */                       \
  vec2_scale(g10, norm[2], g10); /* g10 *= norm.z */                       \
  vec2_scale(g11, norm[3], g11); /* g11 *= norm.w */                       \
}

/* _noiseDetail_i2gxyzw(vec4 ixy, vec4 gx, vec4 gy, vec4 gz, vec4 gw) */
#define _noiseDetail_i2gxyzw(ixy, gx, gy, gz, gw) {      \
  /* gx = ixy / 7.0 */                                       \
  vec4_divs(ixy, 7.0f, gx); /* gx = ixy / 7.0 */         \
                                                             \
  /* gy = fract(gx) / 7.0 */                                 \
  vec4_floor(gx, gy); /* gy = floor(gx) */               \
  vec4_divs(gy, 7.0f, gy); /* gy /= 7.0 */               \
                                                             \
  /* gz = floor(gy) / 6.0 */                                 \
  vec4_floor(gy, gz); /* gz = floor(gy) */               \
  vec4_divs(gz, 6.0f, gz); /* gz /= 6.0 */               \
                                                             \
  /* gx = fract(gx) - 0.5f */                                \
  vec4_fract(gx, gx); /* gx = fract(gx) */               \
  vec4_subs(gx, 0.5f, gx); /* gx -= 0.5f */              \
                                                             \
  /* gy = fract(gy) - 0.5f */                                \
  vec4_fract(gy, gy); /* gy = fract(gy) */               \
  vec4_subs(gy, 0.5f, gy); /* gy -= 0.5f */              \
                                                             \
  /* gz = fract(gz) - 0.5f */                                \
  vec4_fract(gz, gz); /* gz = fract(gz) */               \
  vec4_subs(gz, 0.5f, gz); /* gz -= 0.5f */              \
                                                             \
  /* abs(gx), abs(gy), abs(gz) */                            \
  vec4 gxa, gya, gza;                                        \
  vec4_abs(gx, gxa); /* gxa = abs(gx) */                 \
  vec4_abs(gy, gya); /* gya = abs(gy) */                 \
  vec4_abs(gz, gza); /* gza = abs(gz) */                 \
                                                             \
  /* gw = 0.75 - abs(gx) - abs(gy) - abs(gz) */              \
  vec4_fill(gw, 0.75f); /* gw = 0.75 */                  \
  vec4_sub(gw, gxa, gw); /* gw -= gxa */                 \
  vec4_sub(gw, gza, gw); /* gw -= gza */                 \
  vec4_sub(gw, gya, gw); /* gw -= gya */                 \
                                                             \
  /* sw = step(gw, 0.0); */                                  \
  vec4 sw;                                                   \
  vec4_stepr(gw, 0.0f, sw); /* sw = step(gw, 0.0) */     \
                                                             \
  /* gx -= sw * (step(vec4_new(0), gx) - T(0.5)); */             \
  vec4 temp = {0.0f}; /* temp = 0.0 */                       \
  vec4_step(temp, gx, temp); /* temp = step(temp, gx) */ \
  vec4_subs(temp, 0.5f, temp); /* temp -= 0.5 */         \
  vec4_mul(sw, temp, temp); /* temp *= sw */             \
  vec4_sub(gx, temp, gx); /* gx -= temp */               \
                                                             \
  /* gy -= sw * (step(vec4_new(0), gy) - T(0.5)); */             \
  vec4_zero(temp); /* reset temp */                      \
  vec4_step(temp, gy, temp); /* temp = step(temp, gy) */ \
  vec4_subs(temp, 0.5f, temp); /* temp -= 0.5 */         \
  vec4_mul(sw, temp, temp); /* temp *= sw */             \
  vec4_sub(gy, temp, gy); /* gy -= temp */               \
}

/* NOTE: This function is not *quite* analogous to _noiseDetail_i2gxyzw
 * to try to match the output of glm::perlin. I think it might be a bug in
 * in the original implementation, but for now I'm keeping it consistent. -MK
 *
 * Follow up: The original implementation (glm v 1.0.1) does:
 *
 *   vec<4, T, Q> gx0 = ixy0 * T(1.0 / 7.0);
 *
 * as opposed to:
 *
 *   vec<4, T, Q> gx0 = ixy0 / T(7);
 *
 * This ends up mapping to different simd instructions, at least on AMD.
 * The delta is tiny but it gets amplified by the rest of the noise function.
 * Hence we too need to do `vec4_scale` as opposed to `vec4_divs`, to
 * match it. -MK
 */

/* _noiseDetail_i2gxyz(vec4 i, vec4 gx, vec4 gy, vec4 gz) */
#define _noiseDetail_i2gxyz(ixy, gx, gy, gz) {               \
  /* gx = ixy / 7.0 */                                           \
  vec4_scale(ixy, 1.0f / 7.0f, gx); /* gx = ixy * (1/7.0) */\
                                                                 \
  /* gy = fract(floor(gx0) / 7.0)) - 0.5; */                     \
  vec4_floor(gx, gy); /* gy = floor(gx) */                   \
  vec4_scale(gy, 1.0f / 7.0f, gy); /* gy *= 1 / 7.0 */       \
  vec4_fract(gy, gy); /* gy = fract(gy) */                   \
  vec4_subs(gy, 0.5f, gy); /* gy -= 0.5f */                  \
                                                                 \
  /* gx = fract(gx); */                                          \
  vec4_fract(gx, gx); /* gx = fract(gx) */                   \
                                                                 \
  /* abs(gx), abs(gy) */                                         \
  vec4 gxa, gya;                                                 \
  vec4_abs(gx, gxa); /* gxa = abs(gx) */                     \
  vec4_abs(gy, gya); /* gya = abs(gy) */                     \
                                                                 \
  /* gz = vec4_new(0.5) - abs(gx0) - abs(gy0); */                    \
  vec4_fill(gz, 0.5f); /* gz = 0.5 */                        \
  vec4_sub(gz, gxa, gz); /* gz -= gxa */                     \
  vec4_sub(gz, gya, gz); /* gz -= gya */                     \
                                                                 \
  /* sz = step(gw, 0.0); */                                      \
  vec4 sz;                                                       \
  vec4_stepr(gz, 0.0f, sz); /* sz = step(gz, 0.0) */         \
                                                                 \
  /* gx0 -= sz0 * (step(0.0, gx0) - T(0.5)); */                  \
  vec4 temp = {0.0f}; /* temp = 0.0 */                           \
  vec4_step(temp, gx, temp); /* temp = step(temp, gx) */     \
  vec4_subs(temp, 0.5f, temp); /* temp -= 0.5 */             \
  vec4_mul(sz, temp, temp); /* temp *= sz */                 \
  vec4_sub(gx, temp, gx); /* gx -= temp */                   \
                                                                 \
  /* gy0 -= sz0 * (step(0.0, gy0) - T(0.5)); */                  \
  vec4_zero(temp); /* reset temp */                          \
  vec4_step(temp, gy, temp); /* temp = step(temp, gy) */     \
  vec4_subs(temp, 0.5f, temp); /* temp -= 0.5 */             \
  vec4_mul(sz, temp, temp); /* temp *= sz */                 \
  vec4_sub(gy, temp, gy); /* gy -= temp */                   \
}

/* _noiseDetail_i2gxy(vec4 i, vec4 gx, vec4 gy) */
#define _noiseDetail_i2gxy(i, gx, gy) {                      \
  /* gx = 2.0 * fract(i / 41.0) - 1.0; */                        \
  vec4_divs(i, 41.0f, gx); /* gx = i / 41.0 */               \
  vec4_fract(gx, gx); /* gx = fract(gx) */                   \
  vec4_scale(gx, 2.0f, gx); /* gx *= 2.0 */                  \
  vec4_subs(gx, 1.0f, gx); /* gx -= 1.0 */                   \
                                                                 \
  /* gy = abs(gx) - 0.5; */                                      \
  vec4_abs(gx, gy); /* gy = abs(gx) */                       \
  vec4_subs(gy, 0.5f, gy); /* gy -= 0.5 */                   \
                                                                 \
  /* tx = floor(gx + 0.5); */                                    \
  vec4 tx;                                                       \
  vec4_adds(gx, 0.5f, tx); /* tx = gx + 0.5 */               \
  vec4_floor(tx, tx); /* tx = floor(tx) */                   \
                                                                 \
  /* gx = gx - tx; */                                            \
  vec4_sub(gx, tx, gx); /* gx -= tx */                       \
}

/* ============================================================================
 * Classic perlin noise
 * ============================================================================
 */

/*!
 * @brief Classic perlin noise
 *
 * @param[in]  point  4D vector
 * @returns           perlin noise value
 */
PLAY_CGLM_INLINE
float
perlin_vec4(vec4 point)
{
    /* Integer part of p for indexing */
    vec4 Pi0;
    vec4_floor(point, Pi0); /* Pi0 = floor(point); */

    /* Integer part + 1 */
    vec4 Pi1;
    vec4_adds(Pi0, 1.0f, Pi1); /* Pi1 = Pi0 + 1.0f; */

    vec4_mods(Pi0, 289.0f, Pi0); /* Pi0 = mod(Pi0, 289.0f); */
    vec4_mods(Pi1, 289.0f, Pi1); /* Pi1 = mod(Pi1, 289.0f); */

    /* Fractional part of p for interpolation */
    vec4 Pf0;
    vec4_fract(point, Pf0);

    /* Fractional part - 1.0 */
    vec4 Pf1;
    vec4_subs(Pf0, 1.0f, Pf1);

    vec4 ix = {Pi0[0], Pi1[0], Pi0[0], Pi1[0]};
    vec4 iy = {Pi0[1], Pi0[1], Pi1[1], Pi1[1]};
    vec4 iz0 = {Pi0[2], Pi0[2], Pi0[2], Pi0[2]}; /* iz0 = vec4_new(Pi0.z); */
    vec4 iz1 = {Pi1[2], Pi1[2], Pi1[2], Pi1[2]}; /* iz1 = vec4_new(Pi1.z); */
    vec4 iw0 = {Pi0[3], Pi0[3], Pi0[3], Pi0[3]}; /* iw0 = vec4_new(Pi0.w); */
    vec4 iw1 = {Pi1[3], Pi1[3], Pi1[3], Pi1[3]}; /* iw1 = vec4_new(Pi1.w); */

    /* ------------ */

    /* ixy = permute(permute(ix) + iy) */
    vec4 ixy;
    _noiseDetail_permute(ix, ixy); /* ixy = permute(ix) */
    vec4_add(ixy, iy, ixy); /* ixy += iy; */
    _noiseDetail_permute(ixy, ixy); /* ixy = permute(ixy) */

    /* ixy0 = permute(ixy + iz0) */
    vec4 ixy0;
    vec4_add(ixy, iz0, ixy0); /* ixy0 = ixy + iz0 */
    _noiseDetail_permute(ixy0, ixy0); /* ixy0 = permute(ixy0) */

    /* ixy1 = permute(ixy + iz1) */
    vec4 ixy1;
    vec4_add(ixy, iz1, ixy1); /* ixy1 = ixy, iz1 */
    _noiseDetail_permute(ixy1, ixy1); /* ixy1 = permute(ixy1) */

    /* ixy00 = permute(ixy0 + iw0) */
    vec4 ixy00;
    vec4_add(ixy0, iw0, ixy00); /* ixy00 = ixy0 + iw0 */
    _noiseDetail_permute(ixy00, ixy00); /* ixy00 = permute(ixy00) */

    /* ixy01 = permute(ixy0 + iw1) */
    vec4 ixy01;
    vec4_add(ixy0, iw1, ixy01); /* ixy01 = ixy0 + iw1 */
    _noiseDetail_permute(ixy01, ixy01); /* ixy01 = permute(ixy01) */

    /* ixy10 = permute(ixy1 + iw0) */
    vec4 ixy10;
    vec4_add(ixy1, iw0, ixy10); /* ixy10 = ixy1 + iw0 */
    _noiseDetail_permute(ixy10, ixy10); /* ixy10 = permute(ixy10) */

    /* ixy11 = permute(ixy1 + iw1) */
    vec4 ixy11;
    vec4_add(ixy1, iw1, ixy11); /* ixy11 = ixy1 + iw1 */
    _noiseDetail_permute(ixy11, ixy11); /* ixy11 = permute(ixy11) */

    /* ------------ */

    vec4 gx00, gy00, gz00, gw00;
    _noiseDetail_i2gxyzw(ixy00, gx00, gy00, gz00, gw00);

    vec4 gx01, gy01, gz01, gw01;
    _noiseDetail_i2gxyzw(ixy01, gx01, gy01, gz01, gw01);

    vec4 gx10, gy10, gz10, gw10;
    _noiseDetail_i2gxyzw(ixy10, gx10, gy10, gz10, gw10);

    vec4 gx11, gy11, gz11, gw11;
    _noiseDetail_i2gxyzw(ixy11, gx11, gy11, gz11, gw11);

    /* ------------ */

    vec4 g0000 = {gx00[0], gy00[0], gz00[0], gw00[0]}; /* g0000 = vec4_new(gx00.x, gy00.x, gz00.x, gw00.x); */
    vec4 g0100 = {gx00[2], gy00[2], gz00[2], gw00[2]}; /* g0100 = vec4_new(gx00.z, gy00.z, gz00.z, gw00.z); */
    vec4 g1000 = {gx00[1], gy00[1], gz00[1], gw00[1]}; /* g1000 = vec4_new(gx00.y, gy00.y, gz00.y, gw00.y); */
    vec4 g1100 = {gx00[3], gy00[3], gz00[3], gw00[3]}; /* g1100 = vec4_new(gx00.w, gy00.w, gz00.w, gw00.w); */

    vec4 g0001 = {gx01[0], gy01[0], gz01[0], gw01[0]}; /* g0001 = vec4_new(gx01.x, gy01.x, gz01.x, gw01.x); */
    vec4 g0101 = {gx01[2], gy01[2], gz01[2], gw01[2]}; /* g0101 = vec4_new(gx01.z, gy01.z, gz01.z, gw01.z); */
    vec4 g1001 = {gx01[1], gy01[1], gz01[1], gw01[1]}; /* g1001 = vec4_new(gx01.y, gy01.y, gz01.y, gw01.y); */
    vec4 g1101 = {gx01[3], gy01[3], gz01[3], gw01[3]}; /* g1101 = vec4_new(gx01.w, gy01.w, gz01.w, gw01.w); */

    vec4 g0010 = {gx10[0], gy10[0], gz10[0], gw10[0]}; /* g0010 = vec4_new(gx10.x, gy10.x, gz10.x, gw10.x); */
    vec4 g0110 = {gx10[2], gy10[2], gz10[2], gw10[2]}; /* g0110 = vec4_new(gx10.z, gy10.z, gz10.z, gw10.z); */
    vec4 g1010 = {gx10[1], gy10[1], gz10[1], gw10[1]}; /* g1010 = vec4_new(gx10.y, gy10.y, gz10.y, gw10.y); */
    vec4 g1110 = {gx10[3], gy10[3], gz10[3], gw10[3]}; /* g1110 = vec4_new(gx10.w, gy10.w, gz10.w, gw10.w); */

    vec4 g0011 = {gx11[0], gy11[0], gz11[0], gw11[0]}; /* g0011 = vec4_new(gx11.x, gy11.x, gz11.x, gw11.x); */
    vec4 g0111 = {gx11[2], gy11[2], gz11[2], gw11[2]}; /* g0111 = vec4_new(gx11.z, gy11.z, gz11.z, gw11.z); */
    vec4 g1011 = {gx11[1], gy11[1], gz11[1], gw11[1]}; /* g1011 = vec4_new(gx11.y, gy11.y, gz11.y, gw11.y); */
    vec4 g1111 = {gx11[3], gy11[3], gz11[3], gw11[3]}; /* g1111 = vec4_new(gx11.w, gy11.w, gz11.w, gw11.w); */

    _noiseDetail_gradNorm_vec4(g0000, g0100, g1000, g1100);
    _noiseDetail_gradNorm_vec4(g0001, g0101, g1001, g1101);
    _noiseDetail_gradNorm_vec4(g0010, g0110, g1010, g1110);
    _noiseDetail_gradNorm_vec4(g0011, g0111, g1011, g1111);

    /* ------------ */

    float n0000 = vec4_dot(g0000, Pf0); /* n0000 = dot(g0000, Pf0) */

    /* n1000 = dot(g1000, vec4_new(Pf1.x, Pf0.y, Pf0.z, Pf0.w)) */
    vec4 n1000d = {Pf1[0], Pf0[1], Pf0[2], Pf0[3]};
    float n1000 = vec4_dot(g1000, n1000d);

    /* n0100 = dot(g0100, vec4_new(Pf0.x, Pf1.y, Pf0.z, Pf0.w)) */
    vec4 n0100d = {Pf0[0], Pf1[1], Pf0[2], Pf0[3]};
    float n0100 = vec4_dot(g0100, n0100d);

    /* n1100 = dot(g1100, vec4_new(Pf1.x, Pf1.y, Pf0.z, Pf0.w)) */
    vec4 n1100d = {Pf1[0], Pf1[1], Pf0[2], Pf0[3]};
    float n1100 = vec4_dot(g1100, n1100d);

    /* n0010 = dot(g0010, vec4_new(Pf0.x, Pf0.y, Pf1.z, Pf0.w)) */
    vec4 n0010d = {Pf0[0], Pf0[1], Pf1[2], Pf0[3]};
    float n0010 = vec4_dot(g0010, n0010d);

    /* n1010 = dot(g1010, vec4_new(Pf1.x, Pf0.y, Pf1.z, Pf0.w)) */
    vec4 n1010d = {Pf1[0], Pf0[1], Pf1[2], Pf0[3]};
    float n1010 = vec4_dot(g1010, n1010d);

    /* n0110 = dot(g0110, vec4_new(Pf0.x, Pf1.y, Pf1.z, Pf0.w)) */
    vec4 n0110d = {Pf0[0], Pf1[1], Pf1[2], Pf0[3]};
    float n0110 = vec4_dot(g0110, n0110d);

    /* n1110 = dot(g1110, vec4_new(Pf1.x, Pf1.y, Pf1.z, Pf0.w)) */
    vec4 n1110d = {Pf1[0], Pf1[1], Pf1[2], Pf0[3]};
    float n1110 = vec4_dot(g1110, n1110d);

    /* n0001 = dot(g0001, vec4_new(Pf0.x, Pf0.y, Pf0.z, Pf1.w)) */
    vec4 n0001d = {Pf0[0], Pf0[1], Pf0[2], Pf1[3]};
    float n0001 = vec4_dot(g0001, n0001d);

    /* n1001 = dot(g1001, vec4_new(Pf1.x, Pf0.y, Pf0.z, Pf1.w)) */
    vec4 n1001d = {Pf1[0], Pf0[1], Pf0[2], Pf1[3]};
    float n1001 = vec4_dot(g1001, n1001d);

    /* n0101 = dot(g0101, vec4_new(Pf0.x, Pf1.y, Pf0.z, Pf1.w)) */
    vec4 n0101d = {Pf0[0], Pf1[1], Pf0[2], Pf1[3]};
    float n0101 = vec4_dot(g0101, n0101d);

    /* n1101 = dot(g1101, vec4_new(Pf1.x, Pf1.y, Pf0.z, Pf1.w)) */
    vec4 n1101d = {Pf1[0], Pf1[1], Pf0[2], Pf1[3]};
    float n1101 = vec4_dot(g1101, n1101d);

    /* n0011 = dot(g0011, vec4_new(Pf0.x, Pf0.y, Pf1.z, Pf1.w)) */
    vec4 n0011d = {Pf0[0], Pf0[1], Pf1[2], Pf1[3]};
    float n0011 = vec4_dot(g0011, n0011d);

    /* n1011 = dot(g1011, vec4_new(Pf1.x, Pf0.y, Pf1.z, Pf1.w)) */
    vec4 n1011d = {Pf1[0], Pf0[1], Pf1[2], Pf1[3]};
    float n1011 = vec4_dot(g1011, n1011d);

    /* n0111 = dot(g0111, vec4_new(Pf0.x, Pf1.y, Pf1.z, Pf1.w)) */
    vec4 n0111d = {Pf0[0], Pf1[1], Pf1[2], Pf1[3]};
    float n0111 = vec4_dot(g0111, n0111d);

    float n1111 = vec4_dot(g1111, Pf1); /* n1111 = dot(g1111, Pf1) */

    /* ------------ */

    vec4 fade_xyzw;
    _noiseDetail_fade_vec4(Pf0, fade_xyzw); /* fade_xyzw = fade(Pf0) */

    /* n_0w = lerp(vec4_new(n0000, n1000, n0100, n1100), vec4_new(n0001, n1001, n0101, n1101), fade_xyzw.w) */
    vec4 n_0w1 = {n0000, n1000, n0100, n1100};
    vec4 n_0w2 = {n0001, n1001, n0101, n1101};
    vec4 n_0w;
    vec4_lerp(n_0w1, n_0w2, fade_xyzw[3], n_0w);

    /* n_1w = lerp(vec4_new(n0010, n1010, n0110, n1110), vec4_new(n0011, n1011, n0111, n1111), fade_xyzw.w) */
    vec4 n_1w1 = {n0010, n1010, n0110, n1110};
    vec4 n_1w2 = {n0011, n1011, n0111, n1111};
    vec4 n_1w;
    vec4_lerp(n_1w1, n_1w2, fade_xyzw[3], n_1w);

    /* n_zw = lerp(n_0w, n_1w, fade_xyzw.z) */
    vec4 n_zw;
    vec4_lerp(n_0w, n_1w, fade_xyzw[2], n_zw);

    /* n_yzw = lerp(vec2_new(n_zw.x, n_zw.y), vec2_new(n_zw.z, n_zw.w), fade_xyzw.y) */
    vec2 n_yzw;
    vec2 n_yzw1 = {n_zw[0], n_zw[1]};
    vec2 n_yzw2 = {n_zw[2], n_zw[3]};
    vec2_lerp(n_yzw1, n_yzw2, fade_xyzw[1], n_yzw);

    /* n_xyzw = lerp(n_yzw.x, n_yzw.y, fade_xyzw.x) */
    float n_xyzw = lerp(n_yzw[0], n_yzw[1], fade_xyzw[0]);

    return n_xyzw * 2.2f;
}

/*!
 * @brief Classic perlin noise
 *
 * @param[in]  point  3D vector
 * @returns           perlin noise value
 */
PLAY_CGLM_INLINE
float
perlin_vec3(vec3 point)
{
    /* Integer part of p for indexing */
    vec3 Pi0;
    vec3_floor(point, Pi0); /* Pi0 = floor(point); */

    /* Integer part + 1 */
    vec3 Pi1;
    vec3_adds(Pi0, 1.0f, Pi1); /* Pi1 = Pi0 + 1.0f; */

    vec3_mods(Pi0, 289.0f, Pi0); /* Pi0 = mod(Pi0, 289.0f); */
    vec3_mods(Pi1, 289.0f, Pi1); /* Pi1 = mod(Pi1, 289.0f); */

    /* Fractional part of p for interpolation */
    vec3 Pf0;
    vec3_fract(point, Pf0);

    /* Fractional part - 1.0 */
    vec3 Pf1;
    vec3_subs(Pf0, 1.0f, Pf1);

    vec4 ix = {Pi0[0], Pi1[0], Pi0[0], Pi1[0]};
    vec4 iy = {Pi0[1], Pi0[1], Pi1[1], Pi1[1]};
    vec4 iz0 = {Pi0[2], Pi0[2], Pi0[2], Pi0[2]}; /* iz0 = vec4_new(Pi0.z); */
    vec4 iz1 = {Pi1[2], Pi1[2], Pi1[2], Pi1[2]}; /* iz1 = vec4_new(Pi1.z); */

    /* ------------ */

    /* ixy = permute(permute(ix) + iy) */
    vec4 ixy;
    _noiseDetail_permute(ix, ixy); /* ixy = permute(ix) */
    vec4_add(ixy, iy, ixy); /* ixy += iy; */
    _noiseDetail_permute(ixy, ixy); /* ixy = permute(ixy) */

    /* ixy0 = permute(ixy + iz0) */
    vec4 ixy0;
    vec4_add(ixy, iz0, ixy0); /* ixy0 = ixy + iz0 */
    _noiseDetail_permute(ixy0, ixy0); /* ixy0 = permute(ixy0) */

    /* ixy1 = permute(ixy + iz1) */
    vec4 ixy1;
    vec4_add(ixy, iz1, ixy1); /* ixy1 = ixy, iz1 */
    _noiseDetail_permute(ixy1, ixy1); /* ixy1 = permute(ixy1) */

    /* ------------ */

    vec4 gx0, gy0, gz0;
    _noiseDetail_i2gxyz(ixy0, gx0, gy0, gz0);

    vec4 gx1, gy1, gz1;
    _noiseDetail_i2gxyz(ixy1, gx1, gy1, gz1);

    /* ------------ */

    vec3 g000 = {gx0[0], gy0[0], gz0[0]}; /* g000 = vec3_new(gx0.x, gy0.x, gz0.x); */
    vec3 g100 = {gx0[1], gy0[1], gz0[1]}; /* g100 = vec3_new(gx0.y, gy0.y, gz0.y); */
    vec3 g010 = {gx0[2], gy0[2], gz0[2]}; /* g010 = vec3_new(gx0.z, gy0.z, gz0.z); */
    vec3 g110 = {gx0[3], gy0[3], gz0[3]}; /* g110 = vec3_new(gx0.w, gy0.w, gz0.w); */

    vec3 g001 = {gx1[0], gy1[0], gz1[0]}; /* g001 = vec3_new(gx1.x, gy1.x, gz1.x); */
    vec3 g101 = {gx1[1], gy1[1], gz1[1]}; /* g101 = vec3_new(gx1.y, gy1.y, gz1.y); */
    vec3 g011 = {gx1[2], gy1[2], gz1[2]}; /* g011 = vec3_new(gx1.z, gy1.z, gz1.z); */
    vec3 g111 = {gx1[3], gy1[3], gz1[3]}; /* g111 = vec3_new(gx1.w, gy1.w, gz1.w); */

    _noiseDetail_gradNorm_vec3(g000, g010, g100, g110);
    _noiseDetail_gradNorm_vec3(g001, g011, g101, g111);

    /* ------------ */

    float n000 = vec3_dot(g000, Pf0); /* n000 = dot(g000, Pf0) */

    /* n100 = dot(g100, vec3_new(Pf1.x, Pf0.y, Pf0.z)) */
    vec3 n100d = {Pf1[0], Pf0[1], Pf0[2]};
    float n100 = vec3_dot(g100, n100d);

    /* n010 = dot(g010, vec3_new(Pf0.x, Pf1.y, Pf0.z)) */
    vec3 n010d = {Pf0[0], Pf1[1], Pf0[2]};
    float n010 = vec3_dot(g010, n010d);

    /* n110 = dot(g110, vec3_new(Pf1.x, Pf1.y, Pf0.z)) */
    vec3 n110d = {Pf1[0], Pf1[1], Pf0[2]};
    float n110 = vec3_dot(g110, n110d);

    /* n001 = dot(g001, vec3_new(Pf0.x, Pf0.y, Pf1.z)) */
    vec3 n001d = {Pf0[0], Pf0[1], Pf1[2]};
    float n001 = vec3_dot(g001, n001d);

    /* n101 = dot(g101, vec3_new(Pf1.x, Pf0.y, Pf1.z)) */
    vec3 n101d = {Pf1[0], Pf0[1], Pf1[2]};
    float n101 = vec3_dot(g101, n101d);

    /* n011 = dot(g011, vec3_new(Pf0.x, Pf1.y, Pf1.z)) */
    vec3 n011d = {Pf0[0], Pf1[1], Pf1[2]};
    float n011 = vec3_dot(g011, n011d);

    float n111 = vec3_dot(g111, Pf1); /* n111 = dot(g111, Pf1) */

    /* ------------ */

    vec3 fade_xyz;
    _noiseDetail_fade_vec3(Pf0, fade_xyz); /* fade_xyz = fade(Pf0) */

    /* n_z = lerp(vec4_new(n000, n100, n010, n110), vec4_new(n001, n101, n011, n111), fade_xyz.z); */
    vec4 n_z;
    vec4 n_z1 = {n000, n100, n010, n110};
    vec4 n_z2 = {n001, n101, n011, n111};
    vec4_lerp(n_z1, n_z2, fade_xyz[2], n_z);

    /* vec2 n_yz = lerp(vec2_new(n_z.x, n_z.y), vec2_new(n_z.z, n_z.w), fade_xyz.y); */
    vec2 n_yz;
    vec2 n_yz1 = {n_z[0], n_z[1]};
    vec2 n_yz2 = {n_z[2], n_z[3]};
    vec2_lerp(n_yz1, n_yz2, fade_xyz[1], n_yz);

    /* n_xyz = lerp(n_yz.x, n_yz.y, fade_xyz.x); */
    float n_xyz = lerp(n_yz[0], n_yz[1], fade_xyz[0]);

    return n_xyz * 2.2f;
}

/*!
 * @brief Classic perlin noise
 *
 * @param[in]  point  2D vector
 * @returns           perlin noise value
 */
PLAY_CGLM_INLINE
float
perlin_vec2(vec2 point)
{

    /* Integer part of p for indexing */
    /* Pi = floor(vec4_new(point.x, point.y, point.x, point.y)) + vec4_new(0.0, 0.0, 1.0, 1.0); */
    vec4 Pi = {point[0], point[1], point[0], point[1]}; /* Pi = vec4_new(point.x, point.y, point.x, point.y) */
    vec4_floor(Pi, Pi); /* Pi = floor(Pi) */
    Pi[2] += 1.0f; /* Pi.z += 1.0 */
    Pi[3] += 1.0f; /* Pi.w += 1.0 */

    /* Fractional part of p for interpolation */
    /* vec<4, T, Q> Pf = glm::fract(vec<4, T, Q>(Position.x, Position.y, Position.x, Position.y)) - vec<4, T, Q>(0.0, 0.0, 1.0, 1.0); */
    vec4 Pf = {point[0], point[1], point[0], point[1]}; /* Pf = vec4_new(point.x, point.y, point.x, point.y) */
    vec4_fract(Pf, Pf); /* Pf = fract(Pf) */
    Pf[2] -= 1.0f; /* Pf.z -= 1.0 */
    Pf[3] -= 1.0f; /* Pf.w -= 1.0 */

    /* Mod to avoid truncation effects in permutation */
    vec4_mods(Pi, 289.0f, Pi); /* Pi = mod(Pi, 289.0f); */

    vec4 ix = {Pi[0], Pi[2], Pi[0], Pi[2]}; /* ix = vec4_new(Pi.x, Pi.z, Pi.x, Pi.z) */
    vec4 iy = {Pi[1], Pi[1], Pi[3], Pi[3]}; /* iy = vec4_new(Pi.y, Pi.y, Pi.w, Pi.w) */
    vec4 fx = {Pf[0], Pf[2], Pf[0], Pf[2]}; /* fx = vec4_new(Pf.x, Pf.z, Pf.x, Pf.z) */
    vec4 fy = {Pf[1], Pf[1], Pf[3], Pf[3]}; /* fy = vec4_new(Pf.y, Pf.y, Pf.w, Pf.w) */

    /* ------------ */

    /* i = permute(permute(ix) + iy); */
    vec4 i;
    _noiseDetail_permute(ix, i); /* i = permute(ix) */
    vec4_add(i, iy, i); /* i += iy; */
    _noiseDetail_permute(i, i); /* i = permute(i) */

    /* ------------ */

    vec4 gx, gy;
    _noiseDetail_i2gxy(i, gx, gy);

    /* ------------ */

    vec2 g00 = {gx[0], gy[0]}; /* g00 = vec2_new(gx.x, gy.x) */
    vec2 g10 = {gx[1], gy[1]}; /* g10 = vec2_new(gx.y, gy.y) */
    vec2 g01 = {gx[2], gy[2]}; /* g01 = vec2_new(gx.z, gy.z) */
    vec2 g11 = {gx[3], gy[3]}; /* g11 = vec2_new(gx.w, gy.w) */

    _noiseDetail_gradNorm_vec2(g00, g01, g10, g11);

    /* ------------ */

    /* n00 = dot(g00, vec2_new(fx.x, fy.x)) */
    vec2 n00d = {fx[0], fy[0]}; /* n00d = vec2_new(fx.x, fy.x) */
    float n00 = vec2_dot(g00, n00d); /* n00 = dot(g00, n00d) */

    /* n10 = dot(g10, vec2_new(fx.y, fy.y)) */
    vec2 n10d = {fx[1], fy[1]}; /* n10d = vec2_new(fx.y, fy.y) */
    float n10 = vec2_dot(g10, n10d); /* n10 = dot(g10, n10d) */

    /* n01 = dot(g01, vec2_new(fx.z, fy.z)) */
    vec2 n01d = {fx[2], fy[2]}; /* n01d = vec2_new(fx.z, fy.z) */
    float n01 = vec2_dot(g01, n01d); /* n01 = dot(g01, n01d) */

    /* n11 = dot(g11, vec2_new(fx.w, fy.w)) */
    vec2 n11d = {fx[3], fy[3]}; /* n11d = vec2_new(fx.w, fy.w) */
    float n11 = vec2_dot(g11, n11d); /* n11 = dot(g11, n11d) */

    /* ------------ */

    /* fade_xyz = fade(vec2_new(Pf.x, Pf.y)) */
    vec2 fade_xy;
    vec2 temp2 = {Pf[0], Pf[1]}; /* temp = vec2_new(Pf.x, Pf.y) */
    _noiseDetail_fade_vec2(temp2, fade_xy); /* fade_xy = fade(temp) */

    /* n_x = lerp(vec2_new(n00, n01), vec2_new(n10, n11), fade_xy.x); */
    vec2 n_x;
    vec2 n_x1 = {n00, n01}; /* n_x1 = vec2_new(n00, n01) */
    vec2 n_x2 = {n10, n11}; /* n_x2 = vec2_new(n10, n11) */
    vec2_lerp(n_x1, n_x2, fade_xy[0], n_x); /* n_x = lerp(n_x1, n_x2, fade_xy.x) */

    /* T n_xy = mix(n_x.x, n_x.y, fade_xy.y); */
    /* n_xy = lerp(n_x.x, n_x.y, fade_xy.y); */
    float n_xy = lerp(n_x[0], n_x[1], fade_xy[1]);

    return n_xy * 2.3f;
}

/* Undefine all helper macros */

#undef _noiseDetail_mod289
#undef _noiseDetail_permute
#undef _noiseDetail_fade_vec4
#undef _noiseDetail_fade_vec3
#undef _noiseDetail_fade_vec2
#undef _noiseDetail_taylorInvSqrt
#undef _noiseDetail_gradNorm_vec4
#undef _noiseDetail_gradNorm_vec3
#undef _noiseDetail_gradNorm_vec2
#undef _noiseDetail_i2gxyzw
#undef _noiseDetail_i2gxyz
#undef _noiseDetail_i2gxy



/*** End of inlined file: noise.h ***/


/*** Start of inlined file: aabb2d.h ***/



/* DEPRECATED! use _diag */
#define aabb2d_size(aabb)         aabb2d_diag(aabb)

/*!
 * @brief make [aabb] zero
 *
 * @param[in, out]  aabb aabb
 */
PLAY_CGLM_INLINE
void
aabb2d_zero(vec2 aabb[2])
{
    vec2_zero(aabb[0]);
    vec2_zero(aabb[1]);
}

/*!
 * @brief copy all members of [aabb] to [dest]
 *
 * @param[in]  aabb source
 * @param[out] dest destination
 */
PLAY_CGLM_INLINE
void
aabb2d_copy(vec2 aabb[2], vec2 dest[2])
{
    vec2_copy(aabb[0], dest[0]);
    vec2_copy(aabb[1], dest[1]);
}

/*!
 * @brief apply transform to Axis-Aligned Bounding aabb
 *
 * @param[in]  aabb  bounding aabb
 * @param[in]  m    transform matrix
 * @param[out] dest transformed bounding aabb
 */
PLAY_CGLM_INLINE
void
aabb2d_transform(vec2 aabb[2], mat3 m, vec2 dest[2])
{
    vec2 v[2], xa, xb, ya, yb;

    vec2_scale(m[0], aabb[0][0], xa);
    vec2_scale(m[0], aabb[1][0], xb);

    vec2_scale(m[1], aabb[0][1], ya);
    vec2_scale(m[1], aabb[1][1], yb);

    /* translation + fmin(xa, xb) + fmin(ya, yb) */
    vec2_new(m[2], v[0]);
    vec2_minadd(xa, xb, v[0]);
    vec2_minadd(ya, yb, v[0]);

    /* translation + fmax(xa, xb) + fmax(ya, yb) */
    vec2_new(m[2], v[1]);
    vec2_maxadd(xa, xb, v[1]);
    vec2_maxadd(ya, yb, v[1]);

    vec2_copy(v[0], dest[0]);
    vec2_copy(v[1], dest[1]);
}

/*!
 * @brief merges two AABB bounding aabb and creates new one
 *
 * two aabb must be in same space, if one of aabb is in different space then
 * you should consider to convert it's space by aabb_space
 *
 * @param[in]  aabb1 bounding aabb 1
 * @param[in]  aabb2 bounding aabb 2
 * @param[out] dest merged bounding aabb
 */
PLAY_CGLM_INLINE
void
aabb2d_merge(vec2 aabb1[2], vec2 aabb2[2], vec2 dest[2])
{
    dest[0][0] = fmin(aabb1[0][0], aabb2[0][0]);
    dest[0][1] = fmin(aabb1[0][1], aabb2[0][1]);

    dest[1][0] = fmax(aabb1[1][0], aabb2[1][0]);
    dest[1][1] = fmax(aabb1[1][1], aabb2[1][1]);
}

/*!
 * @brief crops a bounding aabb with another one.
 *
 * this could be useful for getting a baabb which fits with view frustum and
 * object bounding aabbes. In this case you crop view frustum aabb with objects
 * aabb
 *
 * @param[in]  aabb     bounding aabb 1
 * @param[in]  cropAabb crop aabb
 * @param[out] dest     cropped bounding aabb
 */
PLAY_CGLM_INLINE
void
aabb2d_crop(vec2 aabb[2], vec2 cropAabb[2], vec2 dest[2])
{
    dest[0][0] = fmax(aabb[0][0], cropAabb[0][0]);
    dest[0][1] = fmax(aabb[0][1], cropAabb[0][1]);

    dest[1][0] = fmin(aabb[1][0], cropAabb[1][0]);
    dest[1][1] = fmin(aabb[1][1], cropAabb[1][1]);
}

/*!
 * @brief crops a bounding aabb with another one.
 *
 * this could be useful for getting a baabb which fits with view frustum and
 * object bounding aabbes. In this case you crop view frustum aabb with objects
 * aabb
 *
 * @param[in]  aabb      bounding aabb
 * @param[in]  cropAabb  crop aabb
 * @param[in]  clampAabb minimum aabb
 * @param[out] dest      cropped bounding aabb
 */
PLAY_CGLM_INLINE
void
aabb2d_crop_until(vec2 aabb[2],
                  vec2 cropAabb[2],
                  vec2 clampAabb[2],
                  vec2 dest[2])
{
    aabb2d_crop(aabb, cropAabb, dest);
    aabb2d_merge(clampAabb, dest, dest);
}

/*!
 * @brief invalidate AABB min and max values
 *
 * @param[in, out]  aabb bounding aabb
 */
PLAY_CGLM_INLINE
void
aabb2d_invalidate(vec2 aabb[2])
{
    vec2_fill(aabb[0], FLT_MAX);
    vec2_fill(aabb[1], -FLT_MAX);
}

/*!
 * @brief check if AABB is valid or not
 *
 * @param[in]  aabb bounding aabb
 */
PLAY_CGLM_INLINE
bool
aabb2d_isvalid(vec2 aabb[2])
{
    return vec2_max(aabb[0]) != FLT_MAX
           && vec2_min(aabb[1]) != -FLT_MAX;
}

/*!
 * @brief distance between of min and max
 *
 * @param[in]  aabb bounding aabb
 */
PLAY_CGLM_INLINE
float
aabb2d_diag(vec2 aabb[2])
{
    return vec2_distance(aabb[0], aabb[1]);
}

/*!
 * @brief size of aabb
 *
 * @param[in]  aabb bounding aabb
 * @param[out]  dest size
 */
PLAY_CGLM_INLINE
void
aabb2d_sizev(vec2 aabb[2], vec2 dest)
{
    vec2_sub(aabb[1], aabb[0], dest);
}

/*!
 * @brief radius of sphere which surrounds AABB
 *
 * @param[in]  aabb bounding aabb
 */
PLAY_CGLM_INLINE
float
aabb2d_radius(vec2 aabb[2])
{
    return aabb2d_diag(aabb) * 0.5f;
}

/*!
 * @brief computes center point of AABB
 *
 * @param[in]   aabb  bounding aabb
 * @param[out]  dest center of bounding aabb
 */
PLAY_CGLM_INLINE
void
aabb2d_center(vec2 aabb[2], vec2 dest)
{
    vec2_center(aabb[0], aabb[1], dest);
}

/*!
 * @brief check if two AABB intersects
 *
 * @param[in]   aabb    bounding aabb
 * @param[in]   other  other bounding aabb
 */
PLAY_CGLM_INLINE
bool
aabb2d_aabb(vec2 aabb[2], vec2 other[2])
{
    return (aabb[0][0] <= other[1][0] && aabb[1][0] >= other[0][0])
           && (aabb[0][1] <= other[1][1] && aabb[1][1] >= other[0][1]);
}

/*!
 * @brief check if AABB intersects with a circle
 *
 * Circle Representation in cglm: [center.x, center.y, radii]
 *
 * @param[in]   aabb   solid bounding aabb
 * @param[in]   c      solid circle
 */
PLAY_CGLM_INLINE
bool
aabb2d_circle(vec2 aabb[2], vec3 c)
{
    float dmin;
    int   a, b;

    a = (c[0] < aabb[0][0]) + (c[0] > aabb[1][0]);
    b = (c[1] < aabb[0][1]) + (c[1] > aabb[1][1]);

    dmin  = pow2((c[0] - aabb[!(a - 1)][0]) * (a != 0))
            + pow2((c[1] - aabb[!(b - 1)][1]) * (b != 0));

    return dmin <= pow2(c[2]);
}

/*!
 * @brief check if point is inside of AABB
 *
 * @param[in]   aabb    bounding aabb
 * @param[in]   point  point
 */
PLAY_CGLM_INLINE
bool
aabb2d_point(vec2 aabb[2], vec2 point)
{
    return (point[0] >= aabb[0][0] && point[0] <= aabb[1][0])
           && (point[1] >= aabb[0][1] && point[1] <= aabb[1][1]);
}

/*!
 * @brief check if AABB contains other AABB
 *
 * @param[in]   aabb    bounding aabb
 * @param[in]   other  other bounding aabb
 */
PLAY_CGLM_INLINE
bool
aabb2d_contains(vec2 aabb[2], vec2 other[2])
{
    return (aabb[0][0] <= other[0][0] && aabb[1][0] >= other[1][0])
           && (aabb[0][1] <= other[0][1] && aabb[1][1] >= other[1][1]);
}



/*** End of inlined file: aabb2d.h ***/


/*** Start of inlined file: box.h ***/



/*!
 * @brief apply transform to Axis-Aligned Bounding Box
 *
 * @param[in]  box  bounding box
 * @param[in]  m    transform matrix
 * @param[out] dest transformed bounding box
 */
PLAY_CGLM_INLINE
void
aabb_transform(vec3 box[2], mat4 m, vec3 dest[2])
{
    vec3 v[2], xa, xb, ya, yb, za, zb;

    vec3_scale(m[0], box[0][0], xa);
    vec3_scale(m[0], box[1][0], xb);

    vec3_scale(m[1], box[0][1], ya);
    vec3_scale(m[1], box[1][1], yb);

    vec3_scale(m[2], box[0][2], za);
    vec3_scale(m[2], box[1][2], zb);

    /* translation + fmin(xa, xb) + fmin(ya, yb) + fmin(za, zb) */
    vec3_new(m[3], v[0]);
    vec3_minadd(xa, xb, v[0]);
    vec3_minadd(ya, yb, v[0]);
    vec3_minadd(za, zb, v[0]);

    /* translation + fmax(xa, xb) + fmax(ya, yb) + fmax(za, zb) */
    vec3_new(m[3], v[1]);
    vec3_maxadd(xa, xb, v[1]);
    vec3_maxadd(ya, yb, v[1]);
    vec3_maxadd(za, zb, v[1]);

    vec3_copy(v[0], dest[0]);
    vec3_copy(v[1], dest[1]);
}

/*!
 * @brief merges two AABB bounding box and creates new one
 *
 * two box must be in same space, if one of box is in different space then
 * you should consider to convert it's space by box_space
 *
 * @param[in]  box1 bounding box 1
 * @param[in]  box2 bounding box 2
 * @param[out] dest merged bounding box
 */
PLAY_CGLM_INLINE
void
aabb_merge(vec3 box1[2], vec3 box2[2], vec3 dest[2])
{
    dest[0][0] = fmin(box1[0][0], box2[0][0]);
    dest[0][1] = fmin(box1[0][1], box2[0][1]);
    dest[0][2] = fmin(box1[0][2], box2[0][2]);

    dest[1][0] = fmax(box1[1][0], box2[1][0]);
    dest[1][1] = fmax(box1[1][1], box2[1][1]);
    dest[1][2] = fmax(box1[1][2], box2[1][2]);
}

/*!
 * @brief crops a bounding box with another one.
 *
 * this could be useful for getting a bbox which fits with view frustum and
 * object bounding boxes. In this case you crop view frustum box with objects
 * box
 *
 * @param[in]  box     bounding box 1
 * @param[in]  cropBox crop box
 * @param[out] dest    cropped bounding box
 */
PLAY_CGLM_INLINE
void
aabb_crop(vec3 box[2], vec3 cropBox[2], vec3 dest[2])
{
    dest[0][0] = fmax(box[0][0], cropBox[0][0]);
    dest[0][1] = fmax(box[0][1], cropBox[0][1]);
    dest[0][2] = fmax(box[0][2], cropBox[0][2]);

    dest[1][0] = fmin(box[1][0], cropBox[1][0]);
    dest[1][1] = fmin(box[1][1], cropBox[1][1]);
    dest[1][2] = fmin(box[1][2], cropBox[1][2]);
}

/*!
 * @brief crops a bounding box with another one.
 *
 * this could be useful for getting a bbox which fits with view frustum and
 * object bounding boxes. In this case you crop view frustum box with objects
 * box
 *
 * @param[in]  box      bounding box
 * @param[in]  cropBox  crop box
 * @param[in]  clampBox minimum box
 * @param[out] dest     cropped bounding box
 */
PLAY_CGLM_INLINE
void
aabb_crop_until(vec3 box[2],
                vec3 cropBox[2],
                vec3 clampBox[2],
                vec3 dest[2])
{
    aabb_crop(box, cropBox, dest);
    aabb_merge(clampBox, dest, dest);
}

/*!
 * @brief check if AABB intersects with frustum planes
 *
 * this could be useful for frustum culling using AABB.
 *
 * OPTIMIZATION HINT:
 *  if planes order is similar to LEFT, RIGHT, BOTTOM, TOP, NEAR, FAR
 *  then this method should run even faster because it would only use two
 *  planes if object is not inside the two planes
 *  fortunately cglm extracts planes as this order! just pass what you got!
 *
 * @param[in]  box     bounding box
 * @param[in]  planes  frustum planes
 */
PLAY_CGLM_INLINE
bool
aabb_frustum(vec3 box[2], vec4 planes[6])
{
    float *p, dp;
    int    i;

    for (i = 0; i < 6; i++)
    {
        p  = planes[i];
        dp = p[0] * box[p[0] > 0.0f][0]
             + p[1] * box[p[1] > 0.0f][1]
             + p[2] * box[p[2] > 0.0f][2];

        if (dp < -p[3])
            return false;
    }

    return true;
}

/*!
 * @brief invalidate AABB min and max values
 *
 * @param[in, out]  box bounding box
 */
PLAY_CGLM_INLINE
void
aabb_invalidate(vec3 box[2])
{
    vec3_broadcast(FLT_MAX,  box[0]);
    vec3_broadcast(-FLT_MAX, box[1]);
}

/*!
 * @brief check if AABB is valid or not
 *
 * @param[in]  box bounding box
 */
PLAY_CGLM_INLINE
bool
aabb_isvalid(vec3 box[2])
{
    return vec3_max(box[0]) != FLT_MAX
           && vec3_min(box[1]) != -FLT_MAX;
}

/*!
 * @brief distance between of min and max
 *
 * @param[in]  box bounding box
 */
PLAY_CGLM_INLINE
float
aabb_size(vec3 box[2])
{
    return vec3_distance(box[0], box[1]);
}

/*!
 * @brief radius of sphere which surrounds AABB
 *
 * @param[in]  box bounding box
 */
PLAY_CGLM_INLINE
float
aabb_radius(vec3 box[2])
{
    return aabb_size(box) * 0.5f;
}

/*!
 * @brief computes center point of AABB
 *
 * @param[in]   box  bounding box
 * @param[out]  dest center of bounding box
 */
PLAY_CGLM_INLINE
void
aabb_center(vec3 box[2], vec3 dest)
{
    vec3_center(box[0], box[1], dest);
}

/*!
 * @brief check if two AABB intersects
 *
 * @param[in]   box    bounding box
 * @param[in]   other  other bounding box
 */
PLAY_CGLM_INLINE
bool
aabb_aabb(vec3 box[2], vec3 other[2])
{
    return (box[0][0] <= other[1][0] && box[1][0] >= other[0][0])
           && (box[0][1] <= other[1][1] && box[1][1] >= other[0][1])
           && (box[0][2] <= other[1][2] && box[1][2] >= other[0][2]);
}

/*!
 * @brief check if AABB intersects with sphere
 *
 * https://github.com/erich666/GraphicsGems/blob/master/gems/BoxSphere.c
 * Solid Box - Solid Sphere test.
 *
 * Sphere Representation in cglm: [center.x, center.y, center.z, radii]
 *
 * @param[in]   box    solid bounding box
 * @param[in]   s      solid sphere
 */
PLAY_CGLM_INLINE
bool
aabb_sphere(vec3 box[2], vec4 s)
{
    float dmin;
    int   a, b, c;

    a = (s[0] < box[0][0]) + (s[0] > box[1][0]);
    b = (s[1] < box[0][1]) + (s[1] > box[1][1]);
    c = (s[2] < box[0][2]) + (s[2] > box[1][2]);

    dmin  = pow2((s[0] - box[!(a - 1)][0]) * (a != 0))
            + pow2((s[1] - box[!(b - 1)][1]) * (b != 0))
            + pow2((s[2] - box[!(c - 1)][2]) * (c != 0));

    return dmin <= pow2(s[3]);
}

/*!
 * @brief check if point is inside of AABB
 *
 * @param[in]   box    bounding box
 * @param[in]   point  point
 */
PLAY_CGLM_INLINE
bool
aabb_point(vec3 box[2], vec3 point)
{
    return (point[0] >= box[0][0] && point[0] <= box[1][0])
           && (point[1] >= box[0][1] && point[1] <= box[1][1])
           && (point[2] >= box[0][2] && point[2] <= box[1][2]);
}

/*!
 * @brief check if AABB contains other AABB
 *
 * @param[in]   box    bounding box
 * @param[in]   other  other bounding box
 */
PLAY_CGLM_INLINE
bool
aabb_contains(vec3 box[2], vec3 other[2])
{
    return (box[0][0] <= other[0][0] && box[1][0] >= other[1][0])
           && (box[0][1] <= other[0][1] && box[1][1] >= other[1][1])
           && (box[0][2] <= other[0][2] && box[1][2] >= other[1][2]);
}



/*** End of inlined file: box.h ***/


/*** Start of inlined file: color.h ***/



/*!
 * @brief averages the color channels into one value
 *
 * @param[in]  rgb RGB color
 */
PLAY_CGLM_INLINE
float
luminance(vec3 rgb)
{
    vec3 l = {0.212671f, 0.715160f, 0.072169f};
    return dot(rgb, l);
}



/*** End of inlined file: color.h ***/



/*** Start of inlined file: project.h ***/




/*** Start of inlined file: project_zo.h ***/



/*!
 * @brief maps the specified viewport coordinates into specified space [1]
 *        the matrix should contain projection matrix.
 *
 * if you don't have ( and don't want to have ) an inverse matrix then use
 * unproject version. You may use existing inverse of matrix in somewhere
 * else, this is why unprojecti exists to save save inversion cost
 *
 * [1] space:
 *  1- if m = invProj:     View Space
 *  2- if m = invViewProj: World Space
 *  3- if m = invMVP:      Object Space
 *
 * You probably want to map the coordinates into object space
 * so use invMVP as m
 *
 * Computing viewProj:
 *   mat4_mul(proj, view, viewProj);
 *   mat4_mul(viewProj, model, MVP);
 *   mat4_inv(viewProj, invMVP);
 *
 * @param[in]  pos      point/position in viewport coordinates
 * @param[in]  invMat   matrix (see brief)
 * @param[in]  vp       viewport as [x, y, width, height]
 * @param[out] dest     unprojected coordinates
 */
PLAY_CGLM_INLINE
void
unprojecti_zo(vec3 pos, mat4 invMat, vec4 vp, vec3 dest)
{
    vec4 v;

    v[0] = 2.0f * (pos[0] - vp[0]) / vp[2] - 1.0f;
    v[1] = 2.0f * (pos[1] - vp[1]) / vp[3] - 1.0f;
    v[2] = pos[2];
    v[3] = 1.0f;

    mat4_mulv(invMat, v, v);
    vec4_scale(v, 1.0f / v[3], v);
    vec3_new(v, dest);
}

/*!
 * @brief map object coordinates to window coordinates
 *
 * Computing MVP:
 *   mat4_mul(proj, view, viewProj);
 *   mat4_mul(viewProj, model, MVP);
 *
 * @param[in]  pos      object coordinates
 * @param[in]  m        MVP matrix
 * @param[in]  vp       viewport as [x, y, width, height]
 * @param[out] dest     projected coordinates
 */
PLAY_CGLM_INLINE
void
project_zo(vec3 pos, mat4 m, vec4 vp, vec3 dest)
{
    PLAY_CGLM_ALIGN(16) vec4 pos4;

    vec4_new(pos, 1.0f, pos4);

    mat4_mulv(m, pos4, pos4);
    vec4_scale(pos4, 1.0f / pos4[3], pos4); /* pos = pos / pos.w */

    dest[2] = pos4[2];

    vec4_scale(pos4, 0.5f, pos4);
    vec4_adds(pos4,  0.5f, pos4);

    dest[0] = pos4[0] * vp[2] + vp[0];
    dest[1] = pos4[1] * vp[3] + vp[1];
}

/*!
 * @brief map object's z coordinate to window coordinates
 *
 * Computing MVP:
 *   mat4_mul(proj, view, viewProj);
 *   mat4_mul(viewProj, model, MVP);
 *
 * @param[in]  v  object coordinates
 * @param[in]  m  MVP matrix
 *
 * @returns projected z coordinate
 */
PLAY_CGLM_INLINE
float
project_z_zo(vec3 v, mat4 m)
{
    float z, w;

    z = m[0][2] * v[0] + m[1][2] * v[1] + m[2][2] * v[2] + m[3][2];
    w = m[0][3] * v[0] + m[1][3] * v[1] + m[2][3] * v[2] + m[3][3];

    return z / w;
}



/*** End of inlined file: project_zo.h ***/


/*** Start of inlined file: project_no.h ***/



/*!
 * @brief maps the specified viewport coordinates into specified space [1]
 *        the matrix should contain projection matrix.
 *
 * if you don't have ( and don't want to have ) an inverse matrix then use
 * unproject version. You may use existing inverse of matrix in somewhere
 * else, this is why unprojecti exists to save save inversion cost
 *
 * [1] space:
 *  1- if m = invProj:     View Space
 *  2- if m = invViewProj: World Space
 *  3- if m = invMVP:      Object Space
 *
 * You probably want to map the coordinates into object space
 * so use invMVP as m
 *
 * Computing viewProj:
 *   mat4_mul(proj, view, viewProj);
 *   mat4_mul(viewProj, model, MVP);
 *   mat4_inv(viewProj, invMVP);
 *
 * @param[in]  pos      point/position in viewport coordinates
 * @param[in]  invMat   matrix (see brief)
 * @param[in]  vp       viewport as [x, y, width, height]
 * @param[out] dest     unprojected coordinates
 */
PLAY_CGLM_INLINE
void
unprojecti_no(vec3 pos, mat4 invMat, vec4 vp, vec3 dest)
{
    vec4 v;

    v[0] = 2.0f * (pos[0] - vp[0]) / vp[2] - 1.0f;
    v[1] = 2.0f * (pos[1] - vp[1]) / vp[3] - 1.0f;
    v[2] = 2.0f *  pos[2]                  - 1.0f;
    v[3] = 1.0f;

    mat4_mulv(invMat, v, v);
    vec4_scale(v, 1.0f / v[3], v);
    vec3_new(v, dest);
}

/*!
 * @brief map object coordinates to window coordinates
 *
 * Computing MVP:
 *   mat4_mul(proj, view, viewProj);
 *   mat4_mul(viewProj, model, MVP);
 *
 * @param[in]  pos      object coordinates
 * @param[in]  m        MVP matrix
 * @param[in]  vp       viewport as [x, y, width, height]
 * @param[out] dest     projected coordinates
 */
PLAY_CGLM_INLINE
void
project_no(vec3 pos, mat4 m, vec4 vp, vec3 dest)
{
    PLAY_CGLM_ALIGN(16) vec4 pos4;

    vec4_new(pos, 1.0f, pos4);

    mat4_mulv(m, pos4, pos4);
    vec4_scale(pos4, 1.0f / pos4[3], pos4); /* pos = pos / pos.w */
    vec4_scale(pos4, 0.5f, pos4);
    vec4_adds(pos4,  0.5f, pos4);

    dest[0] = pos4[0] * vp[2] + vp[0];
    dest[1] = pos4[1] * vp[3] + vp[1];
    dest[2] = pos4[2];
}

/*!
 * @brief map object's z coordinate to window coordinates
 *
 * Computing MVP:
 *   mat4_mul(proj, view, viewProj);
 *   mat4_mul(viewProj, model, MVP);
 *
 * @param[in]  v  object coordinates
 * @param[in]  m  MVP matrix
 *
 * @returns projected z coordinate
 */
PLAY_CGLM_INLINE
float
project_z_no(vec3 v, mat4 m)
{
    float z, w;

    z = m[0][2] * v[0] + m[1][2] * v[1] + m[2][2] * v[2] + m[3][2];
    w = m[0][3] * v[0] + m[1][3] * v[1] + m[2][3] * v[2] + m[3][3];

    return 0.5f * (z / w) + 0.5f;
}



/*** End of inlined file: project_no.h ***/

/*!
 * @brief maps the specified viewport coordinates into specified space [1]
 *        the matrix should contain projection matrix.
 *
 * if you don't have ( and don't want to have ) an inverse matrix then use
 * unproject version. You may use existing inverse of matrix in somewhere
 * else, this is why unprojecti exists to save save inversion cost
 *
 * [1] space:
 *  1- if m = invProj:     View Space
 *  2- if m = invViewProj: World Space
 *  3- if m = invMVP:      Object Space
 *
 * You probably want to map the coordinates into object space
 * so use invMVP as m
 *
 * Computing viewProj:
 *   mat4_mul(proj, view, viewProj);
 *   mat4_mul(viewProj, model, MVP);
 *   mat4_inv(viewProj, invMVP);
 *
 * @param[in]  pos      point/position in viewport coordinates
 * @param[in]  invMat   matrix (see brief)
 * @param[in]  vp       viewport as [x, y, width, height]
 * @param[out] dest     unprojected coordinates
 */
PLAY_CGLM_INLINE
void
unprojecti(vec3 pos, mat4 invMat, vec4 vp, vec3 dest)
{
#if PLAY_CGLM_CONFIG_CLIP_CONTROL & PLAY_CGLM_CLIP_CONTROL_ZO_BIT
    unprojecti_zo(pos, invMat, vp, dest);
#elif PLAY_CGLM_CONFIG_CLIP_CONTROL & PLAY_CGLM_CLIP_CONTROL_NO_BIT
    unprojecti_no(pos, invMat, vp, dest);
#endif
}

/*!
 * @brief maps the specified viewport coordinates into specified space [1]
 *        the matrix should contain projection matrix.
 *
 * this is same as unprojecti except this function get inverse matrix for
 * you.
 *
 * [1] space:
 *  1- if m = proj:     View Space
 *  2- if m = viewProj: World Space
 *  3- if m = MVP:      Object Space
 *
 * You probably want to map the coordinates into object space
 * so use MVP as m
 *
 * Computing viewProj and MVP:
 *   mat4_mul(proj, view, viewProj);
 *   mat4_mul(viewProj, model, MVP);
 *
 * @param[in]  pos      point/position in viewport coordinates
 * @param[in]  m        matrix (see brief)
 * @param[in]  vp       viewport as [x, y, width, height]
 * @param[out] dest     unprojected coordinates
 */
PLAY_CGLM_INLINE
void
unproject(vec3 pos, mat4 m, vec4 vp, vec3 dest)
{
    mat4 inv;
    mat4_inv(m, inv);
    unprojecti(pos, inv, vp, dest);
}

/*!
 * @brief map object coordinates to window coordinates
 *
 * Computing MVP:
 *   mat4_mul(proj, view, viewProj);
 *   mat4_mul(viewProj, model, MVP);
 *
 * @param[in]  pos      object coordinates
 * @param[in]  m        MVP matrix
 * @param[in]  vp       viewport as [x, y, width, height]
 * @param[out] dest     projected coordinates
 */
PLAY_CGLM_INLINE
void
project(vec3 pos, mat4 m, vec4 vp, vec3 dest)
{
#if PLAY_CGLM_CONFIG_CLIP_CONTROL & PLAY_CGLM_CLIP_CONTROL_ZO_BIT
    project_zo(pos, m, vp, dest);
#elif PLAY_CGLM_CONFIG_CLIP_CONTROL & PLAY_CGLM_CLIP_CONTROL_NO_BIT
    project_no(pos, m, vp, dest);
#endif
}

/*!
 * @brief map object's z coordinate to window coordinates
 *
 * Computing MVP:
 *   mat4_mul(proj, view, viewProj);
 *   mat4_mul(viewProj, model, MVP);
 *
 * @param[in]  v  object coordinates
 * @param[in]  m  MVP matrix
 *
 * @returns projected z coordinate
 */
PLAY_CGLM_INLINE
float
project_z(vec3 v, mat4 m)
{
#if PLAY_CGLM_CONFIG_CLIP_CONTROL & PLAY_CGLM_CLIP_CONTROL_ZO_BIT
    return project_z_zo(v, m);
#elif PLAY_CGLM_CONFIG_CLIP_CONTROL & PLAY_CGLM_CLIP_CONTROL_NO_BIT
    return project_z_no(v, m);
#endif
}

/*!
 * @brief define a picking region
 *
 * @param[in]  center   center [x, y] of a picking region in window coordinates
 * @param[in]  size     size [width, height] of the picking region in window coordinates
 * @param[in]  vp       viewport as [x, y, width, height]
 * @param[out] dest     projected coordinates
 */
PLAY_CGLM_INLINE
void
pickmatrix(vec2 center, vec2 size, vec4 vp, mat4 dest)
{
    mat4 res;
    vec3 v;

    if (size[0] <= 0.0f || size[1] <= 0.0f)
        return;

    /* Translate and scale the picked region to the entire window */
    v[0] = (vp[2] - 2.0f * (center[0] - vp[0])) / size[0];
    v[1] = (vp[3] - 2.0f * (center[1] - vp[1])) / size[1];
    v[2] = 0.0f;

    translate_make(res, v);

    v[0] = vp[2] / size[0];
    v[1] = vp[3] / size[1];
    v[2] = 1.0f;

    scale(res, v);

    mat4_copy(res, dest);
}



/*** End of inlined file: project.h ***/


/*** Start of inlined file: sphere.h ***/



/*
  Sphere Representation in cglm: [center.x, center.y, center.z, radii]

  You could use this representation or you can convert it to vec4 before call
  any function
 */

/*!
 * @brief helper for getting sphere radius
 *
 * @param[in]   s  sphere
 *
 * @return returns radii
 */
PLAY_CGLM_INLINE
float
sphere_radii(vec4 s)
{
    return s[3];
}

/*!
 * @brief apply transform to sphere, it is just wrapper for mat4_mulv3
 *
 * @param[in]  s    sphere
 * @param[in]  m    transform matrix
 * @param[out] dest transformed sphere
 */
PLAY_CGLM_INLINE
void
sphere_transform(vec4 s, mat4 m, vec4 dest)
{
    mat4_mulv3(m, s, 1.0f, dest);
    dest[3] = s[3];
}

/*!
 * @brief merges two spheres and creates a new one
 *
 * two sphere must be in same space, for instance if one in world space then
 * the other must be in world space too, not in local space.
 *
 * @param[in]  s1   sphere 1
 * @param[in]  s2   sphere 2
 * @param[out] dest merged/extended sphere
 */
PLAY_CGLM_INLINE
void
sphere_merge(vec4 s1, vec4 s2, vec4 dest)
{
    float dist, radii;

    dist  = vec3_distance(s1, s2);
    radii = dist + s1[3] + s2[3];

    radii = fmax(radii, s1[3]);
    radii = fmax(radii, s2[3]);

    vec3_center(s1, s2, dest);
    dest[3] = radii;
}

/*!
 * @brief check if two sphere intersects
 *
 * @param[in]   s1  sphere
 * @param[in]   s2  other sphere
 */
PLAY_CGLM_INLINE
bool
sphere_sphere(vec4 s1, vec4 s2)
{
    return vec3_distance2(s1, s2) <= pow2(s1[3] + s2[3]);
}

/*!
 * @brief check if sphere intersects with point
 *
 * @param[in]   s      sphere
 * @param[in]   point  point
 */
PLAY_CGLM_INLINE
bool
sphere_point(vec4 s, vec3 point)
{
    float rr;
    rr = s[3] * s[3];
    return vec3_distance2(point, s) <= rr;
}



/*** End of inlined file: sphere.h ***/


/*** Start of inlined file: ease.h ***/



PLAY_CGLM_INLINE
float
ease_linear(float t)
{
    return t;
}

PLAY_CGLM_INLINE
float
ease_sine_in(float t)
{
    return sinf((t - 1.0f) * PLAY_CGLM_PI_2f) + 1.0f;
}

PLAY_CGLM_INLINE
float
ease_sine_out(float t)
{
    return sinf(t * PLAY_CGLM_PI_2f);
}

PLAY_CGLM_INLINE
float
ease_sine_inout(float t)
{
    return 0.5f * (1.0f - cosf(t * PLAY_CGLM_PIf));
}

PLAY_CGLM_INLINE
float
ease_quad_in(float t)
{
    return t * t;
}

PLAY_CGLM_INLINE
float
ease_quad_out(float t)
{
    return -(t * (t - 2.0f));
}

PLAY_CGLM_INLINE
float
ease_quad_inout(float t)
{
    float tt;

    tt = t * t;
    if (t < 0.5f)
        return 2.0f * tt;

    return (-2.0f * tt) + (4.0f * t) - 1.0f;
}

PLAY_CGLM_INLINE
float
ease_cubic_in(float t)
{
    return t * t * t;
}

PLAY_CGLM_INLINE
float
ease_cubic_out(float t)
{
    float f;
    f = t - 1.0f;
    return f * f * f + 1.0f;
}

PLAY_CGLM_INLINE
float
ease_cubic_inout(float t)
{
    float f;

    if (t < 0.5f)
        return 4.0f * t * t * t;

    f = 2.0f * t - 2.0f;

    return 0.5f * f * f * f + 1.0f;
}

PLAY_CGLM_INLINE
float
ease_quart_in(float t)
{
    float f;
    f = t * t;
    return f * f;
}

PLAY_CGLM_INLINE
float
ease_quart_out(float t)
{
    float f;

    f = t - 1.0f;

    return f * f * f * (1.0f - t) + 1.0f;
}

PLAY_CGLM_INLINE
float
ease_quart_inout(float t)
{
    float f, g;

    if (t < 0.5f)
    {
        f = t * t;
        return 8.0f * f * f;
    }

    f = t - 1.0f;
    g = f * f;

    return -8.0f * g * g + 1.0f;
}

PLAY_CGLM_INLINE
float
ease_quint_in(float t)
{
    float f;
    f = t * t;
    return f * f * t;
}

PLAY_CGLM_INLINE
float
ease_quint_out(float t)
{
    float f, g;

    f = t - 1.0f;
    g = f * f;

    return g * g * f + 1.0f;
}

PLAY_CGLM_INLINE
float
ease_quint_inout(float t)
{
    float f, g;

    if (t < 0.5f)
    {
        f = t * t;
        return 16.0f * f * f * t;
    }

    f = 2.0f * t - 2.0f;
    g = f * f;

    return 0.5f * g * g * f + 1.0f;
}

PLAY_CGLM_INLINE
float
ease_exp_in(float t)
{
    if (t == 0.0f)
        return t;

    return powf(2.0f,  10.0f * (t - 1.0f));
}

PLAY_CGLM_INLINE
float
ease_exp_out(float t)
{
    if (t == 1.0f)
        return t;

    return 1.0f - powf(2.0f, -10.0f * t);
}

PLAY_CGLM_INLINE
float
ease_exp_inout(float t)
{
    if (t == 0.0f || t == 1.0f)
        return t;

    if (t < 0.5f)
        return 0.5f * powf(2.0f, (20.0f * t) - 10.0f);

    return -0.5f * powf(2.0f, (-20.0f * t) + 10.0f) + 1.0f;
}

PLAY_CGLM_INLINE
float
ease_circ_in(float t)
{
    return 1.0f - sqrtf(1.0f - (t * t));
}

PLAY_CGLM_INLINE
float
ease_circ_out(float t)
{
    return sqrtf((2.0f - t) * t);
}

PLAY_CGLM_INLINE
float
ease_circ_inout(float t)
{
    if (t < 0.5f)
        return 0.5f * (1.0f - sqrtf(1.0f - 4.0f * (t * t)));

    return 0.5f * (sqrtf(-((2.0f * t) - 3.0f) * ((2.0f * t) - 1.0f)) + 1.0f);
}

PLAY_CGLM_INLINE
float
ease_back_in(float t)
{
    float o, z;

    o = 1.70158f;
    z = ((o + 1.0f) * t) - o;

    return t * t * z;
}

PLAY_CGLM_INLINE
float
ease_back_out(float t)
{
    float o, z, n;

    o = 1.70158f;
    n = t - 1.0f;
    z = (o + 1.0f) * n + o;

    return n * n * z + 1.0f;
}

PLAY_CGLM_INLINE
float
ease_back_inout(float t)
{
    float o, z, n, m, s, x;

    o = 1.70158f;
    s = o * 1.525f;
    x = 0.5f;
    n = t / 0.5f;

    if (n < 1.0f)
    {
        z = (s + 1) * n - s;
        m = n * n * z;
        return x * m;
    }

    n -= 2.0f;
    z  = (s + 1.0f) * n + s;
    m  = (n * n * z) + 2;

    return x * m;
}

PLAY_CGLM_INLINE
float
ease_elast_in(float t)
{
    return sinf(13.0f * PLAY_CGLM_PI_2f * t) * powf(2.0f, 10.0f * (t - 1.0f));
}

PLAY_CGLM_INLINE
float
ease_elast_out(float t)
{
    return sinf(-13.0f * PLAY_CGLM_PI_2f * (t + 1.0f)) * powf(2.0f, -10.0f * t) + 1.0f;
}

PLAY_CGLM_INLINE
float
ease_elast_inout(float t)
{
    float a;

    a = 2.0f * t;

    if (t < 0.5f)
        return 0.5f * sinf(13.0f * PLAY_CGLM_PI_2f * a)
               * powf(2.0f, 10.0f * (a - 1.0f));

    return 0.5f * (sinf(-13.0f * PLAY_CGLM_PI_2f * a)
                   * powf(2.0f, -10.0f * (a - 1.0f)) + 2.0f);
}

PLAY_CGLM_INLINE
float
ease_bounce_out(float t)
{
    float tt;

    tt = t * t;

    if (t < (4.0f / 11.0f))
        return (121.0f * tt) / 16.0f;

    if (t < 8.0f / 11.0f)
        return ((363.0f / 40.0f) * tt) - ((99.0f / 10.0f) * t) + (17.0f / 5.0f);

    if (t < (9.0f / 10.0f))
        return (4356.0f / 361.0f) * tt
               - (35442.0f / 1805.0f) * t
               + (16061.0f / 1805.0f);

    return ((54.0f / 5.0f) * tt) - ((513.0f / 25.0f) * t) + (268.0f / 25.0f);
}

PLAY_CGLM_INLINE
float
ease_bounce_in(float t)
{
    return 1.0f - ease_bounce_out(1.0f - t);
}

PLAY_CGLM_INLINE
float
ease_bounce_inout(float t)
{
    if (t < 0.5f)
        return 0.5f * (1.0f - ease_bounce_out(t * 2.0f));

    return 0.5f * ease_bounce_out(t * 2.0f - 1.0f) + 0.5f;
}



/*** End of inlined file: ease.h ***/


/*** Start of inlined file: curve.h ***/



/*!
 * @brief helper function to calculate S*M*C multiplication for curves
 *
 * This function does not encourage you to use SMC,
 * instead it is a helper if you use SMC.
 *
 * if you want to specify S as vector then use more generic mat4_rmc() func.
 *
 * Example usage:
 *  B(s) = smc(s, PLAY_CGLM_BEZIER_MAT, (vec4){p0, c0, c1, p1})
 *
 * @param[in]  s  parameter between 0 and 1 (this will be [s3, s2, s, 1])
 * @param[in]  m  basis matrix
 * @param[in]  c  position/control vector
 *
 * @return B(s)
 */
PLAY_CGLM_INLINE
float
smc(float s, mat4 m, vec4 c)
{
    vec4 vs;
    vec4_cubic(s, vs);
    return mat4_rmc(vs, m, c);
}



/*** End of inlined file: curve.h ***/


/*** Start of inlined file: bezier.h ***/



#define PLAY_CGLM_BEZIER_MAT_INIT  {{-1.0f,  3.0f, -3.0f,  1.0f},                   \
                              { 3.0f, -6.0f,  3.0f,  0.0f},                   \
                              {-3.0f,  3.0f,  0.0f,  0.0f},                   \
                              { 1.0f,  0.0f,  0.0f,  0.0f}}
#define PLAY_CGLM_HERMITE_MAT_INIT {{ 2.0f, -3.0f,  0.0f,  1.0f},                   \
                              {-2.0f,  3.0f,  0.0f,  0.0f},                   \
                              { 1.0f, -2.0f,  1.0f,  0.0f},                   \
                              { 1.0f, -1.0f,  0.0f,  0.0f}}
/* for C only */
#define PLAY_CGLM_BEZIER_MAT  ((mat4)PLAY_CGLM_BEZIER_MAT_INIT)
#define PLAY_CGLM_HERMITE_MAT ((mat4)PLAY_CGLM_HERMITE_MAT_INIT)

#define PLAY_CGLM_DECASTEL_EPS   1e-9f
#define PLAY_CGLM_DECASTEL_MAX   1000
#define PLAY_CGLM_DECASTEL_SMALL 1e-20f

/*!
 * @brief cubic bezier interpolation
 *
 * Formula:
 *  B(s) = P0*(1-s)^3 + 3*C0*s*(1-s)^2 + 3*C1*s^2*(1-s) + P1*s^3
 *
 * similar result using matrix:
 *  B(s) = smc(t, PLAY_CGLM_BEZIER_MAT, (vec4){p0, c0, c1, p1})
 *
 * eq(smc(...), bezier(...)) should return TRUE
 *
 * @param[in]  s    parameter between 0 and 1
 * @param[in]  p0   begin point
 * @param[in]  c0   control point 1
 * @param[in]  c1   control point 2
 * @param[in]  p1   end point
 *
 * @return B(s)
 */
PLAY_CGLM_INLINE
float
bezier(float s, float p0, float c0, float c1, float p1)
{
    float x, xx, ss, xs3, a;

    x   = 1.0f - s;
    xx  = x * x;
    ss  = s * s;
    xs3 = (s - ss) * 3.0f;
    a   = p0 * xx + c0 * xs3;

    return a + s * (c1 * xs3 + p1 * ss - a);
}

/*!
 * @brief cubic hermite interpolation
 *
 * Formula:
 *  H(s) = P0*(2*s^3 - 3*s^2 + 1) + T0*(s^3 - 2*s^2 + s)
 *            + P1*(-2*s^3 + 3*s^2) + T1*(s^3 - s^2)
 *
 * similar result using matrix:
 *  H(s) = smc(t, PLAY_CGLM_HERMITE_MAT, (vec4){p0, p1, c0, c1})
 *
 * eq(smc(...), hermite(...)) should return TRUE
 *
 * @param[in]  s    parameter between 0 and 1
 * @param[in]  p0   begin point
 * @param[in]  t0   tangent 1
 * @param[in]  t1   tangent 2
 * @param[in]  p1   end point
 *
 * @return H(s)
 */
PLAY_CGLM_INLINE
float
hermite(float s, float p0, float t0, float t1, float p1)
{
    float ss, d, a, b, c, e, f;

    ss = s  * s;
    a  = ss + ss;
    c  = a  + ss;
    b  = a  * s;
    d  = s  * ss;
    f  = d  - ss;
    e  = b  - c;

    return p0 * (e + 1.0f) + t0 * (f - ss + s) + t1 * f - p1 * e;
}

/*!
 * @brief iterative way to solve cubic equation
 *
 * @param[in]  prm  parameter between 0 and 1
 * @param[in]  p0   begin point
 * @param[in]  c0   control point 1
 * @param[in]  c1   control point 2
 * @param[in]  p1   end point
 *
 * @return parameter to use in cubic equation
 */
PLAY_CGLM_INLINE
float
decasteljau(float prm, float p0, float c0, float c1, float p1)
{
    float u, v, a, b, c, d, e, f;
    int   i;

    if (prm - p0 < PLAY_CGLM_DECASTEL_SMALL)
        return 0.0f;

    if (p1 - prm < PLAY_CGLM_DECASTEL_SMALL)
        return 1.0f;

    u  = 0.0f;
    v  = 1.0f;

    for (i = 0; i < PLAY_CGLM_DECASTEL_MAX; i++)
    {
        /* de Casteljau Subdivision */
        a  = (p0 + c0) * 0.5f;
        b  = (c0 + c1) * 0.5f;
        c  = (c1 + p1) * 0.5f;
        d  = (a  + b)  * 0.5f;
        e  = (b  + c)  * 0.5f;
        f  = (d  + e)  * 0.5f; /* this one is on the curve! */

        /* The curve point is close enough to our wanted t */
        if (fabsf(f - prm) < PLAY_CGLM_DECASTEL_EPS)
            return clamp_zo((u  + v) * 0.5f);

        /* dichotomy */
        if (f < prm)
        {
            p0 = f;
            c0 = e;
            c1 = c;
            u  = (u  + v) * 0.5f;
        }
        else
        {
            c0 = a;
            c1 = d;
            p1 = f;
            v  = (u  + v) * 0.5f;
        }
    }

    return clamp_zo((u  + v) * 0.5f);
}



/*** End of inlined file: bezier.h ***/


/*** Start of inlined file: ray.h ***/
/*
 Functions:
   PLAY_CGLM_INLINE bool ray_triangle(vec3   origin,
                                     vec3   direction,
                                     vec3   v0,
                                     vec3   v1,
                                     vec3   v2,
                                     float *d);
 PLAY_CGLM_INLINE bool ray_sphere(vec3 origin,
                                 vec3 dir,
                                 vec4 s,
                                 float * __restrict t1,
                                 float * __restrict t2)
 PLAY_CGLM_INLINE void ray_at(vec3 orig, vec3 dir, float t, vec3 point);
*/




/*!
 * @brief Möller–Trumbore ray-triangle intersection algorithm
 *
 * @param[in] origin         origin of ray
 * @param[in] direction      direction of ray
 * @param[in] v0             first vertex of triangle
 * @param[in] v1             second vertex of triangle
 * @param[in] v2             third vertex of triangle
 * @param[in, out] d         distance to intersection
 * @return whether there is intersection
 */
PLAY_CGLM_INLINE
bool
ray_triangle(vec3   origin,
             vec3   direction,
             vec3   v0,
             vec3   v1,
             vec3   v2,
             float *d)
{
    vec3        edge1, edge2, p, t, q;
    float       det, inv_det, u, v, dist;
    const float epsilon = 0.000001f;

    vec3_sub(v1, v0, edge1);
    vec3_sub(v2, v0, edge2);
    vec3_cross(direction, edge2, p);

    det = vec3_dot(edge1, p);
    if (det > -epsilon && det < epsilon)
        return false;

    inv_det = 1.0f / det;

    vec3_sub(origin, v0, t);

    u = inv_det * vec3_dot(t, p);
    if (u < 0.0f || u > 1.0f)
        return false;

    vec3_cross(t, edge1, q);

    v = inv_det * vec3_dot(direction, q);
    if (v < 0.0f || u + v > 1.0f)
        return false;

    dist = inv_det * vec3_dot(edge2, q);

    if (d)
        *d = dist;

    return dist > epsilon;
}

/*!
 * @brief ray sphere intersection
 *
 * returns false if there is no intersection if true:
 *
 * - t1 > 0, t2 > 0: ray intersects the sphere at t1 and t2 both ahead of the origin
 * - t1 < 0, t2 > 0: ray starts inside the sphere, exits at t2
 * - t1 < 0, t2 < 0: no intersection ahead of the ray ( returns false )
 * - the caller can check if the intersection points (t1 and t2) fall within a
 *   specific range (for example, tmin < t1, t2 < tmax) to determine if the
 *   intersections are within a desired segment of the ray
 *
 * @param[in]  origin ray origin
 * @param[out] dir    normalized ray direction
 * @param[in]  s      sphere  [center.x, center.y, center.z, radii]
 * @param[in]  t1     near point1 (closer to origin)
 * @param[in]  t2     far point2 (farther from origin)
 *
 * @returns whether there is intersection
 */
PLAY_CGLM_INLINE
bool
ray_sphere(vec3 origin,
           vec3 dir,
           vec4 s,
           float * __restrict t1,
           float * __restrict t2)
{
    vec3  dp;
    float r2, ddp, dpp, dscr, q, tmp, _t1, _t2;

    vec3_sub(s, origin, dp);

    ddp = vec3_dot(dir, dp);
    dpp = vec3_norm2(dp);

    /* compute the remedy term for numerical stability */
    vec3_mulsubs(dir, ddp, dp); /* dp: remedy term */

    r2   = s[3] * s[3];
    dscr = r2 - vec3_norm2(dp);

    if (dscr < 0.0f)
    {
        /* no intersection */
        return false;
    }

    dscr = sqrtf(dscr);
    q    = (ddp >= 0.0f) ? (ddp + dscr) : (ddp - dscr);

    /*
       include Press, William H., Saul A. Teukolsky,
       William T. Vetterling, and Brian P. Flannery,
       "Numerical Recipes in C," Cambridge University Press, 1992.
     */
    _t1 = q;
    _t2 = (dpp - r2) / q;

    /* adjust t1 and t2 to ensure t1 is the closer intersection */
    if (_t1 > _t2)
    {
        tmp = _t1;
        _t1 = _t2;
        _t2 = tmp;
    }

    *t1 = _t1;
    *t2 = _t2;

    /* check if the closest intersection (t1) is behind the ray's origin */
    if (_t1 < 0.0f && _t2 < 0.0f)
    {
        /* both intersections are behind the ray, no visible intersection */
        return false;
    }

    return true;
}

/*!
 * @brief point using t by 𝐏(𝑡)=𝐀+𝑡𝐛
 *
 * @param[in]  orig  origin of ray
 * @param[in]  dir   direction of ray
 * @param[in]  t     parameter
 * @param[out] point point at t
 */
PLAY_CGLM_INLINE
void
ray_at(vec3 orig, vec3 dir, float t, vec3 point)
{
    vec3 dst;
    vec3_scale(dir, t, dst);
    vec3_add(orig, dst, point);
}

#endif

/*** End of inlined file: ray.h ***/


/*** Start of inlined file: affine2d.h ***/
/*
 Functions:
   PLAY_CGLM_INLINE void translate2d(mat3 m, vec2 v)
   PLAY_CGLM_INLINE void translate2d_to(mat3 m, vec2 v, mat3 dest)
   PLAY_CGLM_INLINE void translate2d_x(mat3 m, float x)
   PLAY_CGLM_INLINE void translate2d_y(mat3 m, float y)
   PLAY_CGLM_INLINE void translate2d_make(mat3 m, vec2 v)
   PLAY_CGLM_INLINE void scale2d_to(mat3 m, vec2 v, mat3 dest)
   PLAY_CGLM_INLINE void scale2d_make(mat3 m, vec2 v)
   PLAY_CGLM_INLINE void scale2d(mat3 m, vec2 v)
   PLAY_CGLM_INLINE void scale2d_uni(mat3 m, float s)
   PLAY_CGLM_INLINE void rotate2d_make(mat3 m, float angle)
   PLAY_CGLM_INLINE void rotate2d(mat3 m, float angle)
   PLAY_CGLM_INLINE void rotate2d_to(mat3 m, float angle, mat3 dest)
 */




/*!
 * @brief translate existing 2d transform matrix by v vector
 *        and stores result in same matrix
 *
 * @param[in, out]  m  affine transform
 * @param[in]       v  translate vector [x, y]
 */
PLAY_CGLM_INLINE
void
translate2d(mat3 m, vec2 v)
{
    m[2][0] = m[0][0] * v[0] + m[1][0] * v[1] + m[2][0];
    m[2][1] = m[0][1] * v[0] + m[1][1] * v[1] + m[2][1];
    m[2][2] = m[0][2] * v[0] + m[1][2] * v[1] + m[2][2];
}

/*!
 * @brief translate existing 2d transform matrix by v vector
 *        and store result in dest
 *
 * source matrix will remain same
 *
 * @param[in]  m    affine transform
 * @param[in]  v    translate vector [x, y]
 * @param[out] dest translated matrix
 */
PLAY_CGLM_INLINE
void
translate2d_to(mat3 m, vec2 v, mat3 dest)
{
    mat3_copy(m, dest);
    translate2d(dest, v);
}

/*!
 * @brief translate existing 2d transform matrix by x factor
 *
 * @param[in, out]  m  affine transform
 * @param[in]       x  x factor
 */
PLAY_CGLM_INLINE
void
translate2d_x(mat3 m, float x)
{
    m[2][0] = m[0][0] * x + m[2][0];
    m[2][1] = m[0][1] * x + m[2][1];
    m[2][2] = m[0][2] * x + m[2][2];
}

/*!
 * @brief translate existing 2d transform matrix by y factor
 *
 * @param[in, out]  m  affine transform
 * @param[in]       y  y factor
 */
PLAY_CGLM_INLINE
void
translate2d_y(mat3 m, float y)
{
    m[2][0] = m[1][0] * y + m[2][0];
    m[2][1] = m[1][1] * y + m[2][1];
    m[2][2] = m[1][2] * y + m[2][2];
}

/*!
 * @brief creates NEW translate 2d transform matrix by v vector
 *
 * @param[out]  m  affine transform
 * @param[in]   v  translate vector [x, y]
 */
PLAY_CGLM_INLINE
void
translate2d_make(mat3 m, vec2 v)
{
    mat3_identity(m);
    m[2][0] = v[0];
    m[2][1] = v[1];
}

/*!
 * @brief scale existing 2d transform matrix by v vector
 *        and store result in dest
 *
 * @param[in]  m    affine transform
 * @param[in]  v    scale vector [x, y]
 * @param[out] dest scaled matrix
 */
PLAY_CGLM_INLINE
void
scale2d_to(mat3 m, vec2 v, mat3 dest)
{
    dest[0][0] = m[0][0] * v[0];
    dest[0][1] = m[0][1] * v[0];
    dest[0][2] = m[0][2] * v[0];

    dest[1][0] = m[1][0] * v[1];
    dest[1][1] = m[1][1] * v[1];
    dest[1][2] = m[1][2] * v[1];

    dest[2][0] = m[2][0];
    dest[2][1] = m[2][1];
    dest[2][2] = m[2][2];
}

/*!
 * @brief creates NEW 2d scale matrix by v vector
 *
 * @param[out]  m  affine transform
 * @param[in]   v  scale vector [x, y]
 */
PLAY_CGLM_INLINE
void
scale2d_make(mat3 m, vec2 v)
{
    mat3_identity(m);
    m[0][0] = v[0];
    m[1][1] = v[1];
}

/*!
 * @brief scales existing 2d transform matrix by v vector
 *        and stores result in same matrix
 *
 * @param[in, out]  m  affine transform
 * @param[in]       v  scale vector [x, y]
 */
PLAY_CGLM_INLINE
void
scale2d(mat3 m, vec2 v)
{
    m[0][0] = m[0][0] * v[0];
    m[0][1] = m[0][1] * v[0];
    m[0][2] = m[0][2] * v[0];

    m[1][0] = m[1][0] * v[1];
    m[1][1] = m[1][1] * v[1];
    m[1][2] = m[1][2] * v[1];
}

/*!
 * @brief applies uniform scale to existing 2d transform matrix v = [s, s]
 *        and stores result in same matrix
 *
 * @param[in, out]  m  affine transform
 * @param[in]       s  scale factor
 */
PLAY_CGLM_INLINE
void
scale2d_uni(mat3 m, float s)
{
    m[0][0] = m[0][0] * s;
    m[0][1] = m[0][1] * s;
    m[0][2] = m[0][2] * s;

    m[1][0] = m[1][0] * s;
    m[1][1] = m[1][1] * s;
    m[1][2] = m[1][2] * s;
}

/*!
 * @brief creates NEW rotation matrix by angle around Z axis
 *
 * @param[out] m     affine transform
 * @param[in]  angle angle (radians)
 */
PLAY_CGLM_INLINE
void
rotate2d_make(mat3 m, float angle)
{
    float c, s;

    s = sinf(angle);
    c = cosf(angle);

    m[0][0] = c;
    m[0][1] = s;
    m[0][2] = 0;

    m[1][0] = -s;
    m[1][1] = c;
    m[1][2] = 0;

    m[2][0] = 0.0f;
    m[2][1] = 0.0f;
    m[2][2] = 1.0f;
}

/*!
 * @brief rotate existing 2d transform matrix around Z axis by angle
 *         and store result in same matrix
 *
 * @param[in, out]  m      affine transform
 * @param[in]       angle  angle (radians)
 */
PLAY_CGLM_INLINE
void
rotate2d(mat3 m, float angle)
{
    float m00 = m[0][0],  m10 = m[1][0],
                          m01 = m[0][1],  m11 = m[1][1],
                                          m02 = m[0][2],  m12 = m[1][2];
    float c, s;

    s = sinf(angle);
    c = cosf(angle);

    m[0][0] = m00 * c + m10 * s;
    m[0][1] = m01 * c + m11 * s;
    m[0][2] = m02 * c + m12 * s;

    m[1][0] = m00 * -s + m10 * c;
    m[1][1] = m01 * -s + m11 * c;
    m[1][2] = m02 * -s + m12 * c;
}

/*!
 * @brief rotate existing 2d transform matrix around Z axis by angle
 *        and store result in dest
 *
 * @param[in]  m      affine transform
 * @param[in]  angle  angle (radians)
 * @param[out] dest   destination
 */
PLAY_CGLM_INLINE
void
rotate2d_to(mat3 m, float angle, mat3 dest)
{
    float m00 = m[0][0],  m10 = m[1][0],
                          m01 = m[0][1],  m11 = m[1][1],
                                          m02 = m[0][2],  m12 = m[1][2];
    float c, s;

    s = sinf(angle);
    c = cosf(angle);

    dest[0][0] = m00 * c + m10 * s;
    dest[0][1] = m01 * c + m11 * s;
    dest[0][2] = m02 * c + m12 * s;

    dest[1][0] = m00 * -s + m10 * c;
    dest[1][1] = m01 * -s + m11 * c;
    dest[1][2] = m02 * -s + m12 * c;

    dest[2][0] = m[2][0];
    dest[2][1] = m[2][1];
    dest[2][2] = m[2][2];
}



/*** End of inlined file: affine2d.h ***/


/*** Start of inlined file: affine2d-post.h ***/



/*
 Functions:
   PLAY_CGLM_INLINE void translated2d(mat3 m, vec2 v);
   PLAY_CGLM_INLINE void translated2d_x(mat3 m, float to);
   PLAY_CGLM_INLINE void translated2d_y(mat3 m, float to);
   PLAY_CGLM_INLINE void rotated2d(mat3 m, float angle);
   PLAY_CGLM_INLINE void scaled2d(mat3 m, vec2 v);
   PLAY_CGLM_INLINE void scaled2d_uni(mat3 m, float s);
 */

/*!
 * @brief translate existing transform matrix by v vector
 *        and store result in same matrix
 *
 *  this is POST transform, applies to existing transform as last transform
 *
 * @param[in, out]  m  affine transform
 * @param[in]       v  translate vector [x, y]
 */
PLAY_CGLM_INLINE
void
translated2d(mat3 m, vec2 v)
{
    vec2_add(m[2], v, m[2]);
}

/*!
 * @brief translate existing transform matrix by x factor
 *
 *  this is POST transform, applies to existing transform as last transform
 *
 * @param[in, out]  m  affine transform
 * @param[in]       x  x factor
 */
PLAY_CGLM_INLINE
void
translated2d_x(mat3 m, float x)
{
    m[2][0] += x;
}

/*!
 * @brief translate existing transform matrix by y factor
 *
 *  this is POST transform, applies to existing transform as last transform
 *
 * @param[in, out]  m  affine transform
 * @param[in]       y  y factor
 */
PLAY_CGLM_INLINE
void
translated2d_y(mat3 m, float y)
{
    m[2][1] += y;
}

/*!
 * @brief rotate existing transform matrix by angle
 *
 *  this is POST transform, applies to existing transform as last transform
 *
 * @param[in, out]   m      affine transform
 * @param[in]   angle  angle (radians)
 */
PLAY_CGLM_INLINE
void
rotated2d(mat3 m, float angle)
{
    float c = cosf(angle),
          s = sinf(angle),

          m00 = m[0][0], m10 = m[1][0], m20 = m[2][0],
                                        m01 = m[0][1], m11 = m[1][1], m21 = m[2][1];

    m[0][0] = c * m00 - s * m01;
    m[1][0] = c * m10 - s * m11;
    m[2][0] = c * m20 - s * m21;

    m[0][1] = s * m00 + c * m01;
    m[1][1] = s * m10 + c * m11;
    m[2][1] = s * m20 + c * m21;
}

/*!
 * @brief scale existing 2d transform matrix by v vector
 *
 *  this is POST transform, applies to existing transform as last transform
 *
 * @param[in, out]   m      affine transform
 * @param[in]   v  scale vector [x, y]
 */
PLAY_CGLM_INLINE
void
scaled2d(mat3 m, vec2 v)
{
    m[0][0] *= v[0];
    m[1][0] *= v[0];
    m[2][0] *= v[0];

    m[0][1] *= v[1];
    m[1][1] *= v[1];
    m[2][1] *= v[1];
}

/*!
 * @brief applies uniform scale to existing 2d transform matrix v = [s, s]
 *
 *  this is POST transform, applies to existing transform as last transform
 *
 * @param[in, out]  m  affine transform
 * @param[in]       s  scale factor
 */
PLAY_CGLM_INLINE
void
scaled2d_uni(mat3 m, float s)
{
    m[0][0] *= s;
    m[1][0] *= s;
    m[2][0] *= s;

    m[0][1] *= s;
    m[1][1] *= s;
    m[2][1] *= s;
}



/*** End of inlined file: affine2d-post.h ***/


/*** End of inlined file: cglm.h ***/


/*** Start of inlined file: types-struct.h ***/



/*
 * Anonymous structs are available since C11, but we'd like to be compatible
 * with C99 and C89 too. So let's figure out if we should be using them or not.
 * It's simply a convenience feature, you can e.g. build the library with
 * anonymous structs and your application without them and they'll still be
 * compatible, cglm doesn't use the anonymous structs internally.
 */
#ifndef PLAY_CGLM_USE_ANONYMOUS_STRUCT
/* If the user doesn't explicitly specify if they want anonymous structs or
 * not, then we'll try to intuit an appropriate choice. */
#  if defined(PLAY_CGLM_NO_ANONYMOUS_STRUCT)
/* The user has defined PLAY_CGLM_NO_ANONYMOUS_STRUCT. This used to be the
 * only #define governing the use of anonymous structs, so for backward
 * compatibility, we still honor that choice and disable them. */
#    define PLAY_CGLM_USE_ANONYMOUS_STRUCT 0
/* Disable anonymous structs if strict ANSI mode is enabled for C89 or C99 */
#  elif defined(__STRICT_ANSI__) && \
        (!defined(__STDC_VERSION__) || (__STDC_VERSION__ < 201112L))
/* __STRICT_ANSI__ is defined and we're in C89
 * or C99 mode (C11 or later not detected) */
#    define PLAY_CGLM_USE_ANONYMOUS_STRUCT 0
#  elif (defined(__STDC_VERSION__) && __STDC_VERSION__ >= 201112L) || \
        (defined(__cplusplus)      && __cplusplus >= 201103L)
/* We're compiling for C11 or this is the MSVC compiler. In either
 * case, anonymous structs are available, so use them. */
#    define PLAY_CGLM_USE_ANONYMOUS_STRUCT 1
#  elif defined(__GNUC__) && (__GNUC__ > 4 || (__GNUC__ == 4 && __GNUC_MINOR__ >= 6))
/* GCC 4.6 and onwards support anonymous structs as an extension */
#    define PLAY_CGLM_USE_ANONYMOUS_STRUCT 1
#  elif defined(__clang__) && __clang_major__ >= 3
/* Clang 3.0 and onwards support anonymous structs as an extension */
#    define PLAY_CGLM_USE_ANONYMOUS_STRUCT 1
#  elif defined(_MSC_VER) && (_MSC_VER >= 1900) /*  Visual Studio 2015 */
/* We can support anonymous structs
 * since Visual Studio 2015 or 2017 (1910) maybe? */
#    define PLAY_CGLM_USE_ANONYMOUS_STRUCT 1
#  else
/* Otherwise, we're presumably building for C99 or C89 and can't rely
 * on anonymous structs being available. Turn them off. */
#    define PLAY_CGLM_USE_ANONYMOUS_STRUCT 0
#  endif
#endif

typedef union vec2s
{
    vec2 raw;
#if PLAY_CGLM_USE_ANONYMOUS_STRUCT
    struct
    {
        float x;
        float y;
    };

    struct
    {
        float r;
        float i;
    };

    struct
    {
        float u;
        float v;
    };

    struct
    {
        float s;
        float t;
    };
#endif
} vec2s;

typedef union vec3s
{
    vec3 raw;
#if PLAY_CGLM_USE_ANONYMOUS_STRUCT
    struct
    {
        float x;
        float y;
        float z;
    };

    struct
    {
        float r;
        float g;
        float b;
    };
#endif
} vec3s;

typedef union ivec2s
{
    ivec2 raw;
#if PLAY_CGLM_USE_ANONYMOUS_STRUCT
    struct
    {
        int x;
        int y;
    };

    struct
    {
        int r;
        int i;
    };

    struct
    {
        int u;
        int v;
    };

    struct
    {
        int s;
        int t;
    };
#endif
} ivec2s;

typedef union ivec3s
{
    ivec3 raw;
#if PLAY_CGLM_USE_ANONYMOUS_STRUCT
    struct
    {
        int x;
        int y;
        int z;
    };

    struct
    {
        int r;
        int g;
        int b;
    };
#endif
} ivec3s;

typedef union ivec4s
{
    ivec4 raw;
#if PLAY_CGLM_USE_ANONYMOUS_STRUCT
    struct
    {
        int x;
        int y;
        int z;
        int w;
    };

    struct
    {
        int r;
        int g;
        int b;
        int a;
    };
#endif
} ivec4s;

typedef union
    PLAY_CGLM_ALIGN_IF(16) vec4s
{
    vec4 raw;
#if PLAY_CGLM_USE_ANONYMOUS_STRUCT
    struct
    {
        float x;
        float y;
        float z;
        float w;
    };

    struct
    {
        float r;
        float g;
        float b;
        float a;
    };
#endif
} vec4s;

typedef union
    PLAY_CGLM_ALIGN_IF(16) versors
{
    vec4 raw;
#if PLAY_CGLM_USE_ANONYMOUS_STRUCT
    struct
    {
        float x;
        float y;
        float z;
        float w;
    };

    struct
    {
        vec3s imag;
        float real;
    };
#endif
} versors;

typedef union mat2s
{
    mat2  raw;
    vec2s col[2];
#if PLAY_CGLM_USE_ANONYMOUS_STRUCT
    struct
    {
        float m00, m01;
        float m10, m11;
    };
#endif
} mat2s;

typedef union mat2x3s
{
    mat2x3 raw;
    vec3s  col[2]; /* [col (2), row (3)] */
#if PLAY_CGLM_USE_ANONYMOUS_STRUCT
    struct
    {
        float m00, m01, m02;
        float m10, m11, m12;
    };
#endif
} mat2x3s;

typedef union mat2x4s
{
    mat2x4 raw;
    vec4s  col[2]; /* [col (2), row (4)] */
#if PLAY_CGLM_USE_ANONYMOUS_STRUCT
    struct
    {
        float m00, m01, m02, m03;
        float m10, m11, m12, m13;
    };
#endif
} mat2x4s;

typedef union mat3s
{
    mat3  raw;
    vec3s col[3];
#if PLAY_CGLM_USE_ANONYMOUS_STRUCT
    struct
    {
        float m00, m01, m02;
        float m10, m11, m12;
        float m20, m21, m22;
    };
#endif
} mat3s;

typedef union mat3x2s
{
    mat3x2 raw;
    vec2s  col[3]; /* [col (3), row (2)] */
#if PLAY_CGLM_USE_ANONYMOUS_STRUCT
    struct
    {
        float m00, m01;
        float m10, m11;
        float m20, m21;
    };
#endif
} mat3x2s;

typedef union mat3x4s
{
    mat3x4 raw;
    vec4s  col[3]; /* [col (3), row (4)] */
#if PLAY_CGLM_USE_ANONYMOUS_STRUCT
    struct
    {
        float m00, m01, m02, m03;
        float m10, m11, m12, m13;
        float m20, m21, m22, m23;
    };
#endif
} mat3x4s;

typedef union PLAY_CGLM_ALIGN_MAT mat4s
{
    mat4  raw;
    vec4s col[4];
#if PLAY_CGLM_USE_ANONYMOUS_STRUCT
    struct
    {
        float m00, m01, m02, m03;
        float m10, m11, m12, m13;
        float m20, m21, m22, m23;
        float m30, m31, m32, m33;
    };
#endif
} mat4s;

typedef union mat4x2s
{
    mat4x2 raw;
    vec2s  col[4]; /* [col (4), row (2)] */
#if PLAY_CGLM_USE_ANONYMOUS_STRUCT
    struct
    {
        float m00, m01;
        float m10, m11;
        float m20, m21;
        float m30, m31;
    };
#endif
} mat4x2s;

typedef union mat4x3s
{
    mat4x3 raw;
    vec3s  col[4]; /* [col (4), row (3)] */
#if PLAY_CGLM_USE_ANONYMOUS_STRUCT
    struct
    {
        float m00, m01, m02;
        float m10, m11, m12;
        float m20, m21, m22;
        float m30, m31, m32;
    };
#endif
} mat4x3s;



/*** End of inlined file: types-struct.h ***/


/*** Start of inlined file: vec2.h ***/
/*
 Macros:
   PLAY_CGLM_S_VEC2_ONE_INIT
   PLAY_CGLM_S_VEC2_ZERO_INIT
   PLAY_CGLM_S_VEC2_ONE
   PLAY_CGLM_S_VEC2_ZERO

 Functions:
   PLAY_CGLM_INLINE vec2s vec2_new(vec3s v3)
   PLAY_CGLM_INLINE void  vec2s_pack(vec2s dst[], vec2 src[], size_t len)
   PLAY_CGLM_INLINE void  vec2s_unpack(vec2 dst[], vec2s src[], size_t len)
   PLAY_CGLM_INLINE vec2s vec2s_zero(void)
   PLAY_CGLM_INLINE vec2s vec2s_one(void)
   PLAY_CGLM_INLINE float vec2s_dot(vec2s a, vec2s b)
   PLAY_CGLM_INLINE float vec2s_cross(vec2s a, vec2s b)
   PLAY_CGLM_INLINE float vec2s_norm2(vec2s v)
   PLAY_CGLM_INLINE float vec2s_norm(vec2s v)
   PLAY_CGLM_INLINE vec2s vec2s_add(vec2s a, vec2s b)
   PLAY_CGLM_INLINE vec2s vec2s_adds(vec2s a, float s)
   PLAY_CGLM_INLINE vec2s vec2s_sub(vec2s a, vec2s b)
   PLAY_CGLM_INLINE vec2s vec2s_subs(vec2s a, float s)
   PLAY_CGLM_INLINE vec2s vec2s_mul(vec2s a, vec2s b)
   PLAY_CGLM_INLINE vec2s vec2s_scale(vec2s v, float s)
   PLAY_CGLM_INLINE vec2s vec2s_scale_as(vec2s v, float s)
   PLAY_CGLM_INLINE vec2s vec2s_div(vec2s a, vec2s b)
   PLAY_CGLM_INLINE vec2s vec2s_divs(vec2s a, float s)
   PLAY_CGLM_INLINE vec2s vec2s_addadd(vec2s a, vec2s b, vec2s dest)
   PLAY_CGLM_INLINE vec2s vec2s_subadd(vec2s a, vec2s b, vec2s dest)
   PLAY_CGLM_INLINE vec2s vec2s_muladd(vec2s a, vec2s b, vec2s dest)
   PLAY_CGLM_INLINE vec2s vec2s_muladds(vec2s a, float s, vec2s dest)
   PLAY_CGLM_INLINE vec2s vec2s_maxadd(vec2s a, vec2s b, vec2s dest)
   PLAY_CGLM_INLINE vec2s vec2s_minadd(vec2s a, vec2s b, vec2s dest)
   PLAY_CGLM_INLINE vec2s vec2s_subsub(vec2s a, vec2s b, vec2s dest)
   PLAY_CGLM_INLINE vec2s vec2s_addsub(vec2s a, vec2s b, vec2s dest)
   PLAY_CGLM_INLINE vec2s vec2s_mulsub(vec2s a, vec2s b, vec2s dest)
   PLAY_CGLM_INLINE vec2s vec2s_mulsubs(vec2s a, float s, vec2s dest)
   PLAY_CGLM_INLINE vec2s vec2s_maxsub(vec2s a, vec2s b, vec2s dest)
   PLAY_CGLM_INLINE vec2s vec2s_minsub(vec2s a, vec2s b, vec2s dest)
   PLAY_CGLM_INLINE vec2s vec2s_negate(vec2s v)
   PLAY_CGLM_INLINE vec2s vec2s_normalize(vec2s v)
   PLAY_CGLM_INLINE vec2s vec2s_rotate(vec2s v, float angle, vec2s axis)
   PLAY_CGLM_INLINE vec2s vec2s_center(vec2s a, vec2s b)
   PLAY_CGLM_INLINE float vec2s_distance(vec2s a, vec2s b)
   PLAY_CGLM_INLINE float vec2s_distance2(vec2s a, vec2s b)
   PLAY_CGLM_INLINE vec2s vec2s_maxv(vec2s a, vec2s b)
   PLAY_CGLM_INLINE vec2s vec2s_minv(vec2s a, vec2s b)
   PLAY_CGLM_INLINE vec2s vec2s_clamp(vec2s v, float minVal, float maxVal)
   PLAY_CGLM_INLINE vec2s vec2s_lerp(vec2s from, vec2s to, float t)
   PLAY_CGLM_INLINE vec2s vec2s_step(vec2s edge, vec2s x)
   PLAY_CGLM_INLINE vec2s vec2s_make(float * restrict src)
   PLAY_CGLM_INLINE vec2s vec2s_reflect(vec2s v, vec2s n)
   PLAY_CGLM_INLINE bool  vec2s_refract(vec2s v, vec2s n, float eta, vec2s *dest)
 */





/*** Start of inlined file: vec2-ext.h ***/
/*!
 * @brief SIMD like functions
 */

/*
 Functions:
   PLAY_CGLM_INLINE vec2s vec2s_fill(float val)
   PLAY_CGLM_INLINE bool  vec2s_eq(vec2s v, float val)
   PLAY_CGLM_INLINE bool  vec2s_eq_eps(vec2s v, float val)
   PLAY_CGLM_INLINE bool  vec2s_eq_all(vec2s v)
   PLAY_CGLM_INLINE bool  vec2s_eqv(vec2s a, vec2s b)
   PLAY_CGLM_INLINE bool  vec2s_eqv_eps(vec2s a, vec2s b)
   PLAY_CGLM_INLINE float vec2s_max(vec2s v)
   PLAY_CGLM_INLINE float vec2s_min(vec2s v)
   PLAY_CGLM_INLINE bool  vec2s_isnan(vec2s v)
   PLAY_CGLM_INLINE bool  vec2s_isinf(vec2s v)
   PLAY_CGLM_INLINE bool  vec2s_isvalid(vec2s v)
   PLAY_CGLM_INLINE vec2s vec2s_sign(vec2s v)
   PLAY_CGLM_INLINE vec2s vec2s_abs(vec2s v)
   PLAY_CGLM_INLINE vec2s vec2s_fract(vec2s v)
   PLAY_CGLM_INLINE vec2s vec2s_floor(vec2s v)
   PLAY_CGLM_INLINE vec2s vec2s_mods(vec2s v, float s)
   PLAY_CGLM_INLINE vec2s vec2s_steps(float edge, vec2s v)
   PLAY_CGLM_INLINE vec2s vec2s_stepr(vec2s edge, float v)
   PLAY_CGLM_INLINE vec2s vec2s_sqrt(vec2s v)
 */




/* api definition */
#define vec2s_(NAME) PLAY_CGLM_STRUCTAPI(vec2, NAME)

/*!
 * @brief fill a vector with specified value
 *
 * @param[in]  val  value
 * @returns         dest
 */
PLAY_CGLM_INLINE
vec2s
vec2s_(fill)(float val)
{
    vec2s r;
    vec2_fill(r.raw, val);
    return r;
}

/*!
 * @brief check if vector is equal to value (without epsilon)
 *
 * @param[in] v   vector
 * @param[in] val value
 */
PLAY_CGLM_INLINE
bool
vec2s_(eq)(vec2s v, float val)
{
    return vec2_eq(v.raw, val);
}

/*!
 * @brief check if vector is equal to value (with epsilon)
 *
 * @param[in] v   vector
 * @param[in] val value
 */
PLAY_CGLM_INLINE
bool
vec2s_(eq_eps)(vec2s v, float val)
{
    return vec2_eq_eps(v.raw, val);
}

/*!
 * @brief check if vector members are equal (without epsilon)
 *
 * @param[in] v   vector
 */
PLAY_CGLM_INLINE
bool
vec2s_(eq_all)(vec2s v)
{
    return vec2_eq_all(v.raw);
}

/*!
 * @brief check if vector is equal to another (without epsilon)
 *
 * @param[in] a vector
 * @param[in] b vector
 */
PLAY_CGLM_INLINE
bool
vec2s_(eqv)(vec2s a, vec2s b)
{
    return vec2_eqv(a.raw, b.raw);
}

/*!
 * @brief check if vector is equal to another (with epsilon)
 *
 * @param[in] a vector
 * @param[in] b vector
 */
PLAY_CGLM_INLINE
bool
vec2s_(eqv_eps)(vec2s a, vec2s b)
{
    return vec2_eqv_eps(a.raw, b.raw);
}

/*!
 * @brief max value of vector
 *
 * @param[in] v vector
 */
PLAY_CGLM_INLINE
float
vec2s_(max)(vec2s v)
{
    return vec2_max(v.raw);
}

/*!
 * @brief min value of vector
 *
 * @param[in] v vector
 */
PLAY_CGLM_INLINE
float
vec2s_min(vec2s v)
{
    return vec2_min(v.raw);
}

/*!
 * @brief check if one of items is NaN (not a number)
 *        you should only use this in DEBUG mode or very critical asserts
 *
 * @param[in] v vector
 */
PLAY_CGLM_INLINE
bool
vec2s_(isnan)(vec2s v)
{
    return vec2_isnan(v.raw);
}

/*!
 * @brief check if one of items is INFINITY
 *        you should only use this in DEBUG mode or very critical asserts
 *
 * @param[in] v vector
 */
PLAY_CGLM_INLINE
bool
vec2s_(isinf)(vec2s v)
{
    return vec2_isinf(v.raw);
}

/*!
 * @brief check if all items are valid number
 *        you should only use this in DEBUG mode or very critical asserts
 *
 * @param[in] v vector
 */
PLAY_CGLM_INLINE
bool
vec2s_isvalid(vec2s v)
{
    return vec2_isvalid(v.raw);
}

/*!
 * @brief get sign of 32 bit float as +1, -1, 0
 *
 * Important: It returns 0 for zero/NaN input
 *
 * @param   v   vector
 * @returns     sign vector
 */
PLAY_CGLM_INLINE
vec2s
vec2s_(sign)(vec2s v)
{
    vec2s r;
    vec2_sign(v.raw, r.raw);
    return r;
}

/*!
 * @brief fractional part of each vector item
 *
 * @param   v   vector
 * @returns     abs vector
 */
PLAY_CGLM_INLINE
vec2s
vec2s_(abs)(vec2s v)
{
    vec2s r;
    vec2_abs(v.raw, r.raw);
    return r;
}

/*!
 * @brief fractional part of each vector item
 *
 * @param[in]  v    vector
 * @returns         destination vector
 */
PLAY_CGLM_INLINE
vec2s
vec2s_(fract)(vec2s v)
{
    vec2s r;
    vec2_fract(v.raw, r.raw);
    return r;
}

/*!
 * @brief floor of each vector item
 *
 * @param[in]  v    vector
 * @returns         destination vector
 */
PLAY_CGLM_INLINE
vec2s
vec2s_(floor)(vec2s v)
{
    vec2s r;
    vec2_floor(v.raw, r.raw);
    return r;
}

/*!
 * @brief mod of each vector item by scalar
 *
 * @param[in]  v    vector
 * @param[in]  s    scalar
 * @returns         destination vector
 */
PLAY_CGLM_INLINE
vec2s
vec2s_(mods)(vec2s v, float s)
{
    vec2s r;
    vec2_mods(v.raw, s, r.raw);
    return r;
}

/*!
 * @brief threshold each vector item with scalar
 *        condition is: (x[i] < edge) ? 0.0 : 1.0
 *
 * @param[in]   edge   threshold
 * @param[in]   x      vector to test against threshold
 * @returns            destination
 */
PLAY_CGLM_INLINE
vec2s
vec2s_(steps)(float edge, vec2s x)
{
    vec2s r;
    vec2_steps(edge, x.raw, r.raw);
    return r;
}

/*!
 * @brief threshold a value with *vector* as the threshold
 *        condition is: (x < edge[i]) ? 0.0 : 1.0
 *
 * @param[in]   edge   threshold vector
 * @param[in]   x      value to test against threshold
 * @returns            destination
 */
PLAY_CGLM_INLINE
vec2s
vec2s_(stepr)(vec2s edge, float x)
{
    vec2s r;
    vec2_stepr(edge.raw, x, r.raw);
    return r;
}

/*!
 * @brief square root of each vector item
 *
 * @param[in]  v    vector
 * @returns         destination vector
 */
PLAY_CGLM_INLINE
vec2s
vec2s_(sqrt)(vec2s v)
{
    vec2s r;
    vec2_sqrt(v.raw, r.raw);
    return r;
}

/*!
 * @brief treat vectors as complex numbers and multiply them as such.
 *
 * @param[in]  a    left number
 * @param[in]  b    right number
 * @param[out] dest destination number
 */
PLAY_CGLM_INLINE
vec2s
vec2s_(complex_mul)(vec2s a, vec2s b, vec2s dest)
{
    vec2_complex_mul(a.raw, b.raw, dest.raw);
    return dest;
}

/*!
 * @brief treat vectors as complex numbers and divide them as such.
 *
 * @param[in]  a    left number (numerator)
 * @param[in]  b    right number (denominator)
 * @param[out] dest destination number
 */
PLAY_CGLM_INLINE
vec2s
vec2s_(complex_div)(vec2s a, vec2s b, vec2s dest)
{
    vec2_complex_div(a.raw, b.raw, dest.raw);
    return dest;
}

/*!
 * @brief treat the vector as a complex number and conjugate it as such.
 *
 * @param[in]  a    the number
 * @param[out] dest destination number
 */
PLAY_CGLM_INLINE
vec2s
vec2s_(complex_conjugate)(vec2s a, vec2s dest)
{
    vec2_complex_conjugate(a.raw, dest.raw);
    return dest;
}



/*** End of inlined file: vec2-ext.h ***/

#define PLAY_CGLM_S_VEC2_ONE_INIT   {PLAY_CGLM_VEC2_ONE_INIT}
#define PLAY_CGLM_S_VEC2_ZERO_INIT  {PLAY_CGLM_VEC2_ZERO_INIT}

#define PLAY_CGLM_S_VEC2_ONE  ((vec2s)PLAY_CGLM_S_VEC2_ONE_INIT)
#define PLAY_CGLM_S_VEC2_ZERO ((vec2s)PLAY_CGLM_S_VEC2_ZERO_INIT)

/*!
 * @brief init vec2 using vec2
 *
 * @param[in]  v3   vector3
 * @returns         destination
 */
PLAY_CGLM_INLINE
vec2s
vec2_new(vec3s v3)
{
    vec2s r;
    vec2_new(v3.raw, r.raw);
    return r;
}

/*!
 * @brief pack an array of vec2 into an array of vec2s
 *
 * @param[out] dst array of vec2
 * @param[in]  src array of vec2s
 * @param[in]  len number of elements
 */
PLAY_CGLM_INLINE
void
vec2s_(pack)(vec2s dst[], vec2 src[], size_t len)
{
    size_t i;

    for (i = 0; i < len; i++)
    {
        vec2_copy(src[i], dst[i].raw);
    }
}

/*!
 * @brief unpack an array of vec2s into an array of vec2
 *
 * @param[out] dst array of vec2s
 * @param[in]  src array of vec2
 * @param[in]  len number of elements
 */
PLAY_CGLM_INLINE
void
vec2s_(unpack)(vec2 dst[], vec2s src[], size_t len)
{
    size_t i;

    for (i = 0; i < len; i++)
    {
        vec2_copy(src[i].raw, dst[i]);
    }
}

/*!
 * @brief make vector zero
 *
 * @returns zero vector
 */
PLAY_CGLM_INLINE
vec2s
vec2s_(zero)(void)
{
    vec2s r;
    vec2_zero(r.raw);
    return r;
}

/*!
 * @brief make vector one
 *
 * @returns one vector
 */
PLAY_CGLM_INLINE
vec2s
vec2s_(one)(void)
{
    vec2s r;
    vec2_one(r.raw);
    return r;
}

/*!
 * @brief vec2 dot product
 *
 * @param[in] a vector1
 * @param[in] b vector2
 *
 * @return dot product
 */
PLAY_CGLM_INLINE
float
vec2s_(dot)(vec2s a, vec2s b)
{
    return vec2_dot(a.raw, b.raw);
}

/*!
 * @brief vec2 cross product
 *
 * REF: http://allenchou.net/2013/07/cross-product-of-2d-vectors/
 *
 * @param[in]  a vector1
 * @param[in]  b vector2
 *
 * @return Z component of cross product
 */
PLAY_CGLM_INLINE
float
vec2s_(cross)(vec2s a, vec2s b)
{
    return vec2_cross(a.raw, b.raw);
}

/*!
 * @brief norm * norm (magnitude) of vec
 *
 * we can use this func instead of calling norm * norm, because it would call
 * sqrtf function twice but with this func we can avoid func call, maybe this is
 * not good name for this func
 *
 * @param[in] v vector
 *
 * @return norm * norm
 */
PLAY_CGLM_INLINE
float
vec2s_(norm2)(vec2s v)
{
    return vec2_norm2(v.raw);
}

/*!
 * @brief norm (magnitude) of vec2
 *
 * @param[in] v vector
 *
 * @return norm
 */
PLAY_CGLM_INLINE
float
vec2s_(norm)(vec2s v)
{
    return vec2_norm(v.raw);
}

/*!
 * @brief add a vector to b vector store result in dest
 *
 * @param[in]  a    vector1
 * @param[in]  b    vector2
 * @returns         destination vector
 */
PLAY_CGLM_INLINE
vec2s
vec2s_(add)(vec2s a, vec2s b)
{
    vec2s r;
    vec2_add(a.raw, b.raw, r.raw);
    return r;
}

/*!
 * @brief add scalar to v vector store result in dest (d = v + s)
 *
 * @param[in]  a    vector
 * @param[in]  s    scalar
 * @returns         destination vector
 */
PLAY_CGLM_INLINE
vec2s
vec2s_(adds)(vec2s a, float s)
{
    vec2s r;
    vec2_adds(a.raw, s, r.raw);
    return r;
}

/*!
 * @brief subtract b vector from a vector store result in dest
 *
 * @param[in]  a    vector1
 * @param[in]  b    vector2
 * @returns         destination vector
 */
PLAY_CGLM_INLINE
vec2s
vec2s_(sub)(vec2s a, vec2s b)
{
    vec2s r;
    vec2_sub(a.raw, b.raw, r.raw);
    return r;
}

/*!
 * @brief subtract scalar from v vector store result in dest (d = v - s)
 *
 * @param[in]  a    vector
 * @param[in]  s    scalar
 * @returns         destination vector
 */
PLAY_CGLM_INLINE
vec2s
vec2s_(subs)(vec2s a, float s)
{
    vec2s r;
    vec2_subs(a.raw, s, r.raw);
    return r;
}

/*!
 * @brief multiply two vectors (component-wise multiplication)
 *
 * @param     a     vector1
 * @param     b     vector2
 * @returns         result = (a[0] * b[0], a[1] * b[1])
 */
PLAY_CGLM_INLINE
vec2s
vec2s_(mul)(vec2s a, vec2s b)
{
    vec2s r;
    vec2_mul(a.raw, b.raw, r.raw);
    return r;
}

/*!
 * @brief multiply/scale vec2 vector with scalar: result = v * s
 *
 * @param[in]  v    vector
 * @param[in]  s    scalar
 * @returns         destination vector
 */
PLAY_CGLM_INLINE
vec2s
vec2s_(scale)(vec2s v, float s)
{
    vec2s r;
    vec2_scale(v.raw, s, r.raw);
    return r;
}

/*!
 * @brief make vec2 vector scale as specified: result = unit(v) * s
 *
 * @param[in]  v    vector
 * @param[in]  s    scalar
 * @returns         destination vector
 */
PLAY_CGLM_INLINE
vec2s
vec2s_(scale_as)(vec2s v, float s)
{
    vec2s r;
    vec2_scale_as(v.raw, s, r.raw);
    return r;
}

/*!
 * @brief div vector with another component-wise division: d = a / b
 *
 * @param[in]  a    vector 1
 * @param[in]  b    vector 2
 * @returns         result = (a[0]/b[0], a[1]/b[1])
 */
PLAY_CGLM_INLINE
vec2s
vec2s_(div)(vec2s a, vec2s b)
{
    vec2s r;
    vec2_div(a.raw, b.raw, r.raw);
    return r;
}

/*!
 * @brief div vector with scalar: d = v / s
 *
 * @param[in]  a    vector
 * @param[in]  s    scalar
 * @returns         result = (a[0]/s, a[1]/s)
 */
PLAY_CGLM_INLINE
vec2s
vec2s_(divs)(vec2s a, float s)
{
    vec2s r;
    vec2_divs(a.raw, s, r.raw);
    return r;
}

/*!
 * @brief add two vectors and add result to sum
 *
 * it applies += operator so dest must be initialized
 *
 * @param[in]  a    vector 1
 * @param[in]  b    vector 2
 * @returns         dest += (a + b)
 */
PLAY_CGLM_INLINE
vec2s
vec2s_(addadd)(vec2s a, vec2s b, vec2s dest)
{
    vec2_addadd(a.raw, b.raw, dest.raw);
    return dest;
}

/*!
 * @brief sub two vectors and add result to dest
 *
 * it applies += operator so dest must be initialized
 *
 * @param[in]  a    vector 1
 * @param[in]  b    vector 2
 * @returns         dest += (a + b)
 */
PLAY_CGLM_INLINE
vec2s
vec2s_(subadd)(vec2s a, vec2s b, vec2s dest)
{
    vec2_subadd(a.raw, b.raw, dest.raw);
    return dest;
}

/*!
 * @brief mul two vectors and add result to dest
 *
 * it applies += operator so dest must be initialized
 *
 * @param[in]  a    vector 1
 * @param[in]  b    vector 2
 * @returns         dest += (a * b)
 */
PLAY_CGLM_INLINE
vec2s
vec2s_(muladd)(vec2s a, vec2s b, vec2s dest)
{
    vec2_muladd(a.raw, b.raw, dest.raw);
    return dest;
}

/*!
 * @brief mul vector with scalar and add result to sum
 *
 * it applies += operator so dest must be initialized
 *
 * @param[in]  a    vector
 * @param[in]  s    scalar
 * @returns         dest += (a * b)
 */
PLAY_CGLM_INLINE
vec2s
vec2s_(muladds)(vec2s a, float s, vec2s dest)
{
    vec2_muladds(a.raw, s, dest.raw);
    return dest;
}

/*!
 * @brief add max of two vectors to result/dest
 *
 * it applies += operator so dest must be initialized
 *
 * @param[in]  a    vector 1
 * @param[in]  b    vector 2
 * @returns         dest += fmax(a, b)
 */
PLAY_CGLM_INLINE
vec2s
vec2s_(maxadd)(vec2s a, vec2s b, vec2s dest)
{
    vec2_maxadd(a.raw, b.raw, dest.raw);
    return dest;
}

/*!
 * @brief add min of two vectors to result/dest
 *
 * it applies += operator so dest must be initialized
 *
 * @param[in]  a    vector 1
 * @param[in]  b    vector 2
 * @returns         dest += fmin(a, b)
 */
PLAY_CGLM_INLINE
vec2s
vec2s_(minadd)(vec2s a, vec2s b, vec2s dest)
{
    vec2_minadd(a.raw, b.raw, dest.raw);
    return dest;
}

/*!
 * @brief sub two vectors and sub result to dest
 *
 * it applies -= operator so dest must be initialized
 *
 * @param[in]  a    vector 1
 * @param[in]  b    vector 2
 * @returns         dest -= (a - b)
 */
PLAY_CGLM_INLINE
vec2s
vec2s_(subsub)(vec2s a, vec2s b, vec2s dest)
{
    vec2_subsub(a.raw, b.raw, dest.raw);
    return dest;
}

/*!
 * @brief add two vectors and sub result to dest
 *
 * it applies -= operator so dest must be initialized
 *
 * @param[in]  a    vector 1
 * @param[in]  b    vector 2
 * @returns         dest -= (a + b)
 */
PLAY_CGLM_INLINE
vec2s
vec2s_(addsub)(vec2s a, vec2s b, vec2s dest)
{
    vec2_addsub(a.raw, b.raw, dest.raw);
    return dest;
}

/*!
 * @brief mul two vectors and sub result to dest
 *
 * it applies -= operator so dest must be initialized
 *
 * @param[in]  a    vector 1
 * @param[in]  b    vector 2
 * @returns         dest -= (a * b)
 */
PLAY_CGLM_INLINE
vec2s
vec2s_(mulsub)(vec2s a, vec2s b, vec2s dest)
{
    vec2_mulsub(a.raw, b.raw, dest.raw);
    return dest;
}

/*!
 * @brief mul vector with scalar and sub result to dest
 *
 * it applies -= operator so dest must be initialized
 *
 * @param[in]  a    vector
 * @param[in]  s    scalar
 * @returns         dest -= (a * b)
 */
PLAY_CGLM_INLINE
vec2s
vec2s_(mulsubs)(vec2s a, float s, vec2s dest)
{
    vec2_mulsubs(a.raw, s, dest.raw);
    return dest;
}

/*!
 * @brief sub max of two vectors to dest
 *
 * it applies += operator so dest must be initialized
 *
 * @param[in]  a    vector 1
 * @param[in]  b    vector 2
 * @returns         dest -= fmax(a, b)
 */
PLAY_CGLM_INLINE
vec2s
vec2s_(maxsub)(vec2s a, vec2s b, vec2s dest)
{
    vec2_maxsub(a.raw, b.raw, dest.raw);
    return dest;
}

/*!
 * @brief sub min of two vectors to dest
 *
 * it applies -= operator so dest must be initialized
 *
 * @param[in]  a    vector 1
 * @param[in]  b    vector 2
 * @returns         dest -= fmin(a, b)
 */
PLAY_CGLM_INLINE
vec2s
vec2s_(minsub)(vec2s a, vec2s b, vec2s dest)
{
    vec2_minsub(a.raw, b.raw, dest.raw);
    return dest;
}

/*!
 * @brief negate vector components
 *
 * @param[in]  v  vector
 * @returns       negated vector
 */
PLAY_CGLM_INLINE
vec2s
vec2s_(negate)(vec2s v)
{
    vec2_negate(v.raw);
    return v;
}

/*!
 * @brief normalize vec2 and store result in same vec
 *
 * @param[in] v vector
 * @returns     normalized vector
 */
PLAY_CGLM_INLINE
vec2s
vec2s_(normalize)(vec2s v)
{
    vec2_normalize(v.raw);
    return v;
}

/*!
 * @brief rotate vec2 by angle using Rodrigues' rotation formula
 *
 * @param[in]     v     vector
 * @param[in]     angle angle by radians
 * @returns             rotated vector
 */
PLAY_CGLM_INLINE
vec2s
vec2s_(rotate)(vec2s v, float angle)
{
    vec2s r;
    vec2_rotate(v.raw, angle, r.raw);
    return r;
}

/**
 * @brief find center point of two vector
 *
 * @param[in]  a    vector1
 * @param[in]  b    vector2
 * @returns         center point
 */
PLAY_CGLM_INLINE
vec2s
vec2s_(center)(vec2s a, vec2s b)
{
    vec2s r;
    vec2_center(a.raw, b.raw, r.raw);
    return r;
}

/**
 * @brief distance between two vectors
 *
 * @param[in] a vector1
 * @param[in] b vector2
 * @return      distance
 */
PLAY_CGLM_INLINE
float
vec2s_(distance)(vec2s a, vec2s b)
{
    return vec2_distance(a.raw, b.raw);
}

/**
 * @brief squared distance between two vectors
 *
 * @param[in] a vector1
 * @param[in] b vector2
 * @return      squared distance (distance * distance)
 */
PLAY_CGLM_INLINE
float
vec2s_(distance2)(vec2s a, vec2s b)
{
    return vec2_distance2(a.raw, b.raw);
}

/*!
 * @brief max values of vectors
 *
 * @param[in]  a    vector1
 * @param[in]  b    vector2
 * @returns         destination
 */
PLAY_CGLM_INLINE
vec2s
vec2s_(maxv)(vec2s a, vec2s b)
{
    vec2s r;
    vec2_maxv(a.raw, b.raw, r.raw);
    return r;
}

/*!
 * @brief min values of vectors
 *
 * @param[in]  a    vector1
 * @param[in]  b    vector2
 * @returns         destination
 */
PLAY_CGLM_INLINE
vec2s
vec2s_(minv)(vec2s a, vec2s b)
{
    vec2s r;
    vec2_minv(a.raw, b.raw, r.raw);
    return r;
}

/*!
 * @brief clamp vector's individual members between min and max values
 *
 * @param[in]       v       vector
 * @param[in]       minVal  minimum value
 * @param[in]       maxVal  maximum value
 * @returns                 clamped vector
 */
PLAY_CGLM_INLINE
vec2s
vec2s_(clamp)(vec2s v, float minVal, float maxVal)
{
    vec2_clamp(v.raw, minVal, maxVal);
    return v;
}

/*!
 * @brief linear interpolation between two vectors
 *
 * formula:  from + s * (to - from)
 *
 * @param[in]   from  from value
 * @param[in]   to    to value
 * @param[in]   t     interpolant (amount)
 * @returns           destination
 */
PLAY_CGLM_INLINE
vec2s
vec2s_(lerp)(vec2s from, vec2s to, float t)
{
    vec2s r;
    vec2_lerp(from.raw, to.raw, t, r.raw);
    return r;
}

/*!
 * @brief threshold function
 *
 * @param[in]   edge    threshold
 * @param[in]   x       value to test against threshold
 * @returns             destination
 */
PLAY_CGLM_INLINE
vec2s
vec2s_(step)(vec2s edge, vec2s x)
{
    vec2s r;
    vec2_step(edge.raw, x.raw, r.raw);
    return r;
}

/*!
 * @brief Create two dimensional vector from pointer
 *
 * @param[in]  src  pointer to an array of floats
 * @returns constructed 2D vector from raw pointer
 */
PLAY_CGLM_INLINE
vec2s
vec2s_(make)(const float * __restrict src)
{
    vec2s dest;
    vec2_make(src, dest.raw);
    return dest;
}

/*!
 * @brief reflection vector using an incident ray and a surface normal
 *
 * @param[in]  I    incident vector
 * @param[in]  N    normalized normal vector
 * @returns reflection result
 */
PLAY_CGLM_INLINE
vec2s
vec2s_(reflect)(vec2s v, vec2s n)
{
    vec2s dest;
    vec2_reflect(v.raw, n.raw, dest.raw);
    return dest;
}

/*!
 * @brief computes refraction vector for an incident vector and a surface normal.
 *
 * calculates the refraction vector based on Snell's law. If total internal reflection
 * occurs (angle too great given eta), dest is set to zero and returns false.
 * Otherwise, computes refraction vector, stores it in dest, and returns true.
 *
 * @param[in]  v    normalized incident vector
 * @param[in]  n    normalized normal vector
 * @param[in]  eta  ratio of indices of refraction (incident/transmitted)
 * @param[out] dest refraction vector if refraction occurs; zero vector otherwise
 *
 * @returns true if refraction occurs; false if total internal reflection occurs.
 */
PLAY_CGLM_INLINE
bool
vec2s_(refract)(vec2s v, vec2s n, float eta, vec2s * __restrict dest)
{
    return vec2_refract(v.raw, n.raw, eta, dest->raw);
}



/*** End of inlined file: vec2.h ***/


/*** Start of inlined file: vec3.h ***/
/*
 Macros:
   PLAY_CGLM_S_VEC3_ONE_INIT
   PLAY_CGLM_S_VEC3_ZERO_INIT
   PLAY_CGLM_S_VEC3_ONE
   PLAY_CGLM_S_VEC3_ZERO
   PLAY_CGLM_S_YUP
   PLAY_CGLM_S_ZUP
   PLAY_CGLM_S_XUP

 Functions:
   PLAY_CGLM_INLINE vec3s vec3_new(vec4s v4);
   PLAY_CGLM_INLINE void  vec3s_pack(vec3s dst[], vec3 src[], size_t len);
   PLAY_CGLM_INLINE void  vec3s_unpack(vec3 dst[], vec3s src[], size_t len);
   PLAY_CGLM_INLINE vec3s vec3s_zero(void);
   PLAY_CGLM_INLINE vec3s vec3s_one(void);
   PLAY_CGLM_INLINE float vec3s_dot(vec3s a, vec3s b);
   PLAY_CGLM_INLINE float vec3s_norm2(vec3s v);
   PLAY_CGLM_INLINE float vec3s_norm(vec3s v);
   PLAY_CGLM_INLINE float vec3s_norm_one(vec3s v);
   PLAY_CGLM_INLINE float vec3s_norm_inf(vec3s v);
   PLAY_CGLM_INLINE vec3s vec3s_add(vec3s a, vec3s b);
   PLAY_CGLM_INLINE vec3s vec3s_adds(vec3s a, float s);
   PLAY_CGLM_INLINE vec3s vec3s_sub(vec3s a, vec3s b);
   PLAY_CGLM_INLINE vec3s vec3s_subs(vec3s a, float s);
   PLAY_CGLM_INLINE vec3s vec3s_mul(vec3s a, vec3s b);
   PLAY_CGLM_INLINE vec3s vec3s_scale(vec3s v, float s);
   PLAY_CGLM_INLINE vec3s vec3s_scale_as(vec3s v, float s);
   PLAY_CGLM_INLINE vec3s vec3s_div(vec3s a, vec3s b);
   PLAY_CGLM_INLINE vec3s vec3s_divs(vec3s a, float s);
   PLAY_CGLM_INLINE vec3s vec3s_addadd(vec3s a, vec3s b, vec3s dest);
   PLAY_CGLM_INLINE vec3s vec3s_subadd(vec3s a, vec3s b, vec3s dest);
   PLAY_CGLM_INLINE vec3s vec3s_muladd(vec3s a, vec3s b, vec3s dest);
   PLAY_CGLM_INLINE vec3s vec3s_muladds(vec3s a, float s, vec3s dest);
   PLAY_CGLM_INLINE vec3s vec3s_maxadd(vec3s a, vec3s b, vec3s dest);
   PLAY_CGLM_INLINE vec3s vec3s_minadd(vec3s a, vec3s b, vec3s dest);
   PLAY_CGLM_INLINE vec3s vec3s_subsub(vec3s a, vec3s b, vec3s dest);
   PLAY_CGLM_INLINE vec3s vec3s_addsub(vec3s a, vec3s b, vec3s dest);
   PLAY_CGLM_INLINE vec3s vec3s_mulsub(vec3s a, vec3s b, vec3s dest);
   PLAY_CGLM_INLINE vec3s vec3s_mulsubs(vec3s a, float s, vec3s dest);
   PLAY_CGLM_INLINE vec3s vec3s_maxsub(vec3s a, vec3s b, vec3s dest);
   PLAY_CGLM_INLINE vec3s vec3s_minsub(vec3s a, vec3s b, vec3s dest);
   PLAY_CGLM_INLINE vec3s vec3s_flipsign(vec3s v);
   PLAY_CGLM_INLINE vec3s vec3s_negate(vec3s v);
   PLAY_CGLM_INLINE vec3s vec3s_normalize(vec3s v);
   PLAY_CGLM_INLINE vec3s vec3s_cross(vec3s a, vec3s b);
   PLAY_CGLM_INLINE vec3s vec3s_crossn(vec3s a, vec3s b);
   PLAY_CGLM_INLINE float vec3s_angle(vec3s a, vec3s b);
   PLAY_CGLM_INLINE vec3s vec3s_rotate(vec3s v, float angle, vec3s axis);
   PLAY_CGLM_INLINE vec3s vec3s_rotate_m4(mat4s m, vec3s v);
   PLAY_CGLM_INLINE vec3s vec3s_rotate_m3(mat3s m, vec3s v);
   PLAY_CGLM_INLINE vec3s vec3s_proj(vec3s a, vec3s b);
   PLAY_CGLM_INLINE vec3s vec3s_center(vec3s a, vec3s b);
   PLAY_CGLM_INLINE float vec3s_distance(vec3s a, vec3s b);
   PLAY_CGLM_INLINE float vec3s_distance2(vec3s a, vec3s b);
   PLAY_CGLM_INLINE vec3s vec3s_maxv(vec3s a, vec3s b);
   PLAY_CGLM_INLINE vec3s vec3s_minv(vec3s a, vec3s b);
   PLAY_CGLM_INLINE vec3s vec3s_ortho(vec3s v);
   PLAY_CGLM_INLINE vec3s vec3s_clamp(vec3s v, float minVal, float maxVal);
   PLAY_CGLM_INLINE vec3s vec3s_lerp(vec3s from, vec3s to, float t);
   PLAY_CGLM_INLINE vec3s vec3s_lerpc(vec3s from, vec3s to, float t);
   PLAY_CGLM_INLINE vec3s vec3s_mix(vec3s from, vec3s to, float t);
   PLAY_CGLM_INLINE vec3s vec3s_mixc(vec3s from, vec3s to, float t);
   PLAY_CGLM_INLINE vec3s vec3s_step(vec3s edge, vec3s x);
   PLAY_CGLM_INLINE vec3s vec3s_smoothstep_uni(float edge0, float edge1, vec3s x);
   PLAY_CGLM_INLINE vec3s vec3s_smoothstep(vec3s edge0, vec3s edge1, vec3s x);
   PLAY_CGLM_INLINE vec3s vec3s_smoothinterp(vec3s from, vec3s to, float t);
   PLAY_CGLM_INLINE vec3s vec3s_smoothinterpc(vec3s from, vec3s to, float t);
   PLAY_CGLM_INLINE vec3s vec3s_swizzle(vec3s v, int mask);
   PLAY_CGLM_INLINE vec3s vec3s_make(float * restrict src);
   PLAY_CGLM_INLINE vec3s vec3s_faceforward(vec3s n, vec3s v, vec3s nref);
   PLAY_CGLM_INLINE vec3s vec3s_reflect(vec3s v, vec3s n);
   PLAY_CGLM_INLINE bool  vec3s_refract(vec3s v, vec3s n, float eta, vec3s *dest)

 Convenient:
   PLAY_CGLM_INLINE vec3s cross(vec3s a, vec3s b);
   PLAY_CGLM_INLINE float dot(vec3s a, vec3s b);
   PLAY_CGLM_INLINE vec3s normalize(vec3s v);

 Deprecated:
   vec3s_step_uni  -->  use vec3s_steps
 */





/*** Start of inlined file: vec3-ext.h ***/
/*!
 * @brief SIMD like functions
 */

/*
 Functions:
   PLAY_CGLM_INLINE vec3s vec3s_broadcast(float val);
   PLAY_CGLM_INLINE vec3s vec3s_fill(float val);
   PLAY_CGLM_INLINE bool  vec3s_eq(vec3s v, float val);
   PLAY_CGLM_INLINE bool  vec3s_eq_eps(vec3s v, float val);
   PLAY_CGLM_INLINE bool  vec3s_eq_all(vec3s v);
   PLAY_CGLM_INLINE bool  vec3s_eqv(vec3s a, vec3s b);
   PLAY_CGLM_INLINE bool  vec3s_eqv_eps(vec3s a, vec3s b);
   PLAY_CGLM_INLINE float vec3s_max(vec3s v);
   PLAY_CGLM_INLINE float vec3s_min(vec3s v);
   PLAY_CGLM_INLINE bool  vec3s_isnan(vec3s v);
   PLAY_CGLM_INLINE bool  vec3s_isinf(vec3s v);
   PLAY_CGLM_INLINE bool  vec3s_isvalid(vec3s v);
   PLAY_CGLM_INLINE vec3s vec3s_sign(vec3s v);
   PLAY_CGLM_INLINE vec3s vec3s_abs(vec3s v);
   PLAY_CGLM_INLINE vec3s vec3s_fract(vec3s v);
   PLAY_CGLM_INLINE vec3s vec3s_floor(vec3s v);
   PLAY_CGLM_INLINE vec3s vec3s_mods(vec3s v, float s);
   PLAY_CGLM_INLINE vec3s vec3s_steps(float edge, vec3s v);
   PLAY_CGLM_INLINE vec3s vec3s_stepr(vec3s edge, float v);
   PLAY_CGLM_INLINE float vec3s_hadd(vec3s v);
   PLAY_CGLM_INLINE vec3s vec3s_sqrt(vec3s v);
 */




/* api definition */
#define vec3s_(NAME) PLAY_CGLM_STRUCTAPI(vec3, NAME)

/*!
 * @brief fill a vector with specified value
 *
 * @param[in]  val  value
 * @returns         dest
 */
PLAY_CGLM_INLINE
vec3s
vec3s_(broadcast)(float val)
{
    vec3s r;
    vec3_broadcast(val, r.raw);
    return r;
}

/*!
 * @brief fill a vector with specified value
 *
 * @param[in]  val  value
 * @returns         dest
 */
PLAY_CGLM_INLINE
vec3s
vec3s_(fill)(float val)
{
    vec3s r;
    vec3_fill(r.raw, val);
    return r;
}

/*!
 * @brief check if vector is equal to value (without epsilon)
 *
 * @param[in] v   vector
 * @param[in] val value
 */
PLAY_CGLM_INLINE
bool
vec3s_(eq)(vec3s v, float val)
{
    return vec3_eq(v.raw, val);
}

/*!
 * @brief check if vector is equal to value (with epsilon)
 *
 * @param[in] v   vector
 * @param[in] val value
 */
PLAY_CGLM_INLINE
bool
vec3s_(eq_eps)(vec3s v, float val)
{
    return vec3_eq_eps(v.raw, val);
}

/*!
 * @brief check if vector members are equal (without epsilon)
 *
 * @param[in] v   vector
 */
PLAY_CGLM_INLINE
bool
vec3s_(eq_all)(vec3s v)
{
    return vec3_eq_all(v.raw);
}

/*!
 * @brief check if vector is equal to another (without epsilon)
 *
 * @param[in] a vector
 * @param[in] b vector
 */
PLAY_CGLM_INLINE
bool
vec3s_(eqv)(vec3s a, vec3s b)
{
    return vec3_eqv(a.raw, b.raw);
}

/*!
 * @brief check if vector is equal to another (with epsilon)
 *
 * @param[in] a vector
 * @param[in] b vector
 */
PLAY_CGLM_INLINE
bool
vec3s_(eqv_eps)(vec3s a, vec3s b)
{
    return vec3_eqv_eps(a.raw, b.raw);
}

/*!
 * @brief max value of vector
 *
 * @param[in] v vector
 */
PLAY_CGLM_INLINE
float
vec3s_(max)(vec3s v)
{
    return vec3_max(v.raw);
}

/*!
 * @brief min value of vector
 *
 * @param[in] v vector
 */
PLAY_CGLM_INLINE
float
vec3s_(min)(vec3s v)
{
    return vec3_min(v.raw);
}

/*!
 * @brief check if one of items is NaN (not a number)
 *        you should only use this in DEBUG mode or very critical asserts
 *
 * @param[in] v vector
 */
PLAY_CGLM_INLINE
bool
vec3s_(isnan)(vec3s v)
{
    return vec3_isnan(v.raw);
}

/*!
 * @brief check if one of items is INFINITY
 *        you should only use this in DEBUG mode or very critical asserts
 *
 * @param[in] v vector
 */
PLAY_CGLM_INLINE
bool
vec3s_(isinf)(vec3s v)
{
    return vec3_isinf(v.raw);
}

/*!
 * @brief check if all items are valid number
 *        you should only use this in DEBUG mode or very critical asserts
 *
 * @param[in] v vector
 */
PLAY_CGLM_INLINE
bool
vec3s_(isvalid)(vec3s v)
{
    return vec3_isvalid(v.raw);
}

/*!
 * @brief get sign of 32 bit float as +1, -1, 0
 *
 * Important: It returns 0 for zero/NaN input
 *
 * @param   v   vector
 * @returns     sign vector
 */
PLAY_CGLM_INLINE
vec3s
vec3s_(sign)(vec3s v)
{
    vec3s r;
    vec3_sign(v.raw, r.raw);
    return r;
}

/*!
 * @brief absolute value of each vector item
 *
 * @param[in]  v    vector
 * @return          destination vector
 */
PLAY_CGLM_INLINE
vec3s
vec3s_(abs)(vec3s v)
{
    vec3s r;
    vec3_abs(v.raw, r.raw);
    return r;
}

/*!
 * @brief fractional part of each vector item
 *
 * @param[in]  v    vector
 * @return          dest destination vector
 */
PLAY_CGLM_INLINE
vec3s
vec3s_(fract)(vec3s v)
{
    vec3s r;
    vec3_fract(v.raw, r.raw);
    return r;
}

/*!
 * @brief floor of each vector item
 *
 * @param[in]  v    vector
 * @return          dest destination vector
 */
PLAY_CGLM_INLINE
vec3s
vec3s_(floor)(vec3s v)
{
    vec3s r;
    vec3_floor(v.raw, r.raw);
    return r;
}

/*!
 * @brief mod of each vector item by scalar
 *
 * @param[in]  v    vector
 * @param[in]  s    scalar
 * @returns         destination vector
 */
PLAY_CGLM_INLINE
vec3s
vec3s_(mods)(vec3s v, float s)
{
    vec3s r;
    vec3_mods(v.raw, s, r.raw);
    return r;
}

/*!
 * @brief threshold each vector item with scalar
 *        condition is: (x[i] < edge) ? 0.0 : 1.0
 *
 * @param[in]   edge   threshold
 * @param[in]   x      vector to test against threshold
 * @returns            destination
 */
PLAY_CGLM_INLINE
vec3s
vec3s_(steps)(float edge, vec3s x)
{
    vec3s r;
    vec3_steps(edge, x.raw, r.raw);
    return r;
}

/*!
 * @brief threshold a value with *vector* as the threshold
 *        condition is: (x < edge[i]) ? 0.0 : 1.0
 *
 * @param[in]   edge   threshold vector
 * @param[in]   x      value to test against threshold
 * @returns            destination
 */
PLAY_CGLM_INLINE
vec3s
vec3s_(stepr)(vec3s edge, float x)
{
    vec3s r;
    vec3_stepr(edge.raw, x, r.raw);
    return r;
}

/*!
 * @brief vector reduction by summation
 * @warning could overflow
 *
 * @param[in]  v    vector
 * @return     sum of all vector's elements
 */
PLAY_CGLM_INLINE
float
vec3s_(hadd)(vec3s v)
{
    return vec3_hadd(v.raw);
}

/*!
 * @brief square root of each vector item
 *
 * @param[in]  v    vector
 * @returns         destination vector
 */
PLAY_CGLM_INLINE
vec3s
vec3s_(sqrt)(vec3s v)
{
    vec3s r;
    vec3_sqrt(v.raw, r.raw);
    return r;
}



/*** End of inlined file: vec3-ext.h ***/

/* DEPRECATED! */
#define vec3s_step_uni(edge, x) vec3s_steps(edge, x)

#define PLAY_CGLM_S_VEC3_ONE_INIT   {PLAY_CGLM_VEC3_ONE_INIT}
#define PLAY_CGLM_S_VEC3_ZERO_INIT  {PLAY_CGLM_VEC3_ZERO_INIT}

#define PLAY_CGLM_S_VEC3_ONE  ((vec3s)PLAY_CGLM_S_VEC3_ONE_INIT)
#define PLAY_CGLM_S_VEC3_ZERO ((vec3s)PLAY_CGLM_S_VEC3_ZERO_INIT)

#define PLAY_CGLM_S_YUP  ((vec3s){{0.0f, 1.0f, 0.0f}})
#define PLAY_CGLM_S_ZUP  ((vec3s){{0.0f, 0.0f, 1.0f}})
#define PLAY_CGLM_S_XUP  ((vec3s){{1.0f, 0.0f, 0.0f}})

/*!
 * @brief init vec3 using vec4
 *
 * @param[in]  v4   vector4
 * @returns         destination
 */
PLAY_CGLM_INLINE
vec3s
vec3_new(vec4s v4)
{
    vec3s r;
    vec3_new(v4.raw, r.raw);
    return r;
}

/*!
 * @brief pack an array of vec3 into an array of vec3s
 *
 * @param[out] dst array of vec3
 * @param[in]  src array of vec3s
 * @param[in]  len number of elements
 */
PLAY_CGLM_INLINE
void
vec3s_(pack)(vec3s dst[], vec3 src[], size_t len)
{
    size_t i;

    for (i = 0; i < len; i++)
    {
        vec3_copy(src[i], dst[i].raw);
    }
}

/*!
 * @brief unpack an array of vec3s into an array of vec3
 *
 * @param[out] dst array of vec3s
 * @param[in]  src array of vec3
 * @param[in]  len number of elements
 */
PLAY_CGLM_INLINE
void
vec3s_(unpack)(vec3 dst[], vec3s src[], size_t len)
{
    size_t i;

    for (i = 0; i < len; i++)
    {
        vec3_copy(src[i].raw, dst[i]);
    }
}

/*!
 * @brief make vector zero
 *
 * @returns       zero vector
 */
PLAY_CGLM_INLINE
vec3s
vec3s_(zero)(void)
{
    vec3s r;
    vec3_zero(r.raw);
    return r;
}

/*!
 * @brief make vector one
 *
 * @returns       one vector
 */
PLAY_CGLM_INLINE
vec3s
vec3s_(one)(void)
{
    vec3s r;
    vec3_one(r.raw);
    return r;
}

/*!
 * @brief vec3 dot product
 *
 * @param[in] a vector1
 * @param[in] b vector2
 *
 * @return dot product
 */
PLAY_CGLM_INLINE
float
vec3s_(dot)(vec3s a, vec3s b)
{
    return vec3_dot(a.raw, b.raw);
}

/*!
 * @brief norm * norm (magnitude) of vec
 *
 * we can use this func instead of calling norm * norm, because it would call
 * sqrtf function twice but with this func we can avoid func call, maybe this is
 * not good name for this func
 *
 * @param[in] v vector
 *
 * @return norm * norm
 */
PLAY_CGLM_INLINE
float
vec3s_(norm2)(vec3s v)
{
    return vec3_norm2(v.raw);
}

/*!
 * @brief norm (magnitude) of vec3
 *
 * @param[in] v vector
 *
 * @return norm
 */
PLAY_CGLM_INLINE
float
vec3s_(norm)(vec3s v)
{
    return vec3_norm(v.raw);
}

/*!
 * @brief L1 norm of vec3
 * Also known as Manhattan Distance or Taxicab norm.
 * L1 Norm is the sum of the magnitudes of the vectors in a space.
 * It is calculated as the sum of the absolute values of the vector components.
 * In this norm, all the components of the vector are weighted equally.
 *
 * This computes:
 * R = |v[0]| + |v[1]| + |v[2]|
 *
 * @param[in] v vector
 *
 * @return L1 norm
 */
PLAY_CGLM_INLINE
float
vec3s_(norm_one)(vec3s v)
{
    return vec3_norm_one(v.raw);
}

/*!
 * @brief Infinity norm of vec3
 * Also known as Maximum norm.
 * Infinity Norm is the largest magnitude among each element of a vector.
 * It is calculated as the maximum of the absolute values of the vector components.
 *
 * This computes:
 * inf norm = fmax(|v[0]|, |v[1]|, |v[2]|)
 *
 * @param[in] v vector
 *
 * @return Infinity norm
 */
PLAY_CGLM_INLINE
float
vec3s_(norm_inf)(vec3s v)
{
    return vec3_norm_inf(v.raw);
}

/*!
 * @brief add a vector to b vector store result in dest
 *
 * @param[in]  a    vector1
 * @param[in]  b    vector2
 * @returns         destination vector
 */
PLAY_CGLM_INLINE
vec3s
vec3s_(add)(vec3s a, vec3s b)
{
    vec3s r;
    vec3_add(a.raw, b.raw, r.raw);
    return r;
}

/*!
 * @brief add scalar to v vector store result in dest (d = v + s)
 *
 * @param[in]  a    vector
 * @param[in]  s    scalar
 * @returns         destination vector
 */
PLAY_CGLM_INLINE
vec3s
vec3s_(adds)(vec3s a, float s)
{
    vec3s r;
    vec3_adds(a.raw, s, r.raw);
    return r;
}

/*!
 * @brief subtract b vector from a vector store result in dest
 *
 * @param[in]  a    vector1
 * @param[in]  b    vector2
 * @returns         destination vector
 */
PLAY_CGLM_INLINE
vec3s
vec3s_(sub)(vec3s a, vec3s b)
{
    vec3s r;
    vec3_sub(a.raw, b.raw, r.raw);
    return r;
}

/*!
 * @brief subtract scalar from v vector store result in dest (d = v - s)
 *
 * @param[in]  a    vector
 * @param[in]  s    scalar
 * @returns         destination vector
 */
PLAY_CGLM_INLINE
vec3s
vec3s_(subs)(vec3s a, float s)
{
    vec3s r;
    vec3_subs(a.raw, s, r.raw);
    return r;
}

/*!
 * @brief multiply two vectors (component-wise multiplication)
 *
 * @param     a     vector1
 * @param     b     vector2
 * @returns         v3 = (a[0] * b[0], a[1] * b[1], a[2] * b[2])
 */
PLAY_CGLM_INLINE
vec3s
vec3s_(mul)(vec3s a, vec3s b)
{
    vec3s r;
    vec3_mul(a.raw, b.raw, r.raw);
    return r;
}

/*!
 * @brief multiply/scale vec3 vector with scalar: result = v * s
 *
 * @param[in]  v    vector
 * @param[in]  s    scalar
 * @returns         destination vector
 */
PLAY_CGLM_INLINE
vec3s
vec3s_(scale)(vec3s v, float s)
{
    vec3s r;
    vec3_scale(v.raw, s, r.raw);
    return r;
}

/*!
 * @brief make vec3 vector scale as specified: result = unit(v) * s
 *
 * @param[in]  v    vector
 * @param[in]  s    scalar
 * @returns         destination vector
 */
PLAY_CGLM_INLINE
vec3s
vec3s_(scale_as)(vec3s v, float s)
{
    vec3s r;
    vec3_scale_as(v.raw, s, r.raw);
    return r;
}

/*!
 * @brief div vector with another component-wise division: d = a / b
 *
 * @param[in]  a    vector 1
 * @param[in]  b    vector 2
 * @returns         result = (a[0]/b[0], a[1]/b[1], a[2]/b[2])
 */
PLAY_CGLM_INLINE
vec3s
vec3s_(div)(vec3s a, vec3s b)
{
    vec3s r;
    vec3_div(a.raw, b.raw, r.raw);
    return r;
}

/*!
 * @brief div vector with scalar: d = v / s
 *
 * @param[in]  a    vector
 * @param[in]  s    scalar
 * @returns         result = (a[0]/s, a[1]/s, a[2]/s)
 */
PLAY_CGLM_INLINE
vec3s
vec3s_(divs)(vec3s a, float s)
{
    vec3s r;
    vec3_divs(a.raw, s, r.raw);
    return r;
}

/*!
 * @brief add two vectors and add result to sum
 *
 * it applies += operator so dest must be initialized
 *
 * @param[in]  a    vector 1
 * @param[in]  b    vector 2
 * @returns         dest += (a + b)
 */
PLAY_CGLM_INLINE
vec3s
vec3s_(addadd)(vec3s a, vec3s b, vec3s dest)
{
    vec3_addadd(a.raw, b.raw, dest.raw);
    return dest;
}

/*!
 * @brief sub two vectors and add result to dest
 *
 * it applies += operator so dest must be initialized
 *
 * @param[in]  a    vector 1
 * @param[in]  b    vector 2
 * @returns         dest += (a + b)
 */
PLAY_CGLM_INLINE
vec3s
vec3s_(subadd)(vec3s a, vec3s b, vec3s dest)
{
    vec3_subadd(a.raw, b.raw, dest.raw);
    return dest;
}

/*!
 * @brief mul two vectors and add result to dest
 *
 * it applies += operator so dest must be initialized
 *
 * @param[in]  a    vector 1
 * @param[in]  b    vector 2
 * @returns         dest += (a * b)
 */
PLAY_CGLM_INLINE
vec3s
vec3s_(muladd)(vec3s a, vec3s b, vec3s dest)
{
    vec3_muladd(a.raw, b.raw, dest.raw);
    return dest;
}

/*!
 * @brief mul vector with scalar and add result to sum
 *
 * it applies += operator so dest must be initialized
 *
 * @param[in]  a    vector
 * @param[in]  s    scalar
 * @returns         dest += (a * b)
 */
PLAY_CGLM_INLINE
vec3s
vec3s_(muladds)(vec3s a, float s, vec3s dest)
{
    vec3_muladds(a.raw, s, dest.raw);
    return dest;
}

/*!
 * @brief add max of two vectors to result/dest
 *
 * it applies += operator so dest must be initialized
 *
 * @param[in]  a    vector 1
 * @param[in]  b    vector 2
 * @returns         dest += fmax(a, b)
 */
PLAY_CGLM_INLINE
vec3s
vec3s_(maxadd)(vec3s a, vec3s b, vec3s dest)
{
    vec3_maxadd(a.raw, b.raw, dest.raw);
    return dest;
}

/*!
 * @brief add min of two vectors to result/dest
 *
 * it applies += operator so dest must be initialized
 *
 * @param[in]  a    vector 1
 * @param[in]  b    vector 2
 * @returns         dest += fmin(a, b)
 */
PLAY_CGLM_INLINE
vec3s
vec3s_(minadd)(vec3s a, vec3s b, vec3s dest)
{
    vec3_minadd(a.raw, b.raw, dest.raw);
    return dest;
}

/*!
 * @brief sub two vectors and sub result to dest
 *
 * it applies -= operator so dest must be initialized
 *
 * @param[in]  a    vector 1
 * @param[in]  b    vector 2
 * @returns         dest -= (a - b)
 */
PLAY_CGLM_INLINE
vec3s
vec3s_(subsub)(vec3s a, vec3s b, vec3s dest)
{
    vec3_subsub(a.raw, b.raw, dest.raw);
    return dest;
}

/*!
 * @brief add two vectors and sub result to dest
 *
 * it applies -= operator so dest must be initialized
 *
 * @param[in]  a    vector 1
 * @param[in]  b    vector 2
 * @returns         dest -= (a + b)
 */
PLAY_CGLM_INLINE
vec3s
vec3s_(addsub)(vec3s a, vec3s b, vec3s dest)
{
    vec3_addsub(a.raw, b.raw, dest.raw);
    return dest;
}

/*!
 * @brief mul two vectors and sub result to dest
 *
 * it applies -= operator so dest must be initialized
 *
 * @param[in]  a    vector 1
 * @param[in]  b    vector 2
 * @returns         dest -= (a * b)
 */
PLAY_CGLM_INLINE
vec3s
vec3s_(mulsub)(vec3s a, vec3s b, vec3s dest)
{
    vec3_mulsub(a.raw, b.raw, dest.raw);
    return dest;
}

/*!
 * @brief mul vector with scalar and sub result to dest
 *
 * it applies -= operator so dest must be initialized
 *
 * @param[in]  a    vector
 * @param[in]  s    scalar
 * @returns         dest -= (a * b)
 */
PLAY_CGLM_INLINE
vec3s
vec3s_(mulsubs)(vec3s a, float s, vec3s dest)
{
    vec3_mulsubs(a.raw, s, dest.raw);
    return dest;
}

/*!
 * @brief sub max of two vectors to dest
 *
 * it applies -= operator so dest must be initialized
 *
 * @param[in]  a    vector 1
 * @param[in]  b    vector 2
 * @returns         dest -= fmax(a, b)
 */
PLAY_CGLM_INLINE
vec3s
vec3s_(maxsub)(vec3s a, vec3s b, vec3s dest)
{
    vec3_maxsub(a.raw, b.raw, dest.raw);
    return dest;
}

/*!
 * @brief sub min of two vectors to dest
 *
 * it applies -= operator so dest must be initialized
 *
 * @param[in]  a    vector 1
 * @param[in]  b    vector 2
 * @returns         dest -= fmin(a, b)
 */
PLAY_CGLM_INLINE
vec3s
vec3s_(minsub)(vec3s a, vec3s b, vec3s dest)
{
    vec3_minsub(a.raw, b.raw, dest.raw);
    return dest;
}

/*!
 * @brief negate vector components and store result in dest
 *
 * @param[in]   v     vector
 * @returns           result vector
 */
PLAY_CGLM_INLINE
vec3s
vec3s_(flipsign)(vec3s v)
{
    vec3_flipsign(v.raw);
    return v;
}

/*!
 * @brief negate vector components
 *
 * @param[in]  v  vector
 * @returns       negated vector
 */
PLAY_CGLM_INLINE
vec3s
vec3s_(negate)(vec3s v)
{
    vec3_negate(v.raw);
    return v;
}

/*!
 * @brief normalize vec3 and store result in same vec
 *
 * @param[in] v vector
 * @returns     normalized vector
 */
PLAY_CGLM_INLINE
vec3s
vec3s_(normalize)(vec3s v)
{
    vec3_normalize(v.raw);
    return v;
}

/*!
 * @brief cross product of two vector (RH)
 *
 * @param[in]  a    vector 1
 * @param[in]  b    vector 2
 * @returns         destination
 */
PLAY_CGLM_INLINE
vec3s
vec3s_(cross)(vec3s a, vec3s b)
{
    vec3s r;
    vec3_cross(a.raw, b.raw, r.raw);
    return r;
}

/*!
 * @brief cross product of two vector (RH) and normalize the result
 *
 * @param[in]  a    vector 1
 * @param[in]  b    vector 2
 * @returns         destination
 */
PLAY_CGLM_INLINE
vec3s
vec3s_(crossn)(vec3s a, vec3s b)
{
    vec3s r;
    vec3_crossn(a.raw, b.raw, r.raw);
    return r;
}

/*!
 * @brief angle between two vector
 *
 * @param[in] a  vector1
 * @param[in] b  vector2
 *
 * @return angle as radians
 */
PLAY_CGLM_INLINE
float
vec3s_(angle)(vec3s a, vec3s b)
{
    return vec3_angle(a.raw, b.raw);
}

/*!
 * @brief rotate vec3 around axis by angle using Rodrigues' rotation formula
 *
 * @param[in]     v     vector
 * @param[in]     axis  axis vector (must be unit vector)
 * @param[in]     angle angle by radians
 * @returns             rotated vector
 */
PLAY_CGLM_INLINE
vec3s
vec3s_(rotate)(vec3s v, float angle, vec3s axis)
{
    vec3_rotate(v.raw, angle, axis.raw);
    return v;
}

/*!
 * @brief apply rotation matrix to vector
 *
 *  matrix format should be (no perspective):
 *   a  b  c  x
 *   e  f  g  y
 *   i  j  k  z
 *   0  0  0  w
 *
 * @param[in]  m    affine matrix or rot matrix
 * @param[in]  v    vector
 * @returns         rotated vector
 */
PLAY_CGLM_INLINE
vec3s
vec3s_(rotate_m4)(mat4s m, vec3s v)
{
    vec3s r;
    vec3_rotate_m4(m.raw, v.raw, r.raw);
    return r;
}

/*!
 * @brief apply rotation matrix to vector
 *
 * @param[in]  m    affine matrix or rot matrix
 * @param[in]  v    vector
 * @returns         rotated vector
 */
PLAY_CGLM_INLINE
vec3s
vec3s_(rotate_m3)(mat3s m, vec3s v)
{
    vec3s r;
    vec3_rotate_m3(m.raw, v.raw, r.raw);
    return r;
}

/*!
 * @brief project a vector onto b vector
 *
 * @param[in]  a    vector1
 * @param[in]  b    vector2
 * @returns         projected vector
 */
PLAY_CGLM_INLINE
vec3s
vec3s_(proj)(vec3s a, vec3s b)
{
    vec3s r;
    vec3_proj(a.raw, b.raw, r.raw);
    return r;
}

/**
 * @brief find center point of two vector
 *
 * @param[in]  a    vector1
 * @param[in]  b    vector2
 * @returns         center point
 */
PLAY_CGLM_INLINE
vec3s
vec3s_(center)(vec3s a, vec3s b)
{
    vec3s r;
    vec3_center(a.raw, b.raw, r.raw);
    return r;
}

/**
 * @brief distance between two vectors
 *
 * @param[in] a vector1
 * @param[in] b vector2
 * @return      distance
 */
PLAY_CGLM_INLINE
float
vec3s_(distance)(vec3s a, vec3s b)
{
    return vec3_distance(a.raw, b.raw);
}

/**
 * @brief squared distance between two vectors
 *
 * @param[in] a vector1
 * @param[in] b vector2
 * @return      squared distance (distance * distance)
 */
PLAY_CGLM_INLINE
float
vec3s_(distance2)(vec3s a, vec3s b)
{
    return vec3_distance2(a.raw, b.raw);
}

/*!
 * @brief max values of vectors
 *
 * @param[in]  a    vector1
 * @param[in]  b    vector2
 * @returns         destination
 */
PLAY_CGLM_INLINE
vec3s
vec3s_(maxv)(vec3s a, vec3s b)
{
    vec3s r;
    vec3_maxv(a.raw, b.raw, r.raw);
    return r;
}

/*!
 * @brief min values of vectors
 *
 * @param[in]  a    vector1
 * @param[in]  b    vector2
 * @returns         destination
 */
PLAY_CGLM_INLINE
vec3s
vec3s_(minv)(vec3s a, vec3s b)
{
    vec3s r;
    vec3_minv(a.raw, b.raw, r.raw);
    return r;
}

/*!
 * @brief possible orthogonal/perpendicular vector
 *
 * @param[in]  v    vector
 * @returns         orthogonal/perpendicular vector
 */
PLAY_CGLM_INLINE
vec3s
vec3s_(ortho)(vec3s v)
{
    vec3s r;
    vec3_ortho(v.raw, r.raw);
    return r;
}

/*!
 * @brief clamp vector's individual members between min and max values
 *
 * @param[in]       v       vector
 * @param[in]       minVal  minimum value
 * @param[in]       maxVal  maximum value
 * @returns                 clamped vector
 */
PLAY_CGLM_INLINE
vec3s
vec3s_(clamp)(vec3s v, float minVal, float maxVal)
{
    vec3_clamp(v.raw, minVal, maxVal);
    return v;
}

/*!
 * @brief linear interpolation between two vectors
 *
 * formula:  from + s * (to - from)
 *
 * @param[in]   from  from value
 * @param[in]   to    to value
 * @param[in]   t     interpolant (amount)
 * @returns           destination
 */
PLAY_CGLM_INLINE
vec3s
vec3s_(lerp)(vec3s from, vec3s to, float t)
{
    vec3s r;
    vec3_lerp(from.raw, to.raw, t, r.raw);
    return r;
}

/*!
 * @brief linear interpolation between two vectors (clamped)
 *
 * formula:  from + s * (to - from)
 *
 * @param[in]   from  from value
 * @param[in]   to    to value
 * @param[in]   t     interpolant (amount) clamped between 0 and 1
 * @returns           destination
 */
PLAY_CGLM_INLINE
vec3s
vec3s_(lerpc)(vec3s from, vec3s to, float t)
{
    vec3s r;
    vec3_lerpc(from.raw, to.raw, t, r.raw);
    return r;
}

/*!
 * @brief linear interpolation between two vectors
 *
 * formula:  from + s * (to - from)
 *
 * @param[in]   from  from value
 * @param[in]   to    to value
 * @param[in]   t     interpolant (amount)
 * @returns           destination
 */
PLAY_CGLM_INLINE
vec3s
vec3s_(mix)(vec3s from, vec3s to, float t)
{
    vec3s r;
    vec3_mix(from.raw, to.raw, t, r.raw);
    return r;
}

/*!
 * @brief linear interpolation between two vectors (clamped)
 *
 * formula:  from + s * (to - from)
 *
 * @param[in]   from  from value
 * @param[in]   to    to value
 * @param[in]   t     interpolant (amount) clamped between 0 and 1
 * @returns           destination
 */
PLAY_CGLM_INLINE
vec3s
vec3s_(mixc)(vec3s from, vec3s to, float t)
{
    vec3s r;
    vec3_mixc(from.raw, to.raw, t, r.raw);
    return r;
}

/*!
 * @brief threshold function
 *
 * @param[in]   edge    threshold
 * @param[in]   x       value to test against threshold
 * @returns             0.0 if x < edge, else 1.0
 */
PLAY_CGLM_INLINE
vec3s
vec3s_(step)(vec3s edge, vec3s x)
{
    vec3s r;
    vec3_step(edge.raw, x.raw, r.raw);
    return r;
}

/*!
 * @brief threshold function with a smooth transition (unidimensional)
 *
 * @param[in]   edge0   low threshold
 * @param[in]   edge1   high threshold
 * @param[in]   x       value to test against threshold
 * @returns             destination
 */
PLAY_CGLM_INLINE
vec3s
vec3s_(smoothstep_uni)(float edge0, float edge1, vec3s x)
{
    vec3s r;
    vec3_smoothstep_uni(edge0, edge1, x.raw, r.raw);
    return r;
}

/*!
 * @brief threshold function with a smooth transition
 *
 * @param[in]   edge0   low threshold
 * @param[in]   edge1   high threshold
 * @param[in]   x       value to test against threshold
 * @returns             destination
 */
PLAY_CGLM_INLINE
vec3s
vec3s_(smoothstep)(vec3s edge0, vec3s edge1, vec3s x)
{
    vec3s r;
    vec3_smoothstep(edge0.raw, edge1.raw, x.raw, r.raw);
    return r;
}

/*!
 * @brief smooth Hermite interpolation between two vectors
 *
 * formula:  from + s * (to - from)
 *
 * @param[in]   from    from value
 * @param[in]   to      to value
 * @param[in]   t       interpolant (amount)
 * @returns             destination
 */
PLAY_CGLM_INLINE
vec3s
vec3s_(smoothinterp)(vec3s from, vec3s to, float t)
{
    vec3s r;
    vec3_smoothinterp(from.raw, to.raw, t, r.raw);
    return r;
}

/*!
 * @brief smooth Hermite interpolation between two vectors (clamped)
 *
 * formula:  from + s * (to - from)
 *
 * @param[in]   from    from value
 * @param[in]   to      to value
 * @param[in]   t       interpolant (amount) clamped between 0 and 1
 * @returns             destination
 */
PLAY_CGLM_INLINE
vec3s
vec3s_(smoothinterpc)(vec3s from, vec3s to, float t)
{
    vec3s r;
    vec3_smoothinterpc(from.raw, to.raw, t, r.raw);
    return r;
}

/*!
 * @brief vec3 cross product
 *
 * this is just convenient wrapper
 *
 * @param[in]  a  source 1
 * @param[in]  b  source 2
 * @returns       destination
 */
PLAY_CGLM_INLINE
vec3s
cross(vec3s a, vec3s b)
{
    vec3s r;
    cross(a.raw, b.raw, r.raw);
    return r;
}

/*!
 * @brief vec3 dot product
 *
 * this is just convenient wrapper
 *
 * @param[in] a vector1
 * @param[in] b vector2
 * @return      dot product
 */
PLAY_CGLM_INLINE
float
dot(vec3s a, vec3s b)
{
    return dot(a.raw, b.raw);
}

/*!
 * @brief normalize vec3 and store result in same vec
 *
 * this is just convenient wrapper
 *
 * @param[in]   v   vector
 * @returns         normalized vector
 */
PLAY_CGLM_INLINE
vec3s
normalize(vec3s v)
{
    normalize(v.raw);
    return v;
}

/*!
 * @brief swizzle vector components
 *
 * you can use existing masks e.g. PLAY_CGLM_XXX, PLAY_CGLM_ZYX
 *
 * @param[in]  v    source
 * @param[in]  mask mask
 * @returns swizzled vector
 */
PLAY_CGLM_INLINE
vec3s
vec3s_(swizzle)(vec3s v, int mask)
{
    vec3s dest;
    vec3_swizzle(v.raw, mask, dest.raw);
    return dest;
}

/*!
 * @brief Create three dimensional vector from pointer
 *
 * @param[in]  src  pointer to an array of floats
 * @returns constructed 3D vector from raw pointer
 */
PLAY_CGLM_INLINE
vec3s
vec3s_(make)(const float * __restrict src)
{
    vec3s dest;
    vec3_make(src, dest.raw);
    return dest;
}

/*!
 * @brief a vector pointing in the same direction as another
 *
 * orients a vector to point away from a surface as defined by its normal
 *
 * @param[in] n      vector to orient.
 * @param[in] v      incident vector
 * @param[in] nref   reference vector
 * @returns oriented vector, pointing away from the surface.
 */
PLAY_CGLM_INLINE
vec3s
vec3s_(faceforward)(vec3s n, vec3s v, vec3s nref)
{
    vec3s dest;
    vec3_faceforward(n.raw, v.raw, nref.raw, dest.raw);
    return dest;
}

/*!
 * @brief reflection vector using an incident ray and a surface normal
 *
 * @param[in]  I    incident vector
 * @param[in]  N    normalized normal vector
 * @returns reflection result
 */
PLAY_CGLM_INLINE
vec3s
vec3s_(reflect)(vec3s v, vec3s n)
{
    vec3s dest;
    vec3_reflect(v.raw, n.raw, dest.raw);
    return dest;
}

/*!
 * @brief computes refraction vector for an incident vector and a surface normal.
 *
 * calculates the refraction vector based on Snell's law. If total internal reflection
 * occurs (angle too great given eta), dest is set to zero and returns false.
 * Otherwise, computes refraction vector, stores it in dest, and returns true.
 *
 * @param[in]  v    normalized incident vector
 * @param[in]  n    normalized normal vector
 * @param[in]  eta  ratio of indices of refraction (incident/transmitted)
 * @param[out] dest refraction vector if refraction occurs; zero vector otherwise
 *
 * @returns true if refraction occurs; false if total internal reflection occurs.
 */
PLAY_CGLM_INLINE
bool
vec3s_(refract)(vec3s v, vec3s n, float eta, vec3s * __restrict dest)
{
    return vec3_refract(v.raw, n.raw, eta, dest->raw);
}



/*** End of inlined file: vec3.h ***/


/*** Start of inlined file: vec4.h ***/
/*
 Macros:
   PLAY_CGLM_S_VEC4_ONE_INIT
   PLAY_CGLM_S_VEC4_BLACK_INIT
   PLAY_CGLM_S_VEC4_ZERO_INIT
   PLAY_CGLM_S_VEC4_ONE
   PLAY_CGLM_S_VEC4_BLACK
   PLAY_CGLM_S_VEC4_ZERO

 Functions:
   PLAY_CGLM_INLINE vec4s vec4_new(vec3s v3, float last);
   PLAY_CGLM_INLINE vec3s vec4s_copy3(vec4s v);
   PLAY_CGLM_INLINE vec4s vec4s_copy(vec4s v);
   PLAY_CGLM_INLINE vec4s vec4s_ucopy(vec4s v);
   PLAY_CGLM_INLINE void  vec4s_pack(vec4s dst[], vec4 src[], size_t len);
   PLAY_CGLM_INLINE void  vec4s_unpack(vec4 dst[], vec4s src[], size_t len);
   PLAY_CGLM_INLINE float vec4s_dot(vec4s a, vec4s b);
   PLAY_CGLM_INLINE float vec4s_norm2(vec4s v);
   PLAY_CGLM_INLINE float vec4s_norm(vec4s v);
   PLAY_CGLM_INLINE float vec4s_norm_one(vec4s v);
   PLAY_CGLM_INLINE float vec4s_norm_inf(vec4s v);
   PLAY_CGLM_INLINE vec4s vec4s_add(vec4s a, vec4s b);
   PLAY_CGLM_INLINE vec4s vec4s_adds(vec4s v, float s);
   PLAY_CGLM_INLINE vec4s vec4s_sub(vec4s a, vec4s b);
   PLAY_CGLM_INLINE vec4s vec4s_subs(vec4s v, float s);
   PLAY_CGLM_INLINE vec4s vec4s_mul(vec4s a, vec4s b);
   PLAY_CGLM_INLINE vec4s vec4s_scale(vec4s v, float s);
   PLAY_CGLM_INLINE vec4s vec4s_scale_as(vec4s v, float s);
   PLAY_CGLM_INLINE vec4s vec4s_div(vec4s a, vec4s b);
   PLAY_CGLM_INLINE vec4s vec4s_divs(vec4s v, float s);
   PLAY_CGLM_INLINE vec4s vec4s_addadd(vec4s a, vec4s b, vec4s dest);
   PLAY_CGLM_INLINE vec4s vec4s_subadd(vec4s a, vec4s b, vec4s dest);
   PLAY_CGLM_INLINE vec4s vec4s_muladd(vec4s a, vec4s b, vec4s dest);
   PLAY_CGLM_INLINE vec4s vec4s_muladds(vec4s a, float s, vec4s dest);
   PLAY_CGLM_INLINE vec4s vec4s_maxadd(vec4s a, vec4s b, vec4s dest);
   PLAY_CGLM_INLINE vec4s vec4s_minadd(vec4s a, vec4s b, vec4s dest);
   PLAY_CGLM_INLINE vec4s vec4s_subsub(vec4s a, vec4s b, vec4s dest);
   PLAY_CGLM_INLINE vec4s vec4s_addsub(vec4s a, vec4s b, vec4s dest);
   PLAY_CGLM_INLINE vec4s vec4s_mulsub(vec4s a, vec4s b, vec4s dest);
   PLAY_CGLM_INLINE vec4s vec4s_mulsubs(vec4s a, float s, vec4s dest);
   PLAY_CGLM_INLINE vec4s vec4s_maxsub(vec4s a, vec4s b, vec4s dest);
   PLAY_CGLM_INLINE vec4s vec4s_minsub(vec4s a, vec4s b, vec4s dest);
   PLAY_CGLM_INLINE vec4s vec4s_negate(vec4s v);
   PLAY_CGLM_INLINE vec4s vec4s_normalize(vec4s v);
   PLAY_CGLM_INLINE float vec4s_distance(vec4s a, vec4s b);
   PLAY_CGLM_INLINE float vec4s_distance2(vec4s a, vec4s b);
   PLAY_CGLM_INLINE vec4s vec4s_maxv(vec4s a, vec4s b);
   PLAY_CGLM_INLINE vec4s vec4s_minv(vec4s a, vec4s b);
   PLAY_CGLM_INLINE vec4s vec4s_clamp(vec4s v, float minVal, float maxVal);
   PLAY_CGLM_INLINE vec4s vec4s_lerp(vec4s from, vec4s to, float t);
   PLAY_CGLM_INLINE vec4s vec4s_lerpc(vec4s from, vec4s to, float t);
   PLAY_CGLM_INLINE vec4s vec4s_mix(vec4s from, vec4s to, float t);
   PLAY_CGLM_INLINE vec4s vec4s_mixc(vec4s from, vec4s to, float t);
   PLAY_CGLM_INLINE vec4s vec4s_step(vec4s edge, vec4s x);
   PLAY_CGLM_INLINE vec4s vec4s_smoothstep_uni(float edge0, float edge1, vec4s x);
   PLAY_CGLM_INLINE vec4s vec4s_smoothstep(vec4s edge0, vec4s edge1, vec4s x);
   PLAY_CGLM_INLINE vec4s vec4s_smoothinterp(vec4s from, vec4s to, float t);
   PLAY_CGLM_INLINE vec4s vec4s_smoothinterpc(vec4s from, vec4s to, float t);
   PLAY_CGLM_INLINE vec4s vec4s_cubic(float s);
   PLAY_CGLM_INLINE vec4s vec4s_swizzle(vec4s v, int mask);
   PLAY_CGLM_INLINE vec4s vec4s_make(float * restrict src);
   PLAY_CGLM_INLINE vec4s vec4s_reflect(vec4s v, vec4s n);
   PLAY_CGLM_INLINE bool  vec4s_refract(vec4s v, vec4s n, float eta, vec4s *dest)

 Deprecated:
   vec4s_step_uni  -->  use vec4s_steps
 */





/*** Start of inlined file: vec4-ext.h ***/
/*!
 * @brief SIMD like functions
 */

/*
 Functions:
   PLAY_CGLM_INLINE vec4s vec4s_broadcast(float val);
   PLAY_CGLM_INLINE vec4s vec4s_fill(float val);
   PLAY_CGLM_INLINE bool  vec4s_eq(vec4s v, float val);
   PLAY_CGLM_INLINE bool  vec4s_eq_eps(vec4s v, float val);
   PLAY_CGLM_INLINE bool  vec4s_eq_all(vec4s v);
   PLAY_CGLM_INLINE bool  vec4s_eqv(vec4s a, vec4s b);
   PLAY_CGLM_INLINE bool  vec4s_eqv_eps(vec4s a, vec4s b);
   PLAY_CGLM_INLINE float vec4s_max(vec4s v);
   PLAY_CGLM_INLINE float vec4s_min(vec4s v);
   PLAY_CGLM_INLINE bool  vec4s_isnan(vec4s v);
   PLAY_CGLM_INLINE bool  vec4s_isinf(vec4s v);
   PLAY_CGLM_INLINE bool  vec4s_isvalid(vec4s v);
   PLAY_CGLM_INLINE vec4s vec4s_sign(vec4s v);
   PLAY_CGLM_INLINE vec4s vec4s_abs(vec4s v);
   PLAY_CGLM_INLINE vec4s vec4s_fract(vec4s v);
   PLAY_CGLM_INLINE float vec4s_floor(vec4s v);
   PLAY_CGLM_INLINE float vec4s_mods(vec4s v, float s);
   PLAY_CGLM_INLINE float vec4s_steps(float edge, vec4s v);
   PLAY_CGLM_INLINE void  vec4s_stepr(vec4s edge, float v);
   PLAY_CGLM_INLINE float vec4s_hadd(vec4s v);
   PLAY_CGLM_INLINE vec4s vec4s_sqrt(vec4s v);
 */




/* api definition */
#define vec4s_(NAME) PLAY_CGLM_STRUCTAPI(vec4, NAME)

/*!
 * @brief fill a vector with specified value
 *
 * @param val value
 * @returns   dest
 */
PLAY_CGLM_INLINE
vec4s
vec4s_(broadcast)(float val)
{
    vec4s r;
    vec4_broadcast(val, r.raw);
    return r;
}

/*!
 * @brief fill a vector with specified value
 *
 * @param val value
 * @returns   dest
 */
PLAY_CGLM_INLINE
vec4s
vec4s_(fill)(float val)
{
    vec4s r;
    vec4_fill(r.raw, val);
    return r;
}

/*!
 * @brief check if vector is equal to value (without epsilon)
 *
 * @param v   vector
 * @param val value
 */
PLAY_CGLM_INLINE
bool
vec4s_(eq)(vec4s v, float val)
{
    return vec4_eq(v.raw, val);
}

/*!
 * @brief check if vector is equal to value (with epsilon)
 *
 * @param v   vector
 * @param val value
 */
PLAY_CGLM_INLINE
bool
vec4s_(eq_eps)(vec4s v, float val)
{
    return vec4_eq_eps(v.raw, val);
}

/*!
 * @brief check if vector members are equal (without epsilon)
 *
 * @param v   vector
 */
PLAY_CGLM_INLINE
bool
vec4s_(eq_all)(vec4s v)
{
    return vec4_eq_all(v.raw);
}

/*!
 * @brief check if vector is equal to another (without epsilon)
 *
 * @param a vector
 * @param b vector
 */
PLAY_CGLM_INLINE
bool
vec4s_(eqv)(vec4s a, vec4s b)
{
    return vec4_eqv(a.raw, b.raw);
}

/*!
 * @brief check if vector is equal to another (with epsilon)
 *
 * @param a vector
 * @param b vector
 */
PLAY_CGLM_INLINE
bool
vec4s_(eqv_eps)(vec4s a, vec4s b)
{
    return vec4_eqv_eps(a.raw, b.raw);
}

/*!
 * @brief max value of vector
 *
 * @param v vector
 */
PLAY_CGLM_INLINE
float
vec4s_(max)(vec4s v)
{
    return vec4_max(v.raw);
}

/*!
 * @brief min value of vector
 *
 * @param v vector
 */
PLAY_CGLM_INLINE
float
vec4s_(min)(vec4s v)
{
    return vec4_min(v.raw);
}

/*!
 * @brief check if one of items is NaN (not a number)
 *        you should only use this in DEBUG mode or very critical asserts
 *
 * @param[in] v vector
 */
PLAY_CGLM_INLINE
bool
vec4s_(isnan)(vec4s v)
{
    return vec4_isnan(v.raw);
}

/*!
 * @brief check if one of items is INFINITY
 *        you should only use this in DEBUG mode or very critical asserts
 *
 * @param[in] v vector
 */
PLAY_CGLM_INLINE
bool
vec4s_(isinf)(vec4s v)
{
    return vec4_isinf(v.raw);
}

/*!
 * @brief check if all items are valid number
 *        you should only use this in DEBUG mode or very critical asserts
 *
 * @param[in] v vector
 */
PLAY_CGLM_INLINE
bool
vec4s_(isvalid)(vec4s v)
{
    return vec4_isvalid(v.raw);
}

/*!
 * @brief get sign of 32 bit float as +1, -1, 0
 *
 * Important: It returns 0 for zero/NaN input
 *
 * @param   v   vector
 * @returns     sign vector
 */
PLAY_CGLM_INLINE
vec4s
vec4s_(sign)(vec4s v)
{
    vec4s r;
    vec4_sign(v.raw, r.raw);
    return r;
}

/*!
 * @brief absolute value of each vector item
 *
 * @param[in]  v    vector
 * @returns         destination vector
 */
PLAY_CGLM_INLINE
vec4s
vec4s_(abs)(vec4s v)
{
    vec4s r;
    vec4_abs(v.raw, r.raw);
    return r;
}

/*!
 * @brief fractional part of each vector item
 *
 * @param[in]  v    vector
 * @returns          dest destination vector
 */
PLAY_CGLM_INLINE
vec4s
vec4s_(fract)(vec4s v)
{
    vec4s r;
    vec4_fract(v.raw, r.raw);
    return r;
}

/*!
 * @brief floor of each vector item
 *
 * @param[in]  v    vector
 * @returns          dest destination vector
 */
PLAY_CGLM_INLINE
vec4s
vec4s_(floor)(vec4s v)
{
    vec4s r;
    vec4_floor(v.raw, r.raw);
    return r;
}

/*!
 * @brief mod of each vector item by scalar
 *
 * @param[in]  v    vector
 * @param[in]  s    scalar
 * @returns         destination vector
 */
PLAY_CGLM_INLINE
vec4s
vec4s_(mods)(vec4s v, float s)
{
    vec4s r;
    vec4_mods(v.raw, s, r.raw);
    return r;
}

/*!
 * @brief threshold each vector item with scalar
 *        condition is: (x[i] < edge) ? 0.0 : 1.0
 *
 * @param[in]   edge   threshold
 * @param[in]   x      vector to test against threshold
 * @returns            destination
 */
PLAY_CGLM_INLINE
vec4s
vec4s_(steps)(float edge, vec4s x)
{
    vec4s r;
    vec4_steps(edge, x.raw, r.raw);
    return r;
}

/*!
 * @brief threshold a value with *vector* as the threshold
 *        condition is: (x < edge[i]) ? 0.0 : 1.0
 *
 * @param[in]   edge   threshold vector
 * @param[in]   x      value to test against threshold
 * @returns            destination
 */
PLAY_CGLM_INLINE
vec4s
vec4s_(stepr)(vec4s edge, float x)
{
    vec4s r;
    vec4_stepr(edge.raw, x, r.raw);
    return r;
}

/*!
 * @brief vector reduction by summation
 * @warning could overflow
 *
 * @param[in]  v    vector
 * @return     sum of all vector's elements
 */
PLAY_CGLM_INLINE
float
vec4s_(hadd)(vec4s v)
{
    return vec4_hadd(v.raw);
}

/*!
 * @brief square root of each vector item
 *
 * @param[in]  v    vector
 * @returns         destination vector
 */
PLAY_CGLM_INLINE
vec4s
vec4s_(sqrt)(vec4s v)
{
    vec4s r;
    vec4_sqrt(v.raw, r.raw);
    return r;
}



/*** End of inlined file: vec4-ext.h ***/

/* DEPRECATED! */
#define vec4s_step_uni(edge, x) vec4s_steps(edge, x)

#define PLAY_CGLM_S_VEC4_ONE_INIT   {PLAY_CGLM_VEC4_ONE_INIT}
#define PLAY_CGLM_S_VEC4_BLACK_INIT {PLAY_CGLM_VEC4_BLACK_INIT}
#define PLAY_CGLM_S_VEC4_ZERO_INIT  {PLAY_CGLM_VEC4_ZERO_INIT}

#define PLAY_CGLM_S_VEC4_ONE        ((vec4s)PLAY_CGLM_VEC4_ONE_INIT)
#define PLAY_CGLM_S_VEC4_BLACK      ((vec4s)PLAY_CGLM_VEC4_BLACK_INIT)
#define PLAY_CGLM_S_VEC4_ZERO       ((vec4s)PLAY_CGLM_VEC4_ZERO_INIT)

/*!
 * @brief init vec4 using vec3
 *
 * @param[in]  v3   vector3
 * @param[in]  last last item
 * @returns         destination
 */
PLAY_CGLM_INLINE
vec4s
vec4_new(vec3s v3, float last)
{
    vec4s r;
    vec4_new(v3.raw, last, r.raw);
    return r;
}

/*!
 * @brief copy first 3 members of [a] to [dest]
 *
 * @param[in]  v    source
 * @returns         vec3
 */
PLAY_CGLM_INLINE
vec3s
vec4s_(copy3)(vec4s v)
{
    vec3s r;
    vec4_copy3(v.raw, r.raw);
    return r;
}

/*!
 * @brief copy all members of [a] to [dest]
 *
 * @param[in]  v    source
 * @returns         destination
 */
PLAY_CGLM_INLINE
vec4s
vec4s_(copy)(vec4s v)
{
    vec4s r;
    vec4_copy(v.raw, r.raw);
    return r;
}

/*!
 * @brief copy all members of [a] to [dest]
 *
 * alignment is not required
 *
 * @param[in]  v    source
 * @returns         destination
 */
PLAY_CGLM_INLINE
vec4s
vec4s_(ucopy)(vec4s v)
{
    vec4s r;
    vec4_ucopy(v.raw, r.raw);
    return r;
}

/*!
 * @brief pack an array of vec4 into an array of vec4s
 *
 * @param[out] dst array of vec4
 * @param[in]  src array of vec4s
 * @param[in]  len number of elements
 */
PLAY_CGLM_INLINE
void
vec4s_(pack)(vec4s dst[], vec4 src[], size_t len)
{
    size_t i;

    for (i = 0; i < len; i++)
    {
        vec4_copy(src[i], dst[i].raw);
    }
}

/*!
 * @brief unpack an array of vec4s into an array of vec4
 *
 * @param[out] dst array of vec4s
 * @param[in]  src array of vec4
 * @param[in]  len number of elements
 */
PLAY_CGLM_INLINE
void
vec4s_(unpack)(vec4 dst[], vec4s src[], size_t len)
{
    size_t i;

    for (i = 0; i < len; i++)
    {
        vec4_copy(src[i].raw, dst[i]);
    }
}

/*!
 * @brief make vector zero
 *
 * @returns      zero vector
 */
PLAY_CGLM_INLINE
vec4s
vec4s_(zero)(void)
{
    vec4s r;
    vec4_zero(r.raw);
    return r;
}

/*!
 * @brief make vector one
 *
 * @returns      one vector
 */
PLAY_CGLM_INLINE
vec4s
vec4s_(one)(void)
{
    vec4s r;
    vec4_one(r.raw);
    return r;
}

/*!
 * @brief vec4 dot product
 *
 * @param[in] a vector1
 * @param[in] b vector2
 *
 * @return dot product
 */
PLAY_CGLM_INLINE
float
vec4s_(dot)(vec4s a, vec4s b)
{
    return vec4_dot(a.raw, b.raw);
}

/*!
 * @brief norm * norm (magnitude) of vec
 *
 * we can use this func instead of calling norm * norm, because it would call
 * sqrtf function twice but with this func we can avoid func call, maybe this is
 * not good name for this func
 *
 * @param[in] v vec4
 *
 * @return norm * norm
 */
PLAY_CGLM_INLINE
float
vec4s_(norm2)(vec4s v)
{
    return vec4_norm2(v.raw);
}

/*!
 * @brief norm (magnitude) of vec4
 *
 * @param[in] v vector
 *
 * @return norm
 */
PLAY_CGLM_INLINE
float
vec4s_(norm)(vec4s v)
{
    return vec4_norm(v.raw);
}

/*!
 * @brief L1 norm of vec4
 * Also known as Manhattan Distance or Taxicab norm.
 * L1 Norm is the sum of the magnitudes of the vectors in a space.
 * It is calculated as the sum of the absolute values of the vector components.
 * In this norm, all the components of the vector are weighted equally.
 *
 * This computes:
 * R = |v[0]| + |v[1]| + |v[2]| + |v[3]|
 *
 * @param[in] v vector
 *
 * @return L1 norm
 */
PLAY_CGLM_INLINE
float
vec4s_(norm_one)(vec4s v)
{
    return vec4_norm_one(v.raw);
}

/*!
 * @brief Infinity norm of vec4
 * Also known as Maximum norm.
 * Infinity Norm is the largest magnitude among each element of a vector.
 * It is calculated as the maximum of the absolute values of the vector components.
 *
 * This computes:
 * inf norm = fmax(|v[0]|, |v[1]|, |v[2]|, |v[3]|)
 *
 * @param[in] v vector
 *
 * @return Infinity norm
 */
PLAY_CGLM_INLINE
float
vec4s_(norm_inf)(vec4s v)
{
    return vec4_norm_inf(v.raw);
}

/*!
 * @brief add b vector to a vector store result in dest
 *
 * @param[in]  a    vector1
 * @param[in]  b    vector2
 * @returns         destination vector
 */
PLAY_CGLM_INLINE
vec4s
vec4s_(add)(vec4s a, vec4s b)
{
    vec4s r;
    vec4_add(a.raw, b.raw, r.raw);
    return r;
}

/*!
 * @brief add scalar to v vector store result in dest (d = v + vec(s))
 *
 * @param[in]  v    vector
 * @param[in]  s    scalar
 * @returns         destination vector
 */
PLAY_CGLM_INLINE
vec4s
vec4s_(adds)(vec4s v, float s)
{
    vec4s r;
    vec4_adds(v.raw, s, r.raw);
    return r;
}

/*!
 * @brief subtract b vector from a vector store result in dest (d = a - b)
 *
 * @param[in]  a    vector1
 * @param[in]  b    vector2
 * @returns         destination vector
 */
PLAY_CGLM_INLINE
vec4s
vec4s_(sub)(vec4s a, vec4s b)
{
    vec4s r;
    vec4_sub(a.raw, b.raw, r.raw);
    return r;
}

/*!
 * @brief subtract scalar from v vector store result in dest (d = v - vec(s))
 *
 * @param[in]  v    vector
 * @param[in]  s    scalar
 * @returns         destination vector
 */
PLAY_CGLM_INLINE
vec4s
vec4s_(subs)(vec4s v, float s)
{
    vec4s r;
    vec4_subs(v.raw, s, r.raw);
    return r;
}

/*!
 * @brief multiply two vectors (component-wise multiplication)
 *
 * @param a    vector1
 * @param b    vector2
 * @returns    dest = (a[0] * b[0], a[1] * b[1], a[2] * b[2], a[3] * b[3])
 */
PLAY_CGLM_INLINE
vec4s
vec4s_(mul)(vec4s a, vec4s b)
{
    vec4s r;
    vec4_mul(a.raw, b.raw, r.raw);
    return r;
}

/*!
 * @brief multiply/scale vec4 vector with scalar: result = v * s
 *
 * @param[in]  v    vector
 * @param[in]  s    scalar
 * @returns         destination vector
 */
PLAY_CGLM_INLINE
vec4s
vec4s_(scale)(vec4s v, float s)
{
    vec4s r;
    vec4_scale(v.raw, s, r.raw);
    return r;
}

/*!
 * @brief make vec4 vector scale as specified: result = unit(v) * s
 *
 * @param[in]  v    vector
 * @param[in]  s    scalar
 * @returns         destination vector
 */
PLAY_CGLM_INLINE
vec4s
vec4s_(scale_as)(vec4s v, float s)
{
    vec4s r;
    vec4_scale_as(v.raw, s, r.raw);
    return r;
}

/*!
 * @brief div vector with another component-wise division: d = a / b
 *
 * @param[in]  a    vector 1
 * @param[in]  b    vector 2
 * @returns         result = (a[0]/b[0], a[1]/b[1], a[2]/b[2], a[3]/b[3])
 */
PLAY_CGLM_INLINE
vec4s
vec4s_(div)(vec4s a, vec4s b)
{
    vec4s r;
    vec4_div(a.raw, b.raw, r.raw);
    return r;
}

/*!
 * @brief div vec4 vector with scalar: d = v / s
 *
 * @param[in]  v    vector
 * @param[in]  s    scalar
 * @returns         destination vector
 */
PLAY_CGLM_INLINE
vec4s
vec4s_(divs)(vec4s v, float s)
{
    vec4s r;
    vec4_divs(v.raw, s, r.raw);
    return r;
}

/*!
 * @brief add two vectors and add result to sum
 *
 * it applies += operator so dest must be initialized
 *
 * @param[in]  a    vector 1
 * @param[in]  b    vector 2
 * @returns         dest += (a + b)
 */
PLAY_CGLM_INLINE
vec4s
vec4s_(addadd)(vec4s a, vec4s b, vec4s dest)
{
    vec4_addadd(a.raw, b.raw, dest.raw);
    return dest;
}

/*!
 * @brief sub two vectors and add result to dest
 *
 * it applies += operator so dest must be initialized
 *
 * @param[in]  a    vector 1
 * @param[in]  b    vector 2
 * @returns         dest += (a - b)
 */
PLAY_CGLM_INLINE
vec4s
vec4s_(subadd)(vec4s a, vec4s b, vec4s dest)
{
    vec4_subadd(a.raw, b.raw, dest.raw);
    return dest;
}

/*!
 * @brief mul two vectors and add result to dest
 *
 * it applies += operator so dest must be initialized
 *
 * @param[in]  a    vector 1
 * @param[in]  b    vector 2
 * @returns         dest += (a * b)
 */
PLAY_CGLM_INLINE
vec4s
vec4s_(muladd)(vec4s a, vec4s b, vec4s dest)
{
    vec4_muladd(a.raw, b.raw, dest.raw);
    return dest;
}

/*!
 * @brief mul vector with scalar and add result to sum
 *
 * it applies += operator so dest must be initialized
 *
 * @param[in]  a    vector
 * @param[in]  s    scalar
 * @returns         dest += (a * b)
 */
PLAY_CGLM_INLINE
vec4s
vec4s_(muladds)(vec4s a, float s, vec4s dest)
{
    vec4_muladds(a.raw, s, dest.raw);
    return dest;
}

/*!
 * @brief add max of two vectors to result/dest
 *
 * it applies += operator so dest must be initialized
 *
 * @param[in]  a    vector 1
 * @param[in]  b    vector 2
 * @returns         dest += fmax(a, b)
 */
PLAY_CGLM_INLINE
vec4s
vec4s_(maxadd)(vec4s a, vec4s b, vec4s dest)
{
    vec4_maxadd(a.raw, b.raw, dest.raw);
    return dest;
}

/*!
 * @brief add min of two vectors to result/dest
 *
 * it applies += operator so dest must be initialized
 *
 * @param[in]  a    vector 1
 * @param[in]  b    vector 2
 * @returns         dest += fmin(a, b)
 */
PLAY_CGLM_INLINE
vec4s
vec4s_(minadd)(vec4s a, vec4s b, vec4s dest)
{
    vec4_minadd(a.raw, b.raw, dest.raw);
    return dest;
}

/*!
 * @brief sub two vectors and sub result to dest
 *
 * it applies -= operator so dest must be initialized
 *
 * @param[in]  a    vector 1
 * @param[in]  b    vector 2
 * @returns         dest -= (a + b)
 */
PLAY_CGLM_INLINE
vec4s
vec4s_(subsub)(vec4s a, vec4s b, vec4s dest)
{
    vec4_subsub(a.raw, b.raw, dest.raw);
    return dest;
}

/*!
 * @brief add two vectors and sub result to dest
 *
 * it applies -= operator so dest must be initialized
 *
 * @param[in]  a    vector 1
 * @param[in]  b    vector 2
 * @returns         dest -= (a + b)
 */
PLAY_CGLM_INLINE
vec4s
vec4s_(addsub)(vec4s a, vec4s b, vec4s dest)
{
    vec4_addsub(a.raw, b.raw, dest.raw);
    return dest;
}

/*!
 * @brief mul two vectors and sub result to dest
 *
 * it applies -= operator so dest must be initialized
 *
 * @param[in]  a    vector 1
 * @param[in]  b    vector 2
 * @returns         dest -= (a * b)
 */
PLAY_CGLM_INLINE
vec4s
vec4s_(mulsub)(vec4s a, vec4s b, vec4s dest)
{
    vec4_mulsub(a.raw, b.raw, dest.raw);
    return dest;
}

/*!
 * @brief mul vector with scalar and sub result to dest
 *
 * it applies -= operator so dest must be initialized
 *
 * @param[in]  a    vector
 * @param[in]  s    scalar
 * @returns         dest -= (a * b)
 */
PLAY_CGLM_INLINE
vec4s
vec4s_(mulsubs)(vec4s a, float s, vec4s dest)
{
    vec4_mulsubs(a.raw, s, dest.raw);
    return dest;
}

/*!
 * @brief sub max of two vectors to dest
 *
 * it applies -= operator so dest must be initialized
 *
 * @param[in]  a    vector 1
 * @param[in]  b    vector 2
 * @returns         dest -= fmax(a, b)
 */
PLAY_CGLM_INLINE
vec4s
vec4s_(maxsub)(vec4s a, vec4s b, vec4s dest)
{
    vec4_maxsub(a.raw, b.raw, dest.raw);
    return dest;
}

/*!
 * @brief sub min of two vectors to dest
 *
 * it applies -= operator so dest must be initialized
 *
 * @param[in]  a    vector 1
 * @param[in]  b    vector 2
 * @returns         dest -= fmin(a, b)
 */
PLAY_CGLM_INLINE
vec4s
vec4s_(minsub)(vec4s a, vec4s b, vec4s dest)
{
    vec4_minsub(a.raw, b.raw, dest.raw);
    return dest;
}

/*!
 * @brief negate vector components and store result in dest
 *
 * @param[in]  v     vector
 * @returns          result vector
 */
PLAY_CGLM_INLINE
vec4s
vec4s_(negate)(vec4s v)
{
    vec4_negate(v.raw);
    return v;
}

/*!
 * @brief normalize vec4 and store result in same vec
 *
 * @param[in] v   vector
 * @returns       normalized vector
 */
PLAY_CGLM_INLINE
vec4s
vec4s_(normalize)(vec4s v)
{
    vec4_normalize(v.raw);
    return v;
}

/**
 * @brief distance between two vectors
 *
 * @param[in] a vector1
 * @param[in] b vector2
 * @return returns distance
 */
PLAY_CGLM_INLINE
float
vec4s_(distance)(vec4s a, vec4s b)
{
    return vec4_distance(a.raw, b.raw);
}

/**
 * @brief squared distance between two vectors
 *
 * @param[in] a vector1
 * @param[in] b vector2
 * @return returns squared distance
 */
PLAY_CGLM_INLINE
float
vec4s_(distance2)(vec4s a, vec4s b)
{
    return vec4_distance2(a.raw, b.raw);
}

/*!
 * @brief max values of vectors
 *
 * @param[in]  a    vector1
 * @param[in]  b    vector2
 * @returns         destination
 */
PLAY_CGLM_INLINE
vec4s
vec4s_(maxv)(vec4s a, vec4s b)
{
    vec4s r;
    vec4_maxv(a.raw, b.raw, r.raw);
    return r;
}

/*!
 * @brief min values of vectors
 *
 * @param[in]  a    vector1
 * @param[in]  b    vector2
 * @returns         destination
 */
PLAY_CGLM_INLINE
vec4s
vec4s_(minv)(vec4s a, vec4s b)
{
    vec4s r;
    vec4_minv(a.raw, b.raw, r.raw);
    return r;
}

/*!
 * @brief clamp vector's individual members between min and max values
 *
 * @param[in]       v       vector
 * @param[in]       minVal  minimum value
 * @param[in]       maxVal  maximum value
 * @returns                 clamped vector
 */
PLAY_CGLM_INLINE
vec4s
vec4s_(clamp)(vec4s v, float minVal, float maxVal)
{
    vec4_clamp(v.raw, minVal, maxVal);
    return v;
}

/*!
 * @brief linear interpolation between two vectors
 *
 * formula:  from + s * (to - from)
 *
 * @param[in]   from  from value
 * @param[in]   to    to value
 * @param[in]   t     interpolant (amount)
 * @returns           destination
 */
PLAY_CGLM_INLINE
vec4s
vec4s_(lerp)(vec4s from, vec4s to, float t)
{
    vec4s r;
    vec4_lerp(from.raw, to.raw, t, r.raw);
    return r;
}

/*!
 * @brief linear interpolation between two vectors (clamped)
 *
 * formula:  from + s * (to - from)
 *
 * @param[in]   from  from value
 * @param[in]   to    to value
 * @param[in]   t     interpolant (amount) clamped between 0 and 1
 * @returns           destination
 */
PLAY_CGLM_INLINE
vec4s
vec4s_(lerpc)(vec4s from, vec4s to, float t)
{
    vec4s r;
    vec4_lerpc(from.raw, to.raw, t, r.raw);
    return r;
}

/*!
 * @brief linear interpolation between two vectors
 *
 * formula:  from + s * (to - from)
 *
 * @param[in]   from  from value
 * @param[in]   to    to value
 * @param[in]   t     interpolant (amount)
 * @returns           destination
 */
PLAY_CGLM_INLINE
vec4s
vec4s_(mix)(vec4s from, vec4s to, float t)
{
    vec4s r;
    vec4_mix(from.raw, to.raw, t, r.raw);
    return r;
}

/*!
 * @brief linear interpolation between two vectors (clamped)
 *
 * formula:  from + s * (to - from)
 *
 * @param[in]   from  from value
 * @param[in]   to    to value
 * @param[in]   t     interpolant (amount) clamped between 0 and 1
 * @returns           destination
 */
PLAY_CGLM_INLINE
vec4s
vec4s_(mixc)(vec4s from, vec4s to, float t)
{
    vec4s r;
    vec4_mixc(from.raw, to.raw, t, r.raw);
    return r;
}

/*!
 * @brief threshold function
 *
 * @param[in]   edge    threshold
 * @param[in]   x       value to test against threshold
 * @returns             0.0 if x < edge, else 1.0
 */
PLAY_CGLM_INLINE
vec4s
vec4s_(step)(vec4s edge, vec4s x)
{
    vec4s r;
    vec4_step(edge.raw, x.raw, r.raw);
    return r;
}

/*!
 * @brief threshold function with a smooth transition (unidimensional)
 *
 * @param[in]   edge0   low threshold
 * @param[in]   edge1   high threshold
 * @param[in]   x       value to test against threshold
 * @returns             destination
 */
PLAY_CGLM_INLINE
vec4s
vec4s_(smoothstep_uni)(float edge0, float edge1, vec4s x)
{
    vec4s r;
    vec4_smoothstep_uni(edge0, edge1, x.raw, r.raw);
    return r;
}

/*!
 * @brief threshold function with a smooth transition
 *
 * @param[in]   edge0   low threshold
 * @param[in]   edge1   high threshold
 * @param[in]   x       value to test against threshold
 * @returns             destination
 */
PLAY_CGLM_INLINE
vec4s
vec4s_(smoothstep)(vec4s edge0, vec4s edge1, vec4s x)
{
    vec4s r;
    vec4_smoothstep(edge0.raw, edge1.raw, x.raw, r.raw);
    return r;
}

/*!
 * @brief smooth Hermite interpolation between two vectors
 *
 * formula:  from + s * (to - from)
 *
 * @param[in]   from    from value
 * @param[in]   to      to value
 * @param[in]   t       interpolant (amount)
 * @returns             destination
 */
PLAY_CGLM_INLINE
vec4s
vec4s_(smoothinterp)(vec4s from, vec4s to, float t)
{
    vec4s r;
    vec4_smoothinterp(from.raw, to.raw, t, r.raw);
    return r;
}

/*!
 * @brief smooth Hermite interpolation between two vectors (clamped)
 *
 * formula:  from + s * (to - from)
 *
 * @param[in]   from    from value
 * @param[in]   to      to value
 * @param[in]   t       interpolant (amount) clamped between 0 and 1
 * @returns             destination
 */
PLAY_CGLM_INLINE
vec4s
vec4s_(smoothinterpc)(vec4s from, vec4s to, float t)
{
    vec4s r;
    vec4_smoothinterpc(from.raw, to.raw, t, r.raw);
    return r;
}

/*!
 * @brief helper to fill vec4 as [S^3, S^2, S, 1]
 *
 * @param[in]   s     parameter
 * @returns           destination
 */
PLAY_CGLM_INLINE
vec4s
vec4s_(cubic)(float s)
{
    vec4s r;
    vec4_cubic(s, r.raw);
    return r;
}

/*!
 * @brief swizzle vector components
 *
 * you can use existing masks e.g. PLAY_CGLM_XXXX, PLAY_CGLM_WZYX
 *
 * @param[in]  v    source
 * @param[in]  mask mask
 * @returns swizzled vector
 */
PLAY_CGLM_INLINE
vec4s
vec4s_(swizzle)(vec4s v, int mask)
{
    vec4s dest;
    vec4_swizzle(v.raw, mask, dest.raw);
    return dest;
}

/*!
 * @brief Create four dimensional vector from pointer
 *
 * @param[in]  src  pointer to an array of floats
 * @returns constructed 4D vector from raw pointer
 */
PLAY_CGLM_INLINE
vec4s
vec4s_(make)(const float * __restrict src)
{
    vec4s dest;
    vec4_make(src, dest.raw);
    return dest;
}

/*!
 * @brief reflection vector using an incident ray and a surface normal
 *
 * @param[in]  v    incident vector
 * @param[in]  n    normalized normal vector
 * @returns reflection result
 */
PLAY_CGLM_INLINE
vec4s
vec4s_(reflect)(vec4s v, vec4s n)
{
    vec4s dest;
    vec4_reflect(v.raw, n.raw, dest.raw);
    return dest;
}

/*!
 * @brief computes refraction vector for an incident vector and a surface normal.
 *
 * calculates the refraction vector based on Snell's law. If total internal reflection
 * occurs (angle too great given eta), dest is set to zero and returns false.
 * Otherwise, computes refraction vector, stores it in dest, and returns true.
 *
 * this implementation does not explicitly preserve the 'w' component of the
 * incident vector 'I' in the output 'dest', users requiring the preservation of
 * the 'w' component should manually adjust 'dest' after calling this function.
 *
 * @param[in]  v    normalized incident vector
 * @param[in]  n    normalized normal vector
 * @param[in]  eta  ratio of indices of refraction (incident/transmitted)
 * @param[out] dest refraction vector if refraction occurs; zero vector otherwise
 *
 * @returns true if refraction occurs; false if total internal reflection occurs.
 */
PLAY_CGLM_INLINE
bool
vec4s_(refract)(vec4s v, vec4s n, float eta, vec4s * __restrict dest)
{
    return vec4_refract(v.raw, n.raw, eta, dest->raw);
}



/*** End of inlined file: vec4.h ***/


/*** Start of inlined file: ivec2.h ***/
/*
 Macros:
  PLAY_CGLM_S_IVEC2_ONE_INIT
  PLAY_CGLM_S_IVEC2_ZERO_INIT
  PLAY_CGLM_S_IVEC2_ONE
  PLAY_CGLM_S_IVEC2_ZERO

 Functions:
  PLAY_CGLM_INLINE ivec2s ivec2_new(int * __restrict v)
  PLAY_CGLM_INLINE void ivec2s_pack(ivec2s dst[], ivec2s src[], size_t len)
  PLAY_CGLM_INLINE void ivec2s_unpack(ivec2 dst[], ivec2 src[], size_t len)
  PLAY_CGLM_INLINE ivec2s ivec2s_zero(ivec2s v)
  PLAY_CGLM_INLINE ivec2s ivec2s_one(ivec2s v)
  PLAY_CGLM_INLINE int ivec2s_dot(ivec2s a, ivec2s b)
  PLAY_CGLM_INLINE int ivec2s_cross(ivec2s a, ivec2s b)
  PLAY_CGLM_INLINE ivec2s ivec2s_add(ivec2s a, ivec2s b)
  PLAY_CGLM_INLINE ivec2s ivec2s_adds(ivec2s v, int s)
  PLAY_CGLM_INLINE ivec2s ivec2s_sub(ivec2s a, ivec2s b)
  PLAY_CGLM_INLINE ivec2s ivec2s_subs(ivec2s v, int s)
  PLAY_CGLM_INLINE ivec2s ivec2s_mul(ivec2s a, ivec2s b)
  PLAY_CGLM_INLINE ivec2s ivec2s_scale(ivec2s v, int s)
  PLAY_CGLM_INLINE ivec2s ivec2s_div(ivec2s a, ivec2s b)
  PLAY_CGLM_INLINE ivec2s ivec2s_divs(ivec2s v, int s)
  PLAY_CGLM_INLINE ivec2s ivec2s_mod(ivec2s a, ivec2s b)
  PLAY_CGLM_INLINE ivec2s ivec2s_addadd(ivec2s a, ivec2s b)
  PLAY_CGLM_INLINE ivec2s ivec2s_addadds(ivec2s a, int s)
  PLAY_CGLM_INLINE ivec2s ivec2s_subadd(ivec2s a, ivec2s b)
  PLAY_CGLM_INLINE ivec2s ivec2s_subadds(ivec2s a, int s)
  PLAY_CGLM_INLINE ivec2s ivec2s_muladd(ivec2s a, ivec2s b)
  PLAY_CGLM_INLINE ivec2s ivec2s_muladds(ivec2s a, int s)
  PLAY_CGLM_INLINE ivec2s ivec2s_maxadd(ivec2s a, ivec2s b)
  PLAY_CGLM_INLINE ivec2s ivec2s_minadd(ivec2s a, ivec2s b)
  PLAY_CGLM_INLINE ivec2s ivec2s_subsub(ivec2s a, ivec2s b)
  PLAY_CGLM_INLINE ivec2s ivec2s_subsubs(ivec2s a, int s)
  PLAY_CGLM_INLINE ivec2s ivec2s_addsub(ivec2s a, ivec2s b)
  PLAY_CGLM_INLINE ivec2s ivec2s_addsubs(ivec2s a, int s)
  PLAY_CGLM_INLINE ivec2s ivec2s_mulsub(ivec2s a, ivec2s b)
  PLAY_CGLM_INLINE ivec2s ivec2s_mulsubs(ivec2s a, int s)
  PLAY_CGLM_INLINE ivec2s ivec2s_maxsub(ivec2s a, ivec2s b)
  PLAY_CGLM_INLINE ivec2s ivec2s_minsub(ivec2s a, ivec2s b)
  PLAY_CGLM_INLINE int ivec2s_distance2(ivec2s a, ivec2s b)
  PLAY_CGLM_INLINE float ivec2s_distance(ivec2s a, ivec2s b)
  PLAY_CGLM_INLINE ivec2s ivec2s_fill(int val)
  PLAY_CGLM_INLINE bool ivec2s_eq(ivec2s v, int val);
  PLAY_CGLM_INLINE bool ivec2s_eqv(ivec2s a, ivec2s b);
  PLAY_CGLM_INLINE ivec2s ivec2s_maxv(ivec2s a, ivec2s b)
  PLAY_CGLM_INLINE ivec2s ivec2s_minv(ivec2s a, ivec2s b)
  PLAY_CGLM_INLINE ivec2s ivec2s_clamp(ivec2s v, int minVal, int maxVal)
  PLAY_CGLM_INLINE ivec2s ivec2s_abs(ivec2s v)
 */

#ifndef civec2s_h
#define civec2s_h

#define ivec2s_(NAME) PLAY_CGLM_STRUCTAPI(ivec2, NAME)

#define PLAY_CGLM_S_IVEC2_ONE_INIT   {PLAY_CGLM_IVEC2_ONE_INIT}
#define PLAY_CGLM_S_IVEC2_ZERO_INIT  {PLAY_CGLM_IVEC2_ZERO_INIT}

#define PLAY_CGLM_S_IVEC2_ONE  ((ivec2s)PLAY_CGLM_S_IVEC2_ONE_INIT)
#define PLAY_CGLM_S_IVEC2_ZERO ((ivec2s)PLAY_CGLM_S_IVEC2_ZERO_INIT)

/*!
 * @brief init ivec2 using ivec3 or ivec4
 *
 * @param[in]  v    vector
 * @returns         destination
 */
PLAY_CGLM_INLINE
ivec2s
ivec2_new(int * __restrict v)
{
    ivec2s r;
    ivec2_new(v, r.raw);
    return r;
}

/*!
 * @brief pack an array of ivec2 into an array of ivec2s
 *
 * @param[out] dst array of ivec2s
 * @param[in]  src array of ivec2
 * @param[in]  len number of elements
 */
PLAY_CGLM_INLINE
void
ivec2s_(pack)(ivec2s dst[], ivec2 src[], size_t len)
{
    size_t i;

    for (i = 0; i < len; i++)
    {
        ivec2_copy(src[i], dst[i].raw);
    }
}

/*!
 * @brief unpack an array of ivec2s into an array of ivec2
 *
 * @param[out] dst array of ivec2
 * @param[in]  src array of ivec2s
 * @param[in]  len number of elements
 */
PLAY_CGLM_INLINE
void
ivec2s_(unpack)(ivec2 dst[], ivec2s src[], size_t len)
{
    size_t i;

    for (i = 0; i < len; i++)
    {
        ivec2_copy(src[i].raw, dst[i]);
    }
}

/*!
 * @brief set all members of [v] to zero
 *
 * @returns vector
 */
PLAY_CGLM_INLINE
ivec2s
ivec2s_(zero)(void)
{
    ivec2s r;
    ivec2_zero(r.raw);
    return r;
}

/*!
 * @brief set all members of [v] to one
 *
 * @returns vector
 */
PLAY_CGLM_INLINE
ivec2s
ivec2s_(one)(void)
{
    ivec2s r;
    ivec2_one(r.raw);
    return r;
}

/*!
 * @brief ivec2 dot product
 *
 * @param[in] a vector1
 * @param[in] b vector2
 *
 * @return dot product
 */
PLAY_CGLM_INLINE
int
ivec2s_(dot)(ivec2s a, ivec2s b)
{
    return ivec2_dot(a.raw, b.raw);
}

/*!
 * @brief ivec2 cross product
 *
 * REF: http://allenchou.net/2013/07/cross-product-of-2d-vectors/
 *
 * @param[in]  a vector1
 * @param[in]  b vector2
 *
 * @return Z component of cross product
 */
PLAY_CGLM_INLINE
int
ivec2s_(cross)(ivec2s a, ivec2s b)
{
    return ivec2_cross(a.raw, b.raw);
}

/*!
 * @brief add vector [a] to vector [b] and store result in [dest]
 *
 * @param[in]  a    first vector
 * @param[in]  b    second vector
 * @returns         destination
 */
PLAY_CGLM_INLINE
ivec2s
ivec2s_(add)(ivec2s a, ivec2s b)
{
    ivec2s r;
    ivec2_add(a.raw, b.raw, r.raw);
    return r;
}

/*!
 * @brief add scalar s to vector [v] and store result in [dest]
 *
 * @param[in]  v    vector
 * @param[in]  s    scalar
 * @returns         destination
 */
PLAY_CGLM_INLINE
ivec2s
ivec2s_(adds)(ivec2s v, int s)
{
    ivec2s r;
    ivec2_adds(v.raw, s, r.raw);
    return r;
}

/*!
 * @brief subtract vector [b] from vector [a] and store result in [dest]
 *
 * @param[in]  a    first vector
 * @param[in]  b    second vector
 * @returns         destination
 */
PLAY_CGLM_INLINE
ivec2s
ivec2s_(sub)(ivec2s a, ivec2s b)
{
    ivec2s r;
    ivec2_sub(a.raw, b.raw, r.raw);
    return r;
}

/*!
 * @brief subtract scalar s from vector [v] and store result in [dest]
 *
 * @param[in]  v    vector
 * @param[in]  s    scalar
 * @returns         destination
 */
PLAY_CGLM_INLINE
ivec2s
ivec2s_(subs)(ivec2s v, int s)
{
    ivec2s r;
    ivec2_subs(v.raw, s, r.raw);
    return r;
}

/*!
 * @brief multiply vector [a] with vector [b] and store result in [dest]
 *
 * @param[in]  a    first vector
 * @param[in]  b    second vector
 * @returns         destination
 */
PLAY_CGLM_INLINE
ivec2s
ivec2s_(mul)(ivec2s a, ivec2s b)
{
    ivec2s r;
    ivec2_mul(a.raw, b.raw, r.raw);
    return r;
}

/*!
 * @brief multiply vector [a] with scalar s and store result in [dest]
 *
 * @param[in]  v    vector
 * @param[in]  s    scalar
 * @returns         destination
 */
PLAY_CGLM_INLINE
ivec2s
ivec2s_(scale)(ivec2s v, int s)
{
    ivec2s r;
    ivec2_scale(v.raw, s, r.raw);
    return r;
}

/*!
 * @brief div vector with another component-wise division: d = a / b
 *
 * @param[in]  a    vector 1
 * @param[in]  b    vector 2
 * @returns         result = (a[0]/b[0], a[1]/b[1])
 */
PLAY_CGLM_INLINE
ivec2s
ivec2s_(div)(ivec2s a, ivec2s b)
{
    ivec2s r;
    ivec2_div(a.raw, b.raw, r.raw);
    return r;
}

/*!
 * @brief div vector with scalar: d = v / s
 *
 * @param[in]  v    vector
 * @param[in]  s    scalar
 * @returns         result = (a[0]/s, a[1]/s)
 */
PLAY_CGLM_INLINE
ivec2s
ivec2s_(divs)(ivec2s v, int s)
{
    ivec2s r;
    ivec2_divs(v.raw, s, r.raw);
    return r;
}

/*!
 * @brief mod vector with another component-wise modulo: d = a % b
 *
 * @param[in]  a    vector 1
 * @param[in]  b    vector 2
 * @returns         result = (a[0]%b[0], a[1]%b[1])
 */
PLAY_CGLM_INLINE
ivec2s
ivec2s_(mod)(ivec2s a, ivec2s b)
{
    ivec2s r;
    ivec2_mod(a.raw, b.raw, r.raw);
    return r;
}

/*!
 * @brief add vector [a] with vector [b] and add result to vector [dest]
 *
 * applies += operator so dest must be initialized
 *
 * @param[in]  a    first vector
 * @param[in]  b    second vector
 * @param[in]  dest dest += (a + b)
 * @returns         dest
 */
PLAY_CGLM_INLINE
ivec2s
ivec2s_(addadd)(ivec2s a, ivec2s b, ivec2s dest)
{
    ivec2_addadd(a.raw, b.raw, dest.raw);
    return dest;
}

/*!
 * @brief add scalar [s] onto vector [a] and add result to vector [dest]
 *
 * applies += operator so dest must be initialized
 *
 * @param[in]  a    vector
 * @param[in]  s    scalar
 * @param[in]  dest dest += (a + s)
 * @returns         dest
 */
PLAY_CGLM_INLINE
ivec2s
ivec2s_(addadds)(ivec2s a, int s, ivec2s dest)
{
    ivec2_addadds(a.raw, s, dest.raw);
    return dest;
}

/*!
 * @brief subtract vector [a] from vector [b] and add result to [dest]
 *
 * applies += operator so dest must be initialized
 *
 * @param[in]  a    first vector
 * @param[in]  b    second vector
 * @param[in]  dest dest += (a - b)
 * @returns         dest
 */
PLAY_CGLM_INLINE
ivec2s
ivec2s_(subadd)(ivec2s a, ivec2s b, ivec2s dest)
{
    ivec2_subadd(a.raw, b.raw, dest.raw);
    return dest;
}

/*!
 * @brief subtract scalar [s] from vector [a] and add result to [dest]
 *
 * applies += operator so dest must be initialized
 *
 * @param[in]  a    first
 * @param[in]  s    scalar
 * @param[in]  dest dest += (a - s)
 * @returns         dest
 */
PLAY_CGLM_INLINE
ivec2s
ivec2s_(subadds)(ivec2s a, int s, ivec2s dest)
{
    ivec2_subadds(a.raw, s, dest.raw);
    return dest;
}

/*!
 * @brief multiply vector [a] with vector [b] and add result to [dest]
 *
 * applies += operator so dest must be initialized
 *
 * @param[in]  a    first vector
 * @param[in]  b    second vector
 * @param[in]  dest dest += (a * b)
 * @returns         dest
 */
PLAY_CGLM_INLINE
ivec2s
ivec2s_(muladd)(ivec2s a, ivec2s b, ivec2s dest)
{
    ivec2_muladd(a.raw, b.raw, dest.raw);
    return dest;
}

/*!
 * @brief multiply vector [a] with scalar [s] and add result to [dest]
 *
 * applies += operator so dest must be initialized
 *
 * @param[in]  a    vector
 * @param[in]  s    scalar
 * @param[in]  dest dest += (a * s)
 * @returns         dest
 */
PLAY_CGLM_INLINE
ivec2s
ivec2s_(muladds)(ivec2s a, int s, ivec2s dest)
{
    ivec2_muladds(a.raw, s, dest.raw);
    return dest;
}

/*!
 * @brief add maximum of vector [a] and vector [b] to vector [dest]
 *
 * applies += operator so dest must be initialized
 *
 * @param[in]  a    first vector
 * @param[in]  b    second vector
 * @param[in]  dest dest += fmax(a, b)
 * @returns         dest
 */
PLAY_CGLM_INLINE
ivec2s
ivec2s_(maxadd)(ivec2s a, ivec2s b, ivec2s dest)
{
    ivec2_maxadd(a.raw, b.raw, dest.raw);
    return dest;
}

/*!
 * @brief add minimum of vector [a] and vector [b] to vector [dest]
 *
 * applies += operator so dest must be initialized
 *
 * @param[in]  a    first vector
 * @param[in]  b    second vector
 * @param[in]  dest dest += fmin(a, b)
 * @returns         dest
 */
PLAY_CGLM_INLINE
ivec2s
ivec2s_(minadd)(ivec2s a, ivec2s b, ivec2s dest)
{
    ivec2_minadd(a.raw, b.raw, dest.raw);
    return dest;
}

/*!
 * @brief subtract vector [a] from vector [b] and subtract result from [dest]
 *
 * applies -= operator so dest must be initialized
 *
 * @param[in]  a    first vector
 * @param[in]  b    second vector
 * @param[in]  dest dest -= (a - b)
 * @returns         dest
 */
PLAY_CGLM_INLINE
ivec2s
ivec2s_(subsub)(ivec2s a, ivec2s b, ivec2s dest)
{
    ivec2_subsub(a.raw, b.raw, dest.raw);
    return dest;
}

/*!
 * @brief subtract scalar [s] from vector [a] and subtract result from [dest]
 *
 * applies -= operator so dest must be initialized
 *
 * @param[in]  a    vector
 * @param[in]  s    scalar
 * @param[in]  dest dest -= (a - s)
 * @returns         dest
 */
PLAY_CGLM_INLINE
ivec2s
ivec2s_(subsubs)(ivec2s a, int s, ivec2s dest)
{
    ivec2_subsubs(a.raw, s, dest.raw);
    return dest;
}

/*!
 * @brief add vector [a] to vector [b] and subtract the result from [dest]
 *
 * applies -= operator so dest must be initialized
 *
 * @param[in]  a    vector
 * @param[in]  b    scalar
 * @param[in]  dest dest -= (a + b)
 * @returns         dest
 */
PLAY_CGLM_INLINE
ivec2s
ivec2s_(addsub)(ivec2s a, ivec2s b, ivec2s dest)
{
    ivec2_addsub(a.raw, b.raw, dest.raw);
    return dest;
}

/*!
 * @brief add scalar [s] to vector [a] and subtract the result from [dest]
 *
 * applies -= operator so dest must be initialized
 *
 * @param[in]  a    vector
 * @param[in]  s    scalar
 * @param[in]  dest dest -= (a + b)
 * @returns         dest
 */
PLAY_CGLM_INLINE
ivec2s
ivec2s_(addsubs)(ivec2s a, int s, ivec2s dest)
{
    ivec2_addsubs(a.raw, s, dest.raw);
    return dest;
}

/*!
 * @brief multiply vector [a] and vector [b] and subtract the result from [dest]
 *
 * applies -= operator so dest must be initialized
 *
 * @param[in]  a    vector
 * @param[in]  b    scalar
 * @param[in]  dest dest -= (a * b)
 * @returns         dest
 */
PLAY_CGLM_INLINE
ivec2s
ivec2s_(mulsub)(ivec2s a, ivec2s b, ivec2s dest)
{
    ivec2_mulsub(a.raw, b.raw, dest.raw);
    return dest;
}

/*!
 * @brief multiply vector [a] with scalar [s] and subtract the result from [dest]
 *
 * applies -= operator so dest must be initialized
 *
 * @param[in]  a    vector
 * @param[in]  s    scalar
 * @param[in]  dest dest -= (a * s)
 * @returns         dest
 */
PLAY_CGLM_INLINE
ivec2s
ivec2s_(mulsubs)(ivec2s a, int s, ivec2s dest)
{
    ivec2_mulsubs(a.raw, s, dest.raw);
    return dest;
}

/*!
 * @brief subtract maximum of vector [a] and vector [b] from vector [dest]
 *
 * applies += operator so dest must be initialized
 *
 * @param[in]  a    first vector
 * @param[in]  b    second vector
 * @param[in]  dest dest -= fmax(a, b)
 * @returns         dest
 */
PLAY_CGLM_INLINE
ivec2s
ivec2s_(maxsub)(ivec2s a, ivec2s b, ivec2s dest)
{
    ivec2_maxsub(a.raw, b.raw, dest.raw);
    return dest;
}

/*!
 * @brief subtract minimum of vector [a] and vector [b] from vector [dest]
 *
 * applies -= operator so dest must be initialized
 *
 * @param[in]  a    first vector
 * @param[in]  b    second vector
 * @param[in]  dest dest -= fmin(a, b)
 * @returns         dest
 */
PLAY_CGLM_INLINE
ivec2s
ivec2s_(minsub)(ivec2s a, ivec2s b, ivec2s dest)
{
    ivec2_minsub(a.raw, b.raw, dest.raw);
    return dest;
}

/*!
 * @brief squared distance between two vectors
 *
 * @param[in] a first vector
 * @param[in] b second vector
 * @return returns squared distance (distance * distance)
 */
PLAY_CGLM_INLINE
int
ivec2s_(distance2)(ivec2s a, ivec2s b)
{
    return ivec2_distance2(a.raw, b.raw);
}

/*!
 * @brief distance between two vectors
 *
 * @param[in] a first vector
 * @param[in] b second vector
 * @return returns distance
 */
PLAY_CGLM_INLINE
float
ivec2s_(distance)(ivec2s a, ivec2s b)
{
    return ivec2_distance(a.raw, b.raw);
}

/*!
 * @brief fill a vector with specified value
 *
 * @param[in]  val value
 * @returns        dest
 */
PLAY_CGLM_INLINE
ivec2s
ivec2s_(fill)(int val)
{
    ivec2s r;
    ivec2_fill(r.raw, val);
    return r;
}

/*!
 * @brief check if vector is equal to value
 *
 * @param[in] v   vector
 * @param[in] val value
 */
PLAY_CGLM_INLINE
bool
ivec2s_(eq)(ivec2s v, int val)
{
    return ivec2_eq(v.raw, val);
}

/*!
 * @brief check if vector is equal to another
 *
 * @param[in] a vector
 * @param[in] b vector
 */
PLAY_CGLM_INLINE
bool
ivec2s_(eqv)(ivec2s a, ivec2s b)
{
    return ivec2_eqv(a.raw, b.raw);
}

/*!
 * @brief set each member of dest to greater of vector a and b
 *
 * @param[in]  a    first vector
 * @param[in]  b    second vector
 * @returns         destination
 */
PLAY_CGLM_INLINE
ivec2s
ivec2s_(maxv)(ivec2s a, ivec2s b)
{
    ivec2s r;
    ivec2_maxv(a.raw, b.raw, r.raw);
    return r;
}

/*!
 * @brief set each member of dest to lesser of vector a and b
 *
 * @param[in]  a    first vector
 * @param[in]  b    second vector
 * @returns         destination
 */
PLAY_CGLM_INLINE
ivec2s
ivec2s_(minv)(ivec2s a, ivec2s b)
{
    ivec2s r;
    ivec2_minv(a.raw, b.raw, r.raw);
    return r;
}

/*!
 * @brief clamp each member of [v] between minVal and maxVal (inclusive)
 *
 * @param[in]      v      vector
 * @param[in]      minVal minimum value
 * @param[in]      maxVal maximum value
 * @returns               clamped vector
 */
PLAY_CGLM_INLINE
ivec2s
ivec2s_(clamp)(ivec2s v, int minVal, int maxVal)
{
    ivec2_clamp(v.raw, minVal, maxVal);
    return v;
}

/*!
 * @brief absolute value of v
 *
 * @param[in]	v	vector
 * @returns     destination
 */
PLAY_CGLM_INLINE
ivec2s
ivec2s_(abs)(ivec2s v)
{
    ivec2s r;
    ivec2_abs(v.raw, r.raw);
    return r;
}

#endif /* civec2s_h */

/*** End of inlined file: ivec2.h ***/


/*** Start of inlined file: ivec3.h ***/
/*
 Macros:
  PLAY_CGLM_S_IVEC3_ONE_INIT
  PLAY_CGLM_S_IVEC3_ZERO_INIT
  PLAY_CGLM_S_IVEC3_ONE
  PLAY_CGLM_S_IVEC3_ZERO

 Functions:
  PLAY_CGLM_INLINE ivec3s ivec3_new(ivec4s v4)
  PLAY_CGLM_INLINE void ivec3s_pack(ivec3s dst[], ivec3 src[], size_t len)
  PLAY_CGLM_INLINE void ivec3s_unpack(ivec3 dst[], ivec3s src[], size_t len)
  PLAY_CGLM_INLINE ivec3s ivec3s_zero(void)
  PLAY_CGLM_INLINE ivec3s ivec3s_one(void)
  PLAY_CGLM_INLINE int ivec3s_dot(ivec3s a, ivec3s b)
  PLAY_CGLM_INLINE int ivec3s_norm2(ivec3s v)
  PLAY_CGLM_INLINE int ivec3s_norm(ivec3s v)
  PLAY_CGLM_INLINE ivec3s ivec3s_add(ivec3s a, ivec3s b)
  PLAY_CGLM_INLINE ivec3s ivec3s_adds(ivec3s v, int s)
  PLAY_CGLM_INLINE ivec3s ivec3s_sub(ivec3s a, ivec3s b)
  PLAY_CGLM_INLINE ivec3s ivec3s_subs(ivec3s v, int s)
  PLAY_CGLM_INLINE ivec3s ivec3s_mul(ivec3s a, ivec3s b)
  PLAY_CGLM_INLINE ivec3s ivec3s_scale(ivec3s v, int s)
  PLAY_CGLM_INLINE ivec3s ivec3s_div(ivec3s a, ivec3s b)
  PLAY_CGLM_INLINE ivec3s ivec3s_divs(ivec3s v, int s)
  PLAY_CGLM_INLINE ivec3s ivec3s_mod(ivec3s a, ivec3s b)
  PLAY_CGLM_INLINE ivec3s ivec3s_addadd(ivec3s a, ivec3s b, ivec3s dest)
  PLAY_CGLM_INLINE ivec3s ivec3s_addadds(ivec3s a, int s, ivec3s dest)
  PLAY_CGLM_INLINE ivec3s ivec3s_subadd(ivec3s a, ivec3s b, ivec3s dest)
  PLAY_CGLM_INLINE ivec3s ivec3s_subadds(ivec3s a, int s, ivec3s dest)
  PLAY_CGLM_INLINE ivec3s ivec3s_muladd(ivec3s a, ivec3s b, ivec3s dest)
  PLAY_CGLM_INLINE ivec3s ivec3s_muladds(ivec3s a, int s, ivec3s dest)
  PLAY_CGLM_INLINE ivec3s ivec3s_minadd(ivec3s a, ivec3s b, ivec3s dest)
  PLAY_CGLM_INLINE ivec3s ivec3s_subsub(ivec3s a, ivec3s b, ivec3s dest)
  PLAY_CGLM_INLINE ivec3s ivec3s_subsubs(ivec3s a, int s, ivec3s dest)
  PLAY_CGLM_INLINE ivec3s ivec3s_addsub(ivec3s a, ivec3s b, ivec3s dest)
  PLAY_CGLM_INLINE ivec3s ivec3s_addsubs(ivec3s a, int s, ivec3s dest)
  PLAY_CGLM_INLINE ivec3s ivec3s_mulsub(ivec3s a, ivec3s b, ivec3s dest)
  PLAY_CGLM_INLINE ivec3s ivec3s_mulsubs(ivec3s a, int s, ivec3s dest)
  PLAY_CGLM_INLINE ivec3s ivec3s_maxsub(ivec3s a, ivec3s b, ivec3s dest)
  PLAY_CGLM_INLINE ivec3s ivec3s_minsub(ivec3s a, ivec3s b, ivec3s dest)
  PLAY_CGLM_INLINE int ivec3s_distance2(ivec3s a, ivec3s b)
  PLAY_CGLM_INLINE float ivec3s_distance(ivec3s a, ivec3s b)
  PLAY_CGLM_INLINE ivec3s ivec3s_fill(int val)
  PLAY_CGLM_INLINE bool ivec3s_eq(ivec3s v, int val)
  PLAY_CGLM_INLINE bool ivec3s_eqv(ivec3s a, ivec3s b)
  PLAY_CGLM_INLINE ivec3s ivec3s_maxv(ivec3s a, ivec3s b)
  PLAY_CGLM_INLINE ivec3s ivec3s_minv(ivec3s a, ivec3s b)
  PLAY_CGLM_INLINE ivec3s ivec3s_clamp(ivec3s v, int minVal, int maxVal)
  PLAY_CGLM_INLINE ivec3s ivec3s_abs(ivec3s v)
 */

#ifndef civec3s_h
#define civec3s_h

#define ivec3s_(NAME) PLAY_CGLM_STRUCTAPI(ivec3, NAME)

#define PLAY_CGLM_S_IVEC3_ONE_INIT   {PLAY_CGLM_IVEC3_ONE_INIT}
#define PLAY_CGLM_S_IVEC3_ZERO_INIT  {PLAY_CGLM_IVEC3_ZERO_INIT}

#define PLAY_CGLM_S_IVEC3_ONE  ((ivec3s)PLAY_CGLM_S_IVEC3_ONE_INIT)
#define PLAY_CGLM_S_IVEC3_ZERO ((ivec3s)PLAY_CGLM_S_IVEC3_ZERO_INIT)

/*!
 * @brief init ivec3 using ivec4
 *
 * @param[in]  v4   vector4
 * @returns         destination
 */
PLAY_CGLM_INLINE
ivec3s
ivec3_new(ivec4s v4)
{
    ivec3s r;
    ivec3_new(v4.raw, r.raw);
    return r;
}

/*!
 * @brief pack an array of ivec3 into an array of ivec3s
 *
 * @param[out] dst array of ivec3s
 * @param[in]  src array of ivec3
 * @param[in]  len number of elements
 */
PLAY_CGLM_INLINE
void
ivec3s_(pack)(ivec3s dst[], ivec3 src[], size_t len)
{
    size_t i;

    for (i = 0; i < len; i++)
    {
        ivec3_copy(src[i], dst[i].raw);
    }
}

/*!
 * @brief unpack an array of ivec3s into an array of ivec3
 *
 * @param[out] dst array of ivec3
 * @param[in]  src array of ivec3s
 * @param[in]  len number of elements
 */
PLAY_CGLM_INLINE
void
ivec3s_(unpack)(ivec3 dst[], ivec3s src[], size_t len)
{
    size_t i;

    for (i = 0; i < len; i++)
    {
        ivec3_copy(src[i].raw, dst[i]);
    }
}

/*!
 * @brief set all members of [v] to zero
 *
 * @returns vector
 */
PLAY_CGLM_INLINE
ivec3s
ivec3s_(zero)(void)
{
    ivec3s r;
    ivec3_zero(r.raw);
    return r;
}

/*!
 * @brief set all members of [v] to one
 *
 * @returns vector
 */
PLAY_CGLM_INLINE
ivec3s
ivec3s_(one)(void)
{
    ivec3s r;
    ivec3_one(r.raw);
    return r;
}

/*!
 * @brief ivec3 dot product
 *
 * @param[in] a vector1
 * @param[in] b vector2
 *
 * @return dot product
 */
PLAY_CGLM_INLINE
int
ivec3s_(dot)(ivec3s a, ivec3s b)
{
    return ivec3_dot(a.raw, b.raw);
}

/*!
 * @brief norm * norm (magnitude) of vec
 *
 * we can use this func instead of calling norm * norm, because it would call
 * sqrtf function twice but with this func we can avoid func call, maybe this is
 * not good name for this func
 *
 * @param[in] v vector
 *
 * @return norm * norm
 */
PLAY_CGLM_INLINE
int
ivec3s_(norm2)(ivec3s v)
{
    return ivec3_norm2(v.raw);
}

/*!
 * @brief euclidean norm (magnitude), also called L2 norm
 *        this will give magnitude of vector in euclidean space
 *
 * @param[in] v vector
 *
 * @return norm
 */
PLAY_CGLM_INLINE
int
ivec3s_(norm)(ivec3s v)
{
    return ivec3_norm(v.raw);
}

/*!
 * @brief add vector [a] to vector [b] and store result in [dest]
 *
 * @param[in]  a    first vector
 * @param[in]  b    second vector
 * @returns         destination
 */
PLAY_CGLM_INLINE
ivec3s
ivec3s_(add)(ivec3s a, ivec3s b)
{
    ivec3s r;
    ivec3_add(a.raw, b.raw, r.raw);
    return r;
}

/*!
 * @brief add scalar s to vector [v] and store result in [dest]
 *
 * @param[in]  v    vector
 * @param[in]  s    scalar
 * @returns         destination
 */
PLAY_CGLM_INLINE
ivec3s
ivec3s_(adds)(ivec3s v, int s)
{
    ivec3s r;
    ivec3_adds(v.raw, s, r.raw);
    return r;
}

/*!
 * @brief subtract vector [b] from vector [a] and store result in [dest]
 *
 * @param[in]  a    first vector
 * @param[in]  b    second vector
 * @returns         destination
 */
PLAY_CGLM_INLINE
ivec3s
ivec3s_(sub)(ivec3s a, ivec3s b)
{
    ivec3s r;
    ivec3_sub(a.raw, b.raw, r.raw);
    return r;
}

/*!
 * @brief subtract scalar s from vector [v] and store result in [dest]
 *
 * @param[in]  v    vector
 * @param[in]  s    scalar
 * @returns         destination
 */
PLAY_CGLM_INLINE
ivec3s
ivec3s_(subs)(ivec3s v, int s)
{
    ivec3s r;
    ivec3_subs(v.raw, s, r.raw);
    return r;
}

/*!
 * @brief multiply vector [a] with vector [b] and store result in [dest]
 *
 * @param[in]  a    first vector
 * @param[in]  b    second vector
 * @returns         destination
 */
PLAY_CGLM_INLINE
ivec3s
ivec3s_(mul)(ivec3s a, ivec3s b)
{
    ivec3s r;
    ivec3_mul(a.raw, b.raw, r.raw);
    return r;
}

/*!
 * @brief multiply vector [a] with scalar s and store result in [dest]
 *
 * @param[in]  v    vector
 * @param[in]  s    scalar
 * @returns         destination
 */
PLAY_CGLM_INLINE
ivec3s
ivec3s_(scale)(ivec3s v, int s)
{
    ivec3s r;
    ivec3_scale(v.raw, s, r.raw);
    return r;
}

/*!
 * @brief div vector with another component-wise division: d = a / b
 *
 * @param[in]  a    vector 1
 * @param[in]  b    vector 2
 * @returns         result = (a[0]/b[0], a[1]/b[1], a[2]/b[2])
 */
PLAY_CGLM_INLINE
ivec3s
ivec3s_(div)(ivec3s a, ivec3s b)
{
    ivec3s r;
    ivec3_div(a.raw, b.raw, r.raw);
    return r;
}

/*!
 * @brief div vector with scalar: d = v / s
 *
 * @param[in]  v    vector
 * @param[in]  s    scalar
 * @returns         result = (a[0]/s, a[1]/s, a[2]/s)
 */
PLAY_CGLM_INLINE
ivec3s
ivec3s_(divs)(ivec3s v, int s)
{
    ivec3s r;
    ivec3_divs(v.raw, s, r.raw);
    return r;
}

/*!
 * @brief Element-wise modulo operation on ivec3 vectors: dest = a % b
 *
 * Performs element-wise modulo on each component of vectors `a` and `b`.
 *
 * @param[in]  a    vector 1
 * @param[in]  b    vector 2
 * @returns         result = (a[0]%b[0], a[1]%b[1], a[2]%b[2])
 */
PLAY_CGLM_INLINE
ivec3s
ivec3s_(mod)(ivec3s a, ivec3s b)
{
    ivec3s r;
    ivec3_mod(a.raw, b.raw, r.raw);
    return r;
}

/*!
 * @brief add vector [a] with vector [b] and add result to vector [dest]
 *
 * applies += operator so dest must be initialized
 *
 * @param[in]  a    first vector
 * @param[in]  b    second vector
 * @param[in]  dest dest += (a + b)
 * @returns         dest
 */
PLAY_CGLM_INLINE
ivec3s
ivec3s_(addadd)(ivec3s a, ivec3s b, ivec3s dest)
{
    ivec3_addadd(a.raw, b.raw, dest.raw);
    return dest;
}

/*!
 * @brief add scalar [s] onto vector [a] and add result to vector [dest]
 *
 * applies += operator so dest must be initialized
 *
 * @param[in]  a    vector
 * @param[in]  s    scalar
 * @param[in]  dest dest += (a + s)
 * @returns         dest
 */
PLAY_CGLM_INLINE
ivec3s
ivec3s_(addadds)(ivec3s a, int s, ivec3s dest)
{
    ivec3_addadds(a.raw, s, dest.raw);
    return dest;
}

/*!
 * @brief subtract vector [a] from vector [b] and add result to [dest]
 *
 * applies += operator so dest must be initialized
 *
 * @param[in]  a    first vector
 * @param[in]  b    second vector
 * @param[in]  dest dest += (a - b)
 * @returns         dest
 */
PLAY_CGLM_INLINE
ivec3s
ivec3s_(subadd)(ivec3s a, ivec3s b, ivec3s dest)
{
    ivec3_subadd(a.raw, b.raw, dest.raw);
    return dest;
}

/*!
 * @brief subtract scalar [s] from vector [a] and add result to [dest]
 *
 * applies += operator so dest must be initialized
 *
 * @param[in]  a    first
 * @param[in]  s    scalar
 * @param[in]  dest dest += (a - s)
 * @returns         dest
 */
PLAY_CGLM_INLINE
ivec3s
ivec3s_(subadds)(ivec3s a, int s, ivec3s dest)
{
    ivec3_subadds(a.raw, s, dest.raw);
    return dest;
}

/*!
 * @brief multiply vector [a] with vector [b] and add result to [dest]
 *
 * applies += operator so dest must be initialized
 *
 * @param[in]  a    first vector
 * @param[in]  b    second vector
 * @param[in]  dest dest += (a * b)
 * @returns         dest
 */
PLAY_CGLM_INLINE
ivec3s
ivec3s_(muladd)(ivec3s a, ivec3s b, ivec3s dest)
{
    ivec3_muladd(a.raw, b.raw, dest.raw);
    return dest;
}

/*!
 * @brief multiply vector [a] with scalar [s] and add result to [dest]
 *
 * applies += operator so dest must be initialized
 *
 * @param[in]  a    vector
 * @param[in]  s    scalar
 * @param[in]  dest dest += (a * s)
 * @returns         dest
 */
PLAY_CGLM_INLINE
ivec3s
ivec3s_(muladds)(ivec3s a, int s, ivec3s dest)
{
    ivec3_muladds(a.raw, s, dest.raw);
    return dest;
}

/*!
 * @brief add maximum of vector [a] and vector [b] to vector [dest]
 *
 * applies += operator so dest must be initialized
 *
 * @param[in]  a    first vector
 * @param[in]  b    second vector
 * @param[in]  dest dest += fmax(a, b)
 * @returns         dest
 */
PLAY_CGLM_INLINE
ivec3s
ivec3s_(maxadd)(ivec3s a, ivec3s b, ivec3s dest)
{
    ivec3_maxadd(a.raw, b.raw, dest.raw);
    return dest;
}

/*!
 * @brief add minimum of vector [a] and vector [b] to vector [dest]
 *
 * applies += operator so dest must be initialized
 *
 * @param[in]  a    first vector
 * @param[in]  b    second vector
 * @param[in]  dest dest += fmin(a, b)
 * @returns         dest
 */
PLAY_CGLM_INLINE
ivec3s
ivec3s_(minadd)(ivec3s a, ivec3s b, ivec3s dest)
{
    ivec3_minadd(a.raw, b.raw, dest.raw);
    return dest;
}

/*!
 * @brief subtract vector [a] from vector [b] and subtract result from [dest]
 *
 * applies -= operator so dest must be initialized
 *
 * @param[in]  a    first vector
 * @param[in]  b    second vector
 * @param[in]  dest dest -= (a - b)
 * @returns         dest
 */
PLAY_CGLM_INLINE
ivec3s
ivec3s_(subsub)(ivec3s a, ivec3s b, ivec3s dest)
{
    ivec3_subsub(a.raw, b.raw, dest.raw);
    return dest;
}

/*!
 * @brief subtract scalar [s] from vector [a] and subtract result from [dest]
 *
 * applies -= operator so dest must be initialized
 *
 * @param[in]  a    vector
 * @param[in]  s    scalar
 * @param[in]  dest dest -= (a - s)
 * @returns         dest
 */
PLAY_CGLM_INLINE
ivec3s
ivec3s_(subsubs)(ivec3s a, int s, ivec3s dest)
{
    ivec3_subsubs(a.raw, s, dest.raw);
    return dest;
}

/*!
 * @brief add vector [a] to vector [b] and subtract the result from [dest]
 *
 * applies -= operator so dest must be initialized
 *
 * @param[in]  a    vector
 * @param[in]  b    scalar
 * @param[in]  dest dest -= (a + b)
 * @returns         dest
 */
PLAY_CGLM_INLINE
ivec3s
ivec3s_(addsub)(ivec3s a, ivec3s b, ivec3s dest)
{
    ivec3_addsub(a.raw, b.raw, dest.raw);
    return dest;
}

/*!
 * @brief add scalar [s] to vector [a] and subtract the result from [dest]
 *
 * applies -= operator so dest must be initialized
 *
 * @param[in]  a    vector
 * @param[in]  s    scalar
 * @param[in]  dest dest -= (a + b)
 * @returns         dest
 */
PLAY_CGLM_INLINE
ivec3s
ivec3s_(addsubs)(ivec3s a, int s, ivec3s dest)
{
    ivec3_addsubs(a.raw, s, dest.raw);
    return dest;
}

/*!
 * @brief multiply vector [a] and vector [b] and subtract the result from [dest]
 *
 * applies -= operator so dest must be initialized
 *
 * @param[in]  a    vector
 * @param[in]  b    scalar
 * @param[in]  dest dest -= (a * b)
 * @returns         dest
 */
PLAY_CGLM_INLINE
ivec3s
ivec3s_(mulsub)(ivec3s a, ivec3s b, ivec3s dest)
{
    ivec3_mulsub(a.raw, b.raw, dest.raw);
    return dest;
}

/*!
 * @brief multiply vector [a] with scalar [s] and subtract the result from [dest]
 *
 * applies -= operator so dest must be initialized
 *
 * @param[in]  a    vector
 * @param[in]  s    scalar
 * @param[in]  dest dest -= (a * s)
 * @returns         dest
 */
PLAY_CGLM_INLINE
ivec3s
ivec3s_(mulsubs)(ivec3s a, int s, ivec3s dest)
{
    ivec3_mulsubs(a.raw, s, dest.raw);
    return dest;
}

/*!
 * @brief subtract maximum of vector [a] and vector [b] from vector [dest]
 *
 * applies += operator so dest must be initialized
 *
 * @param[in]  a    first vector
 * @param[in]  b    second vector
 * @param[in]  dest dest -= fmax(a, b)
 * @returns         dest
 */
PLAY_CGLM_INLINE
ivec3s
ivec3s_(maxsub)(ivec3s a, ivec3s b, ivec3s dest)
{
    ivec3_maxsub(a.raw, b.raw, dest.raw);
    return dest;
}

/*!
 * @brief subtract minimum of vector [a] and vector [b] from vector [dest]
 *
 * applies -= operator so dest must be initialized
 *
 * @param[in]  a    first vector
 * @param[in]  b    second vector
 * @param[in]  dest dest -= fmin(a, b)
 * @returns         dest
 */
PLAY_CGLM_INLINE
ivec3s
ivec3s_(minsub)(ivec3s a, ivec3s b, ivec3s dest)
{
    ivec3_minsub(a.raw, b.raw, dest.raw);
    return dest;
}

/*!
 * @brief squared distance between two vectors
 *
 * @param[in] a first vector
 * @param[in] b second vector
 * @return returns squared distance (distance * distance)
 */
PLAY_CGLM_INLINE
int
ivec3s_(distance2)(ivec3s a, ivec3s b)
{
    return ivec3_distance2(a.raw, b.raw);
}

/*!
 * @brief distance between two vectors
 *
 * @param[in] a first vector
 * @param[in] b second vector
 * @return returns distance
 */
PLAY_CGLM_INLINE
float
ivec3s_(distance)(ivec3s a, ivec3s b)
{
    return ivec3_distance(a.raw, b.raw);
}

/*!
 * @brief fill a vector with specified value
 *
 * @param[in]  val value
 * @returns        destination
 */
PLAY_CGLM_INLINE
ivec3s
ivec3s_(fill)(int val)
{
    ivec3s r;
    ivec3_fill(r.raw, val);
    return r;
}

/*!
 * @brief check if vector is equal to value
 *
 * @param[in] v   vector
 * @param[in] val value
 */
PLAY_CGLM_INLINE
bool
ivec3s_(eq)(ivec3s v, int val)
{
    return ivec3_eq(v.raw, val);
}

/*!
 * @brief check if vector is equal to another
 *
 * @param[in] a vector
 * @param[in] b vector
 */
PLAY_CGLM_INLINE
bool
ivec3s_(eqv)(ivec3s a, ivec3s b)
{
    return ivec3_eqv(a.raw, b.raw);
}

/*!
 * @brief set each member of dest to greater of vector a and b
 *
 * @param[in]  a    first vector
 * @param[in]  b    second vector
 * @returns         destination
 */
PLAY_CGLM_INLINE
ivec3s
ivec3s_(maxv)(ivec3s a, ivec3s b)
{
    ivec3s r;
    ivec3_maxv(a.raw, b.raw, r.raw);
    return r;
}

/*!
 * @brief set each member of dest to lesser of vector a and b
 *
 * @param[in]  a    first vector
 * @param[in]  b    second vector
 * @returns         destination
 */
PLAY_CGLM_INLINE
ivec3s
ivec3s_(minv)(ivec3s a, ivec3s b)
{
    ivec3s r;
    ivec3_minv(a.raw, b.raw, r.raw);
    return r;
}

/*!
 * @brief clamp each member of [v] between minVal and maxVal (inclusive)
 *
 * @param[in]      v      vector
 * @param[in]      minVal minimum value
 * @param[in]      maxVal maximum value
 * @returns               clamped vector
 */
PLAY_CGLM_INLINE
ivec3s
ivec3s_(clamp)(ivec3s v, int minVal, int maxVal)
{
    ivec3_clamp(v.raw, minVal, maxVal);
    return v;
}

/*!
 * @brief absolute value of v
 *
 * @param[in]	v	vector
 * @returns     destination
 */
PLAY_CGLM_INLINE
ivec3s
ivec3s_(abs)(ivec3s v)
{
    ivec3s r;
    ivec3_abs(v.raw, r.raw);
    return r;
}

#endif /* civec3s_h */

/*** End of inlined file: ivec3.h ***/


/*** Start of inlined file: ivec4.h ***/
/*
 Macros:
  PLAY_CGLM_S_IVEC4_ONE_INIT
  PLAY_CGLM_S_IVEC4_ZERO_INIT
  PLAY_CGLM_S_IVEC4_ONE
  PLAY_CGLM_S_IVEC4_ZERO

 Functions:
  PLAY_CGLM_INLINE ivec4s ivec4_new(ivec3s v3, int last)
  PLAY_CGLM_INLINE void ivec4s_pack(ivec4s dst[], ivec4 src[], size_t len)
  PLAY_CGLM_INLINE void ivec4s_unpack(ivec4 dst[], ivec4s src[], size_t len)
  PLAY_CGLM_INLINE ivec4s  ivec4s_zero(void)
  PLAY_CGLM_INLINE ivec4s ivec4s_one(void)
  PLAY_CGLM_INLINE ivec4s ivec4s_add(ivec4s a, ivec4s b)
  PLAY_CGLM_INLINE ivec4s ivec4s_adds(ivec4s v, int s)
  PLAY_CGLM_INLINE ivec4s ivec4s_sub(ivec4s a, ivec4s b)
  PLAY_CGLM_INLINE ivec4s ivec4s_subs(ivec4s v, int s)
  PLAY_CGLM_INLINE ivec4s ivec4s_mul(ivec4s a, ivec4s b)
  PLAY_CGLM_INLINE ivec4s ivec4s_scale(ivec4s v, int s)
  PLAY_CGLM_INLINE ivec4s ivec4s_addadd(ivec4s a, ivec4s b, ivec4s dest)
  PLAY_CGLM_INLINE ivec4s ivec4s_addadds(ivec4s a, int s, ivec4s dest)
  PLAY_CGLM_INLINE ivec4s ivec4s_subadd(ivec4s a, ivec4s b, ivec4s dest)
  PLAY_CGLM_INLINE ivec4s ivec4s_subadds(ivec4s a, int s, ivec4s dest)
  PLAY_CGLM_INLINE ivec4s ivec4s_muladd(ivec4s a, ivec4s b, ivec4s dest)
  PLAY_CGLM_INLINE ivec4s ivec4s_muladds(ivec4s a, int s, ivec4s dest)
  PLAY_CGLM_INLINE ivec4s ivec4s_maxadd(ivec4s a, ivec4s b, ivec4s dest)
  PLAY_CGLM_INLINE ivec4s ivec4s_minadd(ivec4s a, ivec4s b, ivec4s dest)
  PLAY_CGLM_INLINE ivec4s ivec4s_subsub(ivec4s a, ivec4s b, ivec4s dest)
  PLAY_CGLM_INLINE ivec4s ivec4s_subsubs(ivec4s a, int s, ivec4s dest)
  PLAY_CGLM_INLINE ivec4s ivec4s_addsub(ivec4s a, ivec4s b, ivec4s dest)
  PLAY_CGLM_INLINE ivec4s ivec4s_addsubs(ivec4s a, int s, ivec4s dest)
  PLAY_CGLM_INLINE ivec4s ivec4s_mulsub(ivec4s a, ivec4s b, ivec4s dest)
  PLAY_CGLM_INLINE ivec4s ivec4s_mulsubs(ivec4s a, int s, ivec4s dest)
  PLAY_CGLM_INLINE ivec4s ivec4s_maxsub(ivec4s a, ivec4s b, ivec4s dest)
  PLAY_CGLM_INLINE ivec4s ivec4s_minsub(ivec4s a, ivec4s b, ivec4s dest)
  PLAY_CGLM_INLINE int ivec4s_distance2(ivec4s a, ivec4s b)
  PLAY_CGLM_INLINE float ivec4s_distance(ivec4s a, ivec4s b)
  PLAY_CGLM_INLINE ivec4s ivec4s_maxv(ivec4s a, ivec4s b)
  PLAY_CGLM_INLINE ivec4s ivec4s_minv(ivec4s a, ivec4s b)
  PLAY_CGLM_INLINE ivec4s ivec4s_clamp(ivec4s v, int minVal, int maxVal)
  PLAY_CGLM_INLINE ivec4s ivec4s_abs(ivec4s v)
 */

#ifndef civec4s_h
#define civec4s_h

#define ivec4s_(NAME) PLAY_CGLM_STRUCTAPI(ivec4, NAME)

#define PLAY_CGLM_S_IVEC4_ONE_INIT   {PLAY_CGLM_IVEC4_ONE_INIT}
#define PLAY_CGLM_S_IVEC4_ZERO_INIT  {PLAY_CGLM_IVEC4_ZERO_INIT}

#define PLAY_CGLM_S_IVEC4_ONE  ((ivec4s)PLAY_CGLM_S_IVEC4_ONE_INIT)
#define PLAY_CGLM_S_IVEC4_ZERO ((ivec4s)PLAY_CGLM_S_IVEC4_ZERO_INIT)

/*!
 * @brief init ivec4 using ivec3
 *
 * @param[in]  v3   vector3
 * @param[in]  last last item
 * @returns         destination
 */
PLAY_CGLM_INLINE
ivec4s
ivec4_new(ivec3s v3, int last)
{
    ivec4s r;
    ivec4_new(v3.raw, last, r.raw);
    return r;
}

/*!
 * @brief pack an array of ivec4 into an array of ivec4s
 *
 * @param[out] dst array of ivec4s
 * @param[in]  src array of ivec4
 * @param[in]  len number of elements
 */
PLAY_CGLM_INLINE
void
ivec4s_(pack)(ivec4s dst[], ivec4 src[], size_t len)
{
    size_t i;

    for (i = 0; i < len; i++)
    {
        ivec4_copy(src[i], dst[i].raw);
    }
}

/*!
 * @brief unpack an array of ivec4s into an array of ivec4
 *
 * @param[out] dst array of ivec4
 * @param[in]  src array of ivec4s
 * @param[in]  len number of elements
 */
PLAY_CGLM_INLINE
void
ivec4s_(unpack)(ivec4 dst[], ivec4s src[], size_t len)
{
    size_t i;

    for (i = 0; i < len; i++)
    {
        ivec4_copy(src[i].raw, dst[i]);
    }
}

/*!
 * @brief set all members of [v] to zero
 *
 * @returns vector
 */
PLAY_CGLM_INLINE
ivec4s
ivec4s_(zero)(void)
{
    ivec4s r;
    ivec4_zero(r.raw);
    return r;
}

/*!
 * @brief set all members of [v] to one
 *
 * @returns vector
 */
PLAY_CGLM_INLINE
ivec4s
ivec4s_(one)(void)
{
    ivec4s r;
    ivec4_one(r.raw);
    return r;
}

/*!
 * @brief add vector [a] to vector [b] and store result in [dest]
 *
 * @param[in]  a    first vector
 * @param[in]  b    second vector
 * @returns         destination
 */
PLAY_CGLM_INLINE
ivec4s
ivec4s_(add)(ivec4s a, ivec4s b)
{
    ivec4s r;
    ivec4_add(a.raw, b.raw, r.raw);
    return r;
}

/*!
 * @brief add scalar s to vector [v] and store result in [dest]
 *
 * @param[in]  v    vector
 * @param[in]  s    scalar
 * @returns         destination
 */
PLAY_CGLM_INLINE
ivec4s
ivec4s_(adds)(ivec4s v, int s)
{
    ivec4s r;
    ivec4_adds(v.raw, s, r.raw);
    return r;
}

/*!
 * @brief subtract vector [b] from vector [a] and store result in [dest]
 *
 * @param[in]  a    first vector
 * @param[in]  b    second vector
 * @returns         destination
 */
PLAY_CGLM_INLINE
ivec4s
ivec4s_(sub)(ivec4s a, ivec4s b)
{
    ivec4s r;
    ivec4_sub(a.raw, b.raw, r.raw);
    return r;
}

/*!
 * @brief subtract scalar s from vector [v] and store result in [dest]
 *
 * @param[in]  v    vector
 * @param[in]  s    scalar
 * @returns         destination
 */
PLAY_CGLM_INLINE
ivec4s
ivec4s_(subs)(ivec4s v, int s)
{
    ivec4s r;
    ivec4_subs(v.raw, s, r.raw);
    return r;
}

/*!
 * @brief multiply vector [a] with vector [b] and store result in [dest]
 *
 * @param[in]  a    first vector
 * @param[in]  b    second vector
 * @returns         destination
 */
PLAY_CGLM_INLINE
ivec4s
ivec4s_(mul)(ivec4s a, ivec4s b)
{
    ivec4s r;
    ivec4_mul(a.raw, b.raw, r.raw);
    return r;
}

/*!
 * @brief multiply vector [a] with scalar s and store result in [dest]
 *
 * @param[in]  v    vector
 * @param[in]  s    scalar
 * @returns         destination
 */
PLAY_CGLM_INLINE
ivec4s
ivec4s_(scale)(ivec4s v, int s)
{
    ivec4s r;
    ivec4_scale(v.raw, s, r.raw);
    return r;
}

/*!
 * @brief add vector [a] with vector [b] and add result to vector [dest]
 *
 * applies += operator so dest must be initialized
 *
 * @param[in]  a    first vector
 * @param[in]  b    second vector
 * @param[in]  dest dest += (a + b)
 * @returns         dest
 */
PLAY_CGLM_INLINE
ivec4s
ivec4s_(addadd)(ivec4s a, ivec4s b, ivec4s dest)
{
    ivec4_addadd(a.raw, b.raw, dest.raw);
    return dest;
}

/*!
 * @brief add scalar [s] onto vector [a] and add result to vector [dest]
 *
 * applies += operator so dest must be initialized
 *
 * @param[in]  a    vector
 * @param[in]  s    scalar
 * @param[in]  dest dest += (a + s)
 * @returns         dest
 */
PLAY_CGLM_INLINE
ivec4s
ivec4s_(addadds)(ivec4s a, int s, ivec4s dest)
{
    ivec4_addadds(a.raw, s, dest.raw);
    return dest;
}

/*!
 * @brief subtract vector [a] from vector [b] and add result to [dest]
 *
 * applies += operator so dest must be initialized
 *
 * @param[in]  a    first vector
 * @param[in]  b    second vector
 * @param[in]  dest dest += (a - b)
 * @returns         dest
 */
PLAY_CGLM_INLINE
ivec4s
ivec4s_(subadd)(ivec4s a, ivec4s b, ivec4s dest)
{
    ivec4_subadd(a.raw, b.raw, dest.raw);
    return dest;
}

/*!
 * @brief subtract scalar [s] from vector [a] and add result to [dest]
 *
 * applies += operator so dest must be initialized
 *
 * @param[in]  a    first
 * @param[in]  s    scalar
 * @param[in]  dest dest += (a - s)
 * @returns         dest
 */
PLAY_CGLM_INLINE
ivec4s
ivec4s_(subadds)(ivec4s a, int s, ivec4s dest)
{
    ivec4_subadds(a.raw, s, dest.raw);
    return dest;
}

/*!
 * @brief multiply vector [a] with vector [b] and add result to [dest]
 *
 * applies += operator so dest must be initialized
 *
 * @param[in]  a    first vector
 * @param[in]  b    second vector
 * @param[in]  dest dest += (a * b)
 * @returns         dest
 */
PLAY_CGLM_INLINE
ivec4s
ivec4s_(muladd)(ivec4s a, ivec4s b, ivec4s dest)
{
    ivec4_muladd(a.raw, b.raw, dest.raw);
    return dest;
}

/*!
 * @brief multiply vector [a] with scalar [s] and add result to [dest]
 *
 * applies += operator so dest must be initialized
 *
 * @param[in]  a    vector
 * @param[in]  s    scalar
 * @param[in]  dest dest += (a * s)
 * @returns         dest
 */
PLAY_CGLM_INLINE
ivec4s
ivec4s_(muladds)(ivec4s a, int s, ivec4s dest)
{
    ivec4_muladds(a.raw, s, dest.raw);
    return dest;
}

/*!
 * @brief add maximum of vector [a] and vector [b] to vector [dest]
 *
 * applies += operator so dest must be initialized
 *
 * @param[in]  a    first vector
 * @param[in]  b    second vector
 * @param[in]  dest dest += fmax(a, b)
 * @returns         dest
 */
PLAY_CGLM_INLINE
ivec4s
ivec4s_(maxadd)(ivec4s a, ivec4s b, ivec4s dest)
{
    ivec4_maxadd(a.raw, b.raw, dest.raw);
    return dest;
}

/*!
 * @brief add minimum of vector [a] and vector [b] to vector [dest]
 *
 * applies += operator so dest must be initialized
 *
 * @param[in]  a    first vector
 * @param[in]  b    second vector
 * @param[in]  dest dest += fmin(a, b)
 * @returns         dest
 */
PLAY_CGLM_INLINE
ivec4s
ivec4s_(minadd)(ivec4s a, ivec4s b, ivec4s dest)
{
    ivec4_minadd(a.raw, b.raw, dest.raw);
    return dest;
}

/*!
 * @brief subtract vector [a] from vector [b] and subtract result from [dest]
 *
 * applies -= operator so dest must be initialized
 *
 * @param[in]  a    first vector
 * @param[in]  b    second vector
 * @param[in]  dest dest -= (a - b)
 * @returns         dest
 */
PLAY_CGLM_INLINE
ivec4s
ivec4s_(subsub)(ivec4s a, ivec4s b, ivec4s dest)
{
    ivec4_subsub(a.raw, b.raw, dest.raw);
    return dest;
}

/*!
 * @brief subtract scalar [s] from vector [a] and subtract result from [dest]
 *
 * applies -= operator so dest must be initialized
 *
 * @param[in]  a    vector
 * @param[in]  s    scalar
 * @param[in]  dest dest -= (a - s)
 * @returns         dest
 */
PLAY_CGLM_INLINE
ivec4s
ivec4s_(subsubs)(ivec4s a, int s, ivec4s dest)
{
    ivec4_subsubs(a.raw, s, dest.raw);
    return dest;
}

/*!
 * @brief add vector [a] to vector [b] and subtract the result from [dest]
 *
 * applies -= operator so dest must be initialized
 *
 * @param[in]  a    vector
 * @param[in]  b    scalar
 * @param[in]  dest dest -= (a + b)
 * @returns         dest
 */
PLAY_CGLM_INLINE
ivec4s
ivec4s_(addsub)(ivec4s a, ivec4s b, ivec4s dest)
{
    ivec4_addsub(a.raw, b.raw, dest.raw);
    return dest;
}

/*!
 * @brief add scalar [s] to vector [a] and subtract the result from [dest]
 *
 * applies -= operator so dest must be initialized
 *
 * @param[in]  a    vector
 * @param[in]  s    scalar
 * @param[in]  dest dest -= (a + b)
 * @returns         dest
 */
PLAY_CGLM_INLINE
ivec4s
ivec4s_(addsubs)(ivec4s a, int s, ivec4s dest)
{
    ivec4_addsubs(a.raw, s, dest.raw);
    return dest;
}

/*!
 * @brief multiply vector [a] and vector [b] and subtract the result from [dest]
 *
 * applies -= operator so dest must be initialized
 *
 * @param[in]  a    vector
 * @param[in]  b    scalar
 * @param[in]  dest dest -= (a * b)
 * @returns         dest
 */
PLAY_CGLM_INLINE
ivec4s
ivec4s_(mulsub)(ivec4s a, ivec4s b, ivec4s dest)
{
    ivec4_mulsub(a.raw, b.raw, dest.raw);
    return dest;
}

/*!
 * @brief multiply vector [a] with scalar [s] and subtract the result from [dest]
 *
 * applies -= operator so dest must be initialized
 *
 * @param[in]  a    vector
 * @param[in]  s    scalar
 * @param[in]  dest dest -= (a * s)
 * @returns         dest
 */
PLAY_CGLM_INLINE
ivec4s
ivec4s_(mulsubs)(ivec4s a, int s, ivec4s dest)
{
    ivec4_mulsubs(a.raw, s, dest.raw);
    return dest;
}

/*!
 * @brief subtract maximum of vector [a] and vector [b] from vector [dest]
 *
 * applies += operator so dest must be initialized
 *
 * @param[in]  a    first vector
 * @param[in]  b    second vector
 * @param[in]  dest dest -= fmax(a, b)
 * @returns         dest
 */
PLAY_CGLM_INLINE
ivec4s
ivec4s_(maxsub)(ivec4s a, ivec4s b, ivec4s dest)
{
    ivec4_maxsub(a.raw, b.raw, dest.raw);
    return dest;
}

/*!
 * @brief subtract minimum of vector [a] and vector [b] from vector [dest]
 *
 * applies -= operator so dest must be initialized
 *
 * @param[in]  a    first vector
 * @param[in]  b    second vector
 * @param[in]  dest dest -= fmin(a, b)
 * @returns         dest
 */
PLAY_CGLM_INLINE
ivec4s
ivec4s_(minsub)(ivec4s a, ivec4s b, ivec4s dest)
{
    ivec4_minsub(a.raw, b.raw, dest.raw);
    return dest;
}

/*!
 * @brief squared distance between two vectors
 *
 * @param[in] a first vector
 * @param[in] b second vector
 * @return returns squared distance (distance * distance)
 */
PLAY_CGLM_INLINE
int
ivec4s_(distance2)(ivec4s a, ivec4s b)
{
    return ivec4_distance2(a.raw, b.raw);
}

/*!
 * @brief distance between two vectors
 *
 * @param[in] a first vector
 * @param[in] b second vector
 * @return returns distance
 */
PLAY_CGLM_INLINE
float
ivec4s_(distance)(ivec4s a, ivec4s b)
{
    return ivec4_distance(a.raw, b.raw);
}

/*!
 * @brief set each member of dest to greater of vector a and b
 *
 * @param[in]  a    first vector
 * @param[in]  b    second vector
 * @returns         destination
 */
PLAY_CGLM_INLINE
ivec4s
ivec4s_(maxv)(ivec4s a, ivec4s b)
{
    ivec4s r;
    ivec4_maxv(a.raw, b.raw, r.raw);
    return r;
}

/*!
 * @brief set each member of dest to lesser of vector a and b
 *
 * @param[in]  a    first vector
 * @param[in]  b    second vector
 * @returns         destination
 */
PLAY_CGLM_INLINE
ivec4s
ivec4s_(minv)(ivec4s a, ivec4s b)
{
    ivec4s r;
    ivec4_minv(a.raw, b.raw, r.raw);
    return r;
}

/*!
 * @brief clamp each member of [v] between minVal and maxVal (inclusive)
 *
 * @param[in]      v      vector
 * @param[in]      minVal minimum value
 * @param[in]      maxVal maximum value
 * @returns               clamped vector
 */
PLAY_CGLM_INLINE
ivec4s
ivec4s_(clamp)(ivec4s v, int minVal, int maxVal)
{
    ivec4_clamp(v.raw, minVal, maxVal);
    return v;
}

/*!
 * @brief absolute value of v
 *
 * @param[in]	v	vector
 * @returns     destination
 */
PLAY_CGLM_INLINE
ivec4s
ivec4s_(abs)(ivec4s v)
{
    ivec4s r;
    ivec4_abs(v.raw, r.raw);
    return r;
}

#endif /* civec4s_h */

/*** End of inlined file: ivec4.h ***/


/*** Start of inlined file: mat2.h ***/
/*
 Macros:
   PLAY_CGLM_MAT2_IDENTITY_INIT
   PLAY_CGLM_MAT2_ZERO_INIT
   PLAY_CGLM_MAT2_IDENTITY
   PLAY_CGLM_MAT2_ZERO

 Functions:
   PLAY_CGLM_INLINE mat2s mat2s_make(const float * __restrict src);
   PLAY_CGLM_INLINE mat2s mat2s_identity(void)
   PLAY_CGLM_INLINE void  mat2s_identity_array(mat2 * restrict mats, size_t count)
   PLAY_CGLM_INLINE mat2s mat2s_zero(void)
   PLAY_CGLM_INLINE mat2s mat2s_mul(mat2 m1, mat2 m2)
   PLAY_CGLM_INLINE vec2s mat2s_mulv(mat2 m, vec2 v)
   PLAY_CGLM_INLINE mat2s mat2s_transpose(mat2 m)
   PLAY_CGLM_INLINE mat2s mat2s_scale(mat2 m, float s)
   PLAY_CGLM_INLINE mat2s mat2s_inv(mat2 m)
   PLAY_CGLM_INLINE mat2s mat2s_swap_col(mat2 mat, int col1, int col2)
   PLAY_CGLM_INLINE mat2s mat2s_swap_row(mat2 mat, int row1, int row2)
   PLAY_CGLM_INLINE float mat2s_det(mat2 m)
   PLAY_CGLM_INLINE float mat2s_trace(mat2 m)
   PLAY_CGLM_INLINE float mat2s_rmc(vec2 r, mat2 m, vec2 c)
 */

#ifndef cmat2s_h
#define cmat2s_h

/* api definition */
#define mat2s_(NAME) PLAY_CGLM_STRUCTAPI(mat2, NAME)

#define PLAY_CGLM_S_MAT2_IDENTITY_INIT {PLAY_CGLM_MAT2_IDENTITY_INIT}
#define PLAY_CGLM_S_MAT2_ZERO_INIT     {PLAY_CGLM_MAT2_ZERO_INIT}

/* for C only */
#define PLAY_CGLM_S_MAT2_IDENTITY ((mat2s)PLAY_CGLM_S_MAT2_IDENTITY_INIT)
#define PLAY_CGLM_S_MAT2_ZERO     ((mat2s)PLAY_CGLM_S_MAT2_ZERO_INIT)

/*!
 * @brief Returns mat2s (r) from pointer (src).
 *
 * @param[in]   src pointer to an array of floats
 * @return[out] r   constructed mat2s from raw pointer
 */
PLAY_CGLM_INLINE
mat2s
mat2s_(make)(const float * __restrict src)
{
    mat2s r;
    mat2_make(src, r.raw);
    return r;
}

/*!
 * @brief Return a identity mat2s (r).
 *
 *        The same thing may be achieved with either of bellow methods,
 *        but it is more easy to do that with this func especially for members
 *        e.g. mat2_identity(aStruct->aMatrix);
 *
 * @code
 * mat2_copy(PLAY_CGLM_MAT2_IDENTITY, mat); // C only
 *
 * // or
 * mat2 mat = PLAY_CGLM_MAT2_IDENTITY_INIT;
 * @endcode
 *
 * @return[out] r constructed mat2s from raw pointer
 */
PLAY_CGLM_INLINE
mat2s
mat2s_(identity)(void)
{
    mat2s r;
    mat2_identity(r.raw);
    return r;
}

/*!
 * @brief Given an array of mat2s’s (mats) make each matrix an identity matrix.
 *
 * @param[in, out] mats  Array of mat2s’s (must be aligned (16/32) if alignment is not disabled)
 * @param[in]      count Array size of mats or number of matrices
 */
PLAY_CGLM_INLINE
void
mat2s_(identity_array)(mat2s * __restrict mats, size_t count)
{
    PLAY_CGLM_ALIGN_MAT mat2s t = PLAY_CGLM_S_MAT2_IDENTITY_INIT;
    size_t i;

    for (i = 0; i < count; i++)
    {
        mat2_copy(t.raw, mats[i].raw);
    }
}

/*!
 * @brief Return zero'd out mat2 (r).
 *
 * @return[out] r constructed mat2s from raw pointer
 */
PLAY_CGLM_INLINE
mat2s
mat2s_(zero)(void)
{
    mat2s r;
    mat2_zero(r.raw);
    return r;
}

/*!
 * @brief Multiply mat2 (m1) by mat2 (m2) and return in mat2s (r)
 *
 *        m1 and m2 matrices can be the same matrix, it is possible to write this:
 *
 * @code
 * mat2 m = PLAY_CGLM_MAT2_IDENTITY_INIT;
 * mat2s r = mat2s_mul(m, m);
 * @endcode
 *
 * @param[in]   m1 mat2s (left)
 * @param[in]   m2 mat2s (right)
 * @return[out] r  constructed mat2s from raw pointers
 */
PLAY_CGLM_INLINE
mat2s
mat2s_(mul)(mat2s m1, mat2s m2)
{
    mat2s r;
    mat2_mul(m1.raw, m2.raw, r.raw);
    return r;
}

/*
 * @brief Multiply mat2s (m) by vec2s (v) and return in vec2s (r).
 *
 * @param[in]   m mat2s (left)
 * @param[in]   v vec2s (right, column vector)
 * @return[out] r constructed vec2s from raw pointers
 */
PLAY_CGLM_INLINE
vec2s
mat2s_(mulv)(mat2s m, vec2s v)
{
    vec2s r;
    mat2_mulv(m.raw, v.raw, r.raw);
    return r;
}

/*!
 * @brief Transpose mat2s (m) and store result in the same matrix.
 *
 * @param[in]   m mat2s (src)
 * @return[out] m constructed mat2s from raw pointers
 */
PLAY_CGLM_INLINE
mat2s
mat2s_(transpose)(mat2s m)
{
    mat2_transpose(m.raw);
    return m;
}

/*!
 * @brief Multiply mat2s (m) by scalar constant (s)
 *
 * @param[in]   m mat2s (src)
 * @param[in]   s scalar value
 * @return[out] m constructed mat2s from raw pointers
 */
PLAY_CGLM_INLINE
mat2s
mat2s_(scale)(mat2s m, float s)
{
    mat2_scale(m.raw, s);
    return m;
}

/*!
 * @brief Inverse mat2s (m) and return in mat2s (r).
 *
 * @param[in]   m mat2s (left, src)
 * @return[out] r constructed mat2s from raw pointers
 */
PLAY_CGLM_INLINE
mat2s
mat2s_(inv)(mat2s m)
{
    mat2s r;
    mat2_inv(m.raw, r.raw);
    return r;
}

/*!
 * @brief Swap two columns in mat2s (mat) and store in same matrix.
 *
 * @param[in]   mat  mat2s
 * @param[in]   col1 column 1 array index
 * @param[in]   col2 column 2 array index
 * @return[out] mat  constructed mat2s from raw pointers columns swapped
 */
PLAY_CGLM_INLINE
mat2s
mat2s_(swap_col)(mat2s mat, int col1, int col2)
{
    mat2_swap_col(mat.raw, col1, col2);
    return mat;
}

/*!
 * @brief Swap two rows in mat2s (mat) and store in same matrix.
 *
 * @param[in]   mat  mat2s
 * @param[in]   row1 row 1 array index
 * @param[in]   row2 row 2 array index
 * @return[out] mat  constructed mat2s from raw pointers rows swapped
 */
PLAY_CGLM_INLINE
mat2s
mat2s_(swap_row)(mat2s mat, int row1, int row2)
{
    mat2_swap_row(mat.raw, row1, row2);
    return mat;
}

/*!
 * @brief Returns mat2 determinant.
 *
 * @param[in] m mat2 (src)
 *
 * @return[out] mat2s raw pointers determinant (float)
 */
PLAY_CGLM_INLINE
float
mat2s_(det)(mat2s m)
{
    return mat2_det(m.raw);
}

/*!
 * @brief Returns trace of matrix. Which is:
 *
 *        The sum of the elements on the main diagonal from
 *        upper left corner to the bottom right corner.
 *
 * @param[in] m mat2 (m)
 *
 * @return[out] mat2s raw pointers trace (float)
 */
PLAY_CGLM_INLINE
float
mat2s_(trace)(mat2s m)
{
    return mat2_trace(m.raw);
}

/*!
 * @brief Helper for  R (row vector) * M (matrix) * C (column vector)
 *
 *        rmc stands for Row * Matrix * Column
 *
 *        the result is scalar because M * C = ResC (1x2, column vector),
 *        then if you take the dot_product(R (2x1), ResC (1x2)) = scalar value.
 *
 * @param[in] r vec2s (2x1, row vector)
 * @param[in] m mat2s (2x2, matrix)
 * @param[in] c vec2s (1x2, column vector)
 *
 * @return[out] Scalar value (float, 1x1)
 */
PLAY_CGLM_INLINE
float
mat2s_(rmc)(vec2s r, mat2s m, vec2s c)
{
    return mat2_rmc(r.raw, m.raw, c.raw);
}

#endif /* cmat2s_h */

/*** End of inlined file: mat2.h ***/


/*** Start of inlined file: mat2x3.h ***/
/*
 Macros:
   PLAY_CGLM_S_MAT2X3_ZERO_INIT
   PLAY_CGLM_S_MAT2X3_ZERO

 Functions:
   PLAY_CGLM_INLINE mat2x3s mat2x3s_zero(void);
   PLAY_CGLM_INLINE mat2x3s mat2x3s_make(const float * __restrict src);
   PLAY_CGLM_INLINE mat2s   mat2x3s_mul(mat2x3s m1, mat3x2s m2);
   PLAY_CGLM_INLINE vec3s   mat2x3s_mulv(mat2x3s m, vec2s v);
   PLAY_CGLM_INLINE mat3x2s mat2x3s_transpose(mat2x3s m);
   PLAY_CGLM_INLINE mat2x3s mat2x3s_scale(mat2x3s m, float s);
 */

#ifndef cmat2x3s_h
#define cmat2x3s_h

/* api definition */
#define mat2x3s_(NAME) PLAY_CGLM_STRUCTAPI(mat2x3, NAME)

#define PLAY_CGLM_S_MAT2X3_ZERO_INIT {PLAY_CGLM_MAT2X3_ZERO_INIT}

/* for C only */
#define PLAY_CGLM_S_MAT2X3_ZERO ((mat2x3s)PLAY_CGLM_S_MAT2X3_ZERO_INIT)

/*!
 * @brief Zero out the mat2x3s (dest).
 *
 * @return[out] dest constructed mat2x3s from raw pointer
 */
PLAY_CGLM_INLINE
mat2x3s
mat2x3s_(zero)(void)
{
    mat2x3s dest;
    mat2x3_zero(dest.raw);
    return dest;
}

/*!
 * @brief Create mat2x3s (dest) from pointer (src).
 *
 * @param[in]   src  pointer to an array of floats
 * @return[out] dest constructed mat2x3s from raw pointer
 */
PLAY_CGLM_INLINE
mat2x3s
mat2x3s_(make)(const float * __restrict src)
{
    mat2x3s dest;
    mat2x3_make(src, dest.raw);
    return dest;
}

/*!
 * @brief Multiply mat2x3s (m1) by mat3x2s (m2) and store in mat3s (dest).
 *
 * @code
 * r = mat2x3s_mul(mat2x3s, mat3x2s);
 * @endcode
 *
 * @param[in]   m1   mat2x3s (left)
 * @param[in]   m2   mat3x2s (right)
 * @return[out] dest constructed mat3s from raw pointers
 */
PLAY_CGLM_INLINE
mat3s
mat2x3s_(mul)(mat2x3s m1, mat3x2s m2)
{
    mat3s dest;
    mat2x3_mul(m1.raw, m2.raw, dest.raw);
    return dest;
}

/*!
 * @brief Multiply mat2x3s (m) by vec2s (v) and store in vec3s (dest).
 *
 * @param[in]   m    mat2x3s (left)
 * @param[in]   v    vec2s (right, column vector)
 * @return[out] dest constructed vec3s from raw pointers
 */
PLAY_CGLM_INLINE
vec3s
mat2x3s_(mulv)(mat2x3s m, vec2s v)
{
    vec3s dest;
    mat2x3_mulv(m.raw, v.raw, dest.raw);
    return dest;
}

/*!
 * @brief Transpose mat2x3s (m) and store in mat3x2s (dest).
 *
 * @param[in]   m    mat2x3s (left)
 * @return[out] dest constructed mat3x2s from raw pointers
 */
PLAY_CGLM_INLINE
mat3x2s
mat2x3s_(transpose)(mat2x3s m)
{
    mat3x2s dest;
    mat2x3_transpose(m.raw, dest.raw);
    return dest;
}

/*!
 * @brief Multiply mat2x3s (m) by scalar constant (s).
 *
 * @param[in, out] m mat2x3 (src, dest)
 * @param[in]      s float (scalar)
 */
PLAY_CGLM_INLINE
mat2x3s
mat2x3s_(scale)(mat2x3s m, float s)
{
    mat2x3_scale(m.raw, s);
    return m;
}

#endif /* cmat2x3s_h */

/*** End of inlined file: mat2x3.h ***/


/*** Start of inlined file: mat2x4.h ***/
/*
 Macros:
   PLAY_CGLM_S_MAT2X4_ZERO_INIT
   PLAY_CGLM_S_MAT2X4_ZERO

 Functions:
   PLAY_CGLM_INLINE mat2x4s mat2x4s_zero(void);
   PLAY_CGLM_INLINE mat2x4s mat2x4s_make(const float * __restrict src);
   PLAY_CGLM_INLINE mat2s   mat2x4s_mul(mat2x4s m1, mat4x2s m2);
   PLAY_CGLM_INLINE vec4s   mat2x4s_mulv(mat2x4s m, vec2s v);
   PLAY_CGLM_INLINE mat4x2s mat2x4s_transpose(mat2x4s m);
   PLAY_CGLM_INLINE mat2x4s mat2x4s_scale(mat2x4s m, float s);
 */

#ifndef cmat2x4s_h
#define cmat2x4s_h

/* api definition */
#define mat2x4s_(NAME) PLAY_CGLM_STRUCTAPI(mat2x4, NAME)

#define PLAY_CGLM_S_MAT2X4_ZERO_INIT {PLAY_CGLM_MAT2X4_ZERO_INIT}

/* for C only */
#define PLAY_CGLM_S_MAT2X4_ZERO ((mat2x4s)PLAY_CGLM_S_MAT2X4_ZERO_INIT)

/*!
 * @brief Zero out the mat2x4s (dest).
 *
 * @return[out] dest constructed mat2x4s from raw pointer
 */
PLAY_CGLM_INLINE
mat2x4s
mat2x4s_(zero)(void)
{
    mat2x4s dest;
    mat2x4_zero(dest.raw);
    return dest;
}

/*!
 * @brief Create mat2x4s (dest) from pointer (src).
 *
 * @param[in]   src  pointer to an array of floats
 * @return[out] dest constructed mat2x4s from raw pointer
 */
PLAY_CGLM_INLINE
mat2x4s
mat2x4s_(make)(const float * __restrict src)
{
    mat2x4s dest;
    mat2x4_make(src, dest.raw);
    return dest;
}

/*!
 * @brief Multiply mat2x4s (m1) by mat4x2s (m2) and store in mat4s (dest).
 *
 * @code
 * r = mat2x4s_mul(mat2x4s, mat4x2s);
 * @endcode
 *
 * @param[in]   m1   mat2x4s (left)
 * @param[in]   m2   mat4x2s (right)
 * @return[out] dest constructed mat4s from raw pointers
 */
PLAY_CGLM_INLINE
mat4s
mat2x4s_(mul)(mat2x4s m1, mat4x2s m2)
{
    mat4s dest;
    mat2x4_mul(m1.raw, m2.raw, dest.raw);
    return dest;
}

/*!
 * @brief Multiply mat2x4s (m) by vec2s (v) and store in vec4s (dest).
 *
 * @param[in]   m    mat2x4s (left)
 * @param[in]   v    vec2s (right, column vector)
 * @return[out] dest constructed vec4s from raw pointers
 */
PLAY_CGLM_INLINE
vec4s
mat2x4s_(mulv)(mat2x4s m, vec2s v)
{
    vec4s dest;
    mat2x4_mulv(m.raw, v.raw, dest.raw);
    return dest;
}

/*!
 * @brief Transpose mat2x4s (m) and store in mat4x2s (dest).
 *
 * @param[in]   m    mat2x4s (left)
 * @return[out] dest constructed mat4x2s from raw pointers
 */
PLAY_CGLM_INLINE
mat4x2s
mat2x4s_(transpose)(mat2x4s m)
{
    mat4x2s dest;
    mat2x4_transpose(m.raw, dest.raw);
    return dest;
}

/*!
 * @brief Multiply mat2x4s (m) by scalar constant (s).
 *
 * @param[in, out] m mat2x4s (src, dest)
 * @param[in]      s float (scalar)
 */
PLAY_CGLM_INLINE
mat2x4s
mat2x4s_(scale)(mat2x4s m, float s)
{
    mat2x4_scale(m.raw, s);
    return m;
}

#endif /* cmat2x4s_h */

/*** End of inlined file: mat2x4.h ***/


/*** Start of inlined file: mat3.h ***/
/*
 Macros:
   PLAY_CGLM_S_MAT3_IDENTITY_INIT
   PLAY_CGLM_S_MAT3_ZERO_INIT
   PLAY_CGLM_S_MAT3_IDENTITY
   PLAY_CGLM_S_MAT3_ZERO

 Functions:
   PLAY_CGLM_INLINE mat3s  mat3s_copy(mat3s mat);
   PLAY_CGLM_INLINE mat3s  mat3s_identity(void);
   PLAY_CGLM_INLINE void   mat3s_identity_array(mat3s * __restrict mat, size_t count);
   PLAY_CGLM_INLINE mat3s  mat3s_zero(void);
   PLAY_CGLM_INLINE mat3s  mat3s_mul(mat3s m1, mat3s m2);
   PLAY_CGLM_INLINE ma3s   mat3s_transpose(mat3s m);
   PLAY_CGLM_INLINE vec3s  mat3s_mulv(mat3s m, vec3s v);
   PLAY_CGLM_INLINE float  mat3s_trace(mat3s m);
   PLAY_CGLM_INLINE versor mat3s_quat(mat3s m);
   PLAY_CGLM_INLINE mat3s  mat3s_scale(mat3s m, float s);
   PLAY_CGLM_INLINE float  mat3s_det(mat3s mat);
   PLAY_CGLM_INLINE mat3s  mat3s_inv(mat3s mat);
   PLAY_CGLM_INLINE mat3s  mat3s_swap_col(mat3s mat, int col1, int col2);
   PLAY_CGLM_INLINE mat3s  mat3s_swap_row(mat3s mat, int row1, int row2);
   PLAY_CGLM_INLINE float  mat3s_rmc(vec3s r, mat3s m, vec3s c);
   PLAY_CGLM_INLINE mat3s  mat3s_make(const float * __restrict src);
   PLAY_CGLM_INLINE mat3s  mat3s_textrans(float sx, float sy, float rot, float tx, float ty);
 */




/* api definition */
#define mat3s_(NAME) PLAY_CGLM_STRUCTAPI(mat3, NAME)

#define PLAY_CGLM_S_MAT3_IDENTITY_INIT  {PLAY_CGLM_MAT3_IDENTITY_INIT}
#define PLAY_CGLM_S_MAT3_ZERO_INIT      {PLAY_CGLM_MAT3_ZERO_INIT}

/* for C only */
#define PLAY_CGLM_S_MAT3_IDENTITY ((mat3s)PLAY_CGLM_S_MAT3_IDENTITY_INIT)
#define PLAY_CGLM_S_MAT3_ZERO     ((mat3s)PLAY_CGLM_S_MAT3_ZERO_INIT)

/*!
 * @brief copy all members of [mat] to [dest]
 *
 * @param[in]  mat  source
 * @returns         destination
 */
PLAY_CGLM_INLINE
mat3s
mat3s_(copy)(mat3s mat)
{
    mat3s r;
    mat3_copy(mat.raw, r.raw);
    return r;
}

/*!
 * @brief make given matrix identity. It is identical with below,
 *        but it is more easy to do that with this func especially for members
 *        e.g. mat3_identity(aStruct->aMatrix);
 *
 * @code
 * mat3_copy(PLAY_CGLM_MAT3_IDENTITY, mat); // C only
 *
 * // or
 * mat3 mat = PLAY_CGLM_MAT3_IDENTITY_INIT;
 * @endcode
 *
 * @returns  destination
 */
PLAY_CGLM_INLINE
mat3s
mat3s_(identity)(void)
{
    mat3s r;
    mat3_identity(r.raw);
    return r;
}

/*!
 * @brief make given matrix array's each element identity matrix
 *
 * @param[in, out]  mat   matrix array (must be aligned (16/32)
 *                        if alignment is not disabled)
 *
 * @param[in]       count count of matrices
 */
PLAY_CGLM_INLINE
void
mat3s_(identity_array)(mat3s * __restrict mat, size_t count)
{
    PLAY_CGLM_ALIGN_MAT mat3s t = PLAY_CGLM_S_MAT3_IDENTITY_INIT;
    size_t i;

    for (i = 0; i < count; i++)
    {
        mat3_copy(t.raw, mat[i].raw);
    }
}

/*!
 * @brief make given matrix zero.
 *
 * @returns  matrix
 */
PLAY_CGLM_INLINE
mat3s
mat3s_(zero)(void)
{
    mat3s r;
    mat3_zero(r.raw);
    return r;
}

/*!
 * @brief multiply m1 and m2 to dest
 *
 * m1, m2 and dest matrices can be same matrix, it is possible to write this:
 *
 * @code
 * mat3 m = PLAY_CGLM_MAT3_IDENTITY_INIT;
 * r = mat3s_mul(m, m);
 * @endcode
 *
 * @param[in]  m1   left matrix
 * @param[in]  m2   right matrix
 * @returns destination matrix
 */
PLAY_CGLM_INLINE
mat3s
mat3s_(mul)(mat3s m1, mat3s m2)
{
    mat3s r;
    mat3_mul(m1.raw, m2.raw, r.raw);
    return r;
}

/*!
 * @brief transpose mat3 and store result in same matrix
 *
 * @param[in, out] m source and dest
 */
PLAY_CGLM_INLINE
mat3s
mat3s_(transpose)(mat3s m)
{
    mat3_transpose(m.raw);
    return m;
}

/*!
 * @brief multiply mat3 with vec3 (column vector) and store in dest vector
 *
 * @param[in]  m    mat3 (left)
 * @param[in]  v    vec3 (right, column vector)
 * @returns         vec3 (result, column vector)
 */
PLAY_CGLM_INLINE
vec3s
mat3s_(mulv)(mat3s m, vec3s v)
{
    vec3s r;
    mat3_mulv(m.raw, v.raw, r.raw);
    return r;
}

/*!
 * @brief trace of matrix
 *
 * sum of the elements on the main diagonal from upper left to the lower right
 *
 * @param[in]  m matrix
 */
PLAY_CGLM_INLINE
float
mat3s_(trace)(mat3s m)
{
    return mat3_trace(m.raw);
}

/*!
 * @brief convert mat3 to quaternion
 *
 * @param[in]  m    rotation matrix
 * @returns         destination quaternion
 */
PLAY_CGLM_INLINE
versors
mat3s_(quat)(mat3s m)
{
    versors r;
    mat3_quat(m.raw, r.raw);
    return r;
}

/*!
 * @brief scale (multiply with scalar) matrix
 *
 * multiply matrix with scalar
 *
 * @param[in]      m matrix
 * @param[in]      s scalar
 * @returns          scaled matrix
 */
PLAY_CGLM_INLINE
mat3s
mat3s_(scale)(mat3s m, float s)
{
    mat3_scale(m.raw, s);
    return m;
}

/*!
 * @brief mat3 determinant
 *
 * @param[in] mat matrix
 *
 * @return determinant
 */
PLAY_CGLM_INLINE
float
mat3s_(det)(mat3s mat)
{
    return mat3_det(mat.raw);
}

/*!
 * @brief inverse mat3 and store in dest
 *
 * @param[in]  mat  matrix
 * @returns         inverse matrix
 */
PLAY_CGLM_INLINE
mat3s
mat3s_(inv)(mat3s mat)
{
    mat3s r;
    mat3_inv(mat.raw, r.raw);
    return r;
}

/*!
 * @brief swap two matrix columns
 *
 * @param[in]     mat  matrix
 * @param[in]     col1 col1
 * @param[in]     col2 col2
 * @returns            matrix
 */
PLAY_CGLM_INLINE
mat3s
mat3s_(swap_col)(mat3s mat, int col1, int col2)
{
    mat3_swap_col(mat.raw, col1, col2);
    return mat;
}

/*!
 * @brief swap two matrix rows
 *
 * @param[in]     mat  matrix
 * @param[in]     row1 row1
 * @param[in]     row2 row2
 * @returns            matrix
 */
PLAY_CGLM_INLINE
mat3s
mat3s_(swap_row)(mat3s mat, int row1, int row2)
{
    mat3_swap_row(mat.raw, row1, row2);
    return mat;
}

/*!
 * @brief helper for  R (row vector) * M (matrix) * C (column vector)
 *
 * rmc stands for Row * Matrix * Column
 *
 * the result is scalar because R * M = Matrix1x3 (row vector),
 * then Matrix1x3 * Vec3 (column vector) = Matrix1x1 (Scalar)
 *
 * @param[in]  r   row vector or matrix1x3
 * @param[in]  m   matrix3x3
 * @param[in]  c   column vector or matrix3x1
 *
 * @return scalar value e.g. Matrix1x1
 */
PLAY_CGLM_INLINE
float
mat3s_(rmc)(vec3s r, mat3s m, vec3s c)
{
    return mat3_rmc(r.raw, m.raw, c.raw);
}

/*!
 * @brief Create mat3 matrix from pointer
 *
 * @param[in]  src  pointer to an array of floats
 * @return constructed matrix from raw pointer
 */
PLAY_CGLM_INLINE
mat3s
mat3s_(make)(const float * __restrict src)
{
    mat3s r;
    mat3_make(src, r.raw);
    return r;
}

/*!
 * @brief Create mat3 matrix from texture transform parameters
 *
 * @param[in]  sx  scale x
 * @param[in]  sy  scale y
 * @param[in]  rot rotation in radians CCW/RH
 * @param[in]  tx  translate x
 * @param[in]  ty  translate y
 * @return texture transform matrix
 */
PLAY_CGLM_INLINE
mat3s
mat3s_(textrans)(float sx, float sy, float rot, float tx, float ty)
{
    mat3s r;
    mat3_textrans(sx, sy, rot, tx, ty, r.raw);
    return r;
}



/*** End of inlined file: mat3.h ***/


/*** Start of inlined file: mat3x2.h ***/
/*
 Macros:
   PLAY_CGLM_S_MAT3X2_ZERO_INIT
   PLAY_CGLM_S_MAT3X2_ZERO

 Functions:
   PLAY_CGLM_INLINE mat3x2s mat3x2s_zero(void);
   PLAY_CGLM_INLINE mat3x2s mat3x2s_make(const float * __restrict src);
   PLAY_CGLM_INLINE mat2s   mat3x2s_mul(mat3x2s m1, mat2x3s m2);
   PLAY_CGLM_INLINE vec2s   mat3x2s_mulv(mat3x2s m, vec3s v);
   PLAY_CGLM_INLINE mat2x3s mat3x2s_transpose(mat3x2s m);
   PLAY_CGLM_INLINE mat3x2s mat3x2s_scale(mat3x2s m, float s);
 */

#ifndef cmat3x2s_h
#define cmat3x2s_h

/* api definition */
#define mat3x2s_(NAME) PLAY_CGLM_STRUCTAPI(mat3x2, NAME)

#define PLAY_CGLM_S_MAT3X2_ZERO_INIT {PLAY_CGLM_MAT3X2_ZERO_INIT}

/* for C only */
#define PLAY_CGLM_S_MAT3X2_ZERO ((mat3x2s)PLAY_CGLM_S_MAT3X2_ZERO_INIT)

/*!
 * @brief Zero out the mat3x2s (dest).
 *
 * @return[out] dest constructed mat3x2s from raw pointer
 */
PLAY_CGLM_INLINE
mat3x2s
mat3x2s_(zero)(void)
{
    mat3x2s dest;
    mat3x2_zero(dest.raw);
    return dest;
}

/*!
 * @brief Create mat3x2s (dest) from pointer (src).
 *
 * @param[in]   src  pointer to an array of floats
 * @return[out] dest constructed mat3x2s from raw pointer
 */
PLAY_CGLM_INLINE
mat3x2s
mat3x2s_(make)(const float * __restrict src)
{
    mat3x2s dest;
    mat3x2_make(src, dest.raw);
    return dest;
}

/*!
 * @brief Multiply mat3x2s (m1) by mat2x3s (m2) and store in mat2s (dest).
 *
 * @code
 * r = mat3x2s_mul(mat3x2s, mat2x3s);
 * @endcode
 *
 * @param[in]   m1   mat3x2s (left)
 * @param[in]   m2   mat2x3s (right)
 * @return[out] dest constructed mat2s from raw pointers
 */
PLAY_CGLM_INLINE
mat2s
mat3x2s_(mul)(mat3x2s m1, mat2x3s m2)
{
    mat2s dest;
    mat3x2_mul(m1.raw, m2.raw, dest.raw);
    return dest;
}

/*!
 * @brief Multiply mat3x2s (m) by vec3s (v) and store in vec2s (dest).
 *
 * @param[in]   m    mat3x2s (left)
 * @param[in]   v    vec3s (right, column vector)
 * @return[out] dest constructed vec2s from raw pointers
 */
PLAY_CGLM_INLINE
vec2s
mat3x2s_(mulv)(mat3x2s m, vec3s v)
{
    vec2s dest;
    mat3x2_mulv(m.raw, v.raw, dest.raw);
    return dest;
}

/*!
 * @brief Transpose mat3x2s (m) and store in mat2x3s (dest).
 *
 * @param[in]   m    mat3x2s (left)
 * @return[out] dest constructed mat2x3s from raw pointers
 */
PLAY_CGLM_INLINE
mat2x3s
mat3x2s_(transpose)(mat3x2s m)
{
    mat2x3s dest;
    mat3x2_transpose(m.raw, dest.raw);
    return dest;
}

/*!
 * @brief Multiply mat3x2s (m) by scalar constant (s).
 *
 * @param[in, out] m mat3x2s (src, dest)
 * @param[in]      s float (scalar)
 */
PLAY_CGLM_INLINE
mat3x2s
mat3x2s_(scale)(mat3x2s m, float s)
{
    mat3x2_scale(m.raw, s);
    return m;
}

#endif /* cmat3x2s_h */

/*** End of inlined file: mat3x2.h ***/


/*** Start of inlined file: mat3x4.h ***/
/*
 Macros:
   PLAY_CGLM_S_MAT3X4_ZERO_INIT
   PLAY_CGLM_S_MAT3X4_ZERO

 Functions:
   PLAY_CGLM_INLINE mat3x4s mat3x4s_zero(void);
   PLAY_CGLM_INLINE mat3x4s mat3x4s_make(const float * __restrict src);
   PLAY_CGLM_INLINE mat4s   mat3x4s_mul(mat3x4s m1, mat4x3s m2);
   PLAY_CGLM_INLINE vec4s   mat3x4s_mulv(mat3x4s m, vec3s v);
   PLAY_CGLM_INLINE mat4x3s mat3x4s_transpose(mat3x4s m);
   PLAY_CGLM_INLINE mat3x4s mat3x4s_scale(mat3x4s m, float s);
 */

#ifndef cmat3x4s_h
#define cmat3x4s_h

/* api definition */
#define mat3x4s_(NAME) PLAY_CGLM_STRUCTAPI(mat3x4, NAME)

#define PLAY_CGLM_S_MAT3X4_ZERO_INIT {PLAY_CGLM_MAT3X4_ZERO_INIT}

/* for C only */
#define PLAY_CGLM_S_MAT3X4_ZERO ((mat3x4s)PLAY_CGLM_S_MAT3X4_ZERO_INIT)

/*!
 * @brief Zero out the mat3x4s (dest).
 *
 * @return[out] dest constructed mat3x4s from raw pointer
 */
PLAY_CGLM_INLINE
mat3x4s
mat3x4s_(zero)(void)
{
    mat3x4s dest;
    mat3x4_zero(dest.raw);
    return dest;
}

/*!
 * @brief Create mat3x4s (dest) from pointer (src).
 *
 * @param[in]   src  pointer to an array of floats
 * @return[out] dest constructed mat3x4s from raw pointer
 */
PLAY_CGLM_INLINE
mat3x4s
mat3x4s_(make)(const float * __restrict src)
{
    mat3x4s dest;
    mat3x4_make(src, dest.raw);
    return dest;
}

/*!
 * @brief Multiply mat3x4s (m1) by mat4x3s (m2) and store in mat4s (dest).
 *
 * @code
 * r = mat3x4s_mul(mat3x4s, mat4x3s);
 * @endcode
 *
 * @param[in]   m1   mat3x4s (left)
 * @param[in]   m2   mat4x3s (right)
 * @return[out] dest constructed mat4s from raw pointers
 */
PLAY_CGLM_INLINE
mat4s
mat3x4s_(mul)(mat3x4s m1, mat4x3s m2)
{
    mat4s dest;
    mat3x4_mul(m1.raw, m2.raw, dest.raw);
    return dest;
}

/*!
 * @brief Multiply mat3x4s (m) by vec3s (v) and store in vec4s (dest).
 *
 * @param[in]   m    mat3x4s (left)
 * @param[in]   v    vec3s (right, column vector)
 * @return[out] dest constructed vec4s from raw pointers
 */
PLAY_CGLM_INLINE
vec4s
mat3x4s_(mulv)(mat3x4s m, vec3s v)
{
    vec4s dest;
    mat3x4_mulv(m.raw, v.raw, dest.raw);
    return dest;
}

/*!
 * @brief Transpose mat3x4s (m) and store in mat4x3s (dest).
 *
 * @param[in]   m    mat3x4s (left)
 * @return[out] dest constructed mat4x3s from raw pointers
 */
PLAY_CGLM_INLINE
mat4x3s
mat3x4s_(transpose)(mat3x4s m)
{
    mat4x3s dest;
    mat3x4_transpose(m.raw, dest.raw);
    return dest;
}

/*!
 * @brief Multiply mat3x4s (m) by scalar constant (s).
 *
 * @param[in, out] m mat3x4s (src, dest)
 * @param[in]      s float (scalar)
 */
PLAY_CGLM_INLINE
mat3x4s
mat3x4s_(scale)(mat3x4s m, float s)
{
    mat3x4_scale(m.raw, s);
    return m;
}

#endif /* cmat3x4s_h */

/*** End of inlined file: mat3x4.h ***/


/*** Start of inlined file: mat4.h ***/
/*!
 * Most of functions in this header are optimized manually with SIMD
 * if available. You dont need to call/incude SIMD headers manually
 */

/*
 Macros:
   PLAY_CGLM_S_MAT4_IDENTITY_INIT
   PLAY_CGLM_S_MAT4_ZERO_INIT
   PLAY_CGLM_S_MAT4_IDENTITY
   PLAY_CGLM_S_MAT4_ZERO

 Functions:
   PLAY_CGLM_INLINE mat4s   mat4s_ucopy(mat4s mat);
   PLAY_CGLM_INLINE mat4s   mat4s_copy(mat4s mat);
   PLAY_CGLM_INLINE mat4s   mat4s_identity(void);
   PLAY_CGLM_INLINE void    mat4s_identity_array(mat4s * __restrict mat, size_t count);
   PLAY_CGLM_INLINE mat4s   mat4s_zero(void);
   PLAY_CGLM_INLINE mat3s   mat4s_pick3(mat4s mat);
   PLAY_CGLM_INLINE mat3s   mat4s_pick3t(mat4s mat);
   PLAY_CGLM_INLINE mat4s   mat4s_ins3(mat3s mat, mat4s dest);
   PLAY_CGLM_INLINE mat4s   mat4s_mul(mat4s m1, mat4s m2);
   PLAY_CGLM_INLINE mat4s   mat4s_mulN(mat4s * __restrict matrices[], uint32_t len);
   PLAY_CGLM_INLINE vec4s   mat4s_mulv(mat4s m, vec4s v);
   PLAY_CGLM_INLINE float   mat4s_trace(mat4s m);
   PLAY_CGLM_INLINE float   mat4s_trace3(mat4s m);
   PLAY_CGLM_INLINE versors mat4s_quat(mat4s m);
   PLAY_CGLM_INLINE vec3s   mat4s_mulv3(mat4s m, vec3s v, float last);
   PLAY_CGLM_INLINE mat4s   mat4s_transpose(mat4s m);
   PLAY_CGLM_INLINE mat4s   mat4s_scale_p(mat4s m, float s);
   PLAY_CGLM_INLINE mat4s   mat4s_scale(mat4s m, float s);
   PLAY_CGLM_INLINE float   mat4s_det(mat4s mat);
   PLAY_CGLM_INLINE mat4s   mat4s_inv(mat4s mat);
   PLAY_CGLM_INLINE mat4s   mat4s_inv_fast(mat4s mat);
   PLAY_CGLM_INLINE mat4s   mat4s_swap_col(mat4s mat, int col1, int col2);
   PLAY_CGLM_INLINE mat4s   mat4s_swap_row(mat4s mat, int row1, int row2);
   PLAY_CGLM_INLINE float   mat4s_rmc(vec4s r, mat4s m, vec4s c);
   PLAY_CGLM_INLINE mat4s   mat4s_make(const float * __restrict src);
   PLAY_CGLM_INLINE mat4s   mat4s_textrans(float sx, float sy, float rot, float tx, float ty);
 */




/* api definition */
#define mat4s_(NAME) PLAY_CGLM_STRUCTAPI(mat4, NAME)

#define PLAY_CGLM_S_MAT4_IDENTITY_INIT  {PLAY_CGLM_MAT4_IDENTITY_INIT}
#define PLAY_CGLM_S_MAT4_ZERO_INIT      {PLAY_CGLM_MAT4_ZERO_INIT}

/* for C only */
#define PLAY_CGLM_S_MAT4_IDENTITY ((mat4s)PLAY_CGLM_S_MAT4_IDENTITY_INIT)
#define PLAY_CGLM_S_MAT4_ZERO     ((mat4s)PLAY_CGLM_S_MAT4_ZERO_INIT)

/*!
 * @brief copy all members of [mat] to [dest]
 *
 * matrix may not be aligned, u stands for unaligned, this may be useful when
 * copying a matrix from external source e.g. asset importer...
 *
 * @param[in]  mat  source
 * @returns         destination
 */
PLAY_CGLM_INLINE
mat4s
mat4s_(ucopy)(mat4s mat)
{
    mat4s r;
    mat4_ucopy(mat.raw, r.raw);
    return r;
}

/*!
 * @brief copy all members of [mat] to [dest]
 *
 * @param[in]  mat  source
 * @returns         destination
 */
PLAY_CGLM_INLINE
mat4s
mat4s_(copy)(mat4s mat)
{
    mat4s r;
    mat4_copy(mat.raw, r.raw);
    return r;
}

/*!
 * @brief make given matrix identity. It is identical with below,
 *        but it is more easy to do that with this func especially for members
 *        e.g. mat4_identity(aStruct->aMatrix);
 *
 * @code
 * mat4_copy(PLAY_CGLM_MAT4_IDENTITY, mat); // C only
 *
 * // or
 * mat4 mat = PLAY_CGLM_MAT4_IDENTITY_INIT;
 * @endcode
 *
 * @returns  destination
 */
PLAY_CGLM_INLINE
mat4s
mat4s_(identity)(void)
{
    mat4s r;
    mat4_identity(r.raw);
    return r;
}

/*!
 * @brief make given matrix array's each element identity matrix
 *
 * @param[in, out]  mat   matrix array (must be aligned (16/32)
 *                        if alignment is not disabled)
 *
 * @param[in]       count count of matrices
 */
PLAY_CGLM_INLINE
void
mat4s_(identity_array)(mat4s * __restrict mat, size_t count)
{
    PLAY_CGLM_ALIGN_MAT mat4s t = PLAY_CGLM_S_MAT4_IDENTITY_INIT;
    size_t i;

    for (i = 0; i < count; i++)
    {
        mat4_copy(t.raw, mat[i].raw);
    }
}

/*!
 * @brief make given matrix zero.
 *
 * @returns  matrix
 */
PLAY_CGLM_INLINE
mat4s
mat4s_(zero)(void)
{
    mat4s r;
    mat4_zero(r.raw);
    return r;
}

/*!
 * @brief copy upper-left of mat4 to mat3
 *
 * @param[in]  mat  source
 * @returns         destination
 */
PLAY_CGLM_INLINE
mat3s
mat4s_(pick3)(mat4s mat)
{
    mat3s r;
    mat4_pick3(mat.raw, r.raw);
    return r;
}

/*!
 * @brief copy upper-left of mat4 to mat3 (transposed)
 *
 * the postfix t stands for transpose
 *
 * @param[in]  mat  source
 * @returns         destination
 */
PLAY_CGLM_INLINE
mat3s
mat4s_(pick3t)(mat4s mat)
{
    mat3s r;
    mat4_pick3t(mat.raw, r.raw);
    return r;
}

/*!
 * @brief copy mat3 to mat4's upper-left
 *
 * @param[in]  mat  source
 * @param[in]  dest destination
 * @returns         destination
 */
PLAY_CGLM_INLINE
mat4s
mat4s_(ins3)(mat3s mat, mat4s dest)
{
    mat4_ins3(mat.raw, dest.raw);
    return dest;
}

/*!
 * @brief multiply m1 and m2 to dest
 *
 * m1, m2 and dest matrices can be same matrix, it is possible to write this:
 *
 * @code
 * mat4 m = PLAY_CGLM_MAT4_IDENTITY_INIT;
 * r = mat4s_mul(m, m);
 * @endcode
 *
 * @param[in]  m1   left matrix
 * @param[in]  m2   right matrix
 * @returns destination matrix
 */
PLAY_CGLM_INLINE
mat4s
mat4s_(mul)(mat4s m1, mat4s m2)
{
    mat4s r;
    mat4_mul(m1.raw, m2.raw, r.raw);
    return r;
}

/*!
 * @brief mupliply N mat4 matrices and store result in dest
 *
 * this function lets you multiply multiple (more than two or more...) matrices
 * <br><br>multiplication will be done in loop, this may reduce instructions
 * size but if <b>len</b> is too small then compiler may unroll whole loop,
 * usage:
 * @code
 * mat4 m1, m2, m3, m4, res;
 *
 * res = mat4_mulN((mat4 *[]){&m1, &m2, &m3, &m4}, 4);
 * @endcode
 *
 * @warning matrices parameter is pointer array not mat4 array!
 *
 * @param[in]  matrices mat4 * array
 * @param[in]  len      matrices count
 * @returns             result matrix
 */
PLAY_CGLM_INLINE
mat4s
mat4s_(mulN)(mat4s * __restrict matrices[], uint32_t len)
{
    PLAY_CGLM_ALIGN_MAT mat4s r = PLAY_CGLM_S_MAT4_IDENTITY_INIT;
    size_t i;

    for (i = 0; i < len; i++)
    {
        r = mat4s_(mul)(r, *matrices[i]);
    }

    return r;
}

/*!
 * @brief multiply mat4 with vec4 (column vector) and store in dest vector
 *
 * @param[in]  m    mat4 (left)
 * @param[in]  v    vec4 (right, column vector)
 * @returns         vec4 (result, column vector)
 */
PLAY_CGLM_INLINE
vec4s
mat4s_(mulv)(mat4s m, vec4s v)
{
    vec4s r;
    mat4_mulv(m.raw, v.raw, r.raw);
    return r;
}

/*!
 * @brief trace of matrix
 *
 * sum of the elements on the main diagonal from upper left to the lower right
 *
 * @param[in]  m matrix
 */
PLAY_CGLM_INLINE
float
mat4s_(trace)(mat4s m)
{
    return mat4_trace(m.raw);
}

/*!
 * @brief trace of matrix (rotation part)
 *
 * sum of the elements on the main diagonal from upper left to the lower right
 *
 * @param[in]  m matrix
 */
PLAY_CGLM_INLINE
float
mat4s_(trace3)(mat4s m)
{
    return mat4_trace3(m.raw);
}

/*!
 * @brief convert mat4's rotation part to quaternion
 *
 * @param[in]  m    affine matrix
 * @returns         destination quaternion
 */
PLAY_CGLM_INLINE
versors
mat4s_(quat)(mat4s m)
{
    versors r;
    mat4_quat(m.raw, r.raw);
    return r;
}

/*!
 * @brief multiply vector with mat4
 *
 * @param[in]  m    mat4_new(affine transform)
 * @param[in]  v    vec3
 * @param[in]  last 4th item to make it vec4
 * @returns         result vector (vec3)
 */
PLAY_CGLM_INLINE
vec3s
mat4s_(mulv3)(mat4s m, vec3s v, float last)
{
    vec3s r;
    mat4_mulv3(m.raw, v.raw, last, r.raw);
    return r;
}

/*!
 * @brief transpose mat4 and store result in same matrix
 *
 * @param[in] m source
 * @returns     result
 */
PLAY_CGLM_INLINE
mat4s
mat4s_(transpose)(mat4s m)
{
    mat4_transpose(m.raw);
    return m;
}

/*!
 * @brief scale (multiply with scalar) matrix without simd optimization
 *
 * multiply matrix with scalar
 *
 * @param[in] m matrix
 * @param[in] s scalar
 * @returns     matrix
 */
PLAY_CGLM_INLINE
mat4s
mat4s_(scale_p)(mat4s m, float s)
{
    mat4_scale_p(m.raw, s);
    return m;
}

/*!
 * @brief scale (multiply with scalar) matrix
 *
 * multiply matrix with scalar
 *
 * @param[in] m matrix
 * @param[in] s scalar
 * @returns     matrix
 */
PLAY_CGLM_INLINE
mat4s
mat4s_(scale)(mat4s m, float s)
{
    mat4_scale(m.raw, s);
    return m;
}

/*!
 * @brief mat4 determinant
 *
 * @param[in] mat matrix
 *
 * @return determinant
 */
PLAY_CGLM_INLINE
float
mat4s_(det)(mat4s mat)
{
    return mat4_det(mat.raw);
}

/*!
 * @brief inverse mat4 and store in dest
 *
 * @param[in]  mat  matrix
 * @returns         inverse matrix
 */
PLAY_CGLM_INLINE
mat4s
mat4s_(inv)(mat4s mat)
{
    mat4s r;
    mat4_inv(mat.raw, r.raw);
    return r;
}

/*!
 * @brief inverse mat4 and store in dest
 *
 * this func uses reciprocal approximation without extra corrections
 * e.g Newton-Raphson. this should work faster than normal,
 * to get more precise use mat4_inv version.
 *
 * NOTE: You will lose precision, mat4_inv is more accurate
 *
 * @param[in]  mat  matrix
 * @returns         inverse matrix
 */
PLAY_CGLM_INLINE
mat4s
mat4s_(inv_fast)(mat4s mat)
{
    mat4s r;
    mat4_inv_fast(mat.raw, r.raw);
    return r;
}

/*!
 * @brief swap two matrix columns
 *
 * @param[in] mat  matrix
 * @param[in] col1 col1
 * @param[in] col2 col2
 * @returns        matrix
 */
PLAY_CGLM_INLINE
mat4s
mat4s_(swap_col)(mat4s mat, int col1, int col2)
{
    mat4_swap_col(mat.raw, col1, col2);
    return mat;
}

/*!
 * @brief swap two matrix rows
 *
 * @param[in] mat  matrix
 * @param[in] row1 row1
 * @param[in] row2 row2
 * @returns        matrix
 */
PLAY_CGLM_INLINE
mat4s
mat4s_(swap_row)(mat4s mat, int row1, int row2)
{
    mat4_swap_row(mat.raw, row1, row2);
    return mat;
}

/*!
 * @brief helper for  R (row vector) * M (matrix) * C (column vector)
 *
 * rmc stands for Row * Matrix * Column
 *
 * the result is scalar because R * M = Matrix1x4 (row vector),
 * then Matrix1x4 * Vec4 (column vector) = Matrix1x1 (Scalar)
 *
 * @param[in]  r   row vector or matrix1x4
 * @param[in]  m   matrix4x4
 * @param[in]  c   column vector or matrix4x1
 *
 * @return scalar value e.g. B(s)
 */
PLAY_CGLM_INLINE
float
mat4s_(rmc)(vec4s r, mat4s m, vec4s c)
{
    return mat4_rmc(r.raw, m.raw, c.raw);
}

/*!
 * @brief Create mat4 matrix from pointer
 *
 * @param[in]  src  pointer to an array of floats
 * @return constructed matrix from raw pointer
 */
PLAY_CGLM_INLINE
mat4s
mat4s_(make)(const float * __restrict src)
{
    mat4s r;
    mat4_make(src, r.raw);
    return r;
}

/*!
 * @brief Create mat4 matrix from texture transform parameters
 *
 * @param[in]  sx  scale x
 * @param[in]  sy  scale y
 * @param[in]  rot rotation in radians CCW/RH
 * @param[in]  tx  translate x
 * @param[in]  ty  translate y
 * @return texture transform matrix
 */
PLAY_CGLM_INLINE
mat4s
mat4s_(textrans)(float sx, float sy, float rot, float tx, float ty)
{
    mat4s r;
    mat4_textrans(sx, sy, rot, tx, ty, r.raw);
    return r;
}



/*** End of inlined file: mat4.h ***/


/*** Start of inlined file: mat4x2.h ***/
/*
 Macros:
   PLAY_CGLM_S_MAT4X2_ZERO_INIT
   PLAY_CGLM_S_MAT4X2_ZERO

 Functions:
   PLAY_CGLM_INLINE mat4x2s mat4x2s_zero(void);
   PLAY_CGLM_INLINE mat4x2s mat4x2s_make(const float * __restrict src);
   PLAY_CGLM_INLINE mat2s   mat4x2s_mul(mat4x2s m1, mat2x4s m2);
   PLAY_CGLM_INLINE vec2s   mat4x2s_mulv(mat4x2s m, vec4s v);
   PLAY_CGLM_INLINE mat2x4s mat4x2s_transpose(mat4x2s m);
   PLAY_CGLM_INLINE mat4x2s mat4x2s_scale(mat4x2s m, float s);
 */

#ifndef cmat4x2s_h
#define cmat4x2s_h

/* api definition */
#define mat4x2s_(NAME) PLAY_CGLM_STRUCTAPI(mat4x2, NAME)

#define PLAY_CGLM_S_MAT4X2_ZERO_INIT {PLAY_CGLM_MAT4X2_ZERO_INIT}

/* for C only */
#define PLAY_CGLM_S_MAT4X2_ZERO ((mat4x2s)PLAY_CGLM_S_MAT4X2_ZERO_INIT)

/*!
 * @brief Zero out the mat4x2s (dest).
 *
 * @return[out] dest constructed mat4x2s from raw pointer
 */
PLAY_CGLM_INLINE
mat4x2s
mat4x2s_(zero)(void)
{
    mat4x2s dest;
    mat4x2_zero(dest.raw);
    return dest;
}

/*!
 * @brief Create mat4x2s (dest) from pointer (src).
 *
 * @param[in]   src  pointer to an array of floats
 * @return[out] dest constructed mat4x2s from raw pointer
 */
PLAY_CGLM_INLINE
mat4x2s
mat4x2s_(make)(const float * __restrict src)
{
    mat4x2s dest;
    mat4x2_make(src, dest.raw);
    return dest;
}

/*!
 * @brief Multiply mat4x2s (m1) by mat2x4s (m2) and store in mat2s (dest).
 *
 * @code
 * r = mat4x2s_mul(mat4x2s, mat2x4s);
 * @endcode
 *
 * @param[in]   m1   mat4x2s (left)
 * @param[in]   m2   mat2x4s (right)
 * @return[out] dest constructed mat2s from raw pointers
 */
PLAY_CGLM_INLINE
mat2s
mat4x2s_(mul)(mat4x2s m1, mat2x4s m2)
{
    mat2s dest;
    mat4x2_mul(m1.raw, m2.raw, dest.raw);
    return dest;
}

/*!
 * @brief Multiply mat4x2s (m) by vec4s (v) and store in vec2s (dest).
 *
 * @param[in]   m    mat4x2s (left)
 * @param[in]   v    vec4s (right, column vector)
 * @return[out] dest constructed vec2s from raw pointers
 */
PLAY_CGLM_INLINE
vec2s
mat4x2s_(mulv)(mat4x2s m, vec4s v)
{
    vec2s dest;
    mat4x2_mulv(m.raw, v.raw, dest.raw);
    return dest;
}

/*!
 * @brief Transpose mat4x2s (m) and store in mat2x4s (dest).
 *
 * @param[in]   m    mat4x2s (left)
 * @return[out] dest constructed mat2x4s from raw pointers
 */
PLAY_CGLM_INLINE
mat2x4s
mat4x2s_(transpose)(mat4x2s m)
{
    mat2x4s dest;
    mat4x2_transpose(m.raw, dest.raw);
    return dest;
}

/*!
 * @brief Multiply mat4x2s (m) by scalar constant (s).
 *
 * @param[in, out] m mat4x2s (src, dest)
 * @param[in]      s float (scalar)
 */
PLAY_CGLM_INLINE
mat4x2s
mat4x2s_(scale)(mat4x2s m, float s)
{
    mat4x2_scale(m.raw, s);
    return m;
}

#endif /* cmat4x2s_h */

/*** End of inlined file: mat4x2.h ***/


/*** Start of inlined file: mat4x3.h ***/
/*
 Macros:
   PLAY_CGLM_S_MAT4X3_ZERO_INIT
   PLAY_CGLM_S_MAT4X3_ZERO

 Functions:
   PLAY_CGLM_INLINE mat4x3s mat4x3s_zero(void);
   PLAY_CGLM_INLINE mat4x3s mat4x3s_make(const float * __restrict src);
   PLAY_CGLM_INLINE mat3s   mat4x3s_mul(mat4x3s m1, mat3x4s m2);
   PLAY_CGLM_INLINE vec3s   mat4x3s_mulv(mat4x3s m, vec4s v);
   PLAY_CGLM_INLINE mat3x4s mat4x3s_transpose(mat4x3s m);
   PLAY_CGLM_INLINE mat4x3s mat4x3s_scale(mat4x3s m, float s);
 */

#ifndef cmat4x3s_h
#define cmat4x3s_h

/* api definition */
#define mat4x3s_(NAME) PLAY_CGLM_STRUCTAPI(mat4x3, NAME)

#define PLAY_CGLM_S_MAT4X3_ZERO_INIT {PLAY_CGLM_MAT4X3_ZERO_INIT}

/* for C only */
#define PLAY_CGLM_S_MAT4X3_ZERO ((mat4x3s)PLAY_CGLM_S_MAT4X3_ZERO_INIT)

/*!
 * @brief Zero out the mat4x3s (dest).
 *
 * @return[out] dest constructed mat4x3s from raw pointer
 */
PLAY_CGLM_INLINE
mat4x3s
mat4x3s_(zero)(void)
{
    mat4x3s dest;
    mat4x3_zero(dest.raw);
    return dest;
}

/*!
 * @brief Create mat4x3s (dest) from pointer (src).
 *
 * @param[in]   src  pointer to an array of floats
 * @return[out] dest constructed mat4x3s from raw pointer
 */
PLAY_CGLM_INLINE
mat4x3s
mat4x3s_(make)(const float * __restrict src)
{
    mat4x3s dest;
    mat4x3_make(src, dest.raw);
    return dest;
}

/*!
 * @brief Multiply mat4x3s (m1) by mat3x4s (m2) and store in mat3s (dest).
 *
 * @code
 * r = mat4x3s_mul(mat4x3s, mat3x4s);
 * @endcode
 *
 * @param[in]   m1   mat4x3s (left)
 * @param[in]   m2   mat3x4s (right)
 * @return[out] dest constructed mat3s from raw pointers
 */
PLAY_CGLM_INLINE
mat3s
mat4x3s_(mul)(mat4x3s m1, mat3x4s m2)
{
    mat3s dest;
    mat4x3_mul(m1.raw, m2.raw, dest.raw);
    return dest;
}

/*!
 * @brief Multiply mat4x3s (m) by vec4s (v) and store in vec3s (dest).
 *
 * @param[in]   m    mat4x3s (left)
 * @param[in]   v    vec4s (right, column vector)
 * @return[out] dest constructed vec3s from raw pointers
 */
PLAY_CGLM_INLINE
vec3s
mat4x3s_(mulv)(mat4x3s m, vec4s v)
{
    vec3s dest;
    mat4x3_mulv(m.raw, v.raw, dest.raw);
    return dest;
}

/*!
 * @brief Transpose mat4x3s (m) and store in mat3x4s (dest).
 *
 * @param[in]   m    mat4x3s (left)
 * @return[out] dest constructed mat3x4s from raw pointers
 */
PLAY_CGLM_INLINE
mat3x4s
mat4x3s_(transpose)(mat4x3s m)
{
    mat3x4s dest;
    mat4x3_transpose(m.raw, dest.raw);
    return dest;
}

/*!
 * @brief Multiply mat4x3s (m) by scalar constant (s).
 *
 * @param[in, out] m mat4x3s (src, dest)
 * @param[in]      s float (scalar)
 */
PLAY_CGLM_INLINE
mat4x3s
mat4x3s_(scale)(mat4x3s m, float s)
{
    mat4x3_scale(m.raw, s);
    return m;
}

#endif /* cmat4x3s_h */

/*** End of inlined file: mat4x3.h ***/


/*** Start of inlined file: affine.h ***/
/*
 Functions:
   PLAY_CGLM_INLINE mat4s translate(mat4s m, vec3s v);
   PLAY_CGLM_INLINE mat4s translate_x(mat4s m, float x);
   PLAY_CGLM_INLINE mat4s translate_y(mat4s m, float y);
   PLAY_CGLM_INLINE mat4s translate_z(mat4s m, float z);
   PLAY_CGLM_INLINE mat4s translate_make(vec3s v);
   PLAY_CGLM_INLINE mat4s scale_to(mat4s m, vec3s v);
   PLAY_CGLM_INLINE mat4s scale_make(vec3s v);
   PLAY_CGLM_INLINE mat4s scale(mat4s m, vec3s v);
   PLAY_CGLM_INLINE mat4s scale_uni(mat4s m, float s);
   PLAY_CGLM_INLINE mat4s rotate_x(mat4s m, float angle);
   PLAY_CGLM_INLINE mat4s rotate_y(mat4s m, float angle);
   PLAY_CGLM_INLINE mat4s rotate_z(mat4s m, float angle);
   PLAY_CGLM_INLINE mat4s rotate_make(float angle, vec3s axis);
   PLAY_CGLM_INLINE mat4s rotate(mat4s m, float angle, vec3s axis);
   PLAY_CGLM_INLINE mat4s rotate_at(mat4s m, vec3s pivot, float angle, vec3s axis);
   PLAY_CGLM_INLINE mat4s rotate_atm(vec3s pivot, float angle, vec3s axis);
   PLAY_CGLM_INLINE mat4s spin(mat4s m, float angle, vec3s axis);
   PLAY_CGLM_INLINE vec3s decompose_scalev(mat4s m);
   PLAY_CGLM_INLINE bool  uniscaled(mat4s m);
   PLAY_CGLM_INLINE void  decompose_rs(mat4s m, mat4s * r, vec3s * s);
   PLAY_CGLM_INLINE void  decompose(mat4s m, vec4s t, mat4s * r, vec3s * s);
 */





/*** Start of inlined file: affine-mat.h ***/
/*
 Functions:
   PLAY_CGLM_INLINE mat4s mul(mat4 m1, mat4 m2);
   PLAY_CGLM_INLINE mat4s mul_rot(mat4 m1, mat4 m2);
   PLAY_CGLM_INLINE mat4s inv_tr();
 */




/*!
 * @brief this is similar to mat4s_mul but specialized to affine transform
 *
 * Matrix format should be:
 *   R  R  R  X
 *   R  R  R  Y
 *   R  R  R  Z
 *   0  0  0  W
 *
 * this reduces some multiplications. It should be faster than mat4_mul.
 * if you are not sure about matrix format then DON'T use this! use mat4_mul
 *
 * @param[in]   m1    affine matrix 1
 * @param[in]   m2    affine matrix 2
 * @returns         destination matrix
 */
PLAY_CGLM_INLINE
mat4s
mul(mat4s m1, mat4s m2)
{
    mat4s r;
    mul(m1.raw, m2.raw, r.raw);
    return r;
}

/*!
 * @brief this is similar to mat4_mul but specialized to affine transform
 *
 * Right Matrix format should be:
 *   R  R  R  0
 *   R  R  R  0
 *   R  R  R  0
 *   0  0  0  1
 *
 * this reduces some multiplications. It should be faster than mat4_mul.
 * if you are not sure about matrix format then DON'T use this! use mat4_mul
 *
 * @param[in]   m1    affine matrix 1
 * @param[in]   m2    affine matrix 2
 * @returns         destination matrix
 */
PLAY_CGLM_INLINE
mat4s
mul_rot(mat4s m1, mat4s m2)
{
    mat4s r;
    mul_rot(m1.raw, m2.raw, r.raw);
    return r;
}

/*!
 * @brief inverse orthonormal rotation + translation matrix (ridig-body)
 *
 * @code
 * X = | R  T |   X' = | R' -R'T |
 *     | 0  1 |        | 0     1 |
 * @endcode
 *
 * @param[in]  m  matrix
 * @returns      destination matrix
 */
PLAY_CGLM_INLINE
mat4s
inv_tr(mat4s m)
{
    inv_tr(m.raw);
    return m;
}


/*** End of inlined file: affine-mat.h ***/

/*!
 * @brief creates NEW translate transform matrix by v vector
 *
 * @param[in]   v   translate vector [x, y, z]
 * @returns         affine transform
 */
PLAY_CGLM_INLINE
mat4s
translate_make(vec3s v)
{
    mat4s m;
    translate_make(m.raw, v.raw);
    return m;
}

/*!
 * @brief creates NEW scale matrix by v vector
 *
 * @param[in]   v  scale vector [x, y, z]
 * @returns affine transform
 */
PLAY_CGLM_INLINE
mat4s
scale_make(vec3s v)
{
    mat4s m;
    scale_make(m.raw, v.raw);
    return m;
}

/*!
 * @brief scales existing transform matrix by v vector
 *        and stores result in same matrix
 *
 * @param[in]    m   affine transform
 * @param[in]    v   scale vector [x, y, z]
 * @returns          affine transform
 */
PLAY_CGLM_INLINE
mat4s
scale(mat4s m, vec3s v)
{
    mat4s r;
    scale_to(m.raw, v.raw, r.raw);
    return r;
}

/*!
 * @brief applies uniform scale to existing transform matrix v = [s, s, s]
 *        and stores result in same matrix
 *
 * @param[in]    m   affine transform
 * @param[in]    s   scale factor
 * @returns          affine transform
 */
PLAY_CGLM_INLINE
mat4s
scale_uni(mat4s m, float s)
{
    scale_uni(m.raw, s);
    return m;
}

/*!
 * @brief creates NEW rotation matrix by angle and axis
 *
 * axis will be normalized so you don't need to normalize it
 *
 * @param[in]  angle  angle (radians)
 * @param[in]  axis   axis
 * @returns           affine transform
 */
PLAY_CGLM_INLINE
mat4s
rotate_make(float angle, vec3s axis)
{
    mat4s m;
    rotate_make(m.raw, angle, axis.raw);
    return m;
}

/*!
 * @brief creates NEW rotation matrix by angle and axis at given point
 *
 * this creates rotation matrix, it assumes you don't have a matrix
 *
 * this should work faster than rotate_at because it reduces
 * one translate.
 *
 * @param[in]  pivot  rotation center
 * @param[in]  angle  angle (radians)
 * @param[in]  axis   axis
 * @returns           affine transform
 */
PLAY_CGLM_INLINE
mat4s
rotate_atm(vec3s pivot, float angle, vec3s axis)
{
    mat4s m;
    rotate_atm(m.raw, pivot.raw, angle, axis.raw);
    return m;
}

/*!
 * @brief decompose scale vector
 *
 * @param[in]  m  affine transform
 * @returns       scale vector (Sx, Sy, Sz)
 */
PLAY_CGLM_INLINE
vec3s
decompose_scalev(mat4s m)
{
    vec3s r;
    decompose_scalev(m.raw, r.raw);
    return r;
}

/*!
 * @brief returns true if matrix is uniform scaled. This is helpful for
 *        creating normal matrix.
 *
 * @param[in] m m
 *
 * @return boolean
 */
PLAY_CGLM_INLINE
bool
uniscaled(mat4s m)
{
    return uniscaled(m.raw);
}

/*!
 * @brief decompose rotation matrix (mat4) and scale vector [Sx, Sy, Sz]
 *        DON'T pass projected matrix here
 *
 * @param[in]  m affine transform
 * @param[out] r rotation matrix
 * @param[out] s scale matrix
 */
PLAY_CGLM_INLINE
void
decompose_rs(mat4s m, mat4s * __restrict r, vec3s * __restrict s)
{
    decompose_rs(m.raw, r->raw, s->raw);
}

/*!
 * @brief decompose affine transform, TODO: extract shear factors.
 *        DON'T pass projected matrix here
 *
 * @param[in]  m affine transform
 * @param[out] t translation vector
 * @param[out] r rotation matrix (mat4)
 * @param[out] s scaling vector [X, Y, Z]
 */
PLAY_CGLM_INLINE
void
decompose(mat4s m, vec4s * __restrict t, mat4s * __restrict r, vec3s * __restrict s)
{
    decompose(m.raw, t->raw, r->raw, s->raw);
}


/*** Start of inlined file: affine-pre.h ***/
/*
 Functions:
   PLAY_CGLM_INLINE mat4s translate(mat4s m, vec3s v);
   PLAY_CGLM_INLINE mat4s translate_x(mat4s m, float x);
   PLAY_CGLM_INLINE mat4s translate_y(mat4s m, float y);
   PLAY_CGLM_INLINE mat4s translate_z(mat4s m, float z);
   PLAY_CGLM_INLINE mat4s rotate_x(mat4s m, float angle);
   PLAY_CGLM_INLINE mat4s rotate_y(mat4s m, float angle);
   PLAY_CGLM_INLINE mat4s rotate_z(mat4s m, float angle);
   PLAY_CGLM_INLINE mat4s rotate(mat4s m, float angle, vec3s axis);
   PLAY_CGLM_INLINE mat4s rotate_at(mat4s m, vec3s pivot, float angle, vec3s axis);
   PLAY_CGLM_INLINE mat4s spin(mat4s m, float angle, vec3s axis);
 */




/*!
 * @brief translate existing transform matrix by v vector
 *        and stores result in same matrix
 *
 * @param[in]       m   affine transform
 * @param[in]       v   translate vector [x, y, z]
 * @returns             affine transform
 */
PLAY_CGLM_INLINE
mat4s
translate(mat4s m, vec3s v)
{
    translate(m.raw, v.raw);
    return m;
}

/*!
 * @brief translate existing transform matrix by x factor
 *
 * @param[in]       m   affine transform
 * @param[in]       x   x factor
 * @returns             affine transform
 */
PLAY_CGLM_INLINE
mat4s
translate_x(mat4s m, float x)
{
    translate_x(m.raw, x);
    return m;
}

/*!
 * @brief translate existing transform matrix by y factor
 *
 * @param[in]       m   affine transform
 * @param[in]       y   y factor
 * @returns             affine transform
 */
PLAY_CGLM_INLINE
mat4s
translate_y(mat4s m, float y)
{
    translate_y(m.raw, y);
    return m;
}

/*!
 * @brief translate existing transform matrix by z factor
 *
 * @param[in]       m   affine transform
 * @param[in]       z   z factor
 * @returns             affine transform
 */
PLAY_CGLM_INLINE
mat4s
translate_z(mat4s m, float z)
{
    translate_z(m.raw, z);
    return m;
}

/*!
 * @brief rotate existing transform matrix around X axis by angle
 *        and store result in dest
 *
 * @param[in]   m       affine transform
 * @param[in]   angle   angle (radians)
 * @returns             rotated matrix
 */
PLAY_CGLM_INLINE
mat4s
rotate_x(mat4s m, float angle)
{
    mat4s r;
    rotate_x(m.raw, angle, r.raw);
    return r;
}

/*!
 * @brief rotate existing transform matrix around Y axis by angle
 *        and store result in dest
 *
 * @param[in]   m       affine transform
 * @param[in]   angle   angle (radians)
 * @returns             rotated matrix
 */
PLAY_CGLM_INLINE
mat4s
rotate_y(mat4s m, float angle)
{
    mat4s r;
    rotate_y(m.raw, angle, r.raw);
    return r;
}

/*!
 * @brief rotate existing transform matrix around Z axis by angle
 *        and store result in dest
 *
 * @param[in]   m       affine transform
 * @param[in]   angle   angle (radians)
 * @returns             rotated matrix
 */
PLAY_CGLM_INLINE
mat4s
rotate_z(mat4s m, float angle)
{
    mat4s r;
    rotate_z(m.raw, angle, r.raw);
    return r;
}

/*!
 * @brief rotate existing transform matrix around given axis by angle
 *
 * @param[in]       m       affine transform
 * @param[in]       angle   angle (radians)
 * @param[in]       axis    axis
 * @returns                affine transform
 */
PLAY_CGLM_INLINE
mat4s
rotate(mat4s m, float angle, vec3s axis)
{
    rotate(m.raw, angle, axis.raw);
    return m;
}

/*!
 * @brief rotate existing transform
 *        around given axis by angle at given pivot point (rotation center)
 *
 * @param[in]       m       affine transform
 * @param[in]       pivot   rotation center
 * @param[in]       angle   angle (radians)
 * @param[in]       axis    axis
 * @returns                 affine transform
 */
PLAY_CGLM_INLINE
mat4s
rotate_at(mat4s m, vec3s pivot, float angle, vec3s axis)
{
    rotate_at(m.raw, pivot.raw, angle, axis.raw);
    return m;
}

/*!
 * @brief rotate existing transform matrix around given axis by angle around self (doesn't affected by position)
 *
 * @param[in]       m       affine transform
 * @param[in]       angle   angle (radians)
 * @param[in]       axis    axis
 * @returns                affine transform
 */
PLAY_CGLM_INLINE
mat4s
spin(mat4s m, float angle, vec3s axis)
{
    spin(m.raw, angle, axis.raw);
    return m;
}



/*** End of inlined file: affine-pre.h ***/


/*** Start of inlined file: affine-post.h ***/
/*
 Functions:
   PLAY_CGLM_INLINE mat4s translated(mat4s m, vec3s v);
   PLAY_CGLM_INLINE mat4s translated_x(mat4s m, float x);
   PLAY_CGLM_INLINE mat4s translated_y(mat4s m, float y);
   PLAY_CGLM_INLINE mat4s translated_z(mat4s m, float z);
   PLAY_CGLM_INLINE mat4s rotated_x(mat4s m, float angle);
   PLAY_CGLM_INLINE mat4s rotated_y(mat4s m, float angle);
   PLAY_CGLM_INLINE mat4s rotated_z(mat4s m, float angle);
   PLAY_CGLM_INLINE mat4s rotated(mat4s m, float angle, vec3s axis);
   PLAY_CGLM_INLINE mat4s rotated_at(mat4s m, vec3s pivot, float angle, vec3s axis);
   PLAY_CGLM_INLINE mat4s spinned(mat4s m, float angle, vec3s axis);
 */




/*!
 * @brief translate existing transform matrix by v vector
 *        and stores result in same matrix
 *
 * @param[in]       m   affine transform
 * @param[in]       v   translate vector [x, y, z]
 * @returns             affine transform
 */
PLAY_CGLM_INLINE
mat4s
translated(mat4s m, vec3s v)
{
    translated(m.raw, v.raw);
    return m;
}

/*!
 * @brief translate existing transform matrix by x factor
 *
 * @param[in]       m   affine transform
 * @param[in]       x   x factor
 * @returns             affine transform
 */
PLAY_CGLM_INLINE
mat4s
translated_x(mat4s m, float x)
{
    translated_x(m.raw, x);
    return m;
}

/*!
 * @brief translate existing transform matrix by y factor
 *
 * @param[in]       m   affine transform
 * @param[in]       y   y factor
 * @returns             affine transform
 */
PLAY_CGLM_INLINE
mat4s
translated_y(mat4s m, float y)
{
    translated_y(m.raw, y);
    return m;
}

/*!
 * @brief translate existing transform matrix by z factor
 *
 * @param[in]       m   affine transform
 * @param[in]       z   z factor
 * @returns             affine transform
 */
PLAY_CGLM_INLINE
mat4s
translated_z(mat4s m, float z)
{
    translated_z(m.raw, z);
    return m;
}

/*!
 * @brief rotate existing transform matrix around X axis by angle
 *        and store result in dest
 *
 * @param[in]   m       affine transform
 * @param[in]   angle   angle (radians)
 * @returns             rotated matrix
 */
PLAY_CGLM_INLINE
mat4s
rotated_x(mat4s m, float angle)
{
    mat4s r;
    rotated_x(m.raw, angle, r.raw);
    return r;
}

/*!
 * @brief rotate existing transform matrix around Y axis by angle
 *        and store result in dest
 *
 * @param[in]   m       affine transform
 * @param[in]   angle   angle (radians)
 * @returns             rotated matrix
 */
PLAY_CGLM_INLINE
mat4s
rotated_y(mat4s m, float angle)
{
    mat4s r;
    rotated_y(m.raw, angle, r.raw);
    return r;
}

/*!
 * @brief rotate existing transform matrix around Z axis by angle
 *        and store result in dest
 *
 * @param[in]   m       affine transform
 * @param[in]   angle   angle (radians)
 * @returns             rotated matrix
 */
PLAY_CGLM_INLINE
mat4s
rotated_z(mat4s m, float angle)
{
    mat4s r;
    rotated_z(m.raw, angle, r.raw);
    return r;
}

/*!
 * @brief rotate existing transform matrix around given axis by angle
 *
 * @param[in]       m       affine transform
 * @param[in]       angle   angle (radians)
 * @param[in]       axis    axis
 * @returns                affine transform
 */
PLAY_CGLM_INLINE
mat4s
rotated(mat4s m, float angle, vec3s axis)
{
    rotated(m.raw, angle, axis.raw);
    return m;
}

/*!
 * @brief rotate existing transform
 *        around given axis by angle at given pivot point (rotation center)
 *
 * @param[in]       m       affine transform
 * @param[in]       pivot   rotation center
 * @param[in]       angle   angle (radians)
 * @param[in]       axis    axis
 * @returns                 affine transform
 */
PLAY_CGLM_INLINE
mat4s
rotated_at(mat4s m, vec3s pivot, float angle, vec3s axis)
{
    rotated_at(m.raw, pivot.raw, angle, axis.raw);
    return m;
}

/*!
 * @brief rotate existing transform matrix around given axis by angle around self (doesn't affected by position)
 *
 * @param[in]       m       affine transform
 * @param[in]       angle   angle (radians)
 * @param[in]       axis    axis
 * @returns                affine transform
 */
PLAY_CGLM_INLINE
mat4s
spinned(mat4s m, float angle, vec3s axis)
{
    spinned(m.raw, angle, axis.raw);
    return m;
}



/*** End of inlined file: affine-post.h ***/



/*** End of inlined file: affine.h ***/


/*** Start of inlined file: frustum.h ***/




/*** Start of inlined file: plane.h ***/



/*
 Plane equation:  Ax + By + Cz + D = 0;

 It stored in vec4 as [A, B, C, D]. (A, B, C) is normal and D is distance
*/

/*
 Functions:
   PLAY_CGLM_INLINE vec4s plane_normalize(vec4s plane);
 */

/*!
 * @brief normalizes a plane
 *
 * @param[in] plane plane to normalize
 * @returns         normalized plane
 */
PLAY_CGLM_INLINE
vec4s
plane_normalize(vec4s plane)
{
    plane_normalize(plane.raw);
    return plane;
}



/*** End of inlined file: plane.h ***/

/* you can override clip space coords
   but you have to provide all with same name
   e.g.: define PLAY_CGLM_CSCOORD_LBN {0.0f, 0.0f, 1.0f, 1.0f} */
#ifndef PLAY_CGLM_CUSTOM_CLIPSPACE

/* near */
#define PLAY_CGLM_S_CSCOORD_LBN {-1.0f, -1.0f, -1.0f, 1.0f}
#define PLAY_CGLM_S_CSCOORD_LTN {-1.0f,  1.0f, -1.0f, 1.0f}
#define PLAY_CGLM_S_CSCOORD_RTN { 1.0f,  1.0f, -1.0f, 1.0f}
#define PLAY_CGLM_S_CSCOORD_RBN { 1.0f, -1.0f, -1.0f, 1.0f}

/* far */
#define PLAY_CGLM_S_CSCOORD_LBF {-1.0f, -1.0f,  1.0f, 1.0f}
#define PLAY_CGLM_S_CSCOORD_LTF {-1.0f,  1.0f,  1.0f, 1.0f}
#define PLAY_CGLM_S_CSCOORD_RTF { 1.0f,  1.0f,  1.0f, 1.0f}
#define PLAY_CGLM_S_CSCOORD_RBF { 1.0f, -1.0f,  1.0f, 1.0f}

#endif

/*!
 * @brief extracts view frustum planes
 *
 * planes' space:
 *  1- if m = proj:     View Space
 *  2- if m = viewProj: World Space
 *  3- if m = MVP:      Object Space
 *
 * You probably want to extract planes in world space so use viewProj as m
 * Computing viewProj:
 *   mat4_mul(proj, view, viewProj);
 *
 * Exracted planes order: [left, right, bottom, top, near, far]
 *
 * @param[in]  m    matrix (see brief)
 * @param[out] dest extracted view frustum planes (see brief)
 */
PLAY_CGLM_INLINE
void
frustum_planes(mat4s m, vec4s dest[6])
{
    vec4 rawDest[6];
    frustum_planes(m.raw, rawDest);
    vec4s_(pack)(dest, rawDest, 6);
}

/*!
 * @brief extracts view frustum corners using clip-space coordinates
 *
 * corners' space:
 *  1- if m = invViewProj: World Space
 *  2- if m = invMVP:      Object Space
 *
 * You probably want to extract corners in world space so use invViewProj
 * Computing invViewProj:
 *   mat4_mul(proj, view, viewProj);
 *   ...
 *   mat4_inv(viewProj, invViewProj);
 *
 * if you have a near coord at i index, you can get it's far coord by i + 4
 *
 * Find center coordinates:
 *   for (j = 0; j < 4; j++) {
 *     vec3_center(corners[i], corners[i + 4], centerCorners[i]);
 *   }
 *
 * @param[in]  invMat matrix (see brief)
 * @param[out] dest   exracted view frustum corners (see brief)
 */
PLAY_CGLM_INLINE
void
frustum_corners(mat4s invMat, vec4s dest[8])
{
    vec4 rawDest[8];
    frustum_corners(invMat.raw, rawDest);
    vec4s_(pack)(dest, rawDest, 8);
}

/*!
 * @brief finds center of view frustum
 *
 * @param[in]  corners view frustum corners
 * @returns            view frustum center
 */
PLAY_CGLM_INLINE
vec4s
frustum_center(vec4s corners[8])
{
    vec4 rawCorners[8];
    vec4s r;

    vec4s_(unpack)(rawCorners, corners, 8);
    frustum_center(rawCorners, r.raw);
    return r;
}

/*!
 * @brief finds bounding box of frustum relative to given matrix e.g. view mat
 *
 * @param[in]  corners view frustum corners
 * @param[in]  m       matrix to convert existing conners
 * @param[out] box     bounding box as array [min, max]
 */
PLAY_CGLM_INLINE
void
frustum_box(vec4s corners[8], mat4s m, vec3s box[2])
{
    vec4 rawCorners[8];
    vec3 rawBox[2];

    vec4s_(unpack)(rawCorners, corners, 8);
    frustum_box(rawCorners, m.raw, rawBox);
    vec3s_(pack)(box, rawBox, 2);
}

/*!
 * @brief finds planes corners which is between near and far planes (parallel)
 *
 * this will be helpful if you want to split a frustum e.g. CSM/PSSM. This will
 * find planes' corners but you will need to one more plane.
 * Actually you have it, it is near, far or created previously with this func ;)
 *
 * @param[in]  corners view  frustum corners
 * @param[in]  splitDist     split distance
 * @param[in]  farDist       far distance (zFar)
 * @param[out] planeCorners  plane corners [LB, LT, RT, RB]
 */
PLAY_CGLM_INLINE
void
frustum_corners_at(vec4s corners[8],
                   float splitDist,
                   float farDist,
                   vec4s planeCorners[4])
{
    vec4 rawCorners[8];
    vec4 rawPlaneCorners[4];

    vec4s_(unpack)(rawCorners, corners, 8);
    frustum_corners_at(rawCorners, splitDist, farDist, rawPlaneCorners);
    vec4s_(pack)(planeCorners, rawPlaneCorners, 8);
}



/*** End of inlined file: frustum.h ***/


/*** Start of inlined file: noise.h ***/



/*
 Functions:
   PLAY_CGLM_INLINE float perlin_vec4(vec4s point);
 */

/*!
 * @brief Classic perlin noise
 *
 * @param[in]  point  4D vector
 * @returns           perlin noise value
 */
PLAY_CGLM_INLINE
float
perlin_vec4(vec4s point)
{
    return perlin_vec4(point.raw);
}

/*!
 * @brief Classic perlin noise
 *
 * @param[in]  point  3D vector
 * @returns           perlin noise value
 */
PLAY_CGLM_INLINE
float
perlin_vec3(vec3s point)
{
    return perlin_vec3(point.raw);
}

/*!
 * @brief Classic perlin noise
 *
 * @param[in]  point  2D vector
 * @returns           perlin noise value
 */
PLAY_CGLM_INLINE
float
perlin_vec2(vec2s point)
{
    return perlin_vec2(point.raw);
}



/*** End of inlined file: noise.h ***/


/*** Start of inlined file: box.h ***/



/* api definition */
#define aabbs_(NAME) PLAY_CGLM_STRUCTAPI(aabb, NAME)

/*!
 * @brief apply transform to Axis-Aligned Bounding Box
 *
 * @param[in]  box  bounding box
 * @param[in]  m    transform matrix
 * @param[out] dest transformed bounding box
 */
PLAY_CGLM_INLINE
void
aabbs_(transform)(vec3s box[2], mat4s m, vec3s dest[2])
{
    vec3 rawBox[2];
    vec3 rawDest[2];

    vec3s_(unpack)(rawBox, box, 2);
    aabb_transform(rawBox, m.raw, rawDest);
    vec3s_(pack)(dest, rawDest, 2);
}

/*!
 * @brief merges two AABB bounding box and creates new one
 *
 * two box must be in same space, if one of box is in different space then
 * you should consider to convert it's space by box_space
 *
 * @param[in]  box1 bounding box 1
 * @param[in]  box2 bounding box 2
 * @param[out] dest merged bounding box
 */
PLAY_CGLM_INLINE
void
aabbs_(merge)(vec3s box1[2], vec3s box2[2], vec3s dest[2])
{
    vec3 rawBox1[2];
    vec3 rawBox2[2];
    vec3 rawDest[2];

    vec3s_(unpack)(rawBox1, box1, 2);
    vec3s_(unpack)(rawBox2, box2, 2);
    aabb_merge(rawBox1, rawBox2, rawDest);
    vec3s_(pack)(dest, rawDest, 2);
}

/*!
 * @brief crops a bounding box with another one.
 *
 * this could be useful for getting a bbox which fits with view frustum and
 * object bounding boxes. In this case you crop view frustum box with objects
 * box
 *
 * @param[in]  box     bounding box 1
 * @param[in]  cropBox crop box
 * @param[out] dest    cropped bounding box
 */
PLAY_CGLM_INLINE
void
aabbs_(crop)(vec3s box[2], vec3s cropBox[2], vec3s dest[2])
{
    vec3 rawBox[2];
    vec3 rawCropBox[2];
    vec3 rawDest[2];

    vec3s_(unpack)(rawBox, box, 2);
    vec3s_(unpack)(rawCropBox, cropBox, 2);
    aabb_crop(rawBox, rawCropBox, rawDest);
    vec3s_(pack)(dest, rawDest, 2);
}

/*!
 * @brief crops a bounding box with another one.
 *
 * this could be useful for getting a bbox which fits with view frustum and
 * object bounding boxes. In this case you crop view frustum box with objects
 * box
 *
 * @param[in]  box      bounding box
 * @param[in]  cropBox  crop box
 * @param[in]  clampBox minimum box
 * @param[out] dest     cropped bounding box
 */
PLAY_CGLM_INLINE
void
aabbs_(crop_until)(vec3s box[2],
                   vec3s cropBox[2],
                   vec3s clampBox[2],
                   vec3s dest[2])
{
    aabbs_(crop)(box, cropBox, dest);
    aabbs_(merge)(clampBox, dest, dest);
}

/*!
 * @brief check if AABB intersects with frustum planes
 *
 * this could be useful for frustum culling using AABB.
 *
 * OPTIMIZATION HINT:
 *  if planes order is similar to LEFT, RIGHT, BOTTOM, TOP, NEAR, FAR
 *  then this method should run even faster because it would only use two
 *  planes if object is not inside the two planes
 *  fortunately cglm extracts planes as this order! just pass what you got!
 *
 * @param[in]  box     bounding box
 * @param[in]  planes  frustum planes
 */
PLAY_CGLM_INLINE
bool
aabbs_(frustum)(vec3s box[2], vec4s planes[6])
{
    vec3 rawBox[2];
    vec4 rawPlanes[6];

    vec3s_(unpack)(rawBox, box, 2);
    vec4s_(unpack)(rawPlanes, planes, 6);
    return aabb_frustum(rawBox, rawPlanes);
}

/*!
 * @brief invalidate AABB min and max values
 *
 * @param[in, out]  box bounding box
 */
PLAY_CGLM_INLINE
void
aabbs_(invalidate)(vec3s box[2])
{
    box[0] = vec3s_(broadcast)(FLT_MAX);
    box[1] = vec3s_(broadcast)(-FLT_MAX);
}

/*!
 * @brief check if AABB is valid or not
 *
 * @param[in]  box bounding box
 */
PLAY_CGLM_INLINE
bool
aabbs_(isvalid)(vec3s box[2])
{
    vec3 rawBox[2];
    vec3s_(unpack)(rawBox, box, 2);
    return aabb_isvalid(rawBox);
}

/*!
 * @brief distance between of min and max
 *
 * @param[in]  box bounding box
 */
PLAY_CGLM_INLINE
float
aabbs_(size)(vec3s box[2])
{
    return vec3_distance(box[0].raw, box[1].raw);
}

/*!
 * @brief radius of sphere which surrounds AABB
 *
 * @param[in]  box bounding box
 */
PLAY_CGLM_INLINE
float
aabbs_(radius)(vec3s box[2])
{
    return aabbs_(size)(box) * 0.5f;
}

/*!
 * @brief computes center point of AABB
 *
 * @param[in]   box  bounding box
 * @returns center of bounding box
 */
PLAY_CGLM_INLINE
vec3s
aabbs_(center)(vec3s box[2])
{
    return vec3s_(center)(box[0], box[1]);
}

/*!
 * @brief check if two AABB intersects
 *
 * @param[in]   box    bounding box
 * @param[in]   other  other bounding box
 */
PLAY_CGLM_INLINE
bool
aabbs_(aabb)(vec3s box[2], vec3s other[2])
{
    vec3 rawBox[2];
    vec3 rawOther[2];

    vec3s_(unpack)(rawBox, box, 2);
    vec3s_(unpack)(rawOther, other, 2);
    return aabb_aabb(rawBox, rawOther);
}

/*!
 * @brief check if AABB intersects with sphere
 *
 * https://github.com/erich666/GraphicsGems/blob/master/gems/BoxSphere.c
 * Solid Box - Solid Sphere test.
 *
 * @param[in]   box    solid bounding box
 * @param[in]   s      solid sphere
 */
PLAY_CGLM_INLINE
bool
aabbs_(sphere)(vec3s box[2], vec4s s)
{
    vec3 rawBox[2];

    vec3s_(unpack)(rawBox, box, 2);
    return aabb_sphere(rawBox, s.raw);
}

/*!
 * @brief check if point is inside of AABB
 *
 * @param[in]   box    bounding box
 * @param[in]   point  point
 */
PLAY_CGLM_INLINE
bool
aabbs_(point)(vec3s box[2], vec3s point)
{
    vec3 rawBox[2];

    vec3s_(unpack)(rawBox, box, 2);
    return aabb_point(rawBox, point.raw);
}

/*!
 * @brief check if AABB contains other AABB
 *
 * @param[in]   box    bounding box
 * @param[in]   other  other bounding box
 */
PLAY_CGLM_INLINE
bool
aabbs_(contains)(vec3s box[2], vec3s other[2])
{
    vec3 rawBox[2];
    vec3 rawOther[2];

    vec3s_(unpack)(rawBox, box, 2);
    vec3s_(unpack)(rawOther, other, 2);
    return aabb_contains(rawBox, rawOther);
}



/*** End of inlined file: box.h ***/


/*** Start of inlined file: color.h ***/



/*!
 * @brief averages the color channels into one value
 *
 * @param[in]  rgb RGB color
 */
PLAY_CGLM_INLINE
float
luminance(vec3s rgb)
{
    return luminance(rgb.raw);
}



/*** End of inlined file: color.h ***/


/*** Start of inlined file: cam.h ***/
/*
 Functions:
   PLAY_CGLM_INLINE mat4s frustum(float left,    float right,
                                  float bottom,  float top,
                                  float nearZ,   float farZ)
   PLAY_CGLM_INLINE mat4s ortho(float left,    float right,
                                float bottom,  float top,
                                float nearZ,   float farZ)
   PLAY_CGLM_INLINE mat4s ortho_aabb(vec3s box[2]);
   PLAY_CGLM_INLINE mat4s ortho_aabb_p(vec3s box[2],  float padding);
   PLAY_CGLM_INLINE mat4s ortho_aabb_pz(vec3s box[2], float padding);
   PLAY_CGLM_INLINE mat4s ortho_default(float aspect)
   PLAY_CGLM_INLINE mat4s ortho_default_s(float aspect, float size)
   PLAY_CGLM_INLINE mat4s perspective(float fovy,
                                      float aspect,
                                      float nearZ,
                                      float farZ)
   PLAY_CGLM_INLINE void  persp_move_far(mat4s proj, float deltaFar)
   PLAY_CGLM_INLINE mat4s perspective_default(float aspect)
   PLAY_CGLM_INLINE void  perspective_resize(mat4s proj, float aspect)
   PLAY_CGLM_INLINE mat4s lookat(vec3s eye, vec3s center, vec3s up)
   PLAY_CGLM_INLINE mat4s look(vec3s eye, vec3s dir, vec3s up)
   PLAY_CGLM_INLINE mat4s look_anyup(vec3s eye, vec3s dir)
   PLAY_CGLM_INLINE void  persp_decomp(mat4s  proj,
                                       float *nearv, float *farv,
                                       float *top,   float *bottom,
                                       float *left,  float *right)
   PLAY_CGLM_INLINE void  persp_decompv(mat4s proj, float dest[6])
   PLAY_CGLM_INLINE void  persp_decomp_x(mat4s proj, float *left, float *right)
   PLAY_CGLM_INLINE void  persp_decomp_y(mat4s proj, float *top, float *bottom)
   PLAY_CGLM_INLINE void  persp_decomp_z(mat4s proj, float *nearv, float *farv)
   PLAY_CGLM_INLINE void  persp_decomp_far(mat4s proj, float *farZ)
   PLAY_CGLM_INLINE void  persp_decomp_near(mat4s proj, float *nearZ)
   PLAY_CGLM_INLINE float persp_fovy(mat4s proj)
   PLAY_CGLM_INLINE float persp_aspect(mat4s proj)
   PLAY_CGLM_INLINE vec4s persp_sizes(mat4s proj, float fovy)
 */





/*** Start of inlined file: ortho_lh_zo.h ***/
/*
 Functions:
   PLAY_CGLM_INLINE mat4s ortho_lh_zo(float left,    float right,
                                      float bottom,  float top,
                                      float nearZ,   float farZ)
   PLAY_CGLM_INLINE mat4s ortho_aabb_lh_zo(vec3s box[2]);
   PLAY_CGLM_INLINE mat4s ortho_aabb_p_lh_zo(vec3s box[2],  float padding);
   PLAY_CGLM_INLINE mat4s ortho_aabb_pz_lh_zo(vec3s box[2], float padding);
   PLAY_CGLM_INLINE mat4s ortho_default_lh_zo(float aspect)
   PLAY_CGLM_INLINE mat4s ortho_default_s_lh_zo(float aspect, float size)
 */




/*!
 * @brief set up orthographic projection matrix
 *        with a left-hand coordinate system and a
 *        clip-space of [0, 1].
 *
 * @param[in]  left    viewport.left
 * @param[in]  right   viewport.right
 * @param[in]  bottom  viewport.bottom
 * @param[in]  top     viewport.top
 * @param[in]  nearZ   near clipping plane
 * @param[in]  farZ    far clipping plane
 * @returns    result matrix
 */
PLAY_CGLM_INLINE
mat4s
ortho_lh_zo(float left,   float right,
            float bottom, float top,
            float nearZ,  float farZ)
{
    mat4s dest;
    ortho_lh_zo(left, right, bottom, top, nearZ, farZ, dest.raw);
    return dest;
}

/*!
 * @brief set up orthographic projection matrix using bounding box
 *        with a left-hand coordinate system and a
 *        clip-space of [0, 1].
 *
 * bounding box (AABB) must be in view space
 *
 * @param[in]  box   AABB
 * @returns    result matrix
 */
PLAY_CGLM_INLINE
mat4s
ortho_aabb_lh_zo(vec3s box[2])
{
    mat4s dest;
    vec3  rawBox[2];

    vec3s_(unpack)(rawBox, box, 2);
    ortho_aabb_lh_zo(rawBox, dest.raw);

    return dest;
}

/*!
 * @brief set up orthographic projection matrix using bounding box
 *        with a left-hand coordinate system and a
 *        clip-space of [0, 1].
 *
 * bounding box (AABB) must be in view space
 *
 * @param[in]  box     AABB
 * @param[in]  padding padding
 * @returns    result matrix
 */
PLAY_CGLM_INLINE
mat4s
ortho_aabb_p_lh_zo(vec3s box[2], float padding)
{
    mat4s dest;
    vec3  rawBox[2];

    vec3s_(unpack)(rawBox, box, 2);
    ortho_aabb_p_lh_zo(rawBox, padding, dest.raw);

    return dest;
}

/*!
 * @brief set up orthographic projection matrix using bounding box
 *        with a left-hand coordinate system and a
 *        clip-space of [0, 1].
 *
 * bounding box (AABB) must be in view space
 *
 * @param[in]  box     AABB
 * @param[in]  padding padding for near and far
 * @returns    result matrix
 */
PLAY_CGLM_INLINE
mat4s
ortho_aabb_pz_lh_zo(vec3s box[2], float padding)
{
    mat4s dest;
    vec3  rawBox[2];

    vec3s_(unpack)(rawBox, box, 2);
    ortho_aabb_pz_lh_zo(rawBox, padding, dest.raw);

    return dest;
}

/*!
 * @brief set up unit orthographic projection matrix
 *        with a left-hand coordinate system and a
 *        clip-space of [0, 1].
 *
 * @param[in]  aspect aspect ration ( width / height )
 * @returns    result matrix
 */
PLAY_CGLM_INLINE
mat4s
ortho_default_lh_zo(float aspect)
{
    mat4s dest;
    ortho_default_lh_zo(aspect, dest.raw);
    return dest;
}

/*!
 * @brief set up orthographic projection matrix with given CUBE size
 *        with a left-hand coordinate system and a
 *        clip-space of [0, 1].
 *
 * @param[in]  aspect aspect ratio ( width / height )
 * @param[in]  size   cube size
 * @returns    result matrix
 */
PLAY_CGLM_INLINE
mat4s
ortho_default_s_lh_zo(float aspect, float size)
{
    mat4s dest;
    ortho_default_s_lh_zo(aspect, size, dest.raw);
    return dest;
}



/*** End of inlined file: ortho_lh_zo.h ***/


/*** Start of inlined file: persp_lh_zo.h ***/
/*
 Functions:
   PLAY_CGLM_INLINE mat4s frustum_lh_zo(float left,    float right,
                                        float bottom,  float top,
                                        float nearZ,   float farZ)
   PLAY_CGLM_INLINE mat4s perspective_lh_zo(float fovy,
                                            float aspect,
                                            float nearZ,
                                            float farZ)
   PLAY_CGLM_INLINE void  persp_move_far_lh_zo(mat4s proj, float deltaFar)
   PLAY_CGLM_INLINE mat4s perspective_default_lh_zo(float aspect)
   PLAY_CGLM_INLINE void  perspective_resize_lh_zo(mat4s proj, float aspect)
   PLAY_CGLM_INLINE void  persp_decomp_lh_zo(mat4s  proj,
                                             float *nearv, float *farv,
                                             float *top,   float *bottom,
                                             float *left,  float *right)
   PLAY_CGLM_INLINE void  persp_decompv_lh_zo(mat4s proj, float dest[6])
   PLAY_CGLM_INLINE void  persp_decomp_x_lh_zo(mat4s proj, float *left, float *right)
   PLAY_CGLM_INLINE void  persp_decomp_y_lh_zo(mat4s proj, float *top, float *bottom)
   PLAY_CGLM_INLINE void  persp_decomp_z_lh_zo(mat4s proj, float *nearv, float *farv)
   PLAY_CGLM_INLINE void  persp_decomp_far_lh_zo(mat4s proj, float *farZ)
   PLAY_CGLM_INLINE void  persp_decomp_near_lh_zo(mat4s proj, float *nearZ)
   PLAY_CGLM_INLINE float persp_fovy_lh_zo(mat4s proj)
   PLAY_CGLM_INLINE float persp_aspect_lh_zo(mat4s proj)
   PLAY_CGLM_INLINE vec4s persp_sizes_lh_zo(mat4s proj, float fovy)
 */




/*!
 * @brief set up perspective peprojection matrix
 *        with a left-hand coordinate system and a
 *        clip-space of [0, 1].
 *
 * @param[in]  left    viewport.left
 * @param[in]  right   viewport.right
 * @param[in]  bottom  viewport.bottom
 * @param[in]  top     viewport.top
 * @param[in]  nearZ   near clipping plane
 * @param[in]  farZ    far clipping plane
 * @returns    result matrix
 */
PLAY_CGLM_INLINE
mat4s
frustum_lh_zo(float left,   float right,
              float bottom, float top,
              float nearZ,  float farZ)
{
    mat4s dest;
    frustum_lh_zo(left, right, bottom, top, nearZ, farZ, dest.raw);
    return dest;
}

/*!
 * @brief set up perspective projection matrix
 *        with a left-hand coordinate system and a
 *        clip-space of [0, 1].
 *
 * @param[in]  fovy    field of view angle
 * @param[in]  aspect  aspect ratio ( width / height )
 * @param[in]  nearZ   near clipping plane
 * @param[in]  farZ    far clipping planes
 * @returns    result matrix
 */
PLAY_CGLM_INLINE
mat4s
perspective_lh_zo(float fovy, float aspect, float nearZ, float farZ)
{
    mat4s dest;
    perspective_lh_zo(fovy, aspect, nearZ, farZ, dest.raw);
    return dest;
}

/*!
 * @brief extend perspective projection matrix's far distance
 *        with a left-hand coordinate system and a
 *        clip-space of [0, 1].
 *
 * NOTE: if you dodn't want to create new matrix then use array api on struct.raw
 *       like persp_move_far_lh_zo(prooj.raw, deltaFar) to avoid create new mat4
 *       each time
 *
 * this function does not guarantee far >= near, be aware of that!
 *
 * @param[in, out] proj      projection matrix to extend
 * @param[in]      deltaFar  distance from existing far (negative to shink)
 */
PLAY_CGLM_INLINE
mat4s
persp_move_far_lh_zo(mat4s proj, float deltaFar)
{
    mat4s dest;
    dest = proj;
    persp_move_far_lh_zo(dest.raw, deltaFar);
    return dest;
}

/*!
 * @brief set up perspective projection matrix with default near/far
 *        and angle values with a left-hand coordinate system and a
 *        clip-space of [0, 1].
 *
 * @param[in]  aspect aspect ratio ( width / height )
 * @returns    result matrix
 */
PLAY_CGLM_INLINE
mat4s
perspective_default_lh_zo(float aspect)
{
    mat4s dest;
    perspective_default_lh_zo(aspect, dest.raw);
    return dest;
}

/*!
 * @brief resize perspective matrix by aspect ratio ( width / height )
 *        this makes very easy to resize proj matrix when window /viewport
 *        reized with a left-hand coordinate system and a
 *        clip-space of [0, 1].
 *
 * NOTE: if you dodn't want to create new matrix then use array api on struct.raw
 *       like perspective_resize_lh_zo(proj.raw, aspect) to avoid create new mat4
 *       each time
 *
 * @param[in, out] proj   perspective projection matrix
 * @param[in]      aspect aspect ratio ( width / height )
 */
PLAY_CGLM_INLINE
mat4s
perspective_resize_lh_zo(mat4s proj, float aspect)
{
    mat4s dest;
    dest = proj;
    perspective_resize_lh_zo(aspect, dest.raw);
    return dest;
}

/*!
 * @brief decomposes frustum values of perspective projection.
 *        with a left-hand coordinate system and a
 *        clip-space of [0, 1].
 *
 * @param[in]  proj    perspective projection matrix
 * @param[out] nearZ   near
 * @param[out] farZ    far
 * @param[out] top     top
 * @param[out] bottom  bottom
 * @param[out] left    left
 * @param[out] right   right
 */
PLAY_CGLM_INLINE
void
persp_decomp_lh_zo(mat4s proj,
                   float * __restrict nearZ, float * __restrict farZ,
                   float * __restrict top,   float * __restrict bottom,
                   float * __restrict left,  float * __restrict right)
{
    persp_decomp_lh_zo(proj.raw, nearZ, farZ, top, bottom, left, right);
}

/*!
 * @brief decomposes frustum values of perspective projection.
 *        this makes easy to get all values at once
 *        with a left-hand coordinate system and a
 *        clip-space of [0, 1].
 *
 * @param[in]  proj   perspective projection matrix
 * @param[out] dest   array
 */
PLAY_CGLM_INLINE
void
persp_decompv_lh_zo(mat4s proj, float dest[6])
{
    persp_decompv_lh_zo(proj.raw, dest);
}

/*!
 * @brief decomposes left and right values of perspective projection
 *        with a left-hand coordinate system and a
 *        clip-space of [0, 1].
 *        x stands for x axis (left / right axis)
 *
 * @param[in]  proj  perspective projection matrix
 * @param[out] left  left
 * @param[out] right right
 */
PLAY_CGLM_INLINE
void
persp_decomp_x_lh_zo(mat4s proj,
                     float * __restrict left,
                     float * __restrict right)
{
    persp_decomp_x_lh_zo(proj.raw, left, right);
}

/*!
 * @brief decomposes top and bottom values of perspective projection
 *        with a left-hand coordinate system and a
 *        clip-space of [0, 1].
 *        y stands for y axis (top / bottom axis)
 *
 * @param[in]  proj   perspective projection matrix
 * @param[out] top    top
 * @param[out] bottom bottom
 */
PLAY_CGLM_INLINE
void
persp_decomp_y_lh_zo(mat4s proj,
                     float * __restrict top,
                     float * __restrict bottom)
{
    persp_decomp_y_lh_zo(proj.raw, top, bottom);
}

/*!
 * @brief decomposes near and far values of perspective projection
 *        with a left-hand coordinate system and a
 *        clip-space of [0, 1].
 *        z stands for z axis (near / far axis)
 *
 * @param[in]  proj    perspective projection matrix
 * @param[out] nearZ   near
 * @param[out] farZ    far
 */
PLAY_CGLM_INLINE
void
persp_decomp_z_lh_zo(mat4s proj,
                     float * __restrict nearZ,
                     float * __restrict farZ)
{
    persp_decomp_z_lh_zo(proj.raw, nearZ, farZ);
}

/*!
 * @brief decomposes far value of perspective projection
 *        with a left-hand coordinate system and a
 *        clip-space of [0, 1].
 *
 * @param[in]  proj   perspective projection matrix
 * @param[out] farZ   far
 */
PLAY_CGLM_INLINE
void
persp_decomp_far_lh_zo(mat4s proj, float * __restrict farZ)
{
    persp_decomp_far_lh_zo(proj.raw, farZ);
}

/*!
 * @brief decomposes near value of perspective projection
 *        with a left-hand coordinate system and a
 *        clip-space of [0, 1].
 *
 * @param[in]  proj  perspective projection matrix
 * @param[out] nearZ near
 */
PLAY_CGLM_INLINE
void
persp_decomp_near_lh_zo(mat4s proj, float * __restrict nearZ)
{
    persp_decomp_near_lh_zo(proj.raw, nearZ);
}

/*!
 * @brief returns field of view angle along the Y-axis (in radians)
 *        with a left-hand coordinate system and a
 *        clip-space of [0, 1].
 *
 * if you need to degrees, use deg to convert it or use this:
 * fovy_deg = deg(persp_fovy(projMatrix))
 *
 * @param[in] proj perspective projection matrix
 */
PLAY_CGLM_INLINE
float
persp_fovy_lh_zo(mat4s proj)
{
    return persp_fovy_lh_zo(proj.raw);
}

/*!
 * @brief returns aspect ratio of perspective projection
 *        with a left-hand coordinate system and a
 *        clip-space of [0, 1].
 *
 * @param[in] proj perspective projection matrix
 */
PLAY_CGLM_INLINE
float
persp_aspect_lh_zo(mat4s proj)
{
    return persp_aspect_lh_zo(proj.raw);
}

/*!
 * @brief returns sizes of near and far planes of perspective projection
 *        with a left-hand coordinate system and a
 *        clip-space of [0, 1].
 *
 * @param[in]  proj perspective projection matrix
 * @param[in]  fovy fovy (see brief)
 * @returns    sizes as vector, sizes order: [Wnear, Hnear, Wfar, Hfar]
 */
PLAY_CGLM_INLINE
vec4s
persp_sizes_lh_zo(mat4s proj, float fovy)
{
    vec4s dest;
    persp_sizes_lh_zo(proj.raw, fovy, dest.raw);
    return dest;
}



/*** End of inlined file: persp_lh_zo.h ***/


/*** Start of inlined file: ortho_lh_no.h ***/
/*
 Functions:
   PLAY_CGLM_INLINE mat4s ortho_lh_no(float left,    float right,
                                      float bottom,  float top,
                                      float nearZ,   float farZ)
   PLAY_CGLM_INLINE mat4s ortho_aabb_lh_no(vec3s box[2]);
   PLAY_CGLM_INLINE mat4s ortho_aabb_p_lh_no(vec3s box[2],  float padding);
   PLAY_CGLM_INLINE mat4s ortho_aabb_pz_lh_no(vec3s box[2], float padding);
   PLAY_CGLM_INLINE mat4s ortho_default_lh_no(float aspect)
   PLAY_CGLM_INLINE mat4s ortho_default_s_lh_no(float aspect, float size)
 */




/*!
 * @brief set up orthographic projection matrix
 *        with a left-hand coordinate system and a
 *        clip-space of [-1, 1].
 *
 * @param[in]  left    viewport.left
 * @param[in]  right   viewport.right
 * @param[in]  bottom  viewport.bottom
 * @param[in]  top     viewport.top
 * @param[in]  nearZ   near clipping plane
 * @param[in]  farZ    far clipping plane
 * @returns    result matrix
 */
PLAY_CGLM_INLINE
mat4s
ortho_lh_no(float left,   float right,
            float bottom, float top,
            float nearZ,  float farZ)
{
    mat4s dest;
    ortho_lh_no(left, right, bottom, top, nearZ, farZ, dest.raw);
    return dest;
}

/*!
 * @brief set up orthographic projection matrix using bounding box
 *        with a left-hand coordinate system and a
 *        clip-space of [-1, 1].
 *
 * bounding box (AABB) must be in view space
 *
 * @param[in]  box   AABB
 * @returns    result matrix
 */
PLAY_CGLM_INLINE
mat4s
ortho_aabb_lh_no(vec3s box[2])
{
    mat4s dest;
    vec3  rawBox[2];

    vec3s_(unpack)(rawBox, box, 2);
    ortho_aabb_lh_no(rawBox, dest.raw);

    return dest;
}

/*!
 * @brief set up orthographic projection matrix using bounding box
 *        with a left-hand coordinate system and a
 *        clip-space of [-1, 1].
 *
 * bounding box (AABB) must be in view space
 *
 * @param[in]  box     AABB
 * @param[in]  padding padding
 * @returns    result matrix
 */
PLAY_CGLM_INLINE
mat4s
ortho_aabb_p_lh_no(vec3s box[2], float padding)
{
    mat4s dest;
    vec3  rawBox[2];

    vec3s_(unpack)(rawBox, box, 2);
    ortho_aabb_p_lh_no(rawBox, padding, dest.raw);

    return dest;
}

/*!
 * @brief set up orthographic projection matrix using bounding box
 *        with a left-hand coordinate system and a
 *        clip-space of [-1, 1].
 *
 * bounding box (AABB) must be in view space
 *
 * @param[in]  box     AABB
 * @param[in]  padding padding for near and far
 * @returns    result matrix
 */
PLAY_CGLM_INLINE
mat4s
ortho_aabb_pz_lh_no(vec3s box[2], float padding)
{
    mat4s dest;
    vec3  rawBox[2];

    vec3s_(unpack)(rawBox, box, 2);
    ortho_aabb_pz_lh_no(rawBox, padding, dest.raw);

    return dest;
}

/*!
 * @brief set up unit orthographic projection matrix
 *        with a left-hand coordinate system and a
 *        clip-space of [-1, 1].
 *
 * @param[in]  aspect aspect ration ( width / height )
 * @returns    result matrix
 */
PLAY_CGLM_INLINE
mat4s
ortho_default_lh_no(float aspect)
{
    mat4s dest;
    ortho_default_lh_no(aspect, dest.raw);
    return dest;
}

/*!
 * @brief set up orthographic projection matrix with given CUBE size
 *        with a left-hand coordinate system and a
 *        clip-space of [-1, 1].
 *
 * @param[in]  aspect aspect ratio ( width / height )
 * @param[in]  size   cube size
 * @returns    result matrix
 */
PLAY_CGLM_INLINE
mat4s
ortho_default_s_lh_no(float aspect, float size)
{
    mat4s dest;
    ortho_default_s_lh_no(aspect, size, dest.raw);
    return dest;
}



/*** End of inlined file: ortho_lh_no.h ***/


/*** Start of inlined file: persp_lh_no.h ***/
/*
 Functions:
   PLAY_CGLM_INLINE mat4s frustum_lh_no(float left,    float right,
                                        float bottom,  float top,
                                        float nearZ,   float farZ)
   PLAY_CGLM_INLINE mat4s perspective_lh_no(float fovy,
                                            float aspect,
                                            float nearZ,
                                            float farZ)
   PLAY_CGLM_INLINE void  persp_move_far_lh_no(mat4s proj, float deltaFar)
   PLAY_CGLM_INLINE mat4s perspective_default_lh_no(float aspect)
   PLAY_CGLM_INLINE void  perspective_resize_lh_no(mat4s proj, float aspect)
   PLAY_CGLM_INLINE void  persp_decomp_lh_no(mat4s  proj,
                                             float *nearv, float *farv,
                                             float *top,   float *bottom,
                                             float *left,  float *right)
   PLAY_CGLM_INLINE void  persp_decompv_lh_no(mat4s proj, float dest[6])
   PLAY_CGLM_INLINE void  persp_decomp_x_lh_no(mat4s proj, float *left, float *right)
   PLAY_CGLM_INLINE void  persp_decomp_y_lh_no(mat4s proj, float *top, float *bottom)
   PLAY_CGLM_INLINE void  persp_decomp_z_lh_no(mat4s proj, float *nearv, float *farv)
   PLAY_CGLM_INLINE void  persp_decomp_far_lh_no(mat4s proj, float *farZ)
   PLAY_CGLM_INLINE void  persp_decomp_near_lh_no(mat4s proj, float *nearZ)
   PLAY_CGLM_INLINE float persp_fovy_lh_no(mat4s proj)
   PLAY_CGLM_INLINE float persp_aspect_lh_no(mat4s proj)
   PLAY_CGLM_INLINE vec4s persp_sizes_lh_no(mat4s proj, float fovy)
 */




/*!
 * @brief set up perspective peprojection matrix
 *        with a left-hand coordinate system and a
 *        clip-space of [-1, 1].
 *
 * @param[in]  left    viewport.left
 * @param[in]  right   viewport.right
 * @param[in]  bottom  viewport.bottom
 * @param[in]  top     viewport.top
 * @param[in]  nearZ   near clipping plane
 * @param[in]  farZ    far clipping plane
 * @returns    result matrix
 */
PLAY_CGLM_INLINE
mat4s
frustum_lh_no(float left,   float right,
              float bottom, float top,
              float nearZ,  float farZ)
{
    mat4s dest;
    frustum_lh_no(left, right, bottom, top, nearZ, farZ, dest.raw);
    return dest;
}

/*!
 * @brief set up perspective projection matrix
 *        with a left-hand coordinate system and a
 *        clip-space of [-1, 1].
 *
 * @param[in]  fovy    field of view angle
 * @param[in]  aspect  aspect ratio ( width / height )
 * @param[in]  nearZ   near clipping plane
 * @param[in]  farZ    far clipping planes
 * @returns    result matrix
 */
PLAY_CGLM_INLINE
mat4s
perspective_lh_no(float fovy, float aspect, float nearZ, float farZ)
{
    mat4s dest;
    perspective_lh_no(fovy, aspect, nearZ, farZ, dest.raw);
    return dest;
}

/*!
 * @brief extend perspective projection matrix's far distance
 *        with a left-hand coordinate system and a
 *        clip-space of [-1, 1].
 *
 * NOTE: if you dodn't want to create new matrix then use array api on struct.raw
 *       like persp_move_far_lh_no(prooj.raw, deltaFar) to avoid create new mat4
 *       each time
 *
 * this function does not guarantee far >= near, be aware of that!
 *
 * @param[in, out] proj      projection matrix to extend
 * @param[in]      deltaFar  distance from existing far (negative to shink)
 */
PLAY_CGLM_INLINE
mat4s
persp_move_far_lh_no(mat4s proj, float deltaFar)
{
    mat4s dest;
    dest = proj;
    persp_move_far_lh_no(dest.raw, deltaFar);
    return dest;
}

/*!
 * @brief set up perspective projection matrix with default near/far
 *        and angle values with a left-hand coordinate system and a
 *        clip-space of [-1, 1].
 *
 * @param[in]  aspect aspect ratio ( width / height )
 * @returns    result matrix
 */
PLAY_CGLM_INLINE
mat4s
perspective_default_lh_no(float aspect)
{
    mat4s dest;
    perspective_default_lh_no(aspect, dest.raw);
    return dest;
}

/*!
 * @brief resize perspective matrix by aspect ratio ( width / height )
 *        this makes very easy to resize proj matrix when window /viewport
 *        reized with a left-hand coordinate system and a
 *        clip-space of [-1, 1].
 *
 * NOTE: if you dodn't want to create new matrix then use array api on struct.raw
 *       like perspective_resize_lh_no(proj.raw, aspect) to avoid create new mat4
 *       each time
 *
 * @param[in, out] proj   perspective projection matrix
 * @param[in]      aspect aspect ratio ( width / height )
 */
PLAY_CGLM_INLINE
mat4s
perspective_resize_lh_no(mat4s proj, float aspect)
{
    mat4s dest;
    dest = proj;
    perspective_resize_lh_no(aspect, dest.raw);
    return dest;
}

/*!
 * @brief decomposes frustum values of perspective projection.
 *        with a left-hand coordinate system and a
 *        clip-space of [-1, 1].
 *
 * @param[in]  proj    perspective projection matrix
 * @param[out] nearZ   near
 * @param[out] farZ    far
 * @param[out] top     top
 * @param[out] bottom  bottom
 * @param[out] left    left
 * @param[out] right   right
 */
PLAY_CGLM_INLINE
void
persp_decomp_lh_no(mat4s proj,
                   float * __restrict nearZ, float * __restrict farZ,
                   float * __restrict top,   float * __restrict bottom,
                   float * __restrict left,  float * __restrict right)
{
    persp_decomp_lh_no(proj.raw, nearZ, farZ, top, bottom, left, right);
}

/*!
 * @brief decomposes frustum values of perspective projection.
 *        this makes easy to get all values at once
 *        with a left-hand coordinate system and a
 *        clip-space of [-1, 1].
 *
 * @param[in]  proj   perspective projection matrix
 * @param[out] dest   array
 */
PLAY_CGLM_INLINE
void
persp_decompv_lh_no(mat4s proj, float dest[6])
{
    persp_decompv_lh_no(proj.raw, dest);
}

/*!
 * @brief decomposes left and right values of perspective projection
 *        with a left-hand coordinate system and a
 *        clip-space of [-1, 1].
 *        x stands for x axis (left / right axis)
 *
 * @param[in]  proj  perspective projection matrix
 * @param[out] left  left
 * @param[out] right right
 */
PLAY_CGLM_INLINE
void
persp_decomp_x_lh_no(mat4s proj,
                     float * __restrict left,
                     float * __restrict right)
{
    persp_decomp_x_lh_no(proj.raw, left, right);
}

/*!
 * @brief decomposes top and bottom values of perspective projection
 *        with a left-hand coordinate system and a
 *        clip-space of [-1, 1].
 *        y stands for y axis (top / bottom axis)
 *
 * @param[in]  proj   perspective projection matrix
 * @param[out] top    top
 * @param[out] bottom bottom
 */
PLAY_CGLM_INLINE
void
persp_decomp_y_lh_no(mat4s proj,
                     float * __restrict top,
                     float * __restrict bottom)
{
    persp_decomp_y_lh_no(proj.raw, top, bottom);
}

/*!
 * @brief decomposes near and far values of perspective projection
 *        with a left-hand coordinate system and a
 *        clip-space of [-1, 1].
 *        z stands for z axis (near / far axis)
 *
 * @param[in]  proj    perspective projection matrix
 * @param[out] nearZ   near
 * @param[out] farZ    far
 */
PLAY_CGLM_INLINE
void
persp_decomp_z_lh_no(mat4s proj,
                     float * __restrict nearZ,
                     float * __restrict farZ)
{
    persp_decomp_z_lh_no(proj.raw, nearZ, farZ);
}

/*!
 * @brief decomposes far value of perspective projection
 *        with a left-hand coordinate system and a
 *        clip-space of [-1, 1].
 *
 * @param[in]  proj   perspective projection matrix
 * @param[out] farZ   far
 */
PLAY_CGLM_INLINE
void
persp_decomp_far_lh_no(mat4s proj, float * __restrict farZ)
{
    persp_decomp_far_lh_no(proj.raw, farZ);
}

/*!
 * @brief decomposes near value of perspective projection
 *        with a left-hand coordinate system and a
 *        clip-space of [-1, 1].
 *
 * @param[in]  proj  perspective projection matrix
 * @param[out] nearZ near
 */
PLAY_CGLM_INLINE
void
persp_decomp_near_lh_no(mat4s proj, float * __restrict nearZ)
{
    persp_decomp_near_lh_no(proj.raw, nearZ);
}

/*!
 * @brief returns field of view angle along the Y-axis (in radians)
 *        with a left-hand coordinate system and a
 *        clip-space of [-1, 1].
 *
 * if you need to degrees, use deg to convert it or use this:
 * fovy_deg = deg(persp_fovy(projMatrix))
 *
 * @param[in] proj perspective projection matrix
 */
PLAY_CGLM_INLINE
float
persp_fovy_lh_no(mat4s proj)
{
    return persp_fovy_lh_no(proj.raw);
}

/*!
 * @brief returns aspect ratio of perspective projection
 *        with a left-hand coordinate system and a
 *        clip-space of [-1, 1].
 *
 * @param[in] proj perspective projection matrix
 */
PLAY_CGLM_INLINE
float
persp_aspect_lh_no(mat4s proj)
{
    return persp_aspect_lh_no(proj.raw);
}

/*!
 * @brief returns sizes of near and far planes of perspective projection
 *        with a left-hand coordinate system and a
 *        clip-space of [-1, 1].
 *
 * @param[in]  proj perspective projection matrix
 * @param[in]  fovy fovy (see brief)
 * @returns    sizes as vector, sizes order: [Wnear, Hnear, Wfar, Hfar]
 */
PLAY_CGLM_INLINE
vec4s
persp_sizes_lh_no(mat4s proj, float fovy)
{
    vec4s dest;
    persp_sizes_lh_no(proj.raw, fovy, dest.raw);
    return dest;
}



/*** End of inlined file: persp_lh_no.h ***/


/*** Start of inlined file: ortho_rh_zo.h ***/
/*
 Functions:
   PLAY_CGLM_INLINE mat4s ortho_rh_zo(float left,    float right,
                                      float bottom,  float top,
                                      float nearZ,   float farZ)
   PLAY_CGLM_INLINE mat4s ortho_aabb_rh_zo(vec3s box[2]);
   PLAY_CGLM_INLINE mat4s ortho_aabb_p_rh_zo(vec3s box[2],  float padding);
   PLAY_CGLM_INLINE mat4s ortho_aabb_pz_rh_zo(vec3s box[2], float padding);
   PLAY_CGLM_INLINE mat4s ortho_default_rh_zo(float aspect)
   PLAY_CGLM_INLINE mat4s ortho_default_s_rh_zo(float aspect, float size)
 */




/*!
 * @brief set up orthographic projection matrix
 *        with a right-hand coordinate system and a
 *        clip-space of [0, 1].
 *
 * @param[in]  left    viewport.left
 * @param[in]  right   viewport.right
 * @param[in]  bottom  viewport.bottom
 * @param[in]  top     viewport.top
 * @param[in]  nearZ   near clipping plane
 * @param[in]  farZ    far clipping plane
 * @returns    result matrix
 */
PLAY_CGLM_INLINE
mat4s
ortho_rh_zo(float left,   float right,
            float bottom, float top,
            float nearZ,  float farZ)
{
    mat4s dest;
    ortho_rh_zo(left, right, bottom, top, nearZ, farZ, dest.raw);
    return dest;
}

/*!
 * @brief set up orthographic projection matrix using bounding box
 *        with a right-hand coordinate system and a
 *        clip-space of [0, 1].
 *
 * bounding box (AABB) must be in view space
 *
 * @param[in]  box   AABB
 * @returns    result matrix
 */
PLAY_CGLM_INLINE
mat4s
ortho_aabb_rh_zo(vec3s box[2])
{
    mat4s dest;
    vec3  rawBox[2];

    vec3s_(unpack)(rawBox, box, 2);
    ortho_aabb_rh_zo(rawBox, dest.raw);

    return dest;
}

/*!
 * @brief set up orthographic projection matrix using bounding box
 *        with a right-hand coordinate system and a
 *        clip-space of [0, 1].
 *
 * bounding box (AABB) must be in view space
 *
 * @param[in]  box     AABB
 * @param[in]  padding padding
 * @returns    result matrix
 */
PLAY_CGLM_INLINE
mat4s
ortho_aabb_p_rh_zo(vec3s box[2], float padding)
{
    mat4s dest;
    vec3  rawBox[2];

    vec3s_(unpack)(rawBox, box, 2);
    ortho_aabb_p_rh_zo(rawBox, padding, dest.raw);

    return dest;
}

/*!
 * @brief set up orthographic projection matrix using bounding box
 *        with a right-hand coordinate system and a
 *        clip-space of [0, 1].
 *
 * bounding box (AABB) must be in view space
 *
 * @param[in]  box     AABB
 * @param[in]  padding padding for near and far
 * @returns    result matrix
 */
PLAY_CGLM_INLINE
mat4s
ortho_aabb_pz_rh_zo(vec3s box[2], float padding)
{
    mat4s dest;
    vec3  rawBox[2];

    vec3s_(unpack)(rawBox, box, 2);
    ortho_aabb_pz_rh_zo(rawBox, padding, dest.raw);

    return dest;
}

/*!
 * @brief set up unit orthographic projection matrix
 *        with a right-hand coordinate system and a
 *        clip-space of [0, 1].
 *
 * @param[in]  aspect aspect ration ( width / height )
 * @returns    result matrix
 */
PLAY_CGLM_INLINE
mat4s
ortho_default_rh_zo(float aspect)
{
    mat4s dest;
    ortho_default_rh_zo(aspect, dest.raw);
    return dest;
}

/*!
 * @brief set up orthographic projection matrix with given CUBE size
 *        with a right-hand coordinate system and a
 *        clip-space of [0, 1].
 *
 * @param[in]  aspect aspect ratio ( width / height )
 * @param[in]  size   cube size
 * @returns    result matrix
 */
PLAY_CGLM_INLINE
mat4s
ortho_default_s_rh_zo(float aspect, float size)
{
    mat4s dest;
    ortho_default_s_rh_zo(aspect, size, dest.raw);
    return dest;
}



/*** End of inlined file: ortho_rh_zo.h ***/


/*** Start of inlined file: persp_rh_zo.h ***/
/*
 Functions:
   PLAY_CGLM_INLINE mat4s frustum_rh_zo(float left,    float right,
                                        float bottom,  float top,
                                        float nearZ,   float farZ)
   PLAY_CGLM_INLINE mat4s perspective_rh_zo(float fovy,
                                            float aspect,
                                            float nearZ,
                                            float farZ)
   PLAY_CGLM_INLINE void  persp_move_far_rh_zo(mat4s proj, float deltaFar)
   PLAY_CGLM_INLINE mat4s perspective_default_rh_zo(float aspect)
   PLAY_CGLM_INLINE void  perspective_resize_rh_zo(mat4s proj, float aspect)
   PLAY_CGLM_INLINE void  persp_decomp_rh_zo(mat4s  proj,
                                             float *nearv, float *farv,
                                             float *top,   float *bottom,
                                             float *left,  float *right)
   PLAY_CGLM_INLINE void  persp_decompv_rh_zo(mat4s proj, float dest[6])
   PLAY_CGLM_INLINE void  persp_decomp_x_rh_zo(mat4s proj, float *left, float *right)
   PLAY_CGLM_INLINE void  persp_decomp_y_rh_zo(mat4s proj, float *top, float *bottom)
   PLAY_CGLM_INLINE void  persp_decomp_z_rh_zo(mat4s proj, float *nearv, float *farv)
   PLAY_CGLM_INLINE void  persp_decomp_far_rh_zo(mat4s proj, float *farZ)
   PLAY_CGLM_INLINE void  persp_decomp_near_rh_zo(mat4s proj, float *nearZ)
   PLAY_CGLM_INLINE float persp_fovy_rh_zo(mat4s proj)
   PLAY_CGLM_INLINE float persp_aspect_rh_zo(mat4s proj)
   PLAY_CGLM_INLINE vec4s persp_sizes_rh_zo(mat4s proj, float fovy)
 */




/*!
 * @brief set up perspective peprojection matrix
 *        with a right-hand coordinate system and a
 *        clip-space of [0, 1].
 *
 * @param[in]  left    viewport.left
 * @param[in]  right   viewport.right
 * @param[in]  bottom  viewport.bottom
 * @param[in]  top     viewport.top
 * @param[in]  nearZ   near clipping plane
 * @param[in]  farZ    far clipping plane
 * @returns    result matrix
 */
PLAY_CGLM_INLINE
mat4s
frustum_rh_zo(float left,   float right,
              float bottom, float top,
              float nearZ,  float farZ)
{
    mat4s dest;
    frustum_rh_zo(left, right, bottom, top, nearZ, farZ, dest.raw);
    return dest;
}

/*!
 * @brief set up perspective projection matrix
 *        with a right-hand coordinate system and a
 *        clip-space of [0, 1].
 *
 * @param[in]  fovy    field of view angle
 * @param[in]  aspect  aspect ratio ( width / height )
 * @param[in]  nearZ   near clipping plane
 * @param[in]  farZ    far clipping planes
 * @returns    result matrix
 */
PLAY_CGLM_INLINE
mat4s
perspective_rh_zo(float fovy, float aspect, float nearZ, float farZ)
{
    mat4s dest;
    perspective_rh_zo(fovy, aspect, nearZ, farZ, dest.raw);
    return dest;
}

/*!
 * @brief extend perspective projection matrix's far distance
 *        with a right-hand coordinate system and a
 *        clip-space of [0, 1].
 *
 * NOTE: if you dodn't want to create new matrix then use array api on struct.raw
 *       like persp_move_far_rh_zo(prooj.raw, deltaFar) to avoid create new mat4
 *       each time
 *
 * this function does not guarantee far >= near, be aware of that!
 *
 * @param[in, out] proj      projection matrix to extend
 * @param[in]      deltaFar  distance from existing far (negative to shink)
 */
PLAY_CGLM_INLINE
mat4s
persp_move_far_rh_zo(mat4s proj, float deltaFar)
{
    mat4s dest;
    dest = proj;
    persp_move_far_rh_zo(dest.raw, deltaFar);
    return dest;
}

/*!
 * @brief set up perspective projection matrix with default near/far
 *        and angle values with a right-hand coordinate system and a
 *        clip-space of [0, 1].
 *
 * @param[in]  aspect aspect ratio ( width / height )
 * @returns    result matrix
 */
PLAY_CGLM_INLINE
mat4s
perspective_default_rh_zo(float aspect)
{
    mat4s dest;
    perspective_default_rh_zo(aspect, dest.raw);
    return dest;
}

/*!
 * @brief resize perspective matrix by aspect ratio ( width / height )
 *        this makes very easy to resize proj matrix when window /viewport
 *        reized with a right-hand coordinate system and a
 *        clip-space of [0, 1].
 *
 * NOTE: if you dodn't want to create new matrix then use array api on struct.raw
 *       like perspective_resize_rh_zo(proj.raw, aspect) to avoid create new mat4
 *       each time
 *
 * @param[in, out] proj   perspective projection matrix
 * @param[in]      aspect aspect ratio ( width / height )
 */
PLAY_CGLM_INLINE
mat4s
perspective_resize_rh_zo(mat4s proj, float aspect)
{
    mat4s dest;
    dest = proj;
    perspective_resize_rh_zo(aspect, dest.raw);
    return dest;
}

/*!
 * @brief decomposes frustum values of perspective projection.
 *        with a right-hand coordinate system and a
 *        clip-space of [0, 1].
 *
 * @param[in]  proj    perspective projection matrix
 * @param[out] nearZ   near
 * @param[out] farZ    far
 * @param[out] top     top
 * @param[out] bottom  bottom
 * @param[out] left    left
 * @param[out] right   right
 */
PLAY_CGLM_INLINE
void
persp_decomp_rh_zo(mat4s proj,
                   float * __restrict nearZ, float * __restrict farZ,
                   float * __restrict top,   float * __restrict bottom,
                   float * __restrict left,  float * __restrict right)
{
    persp_decomp_rh_zo(proj.raw, nearZ, farZ, top, bottom, left, right);
}

/*!
 * @brief decomposes frustum values of perspective projection.
 *        this makes easy to get all values at once
 *        with a right-hand coordinate system and a
 *        clip-space of [0, 1].
 *
 * @param[in]  proj   perspective projection matrix
 * @param[out] dest   array
 */
PLAY_CGLM_INLINE
void
persp_decompv_rh_zo(mat4s proj, float dest[6])
{
    persp_decompv_rh_zo(proj.raw, dest);
}

/*!
 * @brief decomposes left and right values of perspective projection
 *        with a right-hand coordinate system and a
 *        clip-space of [0, 1].
 *        x stands for x axis (left / right axis)
 *
 * @param[in]  proj  perspective projection matrix
 * @param[out] left  left
 * @param[out] right right
 */
PLAY_CGLM_INLINE
void
persp_decomp_x_rh_zo(mat4s proj,
                     float * __restrict left,
                     float * __restrict right)
{
    persp_decomp_x_rh_zo(proj.raw, left, right);
}

/*!
 * @brief decomposes top and bottom values of perspective projection
 *        with a right-hand coordinate system and a
 *        clip-space of [0, 1].
 *        y stands for y axis (top / bottom axis)
 *
 * @param[in]  proj   perspective projection matrix
 * @param[out] top    top
 * @param[out] bottom bottom
 */
PLAY_CGLM_INLINE
void
persp_decomp_y_rh_zo(mat4s proj,
                     float * __restrict top,
                     float * __restrict bottom)
{
    persp_decomp_y_rh_zo(proj.raw, top, bottom);
}

/*!
 * @brief decomposes near and far values of perspective projection
 *        with a right-hand coordinate system and a
 *        clip-space of [0, 1].
 *        z stands for z axis (near / far axis)
 *
 * @param[in]  proj    perspective projection matrix
 * @param[out] nearZ   near
 * @param[out] farZ    far
 */
PLAY_CGLM_INLINE
void
persp_decomp_z_rh_zo(mat4s proj,
                     float * __restrict nearZ,
                     float * __restrict farZ)
{
    persp_decomp_z_rh_zo(proj.raw, nearZ, farZ);
}

/*!
 * @brief decomposes far value of perspective projection
 *        with a right-hand coordinate system and a
 *        clip-space of [0, 1].
 *
 * @param[in]  proj   perspective projection matrix
 * @param[out] farZ   far
 */
PLAY_CGLM_INLINE
void
persp_decomp_far_rh_zo(mat4s proj, float * __restrict farZ)
{
    persp_decomp_far_rh_zo(proj.raw, farZ);
}

/*!
 * @brief decomposes near value of perspective projection
 *        with a right-hand coordinate system and a
 *        clip-space of [0, 1].
 *
 * @param[in]  proj  perspective projection matrix
 * @param[out] nearZ near
 */
PLAY_CGLM_INLINE
void
persp_decomp_near_rh_zo(mat4s proj, float * __restrict nearZ)
{
    persp_decomp_near_rh_zo(proj.raw, nearZ);
}

/*!
 * @brief returns field of view angle along the Y-axis (in radians)
 *        with a right-hand coordinate system and a
 *        clip-space of [0, 1].
 *
 * if you need to degrees, use deg to convert it or use this:
 * fovy_deg = deg(persp_fovy(projMatrix))
 *
 * @param[in] proj perspective projection matrix
 */
PLAY_CGLM_INLINE
float
persp_fovy_rh_zo(mat4s proj)
{
    return persp_fovy_rh_zo(proj.raw);
}

/*!
 * @brief returns aspect ratio of perspective projection
 *        with a right-hand coordinate system and a
 *        clip-space of [0, 1].
 *
 * @param[in] proj perspective projection matrix
 */
PLAY_CGLM_INLINE
float
persp_aspect_rh_zo(mat4s proj)
{
    return persp_aspect_rh_zo(proj.raw);
}

/*!
 * @brief returns sizes of near and far planes of perspective projection
 *        with a right-hand coordinate system and a
 *        clip-space of [0, 1].
 *
 * @param[in]  proj perspective projection matrix
 * @param[in]  fovy fovy (see brief)
 * @returns    sizes as vector, sizes order: [Wnear, Hnear, Wfar, Hfar]
 */
PLAY_CGLM_INLINE
vec4s
persp_sizes_rh_zo(mat4s proj, float fovy)
{
    vec4s dest;
    persp_sizes_rh_zo(proj.raw, fovy, dest.raw);
    return dest;
}



/*** End of inlined file: persp_rh_zo.h ***/


/*** Start of inlined file: ortho_rh_no.h ***/
/*
 Functions:
   PLAY_CGLM_INLINE mat4s ortho_rh_no(float left,    float right,
                                      float bottom,  float top,
                                      float nearZ,   float farZ)
   PLAY_CGLM_INLINE mat4s ortho_aabb_rh_no(vec3s box[2]);
   PLAY_CGLM_INLINE mat4s ortho_aabb_p_rh_no(vec3s box[2],  float padding);
   PLAY_CGLM_INLINE mat4s ortho_aabb_pz_rh_no(vec3s box[2], float padding);
   PLAY_CGLM_INLINE mat4s ortho_default_rh_no(float aspect)
   PLAY_CGLM_INLINE mat4s ortho_default_s_rh_no(float aspect, float size)
 */




/*!
 * @brief set up orthographic projection matrix
 *        with a right-hand coordinate system and a
 *        clip-space of [-1, 1].
 *
 * @param[in]  left    viewport.left
 * @param[in]  right   viewport.right
 * @param[in]  bottom  viewport.bottom
 * @param[in]  top     viewport.top
 * @param[in]  nearZ   near clipping plane
 * @param[in]  farZ    far clipping plane
 * @returns    result matrix
 */
PLAY_CGLM_INLINE
mat4s
ortho_rh_no(float left,   float right,
            float bottom, float top,
            float nearZ,  float farZ)
{
    mat4s dest;
    ortho_rh_no(left, right, bottom, top, nearZ, farZ, dest.raw);
    return dest;
}

/*!
 * @brief set up orthographic projection matrix using bounding box
 *        with a right-hand coordinate system and a
 *        clip-space of [-1, 1].
 *
 * bounding box (AABB) must be in view space
 *
 * @param[in]  box   AABB
 * @returns    result matrix
 */
PLAY_CGLM_INLINE
mat4s
ortho_aabb_rh_no(vec3s box[2])
{
    mat4s dest;
    vec3  rawBox[2];

    vec3s_(unpack)(rawBox, box, 2);
    ortho_aabb_rh_no(rawBox, dest.raw);

    return dest;
}

/*!
 * @brief set up orthographic projection matrix using bounding box
 *        with a right-hand coordinate system and a
 *        clip-space of [-1, 1].
 *
 * bounding box (AABB) must be in view space
 *
 * @param[in]  box     AABB
 * @param[in]  padding padding
 * @returns    result matrix
 */
PLAY_CGLM_INLINE
mat4s
ortho_aabb_p_rh_no(vec3s box[2], float padding)
{
    mat4s dest;
    vec3  rawBox[2];

    vec3s_(unpack)(rawBox, box, 2);
    ortho_aabb_p_rh_no(rawBox, padding, dest.raw);

    return dest;
}

/*!
 * @brief set up orthographic projection matrix using bounding box
 *        with a right-hand coordinate system and a
 *        clip-space of [-1, 1].
 *
 * bounding box (AABB) must be in view space
 *
 * @param[in]  box     AABB
 * @param[in]  padding padding for near and far
 * @returns    result matrix
 */
PLAY_CGLM_INLINE
mat4s
ortho_aabb_pz_rh_no(vec3s box[2], float padding)
{
    mat4s dest;
    vec3  rawBox[2];

    vec3s_(unpack)(rawBox, box, 2);
    ortho_aabb_pz_rh_no(rawBox, padding, dest.raw);

    return dest;
}

/*!
 * @brief set up unit orthographic projection matrix
 *        with a right-hand coordinate system and a
 *        clip-space of [-1, 1].
 *
 * @param[in]  aspect aspect ration ( width / height )
 * @returns    result matrix
 */
PLAY_CGLM_INLINE
mat4s
ortho_default_rh_no(float aspect)
{
    mat4s dest;
    ortho_default_rh_no(aspect, dest.raw);
    return dest;
}

/*!
 * @brief set up orthographic projection matrix with given CUBE size
 *        with a right-hand coordinate system and a
 *        clip-space of [-1, 1].
 *
 * @param[in]  aspect aspect ratio ( width / height )
 * @param[in]  size   cube size
 * @returns    result matrix
 */
PLAY_CGLM_INLINE
mat4s
ortho_default_s_rh_no(float aspect, float size)
{
    mat4s dest;
    ortho_default_s_rh_no(aspect, size, dest.raw);
    return dest;
}



/*** End of inlined file: ortho_rh_no.h ***/


/*** Start of inlined file: persp_rh_no.h ***/
/*
 Functions:
   PLAY_CGLM_INLINE mat4s frustum_rh_no(float left,    float right,
                                        float bottom,  float top,
                                        float nearZ,   float farZ)
   PLAY_CGLM_INLINE mat4s perspective_rh_no(float fovy,
                                            float aspect,
                                            float nearZ,
                                            float farZ)
   PLAY_CGLM_INLINE void  persp_move_far_rh_no(mat4s proj, float deltaFar)
   PLAY_CGLM_INLINE mat4s perspective_default_rh_no(float aspect)
   PLAY_CGLM_INLINE void  perspective_resize_rh_no(mat4s proj, float aspect)
   PLAY_CGLM_INLINE void  persp_decomp_rh_no(mat4s  proj,
                                             float *nearv, float *farv,
                                             float *top,   float *bottom,
                                             float *left,  float *right)
   PLAY_CGLM_INLINE void  persp_decompv_rh_no(mat4s proj, float dest[6])
   PLAY_CGLM_INLINE void  persp_decomp_x_rh_no(mat4s proj, float *left, float *right)
   PLAY_CGLM_INLINE void  persp_decomp_y_rh_no(mat4s proj, float *top, float *bottom)
   PLAY_CGLM_INLINE void  persp_decomp_z_rh_no(mat4s proj, float *nearv, float *farv)
   PLAY_CGLM_INLINE void  persp_decomp_far_rh_no(mat4s proj, float *farZ)
   PLAY_CGLM_INLINE void  persp_decomp_near_rh_no(mat4s proj, float *nearZ)
   PLAY_CGLM_INLINE float persp_fovy_rh_no(mat4s proj)
   PLAY_CGLM_INLINE float persp_aspect_rh_no(mat4s proj)
   PLAY_CGLM_INLINE vec4s persp_sizes_rh_no(mat4s proj, float fovy)
 */




/*!
 * @brief set up perspective peprojection matrix
 *        with a right-hand coordinate system and a
 *        clip-space of [-1, 1].
 *
 * @param[in]  left    viewport.left
 * @param[in]  right   viewport.right
 * @param[in]  bottom  viewport.bottom
 * @param[in]  top     viewport.top
 * @param[in]  nearZ   near clipping plane
 * @param[in]  farZ    far clipping plane
 * @returns    result matrix
 */
PLAY_CGLM_INLINE
mat4s
frustum_rh_no(float left,   float right,
              float bottom, float top,
              float nearZ,  float farZ)
{
    mat4s dest;
    frustum_rh_no(left, right, bottom, top, nearZ, farZ, dest.raw);
    return dest;
}

/*!
 * @brief set up perspective projection matrix
 *        with a right-hand coordinate system and a
 *        clip-space of [-1, 1].
 *
 * @param[in]  fovy    field of view angle
 * @param[in]  aspect  aspect ratio ( width / height )
 * @param[in]  nearZ   near clipping plane
 * @param[in]  farZ    far clipping planes
 * @returns    result matrix
 */
PLAY_CGLM_INLINE
mat4s
perspective_rh_no(float fovy, float aspect, float nearZ, float farZ)
{
    mat4s dest;
    perspective_rh_no(fovy, aspect, nearZ, farZ, dest.raw);
    return dest;
}

/*!
 * @brief extend perspective projection matrix's far distance
 *        with a right-hand coordinate system and a
 *        clip-space of [-1, 1].
 *
 * NOTE: if you dodn't want to create new matrix then use array api on struct.raw
 *       like persp_move_far_rh_no(prooj.raw, deltaFar) to avoid create new mat4
 *       each time
 *       s
 * this function does not guarantee far >= near, be aware of that!
 *
 * @param[in, out] proj      projection matrix to extend
 * @param[in]      deltaFar  distance from existing far (negative to shink)
 */
PLAY_CGLM_INLINE
mat4s
persp_move_far_rh_no(mat4s proj, float deltaFar)
{
    mat4s dest;
    dest = proj;
    persp_move_far_rh_no(dest.raw, deltaFar);
    return dest;
}

/*!
 * @brief set up perspective projection matrix with default near/far
 *        and angle values with a right-hand coordinate system and a
 *        clip-space of [-1, 1].
 *
 * @param[in]  aspect aspect ratio ( width / height )
 * @returns    result matrix
 */
PLAY_CGLM_INLINE
mat4s
perspective_default_rh_no(float aspect)
{
    mat4s dest;
    perspective_default_rh_no(aspect, dest.raw);
    return dest;
}

/*!
 * @brief resize perspective matrix by aspect ratio ( width / height )
 *        this makes very easy to resize proj matrix when window /viewport
 *        reized with a right-hand coordinate system and a
 *        clip-space of [-1, 1].
 *
 * NOTE: if you dodn't want to create new matrix then use array api on struct.raw
 *       like perspective_resize_rh_no(proj.raw, aspect) to avoid create new mat4
 *       each time
 *
 * @param[in, out] proj   perspective projection matrix
 * @param[in]      aspect aspect ratio ( width / height )
 */
PLAY_CGLM_INLINE
mat4s
perspective_resize_rh_no(mat4s proj, float aspect)
{
    mat4s dest;
    dest = proj;
    perspective_resize_rh_no(aspect, dest.raw);
    return dest;
}

/*!
 * @brief decomposes frustum values of perspective projection.
 *        with a right-hand coordinate system and a
 *        clip-space of [-1, 1].
 *
 * @param[in]  proj    perspective projection matrix
 * @param[out] nearZ   near
 * @param[out] farZ    far
 * @param[out] top     top
 * @param[out] bottom  bottom
 * @param[out] left    left
 * @param[out] right   right
 */
PLAY_CGLM_INLINE
void
persp_decomp_rh_no(mat4s proj,
                   float * __restrict nearZ, float * __restrict farZ,
                   float * __restrict top,   float * __restrict bottom,
                   float * __restrict left,  float * __restrict right)
{
    persp_decomp_rh_no(proj.raw, nearZ, farZ, top, bottom, left, right);
}

/*!
 * @brief decomposes frustum values of perspective projection.
 *        this makes easy to get all values at once
 *        with a right-hand coordinate system and a
 *        clip-space of [-1, 1].
 *
 * @param[in]  proj   perspective projection matrix
 * @param[out] dest   array
 */
PLAY_CGLM_INLINE
void
persp_decompv_rh_no(mat4s proj, float dest[6])
{
    persp_decompv_rh_no(proj.raw, dest);
}

/*!
 * @brief decomposes left and right values of perspective projection
 *        with a right-hand coordinate system and a
 *        clip-space of [-1, 1].
 *        x stands for x axis (left / right axis)
 *
 * @param[in]  proj  perspective projection matrix
 * @param[out] left  left
 * @param[out] right right
 */
PLAY_CGLM_INLINE
void
persp_decomp_x_rh_no(mat4s proj,
                     float * __restrict left,
                     float * __restrict right)
{
    persp_decomp_x_rh_no(proj.raw, left, right);
}

/*!
 * @brief decomposes top and bottom values of perspective projection
 *        with a right-hand coordinate system and a
 *        clip-space of [-1, 1].
 *        y stands for y axis (top / bottom axis)
 *
 * @param[in]  proj   perspective projection matrix
 * @param[out] top    top
 * @param[out] bottom bottom
 */
PLAY_CGLM_INLINE
void
persp_decomp_y_rh_no(mat4s proj,
                     float * __restrict top,
                     float * __restrict bottom)
{
    persp_decomp_y_rh_no(proj.raw, top, bottom);
}

/*!
 * @brief decomposes near and far values of perspective projection
 *        with a right-hand coordinate system and a
 *        clip-space of [-1, 1].
 *        z stands for z axis (near / far axis)
 *
 * @param[in]  proj    perspective projection matrix
 * @param[out] nearZ   near
 * @param[out] farZ    far
 */
PLAY_CGLM_INLINE
void
persp_decomp_z_rh_no(mat4s proj,
                     float * __restrict nearZ,
                     float * __restrict farZ)
{
    persp_decomp_z_rh_no(proj.raw, nearZ, farZ);
}

/*!
 * @brief decomposes far value of perspective projection
 *        with a right-hand coordinate system and a
 *        clip-space of [-1, 1].
 *
 * @param[in]  proj   perspective projection matrix
 * @param[out] farZ   far
 */
PLAY_CGLM_INLINE
void
persp_decomp_far_rh_no(mat4s proj, float * __restrict farZ)
{
    persp_decomp_far_rh_no(proj.raw, farZ);
}

/*!
 * @brief decomposes near value of perspective projection
 *        with a right-hand coordinate system and a
 *        clip-space of [-1, 1].
 *
 * @param[in]  proj  perspective projection matrix
 * @param[out] nearZ near
 */
PLAY_CGLM_INLINE
void
persp_decomp_near_rh_no(mat4s proj, float * __restrict nearZ)
{
    persp_decomp_near_rh_no(proj.raw, nearZ);
}

/*!
 * @brief returns field of view angle along the Y-axis (in radians)
 *        with a right-hand coordinate system and a
 *        clip-space of [-1, 1].
 *
 * if you need to degrees, use deg to convert it or use this:
 * fovy_deg = deg(persp_fovy(projMatrix))
 *
 * @param[in] proj perspective projection matrix
 */
PLAY_CGLM_INLINE
float
persp_fovy_rh_no(mat4s proj)
{
    return persp_fovy_rh_no(proj.raw);
}

/*!
 * @brief returns aspect ratio of perspective projection
 *        with a right-hand coordinate system and a
 *        clip-space of [-1, 1].
 *
 * @param[in] proj perspective projection matrix
 */
PLAY_CGLM_INLINE
float
persp_aspect_rh_no(mat4s proj)
{
    return persp_aspect_rh_no(proj.raw);
}

/*!
 * @brief returns sizes of near and far planes of perspective projection
 *        with a right-hand coordinate system and a
 *        clip-space of [-1, 1].
 *
 * @param[in]  proj perspective projection matrix
 * @param[in]  fovy fovy (see brief)
 * @returns    sizes as vector, sizes order: [Wnear, Hnear, Wfar, Hfar]
 */
PLAY_CGLM_INLINE
vec4s
persp_sizes_rh_no(mat4s proj, float fovy)
{
    vec4s dest;
    persp_sizes_rh_no(proj.raw, fovy, dest.raw);
    return dest;
}



/*** End of inlined file: persp_rh_no.h ***/


/*** Start of inlined file: view_lh_zo.h ***/
/*
 Functions:
   PLAY_CGLM_INLINE mat4s lookat_lh_zo(vec3s eye, vec3s center, vec3s up)
   PLAY_CGLM_INLINE mat4s look_lh_zo(vec3s eye, vec3s dir, vec3s up)
   PLAY_CGLM_INLINE mat4s look_anyup_lh_zo(vec3s eye, vec3s dir)
 */




/*!
 * @brief set up view matrix
 *        with a left-hand coordinate system and a
 *        clip-space of [0, 1].
 *
 * NOTE: The UP vector must not be parallel to the line of sight from
 *       the eye point to the reference point
 *
 * @param[in]  eye    eye vector
 * @param[in]  center center vector
 * @param[in]  up     up vector
 * @returns    result matrix
 */
PLAY_CGLM_INLINE
mat4s
lookat_lh_zo(vec3s eye, vec3s center, vec3s up)
{
    mat4s dest;
    lookat_lh_zo(eye.raw, center.raw, up.raw, dest.raw);
    return dest;
}

/*!
 * @brief set up view matrix
 *        with a left-hand coordinate system and a
 *        clip-space of [0, 1].
 *
 * convenient wrapper for lookat: if you only have direction not target self
 * then this might be useful. Because you need to get target from direction.
 *
 * NOTE: The UP vector must not be parallel to the line of sight from
 *       the eye point to the reference point
 *
 * @param[in]  eye    eye vector
 * @param[in]  dir    direction vector
 * @param[in]  up     up vector
 * @returns    result matrix
 */
PLAY_CGLM_INLINE
mat4s
look_lh_zo(vec3s eye, vec3s dir, vec3s up)
{
    mat4s dest;
    look_lh_zo(eye.raw, dir.raw, up.raw, dest.raw);
    return dest;
}

/*!
 * @brief set up view matrix
 *        with a left-hand coordinate system and a
 *        clip-space of [0, 1].
 *
 * convenient wrapper for look: if you only have direction and if you don't
 * care what UP vector is then this might be useful to create view matrix
 *
 * @param[in]  eye    eye vector
 * @param[in]  dir    direction vector
 * @returns    result matrix
 */
PLAY_CGLM_INLINE
mat4s
look_anyup_lh_zo(vec3s eye, vec3s dir)
{
    mat4s dest;
    look_anyup_lh_zo(eye.raw, dir.raw, dest.raw);
    return dest;
}



/*** End of inlined file: view_lh_zo.h ***/


/*** Start of inlined file: view_lh_no.h ***/
/*
 Functions:
   PLAY_CGLM_INLINE mat4s lookat_lh_no(vec3s eye, vec3s center, vec3s up)
   PLAY_CGLM_INLINE mat4s look_lh_no(vec3s eye, vec3s dir, vec3s up)
   PLAY_CGLM_INLINE mat4s look_anyup_lh_no(vec3s eye, vec3s dir)
 */




/*!
 * @brief set up view matrix
 *        with a left-hand coordinate system and a
 *        clip-space of [-1, 1].
 *
 * NOTE: The UP vector must not be parallel to the line of sight from
 *       the eye point to the reference point
 *
 * @param[in]  eye    eye vector
 * @param[in]  center center vector
 * @param[in]  up     up vector
 * @returns    result matrix
 */
PLAY_CGLM_INLINE
mat4s
lookat_lh_no(vec3s eye, vec3s center, vec3s up)
{
    mat4s dest;
    lookat_lh_no(eye.raw, center.raw, up.raw, dest.raw);
    return dest;
}

/*!
 * @brief set up view matrix
 *        with a left-hand coordinate system and a
 *        clip-space of [-1, 1].
 *
 * convenient wrapper for lookat: if you only have direction not target self
 * then this might be useful. Because you need to get target from direction.
 *
 * NOTE: The UP vector must not be parallel to the line of sight from
 *       the eye point to the reference point
 *
 * @param[in]  eye    eye vector
 * @param[in]  dir    direction vector
 * @param[in]  up     up vector
 * @returns    result matrix
 */
PLAY_CGLM_INLINE
mat4s
look_lh_no(vec3s eye, vec3s dir, vec3s up)
{
    mat4s dest;
    look_lh_no(eye.raw, dir.raw, up.raw, dest.raw);
    return dest;
}

/*!
 * @brief set up view matrix
 *        with a left-hand coordinate system and a
 *        clip-space of [-1, 1].
 *
 * convenient wrapper for look: if you only have direction and if you don't
 * care what UP vector is then this might be useful to create view matrix
 *
 * @param[in]  eye    eye vector
 * @param[in]  dir    direction vector
 * @returns    result matrix
 */
PLAY_CGLM_INLINE
mat4s
look_anyup_lh_no(vec3s eye, vec3s dir)
{
    mat4s dest;
    look_anyup_lh_no(eye.raw, dir.raw, dest.raw);
    return dest;
}



/*** End of inlined file: view_lh_no.h ***/


/*** Start of inlined file: view_rh_zo.h ***/
/*
 Functions:
   PLAY_CGLM_INLINE mat4s lookat_rh_zo(vec3s eye, vec3s center, vec3s up)
   PLAY_CGLM_INLINE mat4s look_rh_zo(vec3s eye, vec3s dir, vec3s up)
   PLAY_CGLM_INLINE mat4s look_anyup_rh_zo(vec3s eye, vec3s dir)
 */




/*!
 * @brief set up view matrix
 *        with a right-hand coordinate system and a
 *        clip-space of [0, 1].
 *
 * NOTE: The UP vector must not be parallel to the line of sight from
 *       the eye point to the reference point
 *
 * @param[in]  eye    eye vector
 * @param[in]  center center vector
 * @param[in]  up     up vector
 * @returns    result matrix
 */
PLAY_CGLM_INLINE
mat4s
lookat_rh_zo(vec3s eye, vec3s center, vec3s up)
{
    mat4s dest;
    lookat_rh_zo(eye.raw, center.raw, up.raw, dest.raw);
    return dest;
}

/*!
 * @brief set up view matrix
 *        with a right-hand coordinate system and a
 *        clip-space of [0, 1].
 *
 * convenient wrapper for lookat: if you only have direction not target self
 * then this might be useful. Because you need to get target from direction.
 *
 * NOTE: The UP vector must not be parallel to the line of sight from
 *       the eye point to the reference point
 *
 * @param[in]  eye    eye vector
 * @param[in]  dir    direction vector
 * @param[in]  up     up vector
 * @returns    result matrix
 */
PLAY_CGLM_INLINE
mat4s
look_rh_zo(vec3s eye, vec3s dir, vec3s up)
{
    mat4s dest;
    look_rh_zo(eye.raw, dir.raw, up.raw, dest.raw);
    return dest;
}

/*!
 * @brief set up view matrix
 *        with a right-hand coordinate system and a
 *        clip-space of [0, 1].
 *
 * convenient wrapper for look: if you only have direction and if you don't
 * care what UP vector is then this might be useful to create view matrix
 *
 * @param[in]  eye    eye vector
 * @param[in]  dir    direction vector
 * @returns    result matrix
 */
PLAY_CGLM_INLINE
mat4s
look_anyup_rh_zo(vec3s eye, vec3s dir)
{
    mat4s dest;
    look_anyup_rh_zo(eye.raw, dir.raw, dest.raw);
    return dest;
}



/*** End of inlined file: view_rh_zo.h ***/


/*** Start of inlined file: view_rh_no.h ***/
/*
 Functions:
   PLAY_CGLM_INLINE mat4s lookat_rh_no(vec3s eye, vec3s center, vec3s up)
   PLAY_CGLM_INLINE mat4s look_rh_no(vec3s eye, vec3s dir, vec3s up)
   PLAY_CGLM_INLINE mat4s look_anyup_rh_no(vec3s eye, vec3s dir)
 */




/*!
 * @brief set up view matrix
 *        with a right-hand coordinate system and a
 *        clip-space of [-1, 1].
 *
 * NOTE: The UP vector must not be parallel to the line of sight from
 *       the eye point to the reference point
 *
 * @param[in]  eye    eye vector
 * @param[in]  center center vector
 * @param[in]  up     up vector
 * @returns    result matrix
 */
PLAY_CGLM_INLINE
mat4s
lookat_rh_no(vec3s eye, vec3s center, vec3s up)
{
    mat4s dest;
    lookat_rh_no(eye.raw, center.raw, up.raw, dest.raw);
    return dest;
}

/*!
 * @brief set up view matrix
 *        with a right-hand coordinate system and a
 *        clip-space of [-1, 1].
 *
 * convenient wrapper for lookat: if you only have direction not target self
 * then this might be useful. Because you need to get target from direction.
 *
 * NOTE: The UP vector must not be parallel to the line of sight from
 *       the eye point to the reference point
 *
 * @param[in]  eye    eye vector
 * @param[in]  dir    direction vector
 * @param[in]  up     up vector
 * @returns    result matrix
 */
PLAY_CGLM_INLINE
mat4s
look_rh_no(vec3s eye, vec3s dir, vec3s up)
{
    mat4s dest;
    look_rh_no(eye.raw, dir.raw, up.raw, dest.raw);
    return dest;
}

/*!
 * @brief set up view matrix
 *        with a right-hand coordinate system and a
 *        clip-space of [-1, 1].
 *
 * convenient wrapper for look: if you only have direction and if you don't
 * care what UP vector is then this might be useful to create view matrix
 *
 * @param[in]  eye    eye vector
 * @param[in]  dir    direction vector
 * @returns    result matrix
 */
PLAY_CGLM_INLINE
mat4s
look_anyup_rh_no(vec3s eye, vec3s dir)
{
    mat4s dest;
    look_anyup_rh_no(eye.raw, dir.raw, dest.raw);
    return dest;
}



/*** End of inlined file: view_rh_no.h ***/

/*!
 * @brief set up perspective peprojection matrix
 *
 * @param[in]  left    viewport.left
 * @param[in]  right   viewport.right
 * @param[in]  bottom  viewport.bottom
 * @param[in]  top     viewport.top
 * @param[in]  nearZ   near clipping plane
 * @param[in]  farZ    far clipping plane
 * @returns    result matrix
 */
PLAY_CGLM_INLINE
mat4s
frustum(float left,   float right,
        float bottom, float top,
        float nearZ,  float farZ)
{
#if PLAY_CGLM_CONFIG_CLIP_CONTROL == PLAY_CGLM_CLIP_CONTROL_LH_ZO
    return frustum_lh_zo(left, right, bottom, top, nearZ, farZ);
#elif PLAY_CGLM_CONFIG_CLIP_CONTROL == PLAY_CGLM_CLIP_CONTROL_LH_NO
    return frustum_lh_no(left, right, bottom, top, nearZ, farZ);
#elif PLAY_CGLM_CONFIG_CLIP_CONTROL == PLAY_CGLM_CLIP_CONTROL_RH_ZO
    return frustum_rh_zo(left, right, bottom, top, nearZ, farZ);
#elif PLAY_CGLM_CONFIG_CLIP_CONTROL == PLAY_CGLM_CLIP_CONTROL_RH_NO
    return frustum_rh_no(left, right, bottom, top, nearZ, farZ);
#endif
}

/*!
 * @brief set up orthographic projection matrix
 *
 * @param[in]  left    viewport.left
 * @param[in]  right   viewport.right
 * @param[in]  bottom  viewport.bottom
 * @param[in]  top     viewport.top
 * @param[in]  nearZ   near clipping plane
 * @param[in]  farZ    far clipping plane
 * @returns    result matrix
 */
PLAY_CGLM_INLINE
mat4s
ortho(float left,   float right,
      float bottom, float top,
      float nearZ,  float farZ)
{
#if PLAY_CGLM_CONFIG_CLIP_CONTROL == PLAY_CGLM_CLIP_CONTROL_LH_ZO
    return ortho_lh_zo(left, right, bottom, top, nearZ, farZ);
#elif PLAY_CGLM_CONFIG_CLIP_CONTROL == PLAY_CGLM_CLIP_CONTROL_LH_NO
    return ortho_lh_no(left, right, bottom, top, nearZ, farZ);
#elif PLAY_CGLM_CONFIG_CLIP_CONTROL == PLAY_CGLM_CLIP_CONTROL_RH_ZO
    return ortho_rh_zo(left, right, bottom, top, nearZ, farZ);
#elif PLAY_CGLM_CONFIG_CLIP_CONTROL == PLAY_CGLM_CLIP_CONTROL_RH_NO
    return ortho_rh_no(left, right, bottom, top, nearZ, farZ);
#endif
}

/*!
 * @brief set up orthographic projection matrix using bounding box
 *
 * bounding box (AABB) must be in view space
 *
 * @param[in]  box   AABB
 * @returns    result matrix
 */
PLAY_CGLM_INLINE
mat4s
ortho_aabb(vec3s box[2])
{
#if PLAY_CGLM_CONFIG_CLIP_CONTROL == PLAY_CGLM_CLIP_CONTROL_LH_ZO
    return ortho_aabb_lh_zo(box);
#elif PLAY_CGLM_CONFIG_CLIP_CONTROL == PLAY_CGLM_CLIP_CONTROL_LH_NO
    return ortho_aabb_lh_no(box);
#elif PLAY_CGLM_CONFIG_CLIP_CONTROL == PLAY_CGLM_CLIP_CONTROL_RH_ZO
    return ortho_aabb_rh_zo(box);
#elif PLAY_CGLM_CONFIG_CLIP_CONTROL == PLAY_CGLM_CLIP_CONTROL_RH_NO
    return ortho_aabb_rh_no(box);
#endif
}

/*!
 * @brief set up orthographic projection matrix using bounding box
 *
 * bounding box (AABB) must be in view space
 *
 * @param[in]  box     AABB
 * @param[in]  padding padding
 * @returns    result matrix
 */
PLAY_CGLM_INLINE
mat4s
ortho_aabb_p(vec3s box[2], float padding)
{
#if PLAY_CGLM_CONFIG_CLIP_CONTROL == PLAY_CGLM_CLIP_CONTROL_LH_ZO
    return ortho_aabb_p_lh_zo(box, padding);
#elif PLAY_CGLM_CONFIG_CLIP_CONTROL == PLAY_CGLM_CLIP_CONTROL_LH_NO
    return ortho_aabb_p_lh_no(box, padding);
#elif PLAY_CGLM_CONFIG_CLIP_CONTROL == PLAY_CGLM_CLIP_CONTROL_RH_ZO
    return ortho_aabb_p_rh_zo(box, padding);
#elif PLAY_CGLM_CONFIG_CLIP_CONTROL == PLAY_CGLM_CLIP_CONTROL_RH_NO
    return ortho_aabb_p_rh_no(box, padding);
#endif
}

/*!
 * @brief set up orthographic projection matrix using bounding box
 *
 * bounding box (AABB) must be in view space
 *
 * @param[in]  box     AABB
 * @param[in]  padding padding for near and far
 * @returns    result matrix
 */
PLAY_CGLM_INLINE
mat4s
ortho_aabb_pz(vec3s box[2], float padding)
{
#if PLAY_CGLM_CONFIG_CLIP_CONTROL == PLAY_CGLM_CLIP_CONTROL_LH_ZO
    return ortho_aabb_pz_lh_zo(box, padding);
#elif PLAY_CGLM_CONFIG_CLIP_CONTROL == PLAY_CGLM_CLIP_CONTROL_LH_NO
    return ortho_aabb_pz_lh_no(box, padding);
#elif PLAY_CGLM_CONFIG_CLIP_CONTROL == PLAY_CGLM_CLIP_CONTROL_RH_ZO
    return ortho_aabb_pz_rh_zo(box, padding);
#elif PLAY_CGLM_CONFIG_CLIP_CONTROL == PLAY_CGLM_CLIP_CONTROL_RH_NO
    return ortho_aabb_pz_rh_no(box, padding);
#endif
}

/*!
 * @brief set up unit orthographic projection matrix
 *
 * @param[in]  aspect aspect ration ( width / height )
 * @returns    result matrix
 */
PLAY_CGLM_INLINE
mat4s
ortho_default(float aspect)
{
#if PLAY_CGLM_CONFIG_CLIP_CONTROL == PLAY_CGLM_CLIP_CONTROL_LH_ZO
    return ortho_default_lh_zo(aspect);
#elif PLAY_CGLM_CONFIG_CLIP_CONTROL == PLAY_CGLM_CLIP_CONTROL_LH_NO
    return ortho_default_lh_no(aspect);
#elif PLAY_CGLM_CONFIG_CLIP_CONTROL == PLAY_CGLM_CLIP_CONTROL_RH_ZO
    return ortho_default_rh_zo(aspect);
#elif PLAY_CGLM_CONFIG_CLIP_CONTROL == PLAY_CGLM_CLIP_CONTROL_RH_NO
    return ortho_default_rh_no(aspect);
#endif
}

/*!
 * @brief set up orthographic projection matrix with given CUBE size
 *
 * @param[in]  aspect aspect ratio ( width / height )
 * @param[in]  size   cube size
 * @returns    result matrix
 */
PLAY_CGLM_INLINE
mat4s
ortho_default_s(float aspect, float size)
{
#if PLAY_CGLM_CONFIG_CLIP_CONTROL == PLAY_CGLM_CLIP_CONTROL_LH_ZO
    return ortho_default_s_lh_zo(aspect, size);
#elif PLAY_CGLM_CONFIG_CLIP_CONTROL == PLAY_CGLM_CLIP_CONTROL_LH_NO
    return ortho_default_s_lh_no(aspect, size);
#elif PLAY_CGLM_CONFIG_CLIP_CONTROL == PLAY_CGLM_CLIP_CONTROL_RH_ZO
    return ortho_default_s_rh_zo(aspect, size);
#elif PLAY_CGLM_CONFIG_CLIP_CONTROL == PLAY_CGLM_CLIP_CONTROL_RH_NO
    return ortho_default_s_rh_no(aspect, size);
#endif
}

/*!
 * @brief set up perspective projection matrix
 *
 * @param[in]  fovy    field of view angle
 * @param[in]  aspect  aspect ratio ( width / height )
 * @param[in]  nearZ   near clipping plane
 * @param[in]  farZ    far clipping planes
 * @returns    result matrix
 */
PLAY_CGLM_INLINE
mat4s
perspective(float fovy, float aspect, float nearZ, float farZ)
{
#if PLAY_CGLM_CONFIG_CLIP_CONTROL == PLAY_CGLM_CLIP_CONTROL_LH_ZO
    return perspective_lh_zo(fovy, aspect, nearZ, farZ);
#elif PLAY_CGLM_CONFIG_CLIP_CONTROL == PLAY_CGLM_CLIP_CONTROL_LH_NO
    return perspective_lh_no(fovy, aspect, nearZ, farZ);
#elif PLAY_CGLM_CONFIG_CLIP_CONTROL == PLAY_CGLM_CLIP_CONTROL_RH_ZO
    return perspective_rh_zo(fovy, aspect, nearZ, farZ);
#elif PLAY_CGLM_CONFIG_CLIP_CONTROL == PLAY_CGLM_CLIP_CONTROL_RH_NO
    return perspective_rh_no(fovy, aspect, nearZ, farZ);
#endif
}

/*!
 * @brief extend perspective projection matrix's far distance
 *
 * NOTE: if you dodn't want to create new matrix then use array api on struct.raw
 *       like persp_move_far(prooj.raw, deltaFar) to avoid create new mat4
 *       each time
 *
 * this function does not guarantee far >= near, be aware of that!
 *
 * @param[in, out] proj      projection matrix to extend
 * @param[in]      deltaFar  distance from existing far (negative to shink)
 */
PLAY_CGLM_INLINE
mat4s
persp_move_far(mat4s proj, float deltaFar)
{
#if PLAY_CGLM_CONFIG_CLIP_CONTROL == PLAY_CGLM_CLIP_CONTROL_LH_ZO
    return persp_move_far_lh_zo(proj, deltaFar);
#elif PLAY_CGLM_CONFIG_CLIP_CONTROL == PLAY_CGLM_CLIP_CONTROL_LH_NO
    return persp_move_far_lh_no(proj, deltaFar);
#elif PLAY_CGLM_CONFIG_CLIP_CONTROL == PLAY_CGLM_CLIP_CONTROL_RH_ZO
    return persp_move_far_rh_zo(proj, deltaFar);
#elif PLAY_CGLM_CONFIG_CLIP_CONTROL == PLAY_CGLM_CLIP_CONTROL_RH_NO
    return persp_move_far_rh_no(proj, deltaFar);
#endif
}

/*!
 * @brief set up perspective projection matrix with default near/far
 *        and angle values
 *
 * @param[in]  aspect aspect ratio ( width / height )
 * @returns    result matrix
 */
PLAY_CGLM_INLINE
mat4s
perspective_default(float aspect)
{
#if PLAY_CGLM_CONFIG_CLIP_CONTROL == PLAY_CGLM_CLIP_CONTROL_LH_ZO
    return perspective_default_lh_zo(aspect);
#elif PLAY_CGLM_CONFIG_CLIP_CONTROL == PLAY_CGLM_CLIP_CONTROL_LH_NO
    return perspective_default_lh_no(aspect);
#elif PLAY_CGLM_CONFIG_CLIP_CONTROL == PLAY_CGLM_CLIP_CONTROL_RH_ZO
    return perspective_default_rh_zo(aspect);
#elif PLAY_CGLM_CONFIG_CLIP_CONTROL == PLAY_CGLM_CLIP_CONTROL_RH_NO
    return perspective_default_rh_no(aspect);
#endif
}

/*!
 * @brief resize perspective matrix by aspect ratio ( width / height )
 *        this makes very easy to resize proj matrix when window /viewport
 *        reized
 *
 * NOTE: if you dodn't want to create new matrix then use array api on struct.raw
 *       like perspective_resize(proj.raw, aspect) to avoid create new mat4
 *       each time
 *
 * @param[in, out] proj   perspective projection matrix
 * @param[in]      aspect aspect ratio ( width / height )
 */
PLAY_CGLM_INLINE
mat4s
perspective_resize(mat4s proj, float aspect)
{
#if PLAY_CGLM_CONFIG_CLIP_CONTROL == PLAY_CGLM_CLIP_CONTROL_LH_ZO
    return perspective_resize_lh_zo(proj, aspect);
#elif PLAY_CGLM_CONFIG_CLIP_CONTROL == PLAY_CGLM_CLIP_CONTROL_LH_NO
    return perspective_resize_lh_no(proj, aspect);
#elif PLAY_CGLM_CONFIG_CLIP_CONTROL == PLAY_CGLM_CLIP_CONTROL_RH_ZO
    return perspective_resize_rh_zo(proj, aspect);
#elif PLAY_CGLM_CONFIG_CLIP_CONTROL == PLAY_CGLM_CLIP_CONTROL_RH_NO
    return perspective_resize_rh_no(proj, aspect);
#endif
}

/*!
 * @brief set up view matrix
 *
 * NOTE: The UP vector must not be parallel to the line of sight from
 *       the eye point to the reference point
 *
 * @param[in]  eye    eye vector
 * @param[in]  center center vector
 * @param[in]  up     up vector
 * @returns    result matrix
 */
PLAY_CGLM_INLINE
mat4s
lookat(vec3s eye, vec3s center, vec3s up)
{
#if PLAY_CGLM_CONFIG_CLIP_CONTROL == PLAY_CGLM_CLIP_CONTROL_LH_ZO
    return lookat_lh_zo(eye, center, up);
#elif PLAY_CGLM_CONFIG_CLIP_CONTROL == PLAY_CGLM_CLIP_CONTROL_LH_NO
    return lookat_lh_no(eye, center, up);
#elif PLAY_CGLM_CONFIG_CLIP_CONTROL == PLAY_CGLM_CLIP_CONTROL_RH_ZO
    return lookat_rh_zo(eye, center, up);
#elif PLAY_CGLM_CONFIG_CLIP_CONTROL == PLAY_CGLM_CLIP_CONTROL_RH_NO
    return lookat_rh_no(eye, center, up);
#endif
}

/*!
 * @brief set up view matrix
 *
 * convenient wrapper for lookat: if you only have direction not target self
 * then this might be useful. Because you need to get target from direction.
 *
 * NOTE: The UP vector must not be parallel to the line of sight from
 *       the eye point to the reference point
 *
 * @param[in]  eye    eye vector
 * @param[in]  dir    direction vector
 * @param[in]  up     up vector
 * @returns    result matrix
 */
PLAY_CGLM_INLINE
mat4s
look(vec3s eye, vec3s dir, vec3s up)
{
#if PLAY_CGLM_CONFIG_CLIP_CONTROL == PLAY_CGLM_CLIP_CONTROL_LH_ZO
    return look_lh_zo(eye, dir, up);
#elif PLAY_CGLM_CONFIG_CLIP_CONTROL == PLAY_CGLM_CLIP_CONTROL_LH_NO
    return look_lh_no(eye, dir, up);
#elif PLAY_CGLM_CONFIG_CLIP_CONTROL == PLAY_CGLM_CLIP_CONTROL_RH_ZO
    return look_rh_zo(eye, dir, up);
#elif PLAY_CGLM_CONFIG_CLIP_CONTROL == PLAY_CGLM_CLIP_CONTROL_RH_NO
    return look_rh_no(eye, dir, up);
#endif
}

/*!
 * @brief set up view matrix
 *
 * convenient wrapper for look: if you only have direction and if you don't
 * care what UP vector is then this might be useful to create view matrix
 *
 * @param[in]  eye    eye vector
 * @param[in]  dir    direction vector
 * @returns    result matrix
 */
PLAY_CGLM_INLINE
mat4s
look_anyup(vec3s eye, vec3s dir)
{
#if PLAY_CGLM_CONFIG_CLIP_CONTROL == PLAY_CGLM_CLIP_CONTROL_LH_ZO
    return look_anyup_lh_zo(eye, dir);
#elif PLAY_CGLM_CONFIG_CLIP_CONTROL == PLAY_CGLM_CLIP_CONTROL_LH_NO
    return look_anyup_lh_no(eye, dir);
#elif PLAY_CGLM_CONFIG_CLIP_CONTROL == PLAY_CGLM_CLIP_CONTROL_RH_ZO
    return look_anyup_rh_zo(eye, dir);
#elif PLAY_CGLM_CONFIG_CLIP_CONTROL == PLAY_CGLM_CLIP_CONTROL_RH_NO
    return look_anyup_rh_no(eye, dir);
#endif
}

/*!
 * @brief decomposes frustum values of perspective projection.
 *
 * @param[in]  proj    perspective projection matrix
 * @param[out] nearZ   near
 * @param[out] farZ    far
 * @param[out] top     top
 * @param[out] bottom  bottom
 * @param[out] left    left
 * @param[out] right   right
 */
PLAY_CGLM_INLINE
void
persp_decomp(mat4s proj,
             float * __restrict nearZ, float * __restrict farZ,
             float * __restrict top,   float * __restrict bottom,
             float * __restrict left,  float * __restrict right)
{
#if PLAY_CGLM_CONFIG_CLIP_CONTROL == PLAY_CGLM_CLIP_CONTROL_LH_ZO
    persp_decomp_lh_zo(proj, nearZ, farZ, top, bottom, left, right);
#elif PLAY_CGLM_CONFIG_CLIP_CONTROL == PLAY_CGLM_CLIP_CONTROL_LH_NO
    persp_decomp_lh_no(proj, nearZ, farZ, top, bottom, left, right);
#elif PLAY_CGLM_CONFIG_CLIP_CONTROL == PLAY_CGLM_CLIP_CONTROL_RH_ZO
    persp_decomp_rh_zo(proj, nearZ, farZ, top, bottom, left, right);
#elif PLAY_CGLM_CONFIG_CLIP_CONTROL == PLAY_CGLM_CLIP_CONTROL_RH_NO
    persp_decomp_rh_no(proj, nearZ, farZ, top, bottom, left, right);
#endif
}

/*!
 * @brief decomposes frustum values of perspective projection.
 *        this makes easy to get all values at once
 *
 * @param[in]  proj   perspective projection matrix
 * @param[out] dest   array
 */
PLAY_CGLM_INLINE
void
persp_decompv(mat4s proj, float dest[6])
{
#if PLAY_CGLM_CONFIG_CLIP_CONTROL == PLAY_CGLM_CLIP_CONTROL_LH_ZO
    persp_decompv_lh_zo(proj, dest);
#elif PLAY_CGLM_CONFIG_CLIP_CONTROL == PLAY_CGLM_CLIP_CONTROL_LH_NO
    persp_decompv_lh_no(proj, dest);
#elif PLAY_CGLM_CONFIG_CLIP_CONTROL == PLAY_CGLM_CLIP_CONTROL_RH_ZO
    persp_decompv_rh_zo(proj, dest);
#elif PLAY_CGLM_CONFIG_CLIP_CONTROL == PLAY_CGLM_CLIP_CONTROL_RH_NO
    persp_decompv_rh_no(proj, dest);
#endif
}

/*!
 * @brief decomposes left and right values of perspective projection.
 *        x stands for x axis (left / right axis)
 *
 * @param[in]  proj  perspective projection matrix
 * @param[out] left  left
 * @param[out] right right
 */
PLAY_CGLM_INLINE
void
persp_decomp_x(mat4s proj,
               float * __restrict left,
               float * __restrict right)
{
#if PLAY_CGLM_CONFIG_CLIP_CONTROL == PLAY_CGLM_CLIP_CONTROL_LH_ZO
    persp_decomp_x_lh_zo(proj, left, right);
#elif PLAY_CGLM_CONFIG_CLIP_CONTROL == PLAY_CGLM_CLIP_CONTROL_LH_NO
    persp_decomp_x_lh_no(proj, left, right);
#elif PLAY_CGLM_CONFIG_CLIP_CONTROL == PLAY_CGLM_CLIP_CONTROL_RH_ZO
    persp_decomp_x_rh_zo(proj, left, right);
#elif PLAY_CGLM_CONFIG_CLIP_CONTROL == PLAY_CGLM_CLIP_CONTROL_RH_NO
    persp_decomp_x_rh_no(proj, left, right);
#endif
}

/*!
 * @brief decomposes top and bottom values of perspective projection.
 *        y stands for y axis (top / bottom axis)
 *
 * @param[in]  proj   perspective projection matrix
 * @param[out] top    top
 * @param[out] bottom bottom
 */
PLAY_CGLM_INLINE
void
persp_decomp_y(mat4s proj,
               float * __restrict top,
               float * __restrict bottom)
{
#if PLAY_CGLM_CONFIG_CLIP_CONTROL == PLAY_CGLM_CLIP_CONTROL_LH_ZO
    persp_decomp_y_lh_zo(proj, top, bottom);
#elif PLAY_CGLM_CONFIG_CLIP_CONTROL == PLAY_CGLM_CLIP_CONTROL_LH_NO
    persp_decomp_y_lh_no(proj, top, bottom);
#elif PLAY_CGLM_CONFIG_CLIP_CONTROL == PLAY_CGLM_CLIP_CONTROL_RH_ZO
    persp_decomp_y_rh_zo(proj, top, bottom);
#elif PLAY_CGLM_CONFIG_CLIP_CONTROL == PLAY_CGLM_CLIP_CONTROL_RH_NO
    persp_decomp_y_rh_no(proj, top, bottom);
#endif
}

/*!
 * @brief decomposes near and far values of perspective projection.
 *        z stands for z axis (near / far axis)
 *
 * @param[in]  proj    perspective projection matrix
 * @param[out] nearZ   near
 * @param[out] farZ    far
 */
PLAY_CGLM_INLINE
void
persp_decomp_z(mat4s proj,
               float * __restrict nearZ,
               float * __restrict farZ)
{
#if PLAY_CGLM_CONFIG_CLIP_CONTROL == PLAY_CGLM_CLIP_CONTROL_LH_ZO
    persp_decomp_z_lh_zo(proj, nearZ, farZ);
#elif PLAY_CGLM_CONFIG_CLIP_CONTROL == PLAY_CGLM_CLIP_CONTROL_LH_NO
    persp_decomp_z_lh_no(proj, nearZ, farZ);
#elif PLAY_CGLM_CONFIG_CLIP_CONTROL == PLAY_CGLM_CLIP_CONTROL_RH_ZO
    persp_decomp_z_rh_zo(proj, nearZ, farZ);
#elif PLAY_CGLM_CONFIG_CLIP_CONTROL == PLAY_CGLM_CLIP_CONTROL_RH_NO
    persp_decomp_z_rh_no(proj, nearZ, farZ);
#endif
}

/*!
 * @brief decomposes far value of perspective projection.
 *
 * @param[in]  proj   perspective projection matrix
 * @param[out] farZ   far
 */
PLAY_CGLM_INLINE
void
persp_decomp_far(mat4s proj, float * __restrict farZ)
{
#if PLAY_CGLM_CONFIG_CLIP_CONTROL == PLAY_CGLM_CLIP_CONTROL_LH_ZO
    persp_decomp_far_lh_zo(proj, farZ);
#elif PLAY_CGLM_CONFIG_CLIP_CONTROL == PLAY_CGLM_CLIP_CONTROL_LH_NO
    persp_decomp_far_lh_no(proj, farZ);
#elif PLAY_CGLM_CONFIG_CLIP_CONTROL == PLAY_CGLM_CLIP_CONTROL_RH_ZO
    persp_decomp_far_rh_zo(proj, farZ);
#elif PLAY_CGLM_CONFIG_CLIP_CONTROL == PLAY_CGLM_CLIP_CONTROL_RH_NO
    persp_decomp_far_rh_no(proj, farZ);
#endif
}

/*!
 * @brief decomposes near value of perspective projection.
 *
 * @param[in]  proj  perspective projection matrix
 * @param[out] nearZ near
 */
PLAY_CGLM_INLINE
void
persp_decomp_near(mat4s proj, float * __restrict nearZ)
{
#if PLAY_CGLM_CONFIG_CLIP_CONTROL == PLAY_CGLM_CLIP_CONTROL_LH_ZO
    persp_decomp_near_lh_zo(proj, nearZ);
#elif PLAY_CGLM_CONFIG_CLIP_CONTROL == PLAY_CGLM_CLIP_CONTROL_LH_NO
    persp_decomp_near_lh_no(proj, nearZ);
#elif PLAY_CGLM_CONFIG_CLIP_CONTROL == PLAY_CGLM_CLIP_CONTROL_RH_ZO
    persp_decomp_near_rh_zo(proj, nearZ);
#elif PLAY_CGLM_CONFIG_CLIP_CONTROL == PLAY_CGLM_CLIP_CONTROL_RH_NO
    persp_decomp_near_rh_no(proj, nearZ);
#endif
}

/*!
 * @brief returns field of view angle along the Y-axis (in radians)
 *
 * if you need to degrees, use deg to convert it or use this:
 * fovy_deg = deg(persp_fovy(projMatrix))
 *
 * @param[in] proj perspective projection matrix
 */
PLAY_CGLM_INLINE
float
persp_fovy(mat4s proj)
{
#if PLAY_CGLM_CONFIG_CLIP_CONTROL == PLAY_CGLM_CLIP_CONTROL_LH_ZO
    return persp_fovy_lh_zo(proj);
#elif PLAY_CGLM_CONFIG_CLIP_CONTROL == PLAY_CGLM_CLIP_CONTROL_LH_NO
    return persp_fovy_lh_no(proj);
#elif PLAY_CGLM_CONFIG_CLIP_CONTROL == PLAY_CGLM_CLIP_CONTROL_RH_ZO
    return persp_fovy_rh_zo(proj);
#elif PLAY_CGLM_CONFIG_CLIP_CONTROL == PLAY_CGLM_CLIP_CONTROL_RH_NO
    return persp_fovy_rh_no(proj);
#endif
}

/*!
 * @brief returns aspect ratio of perspective projection
 *
 * @param[in] proj perspective projection matrix
 */
PLAY_CGLM_INLINE
float
persp_aspect(mat4s proj)
{
#if PLAY_CGLM_CONFIG_CLIP_CONTROL == PLAY_CGLM_CLIP_CONTROL_LH_ZO
    return persp_aspect_lh_zo(proj);
#elif PLAY_CGLM_CONFIG_CLIP_CONTROL == PLAY_CGLM_CLIP_CONTROL_LH_NO
    return persp_aspect_lh_no(proj);
#elif PLAY_CGLM_CONFIG_CLIP_CONTROL == PLAY_CGLM_CLIP_CONTROL_RH_ZO
    return persp_aspect_rh_zo(proj);
#elif PLAY_CGLM_CONFIG_CLIP_CONTROL == PLAY_CGLM_CLIP_CONTROL_RH_NO
    return persp_aspect_rh_no(proj);
#endif
}

/*!
 * @brief returns sizes of near and far planes of perspective projection
 *
 * @param[in]  proj perspective projection matrix
 * @param[in]  fovy fovy (see brief)
 * @returns    sizes as vector, sizes order: [Wnear, Hnear, Wfar, Hfar]
 */
PLAY_CGLM_INLINE
vec4s
persp_sizes(mat4s proj, float fovy)
{
#if PLAY_CGLM_CONFIG_CLIP_CONTROL == PLAY_CGLM_CLIP_CONTROL_LH_ZO
    return persp_sizes_lh_zo(proj, fovy);
#elif PLAY_CGLM_CONFIG_CLIP_CONTROL == PLAY_CGLM_CLIP_CONTROL_LH_NO
    return persp_sizes_lh_no(proj, fovy);
#elif PLAY_CGLM_CONFIG_CLIP_CONTROL == PLAY_CGLM_CLIP_CONTROL_RH_ZO
    return persp_sizes_rh_zo(proj, fovy);
#elif PLAY_CGLM_CONFIG_CLIP_CONTROL == PLAY_CGLM_CLIP_CONTROL_RH_NO
    return persp_sizes_rh_no(proj, fovy);
#endif
}



/*** End of inlined file: cam.h ***/


/*** Start of inlined file: quat.h ***/
/*
 Macros:
   PLAY_CGLM_S_QUAT_IDENTITY_INIT
   PLAY_CGLM_S_QUAT_IDENTITY

 Functions:
   PLAY_CGLM_INLINE versors quats_identity(void)
   PLAY_CGLM_INLINE void    quats_identity_array(versor *q, size_t count)
   PLAY_CGLM_INLINE versors quats_init(float x, float y, float z, float w)
   PLAY_CGLM_INLINE versors quatv(float angle, vec3s axis)
   PLAY_CGLM_INLINE versors quat(float angle, float x, float y, float z)
   PLAY_CGLM_INLINE versors quats_from_vecs(vec3s a, vec3s b)
   PLAY_CGLM_INLINE float   quats_norm(versors q)
   PLAY_CGLM_INLINE versors quats_normalize(versors q)
   PLAY_CGLM_INLINE float   quats_dot(versors p, versors q)
   PLAY_CGLM_INLINE versors quats_conjugate(versors q)
   PLAY_CGLM_INLINE versors quats_inv(versors q)
   PLAY_CGLM_INLINE versors quats_add(versors p, versors q)
   PLAY_CGLM_INLINE versors quats_sub(versors p, versors q)
   PLAY_CGLM_INLINE vec3s   quats_imagn(versors q)
   PLAY_CGLM_INLINE float   quats_imaglen(versors q)
   PLAY_CGLM_INLINE float   quats_angle(versors q)
   PLAY_CGLM_INLINE vec3s   quats_axis(versors q)
   PLAY_CGLM_INLINE versors quats_mul(versors p, versors q)
   PLAY_CGLM_INLINE mat4s   quats_mat4(versors q)
   PLAY_CGLM_INLINE mat4s   quats_mat4t(versors q)
   PLAY_CGLM_INLINE mat3s   quats_mat3(versors q)
   PLAY_CGLM_INLINE mat3s   quats_mat3t(versors q)
   PLAY_CGLM_INLINE versors quats_lerp(versors from, versors to, float t)
   PLAY_CGLM_INLINE versors quats_lerpc(versors from, versors to, float t)
   PLAY_CGLM_INLINE versors quats_nlerp(versors from, versors to, float t)
   PLAY_CGLM_INLINE versors quats_slerp(versors from, versors to, float t)
   PLAY_CGLM_INLINE versors quats_slerp_longest(versors from, versors to, float t)
   PLAY_CGLM_INLINE mat4s.  quats_look(vec3s eye, versors ori)
   PLAY_CGLM_INLINE versors quats_for(vec3s dir, vec3s fwd, vec3s up)
   PLAY_CGLM_INLINE versors quats_forp(vec3s from, vec3s to, vec3s fwd, vec3s up)
   PLAY_CGLM_INLINE vec3s   quats_rotatev(versors q, vec3s v)
   PLAY_CGLM_INLINE mat4s   quats_rotate(mat4s m, versors q)
   PLAY_CGLM_INLINE mat4s   quats_rotate_at(mat4s m, versors q, vec3s pivot)
   PLAY_CGLM_INLINE mat4s   quats_rotate_atm(versors q, vec3s pivot)
   PLAY_CGLM_INLINE versors quats_make(float * restrict src)
 */

#ifndef cquats_h
#define cquats_h

/* api definition */
#define quats_(NAME) PLAY_CGLM_STRUCTAPI(quat, NAME)

/*
 * IMPORTANT:
 * ----------------------------------------------------------------------------
 * cglm stores quat as [x, y, z, w] since v0.3.6
 *
 * it was [w, x, y, z] before v0.3.6 it has been changed to [x, y, z, w]
 * with v0.3.6 version.
 * ----------------------------------------------------------------------------
 */

#define PLAY_CGLM_S_QUAT_IDENTITY_INIT  {PLAY_CGLM_QUAT_IDENTITY_INIT}
#define PLAY_CGLM_S_QUAT_IDENTITY       ((versors)PLAY_CGLM_S_QUAT_IDENTITY_INIT)

/*!
 * @brief makes given quat to identity
 *
 * @returns identity quaternion
 */
PLAY_CGLM_INLINE
versors
quats_(identity)(void)
{
    versors dest;
    quat_identity(dest.raw);
    return dest;
}

/*!
 * @brief make given quaternion array's each element identity quaternion
 *
 * @param[in, out]  q     quat array (must be aligned (16)
 *                        if alignment is not disabled)
 *
 * @param[in]       count count of quaternions
 */
PLAY_CGLM_INLINE
void
quats_(identity_array)(versors * __restrict q, size_t count)
{
    PLAY_CGLM_ALIGN(16) versor v = PLAY_CGLM_QUAT_IDENTITY_INIT;
    size_t i;

    for (i = 0; i < count; i++)
    {
        vec4_copy(v, q[i].raw);
    }
}

/*!
 * @brief inits quaternion with raw values
 *
 * @param[in]   x     x
 * @param[in]   y     y
 * @param[in]   z     z
 * @param[in]   w     w (real part)
 * @returns quaternion
 */
PLAY_CGLM_INLINE
versors
quats_(init)(float x, float y, float z, float w)
{
    versors dest;
    quat_init(dest.raw, x, y, z, w);
    return dest;
}

/*!
 * @brief creates NEW quaternion with axis vector
 *
 * @param[in]   angle angle (radians)
 * @param[in]   axis  axis
 * @returns quaternion
 */
PLAY_CGLM_INLINE
versors
quatv(float angle, vec3s axis)
{
    versors dest;
    quatv(dest.raw, angle, axis.raw);
    return dest;
}

/*!
 * @brief creates NEW quaternion with individual axis components
 *
 * @param[in]   angle angle (radians)
 * @param[in]   x     axis.x
 * @param[in]   y     axis.y
 * @param[in]   z     axis.z
 * @returns quaternion
 */
PLAY_CGLM_INLINE
versors
quat(float angle, float x, float y, float z)
{
    versors dest;
    quat(dest.raw, angle, x, y, z);
    return dest;
}

/*!
 * @brief compute quaternion rotating vector A to vector B
 *
 * @param[in]   a     vec3 (must have unit length)
 * @param[in]   b     vec3 (must have unit length)
 * @returns     quaternion (of unit length)
 */
PLAY_CGLM_INLINE
versors
quats_(from_vecs)(vec3s a, vec3s b)
{
    versors dest;
    quat_from_vecs(a.raw, b.raw, dest.raw);
    return dest;
}

/*!
 * @brief returns norm (magnitude) of quaternion
 *
 * @param[in]  q  quaternion
 */
PLAY_CGLM_INLINE
float
quats_(norm)(versors q)
{
    return quat_norm(q.raw);
}

/*!
 * @brief normalize quaternion
 *
 * @param[in]  q  quaternion
 * @returns    quaternion
 */
PLAY_CGLM_INLINE
versors
quats_(normalize)(versors q)
{
    versors dest;
    quat_normalize_to(q.raw, dest.raw);
    return dest;
}

/*!
 * @brief dot product of two quaternion
 *
 * @param[in]  p  quaternion 1
 * @param[in]  q  quaternion 2
 * @returns    dot product
 */
PLAY_CGLM_INLINE
float
quats_(dot)(versors p, versors q)
{
    return quat_dot(p.raw, q.raw);
}

/*!
 * @brief conjugate of quaternion
 *
 * @param[in]   q     quaternion
 * @returns    conjugate
 */
PLAY_CGLM_INLINE
versors
quats_(conjugate)(versors q)
{
    versors dest;
    quat_conjugate(q.raw, dest.raw);
    return dest;
}

/*!
 * @brief inverse of non-zero quaternion
 *
 * @param[in]  q    quaternion
 * @returns    inverse quaternion
 */
PLAY_CGLM_INLINE
versors
quats_(inv)(versors q)
{
    versors dest;
    quat_inv(q.raw, dest.raw);
    return dest;
}

/*!
 * @brief add (componentwise) two quaternions and store result in dest
 *
 * @param[in]   p    quaternion 1
 * @param[in]   q    quaternion 2
 * @returns result quaternion
 */
PLAY_CGLM_INLINE
versors
quats_(add)(versors p, versors q)
{
    versors dest;
    quat_add(p.raw, q.raw, dest.raw);
    return dest;
}

/*!
 * @brief subtract (componentwise) two quaternions and store result in dest
 *
 * @param[in]   p    quaternion 1
 * @param[in]   q    quaternion 2
 * @returns result quaternion
 */
PLAY_CGLM_INLINE
versors
quats_(sub)(versors p, versors q)
{
    versors dest;
    quat_sub(p.raw, q.raw, dest.raw);
    return dest;
}

/*!
 * @brief returns normalized imaginary part of quaternion
 *
 * @param[in]   q    quaternion
 */
PLAY_CGLM_INLINE
vec3s
quats_(imagn)(versors q)
{
    vec3s dest;
    normalize_to(q.raw, dest.raw);
    return dest;
}

/*!
 * @brief returns length of imaginary part of quaternion
 *
 * @param[in]   q    quaternion
 */
PLAY_CGLM_INLINE
float
quats_(imaglen)(versors q)
{
    return quat_imaglen(q.raw);
}

/*!
 * @brief returns angle of quaternion
 *
 * @param[in]   q    quaternion
 */
PLAY_CGLM_INLINE
float
quats_(angle)(versors q)
{
    return quat_angle(q.raw);
}

/*!
 * @brief axis of quaternion
 *
 * @param[in]   q    quaternion
 * @returns axis of quaternion
 */
PLAY_CGLM_INLINE
vec3s
quats_(axis)(versors q)
{
    vec3s dest;
    quat_axis(q.raw, dest.raw);
    return dest;
}

/*!
 * @brief multiplies two quaternion and stores result in dest
 *        this is also called Hamilton Product
 *
 * According to WikiPedia:
 * The product of two rotation quaternions [clarification needed] will be
 * equivalent to the rotation q followed by the rotation p
 *
 * @param[in]   p     quaternion 1
 * @param[in]   q     quaternion 2
 * @returns  result quaternion
 */
PLAY_CGLM_INLINE
versors
quats_(mul)(versors p, versors q)
{
    versors dest;
    quat_mul(p.raw, q.raw, dest.raw);
    return dest;
}

/*!
 * @brief convert quaternion to mat4
 *
 * @param[in]   q     quaternion
 * @returns  result matrix
 */
PLAY_CGLM_INLINE
mat4s
quats_(mat4)(versors q)
{
    mat4s dest;
    quat_mat4(q.raw, dest.raw);
    return dest;
}

/*!
 * @brief convert quaternion to mat4 (transposed)
 *
 * @param[in]   q     quaternion
 * @returns  result matrix as transposed
 */
PLAY_CGLM_INLINE
mat4s
quats_(mat4t)(versors q)
{
    mat4s dest;
    quat_mat4t(q.raw, dest.raw);
    return dest;
}

/*!
 * @brief convert quaternion to mat3
 *
 * @param[in]   q     quaternion
 * @returns  result matrix
 */
PLAY_CGLM_INLINE
mat3s
quats_(mat3)(versors q)
{
    mat3s dest;
    quat_mat3(q.raw, dest.raw);
    return dest;
}

/*!
 * @brief convert quaternion to mat3 (transposed)
 *
 * @param[in]   q     quaternion
 * @returns  result matrix
 */
PLAY_CGLM_INLINE
mat3s
quats_(mat3t)(versors q)
{
    mat3s dest;
    quat_mat3t(q.raw, dest.raw);
    return dest;
}

/*!
 * @brief interpolates between two quaternions
 *        using linear interpolation (LERP)
 *
 * @param[in]   from  from
 * @param[in]   to    to
 * @param[in]   t     interpolant (amount)
 * @returns  result quaternion
 */
PLAY_CGLM_INLINE
versors
quats_(lerp)(versors from, versors to, float t)
{
    versors dest;
    quat_lerp(from.raw, to.raw, t, dest.raw);
    return dest;
}

/*!
 * @brief interpolates between two quaternions
 *        using linear interpolation (LERP)
 *
 * @param[in]   from  from
 * @param[in]   to    to
 * @param[in]   t     interpolant (amount) clamped between 0 and 1
 * @returns  result quaternion
 */
PLAY_CGLM_INLINE
versors
quats_(lerpc)(versors from, versors to, float t)
{
    versors dest;
    quat_lerpc(from.raw, to.raw, t, dest.raw);
    return dest;
}

/*!
 * @brief interpolates between two quaternions
 *        taking the shortest rotation path using
 *        normalized linear interpolation (NLERP)
 *
 * @param[in]   from  from
 * @param[in]   to    to
 * @param[in]   t     interpolant (amount)
 * @returns result quaternion
 */
PLAY_CGLM_INLINE
versors
quats_(nlerp)(versors from, versors to, float t)
{
    versors dest;
    quat_nlerp(from.raw, to.raw, t, dest.raw);
    return dest;
}

/*!
 * @brief interpolates between two quaternions
 *        using spherical linear interpolation (SLERP)
 *
 * @param[in]   from  from
 * @param[in]   to    to
 * @param[in]   t     amount
 * @returns result quaternion
 */
PLAY_CGLM_INLINE
versors
quats_(slerp)(versors from, versors to, float t)
{
    versors dest;
    quat_slerp(from.raw, to.raw, t, dest.raw);
    return dest;
}

/*!
 * @brief interpolates between two quaternions
 *        using spherical linear interpolation (SLERP) and always takes the longest path
 *
 * @param[in]   from  from
 * @param[in]   to    to
 * @param[in]   t     amount
 * @returns result quaternion
 */
PLAY_CGLM_INLINE
versors
quats_(slerp_longest)(versors from, versors to, float t)
{
    versors dest;
    quat_slerp_longest(from.raw, to.raw, t, dest.raw);
    return dest;
}

/*!
 * @brief creates view matrix using quaternion as camera orientation
 *
 * @param[in]   eye   eye
 * @param[in]   ori   orientation in world space as quaternion
 * @returns  view matrix
 */
PLAY_CGLM_INLINE
mat4s
quats_(look)(vec3s eye, versors ori)
{
    mat4s dest;
    quat_look(eye.raw, ori.raw, dest.raw);
    return dest;
}

/*!
 * @brief creates look rotation quaternion
 *
 * @param[in]   dir   direction to look
 * @param[in]   up    up vector
 * @returns  destination quaternion
 */
PLAY_CGLM_INLINE
versors
quats_(for)(vec3s dir, vec3s up)
{
    versors dest;
    quat_for(dir.raw, up.raw, dest.raw);
    return dest;
}

/*!
 * @brief creates look rotation quaternion using source and
 *        destination positions p suffix stands for position
 *
 * @param[in]   from  source point
 * @param[in]   to    destination point
 * @param[in]   up    up vector
 * @returns  destination quaternion
 */
PLAY_CGLM_INLINE
versors
quats_(forp)(vec3s from, vec3s to, vec3s up)
{
    versors dest;
    quat_forp(from.raw, to.raw, up.raw, dest.raw);
    return dest;
}

/*!
 * @brief rotate vector using using quaternion
 *
 * @param[in]   q     quaternion
 * @param[in]   v     vector to rotate
 * @returns  rotated vector
 */
PLAY_CGLM_INLINE
vec3s
quats_(rotatev)(versors q, vec3s v)
{
    vec3s dest;
    quat_rotatev(q.raw, v.raw, dest.raw);
    return dest;
}

/*!
 * @brief rotate existing transform matrix using quaternion
 *
 * @param[in]   m     existing transform matrix
 * @param[in]   q     quaternion
 * @returns  rotated matrix/transform
 */
PLAY_CGLM_INLINE
mat4s
quats_(rotate)(mat4s m, versors q)
{
    quat_rotate(m.raw, q.raw, m.raw);
    return m;
}

/*!
 * @brief rotate existing transform matrix using quaternion at pivot point
 *
 * @param[in, out]   m     existing transform matrix
 * @param[in]        q     quaternion
 * @returns pivot
 */
PLAY_CGLM_INLINE
mat4s
quats_(rotate_at)(mat4s m, versors q, vec3s pivot)
{
    quat_rotate_at(m.raw, q.raw, pivot.raw);
    return m;
}

/*!
 * @brief rotate NEW transform matrix using quaternion at pivot point
 *
 * this creates rotation matrix, it assumes you don't have a matrix
 *
 * this should work faster than quat_rotate_at because it reduces
 * one translate.
 *
 * @param[in]   q     quaternion
 * @returns pivot
 */
PLAY_CGLM_INLINE
mat4s
quats_(rotate_atm)(versors q, vec3s pivot)
{
    mat4s dest;
    quat_rotate_atm(dest.raw, q.raw, pivot.raw);
    return dest;
}

/*!
 * @brief Create CGLM quaternion from pointer
 *
 * @param[in]  src  pointer to an array of floats
 * @returns constructed quaternion from raw pointer
 */
PLAY_CGLM_INLINE
versors
quats_(make)(const float * __restrict src)
{
    versors dest;
    quat_make(src, dest.raw);
    return dest;
}

#endif /* cquats_h */

/*** End of inlined file: quat.h ***/


/*** Start of inlined file: euler.h ***/
/*
 NOTE:
  angles must be passed as [X-Angle, Y-Angle, Z-angle] order
  For instance you don't pass angles as [Z-Angle, X-Angle, Y-angle] to
  euler_zxy function, All RELATED functions accept angles same order
  which is [X, Y, Z].
 */

/*
 Types:
   enum euler_seq

 Functions:
   PLAY_CGLM_INLINE vec3s euler_angles(mat4s m)
   PLAY_CGLM_INLINE mat4s euler_xyz(vec3s angles)
   PLAY_CGLM_INLINE mat4s euler_xzy(vec3s angles)
   PLAY_CGLM_INLINE mat4s euler_yxz(vec3s angles)
   PLAY_CGLM_INLINE mat4s euler_yzx(vec3s angles)
   PLAY_CGLM_INLINE mat4s euler_zxy(vec3s angles)
   PLAY_CGLM_INLINE mat4s euler_zyx(vec3s angles)
   PLAY_CGLM_INLINE mat4s euler_by_order(vec3s angles, euler_seq ord)
   PLAY_CGLM_INLINE versors euler_xyz_quat(vec3s angles)
   PLAY_CGLM_INLINE versors euler_xzy_quat(vec3s angles)
   PLAY_CGLM_INLINE versors euler_yxz_quat(vec3s angles)
   PLAY_CGLM_INLINE versors euler_yzx_quat(vec3s angles)
   PLAY_CGLM_INLINE versors euler_zxy_quat(vec3s angles)
   PLAY_CGLM_INLINE versors euler_zyx_quat(vec3s angles)
 */




/*!
 * @brief extract euler angles (in radians) using xyz order
 *
 * @param[in]  m    affine transform
 * @returns angles vector [x, y, z]
 */
PLAY_CGLM_INLINE
vec3s
euler_angles(mat4s m)
{
    vec3s dest;
    euler_angles(m.raw, dest.raw);
    return dest;
}

/*!
 * @brief build rotation matrix from euler angles
 *
 * @param[in]  angles angles as vector [Xangle, Yangle, Zangle]
 * @returns rotation matrix
 */
PLAY_CGLM_INLINE
mat4s
euler_xyz(vec3s angles)
{
    mat4s dest;
    euler_xyz(angles.raw, dest.raw);
    return dest;
}

/*!
 * @brief build rotation matrix from euler angles
 *
 * @param[in]  angles angles as vector [Xangle, Yangle, Zangle]
 * @returns rotation matrix
 */
PLAY_CGLM_INLINE
mat4s
euler_xzy(vec3s angles)
{
    mat4s dest;
    euler_xzy(angles.raw, dest.raw);
    return dest;
}

/*!
 * @brief build rotation matrix from euler angles
 *
 * @param[in]  angles angles as vector [Xangle, Yangle, Zangle]
 * @returns rotation matrix
 */
PLAY_CGLM_INLINE
mat4s
euler_yxz(vec3s angles)
{
    mat4s dest;
    euler_yxz(angles.raw, dest.raw);
    return dest;
}

/*!
 * @brief build rotation matrix from euler angles
 *
 * @param[in]  angles angles as vector [Xangle, Yangle, Zangle]
 * @returns rotation matrix
 */
PLAY_CGLM_INLINE
mat4s
euler_yzx(vec3s angles)
{
    mat4s dest;
    euler_yzx(angles.raw, dest.raw);
    return dest;
}

/*!
 * @brief build rotation matrix from euler angles
 *
 * @param[in]  angles angles as vector [Xangle, Yangle, Zangle]
 * @returns rotation matrix
 */
PLAY_CGLM_INLINE
mat4s
euler_zxy(vec3s angles)
{
    mat4s dest;
    euler_zxy(angles.raw, dest.raw);
    return dest;
}

/*!
 * @brief build rotation matrix from euler angles
 *
 * @param[in]  angles angles as vector [Xangle, Yangle, Zangle]
 * @returns rotation matrix
 */
PLAY_CGLM_INLINE
mat4s
euler_zyx(vec3s angles)
{
    mat4s dest;
    euler_zyx(angles.raw, dest.raw);
    return dest;
}

/*!
 * @brief build rotation matrix from euler angles
 *
 * @param[in]  angles angles as vector [Xangle, Yangle, Zangle]
 * @param[in]  ord    euler order
 * @returns rotation matrix
 */
PLAY_CGLM_INLINE
mat4s
euler_by_order(vec3s angles, euler_seq ord)
{
    mat4s dest;
    euler_by_order(angles.raw, ord, dest.raw);
    return dest;
}

/*!
 * @brief creates NEW quaternion using rotation angles and does
 *        rotations in x y z order (roll pitch yaw)
 *
 * @param[in]   angles angles x y z (radians)
 * @returns quaternion
 */
PLAY_CGLM_INLINE
versors
euler_xyz_quat(vec3s angles)
{
    versors dest;
    euler_xyz_quat(angles.raw, dest.raw);
    return dest;
}

/*!
 * @brief creates NEW quaternion using rotation angles and does
 *        rotations in x z y order (roll yaw pitch)
 *
 * @param[in]   angles angles x y z (radians)
 * @returns quaternion
 */
PLAY_CGLM_INLINE
versors
euler_xzy_quat(vec3s angles)
{
    versors dest;
    euler_xzy_quat(angles.raw, dest.raw);
    return dest;
}

/*!
 * @brief creates NEW quaternion using rotation angles and does
 *        rotations in y x z order (pitch roll yaw)
 *
 * @param[in]   angles angles x y z (radians)
 * @returns quaternion
 */
PLAY_CGLM_INLINE
versors
euler_yxz_quat(vec3s angles)
{
    versors dest;
    euler_yxz_quat(angles.raw, dest.raw);
    return dest;
}

/*!
 * @brief creates NEW quaternion using rotation angles and does
 *        rotations in y z x order (pitch yaw roll)
 *
 * @param[in]   angles angles x y z (radians)
 * @returns quaternion
 */
PLAY_CGLM_INLINE
versors
euler_yzx_quat(vec3s angles)
{
    versors dest;
    euler_yzx_quat(angles.raw, dest.raw);
    return dest;
}

/*!
 * @brief creates NEW quaternion using rotation angles and does
 *        rotations in z x y order (yaw roll pitch)
 *
 * @param[in]   angles angles x y z (radians)
 * @returns quaternion
 */
PLAY_CGLM_INLINE
versors
euler_zxy_quat(vec3s angles)
{
    versors dest;
    euler_zxy_quat(angles.raw, dest.raw);
    return dest;
}

/*!
 * @brief creates NEW quaternion using rotation angles and does
 *        rotations in z y x order (yaw pitch roll)
 *
 * @param[in]   angles angles x y z (radians)
 * @returns quaternion
 */
PLAY_CGLM_INLINE
versors
euler_zyx_quat(vec3s angles)
{
    versors dest;
    euler_zyx_quat(angles.raw, dest.raw);
    return dest;
}



/*** End of inlined file: euler.h ***/


/*** Start of inlined file: project.h ***/




/*** Start of inlined file: project_zo.h ***/
/*
 Functions:
   PLAY_CGLM_INLINE vec3s unprojecti_no(vec3s pos, mat4s invMat, vec4s vp)
   PLAY_CGLM_INLINE vec3s project_no(vec3s pos, mat4s m, vec4s vp)
   PLAY_CGLM_INLINE float project_z_zo(vec3s v, mat4s m)
 */




/*!
 * @brief maps the specified viewport coordinates into specified space [1]
 *        the matrix should contain projection matrix.
 *
 * if you don't have ( and don't want to have ) an inverse matrix then use
 * unproject version. You may use existing inverse of matrix in somewhere
 * else, this is why unprojecti exists to save save inversion cost
 *
 * [1] space:
 *  1- if m = invProj:     View Space
 *  2- if m = invViewProj: World Space
 *  3- if m = invMVP:      Object Space
 *
 * You probably want to map the coordinates into object space
 * so use invMVP as m
 *
 * Computing viewProj:
 *   mat4_mul(proj, view, viewProj);
 *   mat4_mul(viewProj, model, MVP);
 *   mat4_inv(viewProj, invMVP);
 *
 * @param[in]  pos          point/position in viewport coordinates
 * @param[in]  invMat   matrix (see brief)
 * @param[in]  vp            viewport as [x, y, width, height]
 *
 * @returns unprojected coordinates
 */
PLAY_CGLM_INLINE
vec3s
unprojecti_zo(vec3s pos, mat4s invMat, vec4s vp)
{
    vec3s dest;
    unprojecti_zo(pos.raw, invMat.raw, vp.raw, dest.raw);
    return dest;
}

/*!
 * @brief map object coordinates to window coordinates
 *
 * Computing MVP:
 *   mat4_mul(proj, view, viewProj);
 *   mat4_mul(viewProj, model, MVP);
 *
 * @param[in]  pos      object coordinates
 * @param[in]  m          MVP matrix
 * @param[in]  vp        viewport as [x, y, width, height]
 *
 * @returns projected coordinates
 */
PLAY_CGLM_INLINE
vec3s
project_zo(vec3s pos, mat4s m, vec4s vp)
{
    vec3s dest;
    project_zo(pos.raw, m.raw, vp.raw, dest.raw);
    return dest;
}

/*!
 * @brief map object's z coordinate to window coordinates
 *
 * Computing MVP:
 *   mat4_mul(proj, view, viewProj);
 *   mat4_mul(viewProj, model, MVP);
 *
 * @param[in]  v object coordinates
 * @param[in]  m MVP matrix
 *
 * @returns projected z coordinate
 */
PLAY_CGLM_INLINE
float
project_z_zo(vec3s v, mat4s m)
{
    return project_z_zo(v.raw, m.raw);
}



/*** End of inlined file: project_zo.h ***/


/*** Start of inlined file: project_no.h ***/
/*
 Functions:
   PLAY_CGLM_INLINE vec3s unprojecti_no(vec3s pos, mat4s invMat, vec4s vp)
   PLAY_CGLM_INLINE vec3s project_no(vec3s pos, mat4s m, vec4s vp)
   PLAY_CGLM_INLINE float project_z_no(vec3s v, mat4s m)
 */




/*!
 * @brief maps the specified viewport coordinates into specified space [1]
 *        the matrix should contain projection matrix.
 *
 * if you don't have ( and don't want to have ) an inverse matrix then use
 * unproject version. You may use existing inverse of matrix in somewhere
 * else, this is why unprojecti exists to save save inversion cost
 *
 * [1] space:
 *  1- if m = invProj:     View Space
 *  2- if m = invViewProj: World Space
 *  3- if m = invMVP:      Object Space
 *
 * You probably want to map the coordinates into object space
 * so use invMVP as m
 *
 * Computing viewProj:
 *   mat4_mul(proj, view, viewProj);
 *   mat4_mul(viewProj, model, MVP);
 *   mat4_inv(viewProj, invMVP);
 *
 * @param[in]  pos          point/position in viewport coordinates
 * @param[in]  invMat   matrix (see brief)
 * @param[in]  vp            viewport as [x, y, width, height]
 *
 * @returns unprojected coordinates
 */
PLAY_CGLM_INLINE
vec3s
unprojecti_no(vec3s pos, mat4s invMat, vec4s vp)
{
    vec3s dest;
    unprojecti_no(pos.raw, invMat.raw, vp.raw, dest.raw);
    return dest;
}

/*!
 * @brief map object coordinates to window coordinates
 *
 * Computing MVP:
 *   mat4_mul(proj, view, viewProj);
 *   mat4_mul(viewProj, model, MVP);
 *
 * @param[in]  pos      object coordinates
 * @param[in]  m          MVP matrix
 * @param[in]  vp        viewport as [x, y, width, height]
 *
 * @returns projected coordinates
 */
PLAY_CGLM_INLINE
vec3s
project_no(vec3s pos, mat4s m, vec4s vp)
{
    vec3s dest;
    project_no(pos.raw, m.raw, vp.raw, dest.raw);
    return dest;
}

/*!
 * @brief map object's z coordinate to window coordinates
 *
 * Computing MVP:
 *   mat4_mul(proj, view, viewProj);
 *   mat4_mul(viewProj, model, MVP);
 *
 * @param[in]  v object coordinates
 * @param[in]  m MVP matrix
 *
 * @returns projected z coordinate
 */
PLAY_CGLM_INLINE
float
project_z_no(vec3s v, mat4s m)
{
    return project_z_no(v.raw, m.raw);
}



/*** End of inlined file: project_no.h ***/

/*!
 * @brief maps the specified viewport coordinates into specified space [1]
 *        the matrix should contain projection matrix.
 *
 * if you don't have ( and don't want to have ) an inverse matrix then use
 * unproject version. You may use existing inverse of matrix in somewhere
 * else, this is why unprojecti exists to save save inversion cost
 *
 * [1] space:
 *  1- if m = invProj:     View Space
 *  2- if m = invViewProj: World Space
 *  3- if m = invMVP:      Object Space
 *
 * You probably want to map the coordinates into object space
 * so use invMVP as m
 *
 * Computing viewProj:
 *   mat4_mul(proj, view, viewProj);
 *   mat4_mul(viewProj, model, MVP);
 *   mat4_inv(viewProj, invMVP);
 *
 * @param[in]  pos      point/position in viewport coordinates
 * @param[in]  invMat   matrix (see brief)
 * @param[in]  vp       viewport as [x, y, width, height]
 * @returns             unprojected coordinates
 */
PLAY_CGLM_INLINE
vec3s
unprojecti(vec3s pos, mat4s invMat, vec4s vp)
{
    vec3s r;
    unprojecti(pos.raw, invMat.raw, vp.raw, r.raw);
    return r;
}

/*!
 * @brief maps the specified viewport coordinates into specified space [1]
 *        the matrix should contain projection matrix.
 *
 * this is same as unprojecti except this function get inverse matrix for
 * you.
 *
 * [1] space:
 *  1- if m = proj:     View Space
 *  2- if m = viewProj: World Space
 *  3- if m = MVP:      Object Space
 *
 * You probably want to map the coordinates into object space
 * so use MVP as m
 *
 * Computing viewProj and MVP:
 *   mat4_mul(proj, view, viewProj);
 *   mat4_mul(viewProj, model, MVP);
 *
 * or in struct api:
 *   MVP = mat4_mul(mat4_mul(proj, view), model)
 *
 * @param[in]  pos      point/position in viewport coordinates
 * @param[in]  m        matrix (see brief)
 * @param[in]  vp       viewport as [x, y, width, height]
 * @returns             unprojected coordinates
 */
PLAY_CGLM_INLINE
vec3s
unproject(vec3s pos, mat4s m, vec4s vp)
{
    vec3s r;
    unproject(pos.raw, m.raw, vp.raw, r.raw);
    return r;
}

/*!
 * @brief map object coordinates to window coordinates
 *
 * Computing MVP:
 *   mat4_mul(proj, view, viewProj);
 *   mat4_mul(viewProj, model, MVP);
 *
 * or in struct api:
 *   MVP = mat4_mul(mat4_mul(proj, view), model)
 *
 * @param[in]  pos      object coordinates
 * @param[in]  m        MVP matrix
 * @param[in]  vp       viewport as [x, y, width, height]
 * @returns projected coordinates
 */
PLAY_CGLM_INLINE
vec3s
project(vec3s pos, mat4s m, vec4s vp)
{
    vec3s r;
    project(pos.raw, m.raw, vp.raw, r.raw);
    return r;
}

/*!
 * @brief map object's z coordinate to window coordinates
 *
 * Computing MVP:
 *   mat4_mul(proj, view, viewProj);
 *   mat4_mul(viewProj, model, MVP);
 *
 * or in struct api:
 *   MVP = mat4_mul(mat4_mul(proj, view), model)
 *
 * @param[in]  v  object coordinates
 * @param[in]  m  MVP matrix
 *
 * @returns projected z coordinate
 */
PLAY_CGLM_INLINE
float
project_z(vec3s v, mat4s m)
{
#if PLAY_CGLM_CONFIG_CLIP_CONTROL & PLAY_CGLM_CLIP_CONTROL_ZO_BIT
    return project_z_zo(v, m);
#elif PLAY_CGLM_CONFIG_CLIP_CONTROL & PLAY_CGLM_CLIP_CONTROL_NO_BIT
    return project_z_no(v, m);
#endif
}

/*!
 * @brief define a picking region
 *
 * @param[in]  center   center [x, y] of a picking region in window coordinates
 * @param[in]  size     size [width, height] of the picking region in window coordinates
 * @param[in]  vp       viewport as [x, y, width, height]
 * @returns projected coordinates
 */
PLAY_CGLM_INLINE
mat4s
pickmatrix(vec2s center, vec2s size, vec4s vp)
{
    mat4s res;
    pickmatrix(center.raw, size.raw, vp.raw, res.raw);
    return res;
}



/*** End of inlined file: project.h ***/


/*** Start of inlined file: sphere.h ***/



/*
  Sphere Representation in cglm: [center.x, center.y, center.z, radii]

  You could use this representation or you can convert it to vec4 before call
  any function
 */

/*!
 * @brief helper for getting sphere radius
 *
 * @param[in]   s  sphere
 *
 * @return returns radii
 */
PLAY_CGLM_INLINE
float
sphere_radii(vec4s s)
{
    return sphere_radii(s.raw);
}

/*!
 * @brief apply transform to sphere, it is just wrapper for mat4_mulv3
 *
 * @param[in]  s    sphere
 * @param[in]  m    transform matrix
 * @returns         transformed sphere
 */
PLAY_CGLM_INLINE
vec4s
sphere_transform(vec4s s, mat4s m)
{
    vec4s r;
    sphere_transform(s.raw, m.raw, r.raw);
    return r;
}

/*!
 * @brief merges two spheres and creates a new one
 *
 * two sphere must be in same space, for instance if one in world space then
 * the other must be in world space too, not in local space.
 *
 * @param[in]  s1   sphere 1
 * @param[in]  s2   sphere 2
 * returns          merged/extended sphere
 */
PLAY_CGLM_INLINE
vec4s
sphere_merge(vec4s s1, vec4s s2)
{
    vec4s r;
    sphere_merge(s1.raw, s2.raw, r.raw);
    return r;
}

/*!
 * @brief check if two sphere intersects
 *
 * @param[in]   s1  sphere
 * @param[in]   s2  other sphere
 */
PLAY_CGLM_INLINE
bool
sphere_sphere(vec4s s1, vec4s s2)
{
    return sphere_sphere(s1.raw, s2.raw);
}

/*!
 * @brief check if sphere intersects with point
 *
 * @param[in]   s      sphere
 * @param[in]   point  point
 */
PLAY_CGLM_INLINE
bool
sphere_point(vec4s s, vec3s point)
{
    return sphere_point(s.raw, point.raw);
}



/*** End of inlined file: sphere.h ***/


/*** Start of inlined file: curve.h ***/



/*!
 * @brief helper function to calculate S*M*C multiplication for curves
 *
 * This function does not encourage you to use SMC,
 * instead it is a helper if you use SMC.
 *
 * if you want to specify S as vector then use more generic mat4_rmc() func.
 *
 * Example usage:
 *  B(s) = smc(s, PLAY_CGLM_BEZIER_MAT, (vec4){p0, c0, c1, p1})
 *
 * @param[in]  s  parameter between 0 and 1 (this will be [s3, s2, s, 1])
 * @param[in]  m  basis matrix
 * @param[in]  c  position/control vector
 *
 * @return B(s)
 */
PLAY_CGLM_INLINE
float
smc(float s, mat4s m, vec4s c)
{
    return smc(s, m.raw, c.raw);
}



/*** End of inlined file: curve.h ***/


/*** Start of inlined file: affine2d.h ***/
/*
 Functions:
   PLAY_CGLM_INLINE mat3s translate2d(mat3 m, vec2 v)
   PLAY_CGLM_INLINE mat3s translate2d_x(mat3s m, float x)
   PLAY_CGLM_INLINE mat3s translate2d_y(mat3s m, float y)
   PLAY_CGLM_INLINE mat3s translate2d_make(vec2s v)
   PLAY_CGLM_INLINE mat3s scale2d_make(vec2s v)
   PLAY_CGLM_INLINE mat3s scale2d(mat3s m, vec2s v)
   PLAY_CGLM_INLINE mat3s scale2d_uni(mat3s m, float s)
   PLAY_CGLM_INLINE mat3s rotate2d_make(float angle)
   PLAY_CGLM_INLINE mat3s rotate2d(mat3s m, float angle)
   PLAY_CGLM_INLINE mat3s rotate2d_to(mat3s m, float angle)
 */




/*!
 * @brief translate existing 2d transform matrix by v vector
 *        and stores result in same matrix
 *
 * @param[in] m  affine transform
 * @param[in] v  translate vector [x, y]
 * @returns      affine transform
*/
PLAY_CGLM_INLINE
mat3s
translate2d(mat3s m, vec2s v)
{
    translate2d(m.raw, v.raw);
    return m;
}

/*!
 * @brief translate existing 2d transform matrix by x factor
 *
 * @param[in] m  affine transform
 * @param[in] x  x factor
 * @returns      affine transform
 */
PLAY_CGLM_INLINE
mat3s
translate2d_x(mat3s m, float x)
{
    translate2d_x(m.raw, x);
    return m;
}

/*!
 * @brief translate existing 2d transform matrix by y factor
 *
 * @param[in] m  affine transform
 * @param[in] y  y factor
 * @returns      affine transform
 */
PLAY_CGLM_INLINE
mat3s
translate2d_y(mat3s m, float y)
{
    translate2d_y(m.raw, y);
    return m;
}

/*!
 * @brief creates NEW translate 2d transform matrix by v vector
 *
 * @param[in] v  translate vector [x, y]
 * @returns      affine transform
 */
PLAY_CGLM_INLINE
mat3s
translate2d_make(vec2s v)
{
    mat3s m;
    translate2d_make(m.raw, v.raw);
    return m;
}

/*!
 * @brief creates NEW 2d scale matrix by v vector
 *
 * @param[in]   v  scale vector [x, y]
 * @returns affine transform
 */
PLAY_CGLM_INLINE
mat3s
scale2d_make(vec2s v)
{
    mat3s m;
    scale2d_make(m.raw, v.raw);
    return m;
}

/*!
 * @brief scales existing 2d transform matrix by v vector
 *        and stores result in same matrix
 *
 * @param[in]  m  affine transform
 * @param[in]  v  scale vector [x, y, z]
 * @returns       affine transform
 */
PLAY_CGLM_INLINE
mat3s
scale2d(mat3s m, vec2s v)
{
    mat3s r;
    scale2d_to(m.raw, v.raw, r.raw);
    return r;
}

/*!
 * @brief applies uniform scale to existing 2d transform matrix v = [s, s, s]
 *        and stores result in same matrix
 *
 * @param[in] m  affine transform
 * @param[in] s  scale factor
 * @returns      affine transform
 */
PLAY_CGLM_INLINE
mat3s
scale2d_uni(mat3s m, float s)
{
    scale2d_uni(m.raw, s);
    return m;
}

/*!
 * @brief creates NEW 2d rotation matrix by angle and axis
 *
 * axis will be normalized so you don't need to normalize it
 *
 * @param[in]  angle  angle (radians)
 * @returns           affine transform
 */
PLAY_CGLM_INLINE
mat3s
rotate2d_make(float angle)
{
    mat3s m;
    rotate2d_make(m.raw, angle);
    return m;
}

/*!
 * @brief rotate existing 2d transform matrix around given axis by angle
 *
 * @param[in] m      affine transform
 * @param[in] angle  angle (radians)
 * @returns          affine transform
 */
PLAY_CGLM_INLINE
mat3s
rotate2d(mat3s m, float angle)
{
    rotate2d(m.raw, angle);
    return m;
}

/*!
 * @brief rotate existing 2d transform matrix around given axis by angle
 *
 * @param[in] m      affine transform
 * @param[in] angle  angle (radians)
 * @returns          affine transform
 */
PLAY_CGLM_INLINE
mat3s
rotate2d_to(mat3s m, float angle)
{
    rotate2d(m.raw, angle);
    return m;
}



/*** End of inlined file: affine2d.h ***/


/*** Start of inlined file: ray.h ***/



/* api definition */
#define ray_(NAME) PLAY_CGLM_STRUCTAPI(ray, NAME)

/*!
 * @brief Möller–Trumbore ray-triangle intersection algorithm
 *
 * @param[in] origin         origin of ray
 * @param[in] direction      direction of ray
 * @param[in] v0             first vertex of triangle
 * @param[in] v1             second vertex of triangle
 * @param[in] v2             third vertex of triangle
 * @param[in, out] d         distance to intersection
 * @return whether there is intersection
 */
PLAY_CGLM_INLINE
bool
ray_(triangle)(vec3s  origin,
               vec3s  direction,
               vec3s  v0,
               vec3s  v1,
               vec3s  v2,
               float *d)
{
    return ray_triangle(origin.raw, direction.raw, v0.raw, v1.raw, v2.raw, d);
}

/*!
 * @brief ray sphere intersection
 *
 * returns false if there is no intersection if true:
 *
 * - t1 > 0, t2 > 0: ray intersects the sphere at t1 and t2 both ahead of the origin
 * - t1 < 0, t2 > 0: ray starts inside the sphere, exits at t2
 * - t1 < 0, t2 < 0: no intersection ahead of the ray ( returns false )
 * - the caller can check if the intersection points (t1 and t2) fall within a
 *   specific range (for example, tmin < t1, t2 < tmax) to determine if the
 *   intersections are within a desired segment of the ray
 *
 * @param[in]  origin ray origin
 * @param[out] dir    normalized ray direction
 * @param[in]  s      sphere  [center.x, center.y, center.z, radii]
 * @param[in]  t1     near point1 (closer to origin)
 * @param[in]  t2     far point2 (farther from origin)
 *
 * @returns whether there is intersection
 */
PLAY_CGLM_INLINE
bool
ray_(sphere)(vec3s origin,
             vec3s dir,
             vec4s s,
             float * __restrict t1,
             float * __restrict t2)
{
    return ray_sphere(origin.raw, dir.raw, s.raw, t1, t2);
}

/*!
 * @brief point using t by 𝐏(𝑡)=𝐀+𝑡𝐛
 *
 * @param[in]  orig  origin of ray
 * @param[in]  dir   direction of ray
 * @param[in]  t     parameter
 * @returns point point at t
 */
PLAY_CGLM_INLINE
vec3s
ray_(at)(vec3s orig, vec3s dir, float t)
{
    vec3s r;
    ray_at(orig.raw, dir.raw, t, r.raw);
    return r;
}



/*** End of inlined file: ray.h ***/

#ifdef __cplusplus
}
#endif


/*** End of inlined file: struct.h ***/

