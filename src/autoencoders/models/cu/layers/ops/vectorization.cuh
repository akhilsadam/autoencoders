// define some basic *, /, +, - operations for float2 and other

#ifndef VECTORIZATION_CUH_INCLUDED
#define VECTORIZATION_CUH_INCLUDED

#define BINOP(op) [] __device__ __forceinline__ (auto a, auto b){ return a op b; }

template <class V, class A, class F>
__device__ __forceinline__
V vec_apply(const V& v, const A& x, F f) {
    V r;
    r.x = f(v.x, x);
    r.y = f(v.y, x);
    if constexpr (requires { v.z; }) r.z = f(v.z, x);
    if constexpr (requires { v.w; }) r.w = f(v.w, x);
    return r;
}

// vector-scalar operations

template <class V>
__device__ __forceinline__
V operator*(const V& v, const typename V::value_type& s) { return vec_apply(v, s, BINOP(*)); }

template <class V>
__device__ __forceinline__
V operator+(const V& v, const typename V::value_type& s) { return vec_apply(v, s, BINOP(+)); }

template <class V>
__device__ __forceinline__
V operator-(const V& v, const typename V::value_type& s) { return vec_apply(v, s, BINOP(-)); }

template <class V>
__device__ __forceinline__
V operator/(const V& v, const typename V::value_type& s) { return vec_apply(v, s, BINOP(/)); }

// scalar-vector (comm. ops)

template <class V>
__device__ __forceinline__
V operator*(const typename V::value_type& s, const V& v) { return v * s; }

template <class V>
__device__ __forceinline__
V operator+(const typename V::value_type& s, const V& v) { return v + s; }

// vector-vector operations

template <class V>
__device__ __forceinline__
V operator*(const V& a, const V& b) { return vec_apply(a, b, BINOP(*)); }

template <class V>
__device__ __forceinline__
V operator+(const V& a, const V& b) { return vec_apply(a, b, BINOP(+)); }

template <class V>
__device__ __forceinline__
V operator-(const V& a, const V& b) { return vec_apply(a, b, BINOP(-)); }

template <class V>
__device__ __forceinline__
V operator/(const V& a, const V& b) { return vec_apply(a, b, BINOP(/)); }

#endif // VECTORIZATION_CUH_INCLUDED