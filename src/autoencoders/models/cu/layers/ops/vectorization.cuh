// define some basic *, /, +, - operations for float2 and other

#ifndef VECTORIZATION_CUH_INCLUDED
#define VECTORIZATION_CUH_INCLUDED

// elementwise apply (works for x,y[ ,z[ ,w]])
template<class V, class A, class F>
__device__ __forceinline__ V vop(const V& a, const A& b, F f) {
    V r; r.x=f(a.x,b); r.y=f(a.y,b);
    if constexpr (requires{a.z;}) r.z=f(a.z,b);
    if constexpr (requires{a.w;}) r.w=f(a.w,b);
    return r;
}

template<class V, class A, class F>
__device__ __forceinline__ V rvop(const A& a, const V& b, F f) {
    V r; r.x=f(a,b.x); r.y=f(a,b.y);
    if constexpr (requires{a.z;}) r.z=f(a,b.z);
    if constexpr (requires{a.w;}) r.w=f(a,b.w);
    return r;
}

template<class V>
__device__ __forceinline__ typename V::value_type sum(const V& a) {
    typename V::value_type r = a.x + a.y;
    if constexpr (requires{a.z;}) r += a.z;
    if constexpr (requires{a.w;}) r += a.w;
    return r;
}

// op functors
struct PlusEq{__device__ __forceinline__ void operator()(auto &A,auto B)const{A+=B;}};
struct Add{__device__ __forceinline__ auto operator()(auto A,auto B)const{return A+B;}};
struct Sub{__device__ __forceinline__ auto operator()(auto A,auto B)const{return A-B;}};
struct Mul{__device__ __forceinline__ auto operator()(auto A,auto B)const{return A*B;}};
struct Div{__device__ __forceinline__ auto operator()(auto A,auto B)const{return A/B;}};

// vector ∘ scalar    // vector ∘ vector
#define VEC_OP(OP,FN) \
template<class V> __device__ __forceinline__ V operator OP(const V&a,const V&b){return vop(a,b,FN{});} \
template<class V> __device__ __forceinline__ V operator OP(const V&a,const typename V::value_type&s){return vop(a,s,FN{});} \
template<class V> __device__ __forceinline__ V operator OP(const typename V::value_type&s,const V&a){return rvop(s,a,FN{});}

VEC_OP(*,Mul)
VEC_OP(+,Add)
// VEC_OP(+=,PlusEq) // too dangerous, likely
// VEC_OP(-,Sub)
VEC_OP(/,Div)

#undef VEC_OP


#endif // VECTORIZATION_CUH_INCLUDED