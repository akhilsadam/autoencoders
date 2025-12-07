#ifndef OPS_BASIC_CUH_INCLUDED
    #define OPS_BASIC_CUH_INCLUDED

    struct scale {
        template<class T>
        __device__ __forceinline__ static T op(T& out, const T& in, const ftype& weight) 
        { return in * weight; }
    };

    template<>
    __device__ __forceinline__ static float scale::op(float& out, const float& in, const ftype& weight)
    { return in * weight; }

    template<>
    __device__ __forceinline__ static float2 scale::op(float2& out, const float2& in, const ftype& weight)
    { return float2{in.x * weight, in.y * weight}; }

#endif // OPS_BASIC_CUH_INCLUDED