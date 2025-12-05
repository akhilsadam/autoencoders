
#ifndef LEARNING_RATE
#define LEARNING_RATE 0.001f
#endif

class SGD {
    static const float learning_rate = LEARNING_RATE;
    
    template<typename T>
    static __device__ __forceinline__ T update(T w, T grad_w) {
        return w - learning_rate * grad_w;
    }
};