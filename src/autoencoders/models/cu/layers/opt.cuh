
#ifndef LEARNING_RATE
#define LEARNING_RATE 0.001f
#endif

class SGD {
    public:
        static constexpr float learning_rate = LEARNING_RATE;
        
        template<typename T>
        static __device__ __forceinline__ void update(T& w, T& grad_w) {
            w = w - learning_rate * grad_w;
            grad_w = T(0);
        }
};