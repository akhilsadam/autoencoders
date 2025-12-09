
#ifndef LEARNING_RATE
#define LEARNING_RATE 0.01f
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

template<>
__device__ __forceinline__ void SGD::update<bf16_2>(bf16_2& w, bf16_2& grad_w) {
    w.x = w.x - __float2bfloat16_rn(learning_rate) * grad_w.x;
    w.y = w.y - __float2bfloat16_rn(learning_rate) * grad_w.y;

    grad_w.x = __float2bfloat16_rn(0.0f);
    grad_w.y = __float2bfloat16_rn(0.0f);
}