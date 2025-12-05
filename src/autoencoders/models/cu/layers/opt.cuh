struct SGD {
    float learning_rate;
    
    template<typename T>
    __device__ __forceinline__ T update(T w, T grad_w) {
        return w - learning_rate * grad_w;
    }
};