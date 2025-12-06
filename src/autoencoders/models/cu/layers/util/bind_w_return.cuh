#pragma once

#include "pyutils/pyutils.cuh"   // we need from_object, trait, etc.
#include <type_traits>   // for std::is_void_v

namespace kittens {
namespace py {

template<auto function, typename TGlobal>
inline void bind_function_with_return(auto m, auto name, auto TGlobal::*... member_ptrs) {
    using Ret = decltype(function(std::declval<TGlobal&>()));

    m.def(name,
        [=](object<decltype(member_ptrs)>... args) -> Ret {
            // Construct the struct from Python arguments
            TGlobal __g__ {
                from_object<typename trait<decltype(member_ptrs)>::member_type>::make(args)...
            };

            // Call the function and return the result if non-void
            if constexpr (std::is_void_v<Ret>) {
                function(__g__);
            } else {
                return function(__g__);
            }
        }
    );
}


} // namespace py
} // namespace kittens
