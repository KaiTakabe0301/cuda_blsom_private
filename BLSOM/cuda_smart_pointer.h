#pragma once
#ifndef CUDA_SMART_POINTER_H
#define CUDA_SMART_POINTER_H
#endif

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <memory>
#include <iostream>

template<typename _Ty>
void cuda_deleter(_Ty* Ptr) {
	
	if (cudaFree(Ptr) == cudaSuccess)
		std::cout<<"ŠJ•ú‚µ‚Ü‚µ‚½( ^)o(^ )"<<std::endl;
	else
		std::cout << "Ž¸”s‚µ‚Ü‚µ‚½" << std::endl;
}

template <typename _Ty>
struct cuda_delete
{
	constexpr cuda_delete() noexcept = default;
	template<class _Ty2,
		class = typename std::enable_if<std::is_convertible<_Ty2 *, _Ty *>::value,
		void>::type>
		cuda_delete(const cuda_delete<_Ty2>&) noexcept
	{	// construct from another default_delete
	}

	void operator()(_Ty *_Ptr) const noexcept
	{	// delete a pointer
		static_assert(0 < sizeof(_Ty), "can't delete an incomplete type");
		cudaFree(_Ptr);
	}

};

template<typename _Ty>
using cuda_unique_ptr = std::unique_ptr<_Ty, cuda_delete<_Ty>>;

template<typename _Ty>
inline std::shared_ptr<_Ty> cuda_shared_ptr(int size) {
	std::shared_ptr<_Ty> Ptr(new _Ty[size], cuda_deleter<_Ty>);
	return Ptr;
};