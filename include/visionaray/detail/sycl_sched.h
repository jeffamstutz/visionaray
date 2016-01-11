// This file is distributed under the MIT license.
// See the LICENSE file for details.

#pragma once

#ifndef VSNRAY_DETAIL_SYCL_SCHED_H
#define VSNRAY_DETAIL_SYCL_SCHED_H 1

#include <visionaray/math/math.h>

namespace visionaray
{

template <typename R>
class sycl_sched
{
public:

    sycl_sched() = default;
    sycl_sched(vec2ui block_size);

    template <typename K, typename SP>
    void frame(K kernel, SP sched_params, unsigned frame_num = 0);

private:

    vec2ui block_size_ = vec2ui(16, 16);
};

} // visionaray

#include "sycl_sched.inl"

#endif // VSNRAY_DETAIL_SYCL_SCHED_H
