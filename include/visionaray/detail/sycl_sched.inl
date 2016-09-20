// This file is distributed under the MIT license.
// See the LICENSE file for details.

#include <CL/sycl.hpp>

#include "sched_common.h"
#include "sycl_sched.h"

namespace visionaray
{

//-------------------------------------------------------------------------------------------------
// sycl_sched implementation
//

template <typename R>
sycl_sched<R>::sycl_sched(vec2ui block_size)
    : block_size_(block_size)
{
}

template <typename R>
template <typename K, typename SP>
void sycl_sched<R>::frame(K kernel, SP sched_params, unsigned frame_num)
{
    using namespace cl::sycl;

    sched_params.rt.begin_frame();

    auto& sparams = sched_params;

    auto rt_ref             = sparams.rt.ref();
    auto viewport           = sparams.cam.get_viewport();

    //  front, side, and up vectors form an orthonormal basis
    auto f = normalize( sparams.cam.eye() - sparams.cam.center() );
    auto s = normalize( cross(sparams.cam.up(), f) );
    auto u =            cross(f, s);

    auto eye   = sparams.cam.eye();
    auto cam_u = s * tan(sparams.cam.fovy() / 2.0f) * sparams.cam.aspect();
    auto cam_v = u * tan(sparams.cam.fovy() / 2.0f);
    auto cam_w = -f;

    queue q;

    q.submit([&](handler& cmdgroup)
    {
        cmdgroup.parallel_for<class SYCLKernel>(
            range<2>(sched_params.rt.width(), sched_params.rt.height()),
            [=](id<2> id)
            {
                auto x = id[0];
                auto y = id[1];

                if (x >= sched_params.rt.width() || y >= sched_params.rt.height())
                {
                    return;
                }

                // TODO: support any sampler
                sampler<typename R::scalar_type> samp(detail::tic());

                auto r = detail::make_primary_rays(
                        R{},
                        typename SP::pixel_sampler_type{},
                        samp,
                        x,
                        y,
                        sched_params.rt.width(),
                        sched_params.rt.height(),
                        eye,
                        cam_u,
                        cam_v,
                        cam_w
                        );

                sample_pixel(
                        kernel,
                        typename SP::pixel_sampler_type{},
                        r,
                        samp,
                        frame_num,
                        rt_ref,
                        x,
                        y,
                        sched_params.rt.width(),
                        sched_params.rt.height(),
                        eye,
                        cam_u,
                        cam_v,
                        cam_w
                        );
            }
            );
    }
    );

    sched_params.rt.end_frame();
}

} // visionaray
