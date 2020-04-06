#include <cupy/complex.cuh>
#include <math_constants.h>

extern "C" {

__global__
void slice_gen(const double *model, const double angle, const double scale, const long long size,
               const double *bg, const long long log_flag, double *view) {
    int x = blockIdx.x * blockDim.x + threadIdx.x ;
    int y = blockIdx.y * blockDim.y + threadIdx.y ;
    if (x > size - 1 || y > size - 1)
        return ;
    int t = x*size + y ;
    if (log_flag)
        view[t] = -1000. ;
    else
        view[t] = 0. ;

    int cen = size / 2 ;
    double ac = cos(angle), as = sin(angle) ;
    double tx = (x - cen) * ac - (y - cen) * as + cen ;
    double ty = (x - cen) * as + (y - cen) * ac + cen ;
    int ix = __double2int_rd(tx), iy = __double2int_rd(ty) ;
    if (ix < 0 || ix > size - 2 || iy < 0 || iy > size - 2)
        return ;

    double fx = tx - ix, fy = ty - iy ;
    double cx = 1. - fx, cy = 1. - fy ;

    view[t] = model[ix*size + iy]*cx*cy +
              model[(ix+1)*size + iy]*fx*cy +
              model[ix*size + (iy+1)]*cx*fy +
              model[(ix+1)*size + (iy+1)]*fx*fy ;
    view[t] *= scale ;
    view[t] += bg[t] ;
    if (log_flag) {
        if (view[t] < 1.e-20)
            view[t] = -1000. ;
        else
            view[t] = log(view[t]) ;
    }
}

__global__
void slice_gen_holo(const complex<double> *model, const double shiftx, const double shifty,
               const double diameter, const double rel_scale, const double scale, const long long size,
               const double *bg, const long long log_flag, double *view) {
    int x = blockIdx.x * blockDim.x + threadIdx.x ;
    int y = blockIdx.y * blockDim.y + threadIdx.y ;
    if (x > size - 1 || y > size - 1)
        return ;
    int t = x*size + y ;
    if (log_flag)
        view[t] = -1000. ;
    else
        view[t] = 0. ;

    double cen = floor(size / 2.) ;
    complex<double> ramp, sphere ;

    double phase = 2. * CUDART_PI * ((x-cen) * shiftx + (y-cen) * shifty) / size ;
    double ramp_r = cos(phase) ;
    double ramp_i = sin(phase) ;
    ramp = complex<double>(ramp_r, ramp_i) ;

    double s = sqrt((x-cen)*(x-cen) + (y-cen)*(y-cen)) * CUDART_PI * diameter / size ;
    if (s == 0.)
        s = 1.e-5 ;
    sphere = complex<double>(rel_scale*(sin(s) - s*cos(s)) / (s*s*s), 0) ;

    complex<double> cview = ramp * sphere + model[t] ;
    view[t] = pow(abs(cview), 2.) ;

    view[t] *= scale ;
    view[t] += bg[t] ;
    if (log_flag) {
        if (view[t] < 1.e-20)
            view[t] = -1000. ;
        else
            view[t] = log(view[t]) ;
    }
}

__global__
void slice_merge(const double *view, const double angle, const long long size,
                 double *model, double *mweights) {
    int x = blockIdx.x * blockDim.x + threadIdx.x ;
    int y = blockIdx.y * blockDim.y + threadIdx.y ;
    if (x > size - 1 || y > size - 1)
        return ;
    int t = x*size + y ;

    int cen = size / 2 ;
    double ac = cos(angle), as = sin(angle) ;
    double tx = (x - cen) * ac - (y - cen) * as + cen ;
    double ty = (x - cen) * as + (y - cen) * ac + cen ;
    int ix = __double2int_rd(tx), iy = __double2int_rd(ty) ;
    if (ix < 0 || ix > size - 2 || iy < 0 || iy > size - 2)
        return ;
    double fx = tx - ix, fy = ty - iy ;
    double cx = 1. - fx, cy = 1. - fy ;

    atomicAdd(&model[ix*size + iy], view[t]*cx*cy) ;
    atomicAdd(&mweights[ix*size + iy], cx*cy) ;

    atomicAdd(&model[(ix+1)*size + iy], view[t]*fx*cy) ;
    atomicAdd(&mweights[(ix+1)*size + iy], fx*cy) ;

    atomicAdd(&model[ix*size + (iy+1)], view[t]*cx*fy) ;
    atomicAdd(&mweights[ix*size + (iy+1)], cx*fy) ;

    atomicAdd(&model[(ix+1)*size + (iy+1)], view[t]*fx*fy) ;
    atomicAdd(&mweights[(ix+1)*size + (iy+1)], fx*fy) ;
}

__global__
void calc_prob_all(const double *lview, const int *mask, const long long ndata, const int *ones,
                   const int *multi, const long long *o_acc, const long long *m_acc, const int *p_o,
                   const int *p_m, const int *c_m, const double init, const double *scales, double *prob_r) {
    long long d, t ;
    int pixel ;
    d = blockDim.x * blockIdx.x + threadIdx.x ;
    if (d >= ndata)
        return ;

    prob_r[d] = init * scales[d] ;
    for (t = o_acc[d] ; t < o_acc[d] + ones[d] ; ++t) {
        pixel = p_o[t] ;
        if (mask[pixel] < 1)
            prob_r[d] += lview[pixel] ;
    }
    for (t = m_acc[d] ; t < m_acc[d] + multi[d] ; ++t) {
        pixel = p_m[t] ;
        if (mask[pixel] < 1)
            prob_r[d] += lview[pixel] * c_m[t] ;
    }
}

__global__
void merge_all(const double *prob_r, const long long ndata, const int *ones, const int *multi,
               const long long *o_acc, const long long *m_acc, const int *p_o, const int *p_m,
               const int *c_m, double *view) {
    long long d, t ;
    d = blockDim.x * blockIdx.x + threadIdx.x ;
    if (d >= ndata)
        return ;

    for (t = o_acc[d] ; t < o_acc[d] + ones[d] ; ++t)
        atomicAdd(&view[p_o[t]], prob_r[d]) ;
    for (t = m_acc[d] ; t < m_acc[d] + multi[d] ; ++t)
        atomicAdd(&view[p_m[t]], prob_r[d] * c_m[t]) ;
}

__global__
void proj_divide(const complex<double> *iter_in, const double *fobs, const complex<double> *sphere_ramp,
                 const int *invmask, const long long size, complex<double> *iter_out) {
    int t = blockIdx.x * blockDim.x + threadIdx.x ;
    if (t >= size*size)
        return ;

    complex<double> shifted ;

    if (invmask[t] == 0) {
        shifted = iter_in[t] + sphere_ramp[t] ;
        iter_out[t] = shifted * fobs[t] / abs(shifted) - sphere_ramp[t] ;
    }
    else {
        iter_out[t] = iter_in[t] ;
    }
}

} // extern C
