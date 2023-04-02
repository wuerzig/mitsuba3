#include <mitsuba/core/fwd.h>
#include <mitsuba/core/math.h>
#include <mitsuba/core/properties.h>
#include <mitsuba/core/string.h>
#include <mitsuba/core/transform.h>
#include <mitsuba/core/util.h>
#include <mitsuba/core/warp.h>
#include <mitsuba/render/fwd.h>
#include <mitsuba/render/interaction.h>
#include <mitsuba/render/sdf.h>
#include <drjit/tensor.h>
#include <drjit/texture.h>

#if defined(MI_ENABLE_CUDA)
    #include "optix/sdfgrid.cuh"
#endif

NAMESPACE_BEGIN(mitsuba)

/**!

.. _shape-sdfgrid:

SDF Grid (:monosp:`sdfgrid`)
-------------------------------------------------

Props:
 * "normals": type of normal computation method (analytic, smoth, falcao)
 * "watertight": if the SDF should be watertight (default: true)

Documentation notes:
 * Grid position [0, 0, 0] x [1, 1, 1]
 * Reminder that tensors use [Z, Y, X, C] indexing
 * Does not emit UVs for texturing
 * Cannot be used for area emitters
 * Grid data must be initialized by using `mi.traverse()` (by default the plugin
   is initialized with a 2x2x2 grid of minus ones)

 Temorary issues:
     * Embree does not work

//TODO: Test that instancing works
*/

template <typename Float, typename Spectrum>
class SDFAna final : public SDF<Float, Spectrum> {
public:
    MI_IMPORT_BASE(SDF, m_to_world, m_to_object, m_is_instance, initialize,
                   mark_dirty, get_children_string, parameters_grad_enabled)
    MI_IMPORT_TYPES()

    using typename Base::ScalarSize;

    SDFAna(const Properties &props) : Base(props) {
        update();
        initialize();
    }

    ~SDFAna() {
    }

    void update() {
        auto [S, Q, T] = dr::transform_decompose(m_to_world.scalar().matrix, 25);
        if (dr::abs(Q[0]) > 1e-6f || dr::abs(Q[1]) > 1e-6f ||
            dr::abs(Q[2]) > 1e-6f || dr::abs(Q[3] - 1) > 1e-6f)
            Log(Warn, "'to_world' transform shouldn't perform any rotations, "
                      "use instancing (`shapegroup` and `instance` plugins) "
                      "instead!");

        m_to_object = m_to_world.value().inverse();
        mark_dirty();
   }

    void traverse(TraversalCallback *callback) override {
        Base::traverse(callback);
        callback->put_parameter("to_world", *m_to_world.ptr(), ParamFlags::Differentiable | ParamFlags::Discontinuous);
    }

    void parameters_changed(const std::vector<std::string> &keys) override {
        if (keys.empty() || string::contains(keys, "to_world")  || string::contains(keys, "grid")) {
            // Ensure previous ray-tracing operation are fully evaluated before
            // modifying the scalar values of the fields in this class
            if constexpr (dr::is_jit_v<Float>)
                dr::sync_thread();

            // Update the scalar value of the matrix
            m_to_world = m_to_world.value();

            //m_grid_texture.set_tensor(m_grid_texture.tensor());
            update();
        }

        Base::parameters_changed();
    }

    ScalarSize primitive_count() const override {
        return ScalarSize(1);//m_filled_voxel_count;
    }

    ScalarBoundingBox3f bbox() const override {
        ScalarBoundingBox3f bbox;
        ScalarTransform4f to_world = m_to_world.scalar();

        // bbox from -.5 ... .5
        /*bbox.expand(to_world.transform_affine(ScalarPoint3f(-0.5f, -0.5f, -0.5f)));
        bbox.expand(to_world.transform_affine(ScalarPoint3f( 0.5f, -0.5f, -0.5f)));
        bbox.expand(to_world.transform_affine(ScalarPoint3f(-0.5f,  0.5f, -0.5f)));
        bbox.expand(to_world.transform_affine(ScalarPoint3f( 0.5f,  0.5f, -0.5f)));
        bbox.expand(to_world.transform_affine(ScalarPoint3f(-0.5f, -0.5f,  0.5f)));
        bbox.expand(to_world.transform_affine(ScalarPoint3f( 0.5f, -0.5f,  0.5f)));
        bbox.expand(to_world.transform_affine(ScalarPoint3f(-0.5f,  0.5f,  0.5f)));
        bbox.expand(to_world.transform_affine(ScalarPoint3f( 0.5f,  0.5f,  0.5f)));*/

        bbox.expand(to_world.transform_affine(ScalarPoint3f(-1.5f, -1.5f, -1.5f)));
        bbox.expand(to_world.transform_affine(ScalarPoint3f( 1.5f, -1.5f, -1.5f)));
        bbox.expand(to_world.transform_affine(ScalarPoint3f(-1.5f,  1.5f, -1.5f)));
        bbox.expand(to_world.transform_affine(ScalarPoint3f( 1.5f,  1.5f, -1.5f)));
        bbox.expand(to_world.transform_affine(ScalarPoint3f(-1.5f, -1.5f,  1.5f)));
        bbox.expand(to_world.transform_affine(ScalarPoint3f( 1.5f, -1.5f,  1.5f)));
        bbox.expand(to_world.transform_affine(ScalarPoint3f(-1.5f,  1.5f,  1.5f)));
        bbox.expand(to_world.transform_affine(ScalarPoint3f( 1.5f,  1.5f,  1.5f)));

       /*bbox.expand(to_world.transform_affine(ScalarPoint3f(-3.5f, -3.5f, -3.5f)));
        bbox.expand(to_world.transform_affine(ScalarPoint3f( 3.5f, -3.5f, -3.5f)));
        bbox.expand(to_world.transform_affine(ScalarPoint3f(-3.5f,  3.5f, -3.5f)));
        bbox.expand(to_world.transform_affine(ScalarPoint3f( 3.5f,  3.5f, -3.5f)));
        bbox.expand(to_world.transform_affine(ScalarPoint3f(-3.5f, -3.5f,  3.5f)));
        bbox.expand(to_world.transform_affine(ScalarPoint3f( 3.5f, -3.5f,  3.5f)));
        bbox.expand(to_world.transform_affine(ScalarPoint3f(-3.5f,  3.5f,  3.5f)));
        bbox.expand(to_world.transform_affine(ScalarPoint3f( 3.5f,  3.5f,  3.5f)));*/

        return bbox;
    }

    Float surface_area() const override {
        // TODO: area emitter
        return 0;
    }

    // =============================================================
    //! @{ \name Sampling routines
    // =============================================================

    PositionSample3f sample_position(Float time, const Point2f &sample,
                                     Mask active) const override {
        // TODO: area emitter
        MI_MASK_ARGUMENT(active);
        (void) time;
        (void) sample;
        PositionSample3f ps = dr::zeros<PositionSample3f>();
        return ps;
    }

    Float pdf_position(const PositionSample3f & /*ps*/, Mask active) const override {
        // TODO: area emitter
        MI_MASK_ARGUMENT(active);
        return 0;
    }


    SurfaceInteraction3f eval_parameterization(const Point2f &uv,
                                               uint32_t ray_flags,
                                               Mask active) const override {
        // TODO: area emitter
        MI_MASK_ARGUMENT(active);
        (void) uv;
        (void) ray_flags;

        Log(Warn, "EvalParameterization called");
        SurfaceInteraction3f si = dr::zeros<SurfaceInteraction3f>();
        return si;
    }

    //! @}
    // =============================================================

    // =============================================================
    //! @{ \name Ray tracing routines
    // =============================================================

    template <typename FloatP, typename Ray3fP>
    std::tuple<FloatP, Point<FloatP, 2>, dr::uint32_array_t<FloatP>,
               dr::uint32_array_t<FloatP>>
    ray_intersect_preliminary_impl(const Ray3fP &ray_,
                                   dr::mask_t<FloatP> active) const {
        MI_MASK_ARGUMENT(active);

        using Value = std::conditional_t<dr::is_cuda_v<FloatP> || dr::is_diff_v<Float>,
                                         dr::float32_array_t<FloatP>,
                                         dr::float64_array_t<FloatP>>;
        using ScalarValue = dr::scalar_t<Value>;

        Value radius(1.0); // Constant kept for readability
        Value length(1.0);

        Transform<Point<FloatP, 4>> to_object;
        if constexpr (!dr::is_jit_v<FloatP>)
            to_object = m_to_object.scalar();
        else
            to_object = m_to_object.value();

        Ray3fP ray_xformed = to_object.transform_affine(ray_);

        //Log(Warn, "Orig.    ray o %f %f %f d %f %f %f", ray.o[0], ray.o[1], ray.o[2], ray.d[0], ray.d[1], ray.d[2]);
        //Log(Warn, "xformed  ray o %f %f %f d %f %f %f", ray_xformed.o[0], ray_xformed.o[1], ray_xformed.o[2], ray_xformed.d[0], ray_xformed.d[1], ray_xformed.d[2]);
        //Point<FloatP, 3> local = ray(t);

        Value maxt = Value(ray_xformed.maxt);

        FloatP t = 0;
        UInt32 step = 0;

        dr::Loop<Mask> loop("SDFana sphere tracing loop", 
                                            active, t, step);
        
        // TODO: Also exit loop if no more active lanes present
        //while (loop(dr::neq(step, 32))) {
        
        
        
        ////////
        // Seperate active and hit, for active can be false before we sphere trace
        ////////
        
        while (loop(dr::neq(step, 32))) {
            Point<FloatP, 3> local = ray_xformed(t);
            //Log(Warn, "Sphere tracing loop (ray test) local_xformed =%f %f %f", local[0], local[1], local[2]);
            FloatP dist = (dr::norm(local) - 1.f)+dr::sin(5.f * local[0])*dr::sin(3.f * local[1])*dr::sin(7.f * local[3])*.1f;

            auto hit_next = (dr::abs(dist) > 1e-3);
            active &= hit_next;

            //Log(Warn, "Sphere tracing loop (ray test) %d dist %f t %f local %f %f %f active %d", step, dist, t, local[0], local[1], local[2], active);
            step += 1;
            
            
            // Scale t travel by norm of local ray.d
            // travel further only if no intersection found yet (hit_next)
            //t += dr::select(active, dr::rcp(dr::norm(ray_xformed.d))*dist * 0.975, 0.f);
            t += dr::rcp(dr::norm(ray_xformed.d))*dist;
        }
        active = active || (t > maxt);
        // active here means no intersection found or out of bounds

        //Log(Warn, "SDFAna::rayintersect_preliminary_impl");
        return { dr::select(active, dr::Infinity<FloatP>, t),
                 Point<FloatP, 2>(0, 0), ((uint32_t) -1), 0 };
    }

    template <typename FloatP, typename Ray3fP>
    dr::mask_t<FloatP> ray_test_impl(const Ray3fP &ray_,
                                     dr::mask_t<FloatP> active) const {
        // TODO: Figure out how to do this without code duplication


        MI_MASK_ARGUMENT(active);

        using Value = std::conditional_t<dr::is_cuda_v<FloatP> || dr::is_diff_v<Float>,
                                         dr::float32_array_t<FloatP>,
                                         dr::float64_array_t<FloatP>>;
        using ScalarValue = dr::scalar_t<Value>;

        Value radius(1.0); // Constant kept for readability
        Value length(1.0);

        Transform<Point<FloatP, 4>> to_object;
        if constexpr (!dr::is_jit_v<FloatP>)
            to_object = m_to_object.scalar();
        else
            to_object = m_to_object.value();

        Ray3fP ray_xformed = to_object.transform_affine(ray_);

        //Log(Warn, "Orig.    ray o %f %f %f d %f %f %f", ray.o[0], ray.o[1], ray.o[2], ray.d[0], ray.d[1], ray.d[2]);
        //Log(Warn, "xformed  ray o %f %f %f d %f %f %f", ray_xformed.o[0], ray_xformed.o[1], ray_xformed.o[2], ray_xformed.d[0], ray_xformed.d[1], ray_xformed.d[2]);
        //Point<FloatP, 3> local = ray(t);

        Value maxt = Value(ray_xformed.maxt);

        FloatP t = 0;
        UInt32 step = 0;

        dr::Loop<Mask> loop("SDFana sphere tracing loop", 
                                            active, t, step);
        
        // TODO: Also exit loop if no more active lanes present
        //while (loop(dr::neq(step, 32))) {
        while (loop(dr::neq(step, 32))) {
            Point<FloatP, 3> local = ray_xformed(t);
            //Log(Warn, "Sphere tracing loop (ray test) local_xformed =%f %f %f", local[0], local[1], local[2]);
            FloatP dist = (dr::norm(local) - 1.f)+dr::sin(5.f * local[0])*dr::sin(3.f * local[1])*dr::sin(7.f * local[3])*.1f;

            auto hit_next = (dr::abs(dist) > 1e-3);
            active &= hit_next;

            //Log(Warn, "Sphere tracing loop (ray test) %d dist %f t %f local %f %f %f active %d", step, dist, t, local[0], local[1], local[2], active);
            step += 1;
            
            
            // Scale t travel by norm of local ray.d
            // travel further only if no intersection found yet (hit_next)
            t += dr::rcp(dr::norm(ray_xformed.d))*dist;
        }
        active = active || (t > maxt);
        // active here means no intersection found or out of bounds
        
        //Log(Warn, "SDFAna::rayintersect_preliminary_impl");
        return !active;
    }

    MI_SHAPE_DEFINE_RAY_INTERSECT_METHODS()

    SurfaceInteraction3f compute_surface_interaction(const Ray3f &ray,
                                                     const PreliminaryIntersection3f &pi,
                                                     uint32_t ray_flags,
                                                     uint32_t recursion_depth,
                                                     Mask active) const override {
        MI_MASK_ARGUMENT(active);
        constexpr bool IsDiff = dr::is_diff_v<Float>;

        // Early exit when tracing isn't necessary
        if (!m_is_instance && recursion_depth > 0)
            return dr::zeros<SurfaceInteraction3f>();

        SurfaceInteraction3f si = dr::zeros<SurfaceInteraction3f>();

        bool detach_shape = has_flag(ray_flags, RayFlags::DetachShape);
        bool follow_shape = has_flag(ray_flags, RayFlags::FollowShape);

        Transform4f to_world = m_to_world.value();
        Transform4f to_object = m_to_object.value();

        // TODO: Make sure this is the proper way to detach dr::Texture objects
        //dr::suspend_grad<Float> scope(detach_shape, to_world, to_object, m_grid_texture.tensor().array());

        if constexpr (IsDiff) {
            Log(Warn, "Differentiable implicit SDFs not possible yet!");
        } else {
            si.t = pi.t;
            si.p = ray(si.t);
        }

        si.t = dr::select(active, si.t, dr::Infinity<Float>);

        //Vector3f grad = sdf_grad(m_to_object.value().transform_affine(si.p));
        si.n = sdf_normal(m_to_object.value().transform_affine(si.p));

        if (likely(has_flag(ray_flags, RayFlags::ShadingFrame))) {
            si.sh_frame.n = falcao(m_to_object.value().transform_affine(si.p));
        }

        si.uv = Point2f(0.f, 0.f);
        si.dp_du = Vector3f(0.f);
        si.dp_dv = Vector3f(0.f);
        si.dn_du = si.dn_dv = dr::zeros<Vector3f>();

        si.shape    = this;
        si.instance = nullptr;

        if (unlikely(has_flag(ray_flags, RayFlags::BoundaryTest))) {
            Float dp = dr::dot(si.sh_frame.n, -ray.d);
            // Add non-linearity by squaring the returned value
            si.boundary_test = dr::sqr(dp);
        }

        return si;
    }

    bool parameters_grad_enabled() const override {
        Log(Warn, "sdfana parameters_grad_enabled: Only checking m_to_world so far");
        return dr::grad_enabled(m_to_world);
    }


    std::string to_string() const override {
        std::ostringstream oss;
        oss << "SDFana[" << std::endl
            << "  to_world = " << string::indent(m_to_world, 13) << "," << std::endl
            << "  " << string::indent(get_children_string()) << std::endl
            << "]";
        return oss.str();
    }

    MI_DECLARE_CLASS()
private:
    
    Float eval_sdf(const Point3f& point) const {
        return (dr::norm(point) - 1.f)+dr::sin(5.f * point[0])*dr::sin(3.f * point[1])*dr::sin(7.f * point[3])*.1f;

        //return dr::norm(point) - 1.f;
    }
    

    Normal3f sdf_normal(const Point3f &point) const {
            
            #define __EPS__ 0.00001f

            Normal3f res(
                eval_sdf(point + Vector3f(__EPS__, 0.f, 0.f)) - eval_sdf(point - Vector3f(__EPS__, 0.f, 0.f)),
                eval_sdf(point + Vector3f(0.f, __EPS__, 0.f)) - eval_sdf(point - Vector3f(0.f, __EPS__, 0.f)),
                eval_sdf(point + Vector3f(0.f, 0.f, __EPS__)) - eval_sdf(point - Vector3f(0.f, 0.f, __EPS__))
            );

            return dr::normalize(res);
            
            

    }
    
    /// Very efficient normals (faceted appearance)
    Normal3f falcao(const Point3f& point) const {
        // FALCÃO , P., 2008. Implicit function to distance function.
        // URL: https://www.pouet.net/topic.php?which=5604&page=3#c233266.

        // FIXME: Something is numerically unstable ?!

        // Scale epsilon w.r.t inverse resolution
        //auto shape = m_grid_texture.tensor().shape();
        Vector3f epsilon =
            0.001f * Vector3f(1.f, 1.f, 1.f);

        auto v = [&](const Point3f& p){
            Float out = eval_sdf(p);
            return out;
        };

        Point3f p1(point.x() + epsilon.x(), point.y() - epsilon.y(), point.z() - epsilon.z());
        Point3f p2(point.x() - epsilon.x(), point.y() - epsilon.y(), point.z() + epsilon.z());
        Point3f p3(point.x() - epsilon.x(), point.y() + epsilon.y(), point.z() - epsilon.z());
        Point3f p4(point.x() + epsilon.x(), point.y() + epsilon.y(), point.z() + epsilon.z());

        Float v1 = v(p1);
        Float v2 = v(p2);
        Float v3 = v(p3);
        Float v4 = v(p4);

        Normal3f out = Normal3f(((v4 + v1) / 2.f) - ((v3 + v2) / 2.f),
                                ((v3 + v4) / 2.f) - ((v1 + v2) / 2.f),
                                ((v2 + v4) / 2.f) - ((v3 + v1) / 2.f));

        return dr::normalize(m_to_world.value().transform_affine(out));
    }

    static constexpr uint32_t optix_geometry_flags[1] = {
        OPTIX_GEOMETRY_FLAG_NONE
    };

    // TODO: Store inverse shape using `rcp`
};

MI_IMPLEMENT_CLASS_VARIANT(SDFAna, SDF)
MI_EXPORT_PLUGIN(SDFAna, "SDFAnalytical intersection primitive");
NAMESPACE_END(mitsuba)
