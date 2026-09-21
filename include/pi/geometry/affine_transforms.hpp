#pragma once
// types and concepts
#include <concepts>
#include <cstdint>

// math algorithms
#include <numeric>
#include <valarray>

// containers
#include <array>
#include <list>

#include "linear_algebra.hpp"

namespace pi
{
namespace affine2d_index
{
static constexpr std::size_t dim = 2;
static constexpr std::size_t num_cols = 3;
static constexpr std::size_t num_rows = 3;

static constexpr std::size_t x_index = 0;
static constexpr std::size_t y_index = 1;

static constexpr std::size_t x_component = dim + x_index*num_cols;
static constexpr std::size_t y_component = dim + y_index*num_cols;

inline static const std::slice position_component{ x_component, dim, num_cols };
inline static const std::gslice linear_component{ 0, { dim, dim }, { 1, num_cols } };

inline static const std::slice x_row{ x_index * num_cols, dim, 1 };
inline static const std::slice y_row{ y_index * num_cols, dim, 1 };

inline static const std::slice x_col{ x_index, dim, num_cols };
inline static const std::slice y_col{ y_index, dim, num_cols };
}

template<std::floating_point Field>
class affine_transform2
{
public:
    // Constructors
    /** Construct an affine2 identity transform */
    affine_transform2()
        : basis_{ identity_matrix<Field, 3>()}
    {
    }

    /** Construct an affine2 transform from a pre-existing transform matrix */
    template<std::ranges::input_range MatrixInput>
    requires std::same_as<std::ranges::range_value_t<MatrixInput>, Field>
    explicit affine_transform2(MatrixInput && input)
    {
        static constexpr std::size_t num_values = num_rows_v<mat3x3> * num_cols_v<mat3x3>;
        std::ranges::copy_n(std::ranges::begin(input), std::min(num_values, std::ranges::size(input)),
                            std::begin(basis_.data));

        update_transform_fields();
    }

    // Public Interface
    /** Get the vector mapped to the standard x-vector */
    template<euclidean_vector2 Point>
    requires std::same_as<Field, scalar_field_t<Point>>
    Point x_basis() const
    {
        const std::valarray x_col = basis_.data[affine2d_index::x_col];
        return Point{ x_col[affine2d_index::x_index], x_col[affine2d_index::y_index] };
    }
    void x_basis(std::convertible_to<Field> auto x_value, std::convertible_to<Field> auto y_value)
    {
        basis_.data[affine2d_index::x_col] = std::valarray{ static_cast<Field>(x_value), static_cast<Field>(y_value) };
        update_transform_fields();
    }
    template<euclidean_vector2 Point>
    requires std::convertible_to<scalar_field_t<Point>, Field>
    void x_basis(const Point & p)
    {
        x_basis(p.x, p.y);
    }

    template<euclidean_vector2 Point>
    requires std::same_as<Field, scalar_field_t<Point>>
    Point y_basis() const
    {
        const std::valarray y_col = basis_.data[affine2d_index::y_col];
        return Point{ y_col[affine2d_index::x_index], y_col[affine2d_index::y_index] };
    }
    void y_basis(std::convertible_to<Field> auto x_value, std::convertible_to<Field> auto y_value)
    {
        basis_.data[affine2d_index::y_col] = std::valarray{ static_cast<Field>(x_value), static_cast<Field>(y_value) };
        update_transform_fields();
    }
    template<euclidean_vector2 Point>
    requires std::convertible_to<scalar_field_t<Point>, Field>
    void y_basis(const Point & p)
    {
        y_basis(p.x, p.y);
    }

    template<valarray_matrix Matrix>
    requires (num_cols_v<Matrix> == 2) and (num_rows_v<Matrix> == 2) and std::same_as<scalar_field_t<Matrix>, Field>
    Matrix basis() const
    {
        return Matrix(basis_.data[affine2d_index::linear_component]);
    }
    template<std::output_iterator<Field> MatrixOutput>
    MatrixOutput basis(MatrixOutput into_values) const
    {
        const std::valarray linear_basis = basis_.data[affine2d_index::linear_component];
        return std::copy(linear_basis, into_values).out;
    }
    template<std::input_iterator MatrixInput>
    requires std::same_as<std::ranges::range_value_t<MatrixInput>, Field>
    MatrixInput basis(MatrixInput from_values)
    {
        static constexpr auto num_values = affine2d_index::dim*affine2d_index::dim;
        std::valarray values(Field(0), num_values);
        from_values = std::copy_n(from_values, num_values, std::begin(values)).in;
        update_transform_fields();
        return from_values;
    }

    /** Get the x-translation */
    Field x() const { return basis_.data[affine2d_index::x_component]; }
    /** Set the x-translation */
    void x(std::convertible_to<Field> auto value)
    {
        basis_.data[affine2d_index::x_component] = static_cast<Field>(value);
    }

    /** Get the y-translation */
    Field y() const { return basis_.data[affine2d_index::y_component]; }
    /** Set the y-translation */
    void y(std::convertible_to<Field> auto value)
    {
        basis_.data[affine2d_index::y_component] = static_cast<Field>(value);
    }

    /** The 2d affine translation matrix for this basis (3x3 row-major matrix) */
    template<std::output_iterator<Field> MatrixOutput>
    MatrixOutput translation(MatrixOutput into_matrix) const
    {
        auto [translation_matrix] = identity_matrix<Field, 3>();
        translation_matrix[affine2d_index::position_component] = basis_.data[affine2d_index::position_component];
        return std::ranges::copy(translation_matrix, into_matrix).out;
    }

    /** The 2d translation of this transform */
    template<euclidean_vector2 Vector>
    Vector translation() const { return Vector{ x(), y() }; }

    /** Set the 2d translation */
    void translation(std::convertible_to<Field> auto in_x, std::convertible_to<Field> auto in_y)
    {
        // update basis
        basis_.data[affine2d_index::position_component] = std::valarray
        {
            static_cast<Field>(in_x), static_cast<Field>(in_y)
        };
    }
    /** Set the 2d translation */
    template<euclidean_vector2 Vector> requires std::convertible_to<scalar_field_t<Vector>, Field>
    void translation(const Vector & p) { translation(p.x, p.y); }

    /** Move the transform by a specified amount */
    void translate(std::convertible_to<Field> auto in_x, std::convertible_to<Field> auto in_y)
    {
        basis_.data[affine2d_index::x_component] += static_cast<Field>(in_x);
        basis_.data[affine2d_index::y_component] += static_cast<Field>(in_y);
    }
    template<euclidean_vector2 Vector> requires std::convertible_to<scalar_field_t<Vector>, Field>
    void translate(const Vector & v) { translate(v.x, v.y); }

    /** How much x-coordinates are scaled by */
    Field x_scale() const { return x_scale_; }
    /** Set the x-scale component */
    void x_scale(std::convertible_to<Field> auto in_x)
    {
        x_scale_ = static_cast<Field>(in_x);
        update_linear_transform();
    }
    /** Add to the x-scale component */
    void scale_x_by(std::convertible_to<Field> auto in_x)
    {
        x_scale_ *= static_cast<Field>(in_x);
        basis_.data[affine2d_index::x_row] *= std::valarray(static_cast<Field>(in_x), affine2d_index::dim);
    }

    /** How much y-coordinates are scaled by */
    Field y_scale() const { return y_scale_; }
    /** Set the y-scale component */
    void y_scale(std::convertible_to<Field> auto in_y)
    {
        y_scale_ = static_cast<Field>(in_y);
        update_linear_transform();
    }
    /** Add to the y-scale component */
    void scale_y_by(std::convertible_to<Field> auto in_y)
    {
        y_scale_ *= static_cast<Field>(in_y);
        basis_.data[affine2d_index::y_row] *= std::valarray(static_cast<Field>(in_y), affine2d_index::dim);
    }

    /** The 2d affine scale matrix for this basis (3x3 row-major matrix) */
    template<std::output_iterator<Field> MatrixOutput>
    MatrixOutput scale(MatrixOutput into_matrix) const
    {
        auto [scale_matrix] = identity_matrix<Field, 3>();
        const std::slice slice_component(0, affine2d_index::dim, affine2d_index::num_cols+1);
        scale_matrix[slice_component] = std::valarray{ x_scale_, y_scale_ };
        return std::ranges::copy(scale_matrix, into_matrix).out;
    }

    template<euclidean_vector2 Vector> requires std::convertible_to<Field, scalar_field_t<Vector>>
    Vector scale() const
    {
        using OtherField = scalar_field_t<Vector>;
        return Vector{ static_cast<OtherField>(x_scale_), static_cast<OtherField>(y_scale_) };
    }

    void scale(std::convertible_to<Field> auto in_x, std::convertible_to<Field> auto in_y)
    {
        x_scale_ = static_cast<Field>(in_x);
        y_scale_ = static_cast<Field>(in_y);
        update_linear_transform();
    }
    void scale_by(std::convertible_to<Field> auto in_x, std::convertible_to<Field> auto in_y)
    {
        scale_x_by(in_x);
        scale_y_by(in_y);
    }

    void scale(std::convertible_to<Field> auto in_scale)
    {
        scale(in_scale, in_scale);
    }
    void scale_by(std::convertible_to<Field> auto in_scale)
    {
        x_scale_ *= in_scale;
        y_scale_ *= in_scale;
        basis_.data[affine2d_index::linear_component]
            *= std::valarray(in_scale, affine2d_index::dim*affine2d_index::dim);
    }

    Field x_shear() const { return x_shear_;  }
    void x_shear(Field value)
    {
        x_shear_ = value;
        update_linear_transform();
    }

    Field y_shear() const { return y_shear_; }
    void y_shear(Field value)
    {
        y_shear_ = value;
        update_linear_transform();
    }

    template<euclidean_vector2 Vector>
    Vector shear() const
    {
        return Vector{ x_shear_, y_shear_ };
    }

    template<std::output_iterator<Field> ShearOutput>
    ShearOutput shear(ShearOutput into_shear)
    {
        auto [shear_matrix] = identity_matrix<Field, 3>();
        shear_matrix[affine2d_index::linear_component] = std::valarray
        {
            Field(1) + x_shear_ * y_shear_, x_shear_, y_shear_, Field(1)
        };
        return std::ranges::copy(shear_matrix, into_shear).out;
    }

    void shear(Field x_value, Field y_value)
    {
        x_shear_ = x_value;
        y_shear_ = y_value;
        update_linear_transform();
    }
    template<euclidean_vector2 Vector>
    void shear(const Vector & v) { shear(v.x, v.y); }

    /** The 2d affine rotation matrix for this basis (3x3 row-major matrix) */
    template<std::output_iterator<Field> MatrixOutput>
    MatrixOutput rotation(MatrixOutput into_matrix) const
    {
        auto [rot] = identity_matrix<Field, 3>();
        rot.data[affine2d_index::linear_component] = rotation_matrix(rotation_angle).data;
        return std::ranges::copy(rot, into_matrix).out;
    }

    /** The rotation of this transform in radians */
    Field rotation() const { return rotation_angle; }
    void rotation(Field angle)
    {
        rotation_angle = angle;
        update_linear_transform();
    }
    void rotate(Field angle)
    {
        rotation_angle += angle;
        update_linear_transform();
    }

    auto * data() const { return std::begin(basis_.data); }
    auto * begin() const { return std::begin(basis_.data); }
    auto * end() const { return std::end(basis_.data); }

    template<euclidean_vector2 Vector> requires std::same_as<scalar_field_t<Vector>, Field>
    Vector operator*(const Vector & p) const
    {
        basic_vector<Field, 3> q{ p.x, p.y, Field(1) };
        for (const auto * current = this; current != nullptr; current = current->parent)
        {
            q = matvec_product(current->basis_, q);
        }
        return Vector{ q.x(), q.y() };
    }

    affine_transform2 operator*(const affine_transform2 & other) const
    {
        auto product = other.basis_;
        for (const auto * current = this; current != nullptr; current = current->parent)
        {
            product = matrix_product(current->basis_, product);
        }
        return affine_transform2(product);
    }

    template<euclidean_vector2 Vector>
    requires std::same_as<scalar_field_t<Vector>, Field>
    Vector inverse(const Vector & v) const
    {
        std::list transforms{ this };
        basic_vector<Field, affine2d_index::num_rows> inverse_point{ v.x, v.y, Field(1) };
        for (const auto * current = this->parent; current != nullptr; current = current->parent)
        {
            transforms.push_front(current);
        }
        for (const auto * transform : transforms)
        {
            inverse_point = linear_solve(transform->basis_, inverse_point);
        }
        return Vector{ inverse_point.x(), inverse_point.y() };
    }

    affine_transform2 local() const
    {
        affine_transform2 cpy = *this;
        cpy.parent = nullptr;
        return cpy;
    }
    affine_transform2 * parent = nullptr;

    // Internal Interface
private:
    void update_transform_fields()
    {
        // TODO: use SVD to find rotation, scale and shear
    }
    void update_linear_transform()
    {
        basis_.data[affine2d_index::linear_component] = rotation_matrix();
        basis_.data[affine2d_index::x_row] *= std::valarray(x_scale_, affine2d_index::dim);
        basis_.data[affine2d_index::y_row] *= std::valarray(y_scale_, affine2d_index::dim);
        auto shear_matrix = identity_matrix<Field, 3>();
        shear(std::begin(shear_matrix.data));
        basis_ = matrix_product(shear_matrix, basis_);
    }
    std::valarray<Field> rotation_matrix()
    {
        return std::valarray<Field>{ std::cos(rotation_angle), -std::sin(rotation_angle),
                                     std::sin(rotation_angle),  std::cos(rotation_angle) };
    }

    Field x_scale_ = Field(1);
    Field y_scale_ = Field(1);
    Field x_shear_ = Field(0);
    Field y_shear_ = Field(0);
    Field rotation_angle = Field(0);
    mat3x3 basis_;
};

using transform2f = affine_transform2<float>;
using transform2d = affine_transform2<double>;
}
