#pragma once
#include <ranges>

#include <valarray>
#include <algorithm>
#include <numeric>

#include <cstdio>
#include "vectors.hpp"

namespace pi
{

template<numeric Field, std::size_t Dim>
struct basic_vector
{
    using field_t = Field;
    static constexpr std::size_t dim = Dim;

    basic_vector() : data(Field(0), Dim) {}
    basic_vector(std::initializer_list<Field> values)
        : data(Dim)
    {
        std::ranges::copy_n(values.begin(), std::min(static_cast<std::size_t>(Dim), values.size()),
                            std::begin(data));
    }

    explicit basic_vector(const std::valarray<Field> & vec)
        : data(Dim)
    {
        std::ranges::copy_n(std::begin(vec), std::min(static_cast<std::size_t>(Dim), vec.size()),
                            std::begin(data));
    }

    auto constexpr begin() { return data.begin(); }
    auto constexpr begin() const { return data.begin(); }

    auto constexpr end() { return data.end(); }
    auto constexpr end() const { return data.end(); }

    Field x() const { return data[0]; }

    template<std::size_t N = Dim> requires (Dim > 1)
    Field y() const { return data[1]; }

    template<std::size_t N = Dim> requires (Dim > 2)
    Field z() const { return data[2]; }

    template<std::size_t N = Dim> requires (Dim > 3)
    Field w() const { return data[3]; }

    std::valarray<Field> data{ Dim };
};

template<numeric Field> using vec2 = basic_vector<Field, 2>;
template<numeric Field> using vec3 = basic_vector<Field, 3>;
template<numeric Field> using vec4 = basic_vector<Field, 4>;

using vec2f = basic_vector<float, 2>;
using vec3f = basic_vector<float, 3>;
using vec4f = basic_vector<float, 4>;

template<class Vector>
concept valarray_vector = requires(Vector v)
{
    typename Vector::field_t;
    std::unsigned_integral<std::remove_cvref_t<decltype(Vector::dim)>>;
    { v.data } -> std::same_as<std::add_lvalue_reference_t<std::valarray<typename Vector::field_t>>>;
};

template<valarray_vector Vector>
struct scalar_field<Vector>
{
    using type = Vector::field_t;
};

template<valarray_vector Vector>
constexpr std::size_t vector_dim = Vector::dim;

template<numeric Field, std::size_t NumRows, std::size_t NumCols>
struct basic_matrix
{
    using field_t = Field;
    static constexpr std::size_t num_rows = NumRows;
    static constexpr std::size_t num_cols = NumCols;

    basic_matrix() : data(Field(0), NumRows*NumCols) {}
    std::valarray<Field> data;
};

using mat3x3 = basic_matrix<float, 3, 3>;

template<std::floating_point Field>
basic_matrix<Field, 2, 2> rotation_matrix(Field angle)
{
    return basic_matrix<Field, 2, 2>
    {
        .data { std::cos(angle), -std::sin(angle),
                std::sin(angle),  std::cos(angle) }
    };
}

template<class Matrix>
concept valarray_matrix = requires(Matrix m)
{
    typename Matrix::field_t;
    std::unsigned_integral<std::remove_cvref_t<decltype(Matrix::num_rows)>>;
    std::unsigned_integral<std::remove_cvref_t<decltype(Matrix::num_cols)>>;
    { m.data } -> std::same_as<std::add_lvalue_reference_t<std::valarray<typename Matrix::field_t>>>;
};

template<valarray_matrix Matrix>
struct scalar_field<Matrix>
{
    using type = Matrix::field_t;
};

template<class Matrix>
constexpr std::size_t num_rows_v = Matrix::num_rows;

template<class Matrix>
constexpr std::size_t num_cols_v = Matrix::num_cols;

template<valarray_matrix Matrix>
void print_matrix(const Matrix & A)
{
    std::string_view line_sep = "";
    for (std::size_t row = 0; row < num_rows_v<Matrix>; ++row)
    {
        std::printf("%s", line_sep.data());
        for (std::size_t col = 0; col < num_cols_v<Matrix>; ++col)
        {
            std::printf("    %.1f", A.data[col + row*num_cols_v<Matrix>]);
        }
        line_sep = "\n";
    }
    std::printf("\n");
}

template<numeric Field, std::size_t Dim>
basic_matrix<Field, Dim, Dim> identity_matrix()
{
    basic_matrix<Field, Dim, Dim> identity;
    static const std::slice diagonal(0, Dim, Dim+1);
    identity.data[diagonal] = Field(1);
    return identity;
}

template<valarray_matrix Matrix, valarray_vector Vector>
requires std::same_as<scalar_field_t<Matrix>, scalar_field_t<Vector>>
     and (vector_dim<Vector> == num_cols_v<Matrix>)

basic_vector<scalar_field_t<Matrix>, num_rows_v<Matrix>>
matvec_product(const Matrix & A, const Vector & b)
{
    basic_vector<scalar_field_t<Matrix>, num_rows_v<Matrix>> product;
    for (std::size_t row_index = 0; row_index < num_rows_v<Matrix>; ++row_index)
    {
        const std::slice row_component(row_index*num_cols_v<Matrix>, num_cols_v<Matrix>, 1);
        const std::valarray row = A.data[row_component];
        product.data[row_index] = std::inner_product(std::begin(b.data), std::end(b.data),
                                                     std::begin(row), scalar_field_t<Matrix>(0));
    }
    return product;
}

template<valarray_matrix MatrixA, valarray_matrix MatrixB>
requires (num_cols_v<MatrixA> == num_rows_v<MatrixB>)
     and std::same_as<scalar_field_t<MatrixA>, scalar_field_t<MatrixB>>

basic_matrix<scalar_field_t<MatrixA>, num_rows_v<MatrixA>, num_cols_v<MatrixB>>
matrix_product(const MatrixA & A, const MatrixB & B)
{
    basic_matrix<scalar_field_t<MatrixA>, num_rows_v<MatrixA>, num_cols_v<MatrixB>> product;
    for (std::size_t col_index = 0; col_index < num_cols_v<MatrixA>; ++col_index)
    {
        const std::slice col_component_AB(col_index, num_rows_v<MatrixA>, num_cols_v<MatrixB>);
        const std::slice col_component_B(col_index, num_rows_v<MatrixB>, num_cols_v<MatrixB>);

        using column_vector = basic_vector<scalar_field_t<MatrixB>, num_rows_v<MatrixB>>;
        product.data[col_component_AB] = matvec_product(A, column_vector(B.data[col_component_B])).data;
    }
    return product;
}
}

