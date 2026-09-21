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
        std::ranges::copy_n(std::begin(values), std::min(static_cast<std::size_t>(Dim), values.size()),
                            std::begin(data));
    }

    explicit basic_vector(const std::valarray<Field> & vec)
        : data(Dim)
    {
        std::ranges::copy_n(std::begin(vec), std::min(static_cast<std::size_t>(Dim), vec.size()),
                            std::begin(data));
    }

    auto constexpr begin() { return std::begin(data); }
    auto constexpr begin() const { return std::begin(data); }

    auto constexpr end() { return std::begin(data); }
    auto constexpr end() const { return std::end(data); }

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

template<valarray_vector Vector>
scalar_field_t<Vector> vector_magnitude(const Vector & v)
{
    return std::sqrt(std::inner_product(std::begin(v.data), std::end(v.data),
                                        std::begin(v.data), scalar_field_t<Vector>(0)));
}

template<numeric Field, std::size_t NumRows, std::size_t NumCols>
struct basic_matrix
{
    using field_t = Field;
    static constexpr std::size_t num_rows = NumRows;
    static constexpr std::size_t num_cols = NumCols;

    basic_matrix() : data(Field(0), NumRows*NumCols) {}
    basic_matrix(const std::valarray<Field> & in_data) : basic_matrix()
    {
        std::ranges::copy_n(std::begin(in_data), std::min(in_data.size(), data.size()), std::begin(data));
    }
    std::valarray<Field> data;
};

template<numeric Field> using mat2x2 = basic_matrix<Field, 2, 2>;
template<numeric Field> using mat2x3 = basic_matrix<Field, 2, 3>;
using mat3x3f = basic_matrix<float, 3, 3>;

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
            std::printf("    %.3f", A.data[col + row*num_cols_v<Matrix>]);
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

template<std::floating_point Field>
basic_matrix<Field, 2, 2> rotation_matrix(Field angle)
{
    return basic_matrix<Field, 2, 2>
    {
        .data { std::cos(angle), -std::sin(angle),
                std::sin(angle),  std::cos(angle) }
    };
}

template<valarray_matrix Matrix>
basic_matrix<scalar_field_t<Matrix>, num_cols_v<Matrix>, num_rows_v<Matrix>>
transpose(const Matrix & A)
{
    basic_matrix<scalar_field_t<Matrix>, num_cols_v<Matrix>, num_rows_v<Matrix>> transposed;
    for (int i = 0; i < num_rows_v<Matrix>; ++i)
    {
        const std::slice row_slice(i*num_cols_v<Matrix>, num_cols_v<Matrix>, 1);
        const std::slice transposed_col_slice(i, num_cols_v<Matrix>, num_rows_v<Matrix>);
        transposed.data[transposed_col_slice] = A.data[row_slice];
    }
    return transposed;
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

template<valarray_matrix Matrix, valarray_vector Vector>
requires std::same_as<scalar_field_t<Matrix>, scalar_field_t<Vector>>
     and (vector_dim<Vector> == num_rows_v<Matrix>)

basic_vector<scalar_field_t<Matrix>, num_rows_v<Matrix>>
matvec_product(const Vector & b, const Matrix & A)
{
    return matvec_product(transpose(A), b);
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

template<valarray_vector VectorA, valarray_vector VectorB>
requires std::same_as<scalar_field_t<VectorA>, scalar_field_t<VectorB>>
basic_matrix<scalar_field_t<VectorA>, vector_dim<VectorA>, vector_dim<VectorB>>
outer_product(const VectorA & a, const VectorB & b)
{
    using Field = scalar_field_t<VectorA>;
    constexpr std::size_t M = vector_dim<VectorA>;
    constexpr std::size_t N = vector_dim<VectorB>;

    basic_matrix<Field, M, 1> cv{ a.data };
    basic_matrix<Field, 1, N> rv{ b.data };
    return matrix_product(cv, rv);
}

template<valarray_matrix Matrix>
void backsub(Matrix & A)
{
    using Field = scalar_field_t<Matrix>;
    constexpr std::size_t num_cols = num_rows_v<Matrix>;
    constexpr std::size_t num_rows = num_rows_v<Matrix>;
    constexpr std::size_t num_augmented_cols = num_cols_v<Matrix>;

    for (int pivot = num_rows-1; pivot >= 0; --pivot)
    {
        const int pivot_index = pivot + pivot*num_augmented_cols;
        const std::slice pivot_row_component( num_cols + pivot*num_augmented_cols, num_augmented_cols-num_cols, 1 );
        A.data[pivot_row_component] /= std::valarray(A.data[pivot_index], num_augmented_cols-num_cols);
        A.data[pivot_index] = Field(1);
        const std::valarray pivot_row = A.data[pivot_row_component];

        for (int row_index = pivot-1; row_index >= 0; --row_index)
        {
            const int index_above_pivot = pivot + row_index*num_augmented_cols;
            const std::slice current_row(num_cols + row_index*num_augmented_cols, num_augmented_cols-num_cols, 1);

            A.data[current_row] -= pivot_row * A.data[index_above_pivot];
            A.data[index_above_pivot] = Field(0);
        }
    }
}

template<valarray_matrix Matrix>
bool is_upper_triangular(const Matrix & A)
{
    static constexpr float SENSITIVITY_EPSILON = 0.001f;
    for (int row = 1; row < num_rows_v<Matrix>; ++row)
    {
        for (int col = 0; col < row; ++col)
        {
            if (std::abs(A.data[col + row*num_cols_v<Matrix>]) > SENSITIVITY_EPSILON)
            {
                return false;
            }
        }
    }
    return true;
}

template<valarray_vector Vector>
Vector householder_vector(const Vector & v)
{
    Vector reflector = v;
    for (auto & x : reflector.data) { x = -x; }
    reflector.data[0] -= std::copysign(1.f, v.x()) * vector_magnitude(v);
    reflector.data /= std::valarray(vector_magnitude(reflector), vector_dim<Vector>);
    return reflector;
}

template<valarray_matrix Matrix>
std::pair<basic_matrix<scalar_field_t<Matrix>, num_rows_v<Matrix>, num_rows_v<Matrix>>, Matrix>
hh_qr(const Matrix & A)
{
    using Field = scalar_field_t<Matrix>;
    static constexpr std::size_t M = num_rows_v<Matrix>;
    static constexpr std::size_t N = num_cols_v<Matrix>;

    auto Q = identity_matrix<Field, M>();
    auto R = A;
    template for (constexpr std::size_t k : std::views::iota(0uz, M-1))
    {
        using Vector = basic_vector<Field, M-k>;
        const std::slice householder_column(k, M-k, N);
        auto v = householder_vector(Vector{ A.data[householder_column] });

        {
            using BlockMatrix = basic_matrix<Field, M-k, N-k>;
            const std::gslice block_component(k + k*N, { M-k, N-k }, { N, 1 });
            const BlockMatrix block{ R.data[block_component] };
            R.data[block_component] -= 2.f*outer_product(v, matvec_product(v, block)).data;
        }
        {
            using BlockMatrix = basic_matrix<Field, M, N-k>;
            const std::gslice block_component(k, { M, N-k }, { N, 1 });
            const BlockMatrix block{ Q.data[block_component] };
            Q.data[block_component] -= 2.f*outer_product(matvec_product(block, v), v).data;
        }
    }
    return std::make_pair(Q, R);
}

template<valarray_matrix Matrix>
requires std::floating_point<scalar_field_t<Matrix>>
void gauss_elim(Matrix & A)
{
    using Field = scalar_field_t<Matrix>;
    constexpr std::size_t M = num_rows_v<Matrix>;
    constexpr std::size_t N = num_cols_v<Matrix>;

    std::size_t pivot_row = 0, pivot_col = 0;
    while (pivot_row < M and pivot_col < N)
    {
        int max_row_below_pivot = pivot_row;
        {
            const std::slice col_slice(pivot_col + pivot_row*N, M-pivot_row, N);
            const std::valarray col_below_pivot = A.data[col_slice];
            auto max_below_pivot = std::ranges::max_element(col_below_pivot, [](Field a, Field b)
            {
                return std::abs(a) < std::abs(b);
            });
            max_row_below_pivot += std::distance(std::begin(col_below_pivot), max_below_pivot);
        }
        if (A.data[pivot_col + max_row_below_pivot*N] == 0) // no pivot in this column, skip to next column
        {
            ++pivot_col; continue;
        }
        // swap rows
        for (const auto j : std::views::iota(0uz, N))
        {
            std::swap(A.data[j + pivot_row*N], A.data[j + max_row_below_pivot*N]);
        }
        for (const auto i : std::views::iota(pivot_row+1, M))
        {
            const Field f = A.data[pivot_col + i*N]/A.data[pivot_col + pivot_row*N];
            A.data[pivot_col + i*N] = Field(0);
            for (const auto j : std::views::iota(pivot_col+1, N))
            {
                A.data[j + i*N] -= f * A.data[j + pivot_row*N];
            }
        }
        ++pivot_row, ++pivot_col;
    }
}

template<std::floating_point Field>
void gauss_jordan2x3(mat2x3<Field> & A)
{
    {
        const float a = A.data[0]; const float b = A.data[1]; const float c = A.data[2];
        const float d = A.data[3];
        A.data[3] = Field(0); A.data[4] -= b*d/a; A.data[5] -= c*d/a;
    }
    {
        const float e = A.data[4];
        A.data[4] = Field(1); A.data[5] /= e;
    }
    {
        const float b = A.data[1]; const float f = A.data[5];
        A.data[1] = Field(0); A.data[2] -= b*f;
    }
    {
        const float a = A.data[0];
        A.data[0] = Field(1); A.data[2] /= a;
    }
}

template<valarray_matrix Matrix, valarray_vector Vector>
requires (num_rows_v<Matrix> == vector_dim<Vector>) and std::same_as<scalar_field_t<Matrix>, scalar_field_t<Vector>>
basic_vector<scalar_field_t<Matrix>, num_cols_v<Matrix>>
linear_solve(const Matrix & A, const Vector & b)
{
    using Field = scalar_field_t<Matrix>;
    constexpr std::size_t M = num_rows_v<Matrix>;
    constexpr std::size_t N = num_cols_v<Matrix>;

    using OutVector = basic_vector<Field, N>;

    constexpr std::size_t num_augmented_cols = N + 1;
    const std::gslice original_component(0, { M, N }, { num_augmented_cols, 1 });
    const std::slice inverse_component(N, M, num_augmented_cols);

    using AugmentedMatrix = basic_matrix<Field, M, num_augmented_cols>;
    if (is_upper_triangular(A))
    {
        AugmentedMatrix augmented;
        augmented.data[original_component] = A.data;
        augmented.data[inverse_component] = b.data;
        backsub(augmented);
        return OutVector{ augmented.data[inverse_component] };
    }
    else
    {
        auto [Q, R] = hh_qr(A);
        AugmentedMatrix augmented;
        augmented.data[original_component] = R.data;
        augmented.data[inverse_component] = matvec_product(transpose(Q), b).data;
        backsub(augmented);
        return matvec_product(transpose(Q), OutVector{ augmented.data[inverse_component] });
    }
}
}

