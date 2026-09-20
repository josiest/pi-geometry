#pragma once
#include <concepts>
#include <numeric>

namespace pi
{
template<typename Field>
concept numeric = std::integral<std::remove_cvref_t<Field>> or std::floating_point<std::remove_cvref_t<Field>>;

template<typename Vector>
concept euclidean_vector2 = requires(Vector v)
{
    Vector{};
    { v.x } -> numeric;
    { v.y } -> std::same_as<std::add_lvalue_reference_t<decltype(std::declval<Vector>().x)>>;
    Vector{ v.x, v.y };
};

template<typename Vector>
concept euclidean_vector3 = requires(Vector v)
{
    Vector{};
    { v.x } -> numeric;
    { v.y } -> std::same_as<decltype(std::declval<Vector>().x)>;
    { v.z } -> std::same_as<decltype(std::declval<Vector>().x)>;
    Vector{ v.x, v.y, v.z };
};

template<typename Vector>
concept euclidean_vector4 = requires(Vector v)
{
    Vector{};
    { v.x } -> numeric;
    { v.y } -> std::same_as<decltype(std::declval<Vector>().x)>;
    { v.z } -> std::same_as<decltype(std::declval<Vector>().x)>;
    { v.w } -> std::same_as<decltype(std::declval<Vector>().x)>;
    Vector{ v.x, v.y, v.z, v.w };
};

template<typename Vector>
concept euclidean_vector = euclidean_vector2<Vector> or euclidean_vector3<Vector> or euclidean_vector4<Vector>;

template<typename Vector>
concept color_vector3 = requires(Vector v)
{
    Vector{};
    { v.r } -> numeric;
    { v.g } -> std::same_as<decltype(std::declval<Vector>().r)>;
    { v.b } -> std::same_as<decltype(std::declval<Vector>().r)>;
    Vector{ v.r, v.g, v.b };
};

template<typename Vector>
concept color_vector4 = requires(Vector v)
{
    Vector{};
    { v.r } -> numeric;
    { v.g } -> std::same_as<decltype(std::declval<Vector>().r)>;
    { v.b } -> std::same_as<decltype(std::declval<Vector>().r)>;
    { v.a } -> std::same_as<decltype(std::declval<Vector>().r)>;
    Vector{ v.r, v.g, v.b, v.a };
};

template<typename Vector>
concept color_vector = color_vector3<Vector> or color_vector4<Vector>;

template<typename T>
struct scalar_field
{
};

template<euclidean_vector Vector>
struct scalar_field<Vector>
{
    using type = std::remove_cvref_t<decltype(std::declval<Vector>().x)>;
};

template<color_vector Vector>
struct scalar_field<Vector>
{
    using type = std::remove_cvref_t<decltype(std::declval<Vector>().r)>;
};

template<typename Vector>
using scalar_field_t = scalar_field<Vector>::type;

template<euclidean_vector2 Vector>
requires std::floating_point<scalar_field_t<Vector>>
Vector lerp(Vector a, Vector b, scalar_field_t<Vector> t)
{
    return Vector { std::lerp(a.x, b.x, t), std::lerp(a.y, b.y, t) };
}
}
