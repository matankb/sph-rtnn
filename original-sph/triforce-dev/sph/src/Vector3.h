#ifndef TRIFORCE_VECTOR3_H
#define TRIFORCE_VECTOR3_H

#include <Kokkos_Core.hpp>

template<typename T>
class vec3 {
public:
  KOKKOS_INLINE_FUNCTION vec3() = default;

  // would this be useful?
//  KOKKOS_INLINE_FUNCTION explicit vec3(T x) : e{x, x, x} {}

  KOKKOS_INLINE_FUNCTION vec3(T e0, T e1, T e2) : e{e0, e1, e2} {}

  KOKKOS_INLINE_FUNCTION T x() const { return e[0]; }

  KOKKOS_INLINE_FUNCTION T y() const { return e[1]; }

  KOKKOS_INLINE_FUNCTION T z() const { return e[2]; }

  // Unary Negation
  KOKKOS_INLINE_FUNCTION vec3 operator-() const {
    return {-e[0], -e[1], -e[2]};
  }

  KOKKOS_INLINE_FUNCTION T operator[](uint32_t i) const { return e[i]; }

  KOKKOS_INLINE_FUNCTION T &operator[](uint32_t i) { return e[i]; }

  KOKKOS_INLINE_FUNCTION vec3& operator=(const vec3 &rhs) {
    if (this == &rhs) { return *this; }
    e[0] = rhs.e[0];
    e[1] = rhs.e[1];
    e[2] = rhs.e[2];
    return *this;
  }

  KOKKOS_INLINE_FUNCTION vec3& operator=(const volatile vec3& rhs) {
    if (this == &rhs) { return *this; }
    e[0] = rhs.e[0];
    e[1] = rhs.e[1];
    e[2] = rhs.e[2];
    return *this;
  }

  KOKKOS_INLINE_FUNCTION void operator=(const vec3& rhs) volatile {
    if (this == &rhs) { return; }
    e[0] = rhs.e[0];
    e[1] = rhs.e[1];
    e[2] = rhs.e[2];
  }

  KOKKOS_INLINE_FUNCTION vec3 &operator+=(const vec3 &v) {
    e[0] += v.e[0];
    e[1] += v.e[1];
    e[2] += v.e[2];
    return *this;
  }

  KOKKOS_INLINE_FUNCTION vec3 &operator-=(const vec3 &v) {
    e[0] -= v.e[0];
    e[1] -= v.e[1];
    e[2] -= v.e[2];
    return *this;
  }

  KOKKOS_INLINE_FUNCTION vec3 &operator*=(const T s) {
    e[0] *= s;
    e[1] *= s;
    e[2] *= s;
    return *this;
  }

  KOKKOS_INLINE_FUNCTION vec3 &operator/=(const T s) {
    e[0] /= s;
    e[1] /= s;
    e[2] /= s;
    return *this;
  }

  KOKKOS_INLINE_FUNCTION T length_squared() const {
    return (e[0] * e[0]) + (e[1] * e[1]) + (e[2] * e[2]);
  }

  KOKKOS_INLINE_FUNCTION T length() const {
    return sqrt(length_squared());
  }

public:
  T e[3];
};

template<typename T>
std::ostream &operator<<(std::ostream &out, const vec3<T> &dat) {
  return (out << dat.e[0] << ' ' << dat.e[1] << ' ' << dat.e[2]);
}

template<typename T>
KOKKOS_INLINE_FUNCTION vec3<T> unit_vector(vec3<T> u){
  return u / u.length();
}

/* Vector-Vector Operators */
template<typename T>
KOKKOS_INLINE_FUNCTION vec3<T> operator+(const vec3<T> &u, const vec3<T> &v) {
  return {u.e[0] + v.e[0],
          u.e[1] + v.e[1],
          u.e[2] + v.e[2]};
}

template<typename T>
KOKKOS_INLINE_FUNCTION vec3<T> operator-(const vec3<T> &u, const vec3<T> &v) {
  return {u.e[0] - v.e[0],
          u.e[1] - v.e[1],
          u.e[2] - v.e[2]};
}

template<typename T>
KOKKOS_INLINE_FUNCTION vec3<T> operator*(const vec3<T> &u, const vec3<T> &v) {
  return {u.e[0] * v.e[0],
          u.e[1] * v.e[1],
          u.e[2] * v.e[2]};
}

template<typename T>
KOKKOS_INLINE_FUNCTION vec3<T> operator/(const vec3<T> &u, const vec3<T> &v) {
  return {u.e[0] / v.e[0],
          u.e[1] / v.e[1],
          u.e[2] / v.e[2]};
}

template<typename T>
KOKKOS_INLINE_FUNCTION T dot(const vec3<T> &u, const vec3<T> &v) {
  return (u.e[0] * v.e[0]) + (u.e[1] * v.e[1]) + (u.e[2] * v.e[2]);
}

template<typename T>
KOKKOS_INLINE_FUNCTION vec3<T> cross(const vec3<T> &u, const vec3<T> &v) {
  return {u.e[1] * v.e[2] - u.e[2] * v.e[1],
          u.e[2] * v.e[0] - u.e[0] * v.e[2],
          u.e[0] * v.e[1] - u.e[1] * v.e[0]};
}

// Vector-Scalar Operations
template<typename T>
KOKKOS_INLINE_FUNCTION vec3<T> operator*(T s, const vec3<T> &u)  {
  return {s * u.e[0], s * u.e[1], s * u.e[2]};
}

template<typename T>
KOKKOS_INLINE_FUNCTION vec3<T> operator*(const vec3<T> &u, T s)  {
  return s * u;
}

template<typename T>
KOKKOS_INLINE_FUNCTION vec3<T> operator/(vec3<T> u, T s) {
  return (1 / s) * u;
}

#endif //TRIFORCE_VECTOR3_H
