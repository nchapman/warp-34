// scalar_type.hpp stand-in (vllm/csrc/core/scalar_type.hpp is torch-tied).
//
// The marlin device headers only use ScalarType ids as compile-time
// equality keys (MarlinScalarType<id> specializations + dequant
// specializations + if-constexpr dispatch in marlin_template.h), so the
// id VALUES only need to be distinct and consistent within this build.
// size_bits()/id() are the only members the kernel chain touches.
#pragma once
#include <cstdint>

namespace vllm {

using ScalarTypeId = int64_t; // upstream: ScalarType::Id == int64_t (NOT an
                              // enum — template args must match exactly)
static constexpr ScalarTypeId ST_ID_NONE = 0;
static constexpr ScalarTypeId ST_ID_F16 = 1;
static constexpr ScalarTypeId ST_ID_BF16 = 2;
static constexpr ScalarTypeId ST_ID_FE4M3FN = 3;
static constexpr ScalarTypeId ST_ID_S4 = 4;
static constexpr ScalarTypeId ST_ID_U4 = 5;
static constexpr ScalarTypeId ST_ID_S8 = 6;
static constexpr ScalarTypeId ST_ID_U8 = 7;
static constexpr ScalarTypeId ST_ID_U4B8 = 8;
static constexpr ScalarTypeId ST_ID_U8B128 = 9;
static constexpr ScalarTypeId ST_ID_FE2M1F = 10;
static constexpr ScalarTypeId ST_ID_FE8M0FNU = 11;
static constexpr ScalarTypeId ST_ID_U2B2 = 12;

struct ScalarType {
  ScalarTypeId id_;
  int bits_;
  constexpr ScalarType(ScalarTypeId id, int bits)
      : id_(id), bits_(bits) {}
  constexpr int64_t id() const { return id_; }
  constexpr int size_bits() const { return bits_; }
  constexpr bool operator==(ScalarType const& o) const {
    return id_ == o.id_;
  }
  constexpr bool operator!=(ScalarType const& o) const {
    return id_ != o.id_;
  }

  // marlin_template uses ScalarType::from_id(id) to rebuild the type for
  // if-constexpr dispatch and size_bits(); recover bits from the id
  static constexpr ScalarType from_id(int64_t id) {
    const int bits = (id == ST_ID_F16 || id == ST_ID_BF16)   ? 16
                     : (id == ST_ID_U2B2)                    ? 2
                     : (id == ST_ID_S4 || id == ST_ID_U4 ||
                        id == ST_ID_U4B8)                    ? 4
                                                             : 8;
    return ScalarType(static_cast<ScalarTypeId>(id), bits);
  }
};

static constexpr ScalarType kS4{ST_ID_S4, 4};
static constexpr ScalarType kU4{ST_ID_U4, 4};
static constexpr ScalarType kS8{ST_ID_S8, 8};
static constexpr ScalarType kU8{ST_ID_U8, 8};
static constexpr ScalarType kU2B2{ST_ID_U2B2, 2};
static constexpr ScalarType kU4B8{ST_ID_U4B8, 4};
static constexpr ScalarType kU8B128{ST_ID_U8B128, 8};
static constexpr ScalarType kFE4M3fn{ST_ID_FE4M3FN, 8};
static constexpr ScalarType kFloat16{ST_ID_F16, 16};
static constexpr ScalarType kBFloat16{ST_ID_BF16, 16};
static constexpr ScalarType kFE2M1f{ST_ID_FE2M1F, 4};
static constexpr ScalarType kFE8M0fnu{ST_ID_FE8M0FNU, 8};


// aliases used around vllm csrc
static constexpr ScalarType kUint2b2 = kU2B2;
static constexpr ScalarType kUint4b8 = kU4B8;
static constexpr ScalarType kUint8b128 = kU8B128;
static constexpr ScalarType kInt4 = kS4;
static constexpr ScalarType kInt8 = kS8;

}  // namespace vllm
