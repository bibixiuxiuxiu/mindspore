/**
 * Copyright 2019-2020 Huawei Technologies Co., Ltd
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "src/ops/bn_grad_input.h"

namespace mindspore {
namespace lite {
#ifdef PRIMITIVE_WRITEABLE
float BNGradInput::GetEps() const { return this->primitive_->value.AsBNGradInput()->eps; }
float BNGradInput::GetMomentum() const { return this->primitive_->value.AsBNGradInput()->momentum; }

void BNGradInput::SetEps(float eps) { this->primitive_->value.AsBNGradInput()->eps = eps; }
void BNGradInput::SetMomentum(float momentum) { this->primitive_->value.AsBNGradInput()->momentum = momentum; }
int BNGradInput::UnPackAttr(const Primitive &prim, const std::vector<AnfNodePtr> &inputs) {
  if (this->primitive_ == nullptr) {
    this->primitive_ = new (std::nothrow) schema::PrimitiveT;
    if (this->primitive_ == nullptr) {
      MS_LOG(ERROR) << "new primitiveT failed";
      return RET_ERROR;
    }
    this->primitive_->value.type = schema::PrimitiveType_BNGradInput;
  }
  if (this->primitive_->value.type != schema::PrimitiveType_BNGradInput) {
    MS_LOG(ERROR) << "Primitive type is error :" << this->primitive_->value.type;
    return RET_ERROR;
  }
  if (this->primitive_->value.value == nullptr) {
    auto attr = new (std::nothrow) schema::BNGradInputT();
    if (attr == nullptr) {
      MS_LOG(ERROR) << "new primitiveT value failed";
      return RET_ERROR;
    }
    attr->eps = GetValue<float>(prim.GetAttr("eps"));
    attr->momentum = GetValue<float>(prim.GetAttr("momentum"));
    this->primitive_->value.value = attr;
    if (this->primitive_->value.value == nullptr) {
      MS_LOG(ERROR) << "primitive value is nullptr";
      return RET_ERROR;
    }
  }
  return RET_OK;
}
#else
int BNGradInput::UnPackToFlatBuilder(const schema::Primitive *primitive, flatbuffers::FlatBufferBuilder *fbb) {
  MS_ASSERT(nullptr != primitive);
  MS_ASSERT(nullptr != fbb);
  auto attr = primitive->value_as_BNGradInput();
  if (attr == nullptr) {
    MS_LOG(ERROR) << "value_as_BNGradInputInput return nullptr";
    return RET_ERROR;
  }
  auto val_offset = schema::CreateBNGradInput(*fbb, attr->eps(), attr->momentum());
  auto prim_offset = schema::CreatePrimitive(*fbb, schema::PrimitiveType_BNGradInput, val_offset.o);
  fbb->Finish(prim_offset);
  return RET_OK;
}
float BNGradInput::GetEps() const { return this->primitive_->value_as_BNGradInput()->eps(); }
float BNGradInput::GetMomentum() const { return this->primitive_->value_as_BNGradInput()->momentum(); }

#endif
}  // namespace lite
}  // namespace mindspore
