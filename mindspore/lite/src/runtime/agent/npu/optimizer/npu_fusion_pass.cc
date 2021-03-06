/**
 * Copyright 2020 Huawei Technologies Co., Ltd
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
#include "src/runtime/agent/npu/optimizer/npu_fusion_pass.h"
#include <vector>
#include "src/lite_kernel.h"
#include "nnacl/concat_parameter.h"

namespace mindspore::lite {
bool CheckFusion(kernel::LiteKernel *kernel) {
  auto pre_flag =
    std::all_of(kernel->in_kernels().begin(), kernel->in_kernels().end(), [](const kernel::LiteKernel *in_kernel) {
      return in_kernel->Type() == schema::PrimitiveType_Nchw2Nhwc && in_kernel->out_kernels().size() == 1;
    });
  if (!pre_flag) {
    return false;
  }
  auto post_flag = std::all_of(
    kernel->out_kernels().begin(), kernel->out_kernels().end(),
    [](const kernel::LiteKernel *out_kernel) { return out_kernel->Type() == schema::PrimitiveType_Nhwc2Nchw; });
  return post_flag;
}

bool CheckFormatFusion(kernel::LiteKernel *kernel) {
  if (kernel->out_kernels().empty()) {
    return false;
  }
  if (kernel->Type() == schema::PrimitiveType_Nhwc2Nchw) {
    return std::all_of(
      kernel->out_kernels().begin(), kernel->out_kernels().end(),
      [](const kernel::LiteKernel *kernel) { return kernel->Type() == schema::PrimitiveType_Nchw2Nhwc; });
  }
  if (kernel->Type() == schema::PrimitiveType_Nchw2Nhwc) {
    return std::all_of(
      kernel->out_kernels().begin(), kernel->out_kernels().end(),
      [](const kernel::LiteKernel *kernel) { return kernel->Type() == schema::PrimitiveType_Nhwc2Nchw; });
  }
  return false;
}

void NPUFusionPass::RemoveAndFreeKernel(kernel::LiteKernel *cur_kernel) {
  auto itr = find(kernels->begin(), kernels->end(), cur_kernel);
  if (itr != kernels->end()) {
    kernels->erase(itr);
  }
  delete cur_kernel;
}

void NPUFusionPass::UpdatePreKernels(kernel::LiteKernel *cur_kernel) {
  for (auto in_kernel : cur_kernel->in_kernels()) {
    auto pre_kernel = in_kernel->in_kernels()[0];

    auto pre_out_kernels = pre_kernel->out_kernels();
    for (size_t i = 0; i < pre_out_kernels.size(); i++) {
      if (pre_out_kernels[i] == in_kernel) {
        pre_out_kernels[i] = cur_kernel;
        break;
      }
    }
    pre_kernel->set_out_kernels(pre_out_kernels);

    auto cur_in_kernels = cur_kernel->in_kernels();
    for (size_t i = 0; i < cur_in_kernels.size(); i++) {
      if (cur_in_kernels[i] == in_kernel) {
        cur_in_kernels[i] = pre_kernel;
        break;
      }
    }
    cur_kernel->set_in_kernels(cur_in_kernels);
    RemoveAndFreeKernel(in_kernel);
  }
}

void NPUFusionPass::UpdatePostKernels(kernel::LiteKernel *cur_kernel) {
  for (auto out_kernel : cur_kernel->out_kernels()) {
    auto post_kernel = out_kernel->out_kernels()[0];

    auto post_in_kernels = post_kernel->in_kernels();
    for (size_t i = 0; i < post_in_kernels.size(); i++) {
      if (post_in_kernels[i] == out_kernel) {
        post_in_kernels[i] = cur_kernel;
        break;
      }
    }
    post_kernel->set_in_kernels(post_in_kernels);

    auto cur_out_kernels = cur_kernel->out_kernels();
    for (size_t i = 0; i < cur_out_kernels.size(); i++) {
      if (cur_out_kernels[i] == out_kernel) {
        cur_out_kernels[i] = post_kernel;
        break;
      }
    }
    cur_kernel->set_out_kernels(cur_out_kernels);
    RemoveAndFreeKernel(out_kernel);
  }
}

void UpdatePreTensors(kernel::LiteKernel *cur_kernel) {
  auto tensors_vec = cur_kernel->in_tensors();
  for (auto in_kernel : cur_kernel->in_kernels()) {
    lite::Tensor *cur_tensor = nullptr;
    auto in_tensor = in_kernel->in_tensors()[0];
    auto out_tensor = in_kernel->out_tensors()[0];
    auto pre_kernel = in_kernel->in_kernels()[0];
    for (size_t i = 0; i < pre_kernel->out_tensors().size(); i++) {
      if (pre_kernel->out_tensors()[i] == in_tensor) {
        cur_tensor = pre_kernel->out_tensors()[i];
      }
    }
    for (size_t i = 0; i < tensors_vec.size(); i++) {
      if (tensors_vec[i] == out_tensor) {
        tensors_vec[i] = cur_tensor;
      }
    }
  }
  cur_kernel->set_in_tensors(tensors_vec);
}

void UpdatePostTensors(kernel::LiteKernel *cur_kernel) {
  auto tensor = cur_kernel->out_tensors()[0];
  for (auto out_kernel : cur_kernel->out_kernels()) {
    auto out_tensor = out_kernel->out_tensors()[0];
    for (auto post_kernel : out_kernel->out_kernels()) {
      auto tensors_vec = post_kernel->in_tensors();
      for (int i = 0; i < tensors_vec.size(); i++) {
        if (tensors_vec[i] == out_tensor) {
          tensors_vec[i] = tensor;
        }
      }
      post_kernel->set_in_tensors(tensors_vec);
    }
  }
}

int TransFormAxis(int axis) {
  switch (axis) {
    case 0:
      return 0;
    case 1:
      return 2;
    case 2:
      return 3;
    case 3:
    case -1:
      return 1;
    default:
      return -2;
  }
}

void NPUFusionPass::UpdateKernel(kernel::LiteKernel *kernel) {
  UpdatePreTensors(kernel);
  UpdatePostTensors(kernel);
  UpdatePreKernels(kernel);
  UpdatePostKernels(kernel);
}

int NPUFusionPass::CommonFusion(kernel::LiteKernel *kernel) {
  UpdateKernel(kernel);
  return RET_OK;
}

int NPUFusionPass::ConcatFusion(kernel::LiteKernel *kernel) {
  UpdateKernel(kernel);
  auto concat_param = reinterpret_cast<ConcatParameter *>(kernel->op_parameter());
  concat_param->axis_ = TransFormAxis(concat_param->axis_);
  return RET_OK;
}

int NPUFusionPass::FormatFusion(kernel::LiteKernel *kernel) {
  auto pre_kernel = kernel->in_kernels()[0];
  auto in_tensor = kernel->in_tensors()[0];
  auto out_tensor = kernel->out_tensors()[0];
  auto tensor_itr = std::find(pre_kernel->out_tensors().begin(), pre_kernel->out_tensors().end(), in_tensor);
  if (tensor_itr != pre_kernel->out_tensors().end()) {
    in_tensor = *tensor_itr;
  } else {
    MS_LOG(ERROR) << "Can't find the connneted tensor between kernel " << kernel->name() << " and it's pre_kernel.";
    return RET_ERROR;
  }

  std::vector<kernel::LiteKernel *> pre_insert_kernels;
  for (const auto &trans_kernel : kernel->out_kernels()) {
    for (const auto &post_kernel : trans_kernel->out_kernels()) {
      // update tensor
      auto tensors_vec = post_kernel->in_tensors();
      for (size_t i = 0; i < tensors_vec.size(); i++) {
        if (tensors_vec[i] == out_tensor) {
          tensors_vec[i] = in_tensor;
          break;
        }
      }
      post_kernel->set_in_tensors(tensors_vec);

      // update kernel
      auto post_in_kernels = post_kernel->in_kernels();
      for (size_t i = 0; i < post_in_kernels.size(); i++) {
        if (post_in_kernels[i] == trans_kernel) {
          post_in_kernels[i] = pre_kernel;
          break;
        }
      }
      post_kernel->set_in_kernels(post_in_kernels);
      pre_insert_kernels.push_back(post_kernel);
      RemoveAndFreeKernel(trans_kernel);
    }
  }
  auto pre_out_kernels = pre_kernel->out_kernels();
  auto itr = find(pre_out_kernels.begin(), pre_out_kernels.end(), kernel);
  pre_out_kernels.insert(itr, pre_insert_kernels.begin(), pre_insert_kernels.end());
  pre_kernel->set_in_kernels(pre_out_kernels);
  RemoveAndFreeKernel(kernel);
  return RET_OK;
}

int NPUFusionPass::Run() {
  for (size_t i = 0; i < kernels->size(); i++) {
    auto kernel = (*kernels)[i];
    if (kernel->Type() == schema::PrimitiveType_Nchw2Nhwc || kernel->Type() == schema::PrimitiveType_Nchw2Nhwc) {
      if (CheckFormatFusion(kernel)) {
        i--;
        FormatFusion(kernel);
      }
      continue;
    }
    if (!CheckFusion(kernel)) {
      continue;
    }
    switch (kernel->Type()) {
      case schema::PrimitiveType_Concat:
        i -= kernel->in_kernels().size();
        ConcatFusion(kernel);
        continue;
      case schema::PrimitiveType_Add:
      case schema::PrimitiveType_Activation:
      case schema::PrimitiveType_Eltwise:
        i -= kernel->in_kernels().size();
        CommonFusion(kernel);
        continue;
      default:
        continue;
    }
  }
  return RET_OK;
}
}  // namespace mindspore::lite
