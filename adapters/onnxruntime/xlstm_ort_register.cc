/* Copyright 2026 RAWS labs
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =========================================================================
 * ONNX Runtime shared library entry point — registers sLSTM + mLSTM ops.
 *
 * Build as shared library:
 *   g++ -std=c++17 -shared -fPIC -o libxlstm_ort.so \
 *       adapters/onnxruntime/*.cc src/slstm.c src/mlstm.c \
 *       -Iinclude -Iadapters/onnxruntime -I<onnxruntime>/include -lm
 *
 * Load in consumer:
 *   session_options.RegisterCustomOpsLibrary("libxlstm_ort.so");
 * ===========================================================================*/

#include "slstm_ort.h"
#include "mlstm_ort.h"

#include <memory>
#include <mutex>
#include <vector>

static const char* kOpDomain = "com.raws.xlstm";

static void KeepDomainAlive(Ort::CustomOpDomain&& domain) {
    static std::vector<Ort::CustomOpDomain> s_domains;
    static std::mutex s_mutex;
    std::lock_guard<std::mutex> lock(s_mutex);
    s_domains.push_back(std::move(domain));
}

extern "C" OrtStatus* ORT_API_CALL RegisterCustomOps(
    OrtSessionOptions* options,
    const OrtApiBase* api_base)
{
    Ort::InitApi(api_base->GetApi(ORT_API_VERSION));

    Ort::CustomOpDomain domain{kOpDomain};

    using Ort::Custom::OrtLiteCustomOp;

    static std::unique_ptr<OrtLiteCustomOp> slstm_op{
        Ort::Custom::CreateLiteCustomOp(
            "SLSTM", "CPUExecutionProvider", SLstmOrtKernel)
    };
    domain.Add(slstm_op.get());

    static std::unique_ptr<OrtLiteCustomOp> mlstm_op{
        Ort::Custom::CreateLiteCustomOp(
            "MLSTM", "CPUExecutionProvider", MLstmOrtKernel)
    };
    domain.Add(mlstm_op.get());

    Ort::UnownedSessionOptions session_options(options);
    session_options.Add(domain);
    KeepDomainAlive(std::move(domain));

    return nullptr;
}
