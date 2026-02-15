/* TVM_REGISTER_GLOBAL wrapper for sLSTM/mLSTM packed functions.
 *
 * Our adapters export TVMBackendPackedCFunc signatures expecting DLTensor*
 * handles directly in TVMValue.v_handle. When called from Python, NDArrays
 * arrive as kTVMNDArrayHandle (container pointer, not DLTensor*). This wrapper
 * extracts the DLTensor from each NDArray arg and repacks them as
 * kTVMDLTensorHandle before forwarding to the adapter.
 */

#include <tvm/runtime/ndarray.h>
#include <tvm/runtime/registry.h>

#include <vector>

extern "C" {
#include "slstm_tvm.h"
#include "mlstm_tvm.h"
}

namespace {

/* Extract DLTensor* handles from TVMArgs and repack as kTVMDLTensorHandle. */
void RepackAsDLTensor(tvm::runtime::TVMArgs args,
                      std::vector<TVMValue>& values,
                      std::vector<int>& type_codes) {
    int n = args.size();
    values.resize(n);
    type_codes.resize(n);
    for (int i = 0; i < n; i++) {
        DLTensor* t = args[i].operator DLTensor*();
        values[i].v_handle = t;
        type_codes[i] = kTVMDLTensorHandle;
    }
}

}  // namespace

TVM_REGISTER_GLOBAL("xlstm.slstm_eval")
.set_body([](tvm::runtime::TVMArgs args, tvm::runtime::TVMRetValue* rv) {
    std::vector<TVMValue> values;
    std::vector<int> type_codes;
    RepackAsDLTensor(args, values, type_codes);
    xlstm_tvm_slstm_eval(
        values.data(), type_codes.data(),
        args.size(), nullptr, nullptr, nullptr);
});

TVM_REGISTER_GLOBAL("xlstm.mlstm_eval")
.set_body([](tvm::runtime::TVMArgs args, tvm::runtime::TVMRetValue* rv) {
    std::vector<TVMValue> values;
    std::vector<int> type_codes;
    RepackAsDLTensor(args, values, type_codes);
    xlstm_tvm_mlstm_eval(
        values.data(), type_codes.data(),
        args.size(), nullptr, nullptr, nullptr);
});
