#ifndef __QNN_SETUP_H__
#define __QNN_SETUP_H__

#include "QnnInterface.h"
#include "System/QnnSystemInterface.h"
#include <string>

void QnnInit(const char *backend_path,
             const char *system_path,
             void **out_handle,
             void **out_sys_handle,
             const QnnInterface_t **out_interface,
             const QnnSystemInterface_t **out_sys_interface,
             Qnn_LogHandle_t *out_logger,
             Qnn_DeviceHandle_t *out_device,
             Qnn_BackendHandle_t *out_backend,
             Qnn_ContextHandle_t *out_context = nullptr,
             Qnn_GraphHandle_t *out_graph = nullptr,
             Qnn_ProfileHandle_t *out_profile = nullptr,
             bool is_aot = true);

void QnnGetGraphInfoFromBinary(const QnnSystemInterface_t *sys_interface,
                               void *buffer,
                               uint32_t n_bytes,
                               uint32_t *out_num_graphs,
                               QnnSystemContext_GraphInfo_t **out_graph,
                               const QnnSystemContext_BinaryInfo_t **out_binary_info,
                               std::string &out_graph_name);

void QnnCleanup(void *handle, void *sys_handle);

#endif