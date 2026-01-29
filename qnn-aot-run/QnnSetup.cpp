#include "QnnSetup.h"
#include "HTP/QnnHtpDevice.h"
#include <dlfcn.h>
#include <cstdio>
#include <vector>
#include <array>
#include <cassert>

typedef Qnn_ErrorHandle_t (*QnnInterfaceGetProvidersFn_t)(const QnnInterface_t ***providerList, uint32_t *numProviders);
typedef Qnn_ErrorHandle_t (*QnnSystemInterfaceGetProvidersFn_t)(const QnnSystemInterface_t ***providerList, uint32_t *numProviders);

static void QnnLogHandler(const char *fmt, QnnLog_Level_t LogLevel, uint64_t Timestamp, va_list args)
{
    vfprintf(stdout, fmt, args);
}

void QnnInit(const char *backend_path,
             const char *system_path,
             void **out_handle,
             void **out_sys_handle,
             const QnnInterface_t **out_interface,
             const QnnSystemInterface_t **out_sys_interface,
             Qnn_LogHandle_t *out_logger,
             Qnn_DeviceHandle_t *out_device,
             Qnn_BackendHandle_t *out_backend,
             Qnn_ContextHandle_t *out_context,
             Qnn_GraphHandle_t *out_graph,
             Qnn_ProfileHandle_t *out_profile,
             bool is_aot)
{
    void *handle = dlopen(backend_path, RTLD_NOW | RTLD_LOCAL);

    if (handle == nullptr)
        printf("error: %s\n", dlerror());

    auto *provider = (QnnInterfaceGetProvidersFn_t)dlsym(handle, "QnnInterface_getProviders");
    if (provider == nullptr)
    {
        dlclose(handle);
        printf("error: %s\n", dlerror());
    }

    const QnnInterface_t **providers = nullptr;
    uint32_t provider_count = 0;

    if (provider(&providers, &provider_count) != QNN_SUCCESS)
    {
        dlclose(handle);
        printf("error: QNN returned error\n");
    }

    // create interface
    const QnnInterface_t *selected = nullptr;
    for (auto i = 0u; i < provider_count; ++i)
    {
        auto *prov = providers[i];

        auto &backVer = prov->apiVersion.backendApiVersion;
        auto &coreVer = prov->apiVersion.coreApiVersion;

        printf("QNN Interface - Provider: %s, Backend ID: %u, Backend Version: %u.%u.%u, CoreAPI Version: %u.%u.%u\n",
               prov->providerName, prov->backendId,
               backVer.major, backVer.minor, backVer.patch,
               coreVer.major, coreVer.minor, coreVer.patch);

        selected = prov;
    }

    // create logger
    Qnn_LogHandle_t logger = nullptr;
    if (selected->QNN_INTERFACE_VER_NAME.logCreate(&QnnLogHandler, QNN_LOG_LEVEL_DEBUG, &logger) != QNN_SUCCESS)
    {
        dlclose(handle);
        printf("error: QNN returned error\n");
    }

    // test log
    if (logger == nullptr)
    {
        dlclose(handle);
        printf("error: Logger is null\n");
    }

    // create backend
    const QnnBackend_Config_t *backend_config = nullptr;
    Qnn_BackendHandle_t backend;
    if (selected->QNN_INTERFACE_VER_NAME.backendCreate(logger, &backend_config, &backend) != QNN_SUCCESS)
    {
        dlclose(handle);
        printf("error: QNN returned error\n");
    }

    /**
     * executorch 참고하여 device config 설정: HtpDevice.cpp:295
     */
    std::vector<const QnnDevice_Config_t *> result_config;
    std::vector<QnnDevice_Config_t> device_configs;
    std::vector<QnnDevice_CustomConfig_t> custom_device_configs = {};
    QnnHtpDevice_CustomConfig_t *htp_config = nullptr;

    htp_config = new QnnHtpDevice_CustomConfig_t;
    htp_config->option = QNN_HTP_DEVICE_CONFIG_OPTION_ARCH;
    htp_config->arch.deviceId = 0;
    htp_config->arch.arch = QNN_HTP_DEVICE_ARCH_V73;
    custom_device_configs.push_back(static_cast<QnnDevice_CustomConfig_t>(htp_config));

    device_configs.resize(custom_device_configs.size());
    for (size_t i = 0; i < custom_device_configs.size(); i++)
    {
        device_configs[i].option = QNN_DEVICE_CONFIG_OPTION_CUSTOM;
        device_configs[i].customConfig = custom_device_configs[i];
        result_config.push_back(&device_configs[i]);
    }

    result_config.push_back(nullptr);

    Qnn_DeviceHandle_t device = nullptr;
    if (selected->QNN_INTERFACE_VER_NAME.deviceCreate(logger, result_config.data(), &device) != QNN_SUCCESS)
    {
        dlclose(handle);
        printf("error: QNN returned error\n");
    }

    if (is_aot)
    {
        // create context
        const QnnContext_Config_t *context_config = nullptr;
        Qnn_ContextHandle_t context;
        if (selected->QNN_INTERFACE_VER_NAME.contextCreate(backend, device, &context_config, &context) != QNN_SUCCESS)
        {
            dlclose(handle);
            printf("error: QNN returned error\n");
        }

        // compose graph
        const QnnGraph_Config_t *graph_config = nullptr;
        Qnn_GraphHandle_t graph;
        if (selected->QNN_INTERFACE_VER_NAME.graphCreate(context, "NAME", &graph_config, &graph) != QNN_SUCCESS)
        {
            dlclose(handle);
            printf("error: QNN returned error\n");
        }

        // Runner 의 경우 외부에서 생성함
        *out_context = context;
        *out_graph = graph;
        ////
    }
    else
    {
        // create sys interface
        void *sys_handle = dlopen(system_path, RTLD_NOW | RTLD_GLOBAL);

        printf("sys_handle: %p\n", sys_handle);

        if (sys_handle == nullptr)
            printf("error: %s\n", dlerror());

        auto *sys_provider = (QnnSystemInterfaceGetProvidersFn_t)dlsym(sys_handle, "QnnSystemInterface_getProviders");
        if (sys_provider == nullptr)
        {
            dlclose(sys_handle);
            printf("error: %s\n", dlerror());
        }

        const QnnSystemInterface_t **sys_providers = nullptr;
        uint32_t sys_provider_count = 0;

        if (sys_provider(&sys_providers, &sys_provider_count) != QNN_SUCCESS)
        {
            dlclose(handle);
            printf("error: QNN returned error\n");
        }

        const QnnSystemInterface_t *sys_selected = nullptr;
        sys_selected = sys_providers[0];

        Qnn_ProfileHandle_t profile;
        QnnProfile_Level_t profileLevel = QNN_PROFILE_LEVEL_DETAILED;
        if (selected->QNN_INTERFACE_VER_NAME.profileCreate(backend, profileLevel, &profile) != QNN_SUCCESS)
        {
            dlclose(handle);
            printf("error: QNN returned error\n");
        }

        QnnProfile_Config_t profileConfig = QNN_PROFILE_CONFIG_INIT;
        profileConfig.option = QNN_PROFILE_CONFIG_OPTION_ENABLE_OPTRACE;

        std::array<const QnnProfile_Config_t *, 2> profileConfigs = {&profileConfig, nullptr};

        if (selected->QNN_INTERFACE_VER_NAME.profileSetConfig(profile, profileConfigs.data()) != QNN_SUCCESS)
        {
            dlclose(handle);
            printf("error: QNN returned error\n");
        }

        *out_sys_handle = sys_handle;
        *out_sys_interface = sys_selected;

        // AOT 가 아닐 경우 profile 옵션 활설화
        *out_profile = profile;
    }

    *out_handle = handle;
    *out_interface = selected;
    *out_logger = logger;
    *out_device = device;
    *out_backend = backend;
}

void QnnGetGraphInfoFromBinary(const QnnSystemInterface_t *sys_interface,
                               void *buffer,
                               uint32_t n_bytes,
                               uint32_t *out_num_graphs,
                               QnnSystemContext_GraphInfo_t **out_graph,
                               const QnnSystemContext_BinaryInfo_t **out_binary_info,
                               std::string &out_graph_name)
{
    QnnSystemContext_Handle_t sys_context_handle = nullptr;
    Qnn_ErrorHandle_t error = sys_interface->QNN_SYSTEM_INTERFACE_VER_NAME.systemContextCreate(&sys_context_handle);
    if (error != QNN_SUCCESS)
    {
        // Handle error
        printf("Error systemContextCreate()\n");
    }

    const QnnSystemContext_BinaryInfo_t *binary_info = nullptr;
    Qnn_ContextBinarySize_t binary_info_size = 0;

    error = sys_interface->QNN_SYSTEM_INTERFACE_VER_NAME.systemContextGetBinaryInfo(
        sys_context_handle, buffer, n_bytes, &binary_info, &binary_info_size);

    if (error != QNN_SUCCESS)
    {
        printf("Error systemContextGetBinaryInfo()\n");
    }

    uint32_t num_graphs = 0;
    QnnSystemContext_GraphInfo_t *graphs = nullptr;

    if (binary_info->version == QNN_SYSTEM_CONTEXT_BINARY_INFO_VERSION_1)
    {
        num_graphs = binary_info->contextBinaryInfoV1.numGraphs;
        graphs = binary_info->contextBinaryInfoV1.graphs;
        out_graph_name = graphs[0].graphInfoV1.graphName;
    }
    else if (binary_info->version == QNN_SYSTEM_CONTEXT_BINARY_INFO_VERSION_2)
    {
        num_graphs = binary_info->contextBinaryInfoV2.numGraphs;
        graphs = binary_info->contextBinaryInfoV2.graphs;
        out_graph_name = graphs[0].graphInfoV2.graphName;
    }
#if (QNN_API_VERSION_MAJOR >= 2 && QNN_API_VERSION_MINOR >= 21)
    else if (binary_info->version == QNN_SYSTEM_CONTEXT_BINARY_INFO_VERSION_3)
    {
        num_graphs = binary_info->contextBinaryInfoV3.numGraphs;
        graphs = binary_info->contextBinaryInfoV3.graphs;
        out_graph_name = graphs[0].graphInfoV3.graphName;
    }
#endif
    else
    {
        assert(0);
    }

    *out_binary_info = binary_info;
    *out_num_graphs = num_graphs;
    *out_graph = graphs;
}

void QnnCleanup(void *handle, void *sys_handle)
{
    if (sys_handle)
        dlclose(sys_handle);
    if (handle)
        dlclose(handle);
}
