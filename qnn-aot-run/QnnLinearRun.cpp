#include <iostream>
#include <dlfcn.h>
#include <cassert>
#include <vector>
#include <cstring>
#include <chrono>
#include <fstream>

#include "QnnSetup.h"
#include "QnnUtils.h"
#include "QnnSharedBuffer.h"

uint32_t batch_size = 32;
uint32_t input_shape = 4096 * 8;
uint32_t output_shape = 4096 * 8;
uint32_t num_iter = 10;

static constexpr uint32_t DEFAULT_ALIGNMENT = 8;
static constexpr bool ENABLE_SHARED_BUFFER = false;

std::string backend_lib_file = "libQnnHtp.so";
std::string system_lib_file = "libQnnSystem.so";
std::string context_bin_file = "LinearHtpContext.bin";

void load_context_binary(std::vector<uint8_t> &out_binary, uint32_t &out_binsize)
{
    std::ifstream bin(context_bin_file, std::ios::binary | std::ios::ate);
    assert(bin.is_open());

    size_t binSize = bin.tellg();
    bin.seekg(0);

    out_binary.resize(binSize);
    bin.read(reinterpret_cast<char *>(out_binary.data()), binSize);
    bin.close();

    out_binsize = binSize;

    printf("Loaded context binary: %zu bytes\n", binSize);
}

void export_profile_data(const std::string &OutputPath,
                         const QnnProfile_EventId_t &Event,
                         const QnnProfile_EventData_t &EventData)
{
    auto getUnit = [](QnnProfile_EventUnit_t Unit)
    {
        switch (Unit)
        {
        case QNN_PROFILE_EVENTUNIT_MICROSEC:
            return " (us)";
        case QNN_PROFILE_EVENTUNIT_BYTES:
            return " (bytes)";
        case QNN_PROFILE_EVENTUNIT_COUNT:
            return " (count)";
        case QNN_PROFILE_EVENTUNIT_BACKEND:
        case QNN_PROFILE_EVENTUNIT_CYCLES:
        default:
            return "";
        }
    };

    FILE *file = fopen(OutputPath.c_str(), "a+");

    std::string Identifier = std::string(EventData.identifier) + " " +
                             std::to_string(EventData.value) +
                             getUnit(EventData.unit);

    fprintf(file, "%s \n", Identifier.c_str());

    fclose(file);
}

template <typename INFO>
void prepare_tensors(INFO &info, std::vector<Qnn_Tensor_t> &out_input_tensors, std::vector<Qnn_Tensor_t> &out_output_tensors,
                     const QnnInterface_t *interface,
                     Qnn_ContextHandle_t context,
                     void *input_data,
                     void *output_data)
{
    Qnn_ErrorHandle_t err = QNN_SUCCESS;

    out_input_tensors.resize(info.numGraphInputs);
    out_output_tensors.resize(info.numGraphOutputs);

    for (uint32_t i = 0; i < info.numGraphInputs; i++)
    {
        uint16_t *input_data_fp16 = reinterpret_cast<uint16_t *>(input_data) + i;
        if (ENABLE_SHARED_BUFFER)
        {
            int32_t memfd = SharedBuffer::get_shared_buffer_manager().mem2fd(input_data_fp16);
            Qnn_MemDescriptor_t descriptor =
                {
                    {info.graphInputs[i].v2.rank, info.graphInputs[i].v2.dimensions, nullptr},
                    info.graphInputs[i].v2.dataType,
                    QNN_MEM_TYPE_ION,
                    {{memfd}}};

            Qnn_MemHandle_t mem_handle = nullptr;
            err = interface->QNN_INTERFACE_VER_NAME.memRegister(
                context,
                &descriptor,
                1,
                &mem_handle);

            if (err != QNN_SUCCESS)
            {
                printf("memRegister failed: %lu\n", err);
                return;
            }

            printf("Tensor rank: %u\n", info.graphInputs[i].v2.rank);
            printf("Tensor dimensions: %u x %u\n", info.graphInputs[i].v2.dimensions[0], info.graphInputs[i].v2.dimensions[1]);
            printf("Registered input tensor memhandle: %p\n", mem_handle);

            out_input_tensors[i] = info.graphInputs[i];
            out_input_tensors[i].v2.memHandle = mem_handle;
            out_input_tensors[i].v2.memType = QNN_TENSORMEMTYPE_MEMHANDLE;
        }
        else
        {
            out_input_tensors[i] = info.graphInputs[i];
            out_input_tensors[i].v2.memType = QNN_TENSORMEMTYPE_RAW;
            out_input_tensors[i].v2.clientBuf.data = input_data_fp16;
            out_input_tensors[i].v2.clientBuf.dataSize = batch_size * input_shape * sizeof(uint16_t);
        }
    }

    for (uint32_t i = 0; i < info.numGraphOutputs; i++)
    {
        if (ENABLE_SHARED_BUFFER)
        {
            uint16_t *output_data_fp16 = reinterpret_cast<uint16_t *>(output_data) + i;
            int32_t memfd = SharedBuffer::get_shared_buffer_manager().mem2fd(output_data_fp16);
            Qnn_MemDescriptor_t descriptor =
                {
                    {info.graphOutputs[i].v2.rank, info.graphOutputs[i].v2.dimensions, nullptr},
                    info.graphOutputs[i].v2.dataType,
                    QNN_MEM_TYPE_ION,
                    {{memfd}}};
            Qnn_MemHandle_t mem_handle = nullptr;
            err = interface->QNN_INTERFACE_VER_NAME.memRegister(
                context,
                &descriptor,
                1,
                &mem_handle);

            if (err != QNN_SUCCESS)
            {
                printf("memRegister failed: %lu\n", err);
                return;
            }

            printf("Tensor rank: %u\n", info.graphOutputs[i].v2.rank);
            printf("Tensor dimensions: %u x %u\n", info.graphOutputs[i].v2.dimensions[0], info.graphOutputs[i].v2.dimensions[1]);
            printf("Registered input tensor memhandle: %p\n", mem_handle);

            out_output_tensors[i] = info.graphOutputs[i];
            out_output_tensors[i].v2.memHandle = mem_handle;
            out_output_tensors[i].v2.memType = QNN_TENSORMEMTYPE_MEMHANDLE;
        }
        else
        {
            out_output_tensors[i] = info.graphOutputs[i];
            out_output_tensors[i].v2.memType = QNN_TENSORMEMTYPE_RAW;
            out_output_tensors[i].v2.clientBuf.data = reinterpret_cast<uint16_t *>(output_data) + i;
            out_output_tensors[i].v2.clientBuf.dataSize = batch_size * output_shape * sizeof(uint16_t);
        }
    }
}

int main(int argc, char **argv)
{
    printf("=======================================================\n");
    printf("Qnn Linear Load Smoke Test\n");
    printf("=======================================================\n");

    // parse_arg(argc, argv);

    void *handle;
    void *sys_handle;
    const QnnInterface_t *interface;
    const QnnSystemInterface_t *sys_interface;
    Qnn_LogHandle_t logger;
    Qnn_DeviceHandle_t device;
    Qnn_BackendHandle_t backend;
    Qnn_ContextHandle_t context;
    Qnn_GraphHandle_t graph;
    Qnn_ProfileHandle_t profile;

    QnnInit(backend_lib_file.c_str(),
            system_lib_file.c_str(),
            &handle,
            &sys_handle,
            &interface,
            &sys_interface,
            &logger,
            &device, &backend,
            nullptr, nullptr,
            &profile, false);
    {
        std::vector<uint8_t> bin_data;
        uint32_t bin_size = 0;
        load_context_binary(bin_data, bin_size);

        // Graph Info 를 Binary 로 부터 Load
        uint32_t num_graph = 0;
        QnnSystemContext_GraphInfo_t *graph_info = nullptr;
        const QnnSystemContext_BinaryInfo_t *binary_info = nullptr;
        std::string graph_name;
        QnnGetGraphInfoFromBinary(sys_interface, bin_data.data(), bin_size, &num_graph, &graph_info, &binary_info, graph_name);

        Qnn_ErrorHandle_t err = interface->QNN_INTERFACE_VER_NAME.contextCreateFromBinary(
            backend,
            device,
            nullptr,         // const QnnContext_Config_t** config
            bin_data.data(), // binaryBuffer
            (Qnn_ContextBinarySize_t)bin_size,
            &context,
            nullptr // Qnn_ProfileHandle_t profile
        );

        if (err != QNN_SUCCESS)
        {
            printf("contextCreateFromBinary failed: %lu\n", err);
            return -1;
        }

        printf("Context created from binary\n");
        Qnn_GraphHandle_t graph;
        err = interface->QNN_INTERFACE_VER_NAME.graphRetrieve(context, graph_name.c_str(), &graph);
        if (err != QNN_SUCCESS)
        {
            printf("graphRetrieve failed: %lu\n", err);
            return -1;
        }

        printf("Graph retrieved from context\n");
        QnnGraph_Property_t *graph_property = nullptr;
        err = interface->QNN_INTERFACE_VER_NAME.graphGetProperty(graph, &graph_property);
        if (err != QNN_SUCCESS)
        {
            printf("graphGetProperty failed: %lu\n", err);
            return -1;
        }

        printf("Graph properties from graph\n");
        void *input_data = nullptr;
        void *output_data = nullptr;

        if (ENABLE_SHARED_BUFFER)
        {
            input_data = SharedBuffer::get_shared_buffer_manager().allocmem(batch_size * input_shape * sizeof(uint16_t), DEFAULT_ALIGNMENT);
            output_data = SharedBuffer::get_shared_buffer_manager().allocmem(batch_size * output_shape * sizeof(uint16_t), DEFAULT_ALIGNMENT);
        }
        else
        {
            input_data = aligned_alloc(DEFAULT_ALIGNMENT, batch_size * input_shape * sizeof(uint16_t));
            output_data = aligned_alloc(DEFAULT_ALIGNMENT, batch_size * output_shape * sizeof(uint16_t));
        }

        printf("Fill input data\n");
        uint16_t *input_data_uint16 = reinterpret_cast<uint16_t *>(input_data);
        for (uint32_t i = 0; i < batch_size * input_shape; i++)
        {
            input_data_uint16[i] = fp32_to_fp16(1.0f);
        }

        printf("Prepare input/output tensors\n");
        std::vector<Qnn_Tensor_t> inputTensors;
        std::vector<Qnn_Tensor_t> outputTensors;

        if (binary_info->version == QNN_SYSTEM_CONTEXT_BINARY_INFO_VERSION_1)
        {
            printf("Using GRAPH_INFO_VERSION_1\n");
            prepare_tensors(graph_info->graphInfoV1, inputTensors, outputTensors,
                            interface, context, input_data, output_data);
        }
        else if (binary_info->version == QNN_SYSTEM_CONTEXT_BINARY_INFO_VERSION_2)
        {
            printf("Using GRAPH_INFO_VERSION_2\n");
            prepare_tensors(graph_info->graphInfoV2, inputTensors, outputTensors,
                            interface, context, input_data, output_data);
        }
#if (QNN_API_VERSION_MAJOR >= 2 && QNN_API_VERSION_MINOR >= 21)
        else if (binary_info->version == QNN_SYSTEM_CONTEXT_BINARY_INFO_VERSION_3)
        {
            printf("Using GRAPH_INFO_VERSION_3\n");
            prepare_tensors(graph_info->graphInfoV3, inputTensors, outputTensors,
                            interface, context, input_data, output_data);
        }
#endif

        // Execute graph
        for (uint32_t i = 0; i < num_iter; i++)
        {
            auto start = std::chrono::high_resolution_clock::now();
            err = interface->QNN_INTERFACE_VER_NAME.graphExecute(
                graph,
                inputTensors.data(),  // input tensors
                inputTensors.size(),  // number of inputs
                outputTensors.data(), // output tensors
                outputTensors.size(), // number of outputs
                profile,              // profile
                nullptr               // const QnnExecution_Config_t** executionConfig
            );
            auto end = std::chrono::high_resolution_clock::now();
            auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
            printf("Graph executed successfully\n");
            printf("Graph executed in %lld ms\n", duration);

            if (err != QNN_SUCCESS)
            {
                printf("graphExecute failed: %lu\n", err);
                return -1;
            }

            // profile
            const QnnProfile_EventId_t *events = nullptr;
            const QnnProfile_EventId_t *subEvents = nullptr;
            uint32_t numEvents = 0;
            uint32_t numSubEvents = 0;
            err = interface->QNN_INTERFACE_VER_NAME.profileGetEvents(profile, &events, &numEvents);

            if (err != QNN_SUCCESS)
            {
                printf("profileGetEvents failed: %lu\n", err);
                return -1;
            }

            QnnProfile_EventData_t eventData;
            for (uint32_t i = 0; i < numEvents; i++)
            {
                err = interface->QNN_INTERFACE_VER_NAME.profileGetEventData(events[i], &eventData);
                if (err != QNN_SUCCESS)
                {
                    printf("profileGetEventData failed: %lu\n", err);
                    return -1;
                }

                export_profile_data("qnn_profile_data.txt", events[i], eventData);
            }
        }

        // Print output
        printf("Output values:\n");
        uint16_t *output_data_uint16 = reinterpret_cast<uint16_t *>(output_data);
        for (uint32_t i = 0; i < batch_size * output_shape; i++)
        {
            printf("  [%u]: %f\n", i, fp16_to_fp32(output_data_uint16[i]));
        }

        if (ENABLE_SHARED_BUFFER)
        {
            SharedBuffer::get_shared_buffer_manager().freemem(input_data);
            SharedBuffer::get_shared_buffer_manager().freemem(output_data);
        }
        else
        {
            free(input_data);
            free(output_data);
        }

        input_data = nullptr;
        output_data = nullptr;
    }

    QnnCleanup(handle, sys_handle);

    return 0;
}