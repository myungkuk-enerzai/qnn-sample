#include <iostream>
#include <dlfcn.h>
#include <cassert>
#include <vector>
#include <cstring>
#include <chrono>
#include <fstream>

#include "QnnSetup.h"
#include "QnnUtils.h"

uint32_t batch_size = 16;
uint32_t input_shape = 4096 * 8;
uint32_t output_shape = 4096 * 8;
uint32_t num_iter = 10;

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

    QnnInit("libQnnHtp.so",
            "libQnnHtpSystem.so",
            &handle,
            &sys_handle,
            &interface,
            &sys_interface,
            &logger,
            &device, &backend,
            nullptr, nullptr,
            &profile, false);
    {
        std::ifstream bin("MatmulHtpContext.bin", std::ios::binary | std::ios::ate);
        assert(bin.is_open());

        size_t binSize = bin.tellg();
        bin.seekg(0);

        std::vector<uint8_t> binData(binSize);
        bin.read(reinterpret_cast<char *>(binData.data()), binSize);
        bin.close();

        printf("Loaded context binary: %zu bytes\n", binSize);

        uint32_t num_graph = 0;
        QnnSystemContext_GraphInfo_t *graph_info = nullptr;
        const QnnSystemContext_BinaryInfo_t *binary_info = nullptr;
        QnnGetGraphInfoFromBinary(sys_interface, binData.data(), binSize, &num_graph, &graph_info, &binary_info);

        assert(num_graph);

        Qnn_ErrorHandle_t err =
            interface->QNN_INTERFACE_VER_NAME.contextCreateFromBinary(
                backend,
                device,
                nullptr,        // const QnnContext_Config_t** config
                binData.data(), // binaryBuffer
                (Qnn_ContextBinarySize_t)binSize,
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

        err = interface->QNN_INTERFACE_VER_NAME.graphRetrieve(context, "NAME", &graph);
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

        printf("Graph properties from grpah\n");

        // Prepare input/output tensors
        std::vector<uint16_t> inputData(batch_size * input_shape, fp32_to_fp16(1.0f));
        std::vector<uint16_t> outputData(batch_size * output_shape, fp32_to_fp16(0.0f));

        std::vector<Qnn_Tensor_t> inputTensors;
        std::vector<Qnn_Tensor_t> outputTensors;

        if (binary_info->version == QNN_SYSTEM_CONTEXT_BINARY_INFO_VERSION_1)
        {
            inputTensors.resize(graph_info->graphInfoV1.numGraphInputs);
            outputTensors.resize(graph_info->graphInfoV1.numGraphOutputs);
            for (uint32_t i = 0; i < graph_info->graphInfoV1.numGraphInputs; i++)
            {
                inputTensors[i] = graph_info->graphInfoV1.graphInputs[i];
            }
            for (uint32_t i = 0; i < graph_info->graphInfoV1.numGraphOutputs; i++)
            {
                outputTensors[i] = graph_info->graphInfoV1.graphOutputs[i];
            }
        }
        else if (binary_info->version == QNN_SYSTEM_CONTEXT_BINARY_INFO_VERSION_2)
        {
            inputTensors.resize(graph_info->graphInfoV2.numGraphInputs);
            outputTensors.resize(graph_info->graphInfoV2.numGraphOutputs);
            for (uint32_t i = 0; i < graph_info->graphInfoV2.numGraphInputs; i++)
            {
                inputTensors[i] = graph_info->graphInfoV2.graphInputs[i];
            }
            for (uint32_t i = 0; i < graph_info->graphInfoV2.numGraphOutputs; i++)
            {
                outputTensors[i] = graph_info->graphInfoV2.graphOutputs[i];
            }
        }
#if (QNN_API_VERSION_MAJOR >= 2 && QNN_API_VERSION_MINOR >= 21)
        else if (binary_info->version == QNN_SYSTEM_CONTEXT_BINARY_INFO_VERSION_3)
        {
            inputTensors.resize(graph_info->graphInfoV3.numGraphInputs);
            outputTensors.resize(graph_info->graphInfoV3.numGraphOutputs);
            for (uint32_t i = 0; i < graph_info->graphInfoV3.numGraphInputs; i++)
            {
                inputTensors[i] = graph_info->graphInfoV3.graphInputs[i];
            }
            for (uint32_t i = 0; i < graph_info->graphInfoV3.numGraphOutputs; i++)
            {
                outputTensors[i] = graph_info->graphInfoV3.graphOutputs[i];
            }
        }
#endif

        for (auto &tensor : inputTensors)
        {
            tensor.version = QNN_TENSOR_VERSION_2;
            tensor.v2.clientBuf.data = inputData.data();
            tensor.v2.clientBuf.dataSize = (inputData.size() * sizeof(uint16_t));
        }

        for (auto &tensor : outputTensors)
        {
            tensor.version = QNN_TENSOR_VERSION_2;
            tensor.v2.clientBuf.data = outputData.data();
            tensor.v2.clientBuf.dataSize = (outputData.size() * sizeof(uint16_t));
        }

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
        for (uint32_t i = 0; i < batch_size * output_shape; i++)
        {
            printf("  [%u]: %f\n", i, fp16_to_fp32(outputData[i]));
        }
    }

    QnnCleanup(handle, sys_handle);

    return 0;
}