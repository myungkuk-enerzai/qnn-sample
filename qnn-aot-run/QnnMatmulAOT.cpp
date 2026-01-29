#include <iostream>
#include <dlfcn.h>
#include <cassert>
#include <vector>
#include <cstring>
#include <chrono>

#include "QnnSetup.h"
#include "QnnUtils.h"

uint32_t batch_size = 16;
uint32_t input_shape = 4096 * 8;
uint32_t output_shape = 4096 * 8;

int main(int argc, char **argv)
{
    printf("=======================================================\n");
    printf("Qnn Linear Compile Smoke Test\n");
    printf("=======================================================\n");

    // parse_arg(argc, argv);

    void *handle;
    const QnnInterface_t *interface;
    Qnn_LogHandle_t logger;
    Qnn_DeviceHandle_t device;
    Qnn_BackendHandle_t backend;
    Qnn_ContextHandle_t context;
    Qnn_GraphHandle_t graph;

    QnnInit("libQnnHtp.so", nullptr, &handle, nullptr, &interface, nullptr, &logger, &device, &backend, &context, &graph);
    {
        Qnn_Tensor_t input = QNN_TENSOR_INIT;
        Qnn_Tensor_t weight = QNN_TENSOR_INIT;
        Qnn_Tensor_t output = QNN_TENSOR_INIT;

        uint32_t in_dims[] = {batch_size, input_shape};
        uint32_t w_dims[] = {input_shape, output_shape};
        uint32_t out_dims[] = {batch_size, output_shape};

        input.v1.id = 1;
        input.v1.name = "input";
        input.v1.type = QNN_TENSOR_TYPE_APP_WRITE;
        input.v1.dataFormat = QNN_TENSOR_DATA_FORMAT_DENSE;
        input.v1.dataType = QNN_DATATYPE_FLOAT_16;
        input.v1.quantizeParams = QNN_QUANTIZE_PARAMS_INIT;
        input.v1.rank = 2;
        input.v1.dimensions = in_dims;
        input.v1.memType = QNN_TENSORMEMTYPE_RAW;
        input.v1.clientBuf.data = nullptr;
        input.v1.clientBuf.dataSize = 0;
        interface->QNN_INTERFACE_VER_NAME.tensorCreateGraphTensor(graph, &input);

        std::vector<uint16_t> weight_data(output_shape * input_shape, fp32_to_fp16(1.0f)); // FP16 = 1.0
        weight.v1.id = 2;
        weight.v1.name = "weight";
        weight.v1.type = QNN_TENSOR_TYPE_STATIC;
        weight.v1.dataFormat = QNN_TENSOR_DATA_FORMAT_DENSE;
        weight.v1.dataType = QNN_DATATYPE_FLOAT_16;
        weight.v1.quantizeParams = QNN_QUANTIZE_PARAMS_INIT;
        weight.v1.rank = 2;
        weight.v1.dimensions = w_dims;
        weight.v1.memType = QNN_TENSORMEMTYPE_RAW;
        weight.v1.clientBuf.data = weight_data.data();
        weight.v1.clientBuf.dataSize = weight_data.size() * sizeof(uint16_t);
        interface->QNN_INTERFACE_VER_NAME.tensorCreateGraphTensor(graph, &weight);

        output.v1.id = 3;
        output.v1.name = "output";
        output.v1.type = QNN_TENSOR_TYPE_APP_READ;
        output.v1.dataFormat = QNN_TENSOR_DATA_FORMAT_DENSE;
        output.v1.dataType = QNN_DATATYPE_FLOAT_16;
        output.v1.quantizeParams = QNN_QUANTIZE_PARAMS_INIT;
        output.v1.rank = 2;
        output.v1.dimensions = out_dims;
        output.v1.memType = QNN_TENSORMEMTYPE_RAW;
        output.v1.clientBuf.data = nullptr;
        output.v1.clientBuf.dataSize = 0;
        interface->QNN_INTERFACE_VER_NAME.tensorCreateGraphTensor(graph, &output);

        // add op
        Qnn_Tensor_t inputs[] = {input, weight};

        Qnn_OpConfig_t matmul_op = QNN_OPCONFIG_INIT;
        matmul_op.v1.packageName = QNN_OP_PACKAGE_NAME_QTI_AISW;
        matmul_op.v1.typeName = QNN_OP_MAT_MUL;
        matmul_op.v1.name = "matmul";
        matmul_op.v1.inputTensors = inputs;
        matmul_op.v1.numOfInputs = 2;
        matmul_op.v1.outputTensors = &output;
        matmul_op.v1.numOfOutputs = 1;
        matmul_op.v1.params = nullptr;
        matmul_op.v1.numOfParams = 0;
        interface->QNN_INTERFACE_VER_NAME.graphAddNode(graph, matmul_op);

        // finalize grap
        interface->QNN_INTERFACE_VER_NAME.graphFinalize(graph, nullptr, nullptr);

        // get binary size
        uint64_t binarySize = 0;
        Qnn_ErrorHandle_t err;

        err = interface->QNN_INTERFACE_VER_NAME.contextGetBinarySize(
            context,
            &binarySize);

        if (err != QNN_SUCCESS || binarySize == 0)
        {
            printf("contextGetBinarySize failed (%lu), size=%lu\n", err, binarySize);
            return -1;
        }

        printf("Graph binary size: %lu bytes\n", binarySize);

        // get binary
        uint64_t writtenSize = 0;
        std::vector<uint8_t> binary(binarySize);

        err = interface->QNN_INTERFACE_VER_NAME.contextGetBinary(
            context,
            binary.data(),
            binarySize,
            &writtenSize);

        if (err != QNN_SUCCESS)
        {
            printf("contextGetBinary failed (%lu)\n", err);
            return -1;
        }

        printf("Graph binary retrieved, written size: %lu bytes\n", writtenSize);

        FILE *fp = fopen("MatmulHtpContext.bin", "wb");
        assert(fp != nullptr);
        fwrite(binary.data(), 1, binarySize, fp);
        fclose(fp);

        printf("Complete context binary written\n");
    }

    QnnCleanup(handle, nullptr);

    return 0;
}