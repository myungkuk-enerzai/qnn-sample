#include <iostream>
#include <dlfcn.h>
#include <cassert>
#include <vector>
#include <cstring>
#include <chrono>
#include "QnnInterface.h"
#include "HTP/QnnHtpDevice.h"

#define USING_F16

typedef Qnn_ErrorHandle_t (*QnnInterfaceGetProvidersFn_t)(const QnnInterface_t ***providerList, uint32_t *numProviders);
void qnn_log_handler(const char *fmt, QnnLog_Level_t LogLevel, uint64_t Timestamp, va_list args)
{
	vfprintf(stdout, fmt, args);
}

uint16_t fp32_to_fp16(float f)
{
	uint32_t x = *(uint32_t *)&f;
	uint16_t h = ((x >> 16) & 0x8000) |
				 ((((x & 0x7f800000) - 0x38000000) >> 13) & 0x7c00) |
				 ((x >> 13) & 0x03ff);
	return h;
}

float fp16_to_fp32(uint16_t h)
{
	uint32_t sign = (h & 0x8000) << 16;
	uint32_t exp = (h & 0x7c00) >> 10;
	uint32_t mant = (h & 0x03ff);

	uint32_t f;
	if (exp == 0)
		f = sign;
	else
		f = sign | ((exp + 112) << 23) | (mant << 13);

	float out;
	memcpy(&out, &f, sizeof(out));
	return out;
}

int test_linear()
{
	const char *backend = "libQnnHtp.so";
	void *handle = dlopen(backend, RTLD_NOW | RTLD_LOCAL);

	if (handle == nullptr)
	{
		printf("error: %s\n", dlerror());
		return 1;
	}

	auto *provider = (QnnInterfaceGetProvidersFn_t)dlsym(handle, "QnnInterface_getProviders");
	if (provider == nullptr)
	{
		dlclose(handle);
		printf("error: %s\n", dlerror());
		return 1;
	}

	const QnnInterface_t **providers = nullptr;
	uint32_t provider_count = 0;

	if (provider(&providers, &provider_count) != QNN_SUCCESS)
	{
		dlclose(handle);
		printf("error: QNN returned error\n");
		return 1;
	}

	auto isCapable = [](auto &ver)
	{
		return ver.major == QNN_API_VERSION_MAJOR && ver.minor >= QNN_API_VERSION_MINOR;
	};

	const QnnInterface_t *Selected = nullptr;
	for (auto i = 0u; i < provider_count; ++i)
	{
		auto *prov = providers[i];

		auto &backVer = prov->apiVersion.backendApiVersion;
		auto &coreVer = prov->apiVersion.coreApiVersion;

		printf("QNN Interface - Provider: %s, Backend ID: %u, Backend Version: %u.%u.%u, CoreAPI Version: %u.%u.%u\n",
			   prov->providerName, prov->backendId,
			   backVer.major, backVer.minor, backVer.patch,
			   coreVer.major, coreVer.minor, coreVer.patch);

		Selected = prov;
	}

	// create logger
	Qnn_LogHandle_t Logger = nullptr;
	if (Selected->QNN_INTERFACE_VER_NAME.logCreate(&qnn_log_handler, QNN_LOG_LEVEL_DEBUG, &Logger) != QNN_SUCCESS)
	{
		dlclose(handle);
		printf("error: QNN returned error\n");
		return 1;
	}

	// test log
	if (Logger == nullptr)
	{
		dlclose(handle);
		printf("error: Logger is null\n");
		return 1;
	}

	// create backend
	const QnnBackend_Config_t *BackendConfig = nullptr;
	Qnn_BackendHandle_t Backend;
	if (Selected->QNN_INTERFACE_VER_NAME.backendCreate(Logger, &BackendConfig, &Backend) != QNN_SUCCESS)
	{
		dlclose(handle);
		printf("error: QNN returned error\n");
		return 1;
	}

	/**
	 * executorch 참고하여 device config 설정: HtpDevice.cpp:295
	 */
	std::vector<const QnnDevice_Config_t *> ResultConfig;
	std::vector<QnnDevice_Config_t> DeviceConfigs;
	std::vector<QnnDevice_CustomConfig_t> CustomDeviceConfigs = {};
	QnnHtpDevice_CustomConfig_t *HtpConfig = nullptr;

	HtpConfig = new QnnHtpDevice_CustomConfig_t;
	HtpConfig->option = QNN_HTP_DEVICE_CONFIG_OPTION_ARCH;
	HtpConfig->arch.deviceId = 0;
	HtpConfig->arch.arch = QNN_HTP_DEVICE_ARCH_V73;
	CustomDeviceConfigs.push_back(static_cast<QnnDevice_CustomConfig_t>(HtpConfig));

	DeviceConfigs.resize(CustomDeviceConfigs.size());
	for (size_t i = 0; i < CustomDeviceConfigs.size(); i++)
	{
		DeviceConfigs[i].option = QNN_DEVICE_CONFIG_OPTION_CUSTOM;
		DeviceConfigs[i].customConfig = CustomDeviceConfigs[i];
		ResultConfig.push_back(&DeviceConfigs[i]);
	}

	ResultConfig.push_back(nullptr);

	Qnn_DeviceHandle_t Device = nullptr;
	if (Selected->QNN_INTERFACE_VER_NAME.deviceCreate(Logger, ResultConfig.data(), &Device) != QNN_SUCCESS)
	{
		dlclose(handle);
		printf("error: QNN returned error\n");
		return 1;
	}

	// create context
	const QnnContext_Config_t *ContextConfig = nullptr;
	Qnn_ContextHandle_t Context;
	if (Selected->QNN_INTERFACE_VER_NAME.contextCreate(Backend, Device, &ContextConfig, &Context) != QNN_SUCCESS)
	{
		dlclose(handle);
		printf("error: QNN returned error\n");
		return 1;
	}

	// compose graph
	// Create graph handle
	const QnnGraph_Config_t *GraphConfig = nullptr;
	Qnn_GraphHandle_t Graph;
	if (Selected->QNN_INTERFACE_VER_NAME.graphCreate(Context, "NAME", &GraphConfig, &Graph) != QNN_SUCCESS)
	{
		dlclose(handle);
		printf("error: QNN returned error\n");
		return 1;
	}

	constexpr uint32_t Batch = 1;
	constexpr uint32_t InputShape = 32;
	constexpr uint32_t OutputShape = 32;

	// Add graph tensor
	Qnn_Tensor_t Input = QNN_TENSOR_INIT;
	Qnn_Tensor_t Weight = QNN_TENSOR_INIT;
	Qnn_Tensor_t Bias = QNN_TENSOR_INIT;
	Qnn_Tensor_t Output = QNN_TENSOR_INIT;
	Qnn_Tensor_t Mid0 = QNN_TENSOR_INIT;
	Qnn_Tensor_t Mid1 = QNN_TENSOR_INIT;
	uint32_t InDims[] = {Batch, InputShape};
	uint32_t WeightDims[] = {OutputShape, InputShape};
	uint32_t BiasDims[] = {OutputShape};
	uint32_t OutDims[] = {Batch, OutputShape};
	uint32_t MidDims[] = {Batch, OutputShape};

	constexpr uint32_t WeightCount = OutputShape * InputShape;
#ifdef USING_F16
	static uint16_t WegihtData[WeightCount];
#else
	static float WegihtData[WeightCount];
#endif
	for (uint32_t i = 0; i < WeightCount; i++)
	{
#ifdef USING_F16
		WegihtData[i] = fp32_to_fp16(0.01f);
#else
		WegihtData[i] = 0.01f;
#endif
	}

	constexpr uint32_t BiasCount = OutputShape;
#ifdef USING_F16
	static uint16_t BiasData[BiasCount];
#else
	static float BiasData[BiasCount];
#endif
	for (uint32_t i = 0; i < BiasCount; i++)
	{
#ifdef USING_F16
		BiasData[i] = fp32_to_fp16(1.0f);
#else
		BiasData[i] = 1.0f;
#endif
	}

	Input.v1.id = 0;
	Input.v1.name = "input";
	Input.v1.type = QNN_TENSOR_TYPE_APP_WRITE;
	Input.v1.dataFormat = QNN_TENSOR_DATA_FORMAT_DENSE;
#ifdef USING_F16
	Input.v1.dataType = QNN_DATATYPE_FLOAT_16;
#else
	Input.v1.dataType = QNN_DATATYPE_FLOAT_32;
#endif
	Input.v1.quantizeParams = QNN_QUANTIZE_PARAMS_INIT;
	Input.v1.rank = 2;
	Input.v1.dimensions = InDims;
	Input.v1.memType = QNN_TENSORMEMTYPE_RAW;
	Input.v1.clientBuf.data = nullptr;
	Input.v1.clientBuf.dataSize = 0;
	Selected->QNN_INTERFACE_VER_NAME.tensorCreateGraphTensor(Graph, &Input);

	Output.v1.id = 1;
	Output.v1.name = "output";
	Output.v1.type = QNN_TENSOR_TYPE_APP_READ;
	Output.v1.dataFormat = QNN_TENSOR_DATA_FORMAT_DENSE;
#ifdef USING_F16
	Output.v1.dataType = QNN_DATATYPE_FLOAT_16;
#else
	Output.v1.dataType = QNN_DATATYPE_FLOAT_32;
#endif
	Output.v1.quantizeParams = QNN_QUANTIZE_PARAMS_INIT;
	Output.v1.rank = 2;
	Output.v1.dimensions = OutDims;
	Output.v1.memType = QNN_TENSORMEMTYPE_RAW;
	Output.v1.clientBuf.data = nullptr;
	Output.v1.clientBuf.dataSize = 0;
	Selected->QNN_INTERFACE_VER_NAME.tensorCreateGraphTensor(Graph, &Output);

	Weight.v1.id = 2;
	Weight.v1.name = "weight";
	Weight.v1.type = QNN_TENSOR_TYPE_STATIC;
	Weight.v1.dataFormat = QNN_TENSOR_DATA_FORMAT_DENSE;
#ifdef USING_F16
	Weight.v1.dataType = QNN_DATATYPE_FLOAT_16;
#else
	Weight.v1.dataType = QNN_DATATYPE_FLOAT_32;
#endif
	Weight.v1.rank = 2;
	Weight.v1.dimensions = WeightDims;
	Weight.v1.memType = QNN_TENSORMEMTYPE_RAW;
	Weight.v1.clientBuf.data = WegihtData;
	Weight.v1.clientBuf.dataSize = sizeof(WegihtData);
	Selected->QNN_INTERFACE_VER_NAME.tensorCreateGraphTensor(Graph, &Weight);

	Bias.v1.id = 3;
	Bias.v1.name = "bias";
	Bias.v1.type = QNN_TENSOR_TYPE_STATIC;
	Bias.v1.dataFormat = QNN_TENSOR_DATA_FORMAT_DENSE;
#ifdef USING_F16
	Bias.v1.dataType = QNN_DATATYPE_FLOAT_16;
#else
	Bias.v1.dataType = QNN_DATATYPE_FLOAT_32;
#endif
	Bias.v1.rank = 1;
	Bias.v1.dimensions = BiasDims;
	Bias.v1.memType = QNN_TENSORMEMTYPE_RAW;
	Bias.v1.clientBuf.data = BiasData;
	Bias.v1.clientBuf.dataSize = sizeof(BiasData);
	Selected->QNN_INTERFACE_VER_NAME.tensorCreateGraphTensor(Graph, &Bias);

	Mid0.v1.id = 4;
	Mid0.v1.name = "mid0";
	Mid0.v1.type = QNN_TENSOR_TYPE_NATIVE;
	Mid0.v1.dataFormat = QNN_TENSOR_DATA_FORMAT_DENSE;
	Mid0.v1.dataType =
#ifdef USING_F16
		QNN_DATATYPE_FLOAT_16;
#else
		QNN_DATATYPE_FLOAT_32;
#endif
	Mid0.v1.rank = 2;
	Mid0.v1.dimensions = MidDims;
	Mid0.v1.memType = QNN_TENSORMEMTYPE_RAW;
	Selected->QNN_INTERFACE_VER_NAME.tensorCreateGraphTensor(Graph, &Mid0);

	Mid1 = Mid0;
	Mid1.v1.id = 5;
	Mid1.v1.name = "mid1";
	Selected->QNN_INTERFACE_VER_NAME.tensorCreateGraphTensor(Graph, &Mid1);

	// FC0
	Qnn_Tensor_t FC0_Inputs[] = {Input, Weight, Bias};
	Qnn_OpConfig_t FC0 = QNN_OPCONFIG_INIT;
	FC0.v1.packageName = QNN_OP_PACKAGE_NAME_QTI_AISW;
	FC0.v1.typeName = QNN_OP_FULLY_CONNECTED;
	FC0.v1.name = "FC0";
	FC0.v1.inputTensors = FC0_Inputs;
	FC0.v1.numOfInputs = 3;
	FC0.v1.outputTensors = &Mid0;
	FC0.v1.numOfOutputs = 1;
	FC0.v1.params = nullptr;
	FC0.v1.numOfParams = 0;
	Selected->QNN_INTERFACE_VER_NAME.graphAddNode(Graph, FC0);

	// FC1
	Qnn_Tensor_t FC1_Inputs[] = {Mid0, Weight, Bias};
	Qnn_OpConfig_t FC1 = QNN_OPCONFIG_INIT;
	FC1.v1.packageName = QNN_OP_PACKAGE_NAME_QTI_AISW;
	FC1.v1.typeName = QNN_OP_FULLY_CONNECTED;
	FC1.v1.name = "FC1";
	FC1.v1.inputTensors = FC1_Inputs;
	FC1.v1.numOfInputs = 3;
	FC1.v1.outputTensors = &Mid1;
	FC1.v1.numOfOutputs = 1;
	Selected->QNN_INTERFACE_VER_NAME.graphAddNode(Graph, FC1);

	// FC2
	Qnn_Tensor_t FC2_Inputs[] = {Mid1, Weight, Bias};
	Qnn_OpConfig_t FC2 = QNN_OPCONFIG_INIT;
	FC2.v1.packageName = QNN_OP_PACKAGE_NAME_QTI_AISW;
	FC2.v1.typeName = QNN_OP_FULLY_CONNECTED;
	FC2.v1.name = "FC2";
	FC2.v1.inputTensors = FC2_Inputs;
	FC2.v1.numOfInputs = 3;
	FC2.v1.outputTensors = &Output;
	FC2.v1.numOfOutputs = 1;
	Selected->QNN_INTERFACE_VER_NAME.graphAddNode(Graph, FC2);

	// finalize grap
	Selected->QNN_INTERFACE_VER_NAME.graphFinalize(Graph, nullptr, nullptr);

// execute graph
#ifdef USING_F16
	uint16_t UserInput[Batch * InputShape];
	uint16_t UserOutput[Batch * OutputShape];
#else
	float UserInput[Batch * InputShape];
	float UserOutput[Batch * OutputShape];
#endif

	for (uint32_t i = 0; i < Batch * InputShape; i++)
	{
#ifdef USING_F16
		UserInput[i] = fp32_to_fp16(1.0f);
#else
		UserInput[i] = 1.0f;
#endif
	}

	Input.v1.clientBuf.data = UserInput;
	Input.v1.clientBuf.dataSize = sizeof(UserInput);

	Output.v1.clientBuf.data = UserOutput;
	Output.v1.clientBuf.dataSize = sizeof(UserOutput);

	Selected->QNN_INTERFACE_VER_NAME.graphExecute(Graph, &Input, 1, &Output, 1, nullptr, nullptr);

	for (size_t i = 0; i < Batch * OutputShape; i++)
	{
#ifdef USING_F16
		std::cout << fp16_to_fp32(UserOutput[i]) << " ";
#else
		std::cout << UserOutput[i] << " ";
#endif
	}

	dlclose(handle);

	return 1;
}

int test_matmul()
{
	printf("Test matmul\n");
	const char *backend = "libQnnHtp.so";
	void *handle = dlopen(backend, RTLD_NOW | RTLD_LOCAL);

	if (handle == nullptr)
	{
		printf("error: %s\n", dlerror());
		return 1;
	}

	auto *provider = (QnnInterfaceGetProvidersFn_t)dlsym(handle, "QnnInterface_getProviders");
	if (provider == nullptr)
	{
		dlclose(handle);
		printf("error: %s\n", dlerror());
		return 1;
	}

	const QnnInterface_t **providers = nullptr;
	uint32_t provider_count = 0;

	if (provider(&providers, &provider_count) != QNN_SUCCESS)
	{
		dlclose(handle);
		printf("error: QNN returned error\n");
		return 1;
	}

	auto isCapable = [](auto &ver)
	{
		return ver.major == QNN_API_VERSION_MAJOR && ver.minor >= QNN_API_VERSION_MINOR;
	};

	const QnnInterface_t *Selected = nullptr;
	for (auto i = 0u; i < provider_count; ++i)
	{
		auto *prov = providers[i];

		auto &backVer = prov->apiVersion.backendApiVersion;
		auto &coreVer = prov->apiVersion.coreApiVersion;

		printf("QNN Interface - Provider: %s, Backend ID: %u, Backend Version: %u.%u.%u, CoreAPI Version: %u.%u.%u\n",
			   prov->providerName, prov->backendId,
			   backVer.major, backVer.minor, backVer.patch,
			   coreVer.major, coreVer.minor, coreVer.patch);

		Selected = prov;
	}

	// create logger
	Qnn_LogHandle_t Logger = nullptr;
	if (Selected->QNN_INTERFACE_VER_NAME.logCreate(&qnn_log_handler, QNN_LOG_LEVEL_DEBUG, &Logger) != QNN_SUCCESS)
	{
		dlclose(handle);
		printf("error: QNN returned error\n");
		return 1;
	}

	// test log
	if (Logger == nullptr)
	{
		dlclose(handle);
		printf("error: Logger is null\n");
		return 1;
	}

	// create backend
	const QnnBackend_Config_t *BackendConfig = nullptr;
	Qnn_BackendHandle_t Backend;
	if (Selected->QNN_INTERFACE_VER_NAME.backendCreate(Logger, &BackendConfig, &Backend) != QNN_SUCCESS)
	{
		dlclose(handle);
		printf("error: QNN returned error\n");
		return 1;
	}

	/**
	 * executorch 참고하여 device config 설정: HtpDevice.cpp:295
	 */
	std::vector<const QnnDevice_Config_t *> ResultConfig;
	std::vector<QnnDevice_Config_t> DeviceConfigs;
	std::vector<QnnDevice_CustomConfig_t> CustomDeviceConfigs = {};
	QnnHtpDevice_CustomConfig_t *HtpConfig = nullptr;

	HtpConfig = new QnnHtpDevice_CustomConfig_t;
	HtpConfig->option = QNN_HTP_DEVICE_CONFIG_OPTION_ARCH;
	HtpConfig->arch.deviceId = 0;
	HtpConfig->arch.arch = QNN_HTP_DEVICE_ARCH_V73;
	CustomDeviceConfigs.push_back(static_cast<QnnDevice_CustomConfig_t>(HtpConfig));

	DeviceConfigs.resize(CustomDeviceConfigs.size());
	for (size_t i = 0; i < CustomDeviceConfigs.size(); i++)
	{
		DeviceConfigs[i].option = QNN_DEVICE_CONFIG_OPTION_CUSTOM;
		DeviceConfigs[i].customConfig = CustomDeviceConfigs[i];
		ResultConfig.push_back(&DeviceConfigs[i]);
	}

	ResultConfig.push_back(nullptr);

	Qnn_DeviceHandle_t Device = nullptr;
	if (Selected->QNN_INTERFACE_VER_NAME.deviceCreate(Logger, ResultConfig.data(), &Device) != QNN_SUCCESS)
	{
		dlclose(handle);
		printf("error: QNN returned error\n");
		return 1;
	}

	// create context
	const QnnContext_Config_t *ContextConfig = nullptr;
	Qnn_ContextHandle_t Context;
	if (Selected->QNN_INTERFACE_VER_NAME.contextCreate(Backend, Device, &ContextConfig, &Context) != QNN_SUCCESS)
	{
		dlclose(handle);
		printf("error: QNN returned error\n");
		return 1;
	}

	// compose graph
	// Create graph handle
	const QnnGraph_Config_t *GraphConfig = nullptr;
	Qnn_GraphHandle_t Graph;
	if (Selected->QNN_INTERFACE_VER_NAME.graphCreate(Context, "NAME", &GraphConfig, &Graph) != QNN_SUCCESS)
	{
		dlclose(handle);
		printf("error: QNN returned error\n");
		return 1;
	}

	constexpr uint32_t Batch = 1;
	constexpr uint32_t InputShape = 32;
	constexpr uint32_t OutputShape = 32;

	// Add graph tensor
	Qnn_Tensor_t Input = QNN_TENSOR_INIT;
	Qnn_Tensor_t Weight = QNN_TENSOR_INIT;
	Qnn_Tensor_t Bias = QNN_TENSOR_INIT;
	Qnn_Tensor_t Output = QNN_TENSOR_INIT;
	Qnn_Tensor_t Mid0 = QNN_TENSOR_INIT;
	Qnn_Tensor_t Mid1 = QNN_TENSOR_INIT;
	uint32_t InDims[] = {Batch, InputShape};
	uint32_t WeightDims[] = {InputShape, OutputShape};
	uint32_t BiasDims[] = {OutputShape};
	uint32_t OutDims[] = {Batch, OutputShape};
	uint32_t MidDims[] = {Batch, OutputShape};

	constexpr uint32_t WeightCount = OutputShape * InputShape;
#ifdef USING_F16
	static uint16_t WegihtData[WeightCount];
#else
	static float WegihtData[WeightCount];
#endif
	for (uint32_t i = 0; i < WeightCount; i++)
	{
#ifdef USING_F16
		WegihtData[i] = fp32_to_fp16(0.01f);
#else
		WegihtData[i] = 0.01f;
#endif
	}

	constexpr uint32_t BiasCount = OutputShape;
#ifdef USING_F16
	static uint16_t BiasData[BiasCount];
#else
	static float BiasData[BiasCount];
#endif
	for (uint32_t i = 0; i < BiasCount; i++)
	{
#ifdef USING_F16
		BiasData[i] = fp32_to_fp16(1.0f);
#else
		BiasData[i] = 1.0f;
#endif
	}

	Input.v1.id = 0;
	Input.v1.name = "input";
	Input.v1.type = QNN_TENSOR_TYPE_APP_WRITE;
	Input.v1.dataFormat = QNN_TENSOR_DATA_FORMAT_DENSE;
#ifdef USING_F16
	Input.v1.dataType = QNN_DATATYPE_FLOAT_16;
#else
	Input.v1.dataType = QNN_DATATYPE_FLOAT_32;
#endif
	Input.v1.quantizeParams = QNN_QUANTIZE_PARAMS_INIT;
	Input.v1.rank = 2;
	Input.v1.dimensions = InDims;
	Input.v1.memType = QNN_TENSORMEMTYPE_RAW;
	Input.v1.clientBuf.data = nullptr;
	Input.v1.clientBuf.dataSize = 0;
	Selected->QNN_INTERFACE_VER_NAME.tensorCreateGraphTensor(Graph, &Input);

	Output.v1.id = 1;
	Output.v1.name = "output";
	Output.v1.type = QNN_TENSOR_TYPE_APP_READ;
	Output.v1.dataFormat = QNN_TENSOR_DATA_FORMAT_DENSE;
#ifdef USING_F16
	Output.v1.dataType = QNN_DATATYPE_FLOAT_16;
#else
	Output.v1.dataType = QNN_DATATYPE_FLOAT_32;
#endif
	Output.v1.quantizeParams = QNN_QUANTIZE_PARAMS_INIT;
	Output.v1.rank = 2;
	Output.v1.dimensions = OutDims;
	Output.v1.memType = QNN_TENSORMEMTYPE_RAW;
	Output.v1.clientBuf.data = nullptr;
	Output.v1.clientBuf.dataSize = 0;
	Selected->QNN_INTERFACE_VER_NAME.tensorCreateGraphTensor(Graph, &Output);

	Weight.v1.id = 2;
	Weight.v1.name = "weight";
	Weight.v1.type = QNN_TENSOR_TYPE_STATIC;
	Weight.v1.dataFormat = QNN_TENSOR_DATA_FORMAT_DENSE;
#ifdef USING_F16
	Weight.v1.dataType = QNN_DATATYPE_FLOAT_16;
#else
	Weight.v1.dataType = QNN_DATATYPE_FLOAT_32;
#endif
	Weight.v1.rank = 2;
	Weight.v1.dimensions = WeightDims;
	Weight.v1.memType = QNN_TENSORMEMTYPE_RAW;
	Weight.v1.clientBuf.data = WegihtData;
	Weight.v1.clientBuf.dataSize = sizeof(WegihtData);
	Selected->QNN_INTERFACE_VER_NAME.tensorCreateGraphTensor(Graph, &Weight);

	Bias.v1.id = 3;
	Bias.v1.name = "bias";
	Bias.v1.type = QNN_TENSOR_TYPE_STATIC;
	Bias.v1.dataFormat = QNN_TENSOR_DATA_FORMAT_DENSE;
#ifdef USING_F16
	Bias.v1.dataType = QNN_DATATYPE_FLOAT_16;
#else
	Bias.v1.dataType = QNN_DATATYPE_FLOAT_32;
#endif
	Bias.v1.rank = 1;
	Bias.v1.dimensions = BiasDims;
	Bias.v1.memType = QNN_TENSORMEMTYPE_RAW;
	Bias.v1.clientBuf.data = BiasData;
	Bias.v1.clientBuf.dataSize = sizeof(BiasData);
	Selected->QNN_INTERFACE_VER_NAME.tensorCreateGraphTensor(Graph, &Bias);

	Mid0.v1.id = 4;
	Mid0.v1.name = "mid0";
	Mid0.v1.type = QNN_TENSOR_TYPE_NATIVE;
	Mid0.v1.dataFormat = QNN_TENSOR_DATA_FORMAT_DENSE;
	Mid0.v1.dataType =
#ifdef USING_F16
		QNN_DATATYPE_FLOAT_16;
#else
		QNN_DATATYPE_FLOAT_32;
#endif
	Mid0.v1.rank = 2;
	Mid0.v1.dimensions = MidDims;
	Mid0.v1.memType = QNN_TENSORMEMTYPE_RAW;
	Selected->QNN_INTERFACE_VER_NAME.tensorCreateGraphTensor(Graph, &Mid0);

	Mid1 = Mid0;
	Mid1.v1.id = 5;
	Mid1.v1.name = "mid1";
	Selected->QNN_INTERFACE_VER_NAME.tensorCreateGraphTensor(Graph, &Mid1);

	// MM0
	Qnn_Tensor_t MM0_Inputs[] = {Input, Weight};
	Qnn_OpConfig_t MM0 = QNN_OPCONFIG_INIT;
	MM0.v1.packageName = QNN_OP_PACKAGE_NAME_QTI_AISW;
	MM0.v1.typeName = QNN_OP_MAT_MUL;
	MM0.v1.name = "MM0";
	MM0.v1.inputTensors = MM0_Inputs;
	MM0.v1.numOfInputs = sizeof(MM0_Inputs) / sizeof(MM0_Inputs[0]);
	MM0.v1.outputTensors = &Mid0;
	MM0.v1.numOfOutputs = 1;
	MM0.v1.params = nullptr;
	MM0.v1.numOfParams = 0;
	Selected->QNN_INTERFACE_VER_NAME.graphAddNode(Graph, MM0);

	// MM1
	Qnn_Tensor_t MM1_Inputs[] = {Mid0, Weight};
	Qnn_OpConfig_t MM1 = QNN_OPCONFIG_INIT;
	MM1.v1.packageName = QNN_OP_PACKAGE_NAME_QTI_AISW;
	MM1.v1.typeName = QNN_OP_MAT_MUL;
	MM1.v1.name = "MM1";
	MM1.v1.inputTensors = MM1_Inputs;
	MM1.v1.numOfInputs = sizeof(MM1_Inputs) / sizeof(MM1_Inputs[0]);
	MM1.v1.outputTensors = &Mid1;
	MM1.v1.numOfOutputs = 1;
	MM1.v1.params = nullptr;
	MM1.v1.numOfParams = 0;
	Selected->QNN_INTERFACE_VER_NAME.graphAddNode(Graph, MM1);

	// MM2
	Qnn_Tensor_t MM2_Inputs[] = {Mid1, Weight};
	Qnn_OpConfig_t MM2 = QNN_OPCONFIG_INIT;
	MM2.v1.packageName = QNN_OP_PACKAGE_NAME_QTI_AISW;
	MM2.v1.typeName = QNN_OP_MAT_MUL;
	MM2.v1.name = "MM2";
	MM2.v1.inputTensors = MM2_Inputs;
	MM2.v1.numOfInputs = sizeof(MM2_Inputs) / sizeof(MM2_Inputs[0]);
	MM2.v1.outputTensors = &Output;
	MM2.v1.numOfOutputs = 1;
	MM2.v1.params = nullptr;
	MM2.v1.numOfParams = 0;
	Selected->QNN_INTERFACE_VER_NAME.graphAddNode(Graph, MM2);

	// finalize grap
	Selected->QNN_INTERFACE_VER_NAME.graphFinalize(Graph, nullptr, nullptr);

// execute graph
#ifdef USING_F16
	uint16_t UserInput[Batch * InputShape];
	uint16_t UserOutput[Batch * OutputShape];
#else
	float UserInput[Batch * InputShape];
	float UserOutput[Batch * OutputShape];
#endif

	for (uint32_t i = 0; i < Batch * InputShape; i++)
	{
#ifdef USING_F16
		UserInput[i] = fp32_to_fp16(1.0f);
#else
		UserInput[i] = 1.0f;
#endif
	}

	Input.v1.clientBuf.data = UserInput;
	Input.v1.clientBuf.dataSize = sizeof(UserInput);

	Output.v1.clientBuf.data = UserOutput;
	Output.v1.clientBuf.dataSize = sizeof(UserOutput);

	Selected->QNN_INTERFACE_VER_NAME.graphExecute(Graph, &Input, 1, &Output, 1, nullptr, nullptr);

	for (size_t i = 0; i < Batch * OutputShape; i++)
	{
#ifdef USING_F16
		std::cout << fp16_to_fp32(UserOutput[i]) << " ";
#else
		std::cout << UserOutput[i] << " ";
#endif
	}

	dlclose(handle);

	return 1;
}

int main()
{
	//test_linear();
	test_matmul();
	return 0;
}
