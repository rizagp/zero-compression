// zero: minimal Level Zero (oneAPI L0) declarations and a dynamic loader for the Level Zero loader library
// (ze_loader.dll on Windows, libze_loader.so.1 on Linux; both come with the Intel graphics driver / compute runtime).
// Only what zero uses; the layouts follow ze_api.h (Level Zero spec 1.x). No SDK headers needed.
#pragma once
#include "platform.h"
#include <cstdint>
#include <cstddef>

typedef uint32_t ze_result_t;
#define ZE_RESULT_SUCCESS 0u
#define ZE_RESULT_NOT_READY 1u
#define ZE_RESULT_ERROR_OUT_OF_DEVICE_MEMORY 0x70000002u
typedef struct _ze_driver_handle_t* ze_driver_handle_t;
typedef struct _ze_device_handle_t* ze_device_handle_t;
typedef struct _ze_context_handle_t* ze_context_handle_t;
typedef struct _ze_command_list_handle_t* ze_command_list_handle_t;
typedef struct _ze_module_handle_t* ze_module_handle_t;
typedef struct _ze_module_build_log_handle_t* ze_module_build_log_handle_t;
typedef struct _ze_kernel_handle_t* ze_kernel_handle_t;
typedef struct _ze_event_handle_t* ze_event_handle_t;
typedef struct _ze_event_pool_handle_t* ze_event_pool_handle_t;
typedef uint32_t ze_structure_type_t;

#define ZE_STYPE_DEVICE_PROPERTIES 0x3u
#define ZE_STYPE_DEVICE_COMPUTE_PROPERTIES 0x4u
#define ZE_STYPE_DEVICE_MEMORY_PROPERTIES 0x7u
#define ZE_STYPE_CONTEXT_DESC 0xdu
#define ZE_STYPE_COMMAND_QUEUE_DESC 0xeu
#define ZE_STYPE_DEVICE_MEM_ALLOC_DESC 0x15u
#define ZE_STYPE_HOST_MEM_ALLOC_DESC 0x16u
#define ZE_STYPE_MODULE_DESC 0x1bu
#define ZE_STYPE_KERNEL_DESC 0x1du

struct ze_device_properties_t {
  ze_structure_type_t stype;
  void* pNext;
  uint32_t type, vendorId, deviceId, flags, subdeviceId, coreClockRate;
  uint64_t maxMemAllocSize;
  uint32_t maxHardwareContexts, maxCommandQueuePriority, numThreadsPerEU, physicalEUSimdWidth, numEUsPerSubslice,
      numSubslicesPerSlice, numSlices;
  uint64_t timerResolution;
  uint32_t timestampValidBits, kernelTimestampValidBits;
  uint8_t uuid[16];
  char name[256];
};
#define ZE_DEVICE_TYPE_GPU 1u
#define ZE_DEVICE_PROPERTY_FLAG_INTEGRATED 1u
struct ze_device_compute_properties_t {
  ze_structure_type_t stype;
  void* pNext;
  uint32_t maxTotalGroupSize, maxGroupSizeX, maxGroupSizeY, maxGroupSizeZ, maxGroupCountX, maxGroupCountY,
      maxGroupCountZ, maxSharedLocalMemory, numSubGroupSizes, subGroupSizes[8];
};
struct ze_device_memory_properties_t {
  ze_structure_type_t stype;
  void* pNext;
  uint32_t flags, maxClockRate, maxBusWidth;
  uint64_t totalSize;
  char name[256];
};
struct ze_context_desc_t { ze_structure_type_t stype; const void* pNext; uint32_t flags; };
struct ze_command_queue_desc_t {
  ze_structure_type_t stype;
  const void* pNext;
  uint32_t ordinal, index, flags, mode, priority;  // flags: 2 = in order; mode: 0 default, 1 synchronous, 2 asynchronous
};
struct ze_host_mem_alloc_desc_t { ze_structure_type_t stype; const void* pNext; uint32_t flags; };
struct ze_device_mem_alloc_desc_t { ze_structure_type_t stype; const void* pNext; uint32_t flags, ordinal; };
struct ze_module_constants_t { uint32_t numConstants; const uint32_t* pConstantIds; const void** pConstantValues; };
struct ze_module_desc_t {
  ze_structure_type_t stype;
  const void* pNext;
  uint32_t format;  // 0 SPIR-V, 1 native (the device binary the OpenCL compiler produces)
  size_t inputSize;
  const uint8_t* pInputModule;
  const char* pBuildFlags;
  const ze_module_constants_t* pConstants;
};
#define ZE_MODULE_FORMAT_NATIVE 1u
struct ze_kernel_desc_t { ze_structure_type_t stype; const void* pNext; uint32_t flags; const char* pKernelName; };
struct ze_group_count_t { uint32_t groupCountX, groupCountY, groupCountZ; };
struct ze_kernel_properties_t {  // (stype 0x1e)
  ze_structure_type_t stype;
  void* pNext;
  uint32_t numKernelArgs, requiredGroupSize[3], requiredNumSubGroups, requiredSubgroupSize, maxSubgroupSize,
      maxNumSubgroups, localMemSize, privateMemSize, spillMemSize;
  uint8_t uuid[32];
};

#define ZE_FUNCS(X)                                                                                              \
  X(ze_result_t, zeKernelGetProperties, (ze_kernel_handle_t, ze_kernel_properties_t*))                           \
  X(ze_result_t, zeInit, (uint32_t))                                                                             \
  X(ze_result_t, zeDriverGet, (uint32_t*, ze_driver_handle_t*))                                                  \
  X(ze_result_t, zeDeviceGet, (ze_driver_handle_t, uint32_t*, ze_device_handle_t*))                              \
  X(ze_result_t, zeDeviceGetProperties, (ze_device_handle_t, ze_device_properties_t*))                           \
  X(ze_result_t, zeDeviceGetComputeProperties, (ze_device_handle_t, ze_device_compute_properties_t*))            \
  X(ze_result_t, zeDeviceGetMemoryProperties, (ze_device_handle_t, uint32_t*, ze_device_memory_properties_t*))    \
  X(ze_result_t, zeContextCreate, (ze_driver_handle_t, const ze_context_desc_t*, ze_context_handle_t*))          \
  X(ze_result_t, zeContextDestroy, (ze_context_handle_t))                                                        \
  X(ze_result_t, zeCommandListCreateImmediate,                                                                   \
    (ze_context_handle_t, ze_device_handle_t, const ze_command_queue_desc_t*, ze_command_list_handle_t*))        \
  X(ze_result_t, zeCommandListDestroy, (ze_command_list_handle_t))                                               \
  X(ze_result_t, zeCommandListHostSynchronize, (ze_command_list_handle_t, uint64_t))                             \
  X(ze_result_t, zeCommandListAppendLaunchKernel,                                                                \
    (ze_command_list_handle_t, ze_kernel_handle_t, const ze_group_count_t*, ze_event_handle_t, uint32_t,          \
     ze_event_handle_t*))                                                                                        \
  X(ze_result_t, zeCommandListAppendMemoryCopy,                                                                  \
    (ze_command_list_handle_t, void*, const void*, size_t, ze_event_handle_t, uint32_t, ze_event_handle_t*))     \
  X(ze_result_t, zeCommandListAppendMemoryFill,                                                                  \
    (ze_command_list_handle_t, void*, const void*, size_t, size_t, ze_event_handle_t, uint32_t, ze_event_handle_t*)) \
  X(ze_result_t, zeCommandListAppendBarrier, (ze_command_list_handle_t, ze_event_handle_t, uint32_t, ze_event_handle_t*)) \
  X(ze_result_t, zeMemAllocHost, (ze_context_handle_t, const ze_host_mem_alloc_desc_t*, size_t, size_t, void**))  \
  X(ze_result_t, zeMemAllocDevice,                                                                               \
    (ze_context_handle_t, const ze_device_mem_alloc_desc_t*, size_t, size_t, ze_device_handle_t, void**))        \
  X(ze_result_t, zeMemFree, (ze_context_handle_t, void*))                                                        \
  X(ze_result_t, zeModuleCreate,                                                                                 \
    (ze_context_handle_t, ze_device_handle_t, const ze_module_desc_t*, ze_module_handle_t*,                      \
     ze_module_build_log_handle_t*))                                                                             \
  X(ze_result_t, zeModuleDestroy, (ze_module_handle_t))                                                          \
  X(ze_result_t, zeModuleBuildLogGetString, (ze_module_build_log_handle_t, size_t*, char*))                      \
  X(ze_result_t, zeModuleBuildLogDestroy, (ze_module_build_log_handle_t))                                        \
  X(ze_result_t, zeKernelCreate, (ze_module_handle_t, const ze_kernel_desc_t*, ze_kernel_handle_t*))             \
  X(ze_result_t, zeKernelDestroy, (ze_kernel_handle_t))                                                          \
  X(ze_result_t, zeKernelSetGroupSize, (ze_kernel_handle_t, uint32_t, uint32_t, uint32_t))                       \
  X(ze_result_t, zeKernelSetArgumentValue, (ze_kernel_handle_t, uint32_t, size_t, const void*))

#define ZE_DECL(ret, name, args) typedef ret(ZERO_CALL* PFN_##name) args; inline PFN_##name name = nullptr;
ZE_FUNCS(ZE_DECL)
#undef ZE_DECL

// loads the Level Zero loader and every function above; false (with the reason in err) when it cannot
inline bool ze_load(std::string& err) {
  static zero_lib_t m = nullptr;
  if (m) return true;
  m = zero_lib_open_any(kZeLoaderNames);
  if (!m) { err = std::string("cannot load the Level Zero loader (") + kZeLoaderNames[0] + "): is the Intel graphics driver installed?"; return false; }
#define ZE_LOAD(ret, name, args) \
  name = (PFN_##name)zero_lib_sym(m, #name); \
  if (!name) { err = std::string("the Level Zero loader lacks ") + #name; zero_lib_close(m); m = nullptr; return false; }
  ZE_FUNCS(ZE_LOAD)
#undef ZE_LOAD
  return true;
}
