// zero: the GPU runtime. One Level Zero context, one GPU, one in-order immediate command list. Kernels are written in
// OpenCL C, compiled by the OpenCL runtime of the same driver (loaded dynamically, only for compiling) into the
// device's native binary, which Level Zero loads as a module. The binaries are cached on disk per (source, options,
// device, driver). Buffers are device USM allocations; host reads and writes go through a host USM staging buffer.
// Errors throw std::runtime_error with a plain-words message: the CLI prints it, the GUI shows it.
#pragma once
#include "ze_min.h"
#include <cstring>
#include <fstream>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace gpu {

struct Error : std::runtime_error {
  using std::runtime_error::runtime_error;
};
inline void check(ze_result_t r, const char* what) {
  if (r != ZE_RESULT_SUCCESS) {
    char b[256];
    if ((uint32_t)r == 0x70000001u) snprintf(b, sizeof b, "the GPU was reset (device lost, Level Zero error 0x70000001 in %s)", what);
    else snprintf(b, sizeof b, "Level Zero error 0x%x in %s", r, what);
    throw Error(b);
  }
}
#define ZE_OK(x) gpu::check((x), #x)

// ---- the OpenCL compiler of the same driver (only to turn OpenCL C into the device's native binary)
namespace ocl {
typedef int32_t cl_int;
typedef uint32_t cl_uint;
typedef uint64_t cl_ulong;
typedef cl_int(ZERO_CALL* GetPlatformIDs)(cl_uint, void**, cl_uint*);
typedef cl_int(ZERO_CALL* GetDeviceIDs)(void*, cl_ulong, cl_uint, void**, cl_uint*);
typedef cl_int(ZERO_CALL* GetDeviceInfo)(void*, cl_uint, size_t, void*, size_t*);
typedef void*(ZERO_CALL* CreateContext)(const intptr_t*, cl_uint, void* const*, void*, void*, cl_int*);
typedef void*(ZERO_CALL* CreateProgramWithSource)(void*, cl_uint, const char**, const size_t*, cl_int*);
typedef cl_int(ZERO_CALL* BuildProgram)(void*, cl_uint, void* const*, const char*, void*, void*);
typedef cl_int(ZERO_CALL* GetProgramBuildInfo)(void*, void*, cl_uint, size_t, void*, size_t*);
typedef cl_int(ZERO_CALL* GetProgramInfo)(void*, cl_uint, size_t, void*, size_t*);
typedef cl_int(ZERO_CALL* ReleaseProgram)(void*);
inline GetPlatformIDs getPlatformIDs;
inline GetDeviceIDs getDeviceIDs;
inline GetDeviceInfo getDeviceInfo;
inline CreateContext createContext;
inline CreateProgramWithSource createProgramWithSource;
inline BuildProgram buildProgram;
inline GetProgramBuildInfo getProgramBuildInfo;
inline GetProgramInfo getProgramInfo;
inline ReleaseProgram releaseProgram;
inline void *device = nullptr, *context = nullptr;
inline std::string device_name, driver_version;

// the GPU device of the OpenCL runtime whose name matches the Level Zero device (the same driver: one GPU here)
inline bool load(const std::string& ze_name, std::string& err) {
  if (context) return true;
  zero_lib_t h = zero_lib_open_any(kOpenCLNames);
  if (!h) { err = std::string("cannot load the OpenCL runtime (") + kOpenCLNames[0] + "), needed to compile the GPU kernels"; return false; }
#define RCL(var, name) var = (decltype(var))zero_lib_sym(h, name); if (!var) { err = std::string("OpenCL runtime lacks ") + name; return false; }
  RCL(getPlatformIDs, "clGetPlatformIDs") RCL(getDeviceIDs, "clGetDeviceIDs") RCL(getDeviceInfo, "clGetDeviceInfo")
  RCL(createContext, "clCreateContext") RCL(createProgramWithSource, "clCreateProgramWithSource")
  RCL(buildProgram, "clBuildProgram") RCL(getProgramBuildInfo, "clGetProgramBuildInfo")
  RCL(getProgramInfo, "clGetProgramInfo") RCL(releaseProgram, "clReleaseProgram")
#undef RCL
  cl_uint np = 0;
  if (getPlatformIDs(0, nullptr, &np) != 0 || !np) { err = "no OpenCL platform"; return false; }
  std::vector<void*> plats(np);
  getPlatformIDs(np, plats.data(), nullptr);
  void* fallback = nullptr;
  for (void* p : plats) {
    cl_uint nd = 0;
    if (getDeviceIDs(p, 1u << 2 /* GPU */, 0, nullptr, &nd) != 0 || !nd) continue;
    std::vector<void*> devs(nd);
    getDeviceIDs(p, 1u << 2, nd, devs.data(), nullptr);
    for (void* d : devs) {
      char name[256] = {};
      getDeviceInfo(d, 0x102B /* CL_DEVICE_NAME */, sizeof(name) - 1, name, nullptr);
      if (!fallback) fallback = d;
      if (ze_name == name) { device = d; break; }
    }
    if (device) break;
  }
  if (!device) device = fallback;
  if (!device) { err = "no OpenCL GPU device (the compiler of the kernels)"; return false; }
  char name[256] = {}, drv[256] = {};
  getDeviceInfo(device, 0x102B, sizeof(name) - 1, name, nullptr);
  getDeviceInfo(device, 0x102D /* CL_DRIVER_VERSION */, sizeof(drv) - 1, drv, nullptr);
  device_name = name;
  driver_version = drv;
  cl_int e = 0;
  context = createContext(nullptr, 1, &device, nullptr, nullptr, &e);
  if (e != 0 || !context) { context = nullptr; err = "cannot create the OpenCL compiler context"; return false; }
  return true;
}
// compiles OpenCL C to the device's native binary; throws with the build log on an error
inline std::vector<uint8_t> compile(const std::string& src, const std::string& opts) {
  cl_int err = 0;
  const char* s = src.c_str();
  const size_t sn = src.size();
  void* prog = createProgramWithSource(context, 1, &s, &sn, &err);
  if (err != 0 || !prog) throw Error("clCreateProgramWithSource failed");
  err = buildProgram(prog, 1, &device, opts.c_str(), nullptr, nullptr);
  if (err != 0) {
    size_t ln = 0;
    getProgramBuildInfo(prog, device, 0x1183 /* CL_PROGRAM_BUILD_LOG */, 0, nullptr, &ln);
    std::string log(ln, 0);
    getProgramBuildInfo(prog, device, 0x1183, ln, log.data(), nullptr);
    releaseProgram(prog);
    throw Error("GPU kernel build failed:\n" + log);
  }
  size_t bn = 0;
  getProgramInfo(prog, 0x1165 /* CL_PROGRAM_BINARY_SIZES */, sizeof(bn), &bn, nullptr);
  std::vector<uint8_t> bin(bn);
  uint8_t* bp = bin.data();
  getProgramInfo(prog, 0x1166 /* CL_PROGRAM_BINARIES */, sizeof(bp), &bp, nullptr);
  releaseProgram(prog);
  return bin;
}
}  // namespace ocl

// ---- the device
struct Device {
  ze_driver_handle_t driver = nullptr;
  ze_device_handle_t dev = nullptr;
  ze_context_handle_t ctx = nullptr;
  ze_command_list_handle_t cl = nullptr;  // one in-order immediate command list: launches, copies, fills in order
  std::string name, driver_version;
  uint64_t memory = 0;          // the device's memory (an iGPU: its share of the system RAM)
  uint32_t max_group = 1024;    // work-items per work-group
  uint32_t local_mem = 65536;   // shared local memory per work-group
  uint32_t eus = 0;             // execution units
  bool integrated = false;
  std::string cache_dir;        // compiled kernels; empty: no cache
  char* stage = nullptr;        // host USM bounce buffer
  size_t stage_n = 0;
  uint64_t allocated = 0;

  void open() {
    std::string err;
    if (!ze_load(err)) throw Error(err);
    ZE_OK(zeInit(1 /* GPU only */));
    uint32_t nd = 0;
    ZE_OK(zeDriverGet(&nd, nullptr));
    std::vector<ze_driver_handle_t> drvs(nd);
    ZE_OK(zeDriverGet(&nd, drvs.data()));
    ze_device_properties_t props{};
    for (ze_driver_handle_t d : drvs) {
      uint32_t n = 0;
      if (zeDeviceGet(d, &n, nullptr) != 0 || !n) continue;
      std::vector<ze_device_handle_t> ds(n);
      zeDeviceGet(d, &n, ds.data());
      for (ze_device_handle_t x : ds) {
        ze_device_properties_t p{};
        p.stype = ZE_STYPE_DEVICE_PROPERTIES;
        if (zeDeviceGetProperties(x, &p) == 0 && p.type == ZE_DEVICE_TYPE_GPU) { driver = d; dev = x; props = p; break; }
      }
      if (dev) break;
    }
    if (!dev) throw Error("no Level Zero GPU found (Intel Arc / Iris graphics with a current driver)");
    name = props.name;
    integrated = (props.flags & ZE_DEVICE_PROPERTY_FLAG_INTEGRATED) != 0;
    eus = props.numSlices * props.numSubslicesPerSlice * props.numEUsPerSubslice;
    ze_device_compute_properties_t cp{};
    cp.stype = ZE_STYPE_DEVICE_COMPUTE_PROPERTIES;
    if (zeDeviceGetComputeProperties(dev, &cp) == 0) { max_group = cp.maxTotalGroupSize; local_mem = cp.maxSharedLocalMemory; }
    uint32_t nm = 1;
    ze_device_memory_properties_t mp{};
    mp.stype = ZE_STYPE_DEVICE_MEMORY_PROPERTIES;
    if (zeDeviceGetMemoryProperties(dev, &nm, &mp) == 0) memory = mp.totalSize;
    ze_context_desc_t cd{ZE_STYPE_CONTEXT_DESC, nullptr, 0};
    ZE_OK(zeContextCreate(driver, &cd, &ctx));
    ze_command_queue_desc_t qd{ZE_STYPE_COMMAND_QUEUE_DESC, nullptr, 0, 0, 2 /* in order */, 2 /* asynchronous */, 0};
    ZE_OK(zeCommandListCreateImmediate(ctx, dev, &qd, &cl));
    if (!ocl::load(name, err)) throw Error(err);
    driver_version = ocl::driver_version;
    if (!getenv("ZERO_NO_KCACHE")) cache_dir = zero_cache_dir();
    if (!cache_dir.empty()) {
#ifdef _WIN32
      cache_dir += "\\kernels";
#else
      cache_dir += "/kernels";
#endif
    }
  }
  void close() {
    if (cl) { zeCommandListHostSynchronize(cl, UINT64_MAX); zeCommandListDestroy(cl); cl = nullptr; }
    if (stage) { zeMemFree(ctx, stage); stage = nullptr; stage_n = 0; }
    if (ctx) { zeContextDestroy(ctx); ctx = nullptr; }
  }
  ~Device() { close(); }

  void sync() { ZE_OK(zeCommandListHostSynchronize(cl, UINT64_MAX)); }
  // the same for destructors, which must not throw: a lost device has nothing left to wait for
  void wait_quiet() noexcept { if (cl) zeCommandListHostSynchronize(cl, UINT64_MAX); }

  void* alloc_device(size_t n) {
    ze_device_mem_alloc_desc_t dd{ZE_STYPE_DEVICE_MEM_ALLOC_DESC, nullptr, 0, 0};
    void* p = nullptr;
    const ze_result_t r = zeMemAllocDevice(ctx, &dd, n < 64 ? 64 : n, 4096, dev, &p);
    if (r != ZE_RESULT_SUCCESS) {
      char b[200];
      snprintf(b, sizeof b, "GPU allocation of %.1f MB failed (0x%x; %.0f MB already allocated): the GPU memory is full",
               n / 1e6, r, allocated / 1e6);
      throw Error(b);
    }
    allocated += n;
    return p;
  }
  void free_device(void* p, size_t n) {
    if (!p) return;
    wait_quiet();
    zeMemFree(ctx, p);
    allocated -= n;
  }
  void* alloc_host(size_t n) {
    ze_host_mem_alloc_desc_t hd{ZE_STYPE_HOST_MEM_ALLOC_DESC, nullptr, 0};
    void* p = nullptr;
    ZE_OK(zeMemAllocHost(ctx, &hd, n < 64 ? 64 : n, 4096, &p));
    return p;
  }
  char* staging(size_t n) {
    if (n > stage_n) {
      sync();
      if (stage) zeMemFree(ctx, stage);
      stage_n = n < ((size_t)1 << 22) ? ((size_t)1 << 22) : n;
      stage = (char*)alloc_host(stage_n);
    }
    return stage;
  }
  // host -> device and device -> host, through the staging buffer, both synchronous
  void write(void* dst, const void* src, size_t n) {
    if (!n) return;
    char* s = staging(n);
    memcpy(s, src, n);
    ZE_OK(zeCommandListAppendMemoryCopy(cl, dst, s, n, nullptr, 0, nullptr));
    sync();
  }
  void read(void* dst, const void* src, size_t n) {
    if (!n) return;
    char* s = staging(n);
    ZE_OK(zeCommandListAppendMemoryCopy(cl, s, src, n, nullptr, 0, nullptr));
    sync();
    memcpy(dst, s, n);
  }
  void fill(void* dst, uint32_t v, size_t n) {
    if (!n) return;
    ZE_OK(zeCommandListAppendMemoryFill(cl, dst, &v, 4, n, nullptr, 0, nullptr));
  }
  void copy(void* dst, const void* src, size_t n) {
    if (!n) return;
    ZE_OK(zeCommandListAppendMemoryCopy(cl, dst, src, n, nullptr, 0, nullptr));
  }
};

// a GPU buffer, freed with its scope: device memory, or host memory the GPU reads and writes directly (an iGPU's RAM
// is the CPU's, so host memory saves the copies)
struct Buffer {
  Device* d = nullptr;
  char* p = nullptr;
  size_t n = 0;
  bool host = false;
  Buffer() = default;
  Buffer(Device& dev, size_t bytes) { alloc(dev, bytes); }
  Buffer(const Buffer&) = delete;
  Buffer& operator=(const Buffer&) = delete;
  Buffer(Buffer&& o) noexcept : d(o.d), p(o.p), n(o.n), host(o.host) { o.p = nullptr; o.n = 0; }
  Buffer& operator=(Buffer&& o) noexcept { release(); d = o.d; p = o.p; n = o.n; host = o.host; o.p = nullptr; o.n = 0; return *this; }
  ~Buffer() { release(); }
  void alloc(Device& dev, size_t bytes) { release(); d = &dev; n = bytes; p = (char*)dev.alloc_device(bytes); }
  void alloc_host(Device& dev, size_t bytes) { release(); d = &dev; n = bytes; host = true; p = (char*)dev.alloc_host(bytes); }
  void release() {
    if (p && d) {
      if (host) { d->wait_quiet(); zeMemFree(d->ctx, p); }
      else d->free_device(p, n);
    }
    p = nullptr; n = 0; host = false;
  }
  // grows when smaller than bytes (the old contents are lost)
  void reserve(Device& dev, size_t bytes) { if (bytes > n) alloc(dev, bytes); }
  void write(const void* src, size_t bytes, size_t off = 0) {
    if (host) { d->sync(); if (bytes) memcpy(p + off, src, bytes); }
    else d->write(p + off, src, bytes);
  }
  void read(void* dst, size_t bytes, size_t off = 0) {
    if (host) { d->sync(); if (bytes) memcpy(dst, p + off, bytes); }
    else d->read(dst, p + off, bytes);
  }
  void zero(size_t bytes, size_t off = 0) { d->fill(p + off, 0, bytes); }
  const char* ptr(size_t off = 0) const { return p + off; }
};

// FNV-1a of a string (the cache key)
inline uint64_t fnv1a(const std::string& s) {
  uint64_t h = 1469598103934665603ull;
  for (unsigned char c : s) { h ^= c; h *= 1099511628211ull; }
  return h;
}

// a module (an OpenCL C source built with options) and its kernels
struct Program {
  Device* d = nullptr;
  ze_module_handle_t m = nullptr;
  bool from_cache = false;
  double build_seconds = 0;
  Program() = default;
  Program(const Program&) = delete;
  ~Program() { if (m) zeModuleDestroy(m); }

  void build(Device& dev, const std::string& src, const std::string& opts) {
    d = &dev;
    const double t0 = zero_now();
    std::string path;
    std::vector<uint8_t> bin;
    if (!dev.cache_dir.empty()) {
      char hex[17];
      snprintf(hex, sizeof hex, "%016llx", (unsigned long long)fnv1a(src + "\x01" + opts + "\x01" + dev.name + "|" + dev.driver_version));
      path = dev.cache_dir + "/" + hex + ".bin";
      std::ifstream f(path, std::ios::binary);
      if (f) bin.assign(std::istreambuf_iterator<char>(f), std::istreambuf_iterator<char>());
    }
    bool cached = !bin.empty();
    for (int attempt = 0; attempt < 2; ++attempt) {
      if (bin.empty()) {
        bin = ocl::compile(src, opts);
        if (!path.empty()) {
          std::error_code ec;
          std::filesystem::create_directories(zero_path(dev.cache_dir), ec);
          const std::string tmp = path + ".tmp";
          bool ok = false;
          {
            std::ofstream o(tmp, std::ios::binary);
            o.write((const char*)bin.data(), (std::streamsize)bin.size());
            o.close();
            ok = !o.fail();
          }
          if (ok) std::filesystem::rename(zero_path(tmp), zero_path(path), ec);
          else std::filesystem::remove(zero_path(tmp), ec);
        }
      }
      ze_module_desc_t md{ZE_STYPE_MODULE_DESC, nullptr, ZE_MODULE_FORMAT_NATIVE, bin.size(), bin.data(), "", nullptr};
      ze_module_build_log_handle_t log = nullptr;
      const ze_result_t r = zeModuleCreate(dev.ctx, dev.dev, &md, &m, &log);
      if (r == ZE_RESULT_SUCCESS) {
        if (log) zeModuleBuildLogDestroy(log);
        from_cache = cached;
        break;
      }
      std::string s;
      if (log) {
        size_t ln = 0;
        zeModuleBuildLogGetString(log, &ln, nullptr);
        s.resize(ln);
        zeModuleBuildLogGetString(log, &ln, s.data());
        zeModuleBuildLogDestroy(log);
      }
      m = nullptr;
      if (cached && attempt == 0) {  // a damaged cache file: compiled again from the source
        std::error_code ec;
        std::filesystem::remove(zero_path(path), ec);
        bin.clear();
        cached = false;
        continue;
      }
      char b[128];
      snprintf(b, sizeof b, "zeModuleCreate failed 0x%x: ", r);
      throw Error(b + s);
    }
    build_seconds = zero_now() - t0;
  }
};

struct Kernel {
  Device* d = nullptr;
  ze_kernel_handle_t k = nullptr;
  std::string name;
  uint32_t gx = 0, gy = 0, gz = 0;
  Kernel() = default;
  Kernel(const Kernel&) = delete;
  ~Kernel() { release(); }
  // (the group size belongs to the kernel object: a new one has none, whatever the last launch of the old one used)
  void release() {
    if (k) zeKernelDestroy(k);
    k = nullptr;
    gx = gy = gz = 0;
  }
  void create(Program& p, const char* kname) {
    release();
    d = p.d;
    name = kname;
    ze_kernel_desc_t kd{ZE_STYPE_KERNEL_DESC, nullptr, 0, kname};
    const ze_result_t r = zeKernelCreate(p.m, &kd, &k);
    if (r != ZE_RESULT_SUCCESS) throw Error(std::string("zeKernelCreate(") + kname + ") failed");
    // ZERO_KPROPS=1: each kernel's register spill, local memory and sub-group size (a spill costs memory traffic)
    static const bool kprops = getenv("ZERO_KPROPS") && atoi(getenv("ZERO_KPROPS")) != 0;
    if (kprops) {
      ze_kernel_properties_t kp{};
      kp.stype = (decltype(kp.stype))0x1e;
      if (zeKernelGetProperties(k, &kp) == ZE_RESULT_SUCCESS)
        fprintf(stderr, "[kernel] %-14s spill %u private %u local %u simd %u\n", kname, kp.spillMemSize, kp.privateMemSize,
                kp.localMemSize, kp.maxSubgroupSize);
    }
  }
  void arg(uint32_t i, const Buffer& b, size_t off = 0) {
    const char* p = b.p + off;
    ZE_OK(zeKernelSetArgumentValue(k, i, sizeof(p), &p));
  }
  void arg(uint32_t i, uint32_t v) { ZE_OK(zeKernelSetArgumentValue(k, i, 4, &v)); }
  void arg(uint32_t i, int32_t v) { ZE_OK(zeKernelSetArgumentValue(k, i, 4, &v)); }
  void arg_local(uint32_t i, size_t bytes) { ZE_OK(zeKernelSetArgumentValue(k, i, bytes, nullptr)); }
  // launches groups x work-groups of local work-items each (1-D); the kernel sees get_global_id(0) < groups*local
  void launch(uint32_t groups, uint32_t local) {
    if (local != gx || gy != 1 || gz != 1) {
      ZE_OK(zeKernelSetGroupSize(k, local, 1, 1));
      gx = local, gy = gz = 1;
    }
    ze_group_count_t gc{groups, 1, 1};
    const ze_result_t r = zeCommandListAppendLaunchKernel(d->cl, k, &gc, nullptr, 0, nullptr);
    if (r != ZE_RESULT_SUCCESS) {
      char b[200];
      snprintf(b, sizeof b, "Level Zero error 0x%x launching %s (%u groups x %u)", r, name.c_str(), groups, local);
      throw Error(b);
    }
  }
};

}  // namespace gpu
