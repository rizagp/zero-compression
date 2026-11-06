// zero: the thin layer that differs between Windows and Linux. Everything else is standard C++20.
#pragma once
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <filesystem>

#ifdef _WIN32
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#define ZERO_CALL __stdcall
typedef HMODULE zero_lib_t;
inline zero_lib_t zero_lib_open(const char* name) { return LoadLibraryA(name); }
inline void* zero_lib_sym(zero_lib_t h, const char* name) { return (void*)GetProcAddress(h, name); }
inline void zero_lib_close(zero_lib_t h) { if (h) FreeLibrary(h); }
inline const char* const kZeLoaderNames[] = {"ze_loader.dll", nullptr};
inline const char* const kOpenCLNames[] = {"OpenCL.dll", nullptr};
#else
#include <dlfcn.h>
#define ZERO_CALL
typedef void* zero_lib_t;
inline zero_lib_t zero_lib_open(const char* name) { return dlopen(name, RTLD_NOW | RTLD_LOCAL); }
inline void* zero_lib_sym(zero_lib_t h, const char* name) { return dlsym(h, name); }
inline void zero_lib_close(zero_lib_t h) { if (h) dlclose(h); }
inline const char* const kZeLoaderNames[] = {"libze_loader.so.1", "libze_loader.so", nullptr};
inline const char* const kOpenCLNames[] = {"libOpenCL.so.1", "libOpenCL.so", nullptr};
#endif

// the first of several library names that loads
inline zero_lib_t zero_lib_open_any(const char* const* names) {
  for (; *names; ++names)
    if (zero_lib_t h = zero_lib_open(*names)) return h;
  return nullptr;
}

// Where zero keeps its compiled kernel cache: %LOCALAPPDATA%\zero on Windows, $XDG_CACHE_HOME/zero or
// ~/.cache/zero on Linux. Empty when no such folder can be named (the cache is then off).
inline std::string zero_cache_dir() {
#ifdef _WIN32
  const char* base = getenv("LOCALAPPDATA");
  if (!base || !*base) base = getenv("TEMP");
  if (!base || !*base) return "";
  return std::string(base) + "\\zero";
#else
  if (const char* x = getenv("XDG_CACHE_HOME"); x && *x) return std::string(x) + "/zero";
  if (const char* h = getenv("HOME"); h && *h) return std::string(h) + "/.cache/zero";
  return "";
#endif
}

// A std::filesystem::path from a UTF-8 string, and back. On Windows the exe runs with the UTF-8 code page (the
// embedded manifest), so narrow strings are UTF-8 there too.
inline std::filesystem::path zero_path(const std::string& utf8) { return std::filesystem::path(utf8); }
inline std::string zero_utf8(const std::filesystem::path& p) { return p.string(); }

// a monotonic clock in seconds
#include <chrono>
inline double zero_now() {
  return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count();
}
