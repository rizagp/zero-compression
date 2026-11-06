// zero: the GPU, with the CPU behind it. A GPU can fail in the middle of a job: a lost device, because of a fault here or
// in another program on the same GPU, or a driver reset. Both codecs read and write the same format, so the batch is done
// again on the CPU and so are the ones after it: the job finishes and the archive is right. A lost device stays lost for
// the process, so there is no going back to the GPU. ZERO_FAIL_GPU=n fails the GPU's n+1st batch, for tests.
#pragma once
#include "gpu_codec.h"

namespace zero {

struct SafeCodec : Codec {
  std::unique_ptr<GpuCodec> gpu;
  CpuCodec cpu;
  bool gpu_ok = true;
  int fail_after = -1;

  explicit SafeCodec(std::unique_ptr<GpuCodec> g) : gpu(std::move(g)) {
    if (const char* e = getenv("ZERO_FAIL_GPU")) fail_after = atoi(e);
  }
  // a Report lives as long as its job, and the codec longer: both must always have the current one
  void set_report(Report* r) { rep = r; gpu->rep = r; }
  std::string name() const override { return gpu_ok ? gpu->name() : cpu.name(); }
  size_t batch_bytes(uint32_t bl) const override { return gpu_ok ? gpu->batch_bytes(bl) : cpu.batch_bytes(bl); }
  size_t batch_bytes_decode(uint32_t bl) const override { return gpu_ok ? gpu->batch_bytes_decode(bl) : cpu.batch_bytes_decode(bl); }
  void prepare(uint32_t bl) override { attempt([&] { gpu->prepare(bl); }); }
  void compress(const uint8_t* in, size_t n, uint32_t bl, std::vector<BlockOut>& out) override {
    if (!attempt([&] { gpu->compress(in, n, bl, out); })) cpu.compress(in, n, bl, out);
  }
  const uint8_t* decompress(const std::vector<BlockIn>& blocks) override {
    const uint8_t* raw = nullptr;
    if (attempt([&] { raw = gpu->decompress(blocks); })) return raw;
    return cpu.decompress(blocks);  // (a damaged block is reported from here too)
  }

 private:
  Report* rep = nullptr;
  template <class F> bool attempt(const F& f) {
    if (!gpu_ok) return false;
    try {
      if (fail_after == 0) throw gpu::Error("the GPU was reset (device lost, simulated by ZERO_FAIL_GPU)");
      if (fail_after > 0) --fail_after;
      f();
      return true;
    } catch (const zb::Error&) {
      return false;  // (the data, not the GPU: the CPU decoder says what is wrong, or does better)
    } catch (const gpu::Error& e) {
      gpu_ok = false;
      if (rep) rep->say(std::string("GPU stopped working (") + e.what() + "); the CPU finishes the job");
      return false;
    }
  }
};

}  // namespace zero
