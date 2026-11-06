// zero: the Windows window (Win32, no framework). Files and folders are added by drag and drop or the buttons,
// compressed into one .zero archive; a .zero archive is extracted or tested. The work runs on a thread and reports
// progress and log lines to the window. Built from main.cpp with ZERO_GUI; the paths given on the command line are
// added to the list at start ("Open with", or dropping files on the exe).
#pragma once
#include <windows.h>
#include <commctrl.h>
#include <shellapi.h>
#include <shobjidl.h>
#include <atomic>
#include <memory>
#include <string>
#include <thread>
#include <vector>

namespace zero {
namespace gui {

enum : int {
  ID_LIST = 101, ID_ADD_FILES, ID_ADD_FOLDER, ID_REMOVE, ID_CLEAR, ID_OUT, ID_BROWSE, ID_BLOCK, ID_CPU, ID_COMPRESS,
  ID_EXTRACT, ID_TEST, ID_CANCEL, ID_LOG, ID_PROGRESS, ID_STATUS, ID_LBL_OUT, ID_LBL_BLOCK
};
constexpr UINT WM_LOG = WM_APP + 1;       // lParam: std::string* (the receiver deletes it)
constexpr UINT WM_PROGRESS = WM_APP + 2;  // wParam: done per mille
constexpr UINT WM_DONE = WM_APP + 3;      // lParam: std::string* summary (or empty), wParam: 1 = failed

inline std::string narrow(const wchar_t* w) {
  if (!w || !*w) return "";
  const int n = WideCharToMultiByte(CP_UTF8, 0, w, -1, nullptr, 0, nullptr, nullptr);
  std::string s(n > 0 ? n - 1 : 0, '\0');
  if (n > 1) WideCharToMultiByte(CP_UTF8, 0, w, -1, s.data(), n, nullptr, nullptr);
  return s;
}
inline std::wstring widen(const std::string& s) {
  if (s.empty()) return L"";
  const int n = MultiByteToWideChar(CP_UTF8, 0, s.c_str(), -1, nullptr, 0);
  std::wstring w(n > 0 ? n - 1 : 0, L'\0');
  if (n > 1) MultiByteToWideChar(CP_UTF8, 0, s.c_str(), -1, w.data(), n);
  return w;
}

struct App {
  HWND wnd = nullptr, list = nullptr, out = nullptr, block = nullptr, cpu = nullptr, log = nullptr, progress = nullptr,
       status = nullptr;
  HWND buttons[8] = {};
  HFONT font = nullptr;
  int dpi = 96;
  std::vector<std::string> items;
  std::thread worker;
  std::atomic<bool> cancel{false};
  bool busy = false;
  std::unique_ptr<GpuCodec> gpu;  // kept between operations (the kernels stay loaded); used on the worker only
  std::string gpu_error;

  int px(int v) const { return MulDiv(v, dpi, 96); }

  void post_log(const std::string& s) { PostMessageW(wnd, WM_LOG, 0, (LPARAM) new std::string(s)); }
  void append_log(const std::string& s) {
    const std::wstring w = widen(s) + L"\r\n";
    const int n = GetWindowTextLengthW(log);
    SendMessageW(log, EM_SETSEL, n, n);
    SendMessageW(log, EM_REPLACESEL, FALSE, (LPARAM)w.c_str());
  }
  void set_status(const std::string& s) { SetWindowTextW(status, widen(s).c_str()); }

  // ---- the file dialogs (IFileDialog): files (several), a folder, or a file to save
  std::vector<std::string> pick(bool folder, bool multi, bool save, const wchar_t* def_name = nullptr) {
    std::vector<std::string> r;
    IFileDialog* dlg = nullptr;
    const HRESULT hr = save ? CoCreateInstance(CLSID_FileSaveDialog, nullptr, CLSCTX_INPROC_SERVER, IID_PPV_ARGS(&dlg))
                            : CoCreateInstance(CLSID_FileOpenDialog, nullptr, CLSCTX_INPROC_SERVER, IID_PPV_ARGS(&dlg));
    if (FAILED(hr) || !dlg) return r;
    DWORD opts = 0;
    dlg->GetOptions(&opts);
    opts |= FOS_FORCEFILESYSTEM | FOS_NOCHANGEDIR;
    if (folder) opts |= FOS_PICKFOLDERS;
    if (multi) opts |= FOS_ALLOWMULTISELECT;
    if (save) opts |= FOS_OVERWRITEPROMPT;
    dlg->SetOptions(opts);
    if (save) {
      COMDLG_FILTERSPEC spec[] = {{L"zero archive (*.zero)", L"*.zero"}, {L"All files", L"*.*"}};
      dlg->SetFileTypes(2, spec);
      dlg->SetDefaultExtension(L"zero");
      if (def_name) dlg->SetFileName(def_name);
    } else if (!folder && !multi) {
      COMDLG_FILTERSPEC spec[] = {{L"zero archive (*.zero)", L"*.zero"}, {L"All files", L"*.*"}};
      dlg->SetFileTypes(2, spec);
    }
    if (SUCCEEDED(dlg->Show(wnd))) {
      if (multi) {
        IFileOpenDialog* od = nullptr;
        if (SUCCEEDED(dlg->QueryInterface(IID_PPV_ARGS(&od))) && od) {
          IShellItemArray* arr = nullptr;
          if (SUCCEEDED(od->GetResults(&arr)) && arr) {
            DWORD n = 0;
            arr->GetCount(&n);
            for (DWORD i = 0; i < n; ++i) {
              IShellItem* it = nullptr;
              if (SUCCEEDED(arr->GetItemAt(i, &it)) && it) {
                PWSTR path = nullptr;
                if (SUCCEEDED(it->GetDisplayName(SIGDN_FILESYSPATH, &path)) && path) { r.push_back(narrow(path)); CoTaskMemFree(path); }
                it->Release();
              }
            }
            arr->Release();
          }
          od->Release();
        }
      } else {
        IShellItem* it = nullptr;
        if (SUCCEEDED(dlg->GetResult(&it)) && it) {
          PWSTR path = nullptr;
          if (SUCCEEDED(it->GetDisplayName(SIGDN_FILESYSPATH, &path)) && path) { r.push_back(narrow(path)); CoTaskMemFree(path); }
          it->Release();
        }
      }
    }
    dlg->Release();
    return r;
  }

  // ---- the list of inputs
  void add_item(const std::string& p) {
    for (const auto& s : items) if (s == p) return;
    items.push_back(p);
    SendMessageW(list, LB_ADDSTRING, 0, (LPARAM)widen(p).c_str());
    if (items.size() == 1) suggest_output();
  }
  void remove_selected() {
    int n = (int)SendMessageW(list, LB_GETSELCOUNT, 0, 0);
    if (n <= 0) return;
    std::vector<int> sel(n);
    SendMessageW(list, LB_GETSELITEMS, n, (LPARAM)sel.data());
    for (int i = n - 1; i >= 0; --i) {
      SendMessageW(list, LB_DELETESTRING, sel[i], 0);
      items.erase(items.begin() + sel[i]);
    }
  }
  void clear_items() { items.clear(); SendMessageW(list, LB_RESETCONTENT, 0, 0); }
  bool only_archive() const { return items.size() == 1 && items[0].size() > 5 && items[0].compare(items[0].size() - 5, 5, ".zero") == 0; }
  void suggest_output() {
    if (items.empty() || only_archive()) return;
    fs::path p = zero_path(items[0]);
    while (!p.empty() && !p.has_filename()) p = p.parent_path();
    fs::path o = p.parent_path() / (zero_utf8(p.filename()) + ".zero");
    SetWindowTextW(out, o.wstring().c_str());
  }
  std::string output_path() {
    wchar_t buf[4096];
    GetWindowTextW(out, buf, 4096);
    return narrow(buf);
  }
  uint32_t block_log() { return 16 + (uint32_t)SendMessageW(block, CB_GETCURSEL, 0, 0); }
  bool use_cpu() { return SendMessageW(cpu, BM_GETCHECK, 0, 0) == BST_CHECKED; }

  // ---- the work
  void set_busy(bool b) {
    busy = b;
    for (int id : {ID_ADD_FILES, ID_ADD_FOLDER, ID_REMOVE, ID_CLEAR, ID_BROWSE, ID_COMPRESS, ID_EXTRACT, ID_TEST})
      EnableWindow(GetDlgItem(wnd, id), !b);
    EnableWindow(GetDlgItem(wnd, ID_CANCEL), b);
    EnableWindow(list, !b);
    EnableWindow(out, !b);
    EnableWindow(block, !b);
    EnableWindow(cpu, !b);
    if (!b) SendMessageW(progress, PBM_SETPOS, 0, 0);
  }
  Codec& codec_for(bool want_cpu, Report& rep, std::unique_ptr<CpuCodec>& cpu_holder) {
    if (!want_cpu) {
      if (!gpu && gpu_error.empty()) {
        try {
          gpu = std::make_unique<GpuCodec>();
          gpu->open(rep);
        } catch (const std::exception& e) {
          gpu.reset();
          gpu_error = e.what();
          rep.say(std::string("GPU unavailable: ") + gpu_error + "; using the CPU");
        }
      } else if (gpu) {
        gpu->rep = &rep;
      }
      if (gpu) return *gpu;
    }
    cpu_holder = std::make_unique<CpuCodec>();
    return *cpu_holder;
  }
  void run(std::function<std::string(Report&)> job) {
    if (busy) return;
    if (worker.joinable()) worker.join();
    cancel = false;
    set_busy(true);
    SendMessageW(progress, PBM_SETRANGE32, 0, 1000);
    worker = std::thread([this, job] {
      Report rep;
      rep.cancel = &cancel;
      rep.log = [this](const std::string& s) { post_log(s); };
      double last = 0;
      rep.progress = [this, &last](uint64_t d, uint64_t t) {
        const double now = zero_now();
        if (d < t && now - last < 0.05) return;
        last = now;
        PostMessageW(wnd, WM_PROGRESS, (WPARAM)(t ? d * 1000 / t : 1000), 0);
      };
      std::string summary;
      bool failed = false;
      try {
        summary = job(rep);
      } catch (const Cancelled&) {
        summary = "Cancelled.";
        failed = true;
      } catch (const std::exception& e) {
        summary = std::string("Error: ") + e.what();
        failed = true;
      }
      PostMessageW(wnd, WM_DONE, failed ? 1 : 0, (LPARAM) new std::string(summary));
    });
  }
  void do_compress() {
    if (items.empty()) { MessageBoxW(wnd, L"Add files or folders first (drag them onto the window).", L"zero", MB_ICONINFORMATION); return; }
    const std::string outp = output_path();
    if (outp.empty()) { MessageBoxW(wnd, L"Choose where to write the archive.", L"zero", MB_ICONINFORMATION); return; }
    for (const auto& it : items)
      if (it == outp) { MessageBoxW(wnd, L"The archive cannot be one of its own inputs.", L"zero", MB_ICONWARNING); return; }
    std::vector<std::string> inputs = items;
    const uint32_t bl = block_log();
    const bool want_cpu = use_cpu();
    set_status("Compressing...");
    run([this, inputs, outp, bl, want_cpu](Report& rep) {
      std::unique_ptr<CpuCodec> holder;
      Codec& c = codec_for(want_cpu, rep, holder);
      rep.say("codec: " + c.name());
      const CompressStats st = compress_archive(inputs, outp, c, bl, rep);
      char b[512];
      snprintf(b, sizeof b, "Done: %s -> %s (%.1f %%), %zu files, %zu folders, %.1f s (%s)", human(st.raw).c_str(),
               human(st.compressed).c_str(), st.raw ? 100.0 * st.compressed / st.raw : 0.0, st.files, st.folders, st.seconds,
               rate(st.raw, st.seconds).c_str());
      return std::string(b) + "\n" + outp;
    });
  }
  std::string archive_to_open() {
    if (only_archive()) return items[0];
    std::vector<std::string> r = pick(false, false, false);
    return r.empty() ? "" : r[0];
  }
  void do_extract() {
    const std::string arc = archive_to_open();
    if (arc.empty()) return;
    std::vector<std::string> dir = pick(true, false, false);
    if (dir.empty()) return;
    const bool want_cpu = use_cpu();
    set_status("Extracting...");
    run([this, arc, dir, want_cpu](Report& rep) {
      std::unique_ptr<CpuCodec> holder;
      Codec& c = codec_for(want_cpu, rep, holder);
      rep.say("codec: " + c.name());
      const ExtractStats st = extract_archive(arc, dir[0], c, rep);
      char b[512];
      snprintf(b, sizeof b, "Extracted %zu files, %zu folders, %s in %.1f s (%s); every checksum matched", st.files, st.folders,
               human(st.raw).c_str(), st.seconds, rate(st.raw, st.seconds).c_str());
      return std::string(b) + "\n" + dir[0];
    });
  }
  void do_test() {
    const std::string arc = archive_to_open();
    if (arc.empty()) return;
    const bool want_cpu = use_cpu();
    set_status("Testing...");
    run([this, arc, want_cpu](Report& rep) {
      std::unique_ptr<CpuCodec> holder;
      Codec& c = codec_for(want_cpu, rep, holder);
      rep.say("codec: " + c.name());
      const ExtractStats st = test_archive(arc, c, rep);
      char b[512];
      snprintf(b, sizeof b, "OK: %zu files, %zu folders, %s in %llu blocks, %.1f s (%s); every checksum matched", st.files,
               st.folders, human(st.raw).c_str(), (unsigned long long)st.blocks, st.seconds, rate(st.raw, st.seconds).c_str());
      return std::string(b);
    });
  }

  // ---- the window
  HWND make(const wchar_t* cls, const wchar_t* text, DWORD style, int id, DWORD ex = 0) {
    HWND h = CreateWindowExW(ex, cls, text, style | WS_CHILD | WS_VISIBLE, 0, 0, 10, 10, wnd, (HMENU)(INT_PTR)id, nullptr, nullptr);
    SendMessageW(h, WM_SETFONT, (WPARAM)font, TRUE);
    return h;
  }
  void create_controls() {
    dpi = GetDpiForWindow(wnd);
    font = CreateFontW(-MulDiv(9, dpi, 72), 0, 0, 0, FW_NORMAL, 0, 0, 0, DEFAULT_CHARSET, 0, 0, CLEARTYPE_QUALITY, 0, L"Segoe UI");
    list = make(L"LISTBOX", L"", LBS_EXTENDEDSEL | LBS_NOINTEGRALHEIGHT | WS_VSCROLL | WS_HSCROLL | WS_TABSTOP, ID_LIST, WS_EX_CLIENTEDGE);
    buttons[0] = make(L"BUTTON", L"Add files...", WS_TABSTOP, ID_ADD_FILES);
    buttons[1] = make(L"BUTTON", L"Add folder...", WS_TABSTOP, ID_ADD_FOLDER);
    buttons[2] = make(L"BUTTON", L"Remove", WS_TABSTOP, ID_REMOVE);
    buttons[3] = make(L"BUTTON", L"Clear", WS_TABSTOP, ID_CLEAR);
    make(L"STATIC", L"Archive:", 0, ID_LBL_OUT);
    out = make(L"EDIT", L"", ES_AUTOHSCROLL | WS_TABSTOP, ID_OUT, WS_EX_CLIENTEDGE);
    buttons[4] = make(L"BUTTON", L"Browse...", WS_TABSTOP, ID_BROWSE);
    make(L"STATIC", L"Block:", 0, ID_LBL_BLOCK);
    block = make(L"COMBOBOX", L"", CBS_DROPDOWNLIST | WS_TABSTOP, ID_BLOCK);
    for (const wchar_t* s : {L"64 KiB", L"128 KiB", L"256 KiB", L"512 KiB", L"1 MiB"}) SendMessageW(block, CB_ADDSTRING, 0, (LPARAM)s);
    SendMessageW(block, CB_SETCURSEL, 3, 0);
    cpu = make(L"BUTTON", L"CPU codec", BS_AUTOCHECKBOX | WS_TABSTOP, ID_CPU);
    buttons[5] = make(L"BUTTON", L"Compress", BS_DEFPUSHBUTTON | WS_TABSTOP, ID_COMPRESS);
    buttons[6] = make(L"BUTTON", L"Extract...", WS_TABSTOP, ID_EXTRACT);
    buttons[7] = make(L"BUTTON", L"Test...", WS_TABSTOP, ID_TEST);
    make(L"BUTTON", L"Cancel", WS_TABSTOP | WS_DISABLED, ID_CANCEL);
    progress = make(PROGRESS_CLASSW, L"", 0, ID_PROGRESS);
    status = make(L"STATIC", L"Drop files and folders here, then Compress. Drop a .zero archive to extract or test it.", SS_ENDELLIPSIS, ID_STATUS);
    log = make(L"EDIT", L"", ES_MULTILINE | ES_READONLY | ES_AUTOVSCROLL | WS_VSCROLL, ID_LOG, WS_EX_CLIENTEDGE);
    DragAcceptFiles(wnd, TRUE);
  }
  void layout() {
    RECT rc;
    GetClientRect(wnd, &rc);
    const int W = rc.right, H = rc.bottom, m = px(10), bh = px(26), bw = px(96), gap = px(6);
    auto place = [&](HWND h, int x, int y, int w, int hh) { MoveWindow(h, x, y, w, hh, TRUE); };
    const int list_h = (H - 7 * m - 3 * bh - px(18) - px(20)) * 55 / 100;
    place(list, m, m, W - 3 * m - bw, list_h);
    int y = m;
    for (int i = 0; i < 4; ++i, y += bh + gap) place(buttons[i], W - m - bw, y, bw, bh);
    y = m + list_h + m;
    place(GetDlgItem(wnd, ID_LBL_OUT), m, y + px(4), px(56), px(18));
    place(out, m + px(60), y, W - 3 * m - bw - px(60), bh);
    place(buttons[4], W - m - bw, y, bw, bh);
    y += bh + m;
    place(GetDlgItem(wnd, ID_LBL_BLOCK), m, y + px(4), px(44), px(18));
    place(block, m + px(48), y, px(96), bh * 8);
    place(cpu, m + px(156), y + px(3), px(100), px(20));
    int x = W - m - bw;
    place(GetDlgItem(wnd, ID_CANCEL), x, y, bw, bh);
    x -= bw + gap;
    for (int i = 7; i >= 5; --i, x -= bw + gap) place(buttons[i], x, y, bw, bh);
    y += bh + m;
    place(progress, m, y, W - 2 * m, px(14));
    y += px(14) + gap;
    place(status, m, y, W - 2 * m, px(18));
    y += px(18) + gap;
    place(log, m, y, W - 2 * m, H - y - m);
  }
  void on_drop(HDROP drop) {
    const UINT n = DragQueryFileW(drop, 0xFFFFFFFF, nullptr, 0);
    for (UINT i = 0; i < n; ++i) {
      wchar_t buf[4096];
      if (DragQueryFileW(drop, i, buf, 4096)) add_item(narrow(buf));
    }
    DragFinish(drop);
    if (only_archive()) set_status("A .zero archive: Extract... or Test...");
  }
  void on_command(int id) {
    switch (id) {
      case ID_ADD_FILES: for (const auto& p : pick(false, true, false)) add_item(p); break;
      case ID_ADD_FOLDER: for (const auto& p : pick(true, true, false)) add_item(p); break;
      case ID_REMOVE: remove_selected(); break;
      case ID_CLEAR: clear_items(); break;
      case ID_BROWSE: {
        std::wstring def = L"archive";
        if (!items.empty()) { fs::path p = zero_path(items[0]); while (!p.empty() && !p.has_filename()) p = p.parent_path(); def = p.filename().wstring(); }
        std::vector<std::string> r = pick(false, false, true, def.c_str());
        if (!r.empty()) SetWindowTextW(out, widen(r[0]).c_str());
        break;
      }
      case ID_COMPRESS: do_compress(); break;
      case ID_EXTRACT: do_extract(); break;
      case ID_TEST: do_test(); break;
      case ID_CANCEL: cancel = true; set_status("Cancelling..."); break;
    }
  }
  LRESULT proc(UINT msg, WPARAM wp, LPARAM lp) {
    switch (msg) {
      case WM_CREATE: create_controls(); layout(); return 0;
      case WM_SIZE: layout(); return 0;
      case WM_GETMINMAXINFO: { MINMAXINFO* mm = (MINMAXINFO*)lp; mm->ptMinTrackSize.x = px(560); mm->ptMinTrackSize.y = px(420); return 0; }
      case WM_DROPFILES: on_drop((HDROP)wp); return 0;
      case WM_COMMAND: on_command(LOWORD(wp)); return 0;
      case WM_LOG: { std::string* s = (std::string*)lp; append_log(*s); delete s; return 0; }
      case WM_PROGRESS: SendMessageW(progress, PBM_SETPOS, wp, 0); return 0;
      case WM_DONE: {
        std::string* s = (std::string*)lp;
        set_busy(false);
        const size_t nl = s->find('\n');
        set_status(s->substr(0, nl));
        append_log(*s);
        delete s;
        if (worker.joinable()) worker.join();
        return 0;
      }
      case WM_CLOSE:
        if (busy) { cancel = true; set_status("Cancelling, then closing..."); PostMessageW(wnd, WM_CLOSE, 1, 0); return 0; }
        DestroyWindow(wnd);
        return 0;
      case WM_DESTROY: PostQuitMessage(0); return 0;
    }
    return DefWindowProcW(wnd, msg, wp, lp);
  }
};

inline App* g_app = nullptr;
inline LRESULT CALLBACK wndproc(HWND h, UINT msg, WPARAM wp, LPARAM lp) {
  if (!g_app) return DefWindowProcW(h, msg, wp, lp);
  if (!g_app->wnd) g_app->wnd = h;  // (WM_GETMINMAXINFO and WM_NCCREATE arrive before WM_CREATE)
  return g_app->proc(msg, wp, lp);
}

inline int run_gui(const std::vector<std::string>& args) {
  SetProcessDpiAwarenessContext(DPI_AWARENESS_CONTEXT_PER_MONITOR_AWARE_V2);
  CoInitializeEx(nullptr, COINIT_APARTMENTTHREADED);
  INITCOMMONCONTROLSEX icc{sizeof icc, ICC_PROGRESS_CLASS | ICC_STANDARD_CLASSES};
  InitCommonControlsEx(&icc);
  App app;
  g_app = &app;
  WNDCLASSW wc{};
  wc.lpfnWndProc = wndproc;
  wc.hInstance = GetModuleHandleW(nullptr);
  wc.lpszClassName = L"ZeroMainWindow";
  wc.hCursor = LoadCursorW(nullptr, MAKEINTRESOURCEW(32512) /* IDC_ARROW */);
  wc.hbrBackground = (HBRUSH)(COLOR_WINDOW + 1);
  wc.hIcon = LoadIconW(nullptr, MAKEINTRESOURCEW(32512) /* IDI_APPLICATION */);
  RegisterClassW(&wc);
  const int dpi = GetDpiForSystem();
  HWND h = CreateWindowExW(0, wc.lpszClassName, L"zero - GPU compression", WS_OVERLAPPEDWINDOW, CW_USEDEFAULT, CW_USEDEFAULT,
                           MulDiv(760, dpi, 96), MulDiv(560, dpi, 96), nullptr, nullptr, wc.hInstance, nullptr);
  if (!h) return 1;
  for (const auto& a : args) app.add_item(a);
  if (app.only_archive()) app.set_status("A .zero archive: Extract... or Test...");
  ShowWindow(h, SW_SHOW);
  MSG msg;
  while (GetMessageW(&msg, nullptr, 0, 0)) {
    if (msg.message == WM_KEYDOWN && msg.wParam == VK_DELETE && msg.hwnd == app.list) { app.remove_selected(); continue; }
    if (!IsDialogMessageW(h, &msg)) { TranslateMessage(&msg); DispatchMessageW(&msg); }
  }
  if (app.worker.joinable()) app.worker.join();
  app.gpu.reset();
  CoUninitialize();
  return 0;
}

}  // namespace gui
}  // namespace zero

int WINAPI wWinMain(HINSTANCE, HINSTANCE, PWSTR, int) {
  int argc = 0;
  LPWSTR* argv = CommandLineToArgvW(GetCommandLineW(), &argc);
  std::vector<std::string> args;
  for (int i = 1; i < argc; ++i) args.push_back(zero::gui::narrow(argv[i]));
  LocalFree(argv);
  return zero::gui::run_gui(args);
}
