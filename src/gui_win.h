// zero: the Windows window, in the Windows 11 style: Mica behind the content, the system's light or dark theme and
// accent color, rounded corners, Segoe UI Variable. Win32, Direct2D and DirectWrite, which every Windows has: no
// framework, and the exe stays one file. The whole client area is painted here, the widgets drawn and hit-tested from
// the same list each paint.
//
// One thing at a time, decided by what the person gives it: a start screen; files and folders lead to "New archive",
// a .zero archive to its contents; then progress; then the result with the next step. The choices a person rarely needs
// (block size, processor) stay behind "Advanced". The work runs on a thread. Paths on the command line are opened at
// start ("Open with", or dropping files on the exe).
//
// Explorer: a button on the start screen (or --install-shell, --uninstall-shell) registers .zero and the right-click
// commands for this user only, with no administrator rights. Windows 11 shows them under "Show more options": its first
// menu is only for signed packages. Each selected file starts a process with --add; they gather in one window.
//
// The installer is this same program named zero-setup.exe: it copies itself and the command line it holds as a resource
// to %LOCALAPPDATA%\Programs\zero, for this user only. --silent installs with no window, --uninstall removes it.
#pragma once
#include <windows.h>
#include <dwmapi.h>
#undef DrawText  // (windows.h's macro would rename Direct2D's method)
#include <d2d1.h>
#include <dwrite.h>
#include <shellapi.h>
#include <shobjidl.h>
#include <shlobj.h>
#include <algorithm>
#include <atomic>
#include <functional>
#include <memory>
#include <string>
#include <thread>
#include <vector>

namespace zero {
namespace gui {

enum : int {
  ID_NONE = 0, ID_ADD_FILES = 102, ID_ADD_FOLDER = 103, ID_OPEN = 104, ID_BROWSE = 107, ID_ADVANCED = 108,
  ID_COMPRESS = 110, ID_EXTRACT = 111, ID_TEST = 112, ID_CANCEL = 113, ID_BACK = 114, ID_SHOW = 115, ID_DETAILS = 116,
  ID_AGAIN = 117, ID_SHELL = 118, ID_SETUP_MENU = 119, ID_SETUP_PATH = 120, ID_SETUP_GO = 121, ID_RUN = 122, ID_LEVEL = 200, ID_BLOCK = 210, ID_PROC = 220, ID_ROW = 1000
};
constexpr UINT WM_LOG = WM_APP + 1;       // lParam: std::string* (the receiver deletes it)
constexpr UINT WM_PROGRESS = WM_APP + 2;  // the worker's counters changed
constexpr UINT WM_DONE = WM_APP + 3;      // lParam: DoneInfo* (the receiver deletes it)
constexpr UINT WM_SCANNED = WM_APP + 4;   // lParam: Scanned* (a folder's size is known)
// for the tests: wParam 1 selects block size lParam; wParam 2 returns 0 idle, 1 busy, 2 last job failed; wParam 3 returns
// the screen (0 start, 1 new archive, 2 an archive, 3 working, 4 result); wParam 4 returns the number of items.
// WM_COPYDATA with dwData 1 sets the archive's path. With dwData 2: paths from another process (NUL-separated).
constexpr UINT WM_TEST = WM_APP + 20;

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
inline bool is_archive(const std::string& p) {
  return p.size() > 5 && _stricmp(p.c_str() + p.size() - 5, ".zero") == 0;
}
inline std::string base_name(const std::string& p) {
  fs::path q = zero_path(p);
  while (!q.empty() && !q.has_filename()) q = q.parent_path();
  const std::string n = zero_utf8(q.filename());
  return n.empty() ? p : n;
}
// "name.zero", or "name (2).zero" when that exists
inline std::string unique_path(const fs::path& dir, const std::string& stem, const std::string& ext) {
  std::error_code ec;
  for (int i = 1; i < 1000; ++i) {
    const std::string name = i == 1 ? stem + ext : stem + " (" + std::to_string(i) + ")" + ext;
    const fs::path p = dir / zero_path(name);
    if (!fs::exists(p, ec)) return zero_utf8(p);
  }
  return zero_utf8(dir / zero_path(stem + ext));
}

// the entries in HKEY_CURRENT_USER\Software\Classes
namespace shell {
inline std::wstring exe_path() {
  wchar_t b[4096];
  GetModuleFileNameW(nullptr, b, 4096);
  return b;
}
inline void put(const std::wstring& sub, const wchar_t* name, const std::wstring& v) {
  HKEY k = nullptr;
  if (RegCreateKeyExW(HKEY_CURRENT_USER, (L"Software\\Classes\\" + sub).c_str(), 0, nullptr, 0, KEY_SET_VALUE, nullptr, &k, nullptr) != ERROR_SUCCESS) return;
  RegSetValueExW(k, name, 0, REG_SZ, (const BYTE*)v.c_str(), (DWORD)((v.size() + 1) * sizeof(wchar_t)));
  RegCloseKey(k);
}
inline std::wstring get(const std::wstring& sub, const wchar_t* name = nullptr) {
  wchar_t b[4096] = L"";
  DWORD n = sizeof b;
  if (RegGetValueW(HKEY_CURRENT_USER, (L"Software\\Classes\\" + sub).c_str(), name, RRF_RT_REG_SZ, nullptr, b, &n) != ERROR_SUCCESS) return L"";
  return b;
}
inline void verb(const std::wstring& key, const wchar_t* label, const std::wstring& command, const std::wstring& icon) {
  put(key, nullptr, label);
  put(key, L"Icon", icon);
  put(key + L"\\command", nullptr, command);
}
constexpr const wchar_t* kProgId = L"zero.Archive";
inline bool installed() { return get(std::wstring(kProgId) + L"\\shell\\open\\command").find(exe_path()) != std::wstring::npos; }
inline void install() {
  const std::wstring exe = exe_path(), q = L"\"" + exe + L"\"", icon = exe + L",0", prog = kProgId;
  put(L".zero", nullptr, prog);
  put(L".zero\\OpenWithProgids", kProgId, L"");
  put(prog, nullptr, L"zero archive");
  put(prog + L"\\DefaultIcon", nullptr, icon);
  put(prog + L"\\shell", nullptr, L"open");
  verb(prog + L"\\shell\\open", L"Open with zero", q + L" \"%1\"", icon);
  verb(prog + L"\\shell\\zero.extract", L"Extract here", q + L" --extract-here \"%1\"", icon);
  for (const wchar_t* base : {L"*", L"Directory"}) {
    const std::wstring key = std::wstring(base) + L"\\shell\\zero.compress";
    verb(key, L"Compress with zero", q + L" --add \"%1\"", icon);
    put(key, L"MultiSelectModel", L"Player");
  }
  put(L"*\\shell\\zero.compress", L"AppliesTo", L"NOT System.FileExtension:=.zero");
  SHChangeNotify(SHCNE_ASSOCCHANGED, SHCNF_IDLIST, nullptr, nullptr);
}
inline void uninstall() {
  const std::wstring classes = L"Software\\Classes\\";
  RegDeleteTreeW(HKEY_CURRENT_USER, (classes + L"*\\shell\\zero.compress").c_str());
  RegDeleteTreeW(HKEY_CURRENT_USER, (classes + L"Directory\\shell\\zero.compress").c_str());
  RegDeleteTreeW(HKEY_CURRENT_USER, (classes + kProgId).c_str());
  if (get(L".zero") == kProgId) RegDeleteTreeW(HKEY_CURRENT_USER, (classes + L".zero").c_str());
  SHChangeNotify(SHCNE_ASSOCCHANGED, SHCNF_IDLIST, nullptr, nullptr);
}
}  // namespace shell

// installing: everything lives in this user's profile, so no administrator is needed
namespace setup {
constexpr const wchar_t* kUninstallKey = L"Software\\Microsoft\\Windows\\CurrentVersion\\Uninstall\\zero";
inline std::wstring env(const wchar_t* name) { const wchar_t* v = _wgetenv(name); return v ? v : L""; }
inline std::wstring dir() { return env(L"LOCALAPPDATA") + L"\\Programs\\zero"; }
inline std::wstring shortcut() { return env(L"APPDATA") + L"\\Microsoft\\Windows\\Start Menu\\Programs\\zero.lnk"; }
inline bool same_path(const std::wstring& a, const std::wstring& b) {
  std::error_code ec;
  return fs::equivalent(fs::path(a), fs::path(b), ec) && !ec;
}
inline bool is_setup_exe() { return _wcsnicmp(fs::path(shell::exe_path()).filename().c_str(), L"zero-setup", 10) == 0; }
inline bool running_installed() { return same_path(fs::path(shell::exe_path()).parent_path().wstring(), dir()); }
inline bool installed() {
  std::error_code ec;
  return fs::exists(fs::path(dir()) / L"zero-gui.exe", ec);
}
inline void set_str(HKEY k, const wchar_t* n, const std::wstring& v) { RegSetValueExW(k, n, 0, REG_SZ, (const BYTE*)v.c_str(), (DWORD)((v.size() + 1) * sizeof(wchar_t))); }
inline void set_dword(HKEY k, const wchar_t* n, DWORD v) { RegSetValueExW(k, n, 0, REG_DWORD, (const BYTE*)&v, sizeof v); }
// the user's PATH with our folder added or removed
inline void edit_path(bool add) {
  HKEY k = nullptr;
  if (RegCreateKeyExW(HKEY_CURRENT_USER, L"Environment", 0, nullptr, 0, KEY_READ | KEY_SET_VALUE, nullptr, &k, nullptr) != ERROR_SUCCESS) return;
  std::wstring path;
  DWORD type = REG_EXPAND_SZ, n = 0;
  if (RegQueryValueExW(k, L"Path", nullptr, &type, nullptr, &n) == ERROR_SUCCESS && n) {
    path.resize(n / sizeof(wchar_t));
    RegQueryValueExW(k, L"Path", nullptr, &type, (BYTE*)path.data(), &n);
    while (!path.empty() && path.back() == L'\0') path.pop_back();
  }
  // only our entry changes: everything else stays exactly as it was, down to a trailing semicolon
  const std::wstring d = dir();
  size_t at = std::wstring::npos;
  for (size_t i = 0; i <= path.size();) {
    const size_t e = std::min(path.find(L';', i), path.size());
    if (e - i == d.size() && _wcsnicmp(path.c_str() + i, d.c_str(), d.size()) == 0) { at = i; break; }
    i = e + 1;
  }
  std::wstring out = path;
  if (add && at == std::wstring::npos) {
    if (out.empty()) out = d;
    else if (out.back() == L';') out += d + L";";
    else out += L";" + d;
  } else if (!add && at != std::wstring::npos) {
    if (at + d.size() < out.size()) out.erase(at, d.size() + 1);  // (with the semicolon after it)
    else out.erase(at ? at - 1 : 0, d.size() + (at ? 1 : 0));     // (the last one: with the semicolon before it)
  }
  if (out != path) {
    RegSetValueExW(k, L"Path", 0, type, (const BYTE*)out.c_str(), (DWORD)((out.size() + 1) * sizeof(wchar_t)));
    SendMessageTimeoutW(HWND_BROADCAST, WM_SETTINGCHANGE, 0, (LPARAM)L"Environment", SMTO_ABORTIFHUNG, 1000, nullptr);
  }
  RegCloseKey(k);
}
inline void make_shortcut(const std::wstring& target) {
  IShellLinkW* l = nullptr;
  if (FAILED(CoCreateInstance(CLSID_ShellLink, nullptr, CLSCTX_INPROC_SERVER, IID_PPV_ARGS(&l))) || !l) return;
  l->SetPath(target.c_str());
  l->SetIconLocation(target.c_str(), 0);
  l->SetDescription(L"Compress files and folders, and extract .zero archives");
  IPersistFile* f = nullptr;
  if (SUCCEEDED(l->QueryInterface(IID_PPV_ARGS(&f))) && f) { f->Save(shortcut().c_str(), TRUE); f->Release(); }
  l->Release();
}
inline void wait_run(const std::wstring& exe, const std::wstring& args) {
  SHELLEXECUTEINFOW s{sizeof s};
  s.fMask = SEE_MASK_NOCLOSEPROCESS;
  s.lpFile = exe.c_str();
  s.lpParameters = args.c_str();
  s.nShow = SW_HIDE;
  if (ShellExecuteExW(&s) && s.hProcess) { WaitForSingleObject(s.hProcess, 10000); CloseHandle(s.hProcess); }
}
// false and a reason when it could not be done
inline bool install(bool explorer, bool on_path, std::wstring& why) {
  const fs::path d = dir();
  std::error_code ec;
  fs::create_directories(d, ec);
  const std::wstring gui = (d / L"zero-gui.exe").wstring(), cli = (d / L"zero.exe").wstring();
  if (!CopyFileW(shell::exe_path().c_str(), gui.c_str(), FALSE)) {
    why = GetLastError() == ERROR_SHARING_VIOLATION ? L"zero is running. Close its window and try again." : L"Could not write to " + d.wstring();
    return false;
  }
  HRSRC r = FindResourceW(nullptr, MAKEINTRESOURCEW(2), MAKEINTRESOURCEW(10) /* RT_RCDATA */);
  HGLOBAL g = r ? LoadResource(nullptr, r) : nullptr;
  const void* bytes = g ? LockResource(g) : nullptr;
  bool wrote = false;
  if (bytes) {
    FILE* f = _wfopen(cli.c_str(), L"wb");
    if (f) { wrote = fwrite(bytes, 1, SizeofResource(nullptr, r), f) == SizeofResource(nullptr, r); fclose(f); }
  }
  if (!wrote) { why = L"Could not write " + cli; return false; }
  make_shortcut(gui);
  HKEY k = nullptr;
  if (RegCreateKeyExW(HKEY_CURRENT_USER, kUninstallKey, 0, nullptr, 0, KEY_SET_VALUE, nullptr, &k, nullptr) == ERROR_SUCCESS) {
    set_str(k, L"DisplayName", L"zero");
    set_str(k, L"DisplayVersion", widen(ZERO_VERSION));
    set_str(k, L"DisplayIcon", gui + L",0");
    set_str(k, L"InstallLocation", d.wstring());
    set_str(k, L"UninstallString", L"\"" + gui + L"\" --uninstall");
    set_str(k, L"QuietUninstallString", L"\"" + gui + L"\" --uninstall --silent");
    set_dword(k, L"NoModify", 1);
    set_dword(k, L"NoRepair", 1);
    const uint64_t kb = (fs::file_size(gui, ec) + fs::file_size(cli, ec)) / 1024;
    set_dword(k, L"EstimatedSize", (DWORD)kb);
    RegCloseKey(k);
  }
  // (the entries name the program that registers them: the installed one)
  wait_run(gui, explorer ? L"--install-shell" : L"--uninstall-shell");
  edit_path(on_path);
  return true;
}
// only from the installed copy, so a mistaken call cannot remove anything else. Archives are not touched.
inline void uninstall() {
  if (!running_installed()) return;
  shell::uninstall();
  edit_path(false);
  std::error_code ec;
  fs::remove(fs::path(shortcut()), ec);
  fs::remove_all(fs::path(env(L"LOCALAPPDATA")) / L"zero" / L"kernels", ec);
  fs::remove(fs::path(env(L"LOCALAPPDATA")) / L"zero", ec);  // (only if empty)
  RegDeleteTreeW(HKEY_CURRENT_USER, kUninstallKey);
  fs::remove(fs::path(dir()) / L"zero.exe", ec);
  // a running exe cannot delete itself: a hidden command does, once this process has ended
  const std::wstring cmd = L"/c ping -n 3 127.0.0.1 >nul & del /q \"" + shell::exe_path() + L"\" & rmdir \"" + dir() + L"\"";
  ShellExecuteW(nullptr, L"open", (env(L"ComSpec").empty() ? L"cmd.exe" : env(L"ComSpec")).c_str(), cmd.c_str(), nullptr, SW_HIDE);
}
}  // namespace setup

struct Item {
  std::string path, name;
  bool dir = false;
  bool sized = false;
  uint64_t size = 0;
};
struct Scanned { std::string path; uint64_t size; };
struct DoneInfo {
  bool ok = true, cancelled = false;
  std::wstring title, line;
  std::string show;        // what "Show in folder" opens
  bool select = false;     // the path is a file to select in its folder
};

template <class T> inline void release(T*& p) { if (p) { p->Release(); p = nullptr; } }

struct Theme {
  bool dark = false, mica = false;
  D2D1_COLOR_F bg, text, sub, card, card_line, ctl, ctl_hover, ctl_down, ctl_line, accent, accent_hover, accent_down, accent_text, track, bad, good;
};
inline D2D1_COLOR_F rgba(uint32_t rgb, float a = 1.0f) {
  return D2D1::ColorF((float)((rgb >> 16) & 255) / 255, (float)((rgb >> 8) & 255) / 255, (float)(rgb & 255) / 255, a);
}
inline D2D1_COLOR_F mix(D2D1_COLOR_F a, D2D1_COLOR_F b, float t) {
  return D2D1::ColorF(a.r + (b.r - a.r) * t, a.g + (b.g - a.g) * t, a.b + (b.b - a.b) * t, 1.0f);
}

struct App {
  HWND wnd = nullptr;
  float W = 0, H = 0;  // the client area in device-independent pixels
  UINT dpi = 96;
  ID2D1Factory* d2d = nullptr;
  ID2D1HwndRenderTarget* rt = nullptr;
  ID2D1SolidColorBrush* brush = nullptr;
  ID2D1StrokeStyle* dashed = nullptr;
  IDWriteFactory* dw = nullptr;
  IDWriteTextFormat* f_big = nullptr, *f_title = nullptr, *f_h2 = nullptr, *f_body = nullptr, *f_semi = nullptr, *f_small = nullptr, *f_path = nullptr,
                     *f_wrap = nullptr, *f_icon = nullptr, *f_icon_big = nullptr;
  Theme t;
  ITaskbarList3* taskbar = nullptr;

  enum class View { Home, Compress, Archive, Working, Done, Setup, Installed } view = View::Home, back_to = View::Home;
  bool setup_menu = true, setup_path = false;
  std::wstring setup_error;
  std::vector<Item> items;
  std::string out_path;
  bool out_chosen = false;
  int level_idx = 1;   // Fast, Balanced, Smallest
  int block_idx = 2;   // 64 KiB .. 1 MiB
  int proc_idx = 0;    // GPU, CPU
  bool advanced = false;
  bool shell_on = false;
  float scroll = 0;

  std::string arc_path, dest_dir;
  ArchiveInfo arc;
  uint64_t arc_bytes = 0;
  size_t arc_files = 0;

  // the work
  std::thread worker;
  std::atomic<bool> cancel{false};
  std::atomic<uint64_t> prog_done{0}, prog_total{0};
  bool busy = false, last_failed = false;
  double t_start = 0;
  std::wstring work_title, work_sub;
  DoneInfo done;
  bool show_details = false;
  std::vector<std::string> log;
  std::unique_ptr<SafeCodec> gpu;  // kept between jobs (the kernels stay loaded); used on the worker only
  std::string gpu_error;

  // the widgets of the last paint
  struct Hot { int id; D2D1_RECT_F r; };
  std::vector<Hot> hots;
  int hover = ID_NONE, down = ID_NONE, focus = ID_NONE;
  bool tracking = false;
  D2D1_RECT_F list_clip{};

  // ---- resources
  void load_theme() {
    DWORD light = 1, n = sizeof light;
    RegGetValueW(HKEY_CURRENT_USER, L"Software\\Microsoft\\Windows\\CurrentVersion\\Themes\\Personalize", L"AppsUseLightTheme",
                 RRF_RT_REG_DWORD, nullptr, &light, &n);
    const char* force = getenv("ZERO_GUI_THEME");  // (light or dark: for checking both)
    if (force) light = strcmp(force, "dark") != 0;
    t.dark = light == 0;
    DWORD ab = 0xFF0078D4;  // the accent, as 0xAABBGGRR (Windows' default blue when it cannot be read)
    n = sizeof ab;
    if (RegGetValueW(HKEY_CURRENT_USER, L"Software\\Microsoft\\Windows\\DWM", L"AccentColor", RRF_RT_REG_DWORD, nullptr, &ab, &n) != ERROR_SUCCESS) ab = 0xFFD47800;
    const D2D1_COLOR_F accent = D2D1::ColorF((ab & 255) / 255.0f, ((ab >> 8) & 255) / 255.0f, ((ab >> 16) & 255) / 255.0f);
    const D2D1_COLOR_F white = rgba(0xFFFFFF), black = rgba(0x000000);
    if (t.dark) {
      t.bg = rgba(0x202020); t.text = rgba(0xFFFFFF); t.sub = rgba(0xFFFFFF, 0.70f);
      t.card = rgba(0xFFFFFF, 0.05f); t.card_line = rgba(0x000000, 0.20f);
      t.ctl = rgba(0xFFFFFF, 0.06f); t.ctl_hover = rgba(0xFFFFFF, 0.09f); t.ctl_down = rgba(0xFFFFFF, 0.03f); t.ctl_line = rgba(0xFFFFFF, 0.09f);
      t.accent = mix(accent, white, 0.45f); t.accent_hover = mix(accent, white, 0.35f); t.accent_down = mix(accent, white, 0.25f); t.accent_text = rgba(0x000000);
      t.track = rgba(0xFFFFFF, 0.14f); t.bad = rgba(0xFF99A4); t.good = rgba(0x6CCB5F);
    } else {
      t.bg = rgba(0xF3F3F3); t.text = rgba(0x000000, 0.90f); t.sub = rgba(0x000000, 0.62f);
      t.card = rgba(0xFFFFFF, 0.70f); t.card_line = rgba(0x000000, 0.06f);
      t.ctl = rgba(0xFFFFFF, 0.70f); t.ctl_hover = rgba(0xF9F9F9, 0.50f); t.ctl_down = rgba(0xF9F9F9, 0.30f); t.ctl_line = rgba(0x000000, 0.11f);
      t.accent = mix(accent, black, 0.10f); t.accent_hover = mix(accent, black, 0.0f); t.accent_down = mix(accent, white, 0.15f); t.accent_text = rgba(0xFFFFFF);
      t.track = rgba(0x000000, 0.11f); t.bad = rgba(0xC42B1C); t.good = rgba(0x0F7B0F);
    }
    BOOL dark = t.dark;
    DwmSetWindowAttribute(wnd, 20 /* DWMWA_USE_IMMERSIVE_DARK_MODE */, &dark, sizeof dark);
    int backdrop = 2 /* DWMSBT_MAINWINDOW: Mica */;
    t.mica = SUCCEEDED(DwmSetWindowAttribute(wnd, 38 /* DWMWA_SYSTEMBACKDROP_TYPE */, &backdrop, sizeof backdrop));
    if (t.mica) { MARGINS m{-1, -1, -1, -1}; DwmExtendFrameIntoClientArea(wnd, &m); }  // (the client area shows the backdrop)
  }
  bool has_family(const wchar_t* name) {
    IDWriteFontCollection* c = nullptr;
    dw->GetSystemFontCollection(&c);
    UINT32 i = 0; BOOL ok = FALSE;
    if (c) { c->FindFamilyName(name, &i, &ok); c->Release(); }
    return ok != FALSE;
  }
  IDWriteTextFormat* font(const wchar_t* fam, float size, DWRITE_FONT_WEIGHT w, bool wrap = false, wchar_t delim = 0) {
    IDWriteTextFormat* f = nullptr;
    dw->CreateTextFormat(fam, nullptr, w, DWRITE_FONT_STYLE_NORMAL, DWRITE_FONT_STRETCH_NORMAL, size, L"", &f);
    if (!f) return nullptr;
    f->SetParagraphAlignment(DWRITE_PARAGRAPH_ALIGNMENT_CENTER);
    if (wrap) { f->SetParagraphAlignment(DWRITE_PARAGRAPH_ALIGNMENT_NEAR); return f; }
    f->SetWordWrapping(DWRITE_WORD_WRAPPING_NO_WRAP);
    IDWriteInlineObject* sign = nullptr;
    dw->CreateEllipsisTrimmingSign(f, &sign);
    DWRITE_TRIMMING tr{DWRITE_TRIMMING_GRANULARITY_CHARACTER, (UINT32)delim, delim ? 2u : 0u};
    if (sign) { f->SetTrimming(&tr, sign); sign->Release(); }
    return f;
  }
  void create_resources() {
    D2D1CreateFactory(D2D1_FACTORY_TYPE_SINGLE_THREADED, &d2d);
    DWriteCreateFactory(DWRITE_FACTORY_TYPE_SHARED, __uuidof(IDWriteFactory), reinterpret_cast<IUnknown**>(&dw));
    const bool var = has_family(L"Segoe UI Variable Text");
    const wchar_t* text = var ? L"Segoe UI Variable Text" : L"Segoe UI";
    const wchar_t* disp = var ? L"Segoe UI Variable Display" : L"Segoe UI";
    const wchar_t* icons = has_family(L"Segoe Fluent Icons") ? L"Segoe Fluent Icons" : L"Segoe MDL2 Assets";
    f_big = font(disp, 46, DWRITE_FONT_WEIGHT_SEMI_BOLD);
    f_title = font(disp, 28, DWRITE_FONT_WEIGHT_SEMI_BOLD);
    f_h2 = font(disp, 20, DWRITE_FONT_WEIGHT_SEMI_BOLD);
    f_body = font(text, 14, DWRITE_FONT_WEIGHT_NORMAL);
    f_semi = font(text, 14, DWRITE_FONT_WEIGHT_SEMI_BOLD);
    f_small = font(text, 12, DWRITE_FONT_WEIGHT_NORMAL);
    f_path = font(text, 14, DWRITE_FONT_WEIGHT_NORMAL, false, L'\\');
    f_wrap = font(text, 12, DWRITE_FONT_WEIGHT_NORMAL, true);
    f_icon = font(icons, 16, DWRITE_FONT_WEIGHT_NORMAL);
    f_icon_big = font(icons, 44, DWRITE_FONT_WEIGHT_NORMAL);
    const float dash[] = {4.0f, 3.0f};
    d2d->CreateStrokeStyle(D2D1::StrokeStyleProperties(D2D1_CAP_STYLE_FLAT, D2D1_CAP_STYLE_FLAT, D2D1_CAP_STYLE_FLAT, D2D1_LINE_JOIN_ROUND, 10.0f, D2D1_DASH_STYLE_CUSTOM), dash, 2, &dashed);
  }
  void make_target() {
    release(brush);
    release(rt);
    RECT rc;
    GetClientRect(wnd, &rc);
    // (software: the GPU is busy compressing, and the window should not wait for it)
    auto props = D2D1::RenderTargetProperties(D2D1_RENDER_TARGET_TYPE_SOFTWARE, D2D1::PixelFormat(DXGI_FORMAT_B8G8R8A8_UNORM, D2D1_ALPHA_MODE_PREMULTIPLIED), (float)dpi, (float)dpi);
    d2d->CreateHwndRenderTarget(props, D2D1::HwndRenderTargetProperties(wnd, D2D1::SizeU(rc.right, rc.bottom)), &rt);
    if (rt) { rt->CreateSolidColorBrush(t.text, &brush); rt->SetTextAntialiasMode(D2D1_TEXT_ANTIALIAS_MODE_GRAYSCALE); }
  }
  void redraw() { if (wnd) InvalidateRect(wnd, nullptr, FALSE); }
  void go(View v) { view = v; scroll = 0; hover = down = focus = ID_NONE; redraw(); }

  // ---- drawing
  void text(const std::wstring& s, D2D1_RECT_F r, IDWriteTextFormat* f, D2D1_COLOR_F c, DWRITE_TEXT_ALIGNMENT a = DWRITE_TEXT_ALIGNMENT_LEADING) {
    if (!f) return;
    f->SetTextAlignment(a);
    brush->SetColor(c);
    rt->DrawText(s.c_str(), (UINT32)s.size(), f, r, brush, D2D1_DRAW_TEXT_OPTIONS_CLIP);
  }
  void rrect(D2D1_RECT_F r, float rad, D2D1_COLOR_F fill, const D2D1_COLOR_F* line = nullptr) {
    brush->SetColor(fill);
    rt->FillRoundedRectangle(D2D1::RoundedRect(r, rad, rad), brush);
    if (line) {
      brush->SetColor(*line);
      const D2D1_RECT_F q = D2D1::RectF(r.left + 0.5f, r.top + 0.5f, r.right - 0.5f, r.bottom - 0.5f);
      rt->DrawRoundedRectangle(D2D1::RoundedRect(q, rad, rad), brush, 1.0f);
    }
  }
  void card(D2D1_RECT_F r) { rrect(r, 8, t.card, &t.card_line); }
  void hot(int id, D2D1_RECT_F r) { hots.push_back({id, r}); }
  void focus_ring(int id, D2D1_RECT_F r) {
    if (focus != id) return;
    brush->SetColor(t.text);
    rt->DrawRoundedRectangle(D2D1::RoundedRect(D2D1::RectF(r.left - 2, r.top - 2, r.right + 2, r.bottom + 2), 6, 6), brush, 2.0f);
  }
  // a button: primary (the accent) or standard; an icon glyph, a label, or both
  void button(int id, D2D1_RECT_F r, const std::wstring& label, bool primary = false, wchar_t icon = 0) {
    hot(id, r);
    const bool h = hover == id, d = down == id && h;
    if (primary) rrect(r, 4, d ? t.accent_down : h ? t.accent_hover : t.accent);
    else rrect(r, 4, d ? t.ctl_down : h ? t.ctl_hover : t.ctl, &t.ctl_line);
    const D2D1_COLOR_F c = primary ? t.accent_text : t.text;
    if (icon && label.empty()) {
      text(std::wstring(1, icon), r, f_icon, c, DWRITE_TEXT_ALIGNMENT_CENTER);
    } else if (icon) {
      text(std::wstring(1, icon) , D2D1::RectF(r.left + 12, r.top, r.left + 32, r.bottom), f_icon, c);
      text(label, D2D1::RectF(r.left + 36, r.top, r.right - 8, r.bottom), f_body, c);
    } else {
      text(label, r, f_body, c, DWRITE_TEXT_ALIGNMENT_CENTER);
    }
    focus_ring(id, r);
  }
  // a button without a box, for the less common step
  void link(int id, D2D1_RECT_F r, const std::wstring& label, DWRITE_TEXT_ALIGNMENT a = DWRITE_TEXT_ALIGNMENT_CENTER, wchar_t icon = 0) {
    hot(id, r);
    if (hover == id) rrect(r, 4, t.ctl_hover);
    D2D1_RECT_F q = r;
    if (icon) { text(std::wstring(1, icon), D2D1::RectF(r.left + 8, r.top, r.left + 28, r.bottom), f_icon, t.accent); q.left += 30; }
    else q.left += 8;
    text(label, q, f_body, t.accent, a);
    focus_ring(id, r);
  }
  // one choice out of a few
  void segments(int base, D2D1_RECT_F r, const std::vector<std::wstring>& labels, int sel) {
    rrect(r, 4, t.track);
    const float w = (r.right - r.left) / labels.size();
    for (size_t i = 0; i < labels.size(); ++i) {
      const int id = base + (int)i;
      const D2D1_RECT_F s = D2D1::RectF(r.left + i * w + 2, r.top + 2, r.left + (i + 1) * w - 2, r.bottom - 2);
      hot(id, s);
      if ((int)i == sel) rrect(s, 3, t.accent);
      else if (hover == id) rrect(s, 3, t.ctl_hover);
      text(labels[i], s, f_body, (int)i == sel ? t.accent_text : t.text, DWRITE_TEXT_ALIGNMENT_CENTER);
      focus_ring(id, s);
    }
  }
  void header(const std::wstring& title, const std::wstring& sub) {
    button(ID_BACK, D2D1::RectF(24, 22, 56, 54), L"", false, 0xE72B);
    text(title, D2D1::RectF(68, 16, W - 200, 44), f_h2, t.text);
    text(sub, D2D1::RectF(68, 42, W - 200, 62), f_small, t.sub);
  }
  // a titled card with a path and a way to change it
  void path_card(float y, const std::wstring& label, const std::string& path) {
    const D2D1_RECT_F r = D2D1::RectF(24, y, W - 24, y + 60);
    card(r);
    text(label, D2D1::RectF(40, y + 8, W - 150, y + 28), f_small, t.sub);
    text(widen(path), D2D1::RectF(40, y + 26, W - 150, y + 50), f_path, t.text);
    button(ID_BROWSE, D2D1::RectF(W - 128, y + 14, W - 40, y + 46), L"Change");
  }
  void scrollbar(D2D1_RECT_F r, float content) {
    const float h = r.bottom - r.top;
    if (content <= h) return;
    const float bar = std::max(24.0f, h * h / content), pos = (h - bar) * scroll / (content - h);
    rrect(D2D1::RectF(r.right - 6, r.top + pos, r.right - 3, r.top + pos + bar), 1.5f, t.track);
  }
  float clamp_scroll(float content, float view_h) {
    scroll = std::min(scroll, std::max(0.0f, content - view_h));
    scroll = std::max(scroll, 0.0f);
    return scroll;
  }

  // ---- the screens
  void paint_home() {
    text(L"zero", D2D1::RectF(24, 18, W - 24, 58), f_title, t.text);
    text(L"Compress files and folders, and extract .zero archives", D2D1::RectF(24, 58, W - 24, 80), f_body, t.sub);
    const D2D1_RECT_F r = D2D1::RectF(24, 96, W - 24, H - 24);
    brush->SetColor(t.card);
    rt->FillRoundedRectangle(D2D1::RoundedRect(r, 8, 8), brush);
    brush->SetColor(t.ctl_line);
    const D2D1_RECT_F q = D2D1::RectF(r.left + 0.5f, r.top + 0.5f, r.right - 0.5f, r.bottom - 0.5f);
    rt->DrawRoundedRectangle(D2D1::RoundedRect(q, 8, 8), brush, 1.0f, dashed);
    const float cx = W / 2, cy = (r.top + r.bottom) / 2 - 16;
    text(std::wstring(1, (wchar_t)0xE898), D2D1::RectF(0, cy - 104, W, cy - 48), f_icon_big, t.accent, DWRITE_TEXT_ALIGNMENT_CENTER);
    text(L"Drop files or folders here", D2D1::RectF(24, cy - 44, W - 24, cy - 12), f_h2, t.text, DWRITE_TEXT_ALIGNMENT_CENTER);
    text(L"Drop a .zero archive to extract it", D2D1::RectF(24, cy - 12, W - 24, cy + 12), f_body, t.sub, DWRITE_TEXT_ALIGNMENT_CENTER);
    button(ID_ADD_FILES, D2D1::RectF(cx - 164, cy + 28, cx - 6, cy + 64), L"Choose files", true, 0xE8E5);
    button(ID_ADD_FOLDER, D2D1::RectF(cx + 6, cy + 28, cx + 164, cy + 64), L"Choose a folder", false, 0xE8B7);
    link(ID_OPEN, D2D1::RectF(cx - 130, cy + 76, cx + 130, cy + 108), L"Open an archive to extract");
    const float by = r.bottom - 66;
    if (shell_on) text(L"In Explorer: right-click, Show more options", D2D1::RectF(24, by + 32, W - 24, by + 54), f_small, t.sub, DWRITE_TEXT_ALIGNMENT_CENTER);
    link(ID_SHELL, D2D1::RectF(cx - 150, by, cx + 150, by + 30), shell_on ? L"Remove zero from the Explorer menu" : L"Add zero to the Explorer menu");
  }
  void checkbox(int id, D2D1_RECT_F r, const std::wstring& label, const std::wstring& note, bool on) {
    hot(id, r);
    if (hover == id) rrect(r, 4, t.ctl_hover);
    const D2D1_RECT_F b = D2D1::RectF(r.left + 12, r.top + 14, r.left + 32, r.top + 34);
    if (on) { rrect(b, 4, t.accent); text(std::wstring(1, (wchar_t)0xE73E), b, f_icon, t.accent_text, DWRITE_TEXT_ALIGNMENT_CENTER); }
    else rrect(b, 4, t.ctl, &t.ctl_line);
    text(label, D2D1::RectF(r.left + 44, r.top + 6, r.right - 8, r.top + 28), f_body, t.text);
    text(note, D2D1::RectF(r.left + 44, r.top + 28, r.right - 8, r.top + 46), f_small, t.sub);
    focus_ring(id, r);
  }
  void paint_setup() {
    const bool update = setup::installed();
    text(update ? L"Update zero" : L"Install zero", D2D1::RectF(24, 18, W - 24, 58), f_title, t.text);
    text(L"Compress files and folders, and extract .zero archives", D2D1::RectF(24, 58, W - 24, 80), f_body, t.sub);
    card(D2D1::RectF(24, 100, W - 24, 160));
    text(L"Installs to", D2D1::RectF(40, 108, W - 40, 128), f_small, t.sub);
    text(setup::dir(), D2D1::RectF(40, 126, W - 40, 150), f_path, t.text);
    const D2D1_RECT_F c = D2D1::RectF(24, 172, W - 24, 296);
    card(c);
    checkbox(ID_SETUP_MENU, D2D1::RectF(c.left + 4, c.top + 6, c.right - 4, c.top + 58), L"Add to the Explorer menu", L"Right-click to compress or extract, double-click to open a .zero archive", setup_menu);
    checkbox(ID_SETUP_PATH, D2D1::RectF(c.left + 4, c.top + 62, c.right - 4, c.top + 114), L"Use zero in a terminal", L"Adds the folder to your PATH, for the zero command", setup_path);
    text(L"No administrator rights needed. Remove it any time in Settings, Apps.", D2D1::RectF(24, 306, W - 24, 328), f_small, t.sub);
    if (!setup_error.empty()) text(setup_error, D2D1::RectF(24, 332, W - 24, 356), f_body, t.bad);
    button(ID_SETUP_GO, D2D1::RectF(W - 184, H - 60, W - 24, H - 24), update ? L"Update" : L"Install", true);
  }
  void paint_installed() {
    const float top = H / 2 - 130;
    text(std::wstring(1, (wchar_t)0xE73E), D2D1::RectF(40, top, 100, top + 56), f_icon_big, t.good);
    text(L"zero is installed", D2D1::RectF(40, top + 60, W - 40, top + 96), f_title, t.text);
    text(L"Find it in the Start menu. To remove it, go to Settings, Apps.", D2D1::RectF(40, top + 98, W - 40, top + 130), f_wrap, t.sub);
    button(ID_RUN, D2D1::RectF(40, top + 140, 196, top + 176), L"Open zero", true);
  }
  void paint_rows(D2D1_RECT_F r, size_t n, const std::function<void(size_t, D2D1_RECT_F)>& row) {
    const float rh = 40, content = n * rh;
    card(r);
    clamp_scroll(content, r.bottom - r.top - 2);
    list_clip = r;
    rt->PushAxisAlignedClip(D2D1::RectF(r.left + 1, r.top + 1, r.right - 1, r.bottom - 1), D2D1_ANTIALIAS_MODE_ALIASED);
    const size_t first = (size_t)(scroll / rh);
    const size_t last = std::min(n, first + (size_t)((r.bottom - r.top) / rh) + 2);
    for (size_t i = first; i < last; ++i) {
      const float y = r.top + 1 + i * rh - scroll;
      if (i) { brush->SetColor(t.card_line); rt->DrawLine(D2D1::Point2F(r.left + 12, y), D2D1::Point2F(r.right - 12, y), brush, 1.0f); }
      row(i, D2D1::RectF(r.left, y, r.right, y + rh));
    }
    rt->PopAxisAlignedClip();
    scrollbar(r, content);
  }
  void paint_compress() {
    uint64_t total = 0;
    for (const Item& it : items) total += it.size;
    bool sizing = false;
    for (const Item& it : items) if (!it.sized) sizing = true;
    std::wstring sub = std::to_wstring(items.size()) + (items.size() == 1 ? L" item" : L" items") + L", " + (sizing ? L"counting..." : widen(human(total)));
    header(L"New archive", sub);
    button(ID_ADD_FILES, D2D1::RectF(W - 148, 24, W - 24, 56), L"Add more", false, 0xE710);
    const float bottom = H - 24 - 36;                                // the main button
    const float adv_h = advanced ? 96.0f : 0.0f;
    const float level_y = bottom - 12 - adv_h - 34 - 86;             // the speed choice
    const float path_y = level_y - 12 - 60;
    paint_rows(D2D1::RectF(24, 72, W - 24, path_y - 12), items.size(), [&](size_t i, D2D1_RECT_F r) {
      const Item& it = items[i];
      text(std::wstring(1, it.dir ? (wchar_t)0xE8B7 : (wchar_t)0xE8A5), D2D1::RectF(r.left + 14, r.top, r.left + 38, r.bottom), f_icon, t.sub);
      text(widen(it.name), D2D1::RectF(r.left + 42, r.top, r.right - 160, r.bottom), f_body, t.text);
      text(it.sized ? widen(human(it.size)) : L"counting...", D2D1::RectF(r.right - 160, r.top, r.right - 52, r.bottom), f_small, t.sub, DWRITE_TEXT_ALIGNMENT_TRAILING);
      const D2D1_RECT_F x = D2D1::RectF(r.right - 44, r.top + 6, r.right - 16, r.bottom - 6);
      if (r.top >= list_clip.top - 1 && r.bottom <= list_clip.bottom + 1) {
        const int id = ID_ROW + (int)i;
        hot(id, x);
        if (hover == id) rrect(x, 4, t.ctl_hover);
        text(std::wstring(1, (wchar_t)0xE711), x, f_icon, t.sub, DWRITE_TEXT_ALIGNMENT_CENTER);
        focus_ring(id, x);
      }
    });
    path_card(path_y, L"Save the archive as", out_path);
    const float y = level_y;
    text(L"Compression", D2D1::RectF(24, y, 200, y + 22), f_semi, t.text);
    segments(ID_LEVEL, D2D1::RectF(24, y + 26, W - 24, y + 60), {L"Fast", L"Balanced", L"Smallest"}, level_idx);
    static const wchar_t* notes[] = {L"Finishes soonest. The archive is a little larger.", L"A good mix of speed and size. Best for most files.",
                                     L"Takes about twice as long for a smaller archive."};
    text(notes[level_idx], D2D1::RectF(24, y + 62, W - 24, y + 84), f_small, t.sub);
    paint_advanced(y + 86, true);
    button(ID_COMPRESS, D2D1::RectF(W - 184, bottom, W - 24, bottom + 36), L"Compress", true);
  }
  void paint_advanced(float y, bool with_block) {
    link(ID_ADVANCED, D2D1::RectF(16, y, 150, y + 30), L"Advanced", DWRITE_TEXT_ALIGNMENT_LEADING, advanced ? (wchar_t)0xE70E : (wchar_t)0xE70D);
    if (!advanced) return;
    y += 36;
    const float mid = with_block ? W * 0.62f : W - 24;
    if (with_block) {
      text(L"Block size (larger: a little smaller, smaller: faster to extract)", D2D1::RectF(24, y, mid - 12, y + 20), f_small, t.sub);
      segments(ID_BLOCK, D2D1::RectF(24, y + 22, mid - 12, y + 54), {L"64K", L"128K", L"256K", L"512K", L"1M"}, block_idx);
    }
    const float px0 = with_block ? mid : 24;
    text(L"Run on", D2D1::RectF(px0, y, W - 24, y + 20), f_small, t.sub);
    segments(ID_PROC, D2D1::RectF(px0, y + 22, with_block ? W - 24 : 264, y + 54), {L"GPU", L"CPU"}, proc_idx);
  }
  void paint_archive() {
    wchar_t pct[32] = L"";
    if (arc.total_raw) swprintf(pct, 32, L"%.1f", 100.0 * arc_bytes / arc.total_raw);
    std::wstring sub = std::to_wstring(arc_files) + (arc_files == 1 ? L" file, " : L" files, ") + widen(human(arc.total_raw)) + L" compressed to " + widen(human(arc_bytes)) + L" (" + pct + L" %)";
    header(widen(base_name(arc_path)), sub);
    const float bottom = H - 24 - 36;
    const float adv_h = advanced ? 96.0f : 0.0f;
    const float adv_y = bottom - 12 - adv_h - 34;
    const float path_y = adv_y - 12 - 60;
    std::vector<const Entry*> files;
    files.reserve(arc.entries.size());
    for (const Entry& e : arc.entries) if (!e.dir) files.push_back(&e);
    paint_rows(D2D1::RectF(24, 72, W - 24, path_y - 12), files.size(), [&](size_t i, D2D1_RECT_F r) {
      text(std::wstring(1, (wchar_t)0xE8A5), D2D1::RectF(r.left + 14, r.top, r.left + 38, r.bottom), f_icon, t.sub);
      text(widen(files[i]->name), D2D1::RectF(r.left + 42, r.top, r.right - 130, r.bottom), f_body, t.text);
      text(widen(human(files[i]->size)), D2D1::RectF(r.right - 130, r.top, r.right - 20, r.bottom), f_small, t.sub, DWRITE_TEXT_ALIGNMENT_TRAILING);
    });
    path_card(path_y, L"Extract to", dest_dir);
    paint_advanced(adv_y, false);
    button(ID_EXTRACT, D2D1::RectF(W - 184, bottom, W - 24, bottom + 36), L"Extract", true);
    button(ID_TEST, D2D1::RectF(W - 332, bottom, W - 196, bottom + 36), L"Check for errors");
  }
  static std::wstring time_left(double s) {
    wchar_t b[64];
    if (s < 60) swprintf(b, 64, L"about %d s left", (int)s + 1);
    else if (s < 3600) swprintf(b, 64, L"about %d min left", (int)(s / 60) + 1);
    else swprintf(b, 64, L"about %.1f h left", s / 3600);
    return b;
  }
  void paint_working() {
    const uint64_t d = prog_done.load(), n = prog_total.load();
    const double frac = n ? (double)d / n : 0;
    const float cy = H / 2 - 40;
    text(work_title, D2D1::RectF(40, cy - 96, W - 40, cy - 64), f_h2, t.text);
    text(work_sub, D2D1::RectF(40, cy - 64, W - 40, cy - 40), f_path, t.sub);
    wchar_t pct[16];
    swprintf(pct, 16, L"%d %%", (int)(frac * 100));
    text(pct, D2D1::RectF(40, cy - 36, W - 40, cy + 24), f_big, t.text);
    const D2D1_RECT_F bar = D2D1::RectF(40, cy + 34, W - 40, cy + 40);
    rrect(bar, 3, t.track);
    if (frac > 0) rrect(D2D1::RectF(bar.left, bar.top, bar.left + std::max(6.0f, (float)((bar.right - bar.left) * frac)), bar.bottom), 3, t.accent);
    std::wstring line = widen(human(d)) + L" of " + widen(human(n));
    const double el = zero_now() - t_start;
    if (el > 1.0 && d > 0) {
      line += L", " + widen(rate(d, el));
      if (d < n) line += L", " + time_left((double)(n - d) * el / d);
    }
    text(line, D2D1::RectF(40, cy + 48, W - 40, cy + 72), f_body, t.sub);
    button(ID_CANCEL, D2D1::RectF(W - 160, H - 60, W - 24, H - 24), cancel.load() ? L"Cancelling..." : L"Cancel");
  }
  void paint_done() {
    const float top = show_details ? 56.0f : H / 2 - 130;
    text(std::wstring(1, done.ok ? (wchar_t)0xE73E : (wchar_t)0xE783), D2D1::RectF(40, top, 100, top + 56), f_icon_big, done.ok ? t.good : t.bad);
    text(done.title, D2D1::RectF(40, top + 60, W - 40, top + 96), f_title, t.text);
    text(done.line, D2D1::RectF(40, top + 98, W - 40, top + 150), f_wrap, t.sub);
    float y = top + 152;
    if (done.ok && !done.show.empty()) {
      button(ID_SHOW, D2D1::RectF(40, y, 216, y + 36), L"Show in folder", true, 0xE8B7);
      button(ID_AGAIN, D2D1::RectF(228, y, 360, y + 36), L"Start over");
    } else if (done.ok) {
      button(ID_BACK, D2D1::RectF(40, y, 172, y + 36), L"Back", true);
    } else {
      button(ID_BACK, D2D1::RectF(40, y, 172, y + 36), L"Back", true);
      button(ID_AGAIN, D2D1::RectF(184, y, 316, y + 36), L"Start over");
    }
    if (log.empty()) return;
    y += 46;
    link(ID_DETAILS, D2D1::RectF(32, y, 160, y + 30), L"Details", DWRITE_TEXT_ALIGNMENT_LEADING, show_details ? (wchar_t)0xE70E : (wchar_t)0xE70D);
    if (!show_details) return;
    y += 36;
    const D2D1_RECT_F r = D2D1::RectF(24, y, W - 24, H - 24);
    card(r);
    std::wstring all;
    for (const std::string& s : log) all += widen(s) + L"\n";
    text(all, D2D1::RectF(r.left + 12, r.top + 8, r.right - 12, r.bottom - 8), f_wrap, t.sub);
  }
  void paint() {
    if (!rt) make_target();
    if (!rt) return;
    rt->BeginDraw();
    rt->Clear(t.mica ? D2D1::ColorF(0, 0, 0, 0) : t.bg);
    hots.clear();
    switch (view) {
      case View::Home: paint_home(); break;
      case View::Compress: paint_compress(); break;
      case View::Archive: paint_archive(); break;
      case View::Working: paint_working(); break;
      case View::Done: paint_done(); break;
      case View::Setup: paint_setup(); break;
      case View::Installed: paint_installed(); break;
    }
    if (rt->EndDraw() == D2DERR_RECREATE_TARGET) make_target();
  }

  // ---- the file dialogs (IFileDialog): files (several), a folder, or a file to save
  std::vector<std::string> pick(bool folder, bool multi, bool save, const wchar_t* def_name = nullptr, bool only_zero = false) {
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
    if (save || only_zero) {
      COMDLG_FILTERSPEC spec[] = {{L"zero archive (*.zero)", L"*.zero"}, {L"All files", L"*.*"}};
      dlg->SetFileTypes(2, spec);
    }
    if (save) {
      dlg->SetDefaultExtension(L"zero");
      if (def_name) dlg->SetFileName(def_name);
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

  // ---- what the person gave
  void open_paths(const std::vector<std::string>& paths) {
    if (busy || paths.empty()) return;
    if (paths.size() == 1 && is_archive(paths[0]) && items.empty()) { open_archive(paths[0]); return; }
    std::error_code ec;
    for (const std::string& p : paths) {
      bool dup = false;
      for (const Item& it : items) if (it.path == p) dup = true;
      if (dup) continue;
      Item it;
      it.path = p;
      it.name = base_name(p);
      const fs::path q = zero_path(p);
      it.dir = fs::is_directory(q, ec);
      if (it.dir) {
        // a folder's size may take a while: counted on the side
        HWND w = wnd;
        std::thread([w, p] {
          std::error_code e;
          uint64_t total = 0;
          for (fs::recursive_directory_iterator d(zero_path(p), fs::directory_options::skip_permission_denied, e), end; !e && d != end; d.increment(e))
            if (d->is_regular_file(e)) total += d->file_size(e);
          PostMessageW(w, WM_SCANNED, 0, (LPARAM) new Scanned{p, total});
        }).detach();
      } else {
        it.size = fs::file_size(q, ec);
        it.sized = true;
      }
      items.push_back(std::move(it));
    }
    if (items.empty()) return;
    suggest_output();
    go(View::Compress);
  }
  void suggest_output() {
    if (out_chosen || items.empty()) return;
    const fs::path first = zero_path(items[0].path);
    fs::path p = first;
    while (!p.empty() && !p.has_filename()) p = p.parent_path();
    const fs::path dir = p.parent_path();
    // one input: its name, without its extension when it is a file; several: the name of the folder they are in
    std::string stem = items[0].dir ? zero_utf8(p.filename()) : zero_utf8(p.stem());
    if (items.size() > 1) { const std::string d = zero_utf8(dir.filename()); if (!d.empty()) stem = d; else stem = "archive"; }
    out_path = unique_path(dir, stem, ".zero");
  }
  void open_archive(const std::string& path) {
    try {
      FILE* f = open_file(zero_path(path), "rb");
      if (!f) throw zb::Error("cannot open " + path);
      struct Closer { FILE* f; ~Closer() { fclose(f); } } c{f};
      arc = read_info(f);
    } catch (const std::exception& e) {
      done = DoneInfo{};
      done.ok = false;
      done.title = L"This archive cannot be opened";
      done.line = widen(e.what());
      back_to = View::Home;
      last_failed = true;
      go(View::Done);
      return;
    }
    arc_path = path;
    std::error_code ec;
    arc_bytes = fs::file_size(zero_path(path), ec);
    arc_files = 0;
    for (const Entry& e : arc.entries) if (!e.dir) ++arc_files;
    const fs::path p = zero_path(path);
    dest_dir = unique_path(p.parent_path(), zero_utf8(p.stem()), "");
    go(View::Archive);
  }
  void start_over() {
    items.clear();
    out_path.clear();
    out_chosen = false;
    arc_path.clear();
    arc = ArchiveInfo{};
    log.clear();
    show_details = false;
    last_failed = false;
    go(View::Home);
  }

  // ---- the work
  Codec& codec_for(Report& rep, std::unique_ptr<CpuCodec>& cpu_holder, int level) {
    if (proc_idx == 0) {
      if (!gpu && gpu_error.empty()) {
        try {
          auto g = std::make_unique<GpuCodec>();
          g->open(rep);
          gpu = std::make_unique<SafeCodec>(std::move(g));
        } catch (const std::exception& e) {
          gpu.reset();
          gpu_error = e.what();
          rep.say(std::string("GPU unavailable: ") + gpu_error + "; using the CPU");
        }
      } else if (!gpu) {
        rep.say(std::string("GPU unavailable: ") + gpu_error + "; using the CPU");
      }
      if (gpu) {
        gpu->gpu->level = level;
        gpu->set_report(&rep);
        gpu->cpu.depth = level <= 1 ? 8 : level >= 9 ? 256 : 32;
        return *gpu;
      }
    }
    cpu_holder = std::make_unique<CpuCodec>();
    cpu_holder->depth = level <= 1 ? 8 : level >= 9 ? 256 : 32;
    return *cpu_holder;
  }
  void run(View from, const std::wstring& title, const std::wstring& sub, std::function<DoneInfo(Report&)> job) {
    if (busy) return;
    if (worker.joinable()) worker.join();
    cancel = false;
    busy = true;
    last_failed = false;
    prog_done = 0;
    prog_total = 0;
    log.clear();
    show_details = false;
    t_start = zero_now();
    work_title = title;
    work_sub = sub;
    back_to = from;
    go(View::Working);
    worker = std::thread([this, job] {
      Report rep;
      rep.cancel = &cancel;
      rep.log = [this](const std::string& s) { PostMessageW(wnd, WM_LOG, 0, (LPARAM) new std::string(s)); };
      double last = 0;
      rep.progress = [this, &last](uint64_t d, uint64_t n) {
        prog_done = d;
        prog_total = n;
        const double now = zero_now();
        if (d < n && now - last < 0.1) return;
        last = now;
        PostMessageW(wnd, WM_PROGRESS, 0, 0);
      };
      auto* info = new DoneInfo();
      try {
        *info = job(rep);
      } catch (const Cancelled&) {
        info->cancelled = true;
      } catch (const std::exception& e) {
        info->ok = false;
        info->title = L"That did not work";
        info->line = widen(e.what());
        rep.say(e.what());
        // a lost device stays lost for this process (measured: opening it again fails to build kernels), so the jobs
        // after it say so and use the CPU
        gpu.reset();
        if (strstr(e.what(), "device lost")) {
          gpu_error = "the GPU was reset; reopen the window to use it again";
          info->line += L"\nThe next job will use the CPU. Reopen the window to use the GPU again.";
        }
      }
      PostMessageW(wnd, WM_DONE, 0, (LPARAM)info);
    });
  }
  static std::wstring sized_line(uint64_t raw, uint64_t comp, double secs) {
    wchar_t b[256];
    swprintf(b, 256, L"%s compressed to %s (%.1f %%) in %.1f s, %s", widen(human(raw)).c_str(), widen(human(comp)).c_str(), raw ? 100.0 * comp / raw : 0.0, secs, widen(rate(raw, secs)).c_str());
    return b;
  }
  void do_compress() {
    if (busy || items.empty() || out_path.empty()) return;
    for (const Item& it : items)
      if (it.path == out_path) { MessageBoxW(wnd, L"The archive cannot be one of its own inputs.", L"zero", MB_ICONWARNING); return; }
    std::error_code ec;
    if (!out_chosen && fs::exists(zero_path(out_path), ec))
      if (MessageBoxW(wnd, (widen(base_name(out_path)) + L" already exists. Replace it?").c_str(), L"zero", MB_YESNO | MB_ICONQUESTION) != IDYES) return;
    std::vector<std::string> inputs;
    for (const Item& it : items) inputs.push_back(it.path);
    const std::string outp = out_path;
    const uint32_t bl = 16 + (uint32_t)block_idx;
    const int level = level_idx == 0 ? 1 : level_idx == 2 ? 9 : 5;
    run(View::Compress, L"Compressing", widen(outp), [this, inputs, outp, bl, level](Report& rep) {
      std::unique_ptr<CpuCodec> holder;
      Codec& c = codec_for(rep, holder, level);
      rep.say("codec: " + c.name());
      const CompressStats st = compress_archive(inputs, outp, c, bl, rep);
      DoneInfo d;
      d.title = L"Archive created";
      d.line = sized_line(st.raw, st.compressed, st.seconds);
      d.show = outp;
      d.select = true;
      return d;
    });
  }
  void do_extract() {
    if (busy || arc_path.empty()) return;
    std::error_code ec;
    const fs::path dp = zero_path(dest_dir);
    if (fs::exists(dp, ec) && !fs::is_empty(dp, ec))
      if (MessageBoxW(wnd, L"This folder is not empty. Files with the same names will be replaced. Continue?", L"zero", MB_YESNO | MB_ICONQUESTION) != IDYES) return;
    const std::string a = arc_path, dir = dest_dir;
    run(View::Archive, L"Extracting", widen(dir), [this, a, dir](Report& rep) {
      std::unique_ptr<CpuCodec> holder;
      Codec& c = codec_for(rep, holder, 5);
      rep.say("codec: " + c.name());
      const ExtractStats st = extract_archive(a, dir, c, rep);
      DoneInfo d;
      d.title = L"Extracted";
      wchar_t b[256];
      swprintf(b, 256, L"%zu files, %s in %.1f s. Every checksum matched.", st.files, widen(human(st.raw)).c_str(), st.seconds);
      d.line = b;
      d.show = dir;
      return d;
    });
  }
  void do_test() {
    if (busy || arc_path.empty()) return;
    const std::string a = arc_path;
    run(View::Archive, L"Checking for errors", widen(a), [this, a](Report& rep) {
      std::unique_ptr<CpuCodec> holder;
      Codec& c = codec_for(rep, holder, 5);
      rep.say("codec: " + c.name());
      const ExtractStats st = test_archive(a, c, rep);
      DoneInfo d;
      d.title = L"No errors found";
      wchar_t b[256];
      swprintf(b, 256, L"%zu files, %s in %.1f s. Every checksum matched.", st.files, widen(human(st.raw)).c_str(), st.seconds);
      d.line = b;
      return d;
    });
  }
  void show_result() {
    if (done.select) ShellExecuteW(nullptr, L"open", L"explorer.exe", (L"/select,\"" + widen(done.show) + L"\"").c_str(), nullptr, SW_SHOWNORMAL);
    else ShellExecuteW(nullptr, L"open", widen(done.show).c_str(), nullptr, nullptr, SW_SHOWNORMAL);
  }

  // ---- input
  void command(int id) {
    if (id >= ID_ROW) {
      const size_t i = (size_t)(id - ID_ROW);
      if (i < items.size()) items.erase(items.begin() + i);
      if (items.empty()) start_over();
      else { suggest_output(); redraw(); }
      return;
    }
    if (id >= ID_PROC && id < ID_PROC + 2) { proc_idx = id - ID_PROC; redraw(); return; }
    if (id >= ID_BLOCK && id < ID_BLOCK + 5) { block_idx = id - ID_BLOCK; redraw(); return; }
    if (id >= ID_LEVEL && id < ID_LEVEL + 3) { level_idx = id - ID_LEVEL; redraw(); return; }
    switch (id) {
      case ID_ADD_FILES: open_paths(pick(false, true, false)); break;
      case ID_ADD_FOLDER: open_paths(pick(true, true, false)); break;
      case ID_OPEN: {
        const std::vector<std::string> r = pick(false, false, false, nullptr, true);
        if (!r.empty()) open_archive(r[0]);
        break;
      }
      case ID_BROWSE:
        if (view == View::Archive) {
          const std::vector<std::string> r = pick(true, false, false);
          if (!r.empty()) { dest_dir = r[0]; redraw(); }
        } else {
          const std::wstring def = widen(base_name(out_path));
          const std::vector<std::string> r = pick(false, false, true, def.c_str());
          if (!r.empty()) { out_path = r[0]; out_chosen = true; redraw(); }
        }
        break;
      case ID_ADVANCED: advanced = !advanced; redraw(); break;
      case ID_COMPRESS: do_compress(); break;
      case ID_EXTRACT: do_extract(); break;
      case ID_TEST: do_test(); break;
      case ID_CANCEL: cancel = true; redraw(); break;
      case ID_BACK:
        if (view == View::Done) { go(back_to); break; }
        start_over();
        break;
      case ID_SHOW: show_result(); break;
      case ID_DETAILS: show_details = !show_details; redraw(); break;
      case ID_AGAIN: start_over(); break;
      case ID_SETUP_MENU: setup_menu = !setup_menu; redraw(); break;
      case ID_SETUP_PATH: setup_path = !setup_path; redraw(); break;
      case ID_SETUP_GO:
        setup_error.clear();
        if (setup::install(setup_menu, setup_path, setup_error)) go(View::Installed);
        redraw();
        break;
      case ID_RUN:
        ShellExecuteW(nullptr, L"open", (setup::dir() + L"\\zero-gui.exe").c_str(), nullptr, nullptr, SW_SHOWNORMAL);
        PostMessageW(wnd, WM_CLOSE, 0, 0);
        break;
      case ID_SHELL:
        if (shell_on) shell::uninstall(); else shell::install();
        shell_on = shell::installed();
        redraw();
        break;
    }
  }
  int hit(float x, float y) const {
    for (size_t i = hots.size(); i-- > 0;) {
      const D2D1_RECT_F& r = hots[i].r;
      if (x >= r.left && x < r.right && y >= r.top && y < r.bottom) return hots[i].id;
    }
    return ID_NONE;
  }
  float dip(LPARAM v) const { return (float)(short)v * 96.0f / dpi; }
  LRESULT proc(UINT msg, WPARAM wp, LPARAM lp) {
    switch (msg) {
      case WM_CREATE:
        dpi = GetDpiForWindow(wnd);
        create_resources();
        load_theme();
        shell_on = shell::installed();
        DragAcceptFiles(wnd, TRUE);
        CoCreateInstance(CLSID_TaskbarList, nullptr, CLSCTX_INPROC_SERVER, IID_PPV_ARGS(&taskbar));
        if (taskbar) taskbar->HrInit();
        return 0;
      case WM_ERASEBKGND: return 1;
      case WM_SIZE: {
        W = LOWORD(lp) * 96.0f / dpi;
        H = HIWORD(lp) * 96.0f / dpi;
        if (rt) rt->Resize(D2D1::SizeU(LOWORD(lp), HIWORD(lp)));
        redraw();
        return 0;
      }
      case WM_PAINT: {
        PAINTSTRUCT ps;
        BeginPaint(wnd, &ps);
        paint();
        EndPaint(wnd, &ps);
        return 0;
      }
      case WM_DPICHANGED: {
        dpi = HIWORD(wp);
        const RECT* r = (const RECT*)lp;
        SetWindowPos(wnd, nullptr, r->left, r->top, r->right - r->left, r->bottom - r->top, SWP_NOZORDER | SWP_NOACTIVATE);
        make_target();
        redraw();
        return 0;
      }
      case WM_SETTINGCHANGE:
        if (lp && wcscmp((const wchar_t*)lp, L"ImmersiveColorSet") == 0) { load_theme(); redraw(); }
        return 0;
      case WM_GETMINMAXINFO: {
        MINMAXINFO* mm = (MINMAXINFO*)lp;
        const UINT d = GetDpiForWindow(wnd);
        mm->ptMinTrackSize.x = MulDiv(620, d, 96);
        mm->ptMinTrackSize.y = MulDiv(640, d, 96);
        return 0;
      }
      case WM_MOUSEMOVE: {
        if (!tracking) {
          TRACKMOUSEEVENT e{sizeof e, TME_LEAVE, wnd, 0};
          TrackMouseEvent(&e);
          tracking = true;
        }
        const int h = hit(dip(LOWORD(lp)), dip(HIWORD(lp)));
        if (h != hover) { hover = h; redraw(); }
        return 0;
      }
      case WM_MOUSELEAVE: tracking = false; hover = ID_NONE; redraw(); return 0;
      case WM_LBUTTONDOWN: focus = ID_NONE; down = hit(dip(LOWORD(lp)), dip(HIWORD(lp))); SetCapture(wnd); redraw(); return 0;
      case WM_LBUTTONUP: {
        ReleaseCapture();
        const int id = down;
        down = ID_NONE;
        redraw();
        if (id != ID_NONE && id == hit(dip(LOWORD(lp)), dip(HIWORD(lp)))) command(id);
        return 0;
      }
      case WM_MOUSEWHEEL: scroll -= (float)GET_WHEEL_DELTA_WPARAM(wp) * 0.4f; redraw(); return 0;
      case WM_KEYDOWN:
        if (wp == VK_TAB && !hots.empty()) {
          const bool back = GetKeyState(VK_SHIFT) < 0;
          int cur = -1;
          for (size_t i = 0; i < hots.size(); ++i) if (hots[i].id == focus) cur = (int)i;
          const int n = (int)hots.size();
          cur = cur < 0 ? (back ? n - 1 : 0) : (cur + (back ? n - 1 : 1)) % n;
          focus = hots[cur].id;
          redraw();
        } else if ((wp == VK_RETURN || wp == VK_SPACE) && focus != ID_NONE) {
          command(focus);
        } else if (wp == VK_ESCAPE) {
          if (view == View::Working) command(ID_CANCEL);
          else if (view != View::Home && view != View::Setup && view != View::Installed) command(ID_BACK);
        }
        return 0;
      case WM_DROPFILES: {
        HDROP drop = (HDROP)wp;
        if (view == View::Setup || view == View::Installed) { DragFinish(drop); return 0; }
        const UINT n = DragQueryFileW(drop, 0xFFFFFFFF, nullptr, 0);
        std::vector<std::string> paths;
        for (UINT i = 0; i < n; ++i) {
          wchar_t buf[4096];
          if (DragQueryFileW(drop, i, buf, 4096)) paths.push_back(narrow(buf));
        }
        DragFinish(drop);
        if (view == View::Done || view == View::Archive) start_over();
        open_paths(paths);
        return 0;
      }
      case WM_COMMAND: command(LOWORD(wp)); return 0;
      case WM_COPYDATA: {
        const COPYDATASTRUCT* c = (const COPYDATASTRUCT*)lp;
        if (c->dwData == 1) { out_path = narrow((const wchar_t*)c->lpData); out_chosen = true; redraw(); }
        if (c->dwData == 2 && !busy) {
          std::vector<std::string> paths;
          for (const wchar_t* p = (const wchar_t*)c->lpData; *p; p += wcslen(p) + 1) paths.push_back(narrow(p));
          if (view == View::Done || view == View::Archive) start_over();
          open_paths(paths);
          if (IsIconic(wnd)) ShowWindow(wnd, SW_RESTORE);
          SetForegroundWindow(wnd);
        }
        return TRUE;
      }
      case WM_TEST:
        if (wp == 1) { block_idx = (int)lp; redraw(); return 0; }
        if (wp == 3) return (LRESULT)view;
        if (wp == 4) return (LRESULT)items.size();
        return busy ? 1 : last_failed ? 2 : 0;
      case WM_LOG: {
        std::string* s = (std::string*)lp;
        log.push_back(*s);
        if (log.size() > 200) log.erase(log.begin());
        delete s;
        return 0;
      }
      case WM_SCANNED: {
        Scanned* s = (Scanned*)lp;
        for (Item& it : items) if (it.path == s->path) { it.size = s->size; it.sized = true; }
        delete s;
        redraw();
        return 0;
      }
      case WM_PROGRESS:
        if (taskbar && prog_total.load()) {
          taskbar->SetProgressState(wnd, TBPF_NORMAL);
          taskbar->SetProgressValue(wnd, prog_done.load(), prog_total.load());
        }
        redraw();
        return 0;
      case WM_DONE: {
        DoneInfo* d = (DoneInfo*)lp;
        busy = false;
        if (worker.joinable()) worker.join();
        if (taskbar) taskbar->SetProgressState(wnd, TBPF_NOPROGRESS);
        if (d->cancelled) {
          go(back_to);
        } else {
          done = *d;
          last_failed = !d->ok;
          go(View::Done);
          if (GetForegroundWindow() != wnd) FlashWindow(wnd, TRUE);
        }
        delete d;
        return 0;
      }
      case WM_CLOSE:
        if (busy) { cancel = true; PostMessageW(wnd, WM_CLOSE, 1, 0); Sleep(10); return 0; }
        DestroyWindow(wnd);
        return 0;
      case WM_DESTROY: PostQuitMessage(0); return 0;
    }
    return DefWindowProcW(wnd, msg, wp, lp);
  }
  ~App() {
    release(taskbar);
    release(brush); release(rt); release(dashed);
    for (IDWriteTextFormat** f : {&f_big, &f_title, &f_h2, &f_body, &f_semi, &f_small, &f_path, &f_wrap, &f_icon, &f_icon_big}) release(*f);
    release(dw); release(d2d);
  }
};

inline App* g_app = nullptr;
inline LRESULT CALLBACK wndproc(HWND h, UINT msg, WPARAM wp, LPARAM lp) {
  if (!g_app) return DefWindowProcW(h, msg, wp, lp);
  if (!g_app->wnd) g_app->wnd = h;  // (WM_GETMINMAXINFO and WM_NCCREATE arrive before WM_CREATE)
  return g_app->proc(msg, wp, lp);
}

// Explorer starts one process per selected file: the first opens the window, the others hand it their path and leave.
// Returns true when the paths were handed over.
inline bool hand_over(const std::vector<std::string>& paths, HANDLE& starting) {
  for (int i = 0; i < 250; ++i) {
    HWND w = FindWindowW(L"ZeroMainWindow", nullptr);
    if (w) {
      if (SendMessageW(w, WM_TEST, 2, 0) == 1) return false;  // (it is working: this selection gets its own window)
      std::wstring all;
      for (const std::string& p : paths) { all += widen(p); all.push_back(L'\0'); }
      all.push_back(L'\0');
      COPYDATASTRUCT c{2, (DWORD)(all.size() * sizeof(wchar_t)), (void*)all.data()};
      DWORD pid = 0;
      GetWindowThreadProcessId(w, &pid);
      AllowSetForegroundWindow(pid);
      SendMessageW(w, WM_COPYDATA, 0, (LPARAM)&c);
      return true;
    }
    starting = CreateMutexW(nullptr, FALSE, L"Local\\ZeroWindowStarting");
    if (starting && GetLastError() != ERROR_ALREADY_EXISTS) return false;  // (this process opens the window)
    if (starting) { CloseHandle(starting); starting = nullptr; }
    Sleep(20);
  }
  return false;
}

inline int run_gui(std::vector<std::string> args) {
  SetProcessDpiAwarenessContext(DPI_AWARENESS_CONTEXT_PER_MONITOR_AWARE_V2);
  CoInitializeEx(nullptr, COINIT_APARTMENTTHREADED);
  bool add = false, extract = false;
  const auto has_flag = [&](const char* f) { return std::find(args.begin(), args.end(), f) != args.end(); };
  if (has_flag("--uninstall")) {
    const bool silent = has_flag("--silent");
    if (!setup::running_installed()) {
      if (!silent) MessageBoxW(nullptr, L"This is not the installed copy of zero.", L"zero", MB_ICONWARNING);
      CoUninitialize();
      return 1;
    }
    if (silent || MessageBoxW(nullptr, L"Remove zero from this PC? Your archives stay where they are.", L"Uninstall zero", MB_YESNO | MB_ICONQUESTION) == IDYES) {
      setup::uninstall();
      if (!silent) MessageBoxW(nullptr, L"zero has been removed.", L"Uninstall zero", MB_ICONINFORMATION);
    }
    CoUninitialize();
    return 0;
  }
  const bool setup_mode = setup::is_setup_exe();
  if (setup_mode && has_flag("--silent")) {
    std::wstring why;
    const bool ok = setup::install(!has_flag("--no-explorer"), has_flag("--add-path"), why);
    CoUninitialize();
    return ok ? 0 : 1;
  }
  if (!setup_mode && !args.empty()) {
    if (args[0] == "--install-shell") { shell::install(); CoUninitialize(); return 0; }
    if (args[0] == "--uninstall-shell") { shell::uninstall(); CoUninitialize(); return 0; }
    if (args[0] == "--add") { add = true; args.erase(args.begin()); }
    else if (args[0] == "--extract-here") { extract = true; args.erase(args.begin()); }
  }
  HANDLE starting = nullptr;
  if (add && hand_over(args, starting)) { CoUninitialize(); return 0; }
  {
    App app;
    g_app = &app;
    WNDCLASSW wc{};
    wc.lpfnWndProc = wndproc;
    wc.hInstance = GetModuleHandleW(nullptr);
    wc.lpszClassName = L"ZeroMainWindow";
    wc.hCursor = LoadCursorW(nullptr, MAKEINTRESOURCEW(32512) /* IDC_ARROW */);
    wc.hIcon = LoadIconW(wc.hInstance, MAKEINTRESOURCEW(1));
    RegisterClassW(&wc);
    const int dpi = GetDpiForSystem();
    HWND h = CreateWindowExW(0, wc.lpszClassName, L"zero", WS_OVERLAPPEDWINDOW, CW_USEDEFAULT, CW_USEDEFAULT, MulDiv(760, dpi, 96), MulDiv(700, dpi, 96), nullptr, nullptr, wc.hInstance, nullptr);
    if (!h) return 1;
    if (setup_mode) app.go(App::View::Setup);
    else app.open_paths(args);
    ShowWindow(h, SW_SHOW);
    if (starting) { CloseHandle(starting); starting = nullptr; }  // (the window exists: the next process finds it)
    if (extract && app.view == App::View::Archive) app.do_extract();
    MSG msg;
    while (GetMessageW(&msg, nullptr, 0, 0)) {
      TranslateMessage(&msg);
      DispatchMessageW(&msg);
    }
    if (app.worker.joinable()) app.worker.join();
    app.gpu.reset();
    g_app = nullptr;
  }
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
