// zero: the Linux window, GTK 3 loaded at run time with dlopen (libgtk-3.so.0), so building needs no GTK
// development files and the exe runs without a window system too (the CLI). The same flow as gui_win.h: files
// and folders added by drag and drop or the buttons, compressed into one .zero archive; a .zero archive extracted
// or tested; the work on a thread, progress and log lines sent to the window through g_idle_add.
// Built from main.cpp with ZERO_GUI on Linux. Written against the GTK 3 API; not yet run on a Linux desktop.
#pragma once
#include <dlfcn.h>
#include <atomic>
#include <cstdarg>
#include <functional>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

namespace zero {
namespace gui {

// ---- the GTK / GLib functions used, resolved from libgtk-3 (dlsym searches its dependencies too)
typedef void GtkWidget;
typedef void GtkWindow;
typedef void GtkContainer;
typedef void GtkBox;
typedef void GtkEntry;
typedef void GtkComboBoxText;
typedef void GtkComboBox;
typedef void GtkToggleButton;
typedef void GtkListBox;
typedef void GtkListBoxRow;
typedef void GtkTextView;
typedef void GtkTextBuffer;
typedef void GtkProgressBar;
typedef void GtkLabel;
typedef void GtkScrolledWindow;
typedef void GtkDialog;
typedef void GtkFileChooser;
typedef void GtkSelectionData;
typedef void GdkDragContext;
typedef void* gpointer;
typedef int gint;
typedef int gboolean;
typedef unsigned int guint;
struct GList { gpointer data; GList* next; GList* prev; };
struct GSList { gpointer data; GSList* next; };
struct GtkTextIter { gpointer dummy[14]; };
typedef void (*GCallback)();
typedef gboolean (*GSourceFunc)(gpointer);

#define GTK_FUNCS(X)                                                                                                   \
  X(void, gtk_init, (int*, char***))                                                                                   \
  X(void, gtk_main, (void))                                                                                            \
  X(void, gtk_main_quit, (void))                                                                                       \
  X(GtkWidget*, gtk_window_new, (int))                                                                                 \
  X(void, gtk_window_set_title, (GtkWindow*, const char*))                                                             \
  X(void, gtk_window_set_default_size, (GtkWindow*, int, int))                                                         \
  X(GtkWidget*, gtk_box_new, (int, int))                                                                               \
  X(void, gtk_box_pack_start, (GtkBox*, GtkWidget*, gboolean, gboolean, guint))                                        \
  X(void, gtk_container_add, (GtkContainer*, GtkWidget*))                                                              \
  X(void, gtk_container_set_border_width, (GtkContainer*, guint))                                                      \
  X(GList*, gtk_container_get_children, (GtkContainer*))                                                               \
  X(GtkWidget*, gtk_button_new_with_label, (const char*))                                                              \
  X(GtkWidget*, gtk_check_button_new_with_label, (const char*))                                                        \
  X(gboolean, gtk_toggle_button_get_active, (GtkToggleButton*))                                                        \
  X(GtkWidget*, gtk_entry_new, (void))                                                                                 \
  X(const char*, gtk_entry_get_text, (GtkEntry*))                                                                      \
  X(void, gtk_entry_set_text, (GtkEntry*, const char*))                                                                \
  X(GtkWidget*, gtk_combo_box_text_new, (void))                                                                        \
  X(void, gtk_combo_box_text_append_text, (GtkComboBoxText*, const char*))                                             \
  X(void, gtk_combo_box_set_active, (GtkComboBox*, gint))                                                              \
  X(gint, gtk_combo_box_get_active, (GtkComboBox*))                                                                    \
  X(GtkWidget*, gtk_list_box_new, (void))                                                                              \
  X(void, gtk_list_box_insert, (GtkListBox*, GtkWidget*, gint))                                                        \
  X(void, gtk_list_box_set_selection_mode, (GtkListBox*, int))                                                         \
  X(GList*, gtk_list_box_get_selected_rows, (GtkListBox*))                                                             \
  X(gint, gtk_list_box_row_get_index, (GtkListBoxRow*))                                                                \
  X(GtkWidget*, gtk_label_new, (const char*))                                                                          \
  X(void, gtk_label_set_text, (GtkLabel*, const char*))                                                                \
  X(void, gtk_label_set_xalign, (GtkLabel*, float))                                                                    \
  X(void, gtk_label_set_ellipsize, (GtkLabel*, int))                                                                   \
  X(GtkWidget*, gtk_scrolled_window_new, (void*, void*))                                                               \
  X(void, gtk_scrolled_window_set_policy, (GtkScrolledWindow*, int, int))                                              \
  X(GtkWidget*, gtk_text_view_new, (void))                                                                             \
  X(void, gtk_text_view_set_editable, (GtkTextView*, gboolean))                                                        \
  X(void, gtk_text_view_set_cursor_visible, (GtkTextView*, gboolean))                                                  \
  X(GtkTextBuffer*, gtk_text_view_get_buffer, (GtkTextView*))                                                          \
  X(void, gtk_text_buffer_get_end_iter, (GtkTextBuffer*, GtkTextIter*))                                                \
  X(void, gtk_text_buffer_insert, (GtkTextBuffer*, GtkTextIter*, const char*, gint))                                   \
  X(GtkWidget*, gtk_progress_bar_new, (void))                                                                          \
  X(void, gtk_progress_bar_set_fraction, (GtkProgressBar*, double))                                                    \
  X(void, gtk_widget_show_all, (GtkWidget*))                                                                           \
  X(void, gtk_widget_destroy, (GtkWidget*))                                                                            \
  X(void, gtk_widget_set_sensitive, (GtkWidget*, gboolean))                                                            \
  X(void, gtk_widget_set_hexpand, (GtkWidget*, gboolean))                                                              \
  X(void, gtk_widget_set_vexpand, (GtkWidget*, gboolean))                                                              \
  X(void, gtk_widget_set_size_request, (GtkWidget*, gint, gint))                                                       \
  X(void, gtk_drag_dest_set, (GtkWidget*, int, void*, gint, int))                                                      \
  X(void, gtk_drag_dest_add_uri_targets, (GtkWidget*))                                                                 \
  X(char**, gtk_selection_data_get_uris, (GtkSelectionData*))                                                          \
  X(void, gtk_drag_finish, (GdkDragContext*, gboolean, gboolean, unsigned int))                                        \
  X(gint, gtk_dialog_run, (GtkDialog*))                                                                                \
  X(void, gtk_file_chooser_set_select_multiple, (GtkFileChooser*, gboolean))                                           \
  X(void, gtk_file_chooser_set_do_overwrite_confirmation, (GtkFileChooser*, gboolean))                                 \
  X(void, gtk_file_chooser_set_current_name, (GtkFileChooser*, const char*))                                           \
  X(GSList*, gtk_file_chooser_get_filenames, (GtkFileChooser*))                                                        \
  X(char*, gtk_file_chooser_get_filename, (GtkFileChooser*))                                                           \
  X(unsigned long, g_signal_connect_data, (gpointer, const char*, GCallback, gpointer, void*, int))                    \
  X(guint, g_idle_add, (GSourceFunc, gpointer))                                                                        \
  X(void, g_free, (gpointer))                                                                                          \
  X(void, g_strfreev, (char**))                                                                                        \
  X(void, g_slist_free, (GSList*))                                                                                     \
  X(void, g_list_free, (GList*))                                                                                       \
  X(char*, g_filename_from_uri, (const char*, char**, void**))

#define GTK_DECL(ret, name, args) typedef ret(*PFN_##name) args; inline PFN_##name name = nullptr;
GTK_FUNCS(GTK_DECL)
#undef GTK_DECL
// (variadic: declared by hand)
typedef GtkWidget* (*PFN_gtk_file_chooser_dialog_new)(const char*, GtkWindow*, int, const char*, ...);
inline PFN_gtk_file_chooser_dialog_new gtk_file_chooser_dialog_new = nullptr;
typedef GtkWidget* (*PFN_gtk_message_dialog_new)(GtkWindow*, int, int, int, const char*, ...);
inline PFN_gtk_message_dialog_new gtk_message_dialog_new = nullptr;

inline bool gtk_load(std::string& err) {
  void* h = dlopen("libgtk-3.so.0", RTLD_NOW | RTLD_GLOBAL);
  if (!h) h = dlopen("libgtk-3.so", RTLD_NOW | RTLD_GLOBAL);
  if (!h) { err = "GTK 3 (libgtk-3.so.0) is not installed; the command line works: zero --help"; return false; }
#define GTK_LOAD(ret, name, args) \
  name = (PFN_##name)dlsym(h, #name); \
  if (!name) { err = std::string("libgtk-3 lacks ") + #name; return false; }
  GTK_FUNCS(GTK_LOAD)
#undef GTK_LOAD
  gtk_file_chooser_dialog_new = (PFN_gtk_file_chooser_dialog_new)dlsym(h, "gtk_file_chooser_dialog_new");
  gtk_message_dialog_new = (PFN_gtk_message_dialog_new)dlsym(h, "gtk_message_dialog_new");
  if (!gtk_file_chooser_dialog_new || !gtk_message_dialog_new) { err = "libgtk-3 lacks the dialogs"; return false; }
  return true;
}

// GTK constants used
constexpr int GTK_WINDOW_TOPLEVEL = 0;
constexpr int GTK_ORIENTATION_HORIZONTAL = 0, GTK_ORIENTATION_VERTICAL = 1;
constexpr int GTK_SELECTION_MULTIPLE = 3;
constexpr int GTK_POLICY_AUTOMATIC = 1;
constexpr int PANGO_ELLIPSIZE_MIDDLE = 2;
constexpr int GTK_FILE_CHOOSER_ACTION_OPEN = 0, GTK_FILE_CHOOSER_ACTION_SAVE = 1, GTK_FILE_CHOOSER_ACTION_SELECT_FOLDER = 2;
constexpr int GTK_RESPONSE_ACCEPT = -3, GTK_RESPONSE_CANCEL = -6;
constexpr int GTK_DEST_DEFAULT_ALL = 7;
constexpr int GDK_ACTION_COPY = 2;
constexpr int GTK_MESSAGE_INFO = 0, GTK_MESSAGE_WARNING = 1;
constexpr int GTK_BUTTONS_OK = 1;

struct App;
inline App* g_app = nullptr;

// a function run on the GTK thread
struct Idle {
  std::function<void()> f;
  static gboolean run(gpointer p) { Idle* i = (Idle*)p; i->f(); delete i; return 0; }
};
inline void on_gtk_thread(std::function<void()> f) { g_idle_add(Idle::run, new Idle{std::move(f)}); }

struct App {
  GtkWidget *win = nullptr, *list = nullptr, *out = nullptr, *block = nullptr, *cpu = nullptr, *log = nullptr, *progress = nullptr,
            *status = nullptr;
  std::vector<GtkWidget*> buttons;  // add files, add folder, remove, clear, browse, compress, extract, test, cancel
  std::vector<std::string> items;
  std::thread worker;
  std::atomic<bool> cancel{false};
  bool busy = false;
  std::unique_ptr<GpuCodec> gpu;
  std::string gpu_error;

  void append_log(const std::string& s) {
    GtkTextBuffer* buf = gtk_text_view_get_buffer(log);
    GtkTextIter it;
    gtk_text_buffer_get_end_iter(buf, &it);
    const std::string line = s + "\n";
    gtk_text_buffer_insert(buf, &it, line.c_str(), (gint)line.size());
  }
  void set_status(const std::string& s) { gtk_label_set_text(status, s.c_str()); }
  void message(const std::string& s, bool warn) {
    GtkWidget* d = gtk_message_dialog_new(win, 1, warn ? GTK_MESSAGE_WARNING : GTK_MESSAGE_INFO, GTK_BUTTONS_OK, "%s", s.c_str());
    gtk_dialog_run(d);
    gtk_widget_destroy(d);
  }

  std::vector<std::string> pick(int action, bool multi, const char* def_name = nullptr) {
    std::vector<std::string> r;
    GtkWidget* d = gtk_file_chooser_dialog_new(action == GTK_FILE_CHOOSER_ACTION_SAVE ? "Save archive" : action == GTK_FILE_CHOOSER_ACTION_SELECT_FOLDER ? "Choose a folder" : "Choose files",
                                               win, action, "_Cancel", GTK_RESPONSE_CANCEL,
                                               action == GTK_FILE_CHOOSER_ACTION_SAVE ? "_Save" : "_Open", GTK_RESPONSE_ACCEPT, (const char*)nullptr);
    if (multi) gtk_file_chooser_set_select_multiple(d, 1);
    if (action == GTK_FILE_CHOOSER_ACTION_SAVE) {
      gtk_file_chooser_set_do_overwrite_confirmation(d, 1);
      if (def_name) gtk_file_chooser_set_current_name(d, def_name);
    }
    if (gtk_dialog_run(d) == GTK_RESPONSE_ACCEPT) {
      GSList* l = gtk_file_chooser_get_filenames(d);
      for (GSList* p = l; p; p = p->next) { r.push_back((const char*)p->data); g_free(p->data); }
      g_slist_free(l);
    }
    gtk_widget_destroy(d);
    return r;
  }

  void refresh_list() {
    GList* ch = gtk_container_get_children(list);
    for (GList* p = ch; p; p = p->next) gtk_widget_destroy((GtkWidget*)p->data);
    g_list_free(ch);
    for (const auto& s : items) {
      GtkWidget* l = gtk_label_new(s.c_str());
      gtk_label_set_xalign(l, 0.0f);
      gtk_label_set_ellipsize(l, PANGO_ELLIPSIZE_MIDDLE);
      gtk_list_box_insert(list, l, -1);
    }
    gtk_widget_show_all(list);
  }
  void add_item(const std::string& p) {
    for (const auto& s : items) if (s == p) return;
    items.push_back(p);
    refresh_list();
    if (items.size() == 1) suggest_output();
  }
  void remove_selected() {
    GList* rows = gtk_list_box_get_selected_rows(list);
    std::vector<int> idx;
    for (GList* p = rows; p; p = p->next) idx.push_back(gtk_list_box_row_get_index(p->data));
    g_list_free(rows);
    std::sort(idx.rbegin(), idx.rend());
    for (int i : idx) if (i >= 0 && (size_t)i < items.size()) items.erase(items.begin() + i);
    refresh_list();
  }
  bool only_archive() const { return items.size() == 1 && items[0].size() > 5 && items[0].compare(items[0].size() - 5, 5, ".zero") == 0; }
  void suggest_output() {
    if (items.empty() || only_archive()) return;
    fs::path p = zero_path(items[0]);
    while (!p.empty() && !p.has_filename()) p = p.parent_path();
    const fs::path o = p.parent_path() / (zero_utf8(p.filename()) + ".zero");
    gtk_entry_set_text(out, zero_utf8(o).c_str());
  }
  uint32_t block_log() { return 16 + (uint32_t)gtk_combo_box_get_active(block); }
  bool use_cpu() { return gtk_toggle_button_get_active(cpu) != 0; }

  void set_busy(bool b) {
    busy = b;
    for (size_t i = 0; i < buttons.size(); ++i) gtk_widget_set_sensitive(buttons[i], i == 8 ? b : !b);
    gtk_widget_set_sensitive(list, !b);
    gtk_widget_set_sensitive(out, !b);
    gtk_widget_set_sensitive(block, !b);
    gtk_widget_set_sensitive(cpu, !b);
    if (!b) gtk_progress_bar_set_fraction(progress, 0.0);
  }
  Codec& codec_for(bool want_cpu, Report& rep, std::unique_ptr<CpuCodec>& holder) {
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
    holder = std::make_unique<CpuCodec>();
    return *holder;
  }
  void run(std::function<std::string(Report&)> job) {
    if (busy) return;
    if (worker.joinable()) worker.join();
    cancel = false;
    set_busy(true);
    worker = std::thread([this, job] {
      Report rep;
      rep.cancel = &cancel;
      rep.log = [this](const std::string& s) { on_gtk_thread([this, s] { append_log(s); }); };
      double last = 0;
      rep.progress = [this, &last](uint64_t d, uint64_t t) {
        const double now = zero_now();
        if (d < t && now - last < 0.05) return;
        last = now;
        const double f = t ? (double)d / (double)t : 1.0;
        on_gtk_thread([this, f] { gtk_progress_bar_set_fraction(progress, f); });
      };
      std::string summary;
      try {
        summary = job(rep);
      } catch (const Cancelled&) {
        summary = "Cancelled.";
      } catch (const std::exception& e) {
        summary = std::string("Error: ") + e.what();
      }
      on_gtk_thread([this, summary] {
        set_busy(false);
        const size_t nl = summary.find('\n');
        set_status(summary.substr(0, nl));
        append_log(summary);
      });
    });
  }
  void do_compress() {
    if (items.empty()) { message("Add files or folders first (drag them onto the window).", false); return; }
    const std::string outp = gtk_entry_get_text(out);
    if (outp.empty()) { message("Choose where to write the archive.", false); return; }
    for (const auto& it : items) if (it == outp) { message("The archive cannot be one of its own inputs.", true); return; }
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
    std::vector<std::string> r = pick(GTK_FILE_CHOOSER_ACTION_OPEN, false);
    return r.empty() ? "" : r[0];
  }
  void do_extract() {
    const std::string arc = archive_to_open();
    if (arc.empty()) return;
    std::vector<std::string> dir = pick(GTK_FILE_CHOOSER_ACTION_SELECT_FOLDER, false);
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

  // ---- signal handlers (static: GTK calls plain functions)
  static void on_add_files(GtkWidget*, gpointer) { for (const auto& p : g_app->pick(GTK_FILE_CHOOSER_ACTION_OPEN, true)) g_app->add_item(p); }
  static void on_add_folder(GtkWidget*, gpointer) { for (const auto& p : g_app->pick(GTK_FILE_CHOOSER_ACTION_SELECT_FOLDER, true)) g_app->add_item(p); }
  static void on_remove(GtkWidget*, gpointer) { g_app->remove_selected(); }
  static void on_clear(GtkWidget*, gpointer) { g_app->items.clear(); g_app->refresh_list(); }
  static void on_browse(GtkWidget*, gpointer) {
    std::string def = "archive.zero";
    if (!g_app->items.empty()) { fs::path p = zero_path(g_app->items[0]); while (!p.empty() && !p.has_filename()) p = p.parent_path(); def = zero_utf8(p.filename()) + ".zero"; }
    std::vector<std::string> r = g_app->pick(GTK_FILE_CHOOSER_ACTION_SAVE, false, def.c_str());
    if (!r.empty()) gtk_entry_set_text(g_app->out, r[0].c_str());
  }
  static void on_compress(GtkWidget*, gpointer) { g_app->do_compress(); }
  static void on_extract(GtkWidget*, gpointer) { g_app->do_extract(); }
  static void on_test(GtkWidget*, gpointer) { g_app->do_test(); }
  static void on_cancel(GtkWidget*, gpointer) { g_app->cancel = true; g_app->set_status("Cancelling..."); }
  static void on_drag(GtkWidget*, GdkDragContext* ctx, gint, gint, GtkSelectionData* data, guint, guint time, gpointer) {
    char** uris = gtk_selection_data_get_uris(data);
    if (uris) {
      for (char** u = uris; *u; ++u) {
        char* path = g_filename_from_uri(*u, nullptr, nullptr);
        if (path) { g_app->add_item(path); g_free(path); }
      }
      g_strfreev(uris);
    }
    gtk_drag_finish(ctx, 1, 0, time);
    if (g_app->only_archive()) g_app->set_status("A .zero archive: Extract... or Test...");
  }
  static void on_destroy(GtkWidget*, gpointer) { g_app->cancel = true; gtk_main_quit(); }

  GtkWidget* button(const char* label, void (*cb)(GtkWidget*, gpointer)) {
    GtkWidget* b = gtk_button_new_with_label(label);
    g_signal_connect_data(b, "clicked", (GCallback)cb, nullptr, nullptr, 0);
    buttons.push_back(b);
    return b;
  }
  void build() {
    win = gtk_window_new(GTK_WINDOW_TOPLEVEL);
    gtk_window_set_title(win, "zero - GPU compression");
    gtk_window_set_default_size(win, 760, 560);
    gtk_container_set_border_width(win, 10);
    g_signal_connect_data(win, "destroy", (GCallback)on_destroy, nullptr, nullptr, 0);
    GtkWidget* vbox = gtk_box_new(GTK_ORIENTATION_VERTICAL, 8);
    gtk_container_add(win, vbox);
    // the list and its buttons
    GtkWidget* top = gtk_box_new(GTK_ORIENTATION_HORIZONTAL, 8);
    gtk_box_pack_start(vbox, top, 1, 1, 0);
    GtkWidget* sw = gtk_scrolled_window_new(nullptr, nullptr);
    gtk_scrolled_window_set_policy(sw, GTK_POLICY_AUTOMATIC, GTK_POLICY_AUTOMATIC);
    list = gtk_list_box_new();
    gtk_list_box_set_selection_mode(list, GTK_SELECTION_MULTIPLE);
    gtk_container_add(sw, list);
    gtk_widget_set_hexpand(sw, 1);
    gtk_widget_set_vexpand(sw, 1);
    gtk_box_pack_start(top, sw, 1, 1, 0);
    GtkWidget* side = gtk_box_new(GTK_ORIENTATION_VERTICAL, 6);
    gtk_box_pack_start(top, side, 0, 0, 0);
    gtk_box_pack_start(side, button("Add files...", on_add_files), 0, 0, 0);
    gtk_box_pack_start(side, button("Add folder...", on_add_folder), 0, 0, 0);
    gtk_box_pack_start(side, button("Remove", on_remove), 0, 0, 0);
    gtk_box_pack_start(side, button("Clear", on_clear), 0, 0, 0);
    // the archive path
    GtkWidget* row = gtk_box_new(GTK_ORIENTATION_HORIZONTAL, 8);
    gtk_box_pack_start(vbox, row, 0, 0, 0);
    gtk_box_pack_start(row, gtk_label_new("Archive:"), 0, 0, 0);
    out = gtk_entry_new();
    gtk_box_pack_start(row, out, 1, 1, 0);
    gtk_box_pack_start(row, button("Browse...", on_browse), 0, 0, 0);
    // options and actions
    GtkWidget* row2 = gtk_box_new(GTK_ORIENTATION_HORIZONTAL, 8);
    gtk_box_pack_start(vbox, row2, 0, 0, 0);
    gtk_box_pack_start(row2, gtk_label_new("Block:"), 0, 0, 0);
    block = gtk_combo_box_text_new();
    for (const char* s : {"64 KiB", "128 KiB", "256 KiB", "512 KiB", "1 MiB"}) gtk_combo_box_text_append_text(block, s);
    gtk_combo_box_set_active(block, 3);
    gtk_box_pack_start(row2, block, 0, 0, 0);
    cpu = gtk_check_button_new_with_label("CPU codec");
    gtk_box_pack_start(row2, cpu, 0, 0, 0);
    GtkWidget* spacer = gtk_label_new("");
    gtk_box_pack_start(row2, spacer, 1, 1, 0);
    gtk_box_pack_start(row2, button("Compress", on_compress), 0, 0, 0);
    gtk_box_pack_start(row2, button("Extract...", on_extract), 0, 0, 0);
    gtk_box_pack_start(row2, button("Test...", on_test), 0, 0, 0);
    gtk_box_pack_start(row2, button("Cancel", on_cancel), 0, 0, 0);
    // progress, status, log
    progress = gtk_progress_bar_new();
    gtk_box_pack_start(vbox, progress, 0, 0, 0);
    status = gtk_label_new("Drop files and folders here, then Compress. Drop a .zero archive to extract or test it.");
    gtk_label_set_xalign(status, 0.0f);
    gtk_label_set_ellipsize(status, PANGO_ELLIPSIZE_MIDDLE);
    gtk_box_pack_start(vbox, status, 0, 0, 0);
    GtkWidget* sw2 = gtk_scrolled_window_new(nullptr, nullptr);
    gtk_scrolled_window_set_policy(sw2, GTK_POLICY_AUTOMATIC, GTK_POLICY_AUTOMATIC);
    gtk_widget_set_size_request(sw2, -1, 160);
    log = gtk_text_view_new();
    gtk_text_view_set_editable(log, 0);
    gtk_text_view_set_cursor_visible(log, 0);
    gtk_container_add(sw2, log);
    gtk_box_pack_start(vbox, sw2, 1, 1, 0);
    // drag and drop onto the whole window
    gtk_drag_dest_set(win, GTK_DEST_DEFAULT_ALL, nullptr, 0, GDK_ACTION_COPY);
    gtk_drag_dest_add_uri_targets(win);
    g_signal_connect_data(win, "drag-data-received", (GCallback)on_drag, nullptr, nullptr, 0);
    set_busy(false);
    gtk_widget_show_all(win);
  }
};

inline int run_gui(int argc, char** argv) {
  std::string err;
  if (!gtk_load(err)) { fprintf(stderr, "zero-gui: %s\n", err.c_str()); return 1; }
  std::vector<std::string> args;
  for (int i = 1; i < argc; ++i) args.push_back(argv[i]);
  gtk_init(&argc, &argv);
  App app;
  g_app = &app;
  app.build();
  for (const auto& a : args) app.add_item(a);
  if (app.only_archive()) app.set_status("A .zero archive: Extract... or Test...");
  gtk_main();
  if (app.worker.joinable()) app.worker.join();
  app.gpu.reset();
  return 0;
}

}  // namespace gui
}  // namespace zero

int main(int argc, char** argv) { return zero::gui::run_gui(argc, argv); }
