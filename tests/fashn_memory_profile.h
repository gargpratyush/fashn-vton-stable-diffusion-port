#ifndef __SD_TESTS_FASHN_MEMORY_PROFILE_H__
#define __SD_TESTS_FASHN_MEMORY_PROFILE_H__

#include <chrono>
#include <fstream>
#include <memory>
#include <system_error>
#include "fashn_test_runner.h"
#include "ggml-cpu.h"

#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#include <psapi.h>
#include <tlhelp32.h>
#endif

class FashnMemoryProfile {
    std::ofstream stream_;
    bool failed_ = false;
    bool classify_;
    std::chrono::steady_clock::time_point start_ = std::chrono::steady_clock::now();
    size_t planned_cpu_work_bytes_ = 0;
    std::set<std::string> classified_phases_;

#ifdef _WIN32
    static void require_win(bool success, const char* operation) {
        if (!success)
            throw std::system_error(static_cast<int>(GetLastError()), std::system_category(), operation);
    }
    static uint64_t resident_bytes(uintptr_t address, size_t bytes, size_t page_size) {
        uint64_t result = 0;
        std::vector<PSAPI_WORKING_SET_EX_INFORMATION> pages;
        for (size_t offset = 0; offset < bytes;) {
            size_t count = std::min<size_t>(4096, (bytes - offset + page_size - 1) / page_size);
            pages.assign(count, {});
            for (size_t i = 0; i < count; ++i)
                pages[i].VirtualAddress = reinterpret_cast<void*>(address + offset + i * page_size);
            require_win(QueryWorkingSetEx(GetCurrentProcess(), pages.data(),
                                          static_cast<DWORD>(pages.size() * sizeof(pages[0]))) != 0,
                        "QueryWorkingSetEx");
            for (const auto& page : pages)
                result += page.VirtualAttributes.Valid ? page_size : 0;
            offset += count * page_size;
        }
        return result;
    }
    static uint32_t thread_count() {
        HANDLE handle = CreateToolhelp32Snapshot(TH32CS_SNAPPROCESS, 0);
        require_win(handle != INVALID_HANDLE_VALUE, "CreateToolhelp32Snapshot");
        std::unique_ptr<void, decltype(&CloseHandle)> guard(handle, CloseHandle);
        PROCESSENTRY32 entry{};
        entry.dwSize = sizeof(entry);
        require_win(Process32First(handle, &entry) != 0, "Process32First");
        do {
            if (entry.th32ProcessID == GetCurrentProcessId())
                return entry.cntThreads;
        } while (Process32Next(handle, &entry));
        throw std::runtime_error("Current process absent from process snapshot");
    }
    static nlohmann::ordered_json resident_regions(const SYSTEM_INFO& info) {
        nlohmann::ordered_json result = {{"private", 0ULL}, {"mapped", 0ULL}, {"image", 0ULL}};
        uintptr_t address = 0;
        const auto maximum = reinterpret_cast<uintptr_t>(info.lpMaximumApplicationAddress);
        while (address <= maximum) {
            MEMORY_BASIC_INFORMATION region{};
            require_win(VirtualQueryEx(GetCurrentProcess(), reinterpret_cast<void*>(address),
                                      &region, sizeof(region)) != 0, "VirtualQueryEx");
            if (region.State == MEM_COMMIT) {
                const char* kind = region.Type == MEM_IMAGE ? "image" :
                                   region.Type == MEM_MAPPED ? "mapped" : "private";
                result[kind] = result[kind].get<uint64_t>() +
                               resident_bytes(reinterpret_cast<uintptr_t>(region.BaseAddress),
                                              region.RegionSize, info.dwPageSize);
            }
            uintptr_t next = reinterpret_cast<uintptr_t>(region.BaseAddress) + region.RegionSize;
            if (next <= address)
                throw std::runtime_error("Invalid virtual region extent");
            address = next;
        }
        return result;
    }
#endif

public:
    explicit FashnMemoryProfile(const std::string& path, bool classify)
        : stream_(path), classify_(classify) {
        if (!stream_)
            throw std::runtime_error("Cannot create memory profile: " + path);
    }
    bool good() const { return !failed_ && stream_.good(); }

    void record(const char* phase, const FashnGraphTestContext* context = nullptr,
                const LoadDiagnostics* loads = nullptr, ggml_cgraph* graph = nullptr, int threads = 0) noexcept {
        if (failed_)
            return;
        try {
            auto begin = std::chrono::steady_clock::now();
            nlohmann::ordered_json row = {
                {"phase", phase},
                {"elapsed_seconds", std::chrono::duration<double>(begin - start_).count()},
                {"unix_ns", std::chrono::duration_cast<std::chrono::nanoseconds>(
                                std::chrono::system_clock::now().time_since_epoch()).count()}};
            if (graph != nullptr && threads > 0 && std::string(phase) == "graph_build_end")
                planned_cpu_work_bytes_ = ggml_graph_plan(graph, threads, nullptr).work_size;
            row["planned_cpu_work_bytes"] = planned_cpu_work_bytes_;
            row["backend_internal_allocation_bytes"] = nullptr;
            row["runtime_buffer_bytes"] = context && context->runner ? context->runner->diagnostic_runtime_bytes() : 0;
            row["workspace_bytes_by_backend"] = context && context->runner ?
                context->runner->diagnostic_workspace_bytes() : std::map<std::string, size_t>{};
            row["modulation_cache_bytes"] = context && context->runner ? context->runner->modulation_cache_bytes() : 0;
            row["inactive_modulation_parameter_bytes"] = context && context->runner ? context->runner->inactive_modulation_parameter_bytes() : 0;
            if (context && context->manager) {
                const auto weights = context->manager->params_memory_snapshot();
                row["weights"] = {{"registered_payload_bytes", weights.registered_bytes},
                                  {"assigned_payload_bytes", weights.assigned_bytes},
                                  {"allocated_buffer_bytes", weights.allocated_buffer_bytes},
                                  {"directly_mapped_payload_bytes", weights.directly_mapped_bytes}};
            }
            if (loads) {
                row["conversion"] = {{"read_live_bytes", loads->read_live.load()},
                                     {"read_peak_bytes", loads->read_peak.load()},
                                     {"f32_live_bytes", loads->f32_live.load()},
                                     {"f32_peak_bytes", loads->f32_peak.load()},
                                     {"scratch_live_bytes", loads->scratch_live.load()},
                                     {"scratch_peak_bytes", loads->scratch_peak.load()},
                                     {"converted_tensors", loads->converted_tensors.load()}};
            }
#ifdef _WIN32
            PROCESS_MEMORY_COUNTERS_EX memory{};
            memory.cb = sizeof(memory);
            require_win(GetProcessMemoryInfo(GetCurrentProcess(),
                                             reinterpret_cast<PROCESS_MEMORY_COUNTERS*>(&memory),
                                             sizeof(memory)) != 0, "GetProcessMemoryInfo");
            row["process"] = {{"pid", GetCurrentProcessId()}, {"working_set_bytes", memory.WorkingSetSize},
                              {"peak_working_set_bytes", memory.PeakWorkingSetSize},
                              {"private_commit_bytes", memory.PrivateUsage},
                              {"page_fault_count", memory.PageFaultCount}, {"threads", thread_count()}};
            const std::string name(phase);
            const bool classify = classify_ && name != "conversion_sample" &&
                                  name != "graph_execute_begin" && name != "graph_execute_end" &&
                                  name != "graph_build_begin" && name != "graph_build_end" &&
                                  name != "graph_output_read" && name != "step_observer_begin" &&
                                  name != "step_observer_end" && classified_phases_.insert(name).second;
            SYSTEM_INFO info{};
            GetSystemInfo(&info);
            if (classify) {
                row["resident_bytes_by_region_type"] = resident_regions(info);
                row["classification_note"] = "Non-atomic diagnostic scan by mapping type, not shared/private ownership; excludes no pages by subtraction of unrelated peaks.";
            }
            row["mapped_model_files"] = nlohmann::ordered_json::array();
            if (context && context->manager) {
                for (const auto& mapping : context->manager->loader().mapped_files()) {
                    nlohmann::ordered_json file = {{"path", mapping.path}, {"virtual_bytes", mapping.bytes}};
                    if (classify)
                        file["resident_bytes"] = resident_bytes(mapping.address, mapping.bytes, info.dwPageSize);
                    row["mapped_model_files"].push_back(std::move(file));
                }
            }
#else
            row["process"] = nullptr;
            row["process_memory_unavailable"] = "Windows process/page accounting only";
#endif
            row["diagnostic_seconds"] = std::chrono::duration<double>(std::chrono::steady_clock::now() - begin).count();
            stream_ << row.dump() << "\n";
            stream_.flush();
            if (!stream_)
                throw std::runtime_error("Cannot write memory profile");
        } catch (const std::exception& error) {
            failed_ = true;
            std::cerr << "Memory profiling failed: " << error.what() << "\n";
        }
    }
};

#endif
