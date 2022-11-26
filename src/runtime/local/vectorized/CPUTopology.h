/*
 * Copyright 2021 The DAPHNE Consortium
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#pragma once

#include <runtime/local/vectorized/LoadPartitioningDefs.h>
#include <runtime/local/context/DaphneContext.h>

#include <fstream>
#include <set>
#include <string>
#include <vector>

class CPUTopology {
protected:
    std::string _cpuinfoPath = "/proc/cpuinfo";

    bool _parseStringLine(const std::string &input,
            const std::string &keyword, int *val) {
        auto seperatorLocation = input.find(':');
        if (seperatorLocation != std::string::npos) {
            if (input.find(keyword) == 0) {
                *val = stoi(input.substr(seperatorLocation + 1));
                return true;
            }
        }
        return false;
    }

public:
    std::vector<int> _physicalIds;
    std::vector<int> _uniqueThreads;
    std::vector<int> _responsibleThreads;
    int _totalNumaDomains;
    
    DCTX(_ctx);

    explicit CPUTopology(DCTX(ctx)) : _ctx(ctx) {
        std::ifstream cpuinfoFile(_cpuinfoPath);

        std::vector<int> utilizedThreads;
        std::vector<int> core_ids;

        int index = 0;
        if (cpuinfoFile.is_open()) {
            std::string line;
            int value;
            while (std::getline(cpuinfoFile, line)) {
                if (_parseStringLine(line, "processor", &value )) {
                    utilizedThreads.push_back(value);
                } else if (_parseStringLine(line, "physical id", &value)) {
                    if (_ctx->getUserConfig().queueSetupScheme == PERGROUP) {
                        if (std::find(_physicalIds.begin(), _physicalIds.end(), value) == _physicalIds.end()) {
                            _responsibleThreads.push_back(utilizedThreads[index]);
                        }
                    }
                    _physicalIds.push_back(value);
                } else if (_parseStringLine(line, "core id", &value)) {
                    int found = 0;
                    for (int i = 0; i < index; i++) {
                        if (core_ids[i] == value && _physicalIds[i] == _physicalIds[index]) {
                                found++;
                        }
                    }
                    core_ids.push_back(value);
                    if (_ctx->config.hyperthreadingEnabled || found == 0) {
                        _uniqueThreads.push_back(utilizedThreads[index]);
                        if (_ctx->getUserConfig().queueSetupScheme == PERCPU) {
                            _responsibleThreads.push_back(value);
                        } else if (_ctx->getUserConfig().queueSetupScheme == CENTRALIZED) {
                            _responsibleThreads.push_back(0);
                        }
                    }
                    index++;
                }
            }
            cpuinfoFile.close();
        }
        _totalNumaDomains = std::set<double>(_physicalIds.begin(), _physicalIds.end()).size();
    }

    static void pinCPU(int id) {
        cpu_set_t cpuset;
        CPU_ZERO(&cpuset);
        CPU_SET(id, &cpuset);
        sched_setaffinity(0, sizeof(cpu_set_t), &cpuset);
    }
};
