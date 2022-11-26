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

#include <ir/daphneir/Daphne.h>
#include <runtime/local/context/DaphneContext.h>
#include <runtime/local/vectorized/CPUTopology.h>
#include <runtime/local/vectorized/LoadPartitioningDefs.h>
#include <runtime/local/vectorized/LoadPartitioning.h>
#include <runtime/local/vectorized/Task.h>
#include <runtime/local/vectorized/VectorizedData.h>
#include <runtime/local/vectorized/WorkerCPU.h>

#include <fstream>
#include <queue>
#include <vector>

using mlir::daphne::VectorSplit;

class VectorizedEngine {
    size_t _numCPUThreads{};
    bool _pinWorkers;
    std::vector<std::unique_ptr<Worker>> _cpuWorkers;
    std::vector<TaskQueue *> _queues;

    DCTX(_ctx);

    CPUTopology _topo;
    QueueAttrs _qattrs;

public:
    explicit VectorizedEngine(DCTX(ctx)) : _ctx(ctx), _topo(CPUTopology(ctx)) {
        _numCPUThreads = ctx->config.numberOfThreads > 0 ? ctx->config.numberOfThreads :
            std::thread::hardware_concurrency();

        if (_ctx->getUserConfig().queueSetupScheme != CENTRALIZED)
            _numCPUThreads = _topo._uniqueThreads.size();

        /*
         * If the available CPUs from Slurm is less than the configured num
         * threads, use the value from Slurm
         */
        if (const char *env_m = std::getenv("SLURM_CPUS_ON_NODE"))
            if (std::stoul(env_m) < _numCPUThreads)
                _numCPUThreads = std::stoi(env_m);

        if (std::thread::hardware_concurrency() < _topo._uniqueThreads.size()
                && _ctx->config.hyperthreadingEnabled)
            _topo._uniqueThreads.resize(_numCPUThreads);

        _qattrs._stealLogic = _ctx->getUserConfig().victimSelection;
        if (_ctx->getUserConfig().queueSetupScheme == PERGROUP) {
            _qattrs._queueMode = PERGROUP;
            _qattrs._numQueues = _topo._totalNumaDomains;
        } else if (_ctx->getUserConfig().queueSetupScheme == PERCPU) {
            _qattrs._queueMode = PERCPU;
            _qattrs._numQueues = _numCPUThreads;
        }

        if (_ctx->config.debugMultiThreading) {
            std::cout << "physicalIds:" << std::endl;
            for (const auto &topologyEntry: _topo._physicalIds) {
                std::cout << topologyEntry << ',';
            }
            std::cout << std::endl << "uniqueThreads:" << std::endl;
            for (const auto &topologyEntry: _topo._uniqueThreads) {
                std::cout << topologyEntry << ',';
            }
            std::cout << std::endl << "responsibleThreads:" << std::endl;
            for (const auto &topologyEntry: _topo._responsibleThreads) {
                std::cout << topologyEntry << ',';
            }
            std::cout << std::endl << "_totalNumaDomains=" << _topo._totalNumaDomains << std::endl;
            std::cout << "_numQueues=" << _qattrs._numQueues << std::endl;
        }
#ifndef NDEBUG
        std::cerr << "spawning " << _numCPUThreads << " CPU threads" << std::endl;
#endif
        _pinWorkers = ctx->getUserConfig().pinWorkers;

        for (int i = 0; i < _qattrs._numQueues; i++) {
            _queues.push_back(new BlockingTaskQueue());
        }

        for (size_t i = 0; i < _numCPUThreads; i++) {
            _cpuWorkers.push_back(std::make_unique<CPUWorker>(i,
                        _pinWorkers ? i : -1, _queues, _topo, _qattrs));
        }
    }

    ~VectorizedEngine() = default;
    void destroy() {}

    using PipelineFunc = void(*)(Structure **[], Structure *[], DCTX(ctx));
    void executeCPUQueues(std::vector<PipelineFunc> funcs,
            std::vector<VectorizedOutput *> outputs,
            std::vector<VectorizedInput *> inputs, DCTX(ctx)) {
        // TODO: factor mem usage as well
        size_t batchSize = _ctx->config.minimumBatchSize;
        size_t numRows = 0;

        for (auto input : inputs) {
            if (input->_isScalar || input->_split != VectorSplit::ROWS) {
                continue;
            }
            numRows = std::max(numRows, input->get()->getNumRows());
        }

        int method = ctx->config.taskPartitioningScheme;
        int minChunk = ctx->config.minimumTaskSize;

        std::vector<LoadPartitioning> lps;
        if (ctx->getUserConfig().prePartitionRows) {
            size_t chunk = numRows / _qattrs._numQueues;
            size_t rem = numRows - (chunk * _qattrs._numQueues);
            lps.emplace_back(method, chunk + rem, minChunk, _numCPUThreads, false);
            for (int i = 1; i < _qattrs._numQueues; i++) {
                lps.emplace_back(method, chunk, minChunk, _numCPUThreads, false);
            }
        } else {
            lps.emplace_back(method, numRows, minChunk, _numCPUThreads, false);
        }

        size_t start = 0;
        size_t end = 0;
        size_t target = 0;

        for (auto &w : _cpuWorkers) {
            w->unblock();
        }

        for (auto i = 0; i < _qattrs._numQueues; i++) {
            while (lps[i].hasNextChunk()) {
                end += lps[i].getNextChunk();

                auto task = new VectorizedTask(funcs[0], outputs, inputs,
                        start, end, batchSize, ctx);

                if (ctx->getUserConfig().pinWorkers) {
                    _queues[i]->enqueueTaskPinned(task, _topo._responsibleThreads[i]);
                } else {
                    _queues[target++ % _qattrs._numQueues]->enqueueTask(task);
                }

                start = end;
            }
        }

        for (int i = 0; i < _qattrs._numQueues; i++) {
            _queues[i]->closeInput();
        }

        for (auto &w : _cpuWorkers) {
            w->wait();
        }

        std::cout << "fin" << std::endl;
        while (true) ;
    }
};
