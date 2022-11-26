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

#include <runtime/local/vectorized/CPUTopology.h>
#include <runtime/local/vectorized/LoadPartitioningDefs.h>
#include <runtime/local/vectorized/TaskQueue.h>
#include <runtime/local/vectorized/Worker.h>

#include <array>
#include <cstdlib>
#include <memory>
#include <numeric>
#include <queue>
#include <thread>
#include <vector>
#include <sched.h>
#include <semaphore>

class CPUWorker : public Worker {
protected:
    int _threadId;
    int _coreId;

    std::vector<TaskQueue *> _q;
    // TODO: make this configurable
    std::array<bool, 256> eofWorkers;

    CPUTopology _topo;
    QueueAttrs _qattrs;

    std::binary_semaphore _runLock;
    std::binary_semaphore _workLock;
    
    int getNextQueue(int targetQueue, int initialQueue, int currentDomain, VictimSelectionLogic &stealLogic) {
        auto numQ = _qattrs._numQueues;

        auto should_stop = false; 
        do {
            switch (stealLogic) {
                case SEQ:
                    /* for SEQ, we stop when we wrap around */
                    targetQueue = (targetQueue + 1) % numQ;
                    should_stop = (targetQueue == initialQueue);
                    break;
                case SEQPRI:
                    /* for SEQPRI, we stop when we're out of domain workers / queues */
                    targetQueue = (targetQueue + 1) % numQ;
                    should_stop = (_topo._physicalIds[targetQueue] != currentDomain);
                    break;
                case RANDOM:
                    /* for RANDOM and RANDOMPRI, we stop when every worker is done */
                    targetQueue = rand() % numQ;
                    should_stop = (std::accumulate(eofWorkers.begin(), eofWorkers.end(), 0) < numQ);
                    break;
                case RANDOMPRI: {
                    /* for RANDOMPRI, that means only the domain workers */
                    auto n = std::accumulate(_topo._physicalIds.begin(),
                            _topo._physicalIds.end(), 0,
                            [=](int cur, int id) { return id == currentDomain ? cur + 1 : cur; });
                    targetQueue = rand() % numQ;
                    should_stop = (std::accumulate(eofWorkers.begin(), eofWorkers.end(), 0) < n);
                    break;
                }
                default:
                    throw std::runtime_error("unknown steal logic mode");
            }
        } while (!should_stop);

        /* for the non-PRI variants, we're done when we get out of the loop */
        if (stealLogic == SEQ || stealLogic == RANDOM) {
            return -1;
        }

        /*
         * for the PRI variants, we fall-back to SEQ/RANDOM for the non
         * priority queues, so clear bit 1 and continue
         */
        stealLogic = stealLogic == SEQ ? SEQPRI : RANDOMPRI;

        return targetQueue;
    }

public:
    explicit CPUWorker(int threadId, int coreId, std::vector<TaskQueue *> q,
            CPUTopology topo, QueueAttrs qattrs) :
        Worker(), _threadId(threadId), _coreId(coreId), _q(q), _topo(topo),
        _qattrs(qattrs), _runLock(0), _workLock(0) {
        t = std::make_unique<std::thread>(&CPUWorker::run, this);
    }

    ~CPUWorker() override = default;

    void run() override {
        if (_coreId >= 0) {
            CPUTopology::pinCPU(_coreId);
        }

        auto currentDomain = _topo._physicalIds[_threadId];
        auto targetQueue = _threadId;
        
        switch (_qattrs._queueMode) {
            case CENTRALIZED:
                targetQueue = 0;
                break;
            case PERGROUP:
                targetQueue = currentDomain;
                break;
            case PERCPU:
                targetQueue = _threadId;
                break;
            default:
                std::cerr << "Invalid queue mode" << std::endl;
        }

        while (true) {
            auto startingQueue = targetQueue;
            auto stealLogic = _qattrs._stealLogic;

            _runLock.acquire();

            do {
                Task *tsk = _q[targetQueue]->dequeueTask();
                if (!isEOF(tsk)) {
                    tsk->execute();
                    delete tsk;
                }
                targetQueue = getNextQueue(targetQueue, startingQueue, currentDomain, stealLogic);
            } while (targetQueue >= 0);

            _workLock.release();
        }
    }

    void wait() override {
        _workLock.acquire();
    }

    void unblock() override {
        _runLock.release();
    }
};
