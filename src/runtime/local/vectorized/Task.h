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

#include <runtime/local/vectorized/VectorizedData.h>
#include <runtime/local/datastructures/DataObjectFactory.h>
#include <runtime/local/context/DaphneContext.h>
#include <ir/daphneir/Daphne.h>

#include <vector>

class Task {
public:
    virtual ~Task() = default;

    virtual void execute() = 0;
    virtual size_t getTaskSize() = 0;
};

// task for signaling closed input queue (no more tasks)
class EOFTask : public Task {
public:
    EOFTask() = default;
    ~EOFTask() override = default;
    void execute() override {}
    size_t getTaskSize() override {return 0;}
};

class VectorizedTask : public Task {
private:
    using SinkResults = std::tuple<size_t, size_t, Structure *>;
    using Sink = std::vector<SinkResults>;

    using PipelineFunc = void(*)(Structure **[], Structure *[], DCTX(ctx));
    PipelineFunc _func;

    std::vector<VectorizedOutput *> _outputs;
    const std::vector<VectorizedInput *> _inputs;

    const size_t _rl; // lower row index
    const size_t _ru; // upper row index
    const size_t _batchSize;

    DCTX(_ctx);

    std::vector<Structure *> createFuncInputs(size_t rl, size_t ru) {
        std::vector<Structure *> linputs;
        for (auto input : _inputs) {
            auto data = input->get();

            if (VectorSplit::ROWS == input->_split) {
                data = data->sliceRow(rl, ru);
            }

            /*
             * We need to increase the reference counter, since the pipeline
             * manages the reference counter itself.  This might be a scalar
             * disguised as a Structure.
             */
            if (input->isBroadcast() && !input->_isScalar) {
                /*
                 * Note that increaseRefCounter() synchronizes the access via a
                 * std::mutex. If that turns out to slow down things, creating
                 * a shallow copy of the input would be an alternative.
                 */
                data->increaseRefCounter();
            }

            linputs.push_back(data);
        }
        return linputs;
    }

    void collectResults(std::vector<Structure *> lresults,
            std::vector<Sink> lsinks, size_t rl, size_t ru) {
        for (size_t i = 0; i < _outputs.size(); i++) {
            auto lres = lresults[i];
            auto output = _outputs[i];

            if (output->get() == nullptr && !lres->inlineCombine()) {
                /* 'slowpath' sink combining */
                lsinks[i].push_back(std::make_tuple(rl, rl - ru, lres));
                continue;
            }

            if (output->_combine != VectorCombine::ADD) {
                /* for inline combining, this is a noop */
                continue;
            }

            /* aggregate the result */
            auto sink = std::get<2>(lsinks[i][0]);
            if (lres != sink) {
                //sink->combine(lres, VectorCombine::ADD);
                DataObjectFactory::destroy(lres);
                lres = nullptr;
            }
        }
    }

    void combineResults() {
        for (size_t i = 0; i < _outputs.size(); i++) {
            auto output = _outputs[i];
            
            if (output->get() != nullptr) {
                /* noop for inline combining */
                continue;
            }

            switch (output->_combine) {
                case VectorCombine::ROWS:
                    break;
                case VectorCombine::COLS:
                    break;
                case VectorCombine::ADD:
                    break;
                default:
                    throw std::runtime_error("unknown combine method");
            }
        }
    }

public:
    explicit VectorizedTask(PipelineFunc func,
            std::vector<VectorizedOutput *> outputs,
            std::vector<VectorizedInput *> inputs,
            size_t rl, size_t ru, size_t batchSize, DCTX(ctx))
        :  _func(func), _outputs(outputs), _inputs(inputs),
          _rl(rl), _ru(ru), _batchSize(batchSize), _ctx(ctx) {
        }

    void execute() override {
        std::vector<Structure *> lresults(_outputs.size(), nullptr);
        std::vector<Structure **> loutputs(_outputs.size());
        std::vector<Sink> lsinks(_outputs.size());

        for (size_t i = 0; i < _outputs.size(); i++) {
            loutputs[i] = &lresults[i];
        }

        /*
         * FIXME: We do a first run to create the first partial result, in
         * order to resolve the output type. This could be eliminated if the
         * compiler passed the output type information to the runtime.
         */
        size_t rl = _rl;
        size_t ru = std::min(rl + _batchSize, _ru);
        _func(loutputs.data(), createFuncInputs(rl, ru).data(), _ctx);

        for (size_t i = 0; i < _outputs.size(); i++) {
            auto lres = lresults[i];
            auto output = _outputs[i];

            if (!lres->inlineCombine()) {
                /*
                 * 'slowpath' combining for CSRMatrix, using sinks and
                 * combining at the end
                 */
                size_t runs = getTaskSize() / _batchSize;
                lsinks[i].reserve(runs);
                continue;
            }

            /*
             * for ADD aggregation, we use the first local structure
             * allocated
             */
            if (output->_combine == VectorCombine::ADD) {
                lsinks[i].push_back(std::make_tuple(output->_outRows,
                            output->_outCols, lres));
                continue;
            }

            /* inline combining for DenseMatrix */

            /* create the dest structure and copy the partial result */
            assert(output->get() == nullptr);
            output->createVecOutputFromTile(lres, _rl);
            /* free the intermediate results */
            DataObjectFactory::destroy(lres);
        }

        collectResults(lresults, lsinks, rl, ru);

        for (rl = ru; rl < _ru ; rl += _batchSize) {
            ru = std::min(rl + _batchSize, _ru);

            for (size_t i = 0; i < _outputs.size(); i++) {
                auto lres = lresults[i];
                auto output = _outputs[i];

                if (!lres->inlineCombine() || output->_combine == VectorCombine::ADD) {
                    /* resset lres */
                    lres = nullptr;
                    continue;
                }

                /* inline combining */
                switch (output->_combine) {
                    case VectorCombine::ROWS:
                        /* set lres to the appropriate row slice */
                        lres = output->get()->sliceRow(rl, ru);
                        break;
                    case VectorCombine::COLS:
                        /* set lres to the appropriate column slice */
                        lres = output->get()->sliceCol(rl, ru);
                        break;
                    default:
                        llvm_unreachable("should only get here with row or col combining");
                }
            }

            // exeute function on given data binding (batch size)
            _func(loutputs.data(), createFuncInputs(rl, ru).data(), _ctx);
            collectResults(lresults, lsinks, rl, ru);

            /*
             * Note that a pipeline manages the reference counters of its inputs
             * internally. Thus, we do not need to care about freeing the inputs
             * here.
             */
        }

        combineResults();
    }

    size_t getTaskSize() override {
        return _ru - _rl;
    }
};
