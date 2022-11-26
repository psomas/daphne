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

#include <runtime/local/context/DaphneContext.h>
#include <runtime/local/datastructures/DataObjectFactory.h>
#include <runtime/local/datastructures/Structure.h>
#include <runtime/local/vectorized/VectorizedEngine.h>
#include <runtime/local/vectorized/VectorizedData.h>
#include <ir/daphneir/Daphne.h>

#include <cassert>
#include <cstddef>

using mlir::daphne::VectorSplit;
using mlir::daphne::VectorCombine;

struct VectorizedPipeline {
    static void apply(Structure *outputs[], size_t numOutputs, bool isScalar[],
            Structure *inputs[], size_t numInputs, int64_t outRows[],
            int64_t outCols[], int64_t splits[], int64_t combines[],
            size_t numFuncs, void *funcs[], DCTX(ctx)) {
        using PipelineFunc = void(*)(Structure **[], Structure *[], DCTX(ctx));
        std::vector<PipelineFunc> _funcs;
        for (size_t i = 0; i < numFuncs; ++i) {
            _funcs.push_back(reinterpret_cast<PipelineFunc>(funcs[i]));
        }

        std::vector<VectorizedOutput *> _outputs;
        for (size_t i = 0; i < numOutputs; i++) {
            auto output = new VectorizedOutput(outputs[i],
                    static_cast<VectorCombine>(combines[i]),
                    outRows[i], outCols[i]);
            _outputs.push_back(output);
        }

        std::vector<VectorizedInput *> _inputs;
        for (size_t i = 0; i < numInputs; i++) {
            auto input = new VectorizedInput(inputs[i],
                    static_cast<VectorSplit>(splits[i]), isScalar[i]);
            _inputs.push_back(input);
        }

        // TODO: re-add the per-device and single execute options
        auto engine = std::move(VectorizedContext::get(ctx)->getEngine());
        engine->executeCPUQueues(_funcs, _outputs, _inputs, ctx);
        VectorizedContext::get(ctx)->putEngine(std::move(engine));

        for (size_t i = 0; i < numOutputs; i++) {
            delete _outputs[i];
        }
        for (size_t i = 0; i < numInputs; i++) {
            delete _inputs[i];
        }
    }
};

// ****************************************************************************
// Convenience function
// ****************************************************************************

[[maybe_unused]] void vectorizedPipeline(Structure *outputs[], size_t numOutputs,
        bool isScalar[], Structure *inputs[], size_t numInputs,
        int64_t outRows[], int64_t outCols[], int64_t splits[],
        int64_t combines[], size_t numFuncs, void *funcs[], DCTX(ctx)) {
    VectorizedPipeline::apply(outputs, numOutputs, isScalar, inputs, numInputs,
            outRows, outCols, splits, combines, numFuncs, funcs, ctx);
};
