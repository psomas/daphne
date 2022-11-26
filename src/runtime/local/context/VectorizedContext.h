/*
 * Copyright 2022 The DAPHNE Consortium
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
#include <runtime/local/vectorized/VectorizedEngine.h>

class VectorizedContext final : public IContext {
protected:
    std::unique_ptr<VectorizedEngine> _engine;

public:
    explicit VectorizedContext() {}
    ~VectorizedContext() = default;

    static std::unique_ptr<IContext> createVectorizedContext(DaphneContext *ctx) {
        auto vctx = std::make_unique<VectorizedContext>();
        vctx->_engine = std::make_unique<VectorizedEngine>(ctx);
        return vctx;
    }

    void destroy() override {
        _engine->destroy();
    }

    static VectorizedContext *get(DaphneContext *ctx) {
        return dynamic_cast<VectorizedContext *>(ctx->getVectorizedContext());
    }

    std::unique_ptr<VectorizedEngine> getEngine() {
        assert(_engine != nullptr);
        return std::move(_engine);
    }

    void putEngine(std::unique_ptr<VectorizedEngine> engine) {
        assert(_engine == nullptr);
        _engine = std::move(engine);
    }
};
