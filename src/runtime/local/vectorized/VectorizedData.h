/*
 *  Copyright 2021 The DAPHNE Consortium
 *
 *  Licensed under the Apache License, Version 2.0 (the "License");
 *  you may not use this file except in compliance with the License.
 *  You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 *  Unless required by applicable law or agreed to in writing, software
 *  distributed under the License is distributed on an "AS IS" BASIS,
 *  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  See the License for the specific language governing permissions and
 *  limitations under the License.
 */

#pragma once

#include <ir/daphneir/Daphne.h>

#include <mutex>

using mlir::daphne::VectorSplit;
using mlir::daphne::VectorCombine;

class VectorizedData {
protected:
    Structure *&_data;

public:
    Structure *get() {
        return _data;
    }

    void set(Structure *data) {
        _data = data;
    }

    explicit VectorizedData(Structure *&data) : _data(data) {}
    virtual ~VectorizedData() = default;
};

class VectorizedInput : public VectorizedData {
public:
    const VectorSplit _split;
    const bool _isScalar;

    bool isBroadcast() {
        return _split == VectorSplit::NONE ||
            (_split == VectorSplit::ROWS && _data->getNumRows() == 1);
    }

    explicit VectorizedInput(Structure *&data, VectorSplit split, bool isScalar)
        : VectorizedData(data), _split(split), _isScalar(isScalar) {}

    ~VectorizedInput() override = default;
};

class VectorizedOutput : public VectorizedData {
public:
    const VectorCombine _combine;
    const size_t _outRows;
    const size_t _outCols;
    std::mutex _mtx;

    bool inlineCombine() {
        return _combine != VectorCombine::ADD;
    }

    explicit VectorizedOutput(Structure *&data, VectorCombine combine,
            size_t outRows, size_t outCols)
        : VectorizedData(data), _combine(combine), _outRows(outRows), _outCols(outCols) {}

    ~VectorizedOutput() override = default;

    void createVecOutputFromTile(Structure *result, size_t row) {
        const std::lock_guard<std::mutex> lock(_mtx);

        size_t r = (_combine == VectorCombine::ROWS) ? row : 0;
        size_t c = (_combine == VectorCombine::COLS) ? row : 0;

        _data = result->createVecOutputFromTile(_outRows, _outCols, r, c);
    }
};
