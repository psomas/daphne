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

#ifndef SRC_RUNTIME_LOCAL_KERNELS_GEMV_H
#define SRC_RUNTIME_LOCAL_KERNELS_GEMV_H

#include <runtime/local/context/DaphneContext.h>
#include <runtime/local/datastructures/DataObjectFactory.h>
#include <runtime/local/datastructures/DenseMatrix.h>

#include <cblas.h>

// ****************************************************************************
// Struct for partial template specialization
// ****************************************************************************

template<class DTRes, class DTMat, class DTVec>
struct Gemv {
    static void apply(DTRes *& res, const DTMat * mat, const DTVec * vec, DCTX(ctx)) = delete;
};

// ****************************************************************************
// Convenience function
// ****************************************************************************

template<class DTRes, class DTMat, class DTVec>
void gemv(DTRes *& res, const DTMat * mat, const DTVec * vec, DCTX(ctx)) {
    Gemv<DTRes, DTMat, DTVec>::apply(res, mat, vec, ctx);
}

// ****************************************************************************
// (Partial) template specializations for different data/value types
// ****************************************************************************

// ----------------------------------------------------------------------------
// DenseMatrix <- DenseMatrix
// ----------------------------------------------------------------------------

template<>
struct Gemv<DenseMatrix<double>, DenseMatrix<double>, DenseMatrix<double>> {
    static void apply(DenseMatrix<double> *& res, const DenseMatrix<double> * mat, const DenseMatrix<double> * vec, DCTX(ctx)) {
        const size_t numRows = mat->getNumRows();
        const size_t numCols = mat->getNumCols();

        if(res == nullptr)
            res = DataObjectFactory::create<DenseMatrix<double>>(numCols, 1, false);

        cblas_dgemv(CblasRowMajor,
            CblasTrans,
            numRows,
            numCols,
            1.0,
            mat->getValues(),
            mat->getRowSkip(),
            vec->getValues(),
            vec->getRowSkip(),
            0.0,
            res->getValues(),
            res->getRowSkip()
        );
    }
};

template<>
struct Gemv<DenseMatrix<float>, DenseMatrix<float>, DenseMatrix<float>> {
    static void apply(DenseMatrix<float> *& res, const DenseMatrix<float> * mat, const DenseMatrix<float> * vec, DCTX(ctx)) {
        const size_t numRows = mat->getNumRows();
        const size_t numCols = mat->getNumCols();

        if(res == nullptr)
            res = DataObjectFactory::create<DenseMatrix<float>>(numCols, 1, false);

        cblas_sgemv(CblasRowMajor,
            CblasTrans,
            numRows,
            numCols,
            1.0,
            mat->getValues(),
            mat->getRowSkip(),
            vec->getValues(),
            vec->getRowSkip(),
            0.0,
            res->getValues(),
            res->getRowSkip()
        );
    }
};

// ----------------------------------------------------------------------------
// DenseMatrix <- CSRMatrix, DenseMatrix
// ----------------------------------------------------------------------------

template<typename VT>
struct Gemv<DenseMatrix<VT>, CSRMatrix<VT>, DenseMatrix<VT>> {
    static void apply(DenseMatrix<VT> *& res, const CSRMatrix<VT> * mat,const DenseMatrix<VT> * vec, DCTX(ctx)) {
        const size_t nr1 = mat->getNumRows();

        assert(nr1 == vec->getNumRows() && vec->getNumCols() == 1);

        if(res == nullptr)
            res = DataObjectFactory::create<DenseMatrix<VT>>(nr1, 1, true);

        const VT * valuesVec = vec->getValues();
        VT * valuesRes = res->getValues();

        const size_t rowSkipVec= vec->getRowSkip();
        const size_t rowSkipRes = res->getRowSkip();

        for(size_t r = 0; r < nr1; r++) {
            const size_t rowNumNonZeros = mat->getNumNonZeros(r);
            const size_t * rowColIdxs = mat->getColIdxs(r);
            const VT * rowValues = mat->getValues(r);

            const size_t rowIdxVec = r * rowSkipVec;
            for(size_t i = 0; i < rowNumNonZeros; i++) {
                const size_t c = rowColIdxs[i];
                const size_t rowIdxRes = c * rowSkipRes;
                valuesRes[rowIdxRes] += rowValues[i] * valuesVec[rowIdxVec];
            }
        }
    }
};

#endif //SRC_RUNTIME_LOCAL_KERNELS_GEMV_H
