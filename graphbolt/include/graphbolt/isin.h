/**
 *  Copyright (c) 2023 by Contributors
 *
 * @file graphbolt/isin.h
 * @brief isin op.
 */
#ifndef GRAPHBOLT_ISIN_H_
#define GRAPHBOLT_ISIN_H_

#include <graphbolt/async.h>
#include <torch/torch.h>

namespace graphbolt {
namespace sampling {

/**
 * @brief Tests if each element of elements is in test_elements. Returns a
 * boolean tensor of the same shape as elements that is True for elements
 * in test_elements and False otherwise. Enhance torch.isin by implementing
 * multi-threaded searching, as detailed in the documentation at
 * https://pytorch.org/docs/stable/generated/torch.isin.html."
 *
 * @param elements        Input elements
 * @param test_elements   Values against which to test for each input element.
 *
 * @return
 * A boolean tensor of the same shape as elements that is True for elements
 * in test_elements and False otherwise.
 */
torch::Tensor IsIn(
    const torch::Tensor& elements, const torch::Tensor& test_elements);

/**
 * @brief Tests if each element of elements is not in test_elements. Returns an
 * int64_t tensor of the same shape as elements containing the indexes of the
 * elements not found in test_elements.
 *
 * @param elements        Input elements
 * @param test_elements   Values against which to test for each input element.
 *
 * @return An int64_t tensor of the same shape as elements containing indexes of
 * elements not found in test_elements.
 */
torch::Tensor IsNotInIndex(
    const torch::Tensor& elements, const torch::Tensor& test_elements);

c10::intrusive_ptr<Future<torch::Tensor>> IsNotInIndexAsync(
    const torch::Tensor& elements, const torch::Tensor& test_elements);

}  // namespace sampling
}  // namespace graphbolt

#endif  // GRAPHBOLT_ISIN_H_
