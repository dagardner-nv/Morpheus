#include <cuda_runtime.h>
#include "matx.h"
#include <rmm/cuda_stream_view.hpp>
#include <rmm/device_buffer.hpp>

#include <cassert>
#include <cstdint>
#include <iostream>
#include <memory>
#include <ostream>
#include <vector>

void row_major_test()
{
    // row major 2d [2, 4]
    std::vector<double> input{
        0.1, 0.7, 0.7, 0.7,
        1.0, 0.9, 0.5, 0.9,
        1.0, 0.6, 0.2, 0.6,
        1.0, 0.2, 0.3, 0.2,
    };

    std::vector<double> expected_output{
        1.0, 0.9, 0.7, 0.9,
        1.0, 0.6, 0.3, 0.6
    };

    matx::index_t num_cols      = 4;
    matx::index_t num_rows      = 4;
    matx::index_t expected_rows = expected_output.size() / num_cols;

    assert((num_cols * num_rows) == input.size());
    assert(expected_rows == 2);

    matx::index_t buff_size = input.size() * sizeof(double);
    auto input_buffer     = std::make_shared<rmm::device_buffer>(buff_size, rmm::cuda_stream_per_thread);
    cudaMemcpy(input_buffer->data(), input.data(), input_buffer->size(), cudaMemcpyHostToDevice);

    auto output_buffer = std::make_shared<rmm::device_buffer>(expected_rows * num_cols * sizeof(double), input_buffer->stream(), input_buffer->memory_resource());

    // collapse rows 0 & 1 together
    {
        auto input_ptr = static_cast<double*>(input_buffer->data());
        auto output_ptr = static_cast<double*>(output_buffer->data());

        matx::DefaultDescriptor<2> input_desc{{2, num_cols},
                                              {num_cols, 1}};

        matx::DefaultDescriptor<1> output_desc{{num_cols}, {1}};

        auto input_tensor = matx::make_tensor<double, matx::DefaultDescriptor<2>>(input_ptr, std::move(input_desc));
        auto output_tensor = matx::make_tensor<double, matx::DefaultDescriptor<1>>(output_ptr, std::move(output_desc));

        matx::rmax(output_tensor, input_tensor.Permute({1, 0}), output_buffer->stream().value());
    }

    // collapse rows 2 & 3 together
    {
        auto input_ptr = static_cast<double*>(input_buffer->data()) + (2 * num_cols);
        auto output_ptr = static_cast<double*>(output_buffer->data()) + (1 * num_cols);

        matx::DefaultDescriptor<2> input_desc{{2, num_cols},
                                              {num_cols, 1}};

        matx::DefaultDescriptor<1> output_desc{{num_cols}, {1}};

        auto input_tensor = matx::make_tensor<double, matx::DefaultDescriptor<2>>(input_ptr, std::move(input_desc));
        auto output_tensor = matx::make_tensor<double, matx::DefaultDescriptor<1>>(output_ptr, std::move(output_desc));

        matx::rmax(output_tensor, input_tensor.Permute({1, 0}), output_buffer->stream().value());
    }

    std::vector<double> host_output(expected_output.size());
    cudaMemcpy(host_output.data(), output_buffer->data(), output_buffer->size(), cudaMemcpyDeviceToHost);

    for (std::size_t i = 0; i < host_output.size(); ++i)
    {
        assert(host_output[i] == expected_output[i]);
    }
}

void col_major_test()
{
    // col major 2d [4, 4]
    std::vector<double> input{
        0.1, 1.0, 1.0, 1.0,
        0.7, 0.9, 0.6, 0.2,
        0.7, 0.5, 0.2, 0.3,
        0.7, 0.9, 0.6, 0.2
    };

    std::vector<double> expected_output{
        1.0, 1.0,
        0.9, 0.6,
        0.7, 0.3,
        0.9, 0.6
    };

    matx::index_t num_cols      = 4;
    matx::index_t num_rows      = 4;
    matx::index_t expected_rows = expected_output.size() / num_cols;

    assert((num_cols * num_rows) == input.size());
    assert(expected_rows == 2);

    matx::index_t buff_size = input.size() * sizeof(double);
    auto input_buffer     = std::make_shared<rmm::device_buffer>(buff_size, rmm::cuda_stream_per_thread);
    cudaMemcpy(input_buffer->data(), input.data(), input_buffer->size(), cudaMemcpyHostToDevice);

    auto tmp_buffer = std::make_shared<rmm::device_buffer>(expected_rows * num_cols * sizeof(double), input_buffer->stream(), input_buffer->memory_resource());

    // collapse rows 0 & 1 together
    {
        auto input_ptr = static_cast<double*>(input_buffer->data());
        auto output_ptr = static_cast<double*>(tmp_buffer->data());

        matx::DefaultDescriptor<2> input_desc{{2, num_cols},
                                              {1, num_rows}};

        matx::DefaultDescriptor<1> output_desc{{num_cols}, {1}};

        auto input_tensor = matx::make_tensor<double, matx::DefaultDescriptor<2>>(input_ptr, std::move(input_desc));
        auto output_tensor = matx::make_tensor<double, matx::DefaultDescriptor<1>>(output_ptr, std::move(output_desc));

        matx::rmax(output_tensor, input_tensor.Permute({1, 0}), tmp_buffer->stream().value());
    }

    std::vector<double> host_output(expected_output.size());
    cudaMemcpy(host_output.data(), tmp_buffer->data(), tmp_buffer->size(), cudaMemcpyDeviceToHost);

    /*
    for (std::size_t i = 0; i < host_output.size(); ++i)
    {
        std::cerr << "i= " << i << " v= " << host_output[i] << std::endl << std::flush;
    }
    std::cerr << "\n-----------\n";
    */

    // collapse rows 2 & 3 together
    {
        auto input_ptr = static_cast<double*>(input_buffer->data()) + (2);
        auto output_ptr = static_cast<double*>(tmp_buffer->data()) + (1 * num_cols);

        matx::DefaultDescriptor<2> input_desc{{2, num_cols},
                                              {1, num_rows}};

        matx::DefaultDescriptor<1> output_desc{{num_cols}, {1}};

        auto input_tensor = matx::make_tensor<double, matx::DefaultDescriptor<2>>(input_ptr, std::move(input_desc));
        auto output_tensor = matx::make_tensor<double, matx::DefaultDescriptor<1>>(output_ptr, std::move(output_desc));

        matx::rmax(output_tensor, input_tensor.Permute({1, 0}), tmp_buffer->stream().value());
    }

    cudaMemcpy(host_output.data(), tmp_buffer->data(), tmp_buffer->size(), cudaMemcpyDeviceToHost);

    /*
    for (std::size_t i = 0; i < host_output.size(); ++i)
    {
        std::cerr << "i= " << i << " v= " << host_output[i] << std::endl << std::flush;
    }
    std::cerr << "\n-----------\n";
    */

    auto output_buffer = std::make_shared<rmm::device_buffer>(expected_rows * num_cols * sizeof(double), input_buffer->stream(), input_buffer->memory_resource());
    // copy the row-major tmp_buffer to the output_buffer in column major
    {
        auto tmp_ptr = static_cast<double*>(tmp_buffer->data());
        auto output_ptr = static_cast<double*>(output_buffer->data());

        matx::DefaultDescriptor<2> tmp_desc{{expected_rows, num_cols},
                                              {num_cols, 1}};

        matx::DefaultDescriptor<2> output_desc{{expected_rows, num_cols}, {1, expected_rows}};

        auto tmp_tensor = matx::make_tensor<double, matx::DefaultDescriptor<2>>(tmp_ptr, std::move(tmp_desc));
        auto output_tensor = matx::make_tensor<double, matx::DefaultDescriptor<2>>(output_ptr, std::move(output_desc));

        (output_tensor = tmp_tensor).run(tmp_buffer->stream().value());
    }

    cudaMemcpy(host_output.data(), output_buffer->data(), output_buffer->size(), cudaMemcpyDeviceToHost);

    /*
    for (std::size_t i = 0; i < host_output.size(); ++i)
    {
        std::cerr << "i= " << i << " v= " << host_output[i] << std::endl << std::flush;
    }
    */

    for (std::size_t i = 0; i < host_output.size(); ++i)
    {
        assert(host_output[i] == expected_output[i]);
    }
}


void col_major_slice_test()
{
    // col major 2d [4, 4]
    std::vector<double> input{
        0.1, 1.0, 1.0, 1.0,
        0.7, 0.9, 0.6, 0.2,
        0.7, 0.5, 0.2, 0.3,
        0.7, 0.9, 0.6, 0.2
    };

    std::vector<double> expected_output{
        1.0, 1.0,
        0.9, 0.6,
        0.7, 0.3,
        0.9, 0.6
    };

    matx::index_t num_cols      = 4;
    matx::index_t num_rows      = 4;
    matx::index_t expected_rows = expected_output.size() / num_cols;

    assert((num_cols * num_rows) == input.size());
    assert(expected_rows == 2);

    matx::index_t buff_size = input.size() * sizeof(double);
    auto input_buffer     = std::make_shared<rmm::device_buffer>(buff_size, rmm::cuda_stream_per_thread);
    cudaMemcpy(input_buffer->data(), input.data(), input_buffer->size(), cudaMemcpyHostToDevice);

    auto output_buffer = std::make_shared<rmm::device_buffer>(expected_rows * num_cols * sizeof(double), input_buffer->stream(), input_buffer->memory_resource());

    auto input_ptr = static_cast<double*>(input_buffer->data());
    auto output_ptr = static_cast<double*>(output_buffer->data());

    matx::DefaultDescriptor<2> input_desc{{num_rows, num_cols}, {1, num_rows}};
    matx::DefaultDescriptor<2> output_desc{{expected_rows, num_cols}, {1, expected_rows}};

    auto input_tensor = matx::make_tensor<double, matx::DefaultDescriptor<2>>(input_ptr, std::move(input_desc));
    auto output_tensor = matx::make_tensor<double, matx::DefaultDescriptor<2>>(output_ptr, std::move(output_desc));

    input_tensor.Print();
    std::cerr << "-----------\n";
    output_tensor.Print();

    // collapse rows 0 & 1 together
    {
        auto input_slice = input_tensor.Slice({0, 0}, {2, matx::matxEnd});
        std::cerr << "-----------\nInput Slice\n";
        input_slice.Print();
        std::cerr << "-----------\n";
        auto output_slice = output_tensor.Slice<1>({0, 0}, {matx::matxDropDim, matx::matxEnd}, {1, expected_rows});
        std::cerr << "-----------\nOutput Slice\n";
        output_slice.Print();
        std::cerr << "-----------\n";

        matx::rmax(output_slice, input_slice.Permute({1, 0}), output_buffer->stream().value());
        std::cerr << "-----------\nOutput Slice\n";
        output_slice.Print();
        std::cerr << "-----------\n";
    }

    std::cerr << "-----------\n";
    output_tensor.Print();

    std::vector<double> host_output(expected_output.size());
    cudaMemcpy(host_output.data(), output_buffer->data(), output_buffer->size(), cudaMemcpyDeviceToHost);

    for (std::size_t i = 0; i < host_output.size(); ++i)
    {
        std::cerr << "i= " << i << " v= " << host_output[i] << std::endl << std::flush;
    }
    std::cerr << "\n-----------\n";

    // collapse rows 2 & 3 together
    {
        auto input_slice = input_tensor.Slice({2, 0}, {4, matx::matxEnd});
        auto output_slice = output_tensor.Slice<1>({1, 0}, {matx::matxDropDim, matx::matxEnd}, {expected_rows});

        matx::rmax(output_slice, input_slice.Permute({1, 0}), output_buffer->stream().value());
    }

    cudaMemcpy(host_output.data(), output_buffer->data(), output_buffer->size(), cudaMemcpyDeviceToHost);

    for (std::size_t i = 0; i < host_output.size(); ++i)
    {
        std::cerr << "i= " << i << " v= " << host_output[i] << std::endl << std::flush;
    }
    std::cerr << "\n-----------\n";

    for (std::size_t i = 0; i < host_output.size(); ++i)
    {
        assert(host_output[i] == expected_output[i]);
    }
}


int main([[maybe_unused]] int argc, [[maybe_unused]] char **argv)
{
    row_major_test();
    std::cout << "Row Major test passed" << std::endl;

    col_major_test();
    std::cout << "Col Major test passed" << std::endl;

    col_major_slice_test();
    std::cout << "Col Major Slice test passed" << std::endl;

    return 0;
}
