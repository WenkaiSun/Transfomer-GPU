#include <iostream>
#include <time.h>
#include <vector>
#include <cmath>
#include <algorithm>
#include <torch/torch.h>
#include <random>
using namespace std;

// Function to perform matrix multiplication
vector<vector<double>> matMul(const vector<vector<double>>& a, const vector<vector<double>>& b) {
    int rowsA = a.size();
    int colsA = a[0].size();
    int rowsB = b.size();
    int colsB = b[0].size();
    vector<vector<double>> result(rowsA, vector<double>(rowsB, 0.0));
    for (int i = 0; i < rowsA; ++i) {
        for (int j = 0; j < rowsB; ++j) {
            for (int k = 0; k < colsA; ++k) {
                result[i][j] += a[i][k] * b[j][k];
            }
        }
    }
    return result;
}

vector<vector<vector<double>>> tenMul(const vector<vector<vector<double>>>& a, const vector<vector<vector<double>>>& b) {
    int batch_size = a.size();
    int seq_length = a[0].size();
    int embed_dim = b[0].size();
    vector<vector<vector<double>>> result(batch_size, vector<vector<double>>(seq_length, vector<double>(embed_dim, 0.0)));

    for (int i = 0; i < batch_size; ++i) {
        result[i] = matMul(a[i], b[i]);
    }

    return result;
}

// Function to compute the softmax of a matrix row-wise
vector<vector<vector<double>>> softmax(const vector<vector<vector<double>>>& logits) {
    vector<vector<vector<double>>> result = logits;
    for (auto& batch: result){
        for (auto& row : batch) {
            double maxLogit = *max_element(row.begin(), row.end());
            double sumExp = 0.0;
            for (double& val : row) {
                val = exp(val - maxLogit);
                sumExp += val;
            }
            for (double& val : row) {
                val /= sumExp;
            }
        }
    }
    return result;
}

// Transpose
vector<vector<vector<double>>> transpose(const vector<vector<vector<double>>>& matrix) {
    int batch_size = matrix.size();
    int rows = matrix[0].size();
    int cols = matrix[0][0].size();
    vector<vector<vector<double>>> result(batch_size,vector<vector<double>>(cols, vector<double>(rows, 0.0)));
    for (int i=0; i < batch_size; ++i)
    {
        for (int j = 0; j < rows; ++j) {
            for (int k = 0; k < cols; ++k) {
                result[i][k][j] = matrix[i][j][k];
            }
        }
    }
    return result;
}

// Tensor Slice
vector<vector<vector<double>>> tensor_slice(const vector<vector<vector<double>>>& input, int start,int end){
    vector<vector<vector<double>>> output(input.size(),vector<vector<double>>(input[0].size(),vector<double>(end-start, 0.0)));
    for (int i=0; i<input.size(); i++)
    {
        for (int j=0; j<input[0].size();j++)
        {
            for (int k=start;k<end;k++)
                output[i][j][k - start]=input[i][j][k];
        }
    }
    
    return output;
}


// Self Attention
pair<vector<vector<vector<double>>>, vector<vector<vector<double>>>> SelfAttention(
    const vector<vector<vector<double>>>& Q,
    const vector<vector<vector<double>>>& K,
    const vector<vector<vector<double>>>& V,
    int embed_dim
) {
    // Compute attention scores
    auto scores = tenMul(Q, transpose(K));
    double scaleFactor = sqrt(embed_dim);
    for (auto& batch: scores){
        for (auto& row : batch) {
            for (double& val : row) {
                val /= scaleFactor;
            }
        }
    }
    
    // Apply softmax to the scores
    auto attentionWeights = softmax(scores);

    // Compute the weighted sum of values
    auto attentionOutput = tenMul(attentionWeights, V);

    return {attentionOutput, attentionWeights};
}

vector<vector<vector<double>>> concatenateHeads(
    const vector<vector<vector<vector<double>>>>& attentionOutputs
) {
    int batch_size = attentionOutputs[0].size();
    int seq_length = attentionOutputs[0][0].size();
    int total_embed_dim = 0;

    for (const auto& head : attentionOutputs) {
        total_embed_dim += head[0][0].size();
    }

    vector<vector<vector<double>>> result(batch_size, vector<vector<double>>(seq_length, vector<double>(total_embed_dim, 0.0)));

    for (int i = 0; i < batch_size; ++i) {
        for (int j = 0; j < seq_length; ++j) {
            int offset = 0;
            for (const auto& head : attentionOutputs) {
                for (int k = 0; k < head[i][j].size(); ++k) {
                    result[i][j][offset + k] = head[i][j][k];
                }
                offset += head[i][j].size();
            }
        }
    }

    return result;
}


// Multihead Attention
vector<vector<vector<double>>> Multihead_SelfAttention(
    const vector<vector<vector<double>>>& Q,
    const vector<vector<vector<double>>>& K,
    const vector<vector<vector<double>>>& V,
    int embed_dim,
    int head_num
) {
    
    int head_dim = embed_dim/head_num;
    cout<<head_dim<<endl;
    vector<vector<vector<vector<double>>>> attentionOutputs(head_num);
    vector<vector<vector<vector<double>>>> heads_Q(head_num), heads_K(head_num), heads_V(head_num);
    //vector<vector<vector<vector<double>>>> heads_Q(head_num, vector<vector<vector<double>>>(Q.size()));
    //vector<vector<vector<vector<double>>>> heads_K(head_num, vector<vector<vector<double>>>(K.size()));
    //vector<vector<vector<vector<double>>>> heads_V(head_num, vector<vector<vector<double>>>(V.size()));
    vector<vector<vector<double>>> W_Q(Q.size(), vector<vector<double>>(Q[0].size(), vector<double>(Q[0][0].size(), 0.0)));
    vector<vector<vector<double>>> W_K(Q.size(), vector<vector<double>>(Q[0].size(), vector<double>(Q[0][0].size(), 0.0)));
    vector<vector<vector<double>>> W_V(Q.size(), vector<vector<double>>(Q[0].size(), vector<double>(Q[0][0].size(), 0.0)));
    random_device rd;                  // Seed for random number generator
    mt19937 gen(rd());                 // Standard mersenne_twister_engine
    normal_distribution<> dis(0.0, 1.0); // Normal distribution with mean 0 and std deviation 1
    for (int i = 0; i < Q.size(); ++i) {
            for (int j = 0; j < Q[0].size(); ++j) {
                for (int k = 0; k < Q[0][0].size(); ++k) {
                    W_Q[i][j][k] = dis(gen); // Assign random value
                    W_K[i][j][k] = dis(gen);
                    W_V[i][j][k] = dis(gen);
                }
            }
        }
    for (int i=0; i<head_num; ++i)
    {
        heads_Q[i] = tensor_slice(tenMul(Q, W_Q), i * head_dim, (i + 1) * head_dim);
        heads_K[i] = tensor_slice(tenMul(K, W_K), i * head_dim, (i + 1) * head_dim);
        heads_V[i] = tensor_slice(tenMul(V, W_V), i * head_dim, (i + 1) * head_dim);
    }
    for (int i=0; i<head_num; ++i)
    {
        cout<<heads_Q[i].size()<<" "<<heads_Q[i][0].size()<<" "<<heads_Q[i][0][0].size()<<endl;
        cout<<heads_K[i].size()<<" "<<heads_K[i][0].size()<<" "<<heads_K[i][0][0].size()<<endl;
        cout<<heads_V[i].size()<<" "<<heads_V[i][0].size()<<" "<<heads_V[i][0][0].size()<<endl;
        auto [attentionOutput, attentionWeights] = SelfAttention(heads_Q[i],heads_K[i],heads_V[i], head_dim);
        cout<<attentionOutput.size()<<" "<<attentionOutput[0].size()<<" "<<attentionOutput[0][0].size()<<endl;
        attentionOutputs[i] = attentionOutput;
    }
    auto attention = concatenateHeads(attentionOutputs);
    return attention;
}

int main() {

    const int seed = 42;
    torch::manual_seed(seed);

    // Attention input dimensions
    int batch_size = 3;
    int sequence_length = 4;
    int embed_dim = 8;
    int head_num = 4;

    // Input data generation
    torch::Tensor dummy_input = torch::randn({batch_size, sequence_length, embed_dim});

    // 3D vector to store the input data
    vector<vector<vector<double>>> input_data(batch_size, vector<vector<double>>(sequence_length, vector<double>(embed_dim, 0.0)));

    //Convert torch tensor object to vector object
    auto accessor = dummy_input.accessor<float, 3>();
    for (int b = 0; b < batch_size; ++b) {
        for (int s = 0; s < sequence_length; ++s) {
            for (int e = 0; e < embed_dim; ++e) {
                input_data[b][s][e] = accessor[b][s][e];
            }
        }
    }
    
    vector<vector<vector<double>>> Q = input_data;
    vector<vector<vector<double>>> K = input_data;
    vector<vector<vector<double>>> V = input_data;
    // Compute self-attention
    clock_t start, end;
    start = clock();
    //auto [attentionOutput, attentionWeights] = Multihead_SelfAttention(Q, K, V, embed_dim, head_num);
    auto attentionOutput = Multihead_SelfAttention(Q, K, V, embed_dim, head_num);
    end = clock();
    cout<<attentionOutput.size()<<" "<<attentionOutput[0].size()<<" "<<attentionOutput[0][0].size()<<endl;
    double time_taken = ((double)(end - start))/ CLOCKS_PER_SEC*1000;
    cout << "CPU Time Taken = "<<time_taken<<"ms."<<endl;
    
    return 0;
}
