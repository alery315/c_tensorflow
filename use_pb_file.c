/*
 * use_pb.c
 * To compile: gcc use_pb.c -ltensorflow
 * To run: ./a.out
 *
 * author: alery
 * date: Nov 26, 2019
 */
#include <stdlib.h>
#include <stdio.h>
#include <tensorflow/c/c_api.h>
#include <unistd.h>
#include <assert.h>


TF_Buffer *read_file(const char *file);

void free_buffer(void *data, size_t length) {
    free(data);
}

static void Deallocator(void *data, size_t length, void *arg) {
    free(data);
    // *reinterpret_cast<bool*>(arg) = true;
}


/**
 * check status ok
 * @param status
 * @param step
 */
void check_status_ok(TF_Status *status, char *step) {
    if (TF_GetCode(status) != TF_OK) {
        fprintf(stderr, "Error at step \"%s\", status is: %u\n", step, TF_GetCode(status));
        fprintf(stderr, "Error message: %s\n", TF_Message(status));
        exit(EXIT_FAILURE);
    } else {
        printf("%s\n", step);
    }
}


int main(int argc, char *argv[]) {

    char *input_graph = "../alery.pb";

    // Use read_file to get graph_def as TF_Buffer*
    TF_Buffer *graph_def = read_file(input_graph);
    TF_Graph *graph = TF_NewGraph();

    // Import graph_def into graph
    TF_Status *status = TF_NewStatus();
    TF_ImportGraphDefOptions *graph_opts = TF_NewImportGraphDefOptions();
    TF_GraphImportGraphDef(graph, graph_def, graph_opts, status);
    if (TF_GetCode(status) != TF_OK) {
        fprintf(stderr, "ERROR: Unable to import graph %s", TF_Message(status));
        return -1;
    } else {
        fprintf(stdout, "Successfully imported graph\n");
    }

    int ndims = 3;
    int input_dims = 2 * 3;
    int64_t in_dims[] = {1, 2, 3};
    int out_put_dim = 6;
    float values[2 * 3] = {
            1.0f, 2.0f, 3.0f,
            1.f, 2.f, 3.f,
    };

    float *values_p = malloc(sizeof(float) * input_dims);
    for (int j = 0; j < input_dims; ++j) {
        *(values_p + j) = values[j];
        printf("%f ", *(values_p + j));
    }
    printf("\n");

    // Pass the graph and a string name of your input operation
    // (make sure the operation name is correct)
    // input
    TF_Operation *input_op = TF_GraphOperationByName(graph, "actor_inputs/x");
    TF_Output input_opout = {input_op, 0};

    // Create the input tensor using the dimension (in_dims) and size (num_bytes_in)
    // variables created earlier
    TF_Tensor *input = TF_NewTensor(TF_FLOAT, in_dims, ndims, values_p, sizeof(values), &Deallocator, 0);


    TF_SessionOptions *sess_opts = TF_NewSessionOptions();
    TF_Session *session = TF_NewSession(graph, sess_opts, status);
    assert(TF_GetCode(status) == TF_OK);

    check_status_ok(status, "Running of the image through the model");

    // operation
    const TF_Operation *target_op = TF_GraphOperationByName(graph, "NN_output/op_to_store");
    const TF_Operation *const *target_opers = &target_op;
    // output
    TF_Output output = {(void *) target_op, 0};

    TF_Output *outputs = &output;
    TF_Tensor *output_values;


    TF_SessionRun(
            session,
            NULL,
            &input_opout, &input, 1,
            outputs, &output_values, 1,
            target_opers, 1,
            NULL,
            status
    );


    // get result after run session
    float *out_values = TF_TensorData(output_values);
    printf("output value : ");


    for (int i = 0; i < out_put_dim; ++i) {
        printf("%f ", *(out_values + i));
    }
    printf("\n");

    *(values_p) = 100.0f;
    input = TF_NewTensor(TF_FLOAT, in_dims, ndims, values_p, sizeof(values), &Deallocator, 0);
    TF_SessionRun(
            session,
            NULL,
            &input_opout, &input, 1,
            outputs, &output_values, 1,
            target_opers, 1,
            NULL,
            status
    );

    out_values = TF_TensorData(output_values);
    printf("output value : ");
    for (int i = 0; i < out_put_dim; ++i) {
        printf("%f ", *(out_values + i));
    }
    printf("\n");

    TF_CloseSession(session, status);
    TF_DeleteSession(session, status);
    TF_DeleteSessionOptions(sess_opts);
    TF_DeleteImportGraphDefOptions(graph_opts);
    TF_DeleteGraph(graph);
    TF_DeleteStatus(status);

    return 0;
}

TF_Buffer *read_file(const char *file) {
    FILE *f = fopen(file, "rb");
    fseek(f, 0, SEEK_END);
    long fsize = ftell(f);
    fseek(f, 0, SEEK_SET);  //same as rewind(f);

    void *data = malloc(fsize);
    fread(data, fsize, 1, f);
    fclose(f);

    TF_Buffer *buf = TF_NewBuffer();
    buf->data = data;
    buf->length = fsize;
    buf->data_deallocator = free_buffer;
    return buf;
}