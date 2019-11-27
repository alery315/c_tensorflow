/*
 * use_pb.c
 * To compile: gcc use_pb_file.c -ltensorflow
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
#include <time.h>

/* ------------------Global Variable----------------- */

TF_Buffer *graph_buf;
TF_Graph *graph;
TF_Status *status;
TF_ImportGraphDefOptions *graph_opts;
TF_SessionOptions *sess_opts;
TF_Session *session;

// input
TF_Operation *input_op;
TF_Output input_opout;
TF_Tensor *input;

// operation
const TF_Operation *target_op;
const TF_Operation *const *target_opers;

// output
TF_Output output;
TF_Output *outputs;
TF_Tensor *output_values;

// input data
int ndims = 3;
int input_dims = 2 * 3;
int64_t in_dims[] = {1, 2, 3};
int out_put_dim = 6;


TF_Buffer *read_file(const char *file);

void init(const char *file);

void pre_run_session();

void run_session(float *values_p, int data_length);

void check_status_ok(TF_Status *status, char *step);

void
free_buffer(void *data, size_t length) {
    free(data);
}

static void
Deallocator(void *data, size_t length, void *arg) {
    free(data);
    // *reinterpret_cast<bool*>(arg) = true;
}


int main(int argc, char *argv[]) {

    char *file = "../alery.pb";

    init(file);
    pre_run_session();

    float values[2 * 3] = {
            1.0f, 1.0f, 1.0f,
            1.f, 1.f, 1.f,
    };


    clock_t start, finish;
    double totaltime;
    start = clock();

    for (int i = 0; i < 999999; ++i) {
//        sleep(1);
        float *values_p = malloc(sizeof(float) * input_dims);
//        for (int j = 0; j < input_dims; ++j) {
//            *(values_p + j) = values[j] + (float)i;
//            printf("%f ", *(values_p + j));
//        }
//        printf("%p\n", values_p);

        run_session(values_p, (int)sizeof(float) * input_dims);
    }

    finish = clock();
    totaltime = (double) (finish - start) / CLOCKS_PER_SEC;
    printf("running time is %lf\n", totaltime);



    // free resource
    TF_CloseSession(session, status);
    TF_DeleteSession(session, status);
    TF_DeleteSessionOptions(sess_opts);
    TF_DeleteImportGraphDefOptions(graph_opts);
    TF_DeleteGraph(graph);
    TF_DeleteStatus(status);
    TF_DeleteBuffer(graph_buf);

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


/**
 * check status ok
 * @param status
 * @param step
 */
void
check_status_ok(TF_Status *t_status, char *step) {
    if (TF_GetCode(t_status) != TF_OK) {
        fprintf(stderr, "Error at step \"%s\", status is: %u\n", step, TF_GetCode(t_status));
        fprintf(stderr, "Error message: %s\n", TF_Message(t_status));
        exit(EXIT_FAILURE);
    } else {
        printf("%s\n", step);
    }
}

void
init(const char *file) {
    // Use read_file to get graph_def as TF_Buffer*
    graph_buf = read_file(file);
    graph = TF_NewGraph();

    // Import graph_def into graph
    status = TF_NewStatus();
    graph_opts = TF_NewImportGraphDefOptions();
    TF_GraphImportGraphDef(graph, graph_buf, graph_opts, status);

    check_status_ok(status, "import graph");

    sess_opts = TF_NewSessionOptions();
    session = TF_NewSession(graph, sess_opts, status);

    check_status_ok(status, "create new session");
}

void
pre_run_session() {
    // Pass the graph and a string name of your input operation
    // (make sure the operation name is correct)
    // input
    input_op = TF_GraphOperationByName(graph, "actor_inputs/x");
    input_opout.oper = input_op;
    input_opout.index = 0;

    // operation
    target_op = TF_GraphOperationByName(graph, "NN_output/op_to_store");
    target_opers = &target_op;
    // output
    output.oper = (void *) target_op;
    output.index = 0;
    outputs = &output;
}

void
run_session(float *values_p, int data_length) {

    // Create the input tensor using the dimension (in_dims) and size (num_bytes_in)
    // variables created earlier
    input = TF_NewTensor(TF_FLOAT, in_dims, ndims, values_p, data_length, &Deallocator, 0);

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
//    printf("output value : ");
//    for (int i = 0; i < out_put_dim; ++i) {
//        printf("%f ", *(out_values + i));
//    }
//    printf("\n");

    // 这里不能free,会导致后面TF_NewTensor时候空指针
    // free(values_p);
}