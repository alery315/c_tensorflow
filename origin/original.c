/*
 * function.c
 * To compile: gcc function.c -ltensorflow
 * To run: ./a.out
 *
 * author: Yang Dan
 * date: Aug 9, 2018
 */
//#define _NJ_INCLUDE_HEADER_ONLY

#include <stdlib.h>
#include <stdio.h>
#include <tensorflow/c/c_api.h>

#define BITRATE_LEN 4
#define TARGET_BUF_LEN 4

typedef enum _nj_result {
    NJ_OK = 0,        // no error, decoding successful
    NJ_NO_JPEG,       // not a JPEG file
    NJ_UNSUPPORTED,   // unsupported format
    NJ_OUT_OF_MEM,    // out of memory
    NJ_INTERNAL_ERR,  // internal error
    NJ_SYNTAX_ERROR,  // syntax error
    __NJ_FINISHED,    // used internally, will never be reported
} nj_result_t;

struct TF_Init_Data {
    void *Graph;
    void *Session;
    void *Status;
};

int BITRATE[BITRATE_LEN] = {500, 850, 1200, 1500};
int TARGET_BUF[TARGET_BUF_LEN] = {1500, 2300, 3600, 4300};
/*
 * check_status_ok
 * description:
 * Verifies status OK after each TensorFlow operation.
 * parameters:
 *     input status - the TensorFlow status
 *     input step   - a description of the last operation performed
 */
/*void check_status_ok(TF_Status* status, char* step) {
    if (TF_GetCode(status) != TF_OK) {
        fprintf(stderr, "Error at step \"%s\", status is: %u\n", step, TF_GetCode(status));
        fprintf(stderr, "Error message: %s\n", TF_Message(status));
        exit(EXIT_FAILURE);
    } else {
        printf("%s\n", step);
    }
}*/

/*
 * check_result_ok
 * description:
 * Verifies result OK after read the graph.
 * parameters:
 *     input result - the  result
 *     input step   - a description of operation performed
 */
void check_result_ok(enum _nj_result result, char *step) {
    if (result != NJ_OK) {
        fprintf(stderr, "Error at step \"%s\", status is: %u\n", step, result);
        exit(EXIT_FAILURE);
    } else {
        printf("%s\n", step);
    }
}

/*
 * file_length
 * description:
 * Returns the length of the given file.
 * parameters:
 *     input file - any file
 */
unsigned long file_length(FILE *file) {
    fseek(file, 0, SEEK_END);
    unsigned long length = ftell(file);
    fseek(file, 0, SEEK_SET);
    return length;
}

/*
 * load_file
 * description:
 * Loads a binary buffer from the given file
 * parameters:
 *     input file  - a binary file
 *     inut length - the length of the file
 */
char *load_file(FILE *file, unsigned long length) {
    char *buffer;
    buffer = (char *) malloc(length + 1);
    if (!buffer) {
        fprintf(stderr, "Memory error while reading buffer");
        fclose(file);
        exit(EXIT_FAILURE);
    }
    fread(buffer, length, 1, file);
    return buffer;
}

/*Array 0-15  last_quality
 *      16-31  ebuffer
 *      32-47 data_size/(end_time - start_time)
 *      48-63 end_time - start_time
 *      64-79 gop_flag *      80-95 rtt
 *      86-111 ebitrate
 * */
float *Parse_Data(float Param[16 * 7]) {

    for (int i = 64; i < 80; i++) {
        if (i == 64 || Param[i] == 1.0) {
            int j = 1;
            float chunk_length = Param[i - 16];
            while ((int) (Param[i + j]) != 1 && (i + j) < 80) {
                chunk_length += Param[i + j - 16];
                j++;
            }
            for (int k = i; k < i + j; k++)
                Param[k] = chunk_length;
        }
    }

    /*for(int i = 0; i < 16 * 7; i ++){
        if(i%16 == 0 && i != 0)
            printf("\n");
        printf("%f ", Param[i]);
    }*/
    return Param;

}

/*
 * 1. Initialize TensorFlow session
 * 2. Read in the previosly exported graph
 * 3. Read tensor from input data
 * 4. Run the data through the model
 * 5. Print top labels
 * 6. Close session to release resources
 *
 */

struct TF_Init_Data Init_TF_Session() {
    // 1. Initialize TensorFlow session
    struct TF_Init_Data Result_Init;
    TF_Graph *graph = TF_NewGraph();
    TF_SessionOptions *session_opts = TF_NewSessionOptions();
    TF_Status *status = TF_NewStatus();
    TF_Session *session = TF_NewSession((TF_Graph *) graph, session_opts, status);
    //check_status_ok(status, "Initialization of TensorFlow session");
    //void* result_graph, result_session;
    Result_Init.Graph = (void *) graph;
    Result_Init.Session = (void *) session;
    Result_Init.Status = (void *) status;
    //check_status_ok(status, "Initialization of TensorFlow session");
    return Result_Init;
}

void *Init_Read_Graph(char *input_graph, void *graph, void *status) {
    // 2. Read in the previosly exported graph
    FILE *pb_file = fopen(input_graph, "rb");
    if (!pb_file) {
        fprintf(stderr, "Could not read graph file \"%s\"\n", input_graph);
        exit(EXIT_FAILURE);
    }
    unsigned long pb_file_length = file_length(pb_file);
    char *pb_file_buffer = load_file(pb_file, pb_file_length);
    TF_Buffer *graph_def = TF_NewBufferFromString(pb_file_buffer, pb_file_length);
    TF_ImportGraphDefOptions *graph_opts = TF_NewImportGraphDefOptions();
    TF_GraphImportGraphDef((TF_Graph *) graph, graph_def, graph_opts, (TF_Status *) status);
    fclose(pb_file);
    free(pb_file_buffer);
    //check_status_ok(status, "Loading of .pb graph");
    return graph;
}

int TF_Session_Run(struct TF_Init_Data Init_Data, float data_x[7 * 16], int data_length) {
    int ndims = 3;
    int64_t dims[] = {1, 7, 16};
    int Out_put_dim = 16;
    TF_Tensor *tensor_x = TF_NewTensor(TF_FLOAT, dims, ndims, data_x, data_length, NULL, NULL);
    TF_Graph *graph_t;
    graph_t = (TF_Graph *) Init_Data.Graph;
    // 3. Run the input through the model
    TF_Output outputX;
    outputX.oper = TF_GraphOperationByName(graph_t, "actor/actor_inputs/X");
    outputX.index = 0;
    TF_Output *inputs_x = {&outputX};
    TF_Tensor *const *input_values_x = {&tensor_x};


    const TF_Operation *target_op = TF_GraphOperationByName(graph_t, "actor/NN_output/Softmax");
    TF_Output output;
    output.oper = (void *) target_op;
    output.index = 0;
    TF_Output *outputs = {&output};
    TF_Tensor *output_values;

    const TF_Operation *const *target_opers = {&target_op};

    TF_SessionRun(
            (TF_Session *) Init_Data.Session,
            NULL,
            inputs_x, input_values_x, 1,
            outputs, &output_values, 1,
            target_opers, 1,
            NULL,
            (TF_Status *) Init_Data.Status
    );
    printf("tf code : %d\n", TF_GetCode(Init_Data.Status));
    //check_status_ok(status, "Running of the image through the model");
    float *www = TF_TensorData(output_values);
    float max = 0, decision = -1;
    for (int i = 0; i < Out_put_dim; i++) {
        printf("%f ", *(www + i));
        float temp = (float) (*(www + i));
        if (max < temp) {
            max = temp;
            decision = i;
        }
    }
    printf("\n");
    // 6. Close session to release resources
    //njDone(); // resets NanoJPEG's internal state and frees memory
    return decision;
}

int *main(int argc, char *argv[]) {
    /*if (argc != 2) {
        fprintf(stderr, "3 arguments expected, %d received\n", argc - 1);
        exit(EXIT_FAILURE);
    }*/

    char *input_graph = "../results/model.pb";
    void *graph;
    //TF_Graph* graph = TF_NewGraph();
    struct TF_Init_Data Init_Data = Init_TF_Session();
    Init_Data.Graph = Init_Read_Graph(input_graph, Init_Data.Graph, Init_Data.Status);
    TF_Status *status = TF_NewStatus();


    int ndims = 3;
    int64_t dims[] = {1, 7, 16};

/*Array 0-15  last_quality
 *      16-31  ebuffer
 *      32-47 data_size/(end_time - start_time)
 *      48-63 end_time - start_time
 *      64-79 gop_flag *
 *      80-95 rtt
 *      86-111 ebitrate
 * */
    float data_x[16 * 7] = {
            1.5, 1.5, 1.5, 1.5, 0.5, 1.5, 0.5, 0.5, 0.5, 1.5, 1.5, 1.2, 1.2, 1.2, 1.5, 1.2,
            1.5, 1.5, 1.5, 1.5, 0.5, 1.5, 0.5, 0.5, 0.5, 1.5, 1.5, 1.2, 1.2, 1.2, 1.5, 1.2,
            1.5, 1.5, 1.5, 1.5, 0.5, 1.5, 0.5, 0.5, 0.5, 1.5, 1.5, 1.2, 1.2, 1.2, 1.5, 1.2,
            0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1,
            0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0,
            1.5, 1.5, 1.5, 1.5, 0.5, 1.5, 0.5, 0.5, 0.5, 1.5, 1.5, 1.2, 1.2, 1.2, 1.5, 1.2,
            1.5, 1.5, 1.5, 1.5, 0.5, 1.5, 0.5, 0.5, 0.5, 1.5, 1.5, 1.2, 1.2, 1.2, 1.5, 1.2,
    };
    float *data_parse_base;
    float data_parse[16 * 7];
    //这个地方C没有传引用，很难受，只能用另一个指针接受出来。
    //parse let the gop flag became the chunklength
    data_parse_base = Parse_Data(data_x);
    for (int i = 0; i < 16 * 7; i++) {
        data_parse[i] = *(data_parse_base + i);
        printf("%f\n", data_parse[i]);
    }
    int data_length = sizeof(data_parse);
    printf("%d\n", data_length);
    int res = TF_Session_Run(Init_Data, data_parse, data_length);
    // parse res;
    int result[1];
    result[0] = res / BITRATE_LEN;
    result[1] = res / TARGET_BUF_LEN;
    printf("bit_rate\n%d\n", BITRATE[result[0]]);
    printf("target_buffer\n%d\n", TARGET_BUF[result[1]]);
    return result;
}
