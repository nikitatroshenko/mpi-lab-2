#include <stdio.h>
#include <math.h>
#include <mpi.h>
#include <assert.h>
#include <stdlib.h>
#include <string.h>

#define MPI_TASK_REQUEST_TAG 1
#define MPI_TASK_BODY_TAG 2

double f(double x) {
//    return sin(x);
    return x * x / 1.8e16;
}

struct task_specification {
    double (*func)(double);

    double a;
    double b;
};

struct configuration {
    size_t all_blocks_count;
    size_t task_blocks_count;
    size_t executors_count;
};

struct configuration *parse_args(int argc, char **argv);

void balancer_routine(
        const struct task_specification *spec,
        const struct configuration *config,
        double *answer);

void executor_routine(
        const struct task_specification *spec,
        const struct configuration *config);

double integrate(
        const struct task_specification *spec,
        const struct configuration *config);

int main(int argc, char **argv) {

    const struct task_specification task_spec = {f, -3e5, 3e5};
    struct configuration *config = parse_args(argc, argv);

    MPI_Init(&argc, &argv);

    int world_size;
    int world_rank;

    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

    config->executors_count = world_size - 1;
    if (world_rank == 0) {
        double result;
        balancer_routine(&task_spec, config, &result);

        printf("Result: %.20lf\n", result);
    } else {
        fprintf(stderr, "Executor: starting executor with rank %d\n",
                world_rank);
        executor_routine(&task_spec, config);
    }

    MPI_Finalize();

    free(config);

    return 0;
}

struct configuration *parse_args(int argc, char **argv) {
    struct configuration *config = calloc(1, sizeof *config);

    config->task_blocks_count = 2;
    config->all_blocks_count = 128;
    for (int i = 0; i < argc; i++) {
        if (!strncmp("--task-blocks=", argv[i], 14)) {
            config->task_blocks_count = strtoull(argv[i] + 14, NULL, 0);
        }
        if (!strncmp("--all-blocks=", argv[i], 13)) {
            config->all_blocks_count = strtoull(argv[i] + 13, NULL, 0);
        }
    }
    return config;
}

void balancer_routine(
        const struct task_specification *spec,
        const struct configuration *config,
        double *answer) {
    double last_processed = spec->a;
    double step = (spec->b - spec->a) * config->task_blocks_count / config->all_blocks_count;
    size_t active_executors = config->executors_count;

    fprintf(stderr, "Balancer: working with %lu executors\n", active_executors);

    *answer = 0;
    while (active_executors) {
        MPI_Status status;

        MPI_Probe(MPI_ANY_SOURCE, MPI_TASK_REQUEST_TAG, MPI_COMM_WORLD, &status);


        int executor = status.MPI_SOURCE;
        double slave_prev_op_result;
        double task_body[2];

        task_body[0] = last_processed;
        if (spec->b - last_processed < step) {
            task_body[1] = spec->b;
        } else {
            task_body[1] = last_processed + step;
        }
        last_processed = task_body[1];

        MPI_Recv(&slave_prev_op_result,
                 1,
                 MPI_DOUBLE,
                 executor,
                 MPI_TASK_REQUEST_TAG,
                 MPI_COMM_WORLD,
                 MPI_STATUS_IGNORE);
        fprintf(stderr, "Balancer: accepted request from executor %d: %lf\n",
                executor, slave_prev_op_result);

        *answer += slave_prev_op_result;

        MPI_Send(&task_body,
                 2,
                 MPI_DOUBLE,
                 executor,
                 MPI_TASK_BODY_TAG,
                 MPI_COMM_WORLD);
        if (task_body[0] == task_body[1]) {
            active_executors--;
            fprintf(stderr, "Balancer: sent terminal message to executor %d\n",
                    executor);
        }
    }
}

void executor_routine(
        const struct task_specification *spec,
        const struct configuration *config) {
    double result = 0;
    double task_body[2];

    do {
        MPI_Send(&result,
                 1,
                 MPI_DOUBLE,
                 0,
                 MPI_TASK_REQUEST_TAG,
                 MPI_COMM_WORLD);
        fprintf(stderr, "Executor: sent request %lf to master\n", result);
        MPI_Recv(task_body,
                 2,
                 MPI_DOUBLE,
                 0,
                 MPI_TASK_BODY_TAG,
                 MPI_COMM_WORLD,
                 MPI_STATUS_IGNORE);
        fprintf(stderr, "Executor: received task [%lf,%lf]\n",
                task_body[0], task_body[1]);
        if (task_body[0] == task_body[1]) {
            fprintf(stderr, "Executor: terminating work\n");
            break;
        }

        struct task_specification block_spec = {spec->func, task_body[0], task_body[1]};
        result = integrate(&block_spec, config);
        fprintf(stderr, "Executor: next result is ready: %lf\n", result);
    } while (1);
}

double integrate(
        const struct task_specification *spec,
        const struct configuration *config) {
    double result = 0;
    double step = (spec->b - spec->a) / config->task_blocks_count;

    for (size_t i = 0; i < config->task_blocks_count; i++) {
        double x1 = spec->a + step * i;
        double x2 = spec->b - step * (config->task_blocks_count - i - 1);
        double y1 = spec->func(x1);
        double y2 = spec->func(x2);

        result += (x2 - x1) * (y2 + y1) / 2;
    }
    return result;
}

