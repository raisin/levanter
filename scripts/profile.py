import jax


def matrix_multiply_in_jax():
    # Create two random matrices, one of size 1000x1000 and the other 1000x2000
    A = jax.random.normal(jax.random.PRNGKey(0), (1000, 1000))
    B = jax.random.normal(jax.random.PRNGKey(1), (1000, 2000))
    # Multiply them together
    C = jax.numpy.dot(A, B)
    # Return the result
    return C


def main():
    profiler_path = "profiling/"
    # "gs://levanter-data/dev/ivan/profiling/adhoc/"
    with jax.profiler.trace(profiler_path, create_perfetto_link=True):
        matrix_multiply_in_jax()
    
    # jax.profiler.start_trace(profiler_path)
    # jax.block_until_ready(matrix_multiply_in_jax())
    # jax.profiler.stop_trace()



if __name__ == "__main__":
    main()
