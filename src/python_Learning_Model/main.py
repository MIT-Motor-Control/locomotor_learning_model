import setParameters 


def execute_main(x):
    print("running script: ", x)


if __name__ == "__main__":
    params = setParameters.paramFixed()
    params.print_all_variables()