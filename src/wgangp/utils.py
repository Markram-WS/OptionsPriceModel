def debug(x):
    print("===== [DEBUG] =====")
    if isinstance(x, list):
        for i in x:
            print(i)
    else:
        print(x)
