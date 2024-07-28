def debug(x):
    print("===== [DEBUG] =====")
    if type(x) == list:
        for i in x:
            print(i)
    else:
        print(x)