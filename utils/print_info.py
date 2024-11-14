def print_info(title, info=None):
    print("\n".join(
        [title, 
         "-------------------"]))
    if info:
        print(f"\n".join(info))