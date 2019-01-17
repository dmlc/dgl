
def check_fail(fn, *args, **kwargs):
    try:
        fn(*args, **kwargs)
        return False
    except:
        return True
