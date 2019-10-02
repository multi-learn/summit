import os

def rm_tmp():
    try:
        for file_name in os.listdir("multiview_platform/tests/tmp_tests"):
            os.remove(os.path.join("multiview_platform/tests/tmp_tests", file_name))
        os.rmdir("multiview_platform/tests/tmp_tests")
    except:
        pass
