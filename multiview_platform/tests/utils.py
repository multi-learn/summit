import os


tmp_path = "multiview_platform/tests/tmp_tests/"

def rm_tmp():
    try:
        for file_name in os.listdir(tmp_path):
            os.remove(os.path.join(tmp_path, file_name))
        os.rmdir(tmp_path)
    except:
        pass
