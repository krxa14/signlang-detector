from roboflow import Roboflow

rf = Roboflow(api_key="KPrbQ0Gt0ujyoxWYtktD")
project = rf.workspace("majorproject-25tao").project("american-sign-language-v36cz")
version = project.version(2)
dataset = version.download("yolov5")
