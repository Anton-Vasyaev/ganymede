# 3rd party
import onnxruntime as ort


def draft_onnxruntime_providers():
    gpu_list = ['TensorrtExecutionProvider']

    cpu_list = ['CPUExecutionProvider']

    model_path = r'D:\data\models\work\unifint\human\darknet\yolov7_tiny_person.onnx'

    session = ort.InferenceSession(model_path, providers=cpu_list)

    print('session construct')


if __name__ == '__main__':
    draft_onnxruntime_providers()