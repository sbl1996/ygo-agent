import numpy as np
import optree
import tflite_runtime.interpreter as tf_lite

def tflite_predict(interpreter, rstate, obs):
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    inputs = rstate, obs
    for i, x in enumerate(optree.tree_leaves(inputs)):
        interpreter.set_tensor(input_details[i]["index"], x)
    interpreter.invoke()
    results = [
        interpreter.get_tensor(o["index"]) for o in output_details]
    rstate1, rstate2, probs, value = results
    rstate = (rstate1, rstate2)
    return rstate, probs, value

def predict_fn(interpreter, rstate, obs):
    obs = optree.tree_map(lambda x: np.array([x]), obs)
    rstate, probs, value = tflite_predict(interpreter, rstate, obs)
    prob = probs[0].tolist()
    value = float(value[0])
    return rstate, prob, value

def load_model(checkpoint, *args, **kwargs):
    with open(checkpoint, "rb") as f:
        tflite_model = f.read()
    interpreter = tf_lite.Interpreter(
        model_content=tflite_model, num_threads=kwargs.get("num_threads", 1))
    interpreter.allocate_tensors()
    return interpreter
