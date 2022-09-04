# TensorflowTTS-js

```
tensorflowjs_wizard
? Please provide the path of model file or the directory th
If you are converting TFHub module please provide the URL.
-> models/TF/HIFIGAN/pb/
> Tensorflow Saved Model *
> serve
> signature name: serving_default
    inputs: 1 of 1
        name: serving_Default_input_1:0, dtype: DT_F..
    outputs: 1 of 1
        name: SatefulPartitionedCall:0, dtype: DT_FLOT..
? Do you want to skip op validation? 
This will allow conversion of unsupported ops, 
you can implement them as custom ops in tfjs-converter.  (y
/N) y
? Do you want to strip debug ops? 
This will improve model execution performance.  (Y/n) Y
? Do you want to enable Control Flow V2 ops? 
This will improve branch and loop execution performance.  (
Y/n)
? Which directory do you want to save the converted model i
n?  -> models/TFJS/HIFIGAN/float32
```