import tensorflow as tf
import os
import argparse
os.environ["CUDA_VISIBLE_DEVICES"] = "0"


def parse_args():
    desc = "AnimeGANv2 for pb"
    parser = argparse.ArgumentParser(description=desc)
    parser.add_argument(
        '--ckpt-dir', type=str,
        help='checkpoints Directory name')

    parser.add_argument(
        '--ckpt-prefix', type=str,
        help='file name to save the checkpoints')

    parser.add_argument(
        '--output-dir', type=str,
        default='models/pb',
        help='output directory to store pb file')

    return parser.parse_args()


if __name__ == '__main__':
    arg = parse_args()

    ckpt_prefix = os.path.join(arg.ckpt_dir, arg.ckpt_prefix)
    output_dir = os.path.join(arg.output_dir, os.path.basename(arg.ckpt_dir), arg.ckpt_prefix)  # savedPath

    # input node and output node from the network
    input_op = 'generator_input:0'
    output_op = 'generator/G_MODEL/out_layer/Tanh:0'

    graph = tf.Graph()
    config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=True)
    with tf.Session(graph=graph, config=config) as sess:
        # Restore from checkpoint
        loader = tf.train.import_meta_graph(ckpt_prefix + '.meta')
        loader.restore(sess, ckpt_prefix)

        # the input and output in the ckpt
        x = tf.get_default_graph().get_tensor_by_name(input_op)
        y = tf.get_default_graph().get_tensor_by_name(output_op)

        # Export checkpoint to SavedModel
        builder = tf.saved_model.builder.SavedModelBuilder(output_dir)

        # custom settings of the input and output in the pb
        inputs = {'input': tf.saved_model.utils.build_tensor_info(x)}
        outputs = {'output': tf.saved_model.utils.build_tensor_info(y)}
        signature = tf.saved_model.signature_def_utils.build_signature_def(inputs, outputs, 'AnimeGANv2')

        builder.add_meta_graph_and_variables(sess, [tf.saved_model.tag_constants.SERVING], {
                                             'custom_signature': signature})
        builder.save()

        """
        This will save your protobuf ('saved_model.pb') in the said folder ('models' here) 
        which can then be loaded by Use_pb.py.
        the output file structure as below
        
        └── pb_model_Hayao-64
        ···├── saved_model.pb
        ···└── variables
        ·········├── variables.data-00000-of-00001
        ·········└── variables.index
        """
    print('------------------------------')
    print("output dir :", output_dir)
    print("output dir :", output_dir)
    print('------------------------------')
