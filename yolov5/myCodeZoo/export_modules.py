import torch.onnx
from models.common import earlyFusionConv, Conv

#Function to Convert to ONNX
def Convert_ONNX():

    # set the model to inference mode
    model.eval()

    # Let's create a dummy input tensor
    dummy_input = torch.randn((1, 6, 640, 640), requires_grad=True)

    # Export the model
    torch.onnx.export(model,         # model being run
         dummy_input,       # model input (or a tuple for multiple inputs)
         "earlyFusionConv.onnx",       # where to save the model
         export_params=True,  # store the trained parameter weights inside the model file
         opset_version=10,    # the ONNX version to export the model to
         do_constant_folding=True,  # whether to execute constant folding for optimization
         input_names = ['modelInput'],   # the model's input names
         output_names = ['modelOutput'], # the model's output names
         dynamic_axes={'modelInput' : {0 : 'batch_size'},    # variable length axes
                                'modelOutput' : {0 : 'batch_size'}})
    print(" ")
    print('Model has been converted to ONNX')


if __name__ == "__main__":
    # Let's build our model
    # train(5)
    # print('Finished Training')

    # Test which classes performed well
    # testAccuracy()

    # Let's load the model we just created and test the accuracy per label
    model = earlyFusionConv(6, 32)

    # Test with batch of images
    # testBatch()
    # Test how the classes performed
    # testClassess()

    # Conversion to ONNX
    Convert_ONNX()